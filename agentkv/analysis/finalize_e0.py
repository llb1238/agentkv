"""Audit and present E0 results without rerunning measurements.

Full request payloads remain local; compact metadata makes the recorded queue,
latency, cache partition and stability checks reviewable from the repository.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import RATES,SEEDS,WORKLOAD,SCHEDULE,render,write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import ROOT
from agentkv.bench.shared_pilot import parse_dmon
from agentkv.analysis.recomputation import CATEGORIES


def read(path):return json.loads(path.read_text())


def assert_close(left,right):
    assert math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-9),(left,right)


def check_partition(folder,summary):
    rows=pq.read_table(folder/'recomputation.parquet').to_pylist()
    counts={k:sum(r[k] for r in rows) for k in CATEGORIES}
    assert counts==summary['source_tokens']
    total=sum(r['prompt_blocks_tokens'] for r in rows)
    assert total==summary['prompt_blocks_tokens'] and len(rows)==summary['requests']
    for r in rows:
        assert all(r[k]>=0 and r[k]%16==0 for k in CATEGORIES)
        assert sum(r[k] for k in CATEGORIES)==r['prompt_blocks_tokens']-r['hit']
        assert r['unavoidable']==r['prompt_blocks_tokens']-r['history_prefix']
        assert r['gap_eviction']==max(0,r['trigger_prefix']-r['hit'])
        assert r['other_eviction']==max(0,r['history_prefix']-max(r['trigger_prefix'],r['hit']))
    for k in CATEGORIES:assert_close(counts[k]/total,summary['source_fractions'][k])
    assert_close(sum(counts.values())/total,summary['recomputed_fraction'])
    assert sum(r['hit_exceeds_history'] for r in rows)==summary['hit_exceeds_history_anomalies']
    gap=[r for r in rows if r['gap_eviction']]
    assert len(gap)==summary['gap_requests']
    observed=[r['queued_s'] for r in gap if r['queued_s'] is not None]
    assert len(observed)==summary['gap_queue_observed_requests']
    if observed:assert_close(float(np.mean(observed)),summary['gap_requests_mean_queued_s'])
    return rows


def independent_stability(rows,metrics,events,origin):
    samples=[s for s in metrics if 600<=s['elapsed_s']<=2400 and 'vllm:num_requests_waiting' in s.get('values',{})]
    times=np.array([s['elapsed_s'] for s in samples]);q=np.array([s['values']['vllm:num_requests_waiting'] for s in samples])
    zero=float(np.mean(q==0));p99=float(np.percentile(q,99));increment=float(np.polyfit(times,q,1)[0]*1800)
    halves=[[],[]]
    for r in rows:
        if r['status']=='success' and not r['is_entry'] and r['send_ns'] is not None and r['ttft_ms'] is not None:
            t=(r['send_ns']-origin)/1e9
            if 600<=t<=2400:halves[int(t>=1500)].append(r['ttft_ms'])
    assert all(halves)
    ratio=float(np.median(halves[1])/np.median(halves[0]))
    assert any(e['event']=='kv_observer_ready' for e in events)
    blocked=[e for e in events if e['event']=='kv_allocation_failure' and e.get('admission') and e['waiting']>0 and 600<=(e['ns']-origin)/1e9<=2400]
    checks=dict(queue_distribution=zero>=.8 and p99<=8,fitted_growth_30min=increment<1,
                ttft_half_ratio=ratio<=1.5,conditional_kv_sizing=all(e['running']>=7 for e in blocked))
    return checks,dict(queue_zero_fraction=zero,queue_p99=p99,fitted_increase_30min=increment,
                       ttft_half_ratio=ratio,kv_blocking_events=len(blocked))


def finalize(directory):
    commit=git_commit();report=read(directory/'run.json')
    assert report['status']=='complete' and len(report['runs'])==12
    assert {(r['rate'],r['seed']) for r in report['runs']}=={(r,s) for r in RATES for s in SEEDS}
    assert report['source']['workload_sha256']==sha256_file(WORKLOAD)
    assert report['source']['schedule_sha256']==sha256_file(SCHEDULE)
    validation=read(directory/'model-validation.json')
    assert sha256_file(directory/'models-manifest.json')==validation['input_manifest_sha256']
    criteria=[];payloads=[];model_requests=gap_blocks=0
    for item in report['runs']:
        assert item['status']=='complete'
        for policy in ('lru','oracle'):
            folder=directory/item['name']/'models'/policy;m=read(folder/'run.json')
            assert m==item['models'][policy] and m['git_commit']==report['git_commit']
            assert sha256_file(folder/'requests.parquet')==m['requests_sha256']
            assert sha256_file(folder/'recomputation.parquet')==m['recomputation_sha256']
            assert m['population_sha256']==item['population_sha256']
            classified=check_partition(folder,m['recomputation']);audit=m['summary']['gap_eviction_audit']
            assert audit['blocks']==audit['passed']==m['recomputation']['source_tokens']['gap_eviction']//16
            assert audit['consistency']==1 and not audit['failures']
            model_requests+=len(classified);gap_blocks+=audit['blocks']
        folder=directory/item['name']/'real';run=read(folder/'run.json');manifest=read(folder/'server-manifest.json')
        assert run['status']=='complete' and run['git_commit']==report['git_commit']==manifest['agentkv_commit']
        assert run['server']==manifest and run['source']==report['source']
        assert item['real']['requests_sha256']==run['requests_sha256']==sha256_file(folder/'requests.parquet')
        assert sha256_file(folder/'loop_lag.parquet')==run['loop_lag_sha256']
        assert manifest['gpu_cache_tokens']==182880 and manifest['cpu_cache_tokens']==274320
        assert manifest['max_num_seqs']==8 and manifest['max_model_len']==16384 and manifest['policy']=='lru'
        for key,value in dict(duration=2400,warmup=600,cooldown=0,arrival_rate=item['rate'],population_seed=item['seed'],
                              hint_tier='none',policy='lru',allow_dirty=False,steady_state=True).items():
            assert run['parameters'][key]==value,(key,run['parameters'][key],value)
        import hashlib
        population=hashlib.sha256(json.dumps(run['instances'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert population==item['population_sha256']
        columns=[n for n in pq.ParquetFile(folder/'requests.parquet').schema_arrow.names if n!='prompt_token_ids']
        table=pq.read_table(folder/'requests.parquet',columns=columns,use_threads=False)
        pq.write_table(table,folder/'requests-metadata.parquet',compression='zstd')
        payloads.append(dict(path=str((folder/'requests.parquet').relative_to(directory)),sha256=run['requests_sha256'],
                             metadata_path=str((folder/'requests-metadata.parquet').relative_to(directory)),
                             metadata_sha256=sha256_file(folder/'requests-metadata.parquet'),rows=table.num_rows))
        rows=table.to_pylist();window=[r for r in rows if r['in_window'] and r['status']=='success']
        assert len(window)==run['summary']['window_success']
        for r in window:assert sum(r[k] for k in ('gpu_hit','cpu_hit','recomputed'))==r['prompt_len']
        for k in ('gpu_hit','cpu_hit','recomputed'):
            assert_close(sum(r[k] for r in window)/sum(r['prompt_len'] for r in window),run['summary']['nominal_token_weighted_fraction'][k])
        classified=check_partition(folder,item['real']['recomputation'])
        assert {(r['instance_id'],r['trace_req_idx']) for r in classified}=={(r['instance_id'],r['trace_req_idx']) for r in window}
        assert sha256_file(folder/'recomputation.parquet')==item['real']['recomputation_sha256']
        rounded=sum(sum(r[k] for k in CATEGORIES) for r in classified)
        tail=sum(r['prompt_len']%16 for r in window)
        hit_tail=sum((r['gpu_hit']+r['cpu_hit'])%16 for r in window)
        nominal=sum(r['recomputed'] for r in window)
        assert nominal==rounded+tail-hit_tail
        payloads[-1]['rounding']=dict(classified_recomputed_tokens=rounded,prompt_tail_tokens=tail,
                                     hit_tail_tokens=hit_tail,nominal_recomputed_tokens=nominal)
        events=[json.loads(line) for path in (folder/'server-observations').glob('*.jsonl') for line in path.read_text().splitlines()]
        metrics=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        checks,numbers=independent_stability(rows,metrics,events,run['origin_ns'])
        saved=item['real']['stability'];assert checks==saved['checks'] and saved['stable']==all(checks.values())
        assert_close(numbers['queue_zero_fraction'],saved['details']['queue']['zero_fraction'])
        assert_close(numbers['queue_p99'],saved['details']['queue']['p99'])
        assert_close(numbers['fitted_increase_30min'],saved['details']['regression']['fitted_increase_30min'])
        assert_close(numbers['ttft_half_ratio'],saved['details']['ttft']['ratio'])
        origin=manifest['clock_pair']['wall_time_s']+(run['origin_ns']-manifest['clock_pair']['monotonic_ns'])/1e9
        dmon=parse_dmon((folder/'dmon.log').read_text())
        for gpu in (0,1):assert len([r for r in dmon if r['gpu']==gpu and 600<=r['wall_time_s']-origin<=2400])>=1600
        criteria.append(dict(name=item['name'],rate=item['rate'],seed=item['seed'],status=saved['status'],**numbers,
                             failed_checks=','.join(k for k,v in checks.items() if not v)))
    assert model_requests==validation['requests_checked'] and gap_blocks==validation['gap_blocks_checked']
    # Auditing must preserve the committed LRU baseline's nominal token counts.
    old=read(ROOT/'results/m1-two-tier-seed0/run.json')
    original=next(r for r in old['results'] if r['directory']=='config-11')
    target=next(r for r in report['runs'] if r['name']=='lambda-1-seed-0')
    assert target['models']['lru']['summary']['source_tokens']==original['summary']['source_tokens']
    with (directory/'criteria.csv').open('w',newline='') as sink:
        writer=csv.DictWriter(sink,fieldnames=list(criteria[0]));writer.writeheader();writer.writerows(criteria)
    render(directory,report)
    queue_figure(directory,report)
    append_report(directory,report,criteria,commit)
    result=dict(git_commit=commit,measurement_git_commit=report['git_commit'],passed=True,
                measurement_run_sha256=sha256_file(directory/'run.json'),model_requests_checked=model_requests,
                gap_blocks_checked=gap_blocks,real_window_requests=sum(r['real']['summary']['window_requests'] for r in report['runs']),
                backend_commits=sorted({read(directory/r['name']/'real/server-manifest.json')['vllm_commit'] for r in report['runs']}),
                stable_runs=sum(r['real']['stability']['stable'] for r in report['runs']),
                failed_requests=sum(r['real']['summary']['window_failed'] for r in report['runs']),
                payloads=payloads,checks=['all 12 registered populations','source and output SHA256','exact capacities and slots',
                  'unchanged LRU baseline nominal counts','rounded token conservation','queue timing coverage',
                  'independent OLS/quantiles/TTFT stability','two-card dmon coverage','gap-block loss-time consistency'])
    write_json(directory/'postprocess-run.json',result)
    artifacts={str(p.relative_to(directory)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
               for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='artifacts.json'}
    write_json(directory/'artifacts.json',dict(git_commit=commit,files=artifacts))
    return result


def queue_figure(directory,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6),sharex=True)
    for rate,ax in zip(RATES,axes.flat):
        for item in [r for r in report['runs'] if r['rate']==rate]:
            data=[json.loads(line) for line in (directory/item['name']/'real/metrics.jsonl').read_text().splitlines()]
            good=[r for r in data if 'vllm:num_requests_waiting' in r.get('values',{}) and r['elapsed_s']<=2400]
            ax.plot([r['elapsed_s']/60 for r in good],[r['values']['vllm:num_requests_waiting'] for r in good],
                    lw=.65,label=f"seed {item['seed']} ({item['real']['stability']['status']})")
        ax.axvspan(0,10,color='gray',alpha=.12);ax.axhline(8,color='gray',lw=.6,ls=':')
        ax.set(title=f'Arrivals/min = {rate:g}',xlim=(0,40),ylim=(0,None));ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=6)
    for ax in axes[-1]:ax.set_xlabel('Elapsed minutes')
    for ax in axes[:,0]:ax.set_ylabel('Waiting requests')
    fig.tight_layout()
    for ext in ('png','pdf'):
        kwargs=dict(metadata={'CreationDate':None,'ModDate':None}) if ext=='pdf' else {}
        fig.savefig(directory/f'waiting_queues.{ext}',dpi=300,**kwargs)
    plt.close(fig)


def append_report(directory,report,criteria,commit):
    lines=['','## 统计窗口汇总','','每次先按整块token加权，再对保留的稳定人口种子等权平均；该表是经稳定性筛选后的条件均值，不能当作全部人口的无条件均值。',
           '', '| λ | 稳定/登记 | 不可避免 | 间隙驱逐（含排队） | 其他驱逐 | 总重算 | 同人口LRU模型 | 同人口CPU Oracle | 间隙/可避免 | 间隙请求排队(ms) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rate in RATES:
        kept=[r for r in report['runs'] if r['rate']==rate and r['real']['stability']['stable']]
        if not kept:lines.append(f'| {rate:g} | 0/3 | — | — | — | — | — | — | — | — |');continue
        mean=lambda f:float(np.mean([f(r) for r in kept]))
        values=[mean(lambda r,k=k:r['real']['recomputation']['source_fractions'][k]) for k in CATEGORIES]
        values+=[mean(lambda r:r['real']['recomputation']['recomputed_fraction']),
                 mean(lambda r:r['models']['lru']['recomputation']['recomputed_fraction']),
                 mean(lambda r:r['models']['oracle']['recomputation']['recomputed_fraction']),
                 mean(lambda r:r['real']['recomputation']['gap_share_of_avoidable'])]
        lines.append(f"| {rate:g} | {len(kept)}/3 | "+' | '.join(f'{v:.3%}' for v in values)+f" | {mean(lambda r:r['real']['recomputation']['gap_requests_mean_queued_s'])*1000:.2f} |")
    lines+=['','## 不稳定人口诊断（不纳入曲线）','','以下按失败人口等权平均，仅作诊断；λ=1.4无稳定人口，不能报告稳态均值。',
            '', '| λ | 不稳定人口数 | 实测重算 | 同人口LRU模型 | 同人口CPU Oracle | 间隙/可避免 | 间隙请求排队(s) |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for rate in RATES:
        group=[r for r in report['runs'] if r['rate']==rate and not r['real']['stability']['stable']]
        if not group:continue
        mean=lambda f:float(np.mean([f(r) for r in group]))
        values=[mean(lambda r:r['real']['recomputation']['recomputed_fraction']),
                mean(lambda r:r['models']['lru']['recomputation']['recomputed_fraction']),
                mean(lambda r:r['models']['oracle']['recomputation']['recomputed_fraction']),
                mean(lambda r:r['real']['recomputation']['gap_share_of_avoidable'])]
        lines.append(f"| {rate:g} | {len(group)} | "+' | '.join(f'{v:.3%}' for v in values)+f" | {mean(lambda r:r['real']['recomputation']['gap_requests_mean_queued_s']):.3f} |")
    lines+=['','## 逐次稳定性复核','','| 运行 | 结论 | 队列=0 | 队列P99 | 30分钟拟合增量 | TTFT后/前 | KV阻塞次数 | 未通过项 |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in criteria:
        lines.append(f"| {r['name']} | {r['status']} | {r['queue_zero_fraction']:.2%} | {r['queue_p99']:.2f} | {r['fitted_increase_30min']:.3f} | {r['ttft_half_ratio']:.3f} | {r['kv_blocking_events']} | {r['failed_checks'] or '无'} |")
    lines+=['','## 设备与口径边界','','本批次GPU选择、显存与进程快照见各gpu-check目录，运行全程两卡dmon在各real目录；“显存共享、算力共享”按实际记录，不将总SM利用率等同于本进程利用率。',
      '', 'CPU模型固定服务时间0.5秒、搬运零耗时；Oracle只替换CPU策略，GPU仍是LRU，不宣称全局最优。同人口不保证相同发送序列，不能把模型Oracle重算比例视为真实运行可达的下界。',
      '', '按给定公式划分前缀来源。u取发送历史，尚未完成的并发前缀也属于历史，因此“其他驱逐”是剩余历史前缀未命中，不能仅凭这一项证明物理驱逐。无限缓存零驱逐的单元测试使用串行闭环链。真实服务的整块/末token回放边界仍按名义命中计数；不足16-token的尾部另列。',
      '', '排队时长口径为客户端安全阀等待加EngineCore接收到首次调度的等待；未包含HTTP前端解析耗时。间隙类名称始终包含排队，不能将不稳定运行中的该比例单独归因于工具停顿。',
      '', '![全部人口的队列诊断，含不稳定运行](waiting_queues.png)',
      '', f'测量代码 `{report["git_commit"]}`；复核与作图代码 `{commit}`。完整请求文件本机保留，Git提供去除重复prompt数组的requests-metadata.parquet；原始及元数据SHA256见postprocess-run.json。未复制实现源码。',
      '', '[独立复核与来源](postprocess-run.json) · [全部文件指纹](artifacts.json) · [逐次判据](criteria.csv)',
      '', '```bash','python -m agentkv.analysis.finalize_e0 --run-dir /tmp/completed-e0-run','```']
    devices=sorted({r['gpu']['gpu_index'] for r in report['runs']})
    modes=sorted({r['gpu']['mode'] for r in report['runs']})
    free0=[]
    for item in report['runs']:
        snapshot=read(directory/item['name']/'gpu-check/before.json')
        gpu=next(line.split(',') for line in snapshot['gpus'].splitlines() if line.split(',')[0].strip()=='0')
        free0.append(float(gpu[4])*1024**2/1e9)
    lines+=['',f"本批使用的GPU编号集合为{devices}，资源状态：{'；'.join(modes)}。各次启动前GPU0空闲显存范围{min(free0):.3f}–{max(free0):.3f} GB。"]
    if max(free0)<60:lines+=['全部启动检查中GPU0均未达到60GB空闲显存门槛。']
    if (directory/'availability-initial/README.md').exists():
        lines+=['','[最初的GPU进程采样及其N/A限制](availability-initial/README.md)']
    lines+=['','固定服务时间模型不包含服务时长随重算量或服务端排队变化的反馈，不能用于证明真实系统稳定；不稳定实跑与模型的差异只作诊断，不解释为可直接获得的Oracle收益。']
    with (directory/'README.md').open('a') as sink:sink.write('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(finalize(args.run_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
