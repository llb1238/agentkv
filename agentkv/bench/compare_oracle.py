"""Paired LRU / next-deadline Oracle comparison for the 14 qualified candidates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import ROOT, SCHEMA
from agentkv.bench.steady_state import RootSchedule, SessionInstance
from agentkv.sim.two_tier import load_block_ids, simulate


def compare(source_dir,output_dir,allow_dirty=False):
    commit=git_commit(allow_dirty)
    if (output_dir/'run.json').exists(): raise FileExistsError('New comparison directory required')
    baseline=json.loads((source_dir/'run.json').read_text())
    capacity_dir=Path(baseline['source']['capacity_directory'])
    capacity=json.loads((capacity_dir/'run.json').read_text())
    qualified=[r for r in baseline['results'] if r['recommend_for_gpu_pilot']]
    if len(qualified)!=14: raise ValueError('Expected the 14 qualified configurations')
    output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(git_commit=commit,source=dict(baseline_sha256=sha256_file(source_dir/'run.json'),
                capacity_sha256=sha256_file(capacity_dir/'run.json')),
                parameters=dict(duration_s=2400,warmup_s=600,service_s=.5,block_size=16,target='config-11',
                                noticeably_larger_threshold_pp=2.),
                oracle=dict(access='all requests in each sampled remaining instance lifetime, including after T',
                    deadline='instance_start + est_send_s - tau; remove issued IDs only; overdue queued requests remain',
                    eviction='farthest next unissued expected deadline; never-used first; larger block ID breaks ties',
                    limits='block Belady with forced duplicate writes and chained prefix hits; not claimed globally optimal'),
                results=[],status='running')
    (output_dir/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    for budget in sorted({r['candidate']['context_budget'] for r in qualified},reverse=True):
        cfg=next(c for c in capacity['configurations'] if c['context_budget']==budget)
        workload=Path(cfg['workload_path'])
        if sha256_file(workload)!=cfg['workload_sha256']:raise ValueError('Workload fingerprint mismatch')
        by_root=defaultdict(list); estimates=defaultdict(list)
        for r in pq.read_table(workload,columns=[n for n in SCHEMA.names if n!='prompt_token_ids'],use_threads=False).to_pylist():
            by_root[r['root_id']].append(r)
        for r in pq.read_table(capacity_dir/f'schedule-{budget}.parquet').to_pylist():estimates[r['root_id']].append(r)
        roots={rid:RootSchedule(rows,estimates[rid]) for rid,rows in by_root.items()}
        blocks,_=load_block_ids(workload,16)
        for original in qualified:
            candidate=original['candidate']
            if candidate['context_budget']!=budget:continue
            rate=candidate['rate_per_minute']
            population=next(c for c in capacity['configurations'] if c['context_budget']==budget and c['rate_per_minute']==rate)
            instances=[SessionInstance(**r) for r in population['instances']]
            item=dict(id=original['directory'],candidate=candidate,policies={})
            rows_by_policy={}
            for policy in ('lru','oracle'):
                print('START',item['id'],policy,flush=True)
                summary,records=simulate(roots,instances,blocks,candidate['gpu_tokens'],candidate['cpu_tokens'],
                                         candidate['max_num_seqs'],cpu_policy=policy)
                folder=output_dir/item['id']/policy;folder.mkdir(parents=True)
                pq.write_table(pa.Table.from_pylist(records),folder/'requests.parquet',compression='zstd')
                result=dict(git_commit=commit,policy=policy,summary=summary,requests_sha256=sha256_file(folder/'requests.parquet'))
                (folder/'run.json').write_text(json.dumps(result,indent=2)+'\n')
                item['policies'][policy]=result
                rows_by_policy[policy]=records
                print('DONE',item['id'],policy,summary['source_fractions'],flush=True)
            # Refactoring the policy interface must not change prior LRU output.
            if item['policies']['lru']['summary']!=original['summary']:
                raise AssertionError('LRU regression against the committed baseline')
            left,right=rows_by_policy['lru'],rows_by_policy['oracle']
            assert [(r['instance_id'],r['trace_req_idx'],r['send_s']) for r in left]==[(r['instance_id'],r['trace_req_idx'],r['send_s']) for r in right]
            for name in ('lru','oracle'):
                item[f'{name}_recomputed_fraction']=item['policies'][name]['summary']['source_fractions']['recomputed']
            item['recompute_reduction_pp']=100*(item['lru_recomputed_fraction']-item['oracle_recomputed_fraction'])
            report['results'].append(item)
            if item['id']=='config-11':
                matched,matched_rows=simulate(roots,[i for i in instances if i.start_s<1200],blocks,
                    candidate['gpu_tokens'],candidate['cpu_tokens'],candidate['max_num_seqs'],
                    duration=1200,warmup=300,cpu_policy='lru')
                folder=output_dir/'target-matched-window';folder.mkdir()
                pq.write_table(pa.Table.from_pylist(matched_rows),folder/'requests.parquet',compression='zstd')
                report['target_matched_window']=dict(git_commit=commit,duration_s=1200,warmup_s=300,
                    summary=matched,requests_sha256=sha256_file(folder/'requests.parquet'))
                (folder/'run.json').write_text(json.dumps(report['target_matched_window'],indent=2)+'\n')
            (output_dir/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    report['results'].sort(key=lambda r:(-r['recompute_reduction_pp'],r['id']))
    target=next(r for r in report['results'] if r['id']=='config-11')
    report['noticeably_larger_than_target']=[r['id'] for r in report['results'] if r['recompute_reduction_pp']-target['recompute_reduction_pp']>=2.]
    report['status']='complete'
    (output_dir/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output_dir/'ranking.csv').open('w',newline='') as sink:
        fields=['id','context_budget','lambda','slots','cpu_gpu_ratio','lru_recompute_pct','oracle_recompute_pct','reduction_pp']
        writer=csv.DictWriter(sink,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in report['results']:
            c=r['candidate'];writer.writerow(dict(id=r['id'],context_budget=c['context_budget'],**{'lambda':c['rate_per_minute']},
              slots=c['max_num_seqs'],cpu_gpu_ratio=c['cpu_gpu_ratio'],lru_recompute_pct=r['lru_recomputed_fraction']*100,
              oracle_recompute_pct=r['oracle_recomputed_fraction']*100,reduction_pp=r['recompute_reduction_pp']))
    render(output_dir,report)
    return report


def render(directory,report):
    lines=['# CPU LRU / Oracle 对照','','14 组配置，使用同一固定种子人口、预计调度、16-token 块、s=0.5；40 分钟、预热 10 分钟。新旧 LRU 统计逐项一致；两种策略的实际发送序列相同。',
           '', 'Oracle 按每个已抽中实例全部剩余请求的预计到期时间计算下一次使用，包括窗口后的请求。已发出请求才从未来集合移除，排队但过期的请求仍在集合内。无限远优先驱逐，相同时间按块 ID 从大到小。GPU 仍用 LRU；没有改变负载或容量。链式前缀、GPU 副本与整请求写入约束下不宣称这是全局最优解，负差值也照实保留。',
           '', '| 配置 | 预算 / λ / 槽位 / CPU:GPU | LRU 重算 | Oracle 重算 | 减少百分点 |','|---|---|---:|---:|---:|']
    for r in report['results']:
        c=r['candidate'];lines.append(f"| {r['id']} | {c['context_budget']} / {c['rate_per_minute']:g} / {c['max_num_seqs']} / {c['cpu_gpu_ratio']:.3f} | {r['lru_recomputed_fraction']:.3%} | {r['oracle_recomputed_fraction']:.3%} | {r['recompute_reduction_pp']:.3f} |")
    target=next(r for r in report['results'] if r['id']=='config-11')
    lines+=['',f"指定 GPU 试跑配置 config-11 的 Oracle 空间为 {target['recompute_reduction_pp']:.3f} 个百分点。预先把‘明显更大’定义为比它再大至少2个百分点；符合项：{', '.join(report['noticeably_larger_than_target']) or '无'}。",'',
            'CPU 策略统一接在 `agentkv.sim.two_tier.CPUCachePolicy`：`on_access` 负责写入/访问，`evict(n, protected)` 返回恰好 n 个块或无副作用失败；`on_request` 提供请求级上下文，供 Oracle 维护未来访问，M5 可接同一接口。',
            '', '```bash','python -m agentkv.bench.compare_oracle --output-dir /tmp/new-oracle-run','```','',f"代码提交号：`{report['git_commit']}`。输入指纹、每策略逐请求输出和全部参数见 [run.json](run.json)，排序数据见 [ranking.csv](ranking.csv)。"]
    (directory/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'results/m1-two-tier-seed0')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--allow-dirty',action='store_true',help='DEBUG ONLY')
    args=parser.parse_args();compare(args.source_dir,args.output_dir,args.allow_dirty)


if __name__=='__main__':main()
