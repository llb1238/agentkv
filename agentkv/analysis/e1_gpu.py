"""E1 exit-only source attribution, exact stability and paired reporting."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import current_commit
from agentkv.bench.e0 import write_json,population_hash
from agentkv.bench.steady_state import sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.replay import join_observations,percentiles
from agentkv.bench.policy_overhead import merge_histograms
from agentkv.bench.shared_pilot import parse_dmon
from agentkv.analysis.stability import evaluate
from agentkv.analysis.recomputation import classify_records

LABELS=('lru','lru-final','online','perfect')


def read(path):return json.loads(path.read_text())


def verify_result(directory,result):
    paths=dict(client_run='run.json',server_manifest='server-manifest.json',raw_requests='requests.parquet',
               metrics='metrics.jsonl',dmon='dmon.log')
    for name,relative in paths.items():
        path=directory/relative
        if name=='raw_requests' and not path.exists():continue
        assert sha256_file(path)==result['inputs_sha256'][name],('Artifact changed',path)
    for name,digest in result['light_metrics_sha256'].items():
        assert sha256_file(directory/'server-observations'/name)==digest
    assert sha256_file(directory/'request-metadata.parquet')==result['request_metadata_sha256']
    assert sha256_file(directory/'recomputation.parquet')==result['recomputation_sha256']


def analyze(directory,roots,blocks,protocol,group,label):
    run=read(directory/'run.json');server=read(directory/'server-manifest.json')
    assert run['status']=='complete' and server['vllm_commit']==protocol['backend_commit']==run['vllm_commit']
    for name,value in dict(max_model_len=16384,max_num_seqs=8,gpu_cache_tokens=182880,cpu_cache_tokens=274320,
                           duration_s=2400,warmup_s=600,rate_per_minute=group['rate'],population_seed=group['seed']).items():
        assert server[name]==value,('Runtime configuration differs',name)
    configuration=read(directory/'policy-config.json')
    implementation='vllm builtin lru'
    if label=='lru':assert server['policy']=='lru'
    else:
        assert configuration['hint_tier']==('perfect' if label=='perfect' else 'online')
        assert (configuration.get('policy_variant')=='lru-final')==(label=='lru-final')
        marker='AGENTKV_HINT_POLICY_CONFIG '
        loaded=[json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0] for line in (directory/'server.log').read_text().splitlines() if marker in line]
        assert len(loaded)==1 and loaded[0]['sha256']==sha256_file(directory/'policy-config.json')
        implementation=loaded[0]['implementation']
        if protocol.get('policy_implementation')=='chain':
            assert configuration.get('implementation')=='chain' and implementation=='ChainPrefixAdapter'
    assert sha256_file(directory/'requests.parquet')==run['requests_sha256']
    schema=pq.read_schema(directory/'requests.parquet')
    table=pq.read_table(directory/'requests.parquet',columns=[k for k in schema.names if k!='prompt_token_ids'])
    rows=table.to_pylist();observations=join_observations(rows,directory/'server-observations')
    instances=sample_population({k:r.duration_s for k,r in roots.items()},group['rate'],group['seed'],2400,True)
    from dataclasses import asdict
    assert run['instances']==[asdict(i) for i in instances]
    assert run['source']=={k:protocol['source'][k] for k in ('workload_sha256','schedule_sha256')}
    prediction=next(c for c in protocol['predictions'] if c['group']==group['name'] and c['policy']==label)
    assert population_hash(instances)==prediction['population_sha256']
    logs=sorted((directory/'server-observations').glob('light-*.json'))
    counters=Counter();kv_counts=Counter();kv_observed=False;received_ids=set();window=[run['origin_ns']+600*10**9,run['origin_ns']+2400*10**9]
    for file in logs:
        data=read(file);assert data['closed']
        counters.update(data['counters'])
        received_ids.update(row['hint_message_id'] for row in data['requests'].values() if 'hint_message_id' in row)
        kv=data['kv_blocking_window']
        if kv['observed']:
            assert kv['window_ns']==window
            kv_observed=True;kv_counts.update({int(k):v for k,v in kv['running_counts'].items()})
        for scope in ('histograms','window_histograms'):
            for h in data[scope].values():assert sum(h['buckets'].values())==h['count']
    assert counters['effective_observer_ready']
    assert not counters['effective_invalid_load_batches'] and not counters['effective_missing_model_rows']
    success=[r for r in rows if r['in_window'] and r['status']=='success']
    if success:assert counters['effective_completed_batches']
    assert {r['request_id'] for r in success}<=received_ids,'A primary-window request lost its hint'
    for r in success:
        assert all(r.get(k) is not None for k in ('gpu_hit','cpu_hit','recomputed',
            'recomputed_effective','prefill_forward_token_work','replay_overlap','repeated_prompt_work',
            'effective_unattributed_cache_tokens')),'Missing effective observation'
        assert r['gpu_hit']+r['cpu_hit']+r['recomputed']==r['prompt_len']
        assert r['effective_ranges_complete'],'Missing forward ranges without an observed preemption'
        if not r['effective_unattributed_cache_tokens']:
            assert r['gpu_hit_effective']+r['cpu_hit_effective']+r['recomputed_effective']==r['prompt_len']
        else:assert r['gpu_hit_effective'] is None and r['cpu_hit_effective'] is None
        assert r['prefill_forward_token_work']==r['recomputed_effective']+r['repeated_prompt_work']
        assert r['recomputed_effective']-r['recomputed']==r['replay_overlap']-r['effective_unattributed_cache_tokens']
        assert r['recomputed_effective']>=1
    prompt=sum(r['prompt_len'] for r in success)
    metrics=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
    stable=evaluate(metrics,rows,origin_ns=run['origin_ns'],duration_s=2400,warmup_s=600,max_num_seqs=8,
        kv_summary=dict(window_ns=window,observed=kv_observed,running_counts=dict(kv_counts)))
    categories,classified=classify_records(rows,roots,blocks,real=True)
    assert categories['requests']==len(success)
    assert categories['gap_requests']==categories['gap_queue_observed_requests']
    transport=observations['valid_hints_received'];sent=run['summary']['hint_sent']
    if all(r['status']=='success' for r in rows):assert sent==transport,'Hint transport count mismatch without client failures'
    if label!='lru':
        assert counters['valid_hint_requests_finished']<=counters['valid_hint_requests_any']<=transport
        if all(r['status']=='success' for r in rows):assert counters['valid_hint_requests_finished']==sent,'Policy hint count mismatch'
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),directory/'request-metadata.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(classified),directory/'recomputation.parquet',compression='zstd')
    complete=[r for r in rows if r['status']=='success' and r.get('end_ns') is not None and window[0]<=r['end_ns']<=window[1]]
    dmon=parse_dmon((directory/'dmon.log').read_text());assert {r['gpu'] for r in dmon}=={0,1}
    result=dict(git_commit=run['git_commit'],vllm_commit=run['vllm_commit'],status='complete',policy=label,
        protocol_source=protocol['source'],population_sha256=prediction['population_sha256'],stability=stable,
        window_sent=sum(r['in_window'] for r in rows),window_success=len(success),window_failed=sum(r['in_window'] and r['status']!='success' for r in rows),
        window_completed_all=len(complete),window_completed_non_entry=sum(not r['is_entry'] for r in complete),
        effective_recompute=sum(r['prefill_forward_token_work'] for r in success)/prompt if prompt else None,
        policy_implementation=implementation,policy_config_sha256=sha256_file(directory/'policy-config.json'),
        effective_unique_recompute=sum(r['recomputed_effective'] for r in success)/prompt if prompt else None,
        nominal_recompute=sum(r['recomputed'] for r in success)/prompt if prompt else None,
        replay_overlap_tokens=sum(r['replay_overlap'] for r in success),repeated_prompt_work=sum(r['repeated_prompt_work'] for r in success),
        effective_device_attribution_missing_requests=sum(r['effective_unattributed_cache_tokens']>0 for r in success),
        effective_unattributed_cache_tokens=sum(r['effective_unattributed_cache_tokens'] for r in success),
        preempted_primary_requests=sum(r['prefill_observed_preemptions']>0 for r in success),
        ttft_ms=percentiles([r['ttft_ms'] for r in success]),recomputation=categories,
        predicted_effective_recompute=prediction['predicted_effective_recompute'],
        prediction_gap_pp=100*(sum(r['prefill_forward_token_work'] for r in success)/prompt-prediction['predicted_effective_recompute']) if prompt else None,
        hint_counts=dict(sent=sent,transport_received=transport,match=sent==transport,
            policy_valid_any=None if label=='lru' else counters['valid_hint_requests_any'],
            policy_valid_finished=None if label=='lru' else counters['valid_hint_requests_finished']),
        histograms=merge_histograms(logs),all_histograms=merge_histograms(logs,'histograms'),
        inputs_sha256=dict(client_run=sha256_file(directory/'run.json'),server_manifest=sha256_file(directory/'server-manifest.json'),
            raw_requests=sha256_file(directory/'requests.parquet'),metrics=sha256_file(directory/'metrics.jsonl'),dmon=sha256_file(directory/'dmon.log')),
        light_metrics_sha256={p.name:sha256_file(p) for p in logs},request_metadata_sha256=sha256_file(directory/'request-metadata.parquet'),
        recomputation_sha256=sha256_file(directory/'recomputation.parquet'))
    write_json(directory/'e1-analysis.json',result)
    return result


def mean_range(values):
    return dict(n=len(values),mean=math.fsum(values)/len(values),minimum=min(values),maximum=max(values)) if values else dict(n=0,mean=None,minimum=None,maximum=None)


def aggregates(rows,labels=LABELS,rates=(1.,1.4)):
    output=[]
    for rate in rates:
        for label in labels:
            own=[r for r in rows if r['rate']==rate and r['policy']==label]
            stable=[r for r in own if r.get('stable')]
            paired=[r for r in stable if r.get('paired_eligible')]
            item=dict(rate=rate,policy=label,attempted=len(own),stable_count=len(stable),
                effective_recompute=mean_range([r['effective_recompute'] for r in stable]),
                ttft_p50_ms=mean_range([r['ttft_p50_ms'] for r in stable]),ttft_p99_ms=mean_range([r['ttft_p99_ms'] for r in stable]),
                paired_recompute_pp=mean_range([r['paired_recompute_pp'] for r in paired]),
                paired_ttft_p50_ms=mean_range([r['paired_ttft_p50_ms'] for r in paired]),
                paired_ttft_p99_ms=mean_range([r['paired_ttft_p99_ms'] for r in paired]),
                prediction_gap_all_pp=mean_range([r['prediction_gap_pp'] for r in own if r.get('prediction_gap_pp') is not None]))
            item['completed_all']=mean_range([r['completed'] for r in own if r.get('complete')])
            item['completed_non_entry']=mean_range([r['completed_non_entry'] for r in own if r.get('complete')])
            output.append(item)
    return output


def render(output,state,protocol,*,summary_only=False,saved_common_checks=None,saved_plots=None):
    labels=protocol.get('policies',LABELS)
    rates=sorted({g['rate'] for g in protocol['groups']})
    queue_rate=protocol.get('queue_rate',1.4)
    queue_stem=f'lambda{queue_rate:g}_queues'.replace('.','')
    common_inputs=list(saved_common_checks or []) if summary_only else []
    if not summary_only:
        for group in protocol['groups']:
            complete=[r for r in state['runs'] if r['group']==group['name'] and r['status']=='complete']
            if len(complete)<2:continue
            reference=complete[0]
            def tokens(item):
                columns=['request_id','sent_prompt_sha256','permutation_seed','prompt_len','max_tokens']
                return {r['request_id']:tuple(r[k] for k in columns[1:]) for r in
                    pq.read_table(output/item['selected_attempt']/'request-metadata.parquet',columns=columns).to_pylist()}
            first=tokens(reference)
            for item in complete[1:]:
                second=tokens(item);shared=set(first)&set(second)
                assert all(first[k]==second[k] for k in shared),'Common request tokens changed across policies'
                common_inputs.append(dict(group=group['name'],left='previous_completed_policies',right=item['policy'],requests=len(shared),identical=True))
                first.update(second)
    rows=[]
    for item in state['runs']:
        base={k:item[k] for k in ('id','rate','seed','policy','status')}
        forecast=next(c for c in protocol['predictions'] if c['group']==item['group'] and c['policy']==item['policy'])
        base['predicted']=forecast['predicted_effective_recompute']
        r=item.get('result');base.update(complete=r is not None,stable=bool(r and r['stability']['stable']))
        if r:
            base.update(effective_recompute=r['effective_recompute'],ttft_p50_ms=r['ttft_ms']['p50'],ttft_p99_ms=r['ttft_ms']['p99'],
                completed=r['window_completed_all'],completed_non_entry=r['window_completed_non_entry'],
                predicted=r['predicted_effective_recompute'],prediction_gap_pp=r['prediction_gap_pp'],
                stability=r['stability']['status'],attempts=len(item['attempts']))
            baseline=next((a.get('result') for a in state['runs'] if a['rate']==item['rate'] and a['seed']==item['seed'] and a['policy']=='lru'),None)
            base['paired_eligible']=bool(r['stability']['stable'] and baseline and baseline['stability']['stable'])
            for field in ('recompute_pp','ttft_p50_ms','ttft_p99_ms'):base['paired_'+field]=None
            if base['paired_eligible']:
                base.update(paired_recompute_pp=100*(r['effective_recompute']-baseline['effective_recompute']),
                    paired_ttft_p50_ms=r['ttft_ms']['p50']-baseline['ttft_ms']['p50'],paired_ttft_p99_ms=r['ttft_ms']['p99']-baseline['ttft_ms']['p99'])
        rows.append(base)
    hint_rows=[];timing_rows=[]
    for item in state['runs']:
        if not item.get('result'):continue
        r=item['result'];identity={k:item[k] for k in ('rate','seed','policy')}
        hint_rows.append(dict(identity,**r['hint_counts']))
        for name,h in r['histograms'].items():
            timing_rows.append(dict(identity,callback=name,count=h['count'],
                p50_lower_us=h['p50_ns'][0]/1000,p50_upper_us=h['p50_ns'][1]/1000,
                p99_lower_us=h['p99_ns'][0]/1000,p99_upper_us=h['p99_ns'][1]/1000,
                mean_us=h['total_ns']/h['count']/1000,maximum_us=h['maximum_ns']/1000))
    for filename,values in [('hint_counts.csv',hint_rows),('callback_timings.csv',timing_rows)]:
        if values:
            with (output/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    summary=aggregates(rows,labels,rates)
    write_json(output/'summary.json',dict(git_commit=current_commit(),status=state['status'],rows=rows,aggregates=summary,
        preregistration_commit=state['preregistration_commit'],protocol_sha256=state['protocol_sha256'],common_token_checks=common_inputs))
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (output/'runs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(rates),figsize=(7.1,3.))
    for rate,ax in zip(rates,np.atleast_1d(axes)):
        for i,label in enumerate(labels):
            group=[r for r in rows if r['rate']==rate and r['policy']==label and r.get('effective_recompute') is not None]
            for r in group:
                offset=(r['seed']-4)*.12
                ax.scatter(i+offset,100*r['effective_recompute'],marker='o' if r['stable'] else 'x',color='#0072B2' if r['stable'] else '#777777',s=18)
            for forecast in protocol['predictions']:
                if forecast['rate']==rate and forecast['policy']==label:
                    ax.scatter(i+(forecast['seed']-4)*.12,100*forecast['predicted_effective_recompute'],marker='D',facecolors='none',edgecolors='#D55E00',s=20)
        names={'lru':'LRU','lru-final':'LRU+final','online':'Online','perfect':'Perfect'}
        ax.set_xticks(range(len(labels)),[names[label] for label in labels],rotation=15)
        ax.set(title=f'lambda={rate:g}',ylabel='Recomputed prompt work (%)');ax.grid(axis='y',alpha=.2)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',ls='',color='#0072B2',label='Measured stable'),Line2D([],[],marker='x',ls='',color='#777777',label='Measured unstable/unknown'),
             Line2D([],[],marker='D',ls='',color='#D55E00',markerfacecolor='none',label='Simulation')]
    fig.legend(handles=handles,loc='upper center',ncol=3,fontsize=7);fig.tight_layout(rect=(0,0,1,.9))
    for ext in ('png','pdf'):fig.savefig(output/f'recomputation.{ext}',dpi=300)
    plt.close(fig)
    if summary_only:
        # These empirical CDF/queue plots require raw requests. Retain their
        # original bytes and label them as saved figures, never regenerate
        # synthetic samples from aggregate percentiles.
        import shutil
        for stem in ('ttft_cdf',queue_stem):
            for ext in ('png','pdf'):
                original=Path(saved_plots)/f'{stem}.{ext}'
                shutil.copyfile(original,output/original.name)
    else:
        fig,axes=plt.subplots(1,len(rates),figsize=(7.1,3.))
        for rate,ax in zip(rates,np.atleast_1d(axes)):
            for label in labels:
                kept=[r for r in state['runs'] if r['rate']==rate and r['policy']==label and r.get('result',{}).get('stability',{}).get('stable')]
                values=[];weights=[]
                for item in kept:
                    requests=pq.read_table(output/item['selected_attempt']/'request-metadata.parquet',columns=['in_window','status','ttft_ms']).to_pylist()
                    samples=[r['ttft_ms'] for r in requests if r['in_window'] and r['status']=='success']
                    values.extend(samples);weights.extend([1/(len(kept)*len(samples))]*len(samples))
                if values:
                    order=np.argsort(values);ax.plot(np.asarray(values)[order],np.cumsum(np.asarray(weights)[order]),lw=1,label=f'{label} (n={len(kept)})')
            ax.set(xscale='log',xlabel='TTFT (ms)',ylabel='CDF',title=f'lambda={rate:g}',ylim=(0,1));ax.grid(alpha=.2)
            if ax.lines:ax.legend(frameon=False,fontsize=6)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'ttft_cdf.{ext}',dpi=300)
        plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(7.1,5.2),sharex=True)
        for seed,ax in zip((3,4,5),axes):
            for item in state['runs']:
                if item['rate']!=queue_rate or item['seed']!=seed or item['status']!='complete':continue
                samples=[json.loads(s) for s in (output/item['selected_attempt']/'metrics.jsonl').read_text().splitlines()]
                good=[s for s in samples if 'vllm:num_requests_waiting' in s.get('values',{})]
                ax.plot([s['elapsed_s']/60 for s in good],[s['values']['vllm:num_requests_waiting'] for s in good],lw=.65,label=item['policy'])
            ax.axvspan(0,10,color='gray',alpha=.15);ax.set(ylabel=f'Seed {seed}: waiting');ax.grid(alpha=.2)
            if ax.lines:ax.legend(frameon=False,ncol=4,fontsize=7)
        axes[-1].set(xlabel='Elapsed minutes',xlim=(0,40));fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'{queue_stem}.{ext}',dpi=300)
        plt.close(fig)
    def text(v,scale=1):return 'NA' if not v['n'] else f"{v['mean']*scale:.4f} [{v['minimum']*scale:.4f}, {v['maximum']*scale:.4f}], n={v['n']}"
    total=sum(len(g['order']) for g in protocol['groups'])
    lines=['# E1 GPU config-11','',f"状态：{state['status']}；预注册 `{state['preregistration_commit']}`；已完成{sum(r['status']=='complete' for r in state['runs'])}/{total}。",'',
        '每个人口一个点，报告均值及最小–最大；3个人口不做显著性检验。主指标仅稳定运行；配对差值要求策略和同人口LRU都稳定。负差表示策略较低。不稳定/失败不删除。有效重算为实际前向prompt工作量，名义三分类和重复工作另列于逐次e1-analysis.json。','']
    for rate in rates:
        lines += [f'## λ={rate:g}','','| 策略 | 重算%（稳定） | TTFT P50 ms | TTFT P99 ms | Δ重算pp | ΔP50 ms | ΔP99 ms | 稳定次数/3 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for a in summary:
            if a['rate']!=rate:continue
            lines.append(f"| {a['policy']} | {text(a['effective_recompute'],100)} | {text(a['ttft_p50_ms'])} | {text(a['ttft_p99_ms'])} | {text(a['paired_recompute_pp'])} | {text(a['paired_ttft_p50_ms'])} | {text(a['paired_ttft_p99_ms'])} | {a['stable_count']}/3 |")
    lines+=['',f'## λ={queue_rate:g}完成数（含不稳定运行）','','| 策略 | 稳定/3 | 全部完成数 | 非is_entry完成数 |','|---|---:|---:|---:|']
    for a in summary:
        if a['rate']==queue_rate:lines.append(f"| {a['policy']} | {a['stable_count']}/3 | {text(a['completed_all'])} | {text(a['completed_non_entry'])} |")
    lines+=['','## 全部运行及模拟差距','','| λ | 种子 | 策略 | 状态 | 稳定性 | 窗口完成/有效完成 | 实测重算% | 模拟% | 差pp |','|---:|---:|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r.get('complete'):
            actual='NA' if r['effective_recompute'] is None else f"{100*r['effective_recompute']:.4f}"
            gap='NA' if r['prediction_gap_pp'] is None else f"{r['prediction_gap_pp']:+.4f}"
            lines.append(f"| {r['rate']:g} | {r['seed']} | {r['policy']} | {r['status']} | {r['stability']} | {r['completed']}/{r['completed_non_entry']} | {actual} | {100*r['predicted']:.4f} | {gap} |")
        else:lines.append(f"| {r['rate']:g} | {r['seed']} | {r['policy']} | {r['status']} | NA | NA | NA | {100*r['predicted']:.4f} | NA |")
    lines+=['','![实测和模拟](recomputation.png)','','![TTFT CDF](ttft_cdf.png)','','CDF按稳定人口等权混合，主表仍为每人口分位数的等权均值，不把混合CDF分位数冒充配对均值。','', f'![λ={queue_rate:g}等待队列]({queue_stem}.png)','',
        '[逐次数据](runs.csv) · [hint对账](hint_counts.csv) · [轻量回调耗时](callback_timings.csv) · [机器汇总](summary.json)。', '', '窗口完成数按结束时刻计，主指标按发送时刻入窗；全部与非is_entry完成数都报告。实际闭环请求集合可能不同，共同请求token哈希在组内核对。CPU模拟固定0.5秒、零搬运，不模拟真实TTFT或服务时间依赖。',
        '', '断点续跑使用原持久输出目录：`python -m agentkv.bench.e1_gpu run --output-dir <持久目录> --gpu-index 1 --port 8016`；v2另加`--protocol docs/e1_v2_preregistration.json`。`--max-groups 3`可每晚最多跑3组。状态用`python -m agentkv.bench.e1_gpu status --output-dir <持久目录>`。',
        '', 'checkpoint.json保留每次attempt、原因和选中成功产物SHA；失败不覆盖，第二次非策略失败后停止该配置。未执行显著性检验。']
    if summary_only:
        lines += ['', '精简发布版：表格由保留的e1-analysis.json重新计算；共同token核验为原报告存档结论，本次未重查逐请求证据。TTFT CDF与队列图直接保留原图，不由分位数伪造。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
