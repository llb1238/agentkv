"""CPU-only remaining-life E1 matrix and independent artifact validation."""
import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import ROOT,WORKLOAD,SCHEDULE,RATES,SEEDS,load_inputs,population_hash,write_json
from agentkv.bench.e1_simulation import _schedule_fingerprint,save_plot
from agentkv.bench.steady_state import SessionInstance,sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.remaining_life import DEFAULT_TABLE,RemainingLifeTable
from agentkv.sim.hint_adapter import HintPolicyAdapter
from agentkv.sim.two_tier import simulate

E0=ROOT/'results/m1-e0-config11-20261103'
PREVIOUS=ROOT/'results/m5-e1-simulation-config11-20261104'
MAIN=('lru','lru-final','perfect','noisy-deadline','noisy-remaining','online-deadline','online-remaining','oracle')
# All options are explicit so future defaults cannot change this experiment.
CASES=[('lru',None,{}),('oracle',None,{}),('lru-final','final',dict(scoring='lru_final',tail_first=False,final_first=True,timeout_decay=False)),
       ('perfect','perfect',dict(scoring='deadline',tail_first=False,final_first=True,timeout_decay=True))]
for tier in ('noisy','online'):
    CASES += [(f'{tier}-{mode}',tier,dict(scoring=mode,tail_first=False,final_first=True,timeout_decay=mode=='deadline')) for mode in ('deadline','remaining')]
CASES += [(f'legacy-{tier}',tier,dict(scoring='deadline',tail_first=True,final_first=True,timeout_decay=True)) for tier in ('perfect','noisy','online')]
CASES += [(f'tail-{tier}',tier,dict(scoring='remaining',tail_first=True,final_first=True,timeout_decay=False)) for tier in ('noisy','online')]
_ROOTS=_BLOCKS=_TABLES=_TABLE_SHA=None


def read(path):return json.loads(path.read_text())
def average(values):return math.fsum(values)/len(values)
def fraction(case):return case['summary']['source_fractions']['recomputed']


def calibration_values(model):
    # Re-fitting records a new Git commit / manifest, but the fixed experiment
    # requires identical training membership, parameters and numerical curves.
    return {key:value for key,value in model.items() if key not in ('git_commit','source')}


def _initialize(table_path=DEFAULT_TABLE):
    global _ROOTS,_BLOCKS,_TABLES,_TABLE_SHA
    _ROOTS,_BLOCKS=load_inputs()
    model=read(table_path)
    _TABLES={tier:RemainingLifeTable(model,tier) for tier in ('online','noisy')}
    _TABLE_SHA=sha256_file(table_path)


def _population(arguments):
    item,output,commit=arguments
    name=item['name'];folder=output/name;folder.mkdir()
    original=read(PREVIOUS/name/'run.json');instances=[SessionInstance(**r) for r in item['models']['lru']['instances']]
    rebuilt=sample_population({rid:r.duration_s for rid,r in _ROOTS.items()},item['rate'],item['seed'],2400,True)
    assert population_hash(instances)==population_hash(rebuilt)==item['population_sha256']==original['population_sha256']
    result=dict(name=name,rate=item['rate'],seed=item['seed'],population_sha256=item['population_sha256'],cases={})
    reference=None;fingerprint=None
    for label,tier,options in CASES:
        diagnose=label=='legacy-online' and item['rate'] in (1.,1.4)
        start=time.perf_counter()
        settings=dict(options)
        if options.get('scoring')=='remaining':settings['residual_table']=_TABLES[tier]
        policy=HintPolicyAdapter(274320//16,hint_tier=tier,diagnose_deadlines=diagnose,**settings) if tier else label
        summary,rows=simulate(_ROOTS,instances,_BLOCKS,182880,274320,8,cpu_policy=policy)
        elapsed=time.perf_counter()-start;current=_schedule_fingerprint(rows)
        if fingerprint is None:fingerprint=current;reference=rows
        assert current==fingerprint,('Schedule/GPU regression',name,label)
        if label in ('lru','oracle'):
            expected=item['models'][label]
            assert summary=={k:v for k,v in expected['summary'].items() if k!='gap_eviction_audit'}
            old=pq.read_table(E0/name/'models'/label/'requests.parquet').to_pylist()
            assert sha256_file(E0/name/'models'/label/'requests.parquet')==expected['requests_sha256']
            assert rows==[{k:r[k] for k in rows[0]} for r in old]
        previous_label='hint-'+label[7:] if label.startswith('legacy-') else 'no-tail-'+label.split('-')[0] if label in ('perfect','noisy-deadline','online-deadline') else None
        if previous_label:
            expected=original['cases'][previous_label]
            path=PREVIOUS/name/previous_label/'requests.parquet'
            assert sha256_file(path)==expected['requests_sha256']
            assert rows==pq.read_table(path).to_pylist() and summary==expected['summary'],('Previous E1 regression',name,label)
        if tier:assert policy.policy.stats['valid_hint_requests']==len(rows)
        destination=folder/label;destination.mkdir()
        pq.write_table(pa.Table.from_pylist(rows),destination/'requests.parquet',compression='zstd')
        record=dict(git_commit=commit,status='complete',label=label,hint_tier=tier,options=options,
            table_sha256=_TABLE_SHA if options.get('scoring')=='remaining' else None,
            population_sha256=result['population_sha256'],schedule_gpu_sha256=current,summary=summary,
            elapsed_wall_s=elapsed,policy_stats=dict(policy.policy.stats) if tier else None,
            requests_sha256=sha256_file(destination/'requests.parquet'))
        if diagnose:
            pq.write_table(pa.Table.from_pylist(policy.deadline_decisions),destination/'deadline_decisions.parquet',compression='zstd')
            record['deadline_decisions_sha256']=sha256_file(destination/'deadline_decisions.parquet')
        write_json(destination/'run.json',record);result['cases'][label]=record
        print(name,label,f'{fraction(record):.6%}',f'{elapsed:.2f}s',flush=True)
        del policy;gc.collect()
    write_json(folder/'population.json',dict(instances=item['models']['lru']['instances'],population_sha256=result['population_sha256']))
    write_json(folder/'run.json',dict(git_commit=commit,status='complete',**result));return result


def paired_rows(report):
    pairs=[]
    for r in report['results']:
        lru=fraction(r['cases']['lru']);oracle=fraction(r['cases']['oracle'])
        for label in MAIN:
            if label in ('lru','oracle'):continue
            value=fraction(r['cases'][label])
            pairs.append(dict(name=r['name'],rate=r['rate'],seed=r['seed'],strategy=label,recomputed=value,
                reduction_pp=100*(lru-value),oracle_capture=(lru-value)/(lru-oracle) if lru!=oracle else None))
    return pairs


def aggregate_pairs(pairs):
    output=[]
    for rate in RATES:
        for label in MAIN:
            group=[r for r in pairs if r['rate']==rate and r['strategy']==label]
            if not group:continue
            values=[r['reduction_pp'] for r in group];capture=[r['oracle_capture'] for r in group if r['oracle_capture'] is not None]
            output.append(dict(rate=rate,strategy=label,populations=len(group),mean_recomputed=average([r['recomputed'] for r in group]),
                mean_pp=average(values),minimum_pp=min(values),maximum_pp=max(values),
                capture_mean=average(capture) if capture else None,capture_min=min(capture) if capture else None,capture_max=max(capture) if capture else None))
    return output


def decision_stats(rows):
    if not rows:return dict(decisions=0)
    fractions=[r['overdue_fraction'] for r in rows];total=sum(r['cpu_blocks'] for r in rows)
    return dict(decisions=len(rows),mean_fraction=average(fractions),median_fraction=float(np.median(fractions)),
        p90_fraction=float(np.quantile(fractions,.9)),pooled_fraction=sum(r['overdue_blocks'] for r in rows)/total if total else 0.)


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def render(directory,report):
    pairs=paired_rows(report);aggregate=aggregate_pairs(pairs);diagnostics=[];controls=[]
    for r in report['results']:
        if r['rate'] in (1.,1.4):
            decisions=pq.read_table(directory/r['name']/'legacy-online/deadline_decisions.parquet').to_pylist()
            window=[v for v in decisions if 600<=v['time_s']<2400]
            diagnostics.append(dict(name=r['name'],rate=r['rate'],seed=r['seed'],**decision_stats(window)))
        for tier in ('noisy','online'):
            new=fraction(r['cases'][tier+'-remaining']);old=fraction(r['cases'][tier+'-deadline']);tail=fraction(r['cases']['tail-'+tier])
            controls.append(dict(name=r['name'],rate=r['rate'],seed=r['seed'],tier=tier,old_to_remaining_reduction_pp=100*(old-new),tail_enabled_effect_pp=100*(tail-new)))
    for name,rows in [('paired',pairs),('aggregate',aggregate),('deadline_diagnostics',diagnostics),('controls',controls)]:write_csv(directory/(name+'.csv'),rows)
    lines=['# E1：剩余寿命打分与final基线（纯CPU）','',f"状态：{report['status']}；测量提交 `{report['git_commit']}`；预注册 `{report['preregistration_commit']}`。",'',
        'config-11：16K、8槽位、GPU182880/CPU274320 token；T=40分钟、预热10分钟、固定服务0.5秒、零传输耗时；沿用E0全部12个人口，不按真实GPU稳定性筛选。每人口按原始prompt token加权，排除is_entry和预热，再对三个种子等权平均。没有调用GPU/vLLM，不推断TTFT或真实策略开销。',
        '', '## 预测误差','', '在回放负载全部非final请求上逐间隔计数，不按人口重复加权。实际间隔为截断、压缩后的gap_after_s；比值只对实际间隔>0定义，零间隔另计。','',
        '| 档位 | 正间隔数 | P10 | P50 | P90 | P99 | [0.5,2]占比 | 实际为0 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    errors=read(directory/'calibration/prediction_diagnostics.json')
    for tier,d in errors.items():
        q=d['quantiles'];lines.append(f"| {tier} | {d['positive_actual_intervals']} | {q['p10']:.3f} | {q['p50']:.3f} | {q['p90']:.3f} | {q['p99']:.3f} | {d['within_half_double_fraction']:.2%} | {d['zero_actual_intervals']} |")
    lines+=['','## 旧online策略驱逐时的过deadline占用','',
        '以下只统计600≤t<2400秒的驱逐调用，每次选受害块前采样；链返回以模拟器实际开始执行请求为界，包含已到期但仍排队的情况。分子是至少属于一条waiting且now>原始deadline链的驻留块并集，共享块只计一次，可能仍被其他未超时链引用；不是“可以直接释放”的比例。均值/分位数按决策等权；合并占比=sum(超时块)/sum(CPU块)，不是时间加权。完整模拟全部决策另存各legacy-online/deadline_decisions.parquet。','',
        '| λ | seed | 决策数 | 均值 | 中位数 | P90 | 合并占比 |','|---|---|---:|---:|---:|---:|---:|']
    for d in diagnostics:lines.append(f"| {d['rate']:g} | {d['seed']} | {d['decisions']} | {d['mean_fraction']:.2%} | {d['median_fraction']:.2%} | {d['p90_fraction']:.2%} | {d['pooled_fraction']:.2%} |")
    model=read(directory/'calibration/remaining_life_v1.json')
    lines+=['','## 校准与默认配置','',f"训练集为未抽中的{len(model['training_root_ids'])}个根会话、{model['interval_count']}个链内间隔，与100个回放根无交集。online/noisy分别有"+
        '/'.join(str(len(model['tiers'][t]['buckets'])) for t in ('online','noisy'))+'个预测分位桶。每桶以G>e的样本计算剩余寿命，e=0..120秒、步长1秒，左端点查表，无存活样本退回LRU；不外推。训练对、存活数及均值表均保留。',
        '默认final开启、尾部关闭；perfect仍使用原deadline，noisy/online使用剩余寿命。旧deadline+超时降级与尾部开关保留。新旧打分对照使用相同的尾部关闭配置；legacy额外回归上一轮尾部开启的三档输出。LRU+final完全不读取间隔预测，非final按LRU，final为无穷大，共享块仍取所有拥有者最早分数。',
        '', '## 重算比例（三种子均值）','', '| λ | '+' | '.join(MAIN)+' |','|---|'+'---:|'*len(MAIN)]
    for rate in RATES:
        group=[r for r in report['results'] if r['rate']==rate]
        if group:lines.append(f'| {rate:g} | '+' | '.join(f"{average([fraction(r['cases'][label]) for r in group]):.3%}" for label in MAIN)+' |')
    lines+=['','## 配对减少百分点与Oracle空间捕获','', '减少百分点=LRU−策略。捕获率逐人口计算(LRU−策略)/(LRU−Oracle)，再等权平均；不截断负值，分母为0记NA。括号为三个种子的最小–最大。','',
        '| λ | 策略 | 减少百分点均值（范围） | Oracle捕获率均值（范围） |','|---|---|---:|---:|']
    for a in aggregate:
        cap='NA' if a['capture_mean'] is None else f"{a['capture_mean']:.2%} ({a['capture_min']:.2%}–{a['capture_max']:.2%})"
        lines.append(f"| {a['rate']:g} | {a['strategy']} | {a['mean_pp']:+.3f} ({a['minimum_pp']:+.3f}–{a['maximum_pp']:+.3f}) | {cap} |")
    lines+=['','## online逐人口验收','', '| λ | seed | LRU重算 | 剩余寿命重算 | 减少百分点 | ≥0 |','|---|---|---:|---:|---:|---|']
    for p in pairs:
        if p['strategy']=='online-remaining':
            r=next(r for r in report['results'] if r['name']==p['name'])
            lines.append(f"| {p['rate']:g} | {p['seed']} | {fraction(r['cases']['lru']):.3%} | {p['recomputed']:.3%} | {p['reduction_pp']:+.3f} | {'通过' if p['reduction_pp']>=0 else '失败'} |")
    lines+=['','## 旧打分与尾部组件控制','', '旧→剩余寿命=旧重算−新重算（正数表示新打分更好）；尾部影响=开启尾部−默认关闭（正数表示开启更差）。两者都固定final开启。','',
        '| λ | 档位 | 旧→剩余寿命均值（范围），pp | 开启尾部均值（范围），pp |','|---|---|---:|---:|']
    for rate in RATES:
        for tier in ('noisy','online'):
            group=[v for v in controls if v['rate']==rate and v['tier']==tier]
            if not group:continue
            columns=[]
            for field in ('old_to_remaining_reduction_pp','tail_enabled_effect_pp'):
                v=[d[field] for d in group];columns.append(f'{average(v):+.3f} ({min(v):+.3f}–{max(v):+.3f})')
            lines.append(f'| {rate:g} | {tier} | '+' | '.join(columns)+' |')
    lines+=['','所有人口与GPU/调度路径一致性、旧E0和E1逐请求回归及SHA256核验见[validation.json](validation.json)。Oracle只是当前两层前缀模型中的CPU未来访问启发式，不保证全局最优。新打分的表格来自校准集，无测试人口调参；验收失败时保留原结果。',
       '', '[配对原始值](paired.csv) · [均值/范围](aggregate.csv) · [驱逐诊断](deadline_diagnostics.csv) · [控制](controls.csv) · [run.json](run.json) · [产物指纹](artifacts.json) · [预注册](../../docs/m5_remaining_life_design.md)',
       '', '![收益](recomputed.png)','','![诊断](diagnostics.png)','', '复现（从干净工作区，在Git外生成结果）：','', '```bash',
       "python -m pip install -e '.[simulation]'",
       'python -m agentkv.bench.calibrate_remaining_life --output-dir /tmp/new-calibration',
       'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m agentkv.bench.e1_remaining --output-dir /tmp/new-e1-remaining --calibration-dir /tmp/new-calibration --workers 4',
       'python -m agentkv.bench.e1_remaining --validate /tmp/new-e1-remaining','```']
    (directory/'README.md').write_text('\n'.join(lines)+'\n');plots(directory,report)


def plots(directory,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.1,3.4))
    for label in MAIN:
        x=[];y=[]
        for rate in RATES:
            group=[r for r in report['results'] if r['rate']==rate]
            if group:x.append(rate);y.append(100*average([fraction(r['cases'][label]) for r in group]))
        ax.plot(x,y,'o-',label=label,ms=3,lw=1)
    ax.set(xlabel='Root arrivals / minute',ylabel='Recomputed prompt tokens (%)');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2,frameon=False)
    fig.tight_layout();save_plot(fig,directory,'recomputed');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    rows=pq.read_table(directory/'calibration/prediction_pairs.parquet').to_pylist()
    for tier in ('online','noisy'):
        v=sorted(r[tier+'_s']/r['actual_gap_s'] for r in rows if r['actual_gap_s']>0)
        axes[0].plot(v,np.arange(1,len(v)+1)/len(v),label=tier)
    axes[0].set(xscale='log',xlabel='Predicted / actual gap',ylabel='CDF');axes[0].axvspan(.5,2,alpha=.1,color='green')
    for rate in (1.,1.4):
        values=[]
        for r in report['results']:
            if r['rate']==rate:
                rows=pq.read_table(directory/r['name']/'legacy-online/deadline_decisions.parquet').to_pylist()
                values.extend(100*d['overdue_fraction'] for d in rows if 600<=d['time_s']<2400)
        if values:
            values.sort();axes[1].plot(values,np.arange(1,len(values)+1)/len(values),label=f'lambda={rate:g}')
    axes[1].set(xlabel='CPU blocks owned by overdue chains (%)',ylabel='CDF')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7,frameon=False)
    fig.tight_layout();save_plot(fig,directory,'diagnostics');plt.close(fig)


def validate(directory):
    """Re-read raw rows; independently check counts, tables and paired arithmetic."""
    commit=git_commit();run=read(directory/'run.json');assert run['status']=='complete'
    assert len(run['results'])==12 and {(r['rate'],r['seed']) for r in run['results']}=={(x,y) for x in RATES for y in SEEDS}
    assert run['source']['workload_sha256']==sha256_file(WORKLOAD) and run['source']['schedule_sha256']==sha256_file(SCHEDULE)
    assert run['source']['e0_sha256']==sha256_file(E0/'run.json') and run['source']['previous_e1_sha256']==sha256_file(PREVIOUS/'run.json')
    calibration=directory/'calibration';record=read(calibration/'run.json')
    for path,sha in record['artifacts'].items():assert sha256_file(calibration/path)==sha
    model=read(calibration/'remaining_life_v1.json');training=pq.read_table(calibration/'training_pairs.parquet').to_pylist()
    assert set(model['training_root_ids']).isdisjoint(model['excluded_root_ids']) and len(model['training_root_ids'])==293
    assert len(training)==model['interval_count']==62159 and {r['root_id'] for r in training}<=set(model['training_root_ids'])
    assert sha256_file(calibration/'remaining_life_v1.json')==run['source']['table_sha256']
    assert calibration_values(model)==calibration_values(read(DEFAULT_TABLE))
    import bisect
    for tier in ('online','noisy'):
        table=RemainingLifeTable(model,tier);data=model['tiers'][tier]
        for i,bucket in enumerate(data['buckets']):
            values=np.array([r['actual_gap_s'] for r in training if bisect.bisect_right(table.cuts,r[tier+'_s'])==i])
            assert len(values)==bucket['count']
            for j,age in enumerate(table.grid):
                live=values[values>age];assert len(live)==bucket['survivors'][j]
                if len(live):assert math.isclose(float(np.mean(live-age)),table.values[i][j],rel_tol=1e-12,abs_tol=1e-12)
                else:assert table.values[i][j] is None
    e0=read(E0/'run.json');total=cases=0;failures=[];skips=0
    for population in run['results']:
        folder=directory/population['name'];original=next(r for r in e0['runs'] if r['name']==population['name'])
        data=read(folder/'population.json');instances=[SessionInstance(**r) for r in data['instances']]
        assert population_hash(instances)==data['population_sha256']==population['population_sha256']==original['population_sha256']
        assert set(population['cases'])=={c[0] for c in CASES}
        reference=pq.read_table(folder/'lru/requests.parquet').to_pylist()
        for label,tier,options in CASES:
            case=population['cases'][label];path=folder/label/'requests.parquet'
            assert read(folder/label/'run.json')==case and case['git_commit']==run['git_commit']
            assert case['options']==options and case['requests_sha256']==sha256_file(path)
            rows=pq.read_table(path).to_pylist();summary=case['summary']
            assert len(rows)==len(reference)==summary['requests'] and _schedule_fingerprint(rows)==case['schedule_gpu_sha256']
            for a,b in zip(rows,reference):
                assert {k:v for k,v in a.items() if k not in ('cpu_hit','recomputed')}=={k:v for k,v in b.items() if k not in ('cpu_hit','recomputed')}
                assert a['gpu_hit']+a['cpu_hit']+a['recomputed']==a['prompt_tokens']
                assert a['in_window']==(600<=a['send_s']<2400 and not a['is_entry'])
            window=[r for r in rows if r['in_window']];tokens=sum(r['prompt_tokens'] for r in window)
            assert len(window)==summary['window_requests'] and tokens==summary['window_prompt_tokens']
            for key in ('gpu_hit','cpu_hit','recomputed'):
                count=sum(r[key] for r in window);assert count==summary['source_tokens'][key]
                assert math.isclose(count/tokens,summary['source_fractions'][key],rel_tol=1e-12,abs_tol=1e-12)
            if label in ('lru','oracle'):
                expected=original['models'][label];assert summary=={k:v for k,v in expected['summary'].items() if k!='gap_eviction_audit'}
                old=pq.read_table(E0/population['name']/'models'/label/'requests.parquet').to_pylist()
                assert rows==[{k:r[k] for k in rows[0]} for r in old]
            previous_label='hint-'+label[7:] if label.startswith('legacy-') else 'no-tail-'+label.split('-')[0] if label in ('perfect','noisy-deadline','online-deadline') else None
            if previous_label:
                assert rows==pq.read_table(PREVIOUS/population['name']/previous_label/'requests.parquet').to_pylist()
                assert summary==read(PREVIOUS/population['name']/previous_label/'run.json')['summary']
            if tier:
                assert case['policy_stats']['invalid_hint_callbacks']==0 and case['policy_stats']['valid_hint_requests']==len(rows)
                assert case['policy_stats']['evicted']==summary['cpu_evictions']
                if options['scoring']=='remaining':assert case['policy_stats']['timeout_events']==0 and case['table_sha256']==run['source']['table_sha256']
            if 'deadline_decisions_sha256' in case:
                p=folder/label/'deadline_decisions.parquet';assert sha256_file(p)==case['deadline_decisions_sha256']
                for r in pq.read_table(p).to_pylist():
                    assert 0<=r['overdue_blocks']<=r['cpu_blocks'] and r['needed']>0
                    assert math.isclose(r['overdue_blocks']/r['cpu_blocks'],r['overdue_fraction'],abs_tol=1e-12)
            skips=max(skips,summary['cpu_store_skipped_requests']);total+=len(rows);cases+=1
        if population['cases']['online-remaining']['summary']['source_tokens']['recomputed']>population['cases']['lru']['summary']['source_tokens']['recomputed']:
            failures.append(population['name'])
    expected=paired_rows(run)
    with (directory/'paired.csv').open() as f:paired=list(csv.DictReader(f))
    assert len(paired)==len(expected)==72
    for a,b in zip(paired,expected):
        assert a['name']==b['name'] and a['strategy']==b['strategy']
        for key in ('recomputed','reduction_pp','oracle_capture'):
            assert (a[key]=='' if b[key] is None else math.isclose(float(a[key]),b[key],rel_tol=1e-12,abs_tol=1e-12))
    with (directory/'aggregate.csv').open() as f:aggregate=list(csv.DictReader(f))
    for a,b in zip(aggregate,aggregate_pairs(expected)):
        assert a['strategy']==b['strategy'] and float(a['rate'])==b['rate']
        for key in ('mean_recomputed','mean_pp','minimum_pp','maximum_pp','capture_mean','capture_min','capture_max'):
            assert (a[key]=='' if b[key] is None else math.isclose(float(a[key]),b[key],rel_tol=1e-12,abs_tol=1e-12))
    assert cases==156
    result=dict(git_commit=commit,measurement_git_commit=run['git_commit'],integrity_passed=True,cases=cases,populations=12,requests_checked=total,
        legacy_regression_cases=72,online_acceptance=dict(passed=not failures,nonnegative_populations=12-len(failures),failed_populations=failures),
        max_cpu_store_skips=skips,table_sha256=run['source']['table_sha256'])
    write_json(directory/'validation.json',result)
    write_json(directory/'artifacts.json',dict(git_commit=commit,files={str(p.relative_to(directory)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return result


def run(output,calibration,workers=4):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git worktree')
    assert calibration_values(read(calibration/'remaining_life_v1.json'))==calibration_values(read(DEFAULT_TABLE))
    e0=read(E0/'run.json');assert e0['status']=='complete' and len(e0['runs'])==12
    output.mkdir(parents=True);shutil.copytree(calibration,output/'calibration')
    report=dict(git_commit=commit,preregistration_commit='329824d',status='running',
        source=dict(e0_sha256=sha256_file(E0/'run.json'),previous_e1_sha256=sha256_file(PREVIOUS/'run.json'),
                    workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE),table_sha256=sha256_file(calibration/'remaining_life_v1.json')),
        parameters=dict(gpu_tokens=182880,cpu_tokens=274320,slots=8,block_size=16,duration_s=2400,warmup_s=600,service_s=.5,rates=RATES,seeds=SEEDS,workers=workers,strategies=[c[0] for c in CASES]),results=[])
    write_json(output/'run.json',report)
    try:
        with multiprocessing.get_context('spawn').Pool(processes=workers,initializer=_initialize,initargs=(output/'calibration/remaining_life_v1.json',)) as pool:
            for result in pool.imap_unordered(_population,[(item,output,commit) for item in e0['runs']]):
                report['results'].append(result);report['results'].sort(key=lambda r:(r['rate'],r['seed']))
                print('COMPLETE',len(report['results']),'/12',result['name'],flush=True);write_json(output/'run.json',report)
        report['status']='complete';write_json(output/'run.json',report)
        render(output,report);validate(output)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run.json',report);raise
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path);p.add_argument('--calibration-dir',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4);p.add_argument('--validate',type=Path)
    a=p.parse_args()
    if a.validate:validate(a.validate)
    elif a.output_dir and a.calibration_dir:run(a.output_dir,a.calibration_dir,a.workers)
    else:p.error('--output-dir and --calibration-dir required, or --validate')


if __name__=='__main__':main()
