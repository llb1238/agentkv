"""Preregistered online scoring selection, freeze gate, and held-out E1."""
import argparse
import csv
import gc
import json
import math
import multiprocessing
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import ROOT,WORKLOAD,SCHEDULE,RATES,load_inputs,population_hash,write_json
from agentkv.bench.e1_simulation import _schedule_fingerprint,save_plot
from agentkv.bench.e1_remaining import read,fraction
from agentkv.bench.steady_state import SessionInstance,sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.remaining_life import DEFAULT_TABLE,RemainingLifeTable
from agentkv.policy.horizon import DEFAULT_HORIZON_TABLE,HorizonProbabilityTable
from agentkv.sim.hint_adapter import HintPolicyAdapter
from agentkv.sim.two_tier import simulate

PREVIOUS=ROOT/'results/m5-e1-remaining-config11-20261104'
FREEZE=ROOT/'docs/m5_scoring_frozen.json'
CANDIDATES=('deadline','remaining','horizon')
_ROOTS=_BLOCKS=None


def mean(values):return math.fsum(values)/len(values)
def source_hashes():return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'agentkv').rglob('*.py'))}


def cases(stage,winner=None):
    common=[('lru',None,None),('lru-final','final','lru_final'),('oracle',None,None)]
    if stage=='development':return common+[(c+'-online','online',c) for c in CANDIDATES]
    if stage!='evaluation' or winner not in CANDIDATES:raise ValueError('Frozen winner required')
    return common+[(winner+'-'+tier,tier,winner) for tier in ('perfect','noisy','online')]


def options(tier,candidate):
    scoring='deadline' if tier=='perfect' and candidate=='remaining' else candidate
    result=dict(scoring=scoring,tail_first=False,final_first=True,timeout_decay=scoring=='deadline')
    if scoring=='horizon':result.update(horizon_initial_s=1.,horizon_window=4096,horizon_period_s=1.,perfect_probability=tier=='perfect')
    return result


def initialize():
    global _ROOTS,_BLOCKS
    _ROOTS,_BLOCKS=load_inputs()


def _population(arguments):
    rate,seed,stage,winner,output,commit=arguments
    name=f'lambda-{rate:g}-seed-{seed}';folder=output/name;folder.mkdir()
    instances=sample_population({rid:r.duration_s for rid,r in _ROOTS.items()},rate,seed,2400,True)
    population_sha=population_hash(instances);result=dict(name=name,rate=rate,seed=seed,population_sha256=population_sha,cases={})
    previous=read(PREVIOUS/name/'run.json') if stage=='development' else None
    if previous:assert previous['population_sha256']==population_sha
    reference=None;fingerprint=None
    for label,tier,candidate in cases(stage,winner):
        settings=options(tier,candidate) if tier else {};start=time.perf_counter()
        policy=HintPolicyAdapter(274320//16,hint_tier=tier,**settings) if tier else label
        summary,rows=simulate(_ROOTS,instances,_BLOCKS,182880,274320,8,cpu_policy=policy)
        elapsed=time.perf_counter()-start;current=_schedule_fingerprint(rows)
        if reference is None:reference=rows;fingerprint=current
        assert current==fingerprint,('Schedule/GPU mismatch',stage,name,label)
        old_label={'lru':'lru','oracle':'oracle','lru-final':'lru-final','deadline-online':'online-deadline','remaining-online':'online-remaining'}.get(label)
        if previous and old_label:
            old=previous['cases'][old_label];path=PREVIOUS/name/old_label/'requests.parquet'
            assert sha256_file(path)==old['requests_sha256'] and summary==old['summary']
            assert rows==pq.read_table(path).to_pylist(),('Prior development regression',name,label)
        path=folder/label;path.mkdir()
        pq.write_table(pa.Table.from_pylist(rows),path/'requests.parquet',compression='zstd')
        record=dict(git_commit=commit,status='complete',label=label,tier=tier,candidate=candidate,options=settings,
            population_sha256=population_sha,schedule_gpu_sha256=current,summary=summary,elapsed_wall_s=elapsed,
            requests_sha256=sha256_file(path/'requests.parquet'),policy_stats=dict(policy.policy.stats) if tier else None)
        if candidate=='horizon':
            trace=policy.policy.horizon_trace
            pq.write_table(pa.Table.from_pylist(trace,schema=pa.schema([('time_s',pa.float64()),('horizon_s',pa.float64()),('needed',pa.int64()),('evicted',pa.int64()),('samples_seen',pa.int64()),('window_samples',pa.int64()),('window_mean_s',pa.float64())])),path/'horizon.parquet',compression='zstd')
            h=[r['horizon_s'] for r in trace]
            record['horizon']=dict(decisions=len(trace),median_s=float(np.median(h)) if h else None,p90_s=float(np.quantile(h,.9)) if h else None,
                final_s=policy.policy.horizon_s,window_samples=len(policy.policy.residences),sha256=sha256_file(path/'horizon.parquet'))
        if tier:assert policy.policy.stats['valid_hint_requests']==len(rows)
        write_json(path/'run.json',record);result['cases'][label]=record
        print(stage,name,label,f'{fraction(record):.6%}',f'{elapsed:.2f}s',flush=True)
        del policy;gc.collect()
    from dataclasses import asdict
    write_json(folder/'population.json',dict(population_sha256=population_sha,instances=[asdict(i) for i in instances]))
    write_json(folder/'run.json',dict(git_commit=commit,status='complete',**result));return result


def paired(report):
    result=[]
    for population in report['results']:
        lru=fraction(population['cases']['lru']);oracle=fraction(population['cases']['oracle'])
        for label,case in population['cases'].items():
            if label in ('lru','oracle'):continue
            value=fraction(case)
            result.append(dict(name=population['name'],rate=population['rate'],seed=population['seed'],strategy=label,recomputed=value,
                reduction_pp=100*(lru-value),oracle_capture=(lru-value)/(lru-oracle) if lru!=oracle else None))
    return result


def select(report):
    if report['stage']!='development':raise ValueError('Selection uses development only')
    assert len(report['results'])==12
    assert {(p['rate'],p['seed']) for p in report['results']}=={(r,s) for r in RATES for s in (0,1,2)}
    pairs=paired(report);scores=[]
    for name in CANDIDATES:
        failures=[p['name'] for p in report['results'] if p['cases'][name+'-online']['summary']['source_tokens']['recomputed']>p['cases']['lru']['summary']['source_tokens']['recomputed']]
        captures=[p['oracle_capture'] for p in pairs if p['strategy']==name+'-online' and p['oracle_capture'] is not None]
        scores.append(dict(candidate=name,eligible=not failures,failed_populations=failures,
                           mean_oracle_capture=mean(captures) if captures else None,defined=len(captures),undefined=12-len(captures)))
    eligible=[s for s in scores if s['eligible'] and s['mean_oracle_capture'] is not None]
    winner=max(eligible,key=lambda s:(s['mean_oracle_capture'],-CANDIDATES.index(s['candidate'])))['candidate'] if eligible else None
    return dict(winner=winner,candidates=scores,rule='integer noninferiority at all 12 populations, then mean paired Oracle capture; fixed order breaks exact ties')


def aggregates(pairs):
    result=[]
    for rate in RATES:
        for label in dict.fromkeys(p['strategy'] for p in pairs):
            group=[p for p in pairs if p['rate']==rate and p['strategy']==label]
            if not group:continue
            delta=[r['reduction_pp'] for r in group];caps=[r['oracle_capture'] for r in group if r['oracle_capture'] is not None]
            result.append(dict(rate=rate,strategy=label,mean_recomputed=mean([p['recomputed'] for p in group]),mean_pp=mean(delta),minimum_pp=min(delta),maximum_pp=max(delta),
                capture_mean=mean(caps) if caps else None,capture_min=min(caps) if caps else None,capture_max=max(caps) if caps else None,undefined=len(group)-len(caps)))
    return result


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def render(directory,report):
    pairs=paired(report);summary=aggregates(pairs);csv_write(directory/'paired.csv',pairs);csv_write(directory/'aggregate.csv',summary)
    stage=report['stage'];labels=[c[0] for c in cases(stage,report.get('winner'))]
    title='开发组：仅用于选择' if stage=='development' else '最终留出评估：仅种子3/4/5'
    lines=['# M5 打分选型：'+title,'',f"代码 `{report['git_commit']}`；预注册 `a41033c`；状态 {report['status']}。",'',
        'config-11、四个λ、T=40分钟/预热10分钟、固定服务0.5秒、零搬运耗时、尾部关闭；原始prompt token加权，排除is_entry，再按三个种子等权。只用CPU，不代表真实vLLM TTFT或策略开销。',
        '开发数据与最终数据分开；本目录所有表图只使用当前组人口。新种子是人口留出，同一100根trace仍被复用，不宣称跨trace泛化。',
        '', '## 重算比例','', '| λ | '+' | '.join(labels)+' |','|---|'+'---:|'*len(labels)]
    for rate in RATES:
        group=[p for p in report['results'] if p['rate']==rate]
        lines.append(f'| {rate:g} | '+' | '.join(f"{mean([fraction(p['cases'][label]) for p in group]):.3%}" for label in labels)+' |')
    if stage=='development':
        choice=select(report);write_json(directory/'selection.json',choice)
        lines+=['','## 预注册选择','',f"选中：`{choice['winner']}`。先逐人口要求online不差于LRU，再最大化12个人口等权的Oracle捕获率；以下不能作为最终性能数字。",'',
            '| 候选 | 12人口门槛 | 平均捕获率 | 失败人口 |','|---|---|---:|---|']
        for c in choice['candidates']:
            cap='NA' if c['mean_oracle_capture'] is None else f"{c['mean_oracle_capture']:.3%}"
            lines.append(f"| {c['candidate']} | {c['eligible']} | {cap} | {', '.join(c['failed_populations']) or '无'} |")
    else:
        lines+=['',f"冻结候选 `{report['winner']}`，冻结提交 `{report['freeze_commit']}`；仅根据开发组选择。留出结果不反向改变策略，负收益不删除。"]
    lines+=['','## 配对减少百分点与Oracle空间捕获','', '相对同人口LRU；捕获率先逐人口计算再平均，不截断。括号为三种子的最小–最大，分母0记NA。','',
        '| λ | 策略 | 减少百分点均值（范围） | 捕获率均值（范围） | NA |','|---|---|---:|---:|---:|']
    for r in summary:
        cap='NA' if r['capture_mean'] is None else f"{r['capture_mean']:.2%} ({r['capture_min']:.2%}–{r['capture_max']:.2%})"
        lines.append(f"| {r['rate']:g} | {r['strategy']} | {r['mean_pp']:+.3f} ({r['minimum_pp']:+.3f}–{r['maximum_pp']:+.3f}) | {cap} | {r['undefined']} |")
    lines+=['','## 逐人口记录','', '| λ | seed | 策略 | 重算率 | 减少百分点 | 捕获率 |','|---|---|---|---:|---:|---:|']
    for p in pairs:
        cap='NA' if p['oracle_capture'] is None else f"{p['oracle_capture']:.3%}"
        lines.append(f"| {p['rate']:g} | {p['seed']} | {p['strategy']} | {p['recomputed']:.3%} | {p['reduction_pp']:+.3f} | {cap} |")
    lines+=['','## 运行与H统计','', 'horizon使用最近4096个成功驱逐块的last_access至evict时长均值，每秒刷新；初值1秒。活跃拥有者阻止块被驱逐。概率CDF仅由未抽中的293根校准，online/noisy来源分开，perfect直接使用0/1。下表H分位数按全程驱逐调用采样；原始轨迹在case/horizon.parquet。','',
        '| 人口 | 策略 | CPU写入跳过 | H P50/P90(s) | 刷新数 |','|---|---|---:|---:|---:|']
    for p in report['results']:
        for label,c in p['cases'].items():
            h=c.get('horizon');text='—' if not h or h['median_s'] is None else f"{h['median_s']:.3f}/{h['p90_s']:.3f}"
            lines.append(f"| {p['name']} | {label} | {c['summary']['cpu_store_skipped_requests']} | {text} | {(c['policy_stats'] or {}).get('horizon_updates',0)} |")
    lines+=['','所有请求/GPU/调度路径一致，CPU策略只改变CPU命中与重算；Oracle仍是块级未来访问启发式，不保证全局最优。',
        '', '[run.json](run.json) · [配对数据](paired.csv) · [汇总](aggregate.csv) · [验证](validation.json) · [指纹](artifacts.json) · [预注册](../../docs/m5_scoring_selection_design.md)',
        '', '![重算](recomputed.png)','','![捕获率](capture.png)',
        '', '复现需先在干净工作区开发、冻结提交后才允许最终评估，不能倒序：','', '```bash',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m agentkv.bench.e1_selection develop --output-dir /tmp/new-development',
        'python -m agentkv.bench.e1_selection freeze --development-dir /tmp/new-development --freeze-file docs/new-freeze.json',
        '# 单独commit并push冻结文件，然后：',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m agentkv.bench.e1_selection evaluate --output-dir /tmp/new-evaluation --development-dir /tmp/new-development --freeze-file docs/new-freeze.json',
        'python -m agentkv.bench.e1_selection validate --run-dir /tmp/new-evaluation','```']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('recomputed','capture'):
        fig,ax=plt.subplots(figsize=(7.1,3.2))
        plot_labels=labels if metric=='recomputed' else [label for label in labels if label not in ('lru','oracle')]
        for label in plot_labels:
            values=[]
            for rate in RATES:
                if metric=='recomputed':values.append(100*mean([fraction(p['cases'][label]) for p in report['results'] if p['rate']==rate]))
                else:
                    row=next(v for v in summary if v['rate']==rate and v['strategy']==label)
                    values.append(100*row['capture_mean'] if row['capture_mean'] is not None else np.nan)
            ax.plot(RATES,values,'o-',label=label,ms=3,lw=1)
        ax.set(xlabel='Root arrivals / minute',ylabel='Recomputed prompt tokens (%)' if metric=='recomputed' else 'Oracle space captured (%)')
        ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=7);fig.tight_layout();save_plot(fig,directory,metric);plt.close(fig)


def validate(directory):
    commit=git_commit();run=read(directory/'run.json');assert run['status']=='complete'
    stage=run['stage'];seeds=(0,1,2) if stage=='development' else (3,4,5)
    assert {(p['rate'],p['seed']) for p in run['results']}=={(r,s) for r in RATES for s in seeds}
    assert run['source']['workload_sha256']==sha256_file(WORKLOAD) and run['source']['schedule_sha256']==sha256_file(SCHEDULE)
    assert run['source']['remaining_sha256']==sha256_file(DEFAULT_TABLE) and run['source']['horizon_sha256']==sha256_file(DEFAULT_HORIZON_TABLE)
    cases_checked=records=0
    for p in run['results']:
        folder=directory/p['name'];population=read(folder/'population.json')
        assert population_hash([SessionInstance(**i) for i in population['instances']])==p['population_sha256']==population['population_sha256']
        assert set(p['cases'])=={c[0] for c in cases(stage,run.get('winner'))}
        reference=pq.read_table(folder/'lru/requests.parquet').to_pylist()
        for label,tier,candidate in cases(stage,run.get('winner')):
            c=p['cases'][label];assert read(folder/label/'run.json')==c and c['git_commit']==run['git_commit']
            path=folder/label/'requests.parquet';assert sha256_file(path)==c['requests_sha256']
            rows=pq.read_table(path).to_pylist();s=c['summary']
            assert len(rows)==len(reference)==s['requests'] and _schedule_fingerprint(rows)==c['schedule_gpu_sha256']
            assert c['options']==(options(tier,candidate) if tier else {})
            for a,b in zip(rows,reference):
                assert {k:v for k,v in a.items() if k not in ('cpu_hit','recomputed')}=={k:v for k,v in b.items() if k not in ('cpu_hit','recomputed')}
                assert all(a[k]>=0 for k in ('gpu_hit','cpu_hit','recomputed')) and a['gpu_hit']+a['cpu_hit']+a['recomputed']==a['prompt_tokens']
                assert a['in_window']==(600<=a['send_s']<2400 and not a['is_entry'])
            window=[r for r in rows if r['in_window']];tokens=sum(r['prompt_tokens'] for r in window)
            assert len(window)==s['window_requests'] and tokens==s['window_prompt_tokens']
            for key in ('gpu_hit','cpu_hit','recomputed'):
                count=sum(r[key] for r in window);assert count==s['source_tokens'][key]
                assert math.isclose(count/tokens,s['source_fractions'][key],rel_tol=1e-12,abs_tol=1e-12)
            if tier:assert c['policy_stats']['invalid_hint_callbacks']==0 and c['policy_stats']['valid_hint_requests']==len(rows)
            if candidate=='horizon':
                assert c['policy_stats']['horizon_samples_seen']==c['policy_stats']['evicted']==s['cpu_evictions']
                path=folder/label/'horizon.parquet';assert sha256_file(path)==c['horizon']['sha256']
                trace=pq.read_table(path).to_pylist();assert all(r['horizon_s']>=0 and 0<=r['window_samples']<=4096 for r in trace)
            if stage=='development' and label!='horizon-online':
                previous_label={'deadline-online':'online-deadline','remaining-online':'online-remaining'}.get(label,label)
                assert rows==pq.read_table(PREVIOUS/p['name']/previous_label/'requests.parquet').to_pylist()
                assert s==read(PREVIOUS/p['name']/previous_label/'run.json')['summary']
            records+=len(rows);cases_checked+=1
    pairs=paired(run)
    for file,expected in [('paired.csv',pairs),('aggregate.csv',aggregates(pairs))]:
        with (directory/file).open() as f:actual=list(csv.DictReader(f))
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            for key,value in b.items():
                if value is None:assert a[key]==''
                elif isinstance(value,(int,float)):assert math.isclose(float(a[key]),value,rel_tol=1e-12,abs_tol=1e-12)
                else:assert a[key]==value
    if stage=='development':assert read(directory/'selection.json')==select(run)
    else:
        frozen=read(directory/'freeze.json');assert sha256_file(directory/'freeze.json')==run['source']['freeze_sha256']
        assert run['winner']==frozen['selection']['winner'] and frozen['evaluation_seeds']==[3,4,5]
        assert run['source']['development_sha256']==frozen['development_run_sha256']
    assert cases_checked==72
    result=dict(git_commit=commit,measurement_git_commit=run['git_commit'],stage=stage,passed=True,populations=12,cases=cases_checked,
        requests_checked=records,run_sha256=sha256_file(directory/'run.json'),all_fingerprints_passed=True,development_regression_cases=60 if stage=='development' else 0)
    write_json(directory/'validation.json',result)
    write_json(directory/'artifacts.json',dict(git_commit=commit,files={str(p.relative_to(directory)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='artifacts.json'}))
    print(json.dumps(result,indent=2),flush=True);return result


def freeze(development,path):
    commit=git_commit();run=read(development/'run.json');verification=read(development/'validation.json')
    assert run['stage']=='development' and run['status']=='complete' and verification['passed']
    assert verification['run_sha256']==sha256_file(development/'run.json')
    for name,record in read(development/'artifacts.json')['files'].items():assert sha256_file(development/name)==record['sha256']
    choice=select(run);assert choice==read(development/'selection.json')
    if choice['winner'] is None:raise ValueError('No eligible candidate; held-out population remains unopened')
    if path.exists():raise FileExistsError('Freeze file already exists; never overwrite selection')
    payload=dict(format='agentkv.scoring_freeze.v1',preregistration_commit='a41033c',freeze_code_commit=commit,
        development_run_sha256=sha256_file(development/'run.json'),development_validation_sha256=sha256_file(development/'validation.json'),
        selection=choice,development_seeds=[0,1,2],evaluation_seeds=[3,4,5],rates=RATES,parameters=run['parameters'],
        input_sha256={k:v for k,v in run['source'].items()},source_hashes=source_hashes(),remaining_sha256=sha256_file(DEFAULT_TABLE),horizon_sha256=sha256_file(DEFAULT_HORIZON_TABLE))
    write_json(path,payload);print(json.dumps(payload,indent=2),flush=True)


def run(stage,output,workers=4,development=None,freeze_file=FREEZE):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git')
    winner=None;frozen=None;freeze_commit=None
    if stage=='evaluation':
        frozen=read(freeze_file)
        assert frozen['format']=='agentkv.scoring_freeze.v1' and frozen['evaluation_seeds']==[3,4,5]
        assert frozen['source_hashes']==source_hashes()
        assert frozen['remaining_sha256']==sha256_file(DEFAULT_TABLE) and frozen['horizon_sha256']==sha256_file(DEFAULT_HORIZON_TABLE)
        assert frozen['input_sha256']['workload_sha256']==sha256_file(WORKLOAD) and frozen['input_sha256']['schedule_sha256']==sha256_file(SCHEDULE)
        assert frozen['development_validation_sha256']==sha256_file(development/'validation.json')
        assert frozen['development_run_sha256']==sha256_file(development/'run.json')
        assert frozen['selection']==select(read(development/'run.json'))
        relative=str(freeze_file.resolve().relative_to(ROOT))
        subprocess.check_output(['git','ls-files','--error-unmatch',relative],cwd=ROOT)
        freeze_commit=subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=ROOT,text=True).strip()
        assert freeze_commit and freeze_commit!=frozen['freeze_code_commit']
        winner=frozen['selection']['winner']
    elif stage!='development':raise ValueError('Unknown stage')
    seeds=(0,1,2) if stage=='development' else (3,4,5)
    output.mkdir(parents=True)
    if frozen:shutil.copyfile(freeze_file,output/'freeze.json')
    report=dict(git_commit=commit,preregistration_commit='a41033c',status='running',stage=stage,winner=winner,freeze_commit=freeze_commit,
        source=dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE),remaining_sha256=sha256_file(DEFAULT_TABLE),
                    horizon_sha256=sha256_file(DEFAULT_HORIZON_TABLE)),
        parameters=dict(rates=RATES,seeds=seeds,gpu_tokens=182880,cpu_tokens=274320,slots=8,block_size=16,duration_s=2400,warmup_s=600,service_s=.5,
                        workers=workers,strategies=[c[0] for c in cases(stage,winner)]),results=[])
    if frozen:report['source'].update(freeze_sha256=sha256_file(freeze_file),development_sha256=sha256_file(development/'run.json'))
    write_json(output/'run.json',report)
    try:
        with multiprocessing.get_context('spawn').Pool(processes=workers,initializer=initialize) as pool:
            tasks=[(rate,seed,stage,winner,output,commit) for rate in RATES for seed in seeds]
            for result in pool.imap_unordered(_population,tasks):
                report['results'].append(result);report['results'].sort(key=lambda p:(p['rate'],p['seed']))
                write_json(output/'run.json',report);print('COMPLETE',stage,len(report['results']),'/12',result['name'],flush=True)
        report['status']='complete';write_json(output/'run.json',report);render(output,report);validate(output)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(output/'run.json',report);raise
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ('develop','evaluate'):
        q=sub.add_parser(name);q.add_argument('--output-dir',type=Path,required=True);q.add_argument('--workers',type=int,choices=range(1,5),default=4)
        if name=='evaluate':q.add_argument('--development-dir',type=Path,required=True);q.add_argument('--freeze-file',type=Path,default=FREEZE)
    q=sub.add_parser('freeze');q.add_argument('--development-dir',type=Path,required=True);q.add_argument('--freeze-file',type=Path,default=FREEZE)
    q=sub.add_parser('validate');q.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.development_dir,a.freeze_file)
    elif a.action=='validate':validate(a.run_dir)
    else:run('development' if a.action=='develop' else 'evaluation',a.output_dir,a.workers,getattr(a,'development_dir',None),getattr(a,'freeze_file',FREEZE))


if __name__=='__main__':main()
