"""CPU-only preregistered E1 rehearsal, paired with the 12 E0 populations."""
import argparse
import csv
import gc
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import load_inputs,population_hash,write_json,WORKLOAD,SCHEDULE,RATES,SEEDS
from agentkv.bench.workload import ROOT
from agentkv.bench.steady_state import SessionInstance,sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.sim.two_tier import simulate
from agentkv.sim.hint_adapter import HintPolicyAdapter

TIERS=('perfect','noisy','online')
COMPONENTS={'no-tail':dict(tail_first=False),'no-final':dict(final_first=False),'no-timeout':dict(timeout_decay=False)}
CASES=[('lru',None,{}),('oracle',None,{})]+[(f'hint-{tier}',tier,{}) for tier in TIERS]+[
    (f'{component}-{tier}',tier,options) for component,options in COMPONENTS.items() for tier in TIERS]+[('none-control','none',{})]
_ROOTS=_BLOCKS=None


def _initialize():
    global _ROOTS,_BLOCKS
    _ROOTS,_BLOCKS=load_inputs()


def _schedule_fingerprint(rows):
    fields=('instance_id','trace_req_idx','due_s','send_s','end_s','gpu_hit','is_entry','in_window')
    return hashlib.sha256(json.dumps([[r[k] for k in fields] for r in rows],separators=(',',':')).encode()).hexdigest()


def _population(item,output,e0_dir,commit):
    name=item['name'];folder=output/name;folder.mkdir()
    original=item['models']['lru']['instances']
    instances=[SessionInstance(**r) for r in original]
    rebuilt=sample_population({rid:r.duration_s for rid,r in _ROOTS.items()},item['rate'],item['seed'],2400,True)
    assert population_hash(instances)==population_hash(rebuilt)==item['population_sha256']
    result=dict(name=name,rate=item['rate'],seed=item['seed'],population_sha256=population_hash(instances),cases={})
    reference_rows=None;fingerprint=None
    for label,tier,options in CASES:
        started=time.perf_counter()
        policy=HintPolicyAdapter(274320//16,hint_tier=tier,**(dict(scoring='deadline',tail_first=True,timeout_decay=True)|options)) if tier else label
        summary,rows=simulate(_ROOTS,instances,_BLOCKS,182880,274320,8,cpu_policy=policy)
        elapsed=time.perf_counter()-started
        schedule_fingerprint=_schedule_fingerprint(rows)
        if fingerprint is None:fingerprint=schedule_fingerprint;reference_rows=rows
        assert schedule_fingerprint==fingerprint,('Scheduling/GPU regression',name,label)
        if label in ('lru','oracle'):
            expected=item['models'][label]
            assert summary=={k:v for k,v in expected['summary'].items() if k!='gap_eviction_audit'},(name,label,'E0 summary regression')
            baseline=e0_dir/name/'models'/label/'requests.parquet'
            assert sha256_file(baseline)==expected['requests_sha256']
            old=pq.read_table(baseline).to_pylist()
            assert rows==[{k:r[k] for k in rows[0]} for r in old],(name,label,'E0 request regression')
        if label=='none-control':
            assert rows==reference_rows and summary==result['cases']['lru']['summary'],('None != LRU',name)
        if tier and tier!='none':assert policy.policy.stats['valid_hint_requests']==summary['requests']
        destination=folder/label;destination.mkdir()
        pq.write_table(pa.Table.from_pylist(rows),destination/'requests.parquet',compression='zstd')
        record=dict(git_commit=commit,status='complete',label=label,hint_tier=tier,
            options=dict(tail_first=True,final_first=True,timeout_decay=True)|options,
            population_sha256=result['population_sha256'],schedule_gpu_sha256=schedule_fingerprint,
            summary=summary,elapsed_wall_s=elapsed,
            policy_stats=dict(policy.policy.stats) if tier else None,
            requests_sha256=sha256_file(destination/'requests.parquet'))
        write_json(destination/'run.json',record);result['cases'][label]=record
        print(name,label,f'{summary["source_fractions"]["recomputed"]:.6%}',f'{elapsed:.2f}s',flush=True)
        del policy
        gc.collect()
    write_json(folder/'population.json',dict(instances=original,population_sha256=result['population_sha256']))
    write_json(folder/'run.json',dict(git_commit=commit,status='complete',**result))
    return result


def _task(arguments):
    return _population(*arguments)


def paired(result,tier):
    fraction=lambda name:result['cases'][name]['summary']['source_fractions']['recomputed']
    lru,oracle,hint=fraction('lru'),fraction('oracle'),fraction('hint-'+tier)
    return dict(name=result['name'],rate=result['rate'],seed=result['seed'],tier=tier,
                lru_recomputed=lru,hint_recomputed=hint,oracle_recomputed=oracle,
                reduction_pp=100*(lru-hint),oracle_capture=(lru-hint)/(lru-oracle) if lru!=oracle else None)


def summarize(report):
    pairs=[paired(r,tier) for r in report['results'] for tier in TIERS]
    aggregates=[];ablations=[]
    for rate in RATES:
        for tier in TIERS:
            group=[p for p in pairs if p['rate']==rate and p['tier']==tier]
            if not group:continue
            delta=[p['reduction_pp'] for p in group];capture=[p['oracle_capture'] for p in group if p['oracle_capture'] is not None]
            aggregates.append(dict(rate=rate,tier=tier,populations=len(group),mean_pp=float(np.mean(delta)),minimum_pp=min(delta),maximum_pp=max(delta),
                capture_mean=float(np.mean(capture)) if capture else None,capture_min=min(capture) if capture else None,capture_max=max(capture) if capture else None))
            for component in COMPONENTS:
                values=[]
                for r in report['results']:
                    if r['rate']!=rate:continue
                    full=r['cases']['hint-'+tier]['summary']['source_fractions']['recomputed']
                    changed=r['cases'][component+'-'+tier]['summary']['source_fractions']['recomputed']
                    values.append(100*(changed-full))
                ablations.append(dict(rate=rate,tier=tier,component=component,mean_pp=float(np.mean(values)),minimum_pp=min(values),maximum_pp=max(values)))
    return pairs,aggregates,ablations


def render(directory,report):
    pairs,aggregates,ablations=summarize(report)
    for name,rows in [('paired',pairs),('aggregate',aggregates),('ablations',ablations)]:
        if not rows:continue
        with (directory/f'{name}.csv').open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    main=('lru','hint-perfect','hint-noisy','hint-online','oracle')
    lines=['# E1：config-11 的 CPU-only 模拟预演','',f"状态：{report['status']}；代码 `{report['git_commit']}`；预注册 `{report['preregistration_commit']}`。",'',
       'GPU182880/CPU274320 token、16K、8槽位，40分钟/预热10分钟，固定服务0.5秒、零搬运耗时、完成时CPU重复写prompt。未调用GPU或vLLM。',
       '使用E0全部12个人口，含其不稳定GPU实跑人口；本次不应用GPU稳定性筛选。所有比例以原始prompt token为分母、窗口内排除is_entry；不是E0整块重算分解的分母。每次token加权，再对三个种子等权平均。',
       '', '## 每种策略的重算比例','', '| λ | LRU | perfect | noisy | online | CPU Oracle |','|---|---:|---:|---:|---:|---:|']
    for rate in RATES:
        group=[r for r in report['results'] if r['rate']==rate]
        if group:lines.append(f'| {rate:g} | '+' | '.join(f"{np.mean([r['cases'][label]['summary']['source_fractions']['recomputed'] for r in group]):.3%}" for label in main)+' |')
    lines+=['','## 配对收益与Oracle空间捕获','', '百分点为LRU−hint；捕获率逐人口计算(LRU−hint)/(LRU−Oracle)，再等权平均，不截断负数或超过100%的值。括号为三个种子的最小–最大。','',
            '| λ | hint | 减少百分点均值（范围） | Oracle捕获率均值（范围） |','|---|---|---:|---:|']
    for a in aggregates:
        capture='NA' if a['capture_mean'] is None else f"{a['capture_mean']:.2%} ({a['capture_min']:.2%}–{a['capture_max']:.2%})"
        lines.append(f"| {a['rate']:g} | {a['tier']} | {a['mean_pp']:.3f} ({a['minimum_pp']:.3f}–{a['maximum_pp']:.3f}) | {capture} |")
    lines+=['','## 单组件消融','', '差值=消融重算率−完整hint重算率，单位百分点；正数表示去掉该组件变差。每项仅关闭一个开关，全部三个hint档位均覆盖。','',
            '| λ | hint | 移除组件 | 差值均值（范围） |','|---|---|---|---:|']
    for a in ablations:lines.append(f"| {a['rate']:g} | {a['tier']} | {a['component']} | {a['mean_pp']:+.3f} ({a['minimum_pp']:+.3f}–{a['maximum_pp']:+.3f}) |")
    lines+=['','## 逐人口主比较','', '| λ | seed | LRU | perfect | noisy | online | Oracle |','|---|---|---:|---:|---:|---:|---:|']
    for r in report['results']:
        lines.append(f"| {r['rate']:g} | {r['seed']} | "+' | '.join(f"{r['cases'][label]['summary']['source_fractions']['recomputed']:.3%}" for label in main)+' |')
    lines+=['','## 正确性与边界','',
      '每个人口的LRU、Oracle汇总及逐请求记录与E0模型逐项回归；none适配器与本次LRU逐请求、逐字段相同。180个case均保留请求、人口、输入及调度/GPU指纹。时钟来自模拟事件；策略只读当前请求hint，不查看未来prompt或后续hint。Oracle读取完整剩余人口预计时间（含T之后），不保证链式两层模型全局最优。',
      '原LRU的CPU秩只在完成写入时提升；新策略在开始时仅登记元数据，在完成时交付hint。所有策略必须保持相同请求和GPU路径，CPU写入跳过及hint/超时计数见各run.json。',
      'perfect/noisy的gap_p90为expected的2倍，与现有在线信封一致；可能没有超时事件，消融为0也原样报告。尾部资格限制与已有预测/前缀LRU同分排序可能重叠，消融无收益不能解释为实现失效。固定服务及零传输模型不含真实排队/重算服务时间反馈，此预演没有TTFT测量，不能替代真实E1。',
      '', '![重算](recomputed.png)','','![配对差值](paired.png)',
      '', '[配对原始值](paired.csv) · [均值与范围](aggregate.csv) · [消融](ablations.csv) · [run.json](run.json)',
      '', '```bash','python -m agentkv.bench.e1_simulation --output-dir /tmp/new-e1-simulation --workers 4','```']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    if report['results']:plots(directory,report,aggregates)


def plots(directory,report,aggregates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.1,3.2))
    for label in ('lru','hint-perfect','hint-noisy','hint-online','oracle'):
        x=[];y=[]
        for rate in RATES:
            group=[r for r in report['results'] if r['rate']==rate]
            if group:x.append(rate);y.append(100*np.mean([r['cases'][label]['summary']['source_fractions']['recomputed'] for r in group]))
        ax.plot(x,y,'o-',label=label)
    ax.set(xlabel='Root arrivals / minute',ylabel='Recomputed prompt tokens (%)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,ncol=2)
    fig.tight_layout();save_plot(fig,directory,'recomputed');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    for tier in TIERS:
        data=[r for r in aggregates if r['tier']==tier]
        if not data:continue
        y=np.array([r['mean_pp'] for r in data]);lower=np.array([r['minimum_pp'] for r in data]);upper=np.array([r['maximum_pp'] for r in data])
        axes[0].errorbar([r['rate'] for r in data],y,yerr=[y-lower,upper-y],fmt='o-',capsize=3,label=tier)
        valid=[r for r in data if r['capture_mean'] is not None]
        axes[1].plot([r['rate'] for r in valid],[100*r['capture_mean'] for r in valid],'o-',label=tier)
    axes[0].set(ylabel='LRU - hint (percentage points)');axes[1].set(ylabel='Oracle space captured (%)')
    for ax in axes:ax.set_xlabel('Root arrivals / minute');ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();save_plot(fig,directory,'paired');plt.close(fig)


def save_plot(fig,directory,name):
    fig.savefig(directory/f'{name}.png',dpi=300)
    fig.savefig(directory/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})


def run(output,workers=4,e0_dir=ROOT/'results/m1-e0-config11-20261103'):
    commit=git_commit()
    if output.exists():raise FileExistsError('New output directory required')
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git worktree')
    e0=json.loads((e0_dir/'run.json').read_text());assert e0['status']=='complete' and len(e0['runs'])==12
    assert e0['source']==dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE))
    output.mkdir(parents=True)
    report=dict(git_commit=commit,preregistration_commit='c00b430',status='running',
        source=dict(e0_run_sha256=sha256_file(e0_dir/'run.json'),**e0['source']),
        parameters=dict(gpu_tokens=182880,cpu_tokens=274320,slots=8,block_size=16,duration_s=2400,warmup_s=600,service_s=.5,
                        rates=RATES,seeds=SEEDS,workers=workers,strategies=[c[0] for c in CASES]),results=[])
    write_json(output/'run.json',report)
    try:
        with multiprocessing.get_context('spawn').Pool(processes=workers,initializer=_initialize) as pool:
            tasks=[(item,output,e0_dir,commit) for item in e0['runs']]
            for result in pool.imap_unordered(_task,tasks):
                report['results'].append(result);report['results'].sort(key=lambda r:(r['rate'],r['seed']))
                print('COMPLETE',len(report['results']),'/12',result['name'],flush=True)
                write_json(output/'run.json',report)
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        write_json(output/'run.json',report);render(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4);p.add_argument('--e0-dir',type=Path,default=ROOT/'results/m1-e0-config11-20261103')
    args=p.parse_args();run(args.output_dir,args.workers,args.e0_dir)


if __name__=='__main__':main()
