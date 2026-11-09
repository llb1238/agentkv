"""Preregistered config-11 E0: paired populations, audited CPU models and GPU runs.

Run outside the Git worktree. Interrupted/failed runs are retained and stop the
batch; resume never replaces an existing failed or incomplete run.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.workload import ROOT,SCHEMA
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.steady_state import RootSchedule,sample_population
from agentkv.analysis.recomputation import classify_records,CATEGORIES
from agentkv.sim.two_tier import load_block_ids,simulate

RATES=(.5,.7,1.,1.4)
SEEDS=(0,1,2)
WORKLOAD=ROOT/'results/m1-workload-seed0-v3-16k/workload.parquet'
SCHEDULE=ROOT/'results/m1-capacity-seed0-v3/schedule-16384.parquet'


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def population_hash(instances):
    return hashlib.sha256(json.dumps([asdict(i) for i in instances],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_inputs():
    by_root=defaultdict(list);estimates=defaultdict(list)
    rows=pq.read_table(WORKLOAD,columns=[n for n in SCHEMA.names if n!='prompt_token_ids'],use_threads=False).to_pylist()
    for r in rows:by_root[r['root_id']].append(r)
    for r in pq.read_table(SCHEDULE).to_pylist():estimates[r['root_id']].append(r)
    roots={rid:RootSchedule(rs,estimates[rid]) for rid,rs in by_root.items()}
    blocks,_=load_block_ids(WORKLOAD,16)
    return roots,blocks


def models(folder,roots,blocks,instances,commit):
    folder.mkdir(parents=True,exist_ok=True)
    result={}
    for policy in ('lru','oracle'):
        out=folder/policy
        if out.exists():raise FileExistsError(out)
        out.mkdir()
        print('MODEL',folder.name,policy,flush=True)
        summary,records=simulate(roots,instances,blocks,182880,274320,8,
                                duration=2400,warmup=600,cpu_policy=policy,audit_evictions=True)
        categories,classified=classify_records(records,roots,blocks)
        # An exact correspondence between source formulas and causal auditing.
        if categories['source_tokens']['gap_eviction']//16 != summary['gap_eviction_audit']['blocks']:
            raise AssertionError('Classification/audit counts differ')
        pq.write_table(pa.Table.from_pylist(records),out/'requests.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(classified),out/'recomputation.parquet',compression='zstd')
        record=dict(git_commit=commit,status='complete',policy=policy,population_sha256=population_hash(instances),
                    instances=[asdict(i) for i in instances],summary=summary,recomputation=categories,
                    requests_sha256=sha256_file(out/'requests.parquet'),recomputation_sha256=sha256_file(out/'recomputation.parquet'))
        write_json(out/'run.json',record);result[policy]=record
    return result


def gpu_snapshot():
    query=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    text=subprocess.check_output(query,text=True)
    return dict(wall_time_s=time.time(),gpus=text,processes=subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True))


def choose_gpu(folder):
    folder.mkdir(parents=True,exist_ok=True)
    before=gpu_snapshot();write_json(folder/'before.json',before)
    gpu0=next(line.split(',') for line in before['gpus'].splitlines() if line.split(',')[0].strip()=='0')
    # nvidia-smi reports MiB; require at least 60 decimal GB.
    if float(gpu0[4])*1024**2<60e9:
        decision=dict(gpu_index=1,mode='显存共享、算力共享',reason='GPU0 free memory <60 GB')
    else:
        with (folder/'dmon.log').open('w') as sink:
            subprocess.run(['nvidia-smi','dmon','-i','0,1','-d','1','-c','60','-s','pucm','-o','DT'],
                           stdout=sink,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,TZ='UTC'))
        from agentkv.bench.shared_pilot import parse_dmon
        data=[r for r in parse_dmon((folder/'dmon.log').read_text()) if r['gpu']==0]
        after=gpu_snapshot();write_json(folder/'after.json',after)
        free=float(next(line.split(',') for line in after['gpus'].splitlines() if line.split(',')[0].strip()=='0')[4])*1024**2
        idle=len(data)>=50 and max(r['sm'] for r in data)<=1 and free>=60e9
        decision=dict(gpu_index=0 if idle else 1,mode='显存共享、算力独占' if idle else '显存共享、算力共享',
                      reason='60-second aggregate SM and free-memory check',gpu0_sm_samples=len(data),
                      gpu0_sm_max=max((r['sm'] for r in data),default=None))
    write_json(folder/'decision.json',decision);return decision


def analyze_real(folder,roots,blocks,instances):
    from agentkv.bench.analyze_replay import analyze
    import contextlib
    with (folder/'analysis.log').open('w') as sink,contextlib.redirect_stdout(sink):analysis=analyze(folder)
    run=json.loads((folder/'run.json').read_text())
    if population_hash(instances)!=hashlib.sha256(json.dumps(run['instances'],sort_keys=True,separators=(',',':')).encode()).hexdigest():
        raise AssertionError('Real/simulated population mismatch')
    columns=['request_id','root_id','trace_req_idx','instance_id','send_ns','end_ns','prompt_len','is_entry',
             'in_window','status','gpu_hit','cpu_hit','queued_ms','server_queued_ms']
    rows=pq.read_table(folder/'requests.parquet',columns=columns,use_threads=False).to_pylist()
    categories,classified=classify_records(rows,roots,blocks,real=True)
    if categories['requests']!=analysis['success']:raise AssertionError('Missing nominal observation in successful window requests')
    if categories['gap_requests']!=categories['gap_queue_observed_requests']:
        raise AssertionError('Missing queue timing for gap-eviction requests')
    pq.write_table(pa.Table.from_pylist(classified),folder/'recomputation.parquet',compression='zstd')
    record=dict(git_commit=run['git_commit'],population_sha256=population_hash(instances),
                stability=analysis['stability'],recomputation=categories,summary=run['summary'],
                requests_sha256=run['requests_sha256'],recomputation_sha256=sha256_file(folder/'recomputation.parquet'))
    write_json(folder/'e0-analysis.json',record);return record


def run(output,*,models_only=False):
    commit=git_commit()
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage E0 outside Git worktree')
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'run.json'
    if manifest_path.exists():
        report=json.loads(manifest_path.read_text())
        if report['git_commit']!=commit:raise ValueError('Cannot resume with different source commit')
    else:
        report=dict(git_commit=commit,status='running',parameters=dict(rates=RATES,seeds=SEEDS,duration=2400,warmup=600,
            config='config-11',hint_tier='none',policy='lru',service_time_model_s=.5),
            source=dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE)),runs=[])
        write_json(manifest_path,report)
    report['status']='running';report.pop('error',None)
    roots,blocks=load_inputs()
    try:
        # Audit the entire unchanged matrix before allocating GPU resources.
        for rate in RATES:
            for seed in SEEDS:
                name=f'lambda-{rate:g}-seed-{seed}'
                existing=next((r for r in report['runs'] if r['name']==name),None)
                if existing:continue
                instances=sample_population({rid:r.duration_s for rid,r in roots.items()},rate,seed,2400,True)
                comparison=models(output/name/'models',roots,blocks,instances,commit)
                report['runs'].append(dict(name=name,rate=rate,seed=seed,population_sha256=population_hash(instances),
                    M0=sum(i.initial for i in instances),models=comparison,status='models_complete'))
                write_json(manifest_path,report)
        if not models_only:
            from agentkv.bench.shared_pilot import run as pilot
            for item in report['runs']:
                if item['status']=='complete':continue
                real=output/item['name']/'real'
                if real.exists():raise FileExistsError(f'Incomplete/failed run retained; explicit investigation required: {real}')
                decision=choose_gpu(output/item['name']/'gpu-check')
                item.update(status='running',gpu=decision);write_json(manifest_path,report)
                instances=sample_population({rid:r.duration_s for rid,r in roots.items()},item['rate'],item['seed'],2400,True)
                pilot(real,gpu_index=decision['gpu_index'],duration=2400,warmup=600,rate=item['rate'],
                      population_seed=item['seed'],run_id='e0-'+item['name'],analyze_output=False)
                item['real']=analyze_real(real,roots,blocks,instances)
                item['status']='complete';write_json(manifest_path,report)
                render(output,report)
        report['status']='models_complete' if models_only else 'complete'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(manifest_path,report)
        render(output,report)
    return report


def render(output,report):
    """One point per population, then arithmetic mean across retained seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete=[r for r in report['runs'] if r['status']=='complete']
    stable=[r for r in complete if r['real']['stability']['stable']]
    rows=[]
    for item in complete:
        row=dict(rate=item['rate'],seed=item['seed'],status=item['real']['stability']['status'])
        for label,source in [('real',item['real']),('lru',item['models']['lru']),('oracle',item['models']['oracle'])]:
            value=source['recomputation']
            row.update({label+'_'+key:value['source_fractions'][key] for key in CATEGORIES})
            row[label+'_recompute']=value['recomputed_fraction']
            row[label+'_gap_share_avoidable']=value['gap_share_of_avoidable']
            row[label+'_gap_mean_queued_s']=value['gap_requests_mean_queued_s']
        rows.append(row)
    if rows:
        with (output/'runs.csv').open('w',newline='') as sink:
            writer=csv.DictWriter(sink,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(7.1,3.2))
    points=[]
    for rate in RATES:
        kept=[r for r in stable if r['rate']==rate]
        if kept:
            points.append((rate,np.mean([[r['real']['recomputation']['source_fractions'][k] for k in CATEGORIES] for r in kept],axis=0),kept))
    if points:
        x=[p[0] for p in points];values=np.array([p[1] for p in points]);bottom=np.zeros(len(points))
        for i,(key,color) in enumerate(zip(CATEGORIES,['#999999','#D55E00','#E69F00'])):
            ax.bar(x,values[:,i]*100,bottom=bottom,width=.09,label={'unavoidable':'Unavoidable','gap_eviction':'Gap eviction (including queue)','other_eviction':'Other eviction'}[key],color=color);bottom+=values[:,i]*100
        for policy,style,color in [('lru','o--','#0072B2'),('oracle','s:','#009E73')]:
            y=[np.mean([r['models'][policy]['recomputation']['recomputed_fraction'] for r in p[2]])*100 for p in points]
            ax.plot(x,y,style,color=color,label='Simulation '+('CPU Oracle' if policy=='oracle' else 'LRU'))
        ax.legend(frameon=False,fontsize=7,ncol=1,loc='upper right')
    else:ax.text(.5,.5,'No stable completed E0 runs',ha='center',transform=ax.transAxes)
    counts=[sum(r['rate']==rate for r in stable) for rate in RATES]
    ax.set_xticks(RATES,[f'{rate:g}\n(n={n})' for rate,n in zip(RATES,counts)])
    ax.set(xlabel='Root arrivals / minute',ylabel='Recomputed full-block tokens (%)',xlim=(.38,1.55))
    for rate,n in zip(RATES,counts):
        if not n:ax.text(rate,.45*ax.get_ylim()[1],'No stable\nruns',ha='center',color='gray',fontsize=9)
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):
        kwargs=dict(metadata={'CreationDate':None,'ModDate':None}) if ext=='pdf' else {}
        fig.savefig(output/f'recomputation.{ext}',dpi=300,**kwargs)
    plt.close(fig)
    lines=['# E0 config-11', '',f"状态：{report['status']}；代码 `{report['git_commit']}`。12次登记运行，完成{len(complete)}，其中稳定{len(stable)}。",'',
           '16K、8槽位、GPU182880/CPU274320 token；LRU、none；40分钟/预热10分钟；人口种子0/1/2。',
           '三类使用16-token整块，尾块与hit>u异常逐次单列。曲线只纳入新判据下稳定的真实运行及其同人口模型，先算每次token加权比例，再对保留种子等权平均；无稳定种子的λ不画点。', '',
           '| λ | 种子 | 稳定性 | 真实重算 | 模拟LRU | Oracle | 间隙/可避免 | 间隙请求平均排队(s) | hit>u异常 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for item in complete:
        real=item['real'];v=real['recomputation'];gap=v['gap_share_of_avoidable'];wait=v['gap_requests_mean_queued_s']
        gap_text=f'{gap:.2%}' if gap is not None else 'NA'
        wait_text=f'{wait:.4f}' if wait is not None else 'NA'
        lines.append(f"| {item['rate']:g} | {item['seed']} | {real['stability']['status']} | {v['recomputed_fraction']:.3%} | {item['models']['lru']['recomputation']['recomputed_fraction']:.3%} | {item['models']['oracle']['recomputation']['recomputed_fraction']:.3%} | {gap_text} | {wait_text} | {v['hit_exceeds_history_anomalies']} |")
    lines+=['','## 固定服务时间模型（完整登记人口）','','| λ | 种子 | LRU重算 | Oracle重算 | LRU间隙/可避免 | 驱逐时刻一致率（LRU/Oracle） |','|---|---|---:|---:|---:|---:|']
    for item in report['runs']:
        lru=item['models']['lru'];oracle=item['models']['oracle']
        lines.append(f"| {item['rate']:g} | {item['seed']} | {lru['recomputation']['recomputed_fraction']:.3%} | {oracle['recomputation']['recomputed_fraction']:.3%} | {lru['recomputation']['gap_share_of_avoidable']:.2%} | {lru['summary']['gap_eviction_audit']['consistency']:.1%} / {oracle['summary']['gap_eviction_audit']['consistency']:.1%} |")
    lines+=['','## 未纳入曲线的运行','']
    for item in report['runs']:
        if item['status']!='complete':lines.append(f"- {item['name']}：{item['status']}")
        elif not item['real']['stability']['stable']:
            failed=', '.join(k for k,v in item['real']['stability']['checks'].items() if not v)
            lines.append(f"- [{item['name']}]({item['name']}/real/e0-analysis.json)：{failed}")
    lines+=['','![重算与同人口模型](recomputation.png)','','逐次新判据数值见run.json各real.stability；队列曲线及两卡dmon保存在各real目录。',
            '', '```bash', 'python -m agentkv.bench.e0 --output-dir /tmp/new-e0-config11', '```']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--models-only',action='store_true')
    args=parser.parse_args();run(args.output_dir,models_only=args.models_only)


if __name__=='__main__':main()
