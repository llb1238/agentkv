"""Independent integrity and arithmetic audit of the CPU-only E1 rehearsal."""
import argparse
import csv
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.e0 import WORKLOAD,SCHEDULE,population_hash,write_json
from agentkv.bench.steady_state import SessionInstance
from agentkv.bench.workload import ROOT

TIERS=('perfect','noisy','online')
COMPONENTS={'no-tail':'tail_first','no-final':'final_first','no-timeout':'timeout_decay'}
LABELS={'lru','oracle','none-control'}|{'hint-'+t for t in TIERS}|{a+'-'+t for a in COMPONENTS for t in TIERS}


def read(path):return json.loads(path.read_text())
def near(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
def mean(values):return math.fsum(values)/len(values)


def validate(directory,e0_directory=ROOT/'results/m1-e0-config11-20261103'):
    commit=git_commit();run=read(directory/'run.json');e0=read(e0_directory/'run.json')
    assert run['status']=='complete' and len(run['results'])==12
    assert run['source']['e0_run_sha256']==sha256_file(e0_directory/'run.json')
    assert run['source']['workload_sha256']==sha256_file(WORKLOAD)==e0['source']['workload_sha256']
    assert run['source']['schedule_sha256']==sha256_file(SCHEDULE)==e0['source']['schedule_sha256']
    p=run['parameters']
    for name,value in dict(gpu_tokens=182880,cpu_tokens=274320,slots=8,block_size=16,duration_s=2400,warmup_s=600,service_s=.5).items():assert p[name]==value
    assert {(r['rate'],r['seed']) for r in run['results']}=={(rate,seed) for rate in (.5,.7,1.,1.4) for seed in (0,1,2)}
    pairs=[];effects=[];records_checked=cases=0;stats={}
    for population in run['results']:
        name=population['name'];folder=directory/name;original=next(r for r in e0['runs'] if r['name']==name)
        assert read(folder/'run.json')==dict(git_commit=run['git_commit'],status='complete',**population)
        data=read(folder/'population.json');instances=[SessionInstance(**r) for r in data['instances']]
        assert population_hash(instances)==data['population_sha256']==population['population_sha256']==original['population_sha256']
        assert set(population['cases'])==LABELS
        reference=pq.read_table(folder/'lru/requests.parquet').to_pylist()
        fractions={}
        for label,record in population['cases'].items():
            path=folder/label
            assert read(path/'run.json')==record and record['git_commit']==run['git_commit']
            assert record['status']=='complete' and record['population_sha256']==population['population_sha256']
            assert sha256_file(path/'requests.parquet')==record['requests_sha256']
            rows=pq.read_table(path/'requests.parquet').to_pylist();summary=record['summary']
            assert len(rows)==len(reference)==summary['requests']
            for actual,expected in zip(rows,reference):
                assert {k:v for k,v in actual.items() if k not in ('cpu_hit','recomputed')}=={k:v for k,v in expected.items() if k not in ('cpu_hit','recomputed')}
                assert actual['gpu_hit']+actual['cpu_hit']+actual['recomputed']==actual['prompt_tokens']
                assert all(actual[k]>=0 for k in ('gpu_hit','cpu_hit','recomputed'))
                assert actual['in_window']==(600<=actual['send_s']<2400 and not actual['is_entry'])
                near(actual['end_s']-actual['send_s'],.5)
            window=[r for r in rows if r['in_window']];total=sum(r['prompt_tokens'] for r in window)
            assert len(window)==summary['window_requests'] and total==summary['window_prompt_tokens']
            counts={k:sum(r[k] for r in window) for k in ('gpu_hit','cpu_hit','recomputed')}
            assert counts==summary['source_tokens'] and sum(counts.values())==total
            for key in counts:near(counts[key]/total,summary['source_fractions'][key])
            fractions[label]=counts['recomputed']/total
            if label in ('lru','oracle'):
                baseline=original['models'][label]
                assert summary=={k:v for k,v in baseline['summary'].items() if k!='gap_eviction_audit'}
                source=e0_directory/name/'models'/label/'requests.parquet'
                assert sha256_file(source)==baseline['requests_sha256']
                old=pq.read_table(source).to_pylist()
                assert rows==[{k:r[k] for k in rows[0]} for r in old]
            else:
                stat=record['policy_stats'];stats[name+'/'+label]=stat
                assert stat['evicted']==summary['cpu_evictions'] and stat['invalid_hint_callbacks']==0
                assert stat['valid_hint_requests']==(0 if label=='none-control' else len(rows))
                options=record['options'];disabled=next((field for component,field in COMPONENTS.items() if label.startswith(component+'-')),None)
                assert options=={field:field!=disabled for field in COMPONENTS.values()}
                if label.startswith('no-timeout'):assert stat['timeout_events']==0
            if label=='none-control':
                assert rows==reference and summary==population['cases']['lru']['summary']
                assert record['requests_sha256']==population['cases']['lru']['requests_sha256']
            records_checked+=len(rows);cases+=1
        for tier in TIERS:
            delta=100*(fractions['lru']-fractions['hint-'+tier]);space=fractions['lru']-fractions['oracle']
            pairs.append(dict(name=name,rate=population['rate'],seed=population['seed'],tier=tier,reduction_pp=delta,
                              oracle_capture=delta/100/space if space else None))
            for component in COMPONENTS:
                effects.append(dict(rate=population['rate'],seed=population['seed'],tier=tier,component=component,
                                    delta_pp=100*(fractions[component+'-'+tier]-fractions['hint-'+tier])))
    assert cases==180
    with (directory/'paired.csv').open() as handle:table=list(csv.DictReader(handle))
    assert len(table)==36
    for expected in pairs:
        row=next(r for r in table if r['name']==expected['name'] and r['tier']==expected['tier'])
        near(float(row['reduction_pp']),expected['reduction_pp'])
        if expected['oracle_capture'] is None:assert not row['oracle_capture']
        else:near(float(row['oracle_capture']),expected['oracle_capture'])
    with (directory/'aggregate.csv').open() as handle:aggregate=list(csv.DictReader(handle))
    assert len(aggregate)==12
    for row in aggregate:
        group=[r for r in pairs if r['rate']==float(row['rate']) and r['tier']==row['tier']]
        assert len(group)==int(row['populations'])==3
        values=[r['reduction_pp'] for r in group]
        for field,value in [('mean_pp',mean(values)),('minimum_pp',min(values)),('maximum_pp',max(values))]:near(float(row[field]),value)
        capture=[r['oracle_capture'] for r in group if r['oracle_capture'] is not None]
        if capture:
            for field,value in [('capture_mean',mean(capture)),('capture_min',min(capture)),('capture_max',max(capture))]:near(float(row[field]),value)
    with (directory/'ablations.csv').open() as handle:ablations=list(csv.DictReader(handle))
    assert len(ablations)==36
    for row in ablations:
        values=[r['delta_pp'] for r in effects if r['rate']==float(row['rate']) and r['tier']==row['tier'] and r['component']==row['component']]
        assert len(values)==3
        for field,value in [('mean_pp',mean(values)),('minimum_pp',min(values)),('maximum_pp',max(values))]:near(float(row[field]),value)
    result=dict(git_commit=commit,measurement_git_commit=run['git_commit'],passed=True,case_count=cases,
        population_count=12,requests_checked=records_checked,none_lru_identical_populations=12,e0_regression_populations=12,
        source_run_sha256=sha256_file(directory/'run.json'),
        mean_reduction_pp={tier:mean([r['reduction_pp'] for r in pairs if r['tier']==tier]) for tier in TIERS},
        mean_oracle_capture={tier:mean([r['oracle_capture'] for r in pairs if r['tier']==tier and r['oracle_capture'] is not None]) for tier in TIERS},
        mean_ablation_effect_pp={component:{tier:mean([r['delta_pp'] for r in effects if r['component']==component and r['tier']==tier]) for tier in TIERS} for component in COMPONENTS},
        max_cpu_store_skips=max(c['summary']['cpu_store_skipped_requests'] for r in run['results'] for c in r['cases'].values()),
        checks=['all source/output hashes','12 paired E0 populations','all token sums','unchanged request/schedule/GPU paths',
                'LRU/Oracle E0 regression','none-adapter byte-identical requests','paired percentage points and unclipped Oracle capture',
                'three-seed means and ranges','three independent component ablations','hint and timeout callback counts'])
    write_json(directory/'validation.json',result)
    append_observations(directory,result,pairs,effects)
    write_json(directory/'artifacts.json',dict(git_commit=commit,files={str(path.relative_to(directory)):dict(bytes=path.stat().st_size,sha256=sha256_file(path))
        for path in sorted(directory.rglob('*')) if path.is_file() and path.name!='artifacts.json'}))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result



def append_observations(directory,result,pairs,effects):
    marker='\n## 独立复核与组件贡献\n'
    body=(directory/'README.md').read_text().split(marker)[0]
    lines=[marker.strip(),'','完整矩阵180次、12个人口；none控制的逐请求parquet与LRU逐字节相同，LRU/Oracle与E0模型逐项回归。所有策略的调度、GPU命中、请求身份和prompt一致。',
           '',f"测量代码 `{result['measurement_git_commit']}`；复核代码 `{result['git_commit']}`。核验{result['requests_checked']}条逐请求记录及全部输出SHA256。",
           '', '| hint | 12人口平均减少百分点 | 12人口平均Oracle捕获率 | 收益为正人口数 |','|---|---:|---:|---:|']
    for tier in TIERS:
        positive=sum(r['reduction_pp']>0 for r in pairs if r['tier']==tier)
        lines.append(f"| {tier} | {result['mean_reduction_pp'][tier]:.3f} | {result['mean_oracle_capture'][tier]:.2%} | {positive}/12 |")
    lines+=['','下表是三个种子×四个λ全部12个人口的组件消融差值，单位百分点；负数表示去掉该组件反而减少重算。各λ分表仍以上文和ablations.csv为准。',
            '', '| 移除组件 | hint | 平均影响 | 全12人口最小–最大 |','|---|---|---:|---:|']
    for component in COMPONENTS:
        for tier in TIERS:
            values=[r['delta_pp'] for r in effects if r['component']==component and r['tier']==tier]
            lines.append(f"| {component} | {tier} | {mean(values):+.3f} | {min(values):+.3f}–{max(values):+.3f} |")
    lines+=['',f"全矩阵CPU写入跳过次数的单case最大值为{result['max_cpu_store_skips']}。本结果只适用于当前截断负载、固定服务时间和CPU镜像模型；不替代真实vLLM的收益或开销测量。",'',
           '[独立验证](validation.json) · [全部产物指纹](artifacts.json) · [策略与实验预注册](../../docs/m5_e1_simulation_design.md)',
           '', '纯CPU依赖与复现：','', '```bash',"python -m pip install -e '.[simulation]'",
           'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m agentkv.bench.e1_simulation --output-dir /tmp/new-e1 --workers 4',
           'python -m agentkv.analysis.validate_e1_simulation --run-dir /tmp/new-e1','```']
    (directory/'README.md').write_text(body+'\n'+'\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();validate(args.run_dir)


if __name__=='__main__':main()
