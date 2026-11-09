"""Independent paired-source and shared-pilot audit; does not rerun GPU work."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import ROOT,SCHEMA
from agentkv.bench.shared_pilot import parse_dmon


def validate(oracle_dir,pilot_dir,output):
    oracle=json.loads((oracle_dir/'run.json').read_text())
    assert oracle['status']=='complete' and len(oracle['results'])==14
    baseline_path=ROOT/'results/m1-two-tier-seed0/run.json'
    assert sha256_file(baseline_path)==oracle['source']['baseline_sha256']
    baseline=json.loads(baseline_path.read_text())
    old={r['directory']:r for r in baseline['results'] if r['recommend_for_gpu_pilot']}
    assert {r['id'] for r in oracle['results']}==set(old)
    last=math.inf;sim_checked=0
    for result in oracle['results']:
        tables={}
        assert result['candidate']==old[result['id']]['candidate']
        for policy in ('lru','oracle'):
            directory=oracle_dir/result['id']/policy
            record=json.loads((directory/'run.json').read_text())
            assert record==result['policies'][policy]
            assert record['git_commit']==oracle['git_commit']
            assert sha256_file(directory/'requests.parquet')==record['requests_sha256']
            rows=pq.read_table(directory/'requests.parquet').to_pylist();tables[policy]=rows
            window=[r for r in rows if r['in_window']]
            counts={k:sum(r[k] for r in window) for k in ('gpu_hit','cpu_hit','recomputed')}
            total=sum(r['prompt_tokens'] for r in window)
            assert counts==record['summary']['source_tokens']
            assert sum(counts.values())==total
            for k in counts:assert math.isclose(counts[k]/total,record['summary']['source_fractions'][k],abs_tol=1e-12)
            for r in rows:
                assert r['send_s']>=r['due_s']-1e-9
                assert math.isclose(r['end_s']-r['send_s'],.5,abs_tol=1e-9)
                assert sum(r[k] for k in counts)==r['prompt_tokens']
            sim_checked+=len(rows)
        assert result['policies']['lru']['summary']==old[result['id']]['summary']
        assert [(r['instance_id'],r['trace_req_idx'],r['send_s'],r['gpu_hit']) for r in tables['lru']]==[
               (r['instance_id'],r['trace_req_idx'],r['send_s'],r['gpu_hit']) for r in tables['oracle']]
        delta=100*(result['policies']['lru']['summary']['source_fractions']['recomputed']-
                   result['policies']['oracle']['summary']['source_fractions']['recomputed'])
        assert math.isclose(delta,result['recompute_reduction_pp'],abs_tol=1e-10)
        assert delta<=last+1e-12;last=delta
    target=next(r for r in oracle['results'] if r['id']=='config-11')
    assert oracle['noticeably_larger_than_target']==[r['id'] for r in oracle['results'] if r['recompute_reduction_pp']-target['recompute_reduction_pp']>=2]
    run=json.loads((pilot_dir/'run.json').read_text());manifest=json.loads((pilot_dir/'server-manifest.json').read_text())
    assert run['status']=='complete' and run['git_commit']==oracle['git_commit']==manifest['agentkv_commit']
    assert all(k not in run for k in ('git','git_at_end','implementation_sha256'))
    p=run['parameters']
    for k,v in dict(duration=1200,warmup=300,arrival_rate=1,max_inflight=128,hint_tier='none',policy='lru',steady_state=True,allow_dirty=False).items():assert p[k]==v,(k,p[k])
    assert manifest['gpu_cache_tokens']==182880 and manifest['cpu_cache_tokens']==274320
    assert manifest['gpu_blocks']==11430 and manifest['max_num_seqs']==8 and manifest['max_model_len']==16384
    assert manifest['gpu_cache_bytes']==26966753280 and manifest['cpu_bytes']==40450129920
    assert sha256_file(pilot_dir/'requests.parquet')==run['requests_sha256']
    assert sha256_file(pilot_dir/'loop_lag.parquet')==run['loop_lag_sha256']
    columns=[n for n in SCHEMA.names if n!='prompt_token_ids']
    source=pq.read_table(ROOT/'results/m1-workload-seed0-v3-16k/workload.parquet',columns=columns,use_threads=False).to_pylist()
    original={r['trace_req_idx']:r for r in source}
    records=pq.read_table(pilot_dir/'requests.parquet',columns=[*columns,'instance_id','is_entry','request_id','server_request_id','send_ns','end_ns','due_ns','in_window','status','completion_tokens','gpu_hit','cpu_hit','recomputed']).to_pylist()
    for r in records:
        for k in columns:assert r[k]==original[r['trace_req_idx']][k],(r['trace_req_idx'],k)
        elapsed=(r['send_ns']-run['origin_ns'])/1e9
        assert elapsed<1200
        assert r['in_window']==(300<=elapsed<=1200 and not r['is_entry'])
        assert r['end_ns']>=r['send_ns']>=r['due_ns']
        if r['status']=='success':
            assert r['completion_tokens']==r['max_tokens']
            assert sum(r[k] for k in ('gpu_hit','cpu_hit','recomputed'))==r['prompt_len']
    window=[r for r in records if r['in_window'] and r['status']=='success']
    total=sum(r['prompt_len'] for r in window)
    actual={k:sum(r[k] for r in window)/total for k in ('gpu_hit','cpu_hit','recomputed')}
    assert actual==run['summary']['nominal_token_weighted_fraction']
    analysis=json.loads((pilot_dir/'pilot-analysis.json').read_text())
    for k in actual:assert math.isclose(analysis['difference_pp'][k],100*(actual[k]-old['config-11']['summary']['source_fractions'][k]),abs_tol=1e-10)
    dmon=parse_dmon((pilot_dir/'dmon.log').read_text())
    clock=manifest['clock_pair'];origin_epoch=clock['wall_time_s']+(run['origin_ns']-clock['monotonic_ns'])/1e9
    for gpu in (0,1):
        data=[r for r in dmon if r['gpu']==gpu and 300<=r['wall_time_s']-origin_epoch<=1200]
        assert len(data)>=800,(gpu,len(data))
        assert all(0<=r['sm']<=100 for r in data)
    assert sha256_file(pilot_dir/'dmon.log')==analysis['dmon_sha256']
    metrics=[json.loads(s) for s in (pilot_dir/'metrics.jsonl').read_text().splitlines()]
    valid=[r for r in metrics if 300<=r['elapsed_s']<=1200 and 'vllm:num_requests_running' in r.get('values',{})]
    running=[r['values']['vllm:num_requests_running'] for r in valid]
    waiting=[r['values']['vllm:num_requests_waiting'] for r in valid]
    assert np.median(running)==analysis['running']['p50']
    assert np.quantile(running,.9)==analysis['running']['p90']
    assert np.median(waiting)==analysis['waiting']['p50']
    matched=oracle['target_matched_window'];matched_dir=oracle_dir/'target-matched-window'
    assert sha256_file(matched_dir/'requests.parquet')==matched['requests_sha256']
    result=dict(passed=True,git_commit=run['git_commit'],simulated_configurations=14,simulation_requests_checked=sim_checked,
                real_requests_checked=len(records),window_success=len(window),two_gpu_dmon_samples=len(dmon),
                target_oracle_reduction_pp=target['recompute_reduction_pp'],
                noticeably_larger=oracle['noticeably_larger_than_target'],actual=actual,
                checks=['input/source equality','paired send/GPU identity','all token sums','LRU baseline exact regression',
                        'oracle ranking arithmetic','clean commit-only provenance','exact resource configuration',
                        'window selection','two-GPU monitor coverage','queue and running statistics'])
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',type=Path,required=True);parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();validate(args.oracle,args.pilot,args.output)
