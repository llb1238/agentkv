from dataclasses import asdict
from types import SimpleNamespace as NS
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentkv.bench import e1_gpu
from agentkv.analysis.e1_gpu import analyze,aggregates,verify_result
from agentkv.bench.e0 import write_json,population_hash
from agentkv.bench.steady_state import sample_population
from agentkv.bench.replay import OUTPUT_SCHEMA
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.light_metrics import LightMetrics


def test_order_fixed_grouped_and_complete():
    a=e1_gpu.plan();assert a==e1_gpu.plan()
    assert [(g['rate'],g['seed']) for g in a]==[(r,s) for r in (1.,1.4) for s in (3,4,5)]
    assert len(a)==6 and all(set(g['order'])==set(e1_gpu.POLICIES) and len(g['order'])==4 for g in a)
    assert len({tuple(g['order']) for g in a})>1


def test_analysis_effective_replay_uint64_completion_and_stability(tmp_path):
    roots={'r':NS(duration_s=3000,triggers={'c':(None,0),'a':(None,0),'b':('a',0)})}
    blocks={'a':np.array([1]),'b':np.array([1,2]),'c':np.array([1,2])}
    instances=sample_population({'r':3000},1.,3,2400,True)
    rows=[];store=LightMetrics(tmp_path/'server-observations',pid=1);store.set_window(0,600,2400)
    store.record('kv_observer_ready',max_num_seqs=8)
    store.counters.update(effective_observer_ready=1,effective_completed_batches=3)
    for idx,t,prompt,g,c,start,entry in [('c',100,32,0,0,0,True),('a',610,16,0,0,0,False),('b',1600,32,16,16,31,False)]:
        row=dict(request_id=idx,server_request_id='server-'+idx,trace_req_idx=idx,root_id='r',chain_id=idx,chain_turn=0,
            instance_id=instances[0].instance_id,prompt_len=prompt,prompt_token_ids=list(range(prompt)),permutation_seed=2**64-1,
            send_ns=t*10**9,end_ns=int((t+.2)*1e9),ttft_ms=100.,status='success',is_entry=entry,in_window=not entry,
            queued_ms=0.,sent_prompt_sha256=idx,max_tokens=1)
        rows.append(row);rid='server-'+idx
        store.record('request_received',ns=t*10**9,request_id=rid)
        store.record('request_scheduled',ns=t*10**9+1000,request_id=rid)
        store.record('hint_received',ns=t*10**9,request_id=rid,message_id=idx)
        store.record('prefill',ns=t*10**9,request_id=rid,prompt_tokens=prompt,gpu_hit=g,cpu_hit=c,recomputed=prompt-g-c)
        store.record_forward(rid,start,prompt,prompt)
    store.close()
    pq.write_table(pa.Table.from_pylist(rows,schema=OUTPUT_SCHEMA),tmp_path/'requests.parquet')
    source=dict(workload_sha256='input',schedule_sha256='schedule')
    write_json(tmp_path/'run.json',dict(git_commit='code',vllm_commit=e1_gpu.BACKEND,status='complete',origin_ns=0,
        instances=[asdict(i) for i in instances],source=source,requests_sha256=sha256_file(tmp_path/'requests.parquet'),
        summary=dict(hint_sent=3,window_failed=0)))
    original=(tmp_path/'run.json').read_bytes()
    write_json(tmp_path/'server-manifest.json',dict(vllm_commit=e1_gpu.BACKEND,policy='lru',max_model_len=16384,
        max_num_seqs=8,gpu_cache_tokens=182880,cpu_cache_tokens=274320,duration_s=2400,warmup_s=600,rate_per_minute=1.,population_seed=3))
    write_json(tmp_path/'policy-config.json',dict(hint_tier='online'))
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(dict(elapsed_s=t,values={
        'vllm:num_requests_waiting':0,'vllm:num_requests_running':1}))+'\n' for t in range(600,2401,10)))
    (tmp_path/'dmon.log').write_text('20261104 13:00:00 0 94 32 34 0 0 0 0 0 0 3201 1785 133902 133902 0\n20261104 13:00:00 1 94 32 34 0 0 0 0 0 0 3201 1785 133902 133902 0\n')
    group=dict(name='lambda-1-seed-3',rate=1.,seed=3)
    prediction=dict(group=group['name'],policy='lru',population_sha256=population_hash(instances),predicted_effective_recompute=.5)
    result=analyze(tmp_path,roots,blocks,dict(backend_commit=e1_gpu.BACKEND,source=source,predictions=[prediction]),group,'lru')
    assert result['stability']['stable'] and result['window_completed_all']==2
    assert result['effective_recompute']==17/48 and result['nominal_recompute']==16/48
    assert result['replay_overlap_tokens']==1 and result['repeated_prompt_work']==0
    assert (tmp_path/'run.json').read_bytes()==original
    assert pq.read_table(tmp_path/'request-metadata.parquet')['permutation_seed'].to_pylist()==[2**64-1]*3
    verify_result(tmp_path,result)
    (tmp_path/'metrics.jsonl').write_text('corrupt')
    with pytest.raises(AssertionError):verify_result(tmp_path,result)


def test_pairs_require_both_stable_and_do_not_fill_missing_seeds():
    rows=[dict(rate=1.,policy='online',stable=s,paired_eligible=p,complete=True,effective_recompute=v,
        ttft_p50_ms=100,ttft_p99_ms=200,paired_recompute_pp=-10,paired_ttft_p50_ms=-1,paired_ttft_p99_ms=-2,
        prediction_gap_pp=0,completed=2,completed_non_entry=2) for s,p,v in [(True,True,.2),(False,False,.3),(True,False,.4)]]
    a=next(a for a in aggregates(rows) if a['rate']==1 and a['policy']=='online')
    assert a['stable_count']==2 and a['effective_recompute']['mean']==pytest.approx(.3)
    assert a['paired_recompute_pp']['n']==1 and a['paired_recompute_pp']['mean']==-10


def test_interrupted_retry_once_and_checkpoint_resume(tmp_path,monkeypatch):
    from agentkv.bench import shared_pilot
    from agentkv.analysis import e1_gpu as analysis
    protocol=dict(groups=[dict(name='g',rate=1.,seed=3,order=['lru'])])
    monkeypatch.setattr(e1_gpu,'verify_protocol',lambda:('code','registration',protocol))
    monkeypatch.setattr(e1_gpu,'PROTOCOL',tmp_path/'protocol.json');write_json(e1_gpu.PROTOCOL,protocol)
    monkeypatch.setattr(e1_gpu,'load_inputs',lambda:({},{}))
    monkeypatch.setattr(e1_gpu.subprocess,'check_output',lambda *a,**kw:'0, 64113\n1, 64113\n')
    monkeypatch.setattr(analysis,'render',lambda *a:None)
    attempts=[]
    def fail(directory,*a,**kw):directory.mkdir();attempts.append(directory);raise RuntimeError('service interrupted')
    monkeypatch.setattr(shared_pilot,'run',fail)
    s=e1_gpu.run(tmp_path/'output',port=0)
    assert len(attempts)==2 and s['runs'][0]['status']=='failed'
    assert all((p/'failure.json').exists() for p in attempts)
    e1_gpu.run(tmp_path/'output',port=0)
    assert len(attempts)==2,'Resume must not silently retry a twice-failed configuration'


def test_correctness_failure_is_not_retried(tmp_path,monkeypatch):
    from agentkv.bench import shared_pilot
    from agentkv.analysis import e1_gpu as analysis
    protocol=dict(groups=[dict(name='g',rate=1.,seed=3,order=['lru'])])
    monkeypatch.setattr(e1_gpu,'verify_protocol',lambda:('code','registration',protocol))
    monkeypatch.setattr(e1_gpu,'PROTOCOL',tmp_path/'protocol.json');write_json(e1_gpu.PROTOCOL,protocol)
    monkeypatch.setattr(e1_gpu,'load_inputs',lambda:({},{}))
    monkeypatch.setattr(e1_gpu.subprocess,'check_output',lambda *a,**kw:'0, 64113\n1, 64113\n')
    monkeypatch.setattr(analysis,'render',lambda *a:None)
    def fail(directory,*a,**kw):directory.mkdir();raise AssertionError('missing observation')
    monkeypatch.setattr(shared_pilot,'run',fail)
    with pytest.raises(AssertionError):e1_gpu.run(tmp_path/'output',port=0)
    s=json.loads((tmp_path/'output/checkpoint.json').read_text())
    assert s['runs'][0]['status']=='blocked' and len(s['runs'][0]['attempts'])==1


def test_completed_checkpoint_is_verified_and_never_replayed(tmp_path,monkeypatch):
    from agentkv.analysis import e1_gpu as analysis
    protocol=dict(groups=[dict(name='g',rate=1.,seed=3,order=['lru'])])
    monkeypatch.setattr(e1_gpu,'verify_protocol',lambda:('code','registration',protocol))
    monkeypatch.setattr(e1_gpu,'PROTOCOL',tmp_path/'protocol.json');write_json(e1_gpu.PROTOCOL,protocol)
    monkeypatch.setattr(e1_gpu,'load_inputs',lambda:({},{}));monkeypatch.setattr(analysis,'render',lambda *a:None)
    calls=[];monkeypatch.setattr(analysis,'verify_result',lambda *a:calls.append('verified'))
    def no_gpu(*a,**kw):raise AssertionError('Completed work must not query or start GPU')
    monkeypatch.setattr(e1_gpu.subprocess,'check_output',no_gpu)
    out=tmp_path/'output';directory=out/'g/lru/attempt-1';directory.mkdir(parents=True)
    write_json(directory/'e1-analysis.json',dict(status='complete'))
    item=dict(id='g/lru',group='g',rate=1.,seed=3,policy='lru',status='complete',attempts=[dict(status='complete')],
        selected_attempt='g/lru/attempt-1',analysis_sha256=sha256_file(directory/'e1-analysis.json'))
    write_json(out/'checkpoint.json',dict(git_commit='code',preregistration_commit='registration',
        protocol_sha256=sha256_file(e1_gpu.PROTOCOL),status='partial',runs=[item]))
    e1_gpu.run(out)
    assert calls==['verified']


@pytest.mark.parametrize('labels,rate',[(e1_gpu.POLICIES,None),(('lru','online','perfect'),None),(('lru','online','perfect'),1.2)])
def test_report_all_planned_cases_and_cpu_predictions_without_gpu(tmp_path,labels,rate):
    pytest.importorskip('matplotlib')
    from agentkv.analysis.e1_gpu import render
    predictions=[];runs=[]
    groups=e1_gpu.plan() if rate is None else e1_gpu.registered_plan(dict(format='agentkv.stable_limit.v1',
        policies=list(labels),order_seed=20261107))
    for group in groups:
        group['order']=[p for p in group['order'] if p in labels]
        for label in labels:
            predictions.append(dict(group=group['name'],rate=group['rate'],seed=group['seed'],policy=label,
                predicted_effective_recompute=.2))
            runs.append(dict(id=group['name']+'/'+label,group=group['name'],rate=group['rate'],seed=group['seed'],
                policy=label,status='pending',attempts=[]))
    state=dict(status='pending',preregistration_commit='registration',protocol_sha256='protocol',runs=runs)
    render(tmp_path,state,dict(groups=groups,policies=labels,predictions=predictions,queue_rate=rate or 1.4))
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['rows'])==len(groups)*len(labels) and all(r['predicted']==.2 for r in summary['rows'])
    assert {a['policy'] for a in summary['aggregates']}==set(labels)
    assert all(a['stable_count']==0 for a in summary['aggregates'])
    queues='lambda14_queues.pdf' if rate is None else 'lambda12_queues.pdf'
    assert (tmp_path/'recomputation.png').exists() and (tmp_path/queues).exists()


def test_v2_protocol_chain_config_and_retry_resume(tmp_path,monkeypatch):
    from agentkv.bench import shared_pilot,e1_v2
    from agentkv.analysis import e1_gpu as analysis
    groups=e1_v2.groups()
    assert all(g['order']==[p for p in original['order'] if p!='lru-final']
               for g,original in zip(groups,e1_gpu.plan()))
    assert sum(len(g['order']) for g in groups)==18
    protocol=dict(groups=[dict(name='g',rate=1.,seed=3,order=['online'])],policy_implementation='chain')
    path=tmp_path/'custom.json';write_json(path,protocol);verified=[]
    def verify(p):
        assert p==path;verified.append(p);return 'code','registration',protocol
    monkeypatch.setattr(e1_gpu,'verify_protocol',verify)
    monkeypatch.setattr(e1_gpu,'load_inputs',lambda:({},{}))
    monkeypatch.setattr(e1_gpu.subprocess,'check_output',lambda *a,**kw:'0, 64113\n1, 64113\n')
    monkeypatch.setattr(analysis,'render',lambda *a:None)
    configs=[]
    def fail(directory,*a,**kw):
        directory.mkdir();configs.append(json.loads(Path(kw['extra_environment']['AGENTKV_HINT_POLICY_CONFIG']).read_text()))
        raise RuntimeError('interrupted')
    from pathlib import Path
    monkeypatch.setattr(shared_pilot,'run',fail)
    state=e1_gpu.run(tmp_path/'output',port=0,protocol_path=path)
    assert state['status']=='complete_with_failures'
    assert len(configs)==2 and all(c['implementation']=='chain' for c in configs)
    e1_gpu.run(tmp_path/'output',port=0,protocol_path=path)
    assert len(configs)==2 and len(verified)==4
