"""Independent checks of exact prefix unions, temporal upper bounds and KV simulation."""
import numpy as np
import pytest

from agentkv.bench.capacity import PrefixSet, root_profile, running_kv, saturated_kv, weighted_quantile
from agentkv.bench.estimate_replay import simulate_requests
from agentkv.bench.steady_state import RootSchedule, SessionInstance
from agentkv.bench.synth import common_prefix


def key(tokens):
    return np.asarray(tokens,dtype='<i4').tobytes()


def test_prefix_multiset_matches_explicit_trie_under_insert_delete():
    rng=np.random.default_rng(40)
    seq=[rng.integers(0,4,size=int(rng.integers(0,15))).tolist() for _ in range(80)]
    seq.extend(seq[:5])
    live=[]; prefixes=PrefixSet()
    for value in seq:
        prefixes.add(key(value)); live.append(value)
        expected={tuple(v[:i]) for v in live for i in range(1,len(v)+1)}
        assert prefixes.tokens==len(expected)
        query=rng.integers(0,4,size=10)
        assert prefixes.best(key(query))==max(common_prefix(query,np.array(v)) for v in live)
    rng.shuffle(seq)
    for value in seq:
        prefixes.remove(key(value)); live.remove(value)
        assert prefixes.tokens==len({tuple(v[:i]) for v in live for i in range(1,len(v)+1)})
    assert prefixes.tokens==0


def test_upper_uses_any_prior_chain_and_excludes_equal_time_requests():
    rows=[dict(trace_req_idx='a',chain_id='x',final=True,_tokens=key([1,2,3])),
          dict(trace_req_idx='b',chain_id='y',final=True,_tokens=key([1,2,4])),
          dict(trace_req_idx='c',chain_id='z',final=True,_tokens=key([1,2,3,5]))]
    hits,profile=root_profile(rows,{'a':0.,'b':0.,'c':.1})
    assert hits=={'a':0,'b':0,'c':3}
    i=np.searchsorted(profile['times'],.2,side='right')-1
    assert profile['tokens'][i]==5  # union [1,2], then 3,4,5
    assert profile['tokens'][-1]==0


def test_time_weighted_quantile_includes_idle_time():
    assert weighted_quantile([0,100],[99,1],.9)==0
    assert weighted_quantile([0,100],[1,99],.9)==100


def test_running_slots_and_kv_use_actual_completion_dependencies():
    rows=[]
    for chain in ('a','b'):
        for turn in (0,1):
            rows.append(dict(root_id='r',chain_id=chain,chain_turn=turn,trace_req_idx=chain+str(turn),
                             prompt_len=17,max_tokens=2,anchor_trace_req_idx=None,
                             start_delay_s=0. if not turn else None,gap_after_s=0. if not turn else None))
    sends=simulate_requests(rows,.5)
    schedule=[dict(root_id='r',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],
                   est_send_s=sends[r['trace_req_idx']],est_duration_s=1.) for r in rows]
    roots={'r':RootSchedule(rows,schedule)}
    instances=[SessionInstance('r',0,0,0,False)]
    single=running_kv(roots,instances,1,warmup=0,duration=2)
    double=running_kv(roots,instances,2,warmup=0,duration=2)
    assert single['kv_tokens_p90']==32 and single['mean_running']==1
    assert double['kv_tokens_p90']==64 and double['mean_running']==1
    assert saturated_kv(rows,8,0)['p90']==256


def test_two_tier_screening_bounds_and_quantitative_failures():
    from agentkv.bench.capacity import tier_constraints
    checks,failures,_=tier_constraints(256,384,100,1000,1,1e10,10,0)
    assert all(checks.values()) and failures=={}
    checks,failures,_=tier_constraints(256,512,100,600,1,1e10,10,0)
    assert not checks['cpu_le_80pct_working_set']
    assert failures['cpu_le_80pct_working_set']==32
    checks,failures,_=tier_constraints(256,368,100,1000,1,1e10,10,0)
    assert not checks['cpu_gpu_ratio_min'] and failures['cpu_gpu_ratio_min']==16
    checks,failures,_=tier_constraints(240,480,100,1000,1,1e10,10,0)
    assert not checks['gpu_safety'] and failures['gpu_safety']==10


def test_report_handles_no_feasible_configuration_without_gpu(tmp_path,monkeypatch):
    import json
    from agentkv.bench import report_two_tier
    from agentkv.bench.capacity import tier_constraints
    capacity_dir=tmp_path/'capacity'; capacity_dir.mkdir()
    simulation_dir=tmp_path/'sim'; simulation_dir.mkdir()
    checks,violations,limits=tier_constraints(256,384,100,400,1,1e10,10,0)
    rejected=dict(context_budget=32512,rate_per_minute=.5,max_num_seqs=8,
                  gpu_tokens=256,cpu_tokens=384,cpu_gpu_ratio=1.5,
                  sizing_kv_p90=100,working_set_p50=400,working_set_p90=500,
                  constraints=checks,violations_tokens=violations,limits=limits)
    capacity=dict(kv_bytes_per_token=147456,parameters={'non_kv_gpu_reserve_gib':30},
        candidates=[],configurations=[dict(context_budget=32512,rate_per_minute=.5,
                    working_set_tokens={'p50':400,'p90':500})],
        screened_configurations=[rejected],closest_rejected=[rejected],memory={'MemAvailable_bytes':10**10})
    (capacity_dir/'run.json').write_text(json.dumps(capacity))
    (simulation_dir/'run.json').write_text(json.dumps({'results':[]}))
    def no_gpu(*args,**kwargs): raise FileNotFoundError('nvidia-smi')
    monkeypatch.setattr(report_two_tier.subprocess,'check_output',no_gpu)
    monkeypatch.setattr(report_two_tier,'current_commit',lambda:'0'*40)
    monkeypatch.setattr(report_two_tier,'plot',lambda *args:None)
    report_two_tier.render(capacity_dir,simulation_dir)
    assert json.loads((simulation_dir/'analysis.json').read_text())['priority_pilots']==[]
    assert 'cpu_le_80pct_working_set' in (capacity_dir/'README.md').read_text()
    assert '64.0' in (capacity_dir/'README.md').read_text()
    assert '没有容量合格项' in (simulation_dir/'README.md').read_text()
