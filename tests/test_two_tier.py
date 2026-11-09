"""Semantic tests: prefix upper bound, duplicate CPU writes, pinning and timing."""
import numpy as np
import pytest

from agentkv.bench.capacity import root_profile
from agentkv.bench.estimate_replay import simulate_requests
from agentkv.bench.steady_state import RootSchedule, SessionInstance
from agentkv.sim.two_tier import TwoTierLRU, canonical_prefix_blocks, simulate


def plan(prompts, sends=None):
    rows=[]
    for i,prompt in enumerate(prompts):
        chain='chain' if sends is None else f'chain-{i}'
        turn=i if sends is None else 0
        final=sends is not None or i==len(prompts)-1
        rows.append(dict(root_id='r',chain_id=chain,chain_turn=turn,trace_req_idx=str(i),
                         prompt_len=len(prompt),max_tokens=1,anchor_trace_req_idx=None,
                         start_delay_s=(0 if sends is None else sends[i]) if not turn else None,
                         gap_after_s=None if final else .1,final=final,
                         _tokens=np.array(prompt,dtype='<i4').tobytes()))
    times=simulate_requests(rows,.5)
    end=max(times.values())+.5
    schedule=[dict(root_id='r',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],
                   est_send_s=times[r['trace_req_idx']],est_duration_s=end) for r in rows]
    root=RootSchedule(rows,schedule)
    return rows,root,times


def test_single_chain_unlimited_cache_matches_all_prior_lcp_upper_exactly():
    rows,root,times=plan([[1,2,3],[1,2,3,4,5],[1,2,8],[1,2,3,4,5,6]])
    hits,_=root_profile(rows,times)
    blocks,_=canonical_prefix_blocks(rows,1)
    summary,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                              10000,20000,1,duration=10,warmup=0,block_size=1)
    assert [r['gpu_hit']+r['cpu_hit'] for r in records]==[hits[r['trace_req_idx']] for r in records]
    total=sum(r['prompt_len'] for r in rows)
    assert sum(hits.values())/total==1-summary['source_fractions']['recomputed']
    assert summary['gpu_evictions']==summary['cpu_evictions']==0


def test_cpu_smaller_than_gpu_adds_no_hits_for_sequential_cyclic_trace():
    # Working set exceeds GPU, so this also distinguishes duplicate CPU
    # writes from an exclusive victim cache that could help despite its size.
    prompts=[[j*10+k for k in range(4)] for j in range(6)]*20
    rows,root,_=plan(prompts)
    blocks,_=canonical_prefix_blocks(rows,1)
    small,_=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                     20,8,1,duration=80,warmup=0,block_size=1)
    assert small['source_fractions']['cpu_hit']==0
    assert small['gpu_evictions']>0
    # A larger CPU mirror retains entries after GPU eviction and adds hits.
    larger,_=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                      8,32,1,duration=80,warmup=0,block_size=1)
    assert larger['source_fractions']['cpu_hit']>.1
    assert larger['gpu_evictions']>0


def test_pending_requests_do_not_create_hits_and_instances_do_not_share():
    rows,root,_=plan([[1,2,3]]*3,sends=[0,.25,.75])
    blocks,_=canonical_prefix_blocks(rows,1)
    summary,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                              100,200,8,duration=2,warmup=0,block_size=1)
    assert [r['gpu_hit'] for r in records]==[0,0,3]
    _,isolated=simulate({'r':root},[SessionInstance('r',0,0,0,False),SessionInstance('r',2,0,.6,False)],blocks,
                              100,200,8,duration=3,warmup=0,block_size=1)
    second_first=next(r for r in isolated if r['instance_id']==2 and r['trace_req_idx']=='0')
    assert second_first['gpu_hit']==second_first['cpu_hit']==0


def test_running_kv_is_pinned_and_queues_until_completion():
    rows,root,_=plan([[1,2,3],[4,5,6]],sends=[0,0])
    blocks,_=canonical_prefix_blocks(rows,1)
    summary,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                              4,8,8,duration=2,warmup=0,block_size=1)
    assert [r['send_s'] for r in records]==[0,.5]
    assert summary['peak_gpu_tokens']==4
    assert summary['queued_requests']==1 and summary['kv_blocked_checks']>0


def test_partial_blocks_recompute_and_parent_prefix_is_part_of_identity():
    rows,root,_=plan([[1,2,3],[1,2,3,4],[8,9,3,4]])
    blocks,_=canonical_prefix_blocks(rows,2)
    assert blocks['0'][0]==blocks['1'][0]
    assert blocks['1'][1]!=blocks['2'][1]  # same suffix with different prefix
    _,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,
                       100,200,1,duration=3,warmup=0,block_size=2)
    assert records[1]['gpu_hit']==2
    assert records[1]['recomputed']==2


def test_single_request_exceeding_gpu_pool_is_rejected():
    cache=TwoTierLRU(2,4,1)
    with pytest.raises(ValueError,match='exceeds'):
        cache.begin([1,2],2,1)


def test_oracle_farthest_next_use_never_used_first_and_atomic_protection():
    from agentkv.sim.two_tier import FutureUses, OraclePolicy, CPUCachePolicy
    future=FutureUses([(10,'a',[1]),(20,'b',[2])])
    policy=OraclePolicy(future)
    assert isinstance(policy,CPUCachePolicy)
    for block in (1,2,3): policy.on_access(block)
    assert policy.evict(2)==[3,2]
    policy.on_access(2);policy.on_access(3)
    assert policy.evict(3,{1}) is None
    assert len(policy)==3
    assert policy.evict(2,{1})==[3,2]


def test_oracle_does_not_drop_overdue_unissued_requests():
    from agentkv.sim.two_tier import FutureUses
    future=FutureUses([(1,'queued',[1]),(2,'issued',[1]),(3,'other',[2])])
    future.consume('issued')
    assert future.next_use(1)==1
    future.consume('queued')
    assert future.next_use(1)==float('inf')


def test_oracle_cpu_hits_at_least_lru_on_small_access_sequence():
    rows,root,_=plan([[x] for x in [1,2,3,1,2,4,1,2,3,4]])
    blocks,_=canonical_prefix_blocks(rows,1)
    args=({'r':root},[SessionInstance('r',0,0,0,False)],blocks,2,3,1)
    lru,_=simulate(*args,duration=10,warmup=0,block_size=1,cpu_policy='lru')
    oracle,_=simulate(*args,duration=10,warmup=0,block_size=1,cpu_policy='oracle')
    assert oracle['source_tokens']['cpu_hit']>=lru['source_tokens']['cpu_hit']
    assert oracle['source_tokens']['recomputed']<lru['source_tokens']['recomputed']


def test_infinite_tiers_lru_and_oracle_have_identical_sources():
    rows,root,_=plan([[1,2,3],[1,2,3,4,5],[1,2,8],[1,2,3,4,5,6]])
    blocks,_=canonical_prefix_blocks(rows,1)
    args=({'r':root},[SessionInstance('r',0,0,0,False)],blocks,10000,20000,1)
    lru,lru_rows=simulate(*args,duration=10,warmup=0,block_size=1,cpu_policy='lru')
    oracle,oracle_rows=simulate(*args,duration=10,warmup=0,block_size=1,cpu_policy='oracle')
    assert lru==oracle and lru_rows==oracle_rows


def test_oracle_lazy_heap_matches_independent_reference_with_overdue_queue():
    from agentkv.sim.two_tier import FutureUses,OraclePolicy
    rng=np.random.default_rng(8712)
    events=[(float(i),i,tuple(int(v) for v in rng.choice(50,size=4,replace=False))) for i in range(600)]
    actual=OraclePolicy(FutureUses(events));reference=FutureUses(events);resident=set()
    for step,number in enumerate(rng.permutation(len(events))):
        _,request,keys=events[int(number)]
        now=1000.+step  # all expected deadlines have passed; queue is unissued
        actual.on_request(request,keys,time_s=now);reference.consume(request)
        count=max(0,len(resident)+sum(key not in resident for key in keys)-20)
        expected=sorted(resident-set(keys),key=lambda key:(reference.next_use(key),key),reverse=True)[:count]
        victims=actual.evict(count,set(keys),time_s=now)
        assert victims==expected
        resident.difference_update(victims)
        for key in reversed(keys):
            actual.on_access(key,time_s=now);resident.add(key)
        assert set(actual.resident)==resident
