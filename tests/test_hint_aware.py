"""Policy correctness without any engine classes or device dependencies."""
import copy
import random
import subprocess
import sys
from collections import OrderedDict
from types import SimpleNamespace as NS
import pytest
from agentkv.policy.hint_aware import HintAwarePolicy


def legacy_policy(*args,**options):
    return HintAwarePolicy(*args,**(dict(scoring='deadline',tail_first=True)|options))


class Clock:
    now=0.
    def __call__(self):return self.now


def context(chain,keys,*,gap=1000.,margin=1000.,final=False,hint=True,turn=0):
    payload=dict(session_id=chain,parent_session='r#0',turn=turn,final=final,
                 expected_gap_ms=None if final else gap,gap_p90_ms=None if final else margin)
    envelope=dict(protocol_version=1,actions=[dict(action_type='agentkv.tool_gap',action_version=1,payload=payload)])
    return NS(req_id=f'{chain}/{turn}',instance_id='r#0',chain_id=chain,kv_hints=envelope if hint else None,
              kv_transfer_params=None,get_offload_key_position={key:i+1 for i,key in enumerate(keys)}.get)


def store(policy,keys,ctx):
    policy.on_store_miss(keys,ctx)
    for key in reversed(keys):
        if policy.get(key) is None:policy.insert(key,NS(ref_cnt=0))
    policy.on_request_finished((keys,),set(),set(),ctx)


def test_ended_chain_before_waiting_chain_and_tail_first():
    clock=Clock();policy=legacy_policy(10,clock=clock)
    store(policy,[1,2],context('a',[1,2],gap=5000))
    store(policy,[3,4],context('b',[3,4],final=True))
    assert [k for k,_ in policy.evict(3,set())]==[4,3,2]


def test_shared_prefix_keeps_earliest_live_owner():
    clock=Clock();policy=legacy_policy(10,clock=clock)
    store(policy,[1,2],context('a',[1,2],final=True))
    store(policy,[1,3],context('b',[1,3],gap=1000))
    assert [k for k,_ in policy.evict(1,set())]==[2]
    assert policy.get(1) is not None


def test_native_position_metadata_orders_scrambled_groups():
    policy=legacy_policy(10,clock=Clock());ctx=context('a',[1,2,3])
    policy.on_store_miss([3,1,2],ctx)
    for key in [1,2,3]:policy.insert(key,NS(ref_cnt=0))
    policy.on_request_finished(([3,1,2],),set(),set(),ctx)
    assert [key for key,_ in policy.evict(3,set())]==[3,2,1]


def test_no_hints_matches_lru_with_ties_and_protection():
    clock=Clock();policy=legacy_policy(10,clock=clock);reference=OrderedDict()
    rng=random.Random(913)
    for step in range(300):
        clock.now=step//3
        keys=rng.sample(range(10),rng.randint(1,5));ctx=context('a',keys,hint=False,turn=step)
        store(policy,keys,ctx)
        for key in reversed(keys):reference[key]=None;reference.move_to_end(key)
        protected=set(rng.sample(list(reference),rng.randrange(len(reference)+1)))
        n=rng.randrange(4);expected=[key for key in reference if key not in protected][:n]
        actual=policy.evict(n,protected)
        if len(expected)<n:assert actual is None
        else:
            assert [key for key,_ in actual]==expected
            for key in expected:del reference[key]
        assert set(policy.chunks)==set(reference)


def test_protected_tail_and_pinning_failure_rolls_back_partial_selection():
    p=legacy_policy(10,clock=Clock())
    store(p,[1,2],context('a',[1,2]));store(p,[3],context('b',[3],final=True))
    before=(dict(p.chunks),dict(p.ranks),copy.deepcopy(p.owners))
    assert p.evict(2,{2}) is None
    assert (p.chunks,p.ranks,p.owners)==before
    assert [key for key,_ in p.evict(1,{2})]==[3]
    p.get(2).ref_cnt=-1;p.mark_non_evictable(2)
    assert p.evict(1,set()) is None
    p.get(2).ref_cnt=1;p.mark_non_evictable(2)
    assert p.evict(1,set()) is None
    p.get(2).ref_cnt=0;p.mark_evictable(2)
    assert p.evict(0,set())==[]
    assert [k for k,_ in p.evict(2,set())]==[2,1]
    with pytest.raises(ValueError):p.evict(-1,set())


def test_timeout_degradation_changes_victim_without_another_hint():
    for enabled,expected in [(True,1),(False,2)]:
        clock=Clock();p=legacy_policy(10,clock=clock,timeout_decay=enabled)
        store(p,[1],context('a',[1],gap=0,margin=1000))
        store(p,[2],context('b',[2],gap=10000,margin=1000))
        clock.now=11
        assert p.evict(1,set())[0][0]==expected


def test_shared_lru_line_crossing_cannot_stay_hidden_in_heap():
    for now,expected in [(1,2),(6,1)]:
        clock=Clock();p=legacy_policy(10,clock=clock)
        store(p,[1],context('a',[1],gap=10000,margin=100000))
        store(p,[1],context('unknown',[1],hint=False))
        store(p,[2],context('b',[2],gap=8000,margin=100000))
        clock.now=now
        assert p.evict(1,set())[0][0]==expected


def test_new_request_cancels_old_timeout_and_final_is_applied_only_at_finish():
    clock=Clock();p=legacy_policy(10,clock=clock)
    store(p,[1],context('a',[1],gap=1000,margin=1000))
    clock.now=1;p.on_store_miss([1],context('a',[1],turn=1,final=True))
    store(p,[2],context('b',[2],gap=10000))
    clock.now=5
    assert p.evict(1,set())[0][0]==2  # A is active; final has not happened.
    assert p.stats['timeout_events']==0
    p.on_request_finished(([1],),set(),set(),context('a',[1],turn=1,final=True))
    assert p.chains['r#0','a'].mode=='final'


def test_unknown_actions_and_malformed_hint_fall_back(caplog):
    p=legacy_policy(10,clock=Clock());ctx=context('a',[1,2])
    ctx.kv_hints['actions'][0]['payload']['expected_gap_ms']=True
    store(p,[1,2],ctx)
    assert p.stats['invalid_hint_callbacks']>0 and 'using LRU' in caplog.text
    assert [k for k,_ in p.evict(1,set())]==[2]
    ctx=context('b',[3]);ctx.kv_hints['actions'][0]['action_type']='unrelated.action'
    store(p,[3],ctx)
    assert p.stats['valid_hint_requests']==0


def test_transfer_envelope_and_typed_envelope_supported():
    p=legacy_policy(10,clock=Clock());ctx=context('a',[1],final=True)
    ctx.kv_transfer_params={'kv_hints':ctx.kv_hints};ctx.kv_hints=None
    store(p,[1],ctx)
    ctx=context('b',[2]);d=ctx.kv_hints;d['actions']=[NS(**d['actions'][0])];ctx.kv_hints=NS(**d)
    store(p,[2],ctx)
    assert p.stats['valid_hint_requests']==2 and p.evict(1,set())[0][0]==1


def test_policy_import_needs_only_standard_library():
    code='import sys;from agentkv.policy.hint_aware import HintAwarePolicy;assert not any(m in sys.modules for m in ("vllm","torch","numpy"))'
    subprocess.run([sys.executable,'-S','-c',code],check=True)


def test_lazy_heap_matches_direct_affine_reference_with_shared_blocks():
    clock=Clock();p=legacy_policy(40,clock=clock);rng=random.Random(7731)
    for turn in range(300):
        clock.now+=rng.random()*2
        chain=str(rng.randrange(5));keys=list(range(1,rng.randint(2,10)))
        ctx=context(chain,keys,gap=rng.random()*20000,margin=rng.random()*1000,turn=turn)
        store(p,keys,ctx)
        protected=set(rng.sample(list(p.chunks),rng.randrange(len(p.chunks)+1)))
        candidates=[]
        for key,chunk in p.chunks.items():
            owners=p.owners.get(key,{})
            if key in protected or chunk.ref_cnt!=0:continue
            if any(c.known and depth!=max(c.keys.values()) for identity,depth in owners.items() for c in [p.chains[identity]]):continue
            scores=[]
            for identity in owners:
                c=p.chains[identity]
                if c.mode=='final':scores.append(float('inf'))
                elif c.mode=='active':scores.append(clock.now)
                elif c.mode=='waiting':scores.append(c.deadline+2*max(0,clock.now-c.deadline-c.margin))
                else:scores.append(2*clock.now-p.last_access[key])
            score=min(scores) if scores else 2*clock.now-p.last_access[key]
            candidates.append((-score,p.ranks[key],key))
        actual=p.evict(1,protected)
        assert (None if actual is None else actual[0][0])==(min(candidates)[2] if candidates else None)
