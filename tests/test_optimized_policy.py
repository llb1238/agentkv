import random
from types import SimpleNamespace as NS
import pytest
from agentkv.hints import make_envelope
from agentkv.policy.hint_aware import HintAwarePolicy
from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy


@pytest.mark.parametrize('scoring',['deadline','remaining','lru_final'])
@pytest.mark.parametrize('tail',[False,True])
def test_indexed_priorities_preserve_all_eviction_decisions(scoring,tail):
    now=[100.];rng=random.Random(771)
    options=dict(clock=lambda:now[0],scoring=scoring,tail_first=tail)
    a=HintAwarePolicy(32,**options);b=OptimizedHintAwarePolicy(32,**options)
    pool=list(range(20));turns={}
    for step in range(200):
        now[0]+=rng.random()*2
        chain=str(rng.randrange(4));turn=turns.get(chain,0);turns[chain]=turn+1
        keys=rng.sample(pool,rng.randrange(1,9));final=rng.randrange(5)==0
        hint=make_envelope(dict(chain_id=chain,chain_turn=turn,final=final,
            hint_perfect_ms=None if final else rng.random()*3000),'perfect',f'{chain}/{turn}','r#0')
        ctx=NS(req_id=f'{chain}/{turn}',kv_hints=hint,kv_transfer_params=None,
            get_offload_key_position={k:16*(i+1) for i,k in enumerate(keys)}.get)
        for actor in (a,b):
            actor.on_store_miss(keys,ctx)
            for k in keys:
                if actor.get(k) is None:actor.insert(k,NS(ref_cnt=0,chunk_id=k))
            actor.on_request_finished((keys,),set(keys),set(),ctx)
        if a.chunks and step%3==0:
            key=next(iter(a.chunks))
            for actor in (a,b):
                actor.chunks[key].ref_cnt=1;actor.mark_non_evictable(key)
        protected=set(rng.sample(pool,rng.randrange(4)));n=rng.randrange(6)
        first=a.evict(n,protected);second=b.evict(n,protected)
        assert (None if first is None else [k for k,_ in first])==(None if second is None else [k for k,_ in second])
        assert set(a.chunks)==set(b.chunks)
        for actor in (a,b):
            for key,chunk in actor.chunks.items():
                if chunk.ref_cnt:
                    chunk.ref_cnt=0;actor.mark_evictable(key)


def test_duplicate_refresh_does_not_accumulate_priority_garbage():
    now=[0.];p=OptimizedHintAwarePolicy(4,clock=lambda:now[0],scoring='deadline')
    p.insert(1,NS(ref_cnt=0))
    for _ in range(1000):p.mark_evictable(1)
    assert sum(map(len,p._heaps.values()))==1
    assert p.evict(1,set())[0][0]==1


@pytest.mark.parametrize('implementation',[HintAwarePolicy,OptimizedHintAwarePolicy])
def test_cancelled_crossovers_preserve_timeout_cleanup_before_chain_reuse(implementation):
    now=[0.];p=implementation(8,clock=lambda:now[0],scoring='deadline');key=b'prefix'
    def context(turn,*,final=False,gap=1000.):
        hint=make_envelope(dict(chain_id='a',chain_turn=turn,final=final,
            hint_perfect_ms=None if final else gap),'perfect',str(turn),'r#0')
        return NS(req_id=str(turn),kv_hints=hint,kv_transfer_params=None,
            get_offload_key_position=lambda k:16)
    unknown=NS(req_id='b',kv_hints=None,kv_transfer_params=None,instance_id='r#0',chain_id='b',
        get_offload_key_position=lambda k:16)
    p.on_store_miss([key],unknown);p.insert(key,NS(ref_cnt=0))
    p.on_request_finished(((key,),),set(),set(),context(0))
    # Shared known/unknown ownership creates cancellable score crossings.
    for _ in range(160):p.mark_evictable(key)
    p.on_request_finished(((key,),),set(),set(),context(1,final=True))
    p.evict(1,set())
    identity=('r#0','r#0/a')
    assert identity not in p.chains
    p.on_request_finished((),set(),set(),context(2,gap=100000.))
    now[0]=4.
    p.on_store_miss([],context(2,gap=100000.))
    # A discarded timeout from the old generation must not expire this chain.
    assert p.chains[identity].deadline==100. and not p.chains[identity].expired
