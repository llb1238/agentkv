import math
import random
from types import SimpleNamespace as NS
from agentkv.policy.hint_aware import HintAwarePolicy
from agentkv.policy.remaining_life import RemainingLifeTable
from tests.test_hint_aware import Clock,context,store


def table():
    # First bucket's residual jumps up at 1s, then exhausts at 3s.
    return RemainingLifeTable(dict(format='agentkv.remaining_life.v1',elapsed_grid_s=[0.,1.,3.],
        tiers={'online':dict(prediction_cuts_s=[5.],buckets=[
            dict(mean_remaining_s=[.2,10.,None]),dict(mean_remaining_s=[5.,5.,None])])}))


def test_residual_grid_refreshes_without_new_hint_and_defaults_disable_tail():
    for now,expected in [(0.,2),(1.,1),(3.,1)]:
        clock=Clock();p=HintAwarePolicy(10,clock=clock,residual_table=table())
        assert not p.tail_first and p.final_first and not p.timeout_decay
        store(p,[1],context('a',[1],gap=200))
        store(p,[2],context('b',[2],gap=6000))
        clock.now=now
        assert p.evict(1,set())[0][0]==expected
        assert p.stats['timeout_events']==0


def test_large_absolute_times_grid_lookup_does_not_reschedule_same_knot():
    clock=Clock();clock.now=2101.23456789
    p=HintAwarePolicy(10,clock=clock,residual_table=table())
    store(p,[1],context('a',[1],gap=200))
    clock.now+=1
    p.evict(0,set())  # zero count is a no-op
    p.mark_evictable(1)
    c=p.chains['r#0','a']
    assert c.life_index==1 and c.life_remaining==10.
    assert all(e[0]>clock.now for e in p._events if e[2]=='life' and e[4]==c.version)


def test_new_return_cancels_old_grid_event_and_shared_scores_match_direct_reference():
    clock=Clock();p=HintAwarePolicy(30,clock=clock,residual_table=table());rng=random.Random(703)
    for turn in range(140):
        clock.now+=rng.random()*1.5
        keys=list(range(1,rng.randint(2,9)));chain=str(rng.randrange(4))
        ctx=context(chain,keys,gap=rng.choice([200,6000]),turn=turn,final=rng.random()<.2)
        store(p,keys,ctx)
        protected=set(rng.sample(list(p.chunks),rng.randrange(len(p.chunks)+1)))
        expected=[]
        for key in p.chunks:
            if key in protected:continue
            scores=[]
            for identity in p.owners[key]:
                c=p.chains[identity]
                if c.mode=='final':score=math.inf
                elif c.mode=='active':score=clock.now
                elif c.mode=='waiting':
                    # Find the grid by absolute times, independently of cached index.
                    age_index=max(i for i,g in enumerate(p.residual_table.grid) if c.finished_at+g<=clock.now)
                    remaining=p.residual_table.values[c.life_bucket][age_index]
                    score=clock.now+remaining if remaining is not None else 2*clock.now-p.last_access[key]
                else:score=2*clock.now-p.last_access[key]
                scores.append(score)
            expected.append((-min(scores),p.ranks[key],key))
        actual=p.evict(1,protected)
        assert (None if actual is None else actual[0][0])==(min(expected)[2] if expected else None)


def test_final_only_uses_lru_for_nonfinal_and_ignores_interval_values():
    clock=Clock();p=HintAwarePolicy(10,clock=clock,scoring='lru_final')
    store(p,[1,2],context('a',[1,2],gap=1e9))
    clock.now=1;store(p,[3,4],context('b',[3,4],gap=1))
    assert [k for k,_ in p.evict(1,set())]==[2]  # ordinary prefix-aware LRU
    clock.now=2;store(p,[3,4],context('b',[3,4],final=True,turn=1))
    assert [k for k,_ in p.evict(2,set())]==[4,3]


def test_expired_diagnostic_deduplicates_shared_blocks_and_excludes_final():
    from agentkv.sim.hint_adapter import HintPolicyAdapter
    a=HintPolicyAdapter(10,hint_tier='online',scoring='deadline',diagnose_deadlines=True)
    store(a.policy,[1,2],context('x',[1,2],gap=1000))
    store(a.policy,[1,3],context('y',[1,3],gap=2000))
    store(a.policy,[4],context('z',[4],final=True))
    a.evict(1,time_s=3.)
    d=a.deadline_decisions[0]
    assert d['overdue_blocks']==3 and d['cpu_blocks']==4 and d['overdue_fraction']==.75
    assert d['overdue_chains']==2


def test_adapter_defaults_and_final_only_require_no_numeric_hint_columns():
    from agentkv.sim.hint_adapter import HintPolicyAdapter
    from tests.test_hint_adapter import fixture
    from agentkv.sim.two_tier import simulate
    for tier in ('online','noisy'):
        adapter=HintPolicyAdapter(10,hint_tier=tier)
        assert adapter.policy.scoring=='remaining' and not adapter.policy.tail_first
        assert adapter.policy.residual_table.tier==tier
    assert HintPolicyAdapter(10,hint_tier='perfect').policy.scoring=='deadline'
    roots,instances,blocks=fixture()
    for root in roots.values():
        for request in root.requests:
            for name in list(request):
                if name.startswith('hint_'):del request[name]
    adapter=HintPolicyAdapter(10,hint_tier='final')
    summary,rows=simulate(roots,instances,blocks,8,10,2,duration=20,warmup=0,block_size=1,cpu_policy=adapter)
    assert adapter.policy.stats['valid_hint_requests']==len(rows)


def test_evicted_then_recreated_chain_cannot_consume_previous_grid_events():
    clock=Clock();p=HintAwarePolicy(10,clock=clock,residual_table=table())
    store(p,[1],context('a',[1],gap=200))
    p.evict(1,set())
    assert ('r#0','a') not in p.chains
    clock.now=.2
    store(p,[1],context('a',[1],gap=200,turn=1))
    clock.now=1.
    p.mark_evictable(1)
    assert p.chains['r#0','a'].life_index==0
    clock.now=1.2
    p.mark_evictable(1)
    assert p.chains['r#0','a'].life_index==1
