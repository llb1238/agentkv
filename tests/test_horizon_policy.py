import copy
import random
from types import SimpleNamespace as NS
import pytest
from agentkv.policy.horizon_aware import HorizonPolicy
from agentkv.policy.horizon import HorizonProbabilityTable
from tests.test_hint_aware import Clock,context,store


def table():
    return HorizonProbabilityTable(dict(format='agentkv.horizon_probability.v1',elapsed_grid_s=[0.,1.,2.,3.,5.],
        tiers={'online':dict(prediction_cuts_s=[5.],buckets=[dict(support_s=[1.,3.,5.],cumulative_counts=[1,2,3]),dict(support_s=[4.,5.],cumulative_counts=[1,2])])}))


def test_probability_lowest_first_and_final_and_shared_live_references():
    clock=Clock();p=HorizonPolicy(10,clock=clock,probability_table=table())
    store(p,[1],context('soon',[1],gap=1000))
    store(p,[2],context('later',[2],gap=6000))
    store(p,[1,3],context('ended',[1,3],final=True))
    assert [k for k,_ in p.evict(2,set())]==[2,3]  # P=0; shared 1 has positive probability.


def test_perfect_needs_no_calibration_and_crosses_deadline_minus_horizon():
    for now,expected in [(0.,1),(1.,2)]:
        clock=Clock();p=HorizonPolicy(10,clock=clock,perfect_probability=True)
        store(p,[1],context('a',[1],gap=2000))
        store(p,[2],context('b',[2],gap=5000))
        clock.now=now
        assert p.evict(1,set())[0][0]==expected
        assert p.probability_table is None


def test_active_owner_protects_shared_block_and_failure_is_atomic():
    clock=Clock();p=HorizonPolicy(10,clock=clock,probability_table=table())
    store(p,[1,2],context('a',[1,2],final=True))
    p.on_store_miss([1],context('active',[1],turn=0))
    before=(dict(p.chunks),dict(p.ranks),copy.deepcopy(p.owners))
    assert p.evict(2,set()) is None
    assert (p.chunks,p.ranks,p.owners)==before and p.stats['horizon_samples_seen']==0
    assert [k for k,_ in p.evict(1,set())]==[2]
    assert p.evict(1,{1}) is None
    p.on_request_finished(([1],),set(),set(),context('active',[1],final=True))
    assert p.evict(1,set())[0][0]==1


def test_sliding_residence_mean_updates_at_tick_only_and_uses_last_touch():
    clock=Clock();p=HorizonPolicy(10,clock=clock,probability_table=table(),horizon_window=2)
    for i in (1,2,3):store(p,[i],context(str(i),[i],final=True))
    clock.now=2.;p.touch([1],context('1',[1],final=True))  # last access 2, active now
    p.on_request_finished(([1],),set(),set(),context('1',[1],final=True))
    assert p.evict(2,set()) is not None
    assert list(p.residences)==[2.,2.] and p.horizon_s==1.
    clock.now=3.;p.mark_evictable(1)
    assert p.horizon_s==2.
    p.evict(1,set())
    assert list(p.residences)==[2.,1.]
    clock.now=4.;p._advance()
    assert p.horizon_s==1.5


def test_elapsed_knots_and_recreated_chain_do_not_use_stale_events():
    clock=Clock();clock.now=2101.23456789
    p=HorizonPolicy(10,clock=clock,probability_table=table(),horizon_period_s=1000.)
    store(p,[1],context('a',[1],gap=1000))
    p.evict(1,set());clock.now+=.2
    store(p,[1],context('a',[1],gap=1000,turn=1))
    clock.now=2102.23456789;p.mark_evictable(1)
    assert p.chains['r#0','a'].return_probability==1/3
    clock.now=2102.43456789;p.mark_evictable(1)
    assert p.chains['r#0','a'].return_probability==0.


def test_unknown_hints_idle_order_is_lru_and_pins_are_skipped():
    clock=Clock();p=HorizonPolicy(10,clock=clock,probability_table=table())
    store(p,[1,2],context('a',[1,2],hint=False))
    clock.now=.3;store(p,[3],context('b',[3],hint=False))
    p.get(2).ref_cnt=-1;p.mark_non_evictable(2)
    assert p.evict(1,set())[0][0]==1
    p.get(2).ref_cnt=1;p.mark_non_evictable(2)
    assert p.evict(2,set()) is None
    p.get(2).ref_cnt=0;p.mark_evictable(2)
    assert [k for k,_ in p.evict(2,set())]==[2,3]


@pytest.mark.parametrize('tier',['perfect','noisy','online'])
def test_simulator_horizon_keeps_gpu_schedule_and_token_sums(tier):
    from agentkv.sim.two_tier import simulate
    from tests.test_hint_adapter import fixture
    _,base=simulate(*fixture(),8,10,2,duration=20,warmup=0,block_size=1)
    summary,rows=simulate(*fixture(),8,10,2,duration=20,warmup=0,block_size=1,cpu_policy='hint',hint_tier=tier,hint_options=dict(scoring='horizon'))
    assert [(r['trace_req_idx'],r['send_s'],r['end_s'],r['gpu_hit']) for r in rows]==[(r['trace_req_idx'],r['send_s'],r['end_s'],r['gpu_hit']) for r in base]
    assert all(r['gpu_hit']+r['cpu_hit']+r['recomputed']==r['prompt_tokens'] for r in rows)


def test_horizon_import_needs_only_standard_library():
    import subprocess,sys
    subprocess.run([sys.executable,'-S','-c','import sys;from agentkv.policy.horizon_aware import HorizonPolicy;assert not any(m in sys.modules for m in ("numpy","vllm","torch"))'],check=True)


def test_perfect_event_is_invalidated_when_horizon_changes():
    clock=Clock();p=HorizonPolicy(10,clock=clock,perfect_probability=True)
    store(p,[0],context('ended',[0],final=True))
    store(p,[1],context('a',[1],gap=5000))
    store(p,[2],context('b',[2],gap=10000))
    clock.now=3.;assert p.evict(1,set())[0][0]==0
    clock.now=4.;p.mark_evictable(1)
    assert p.horizon_s==3. and p.chains['r#0','a'].return_probability==1.
    assert p.evict(1,set())[0][0]==2
    p.clear();assert not p.chunks and not p.residences and p.horizon_s==1.


def test_lazy_probabilities_match_direct_shared_block_reference():
    clock=Clock();p=HorizonPolicy(30,clock=clock,probability_table=table());rng=random.Random(8924)
    support=([1.,3.,5.],[4.,5.]);grid=(0.,1.,2.,3.,5.);active_ctx=None;active_keys=None
    for turn in range(240):
        clock.now+=rng.random()*2.5
        if active_ctx is not None:
            p.on_request_finished((active_keys,),set(),set(),active_ctx);active_ctx=None
        keys=list(range(1,rng.randint(2,9)));chain=str(rng.randrange(5))
        ctx=context(chain,keys,gap=rng.choice([1000.,6000.]),turn=turn,final=rng.random()<.2)
        store(p,keys,ctx)
        if rng.random()<.2:
            active_ctx=context('active',keys,gap=1000.,turn=turn);active_keys=keys
            p.on_store_miss(keys,active_ctx)
        protected=set(rng.sample(list(p.chunks),rng.randrange(len(p.chunks)+1)))
        expected=[]
        for key in p.chunks:
            if key in protected or any(not p.chains[i].finished for i in p.owners[key]):continue
            probabilities=[]
            for identity in p.owners[key]:
                c=p.chains[identity]
                if c.mode=='final':probability=0.
                else:
                    elapsed=max(g for g in grid if c.finished_at+g<=clock.now)
                    live=[g for g in support[c.probability_bucket] if g>elapsed]
                    probability=sum(g<=elapsed+p.horizon_s for g in live)/len(live) if live else 0.
                probabilities.append(probability)
            expected.append((max(probabilities),p.ranks[key],key))
        actual=p.evict(1,protected)
        assert (None if actual is None else actual[0][0])==(min(expected)[2] if expected else None)
