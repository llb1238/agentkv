"""Actual vLLM LRU equivalence, using ctypes chunks and no model/GPU runtime."""
import importlib.util
import json
import random
import struct
from types import SimpleNamespace
import pytest


@pytest.fixture
def classes():
    if importlib.util.find_spec('vllm') is None:pytest.skip('Requires installed vLLM source environment, no GPU needed')
    from vllm.v1.kv_offload.base import ReqContext,make_offload_key
    from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    from agentkv.policy.lru_probe import LRUProbePolicy
    return ReqContext,make_offload_key,ChunkStatus,LRUCachePolicy,LRUProbePolicy


def test_probe_exact_lru_state_and_eviction_equivalence(tmp_path,monkeypatch,classes):
    ReqContext,key,Chunk,LRU,Probe=classes;monkeypatch.setenv('AGENTKV_POLICY_OBSERVATION_DIR',str(tmp_path))
    a=LRU(30);b=Probe(30);rng=random.Random(7781)
    keys=[key(i.to_bytes(8,'big'),0) for i in range(20)];ctx=ReqContext('r')
    for i,k in enumerate(keys):ctx.set_offload_key_position(k,16*(i+1))
    for step in range(250):
        selected=rng.sample(keys,rng.randint(1,7))
        a.on_store_miss(selected,ctx);b.on_store_miss(iter(selected),ctx)
        for k in selected:
            if a.get(k) is None:
                c1=Chunk(step);c2=Chunk(step);c1.ref_cnt=c2.ref_cnt=0;a.insert(k,c1);b.insert(k,c2)
        # Scrambled groups exercise native position-based global sorting.
        groups=(selected[::2],selected[1::2]);inserted=set(selected[::2]);reused=set(selected[1::2])
        a.on_request_finished(groups,inserted,reused,ctx);b.on_request_finished(groups,inserted,reused,ctx)
        assert a._ranks==b._ranks and a._evictable==b._evictable
        if step%7==0 and a.chunks:
            k=next(iter(a.chunks));a.get(k).ref_cnt=b.get(k).ref_cnt=1
            a.mark_non_evictable(k);b.mark_non_evictable(k)
            a.get(k).ref_cnt=b.get(k).ref_cnt=0;a.mark_evictable(k);b.mark_evictable(k)
        protected=set(rng.sample(list(a.chunks),rng.randrange(len(a.chunks)+1)));n=rng.randrange(6)
        av=a.evict(n,protected);bv=b.evict(n,protected)
        assert (None if av is None else [k for k,_ in av])==(None if bv is None else [k for k,_ in bv])
        assert a._ranks==b._ranks and set(a.chunks)==set(b.chunks)
        if step%13==0 and a.chunks:
            k=next(iter(a.chunks));a.remove(k);b.remove(k)
    a.clear();b.clear();assert not a.chunks and not b.chunks
    b.probe.close()
    from agentkv.policy.lru_probe import SCALAR,METHODS
    raw=next(tmp_path.glob('*.bin')).read_bytes();assert len(raw)%SCALAR.size==0
    decoded=list(SCALAR.iter_unpack(raw));assert all(0<=method<len(METHODS) for _,method in decoded)
    assert all(x[0]<=y[0] for x,y in zip(decoded,decoded[1:]))
    records=[json.loads(line) for line in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    assert records[-1]['event']=='probe_close' and sum(records[-1]['counts'].values())==len(decoded)


def test_hint_routes_and_raw_key_order_are_observed_without_sorting(tmp_path,monkeypatch,classes):
    ReqContext,key,Chunk,LRU,Probe=classes;monkeypatch.setenv('AGENTKV_POLICY_OBSERVATION_DIR',str(tmp_path))
    from agentkv.hints import make_envelope
    from agentkv.policy.lru_probe import inspect_hints
    row=dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=2000.)
    envelope=make_envelope(row,'perfect','message-1','r#0')
    ctx=ReqContext('r',kv_transfer_params={'kv_hints':envelope})
    assert inspect_hints(ctx)['selected_source']=='nested' and inspect_hints(ctx)['valid']
    direct=SimpleNamespace(model_dump=lambda **kwargs:envelope)
    ctx.kv_hints=direct;assert inspect_hints(ctx)['selected_source']=='direct'
    keys=[key(i.to_bytes(8,'big'),0) for i in range(3)]
    for i,k in enumerate(keys):ctx.set_offload_key_position(k,16*(i+1))
    p=Probe(10)
    for i,k in enumerate(keys):
        c=Chunk(i);c.ref_cnt=-1;p.insert(k,c)
    p.on_request_finished((list(reversed(keys)),),set(keys),set(),ctx);p.probe.close()
    records=[json.loads(line) for line in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    observed=next(r for r in records if r['event']=='on_request_finished')
    assert observed['position_groups']==[[48,32,16]] and observed['ref_count_groups']==[[-1,-1,-1]]


def test_generation_end_observer_forwards_once(monkeypatch):
    from agentkv.bench import observability_plugin as observer
    events=[];monkeypatch.setattr(observer,'_record',lambda event,**fields:events.append((event,fields)))
    class Scheduler:
        def _free_request(self,request,*args,**kwargs):return ('original',request.request_id,args,kwargs)
    observer.install_request_finish_observation(Scheduler)
    req=SimpleNamespace(request_id='r',status='FINISHED')
    scheduler=Scheduler();assert scheduler._free_request(req,True)==('original','r',(True,),{})
    scheduler._free_request(req)
    assert len(events)==1 and events[0][0]=='request_finished'


def test_shutdown_flushes_before_teardown_and_closes_after(monkeypatch,classes):
    from agentkv.bench import observability_plugin as observer
    from agentkv.policy import lru_probe
    calls=[]
    monkeypatch.setattr(lru_probe,'checkpoint_logs',lambda reason:calls.append(reason))
    monkeypatch.setattr(lru_probe,'close_logs',lambda:calls.append('close'))
    class Core:
        def shutdown(self):
            calls.append('teardown')
            raise RuntimeError('teardown failure')
    observer.install_engine_shutdown_observation(Core)
    observer.install_engine_shutdown_observation(Core)
    with pytest.raises(RuntimeError,match='teardown failure'):Core().shutdown()
    assert calls==['engine_shutdown_enter','teardown','close']


def test_policy_destructor_flushes_close_footer(tmp_path,monkeypatch,classes):
    *_,Probe=classes
    monkeypatch.setenv('AGENTKV_POLICY_OBSERVATION_DIR',str(tmp_path))
    policy=Probe(1);policy.__del__();policy.__del__()
    records=[json.loads(line) for line in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    assert [r['event'] for r in records]==['probe_open','probe_close']
