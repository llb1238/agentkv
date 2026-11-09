from types import SimpleNamespace as NS
import json
import pytest
from agentkv.policy.light_metrics import LightMetrics,read_light_events
from agentkv.analysis.stability import evaluate
from agentkv.bench.effective_observer import install_effective_prefill,install_builtin_lru_timing


def test_forward_chunks_replay_and_preemption_work_are_distinct(tmp_path):
    s=LightMetrics(tmp_path,pid=1)
    s.record('prefill',ns=1,request_id='r',prompt_tokens=32,gpu_hit=16,cpu_hit=16,recomputed=0)
    s.record_forward('r',31,33,32)  # terminal prompt token, decode excluded
    s.record_forward('r',0,16,32)   # preemption replay
    s.record_forward('r',16,32,32)
    s.close()
    e=next(e for e in read_light_events(tmp_path) if e['event']=='effective_prefill')
    assert e['recomputed_effective']==32 and e['prefill_forward_token_work']==33
    assert e['repeated_prompt_work']==1 and e['replay_overlap']==32
    assert e['gpu_hit_effective']==e['cpu_hit_effective']==0


def test_actual_output_hook_handles_async_ahead_and_unexecuted_batches(tmp_path,monkeypatch):
    from agentkv.bench import effective_observer as module
    store=LightMetrics(tmp_path,pid=1);monkeypatch.setattr(module,'get_store',lambda:store)
    class Scheduler:
        def __init__(self):self.requests={'r':NS(num_computed_tokens=16,num_prompt_tokens=32)}
        def _update_after_schedule(self,out):
            for k,n in out.num_scheduled_tokens.items():self.requests[k].num_computed_tokens+=n
        def update_from_output(self,out,result):return 'unchanged'
    install_effective_prefill(Scheduler);s=Scheduler()
    a=NS(num_scheduled_tokens={'r':8});b=NS(num_scheduled_tokens={'r':8})
    s._update_after_schedule(a);s._update_after_schedule(b)
    assert s.requests['r'].num_computed_tokens==32 and not store.requests
    result=NS(req_id_to_index={'r':0},kv_connector_output=None)
    assert s.update_from_output(a,result)=='unchanged'
    assert s.update_from_output(b,result)=='unchanged'
    assert store.requests['r']['forward_ranges']==[[16,32]]
    assert store.requests['r']['prefill_forward_token_work']==16
    c=NS(num_scheduled_tokens={'r':5});s._update_after_schedule(c)
    s.update_from_output(c,result)  # decode only
    assert store.requests['r']['prefill_forward_token_work']==16


def test_light_kv_summary_exactly_matches_event_criteria(tmp_path):
    store=LightMetrics(tmp_path,pid=1);store.set_window(0,600,2400)
    store.record('kv_observer_ready',max_num_seqs=8)
    events=[dict(ns=t*10**9,admission=a,waiting=w,running=r,event='kv_allocation_failure')
        for t,a,w,r in [(599,True,1,0),(600,True,1,7),(700,True,1,6),(700,True,1,8),
                        (701,False,1,0),(702,True,0,0),(2400,True,1,7),(2401,True,1,0)]]
    for e in events:store.record(**e)
    store.close();summary=json.loads((tmp_path/'light-1.json').read_text())['kv_blocking_window']
    metrics=[dict(elapsed_s=t,values={'vllm:num_requests_waiting':0,'vllm:num_requests_running':1}) for t in range(600,2401,10)]
    req=[dict(send_ns=t*10**9,status='success',is_entry=False,ttft_ms=100) for t in (601,2399)]
    args=dict(origin_ns=0,duration_s=2400,warmup_s=600,max_num_seqs=8)
    raw=evaluate(metrics,req,kv_events=events,kv_observed=True,**args)
    light=evaluate(metrics,req,kv_summary=summary,**args)
    assert raw==light and not light['stable']
    with pytest.raises(ValueError):evaluate(metrics,req,kv_summary=dict(summary,window_ns=[0,1]),**args)


def test_final_only_does_not_use_or_mutate_interval_predictions():
    pytest.importorskip('vllm.v1.kv_offload.base', reason='Requires optional patched vLLM, CPU only')
    from vllm.v1.kv_offload.base import ReqContext
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    from agentkv.hints import make_envelope
    def run(gap):
        p=OptimizedPrefixAdapter(8,clock=lambda:1.,hint_tier='online',final_only=True,
            options=dict(scoring='lru_final',tail_first=False,final_first=True,timeout_decay=False))
        for i in range(3):
            env=make_envelope(dict(chain_id=str(i),chain_turn=0,final=i==1,hint_online_ms=None if i==1 else gap,
                hint_online_p90_ms=None if i==1 else gap*2),'online',str(i),'r#0')
            original=json.dumps(env,sort_keys=True)
            ctx=ReqContext(str(i),kv_hints=env,num_prompt_tokens=16);ctx.set_offload_key_position(i,16)
            p.on_store_miss([i],ctx);p.insert(i,NS(ref_cnt=0,chunk_id=i));p.on_request_finished(([i],),{i},set(),ctx)
            assert json.dumps(env,sort_keys=True)==original
        return [k for k,_ in p.evict(3,set())]
    assert run(1)==run(1000000)==[1,0,2]


def test_builtin_lru_timing_preserves_evictions_and_suppresses_nested_touch(tmp_path,monkeypatch):
    pytest.importorskip('vllm.v1.kv_offload.cpu.policies.lru', reason='Requires optional vLLM, CPU only')
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    from vllm.v1.kv_offload.base import ReqContext
    from agentkv.bench import effective_observer as module
    class ObservedLRU(LRUCachePolicy):pass
    store=LightMetrics(tmp_path,pid=1);monkeypatch.setattr(module,'get_store',lambda:store)
    install_builtin_lru_timing(ObservedLRU)
    a=LRUCachePolicy(8);b=ObservedLRU(8);ctx=ReqContext('r',num_prompt_tokens=48)
    for k in range(3):ctx.set_offload_key_position(k,16*(k+1))
    for p in (a,b):
        p.on_store_miss([0,1,2],ctx)
        for k in range(3):p.insert(k,NS(ref_cnt=0,chunk_id=k))
        p.on_request_finished(([2,0],[1]),set(range(3)),set(),ctx)
    assert 'policy.touch' not in store.histograms
    assert [k for k,_ in a.evict(2,{1})]==[k for k,_ in b.evict(2,{1})]
    assert store.histograms['policy.evict'].count==1


def test_pinned_backend_full_cpu_hit_rewinds_terminal_prompt_token():
    pytest.importorskip('vllm.v1.core.sched.scheduler', reason='Requires optional pinned vLLM, CPU only')
    from vllm.v1.core.sched.scheduler import Scheduler
    calls=[]
    request=NS(request_id='r',num_computed_tokens=32,num_tokens=32)
    scheduler=NS(connector=object(),failed_recving_kv_req_ids=set(),finished_recving_kv_req_ids={'r'},
        kv_cache_manager=NS(cache_blocks=lambda *a:calls.append(a)),_mark_prefix_replay=lambda *a,**kw:0,
        _load_restores_replay_window=lambda *a:False)
    Scheduler._update_waiting_for_remote_kv(scheduler,request)
    assert request.num_computed_tokens==31 and calls and not scheduler.finished_recving_kv_req_ids


def test_post_preemption_new_cache_hits_do_not_hide_measured_work(tmp_path):
    s=LightMetrics(tmp_path,pid=1)
    s.record('prefill',ns=1,request_id='r',prompt_tokens=32,gpu_hit=0,cpu_hit=0,recomputed=32)
    s.record_forward('r',0,16,32)
    # A later admission loads positions16..30; the terminal token forwards.
    s.record_forward('r',31,32,32,preemptions=1)
    s.close()
    e=next(e for e in read_light_events(tmp_path) if e['event']=='effective_prefill')
    assert e['prefill_forward_token_work']==17 and e['recomputed_effective']==17
    assert e['effective_unattributed_cache_tokens']==15
    assert e['effective_ranges_complete']
    assert e['gpu_hit_effective'] is None and e['cpu_hit_effective'] is None
