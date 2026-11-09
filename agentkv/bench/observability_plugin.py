"""Opt-in vLLM plugin: one nominal prefill-source record per request."""

import functools
import json
import os
import time
import threading
from pathlib import Path

_installed = False


def _record(event, **fields):
    from agentkv.policy.light_metrics import get_store, instrumentation_mode
    if instrumentation_mode() == 'light':
        store = get_store()
        if store is not None:store.record(event, **fields)
        return
    if instrumentation_mode() == 'off':return
    directory = os.environ.get("AGENTKV_OBSERVATION_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"requests-{os.getpid()}.jsonl").open("a") as sink:
            sink.write(json.dumps(dict(event=event, ns=time.monotonic_ns(), **fields)) + "\n")


_scheduler_context=threading.local()


def install_scheduler_observation(Scheduler,KVCacheManager):
    """Observe failed admissions without treating every wait as KV shortage."""
    if getattr(Scheduler,'_agentkv_kv_wrapped',False):return
    Scheduler._agentkv_kv_wrapped=True
    original_init=Scheduler.__init__
    original_schedule=Scheduler.schedule
    original_allocate=KVCacheManager.allocate_slots
    from agentkv.policy.light_metrics import get_store
    metrics = get_store()

    @functools.wraps(original_init)
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        _record('kv_observer_ready',max_num_seqs=self.max_num_running_reqs)

    @functools.wraps(original_schedule)
    def schedule(self,*args,**kwargs):
        started = time.perf_counter_ns() if metrics is not None else 0
        previous=getattr(_scheduler_context,'scheduler',None)
        _scheduler_context.scheduler=self
        try:return original_schedule(self,*args,**kwargs)
        finally:
            if metrics is not None:
                ended = time.perf_counter_ns()
                metrics.observe('scheduler.schedule',ended-started,ended)
            _scheduler_context.scheduler=previous
            if os.environ.get('AGENTKV_POLICY_PROBE') == '1':
                from agentkv.policy.lru_probe import flush_logs
                flush_logs()
            if os.environ.get('AGENTKV_CALLBACK_TRACE_DIR'):
                from agentkv.policy.callback_trace import flush_traces
                flush_traces()

    @functools.wraps(original_allocate)
    def allocate(self,request,*args,**kwargs):
        result=original_allocate(self,request,*args,**kwargs)
        scheduler=getattr(_scheduler_context,'scheduler',None)
        if result is None and scheduler is not None:
            running,waiting=scheduler.get_request_counts()
            _record('kv_allocation_failure',request_id=request.request_id,
                    admission=request not in scheduler.running,running=running,waiting=waiting,
                    request_status=str(request.status))
        return result
    Scheduler.__init__=initialize
    Scheduler.schedule=schedule
    KVCacheManager.allocate_slots=allocate


def install_request_finish_observation(Scheduler):
    """Observe generation completion at the existing release hook; no mutation."""
    if getattr(Scheduler,'_agentkv_finish_wrapped',False):return
    Scheduler._agentkv_finish_wrapped=True
    original=Scheduler._free_request
    seen=set()
    @functools.wraps(original)
    def release(self,request,*args,**kwargs):
        if request.request_id not in seen:
            seen.add(request.request_id)
            _record('request_finished',request_id=request.request_id,status=str(request.status))
        return original(self,request,*args,**kwargs)
    Scheduler._free_request=release
    if hasattr(Scheduler,'shutdown'):
        shutdown=Scheduler.shutdown
        @functools.wraps(shutdown)
        def close(self,*args,**kwargs):
            try:return shutdown(self,*args,**kwargs)
            finally:
                from agentkv.policy.lru_probe import close_logs
                close_logs()
                from agentkv.policy.callback_trace import close_traces
                close_traces()
        Scheduler.shutdown=close


def install_engine_shutdown_observation(EngineCore):
    """Flush before model-executor teardown, and close after final callbacks."""
    if getattr(EngineCore,'_agentkv_shutdown_wrapped',False):return
    EngineCore._agentkv_shutdown_wrapped=True
    original=EngineCore.shutdown
    @functools.wraps(original)
    def shutdown(self,*args,**kwargs):
        from agentkv.policy.lru_probe import checkpoint_logs,close_logs
        from agentkv.policy.callback_trace import flush_traces,close_traces
        checkpoint_logs('engine_shutdown_enter')
        flush_traces('engine_shutdown_enter')
        try:return original(self,*args,**kwargs)
        finally:
            close_logs()
            close_traces()
            from agentkv.policy.light_metrics import close_metrics
            close_metrics()
    EngineCore.shutdown=shutdown


def install_chunk_ref_observation(Manager):
    """Observe manager writes outside CachePolicy, including 0 -> 1 pins."""
    if getattr(Manager,'_agentkv_refs_wrapped',False):return
    Manager._agentkv_refs_wrapped=True
    for name in ('_prepare_load','complete_load','complete_store'):
        original=getattr(Manager,name)
        def wrap(original=original,name=name):
            @functools.wraps(original)
            def invoke(self,keys,*args,**kwargs):
                trace=getattr(self._policy,'trace',None)
                if trace is None:return original(self,keys,*args,**kwargs)
                keys=list(keys)
                def state(key):
                    chunk=self._policy.chunks.get(key)
                    return None if chunk is None else [chunk.ref_cnt,chunk.chunk_id]
                before={key:state(key) for key in keys}
                result=original(self,keys,*args,**kwargs)
                changes=[[trace.key(key),before[key],state(key)] for key in keys if before[key]!=state(key)]
                trace.write(dict(event='manager_refs',ns=time.monotonic_ns(),method=name,
                    transitions=changes,refs=trace.delta(keys)))
                return result
            return invoke
        setattr(Manager,name,wrap())


def install():
    global _installed
    if _installed or not os.environ.get("AGENTKV_OBSERVATION_DIR"):
        return
    _installed = True
    from vllm.v1.request import Request
    from vllm.v1.metrics.stats import PrefillStats
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if os.environ.get('AGENTKV_EFFECTIVE_PREFILL') == '1':
        from agentkv.bench.effective_observer import install_effective_prefill
        install_effective_prefill(Scheduler)
    if os.environ.get('AGENTKV_BUILTIN_LRU_TIMING') == '1':
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
        from agentkv.bench.effective_observer import install_builtin_lru_timing
        install_builtin_lru_timing(LRUCachePolicy)
    install_scheduler_observation(Scheduler,KVCacheManager)
    install_request_finish_observation(Scheduler)
    from vllm.v1.engine.core import EngineCore
    install_engine_shutdown_observation(EngineCore)
    if os.environ.get('AGENTKV_CALLBACK_TRACE_DIR'):
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
        install_chunk_ref_observation(CPUOffloadingManager)
    if os.environ.get('AGENTKV_CONTEXT_TIMING') == '1':
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as offloading
        original_context=offloading._create_req_context
        @functools.wraps(original_context)
        def create_context(request,*args,**kwargs):
            start=time.perf_counter_ns()
            context=original_context(request,*args,**kwargs)
            elapsed=time.perf_counter_ns()-start
            _record('create_req_context',request_id=request.request_id,duration_us=elapsed/1e3,
                    prompt_tokens=context.num_prompt_tokens,prefix_keys=len(context.get_offload_prefix_keys()))
            return context
        offloading._create_req_context=create_context

    old_init = Request.__init__
    @functools.wraps(old_init)
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        _record("request_received",request_id=self.request_id)
        envelope = (self.kv_transfer_params or {}).get("kv_hints")
        if envelope is not None:
            from agentkv.hints import validate_envelope
            validate_envelope(envelope)
            _record("hint_received", request_id=self.request_id,
                    message_id=envelope["message_id"])
    Request.__init__ = init

    old_event = Request.record_event
    @functools.wraps(old_event)
    def record_event(self, event_type, *args, **kwargs):
        result = old_event(self,event_type,*args,**kwargs)
        if event_type.name == 'SCHEDULED':
            _record('request_scheduled',request_id=self.request_id)
        return result
    Request.record_event = record_event

    old_take = Request.take_prefill_stats
    @functools.wraps(old_take)
    def take(self):
        stats = old_take(self)
        if stats is not None:
            stats._agentkv_request_id = self.request_id
        return stats
    Request.take_prefill_stats = take

    old_finalize = PrefillStats.finalize
    @functools.wraps(old_finalize)
    def finalize(self, *args, **kwargs):
        old_finalize(self, *args, **kwargs)
        idx = getattr(self, "_agentkv_request_id", None)
        if idx is not None:
            _record("prefill", request_id=idx, prompt_tokens=self.num_prompt_tokens,
                    gpu_hit=self.num_local_cached_tokens,
                    cpu_hit=self.num_external_cached_tokens,
                    recomputed=self.num_computed_tokens,
                    cache_creation=self.num_cache_creation_tokens)
    PrefillStats.finalize = finalize
    _record("plugin_installed", pid=os.getpid())
