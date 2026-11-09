"""Driver-only observation of executed prompt ranges; no engine source patch."""
import functools
import time
from agentkv.policy.light_metrics import get_store


def install_effective_prefill(Scheduler):
    if getattr(Scheduler,'_agentkv_effective_wrapped',False):return
    Scheduler._agentkv_effective_wrapped=True
    advance=Scheduler._update_after_schedule
    update=Scheduler.update_from_output

    @functools.wraps(advance)
    def capture(self,output):
        spans=[]
        for identity,n in output.num_scheduled_tokens.items():
            request=self.requests[identity]
            start=request.num_computed_tokens
            if n>0 and start<request.num_prompt_tokens:
                spans.append((identity,start,start+n,request.num_prompt_tokens,getattr(request,'num_preemptions',0)))
        if not hasattr(self,'_agentkv_prompt_pending'):self._agentkv_prompt_pending={}
        self._agentkv_prompt_pending[id(output)]=(output,spans)
        return advance(self,output)

    @functools.wraps(update)
    def completed(self,output,result,*args,**kwargs):
        held,spans=self._agentkv_prompt_pending.pop(id(output))
        assert held is output
        store=get_store()
        if store is not None:
            connector=getattr(result,'kv_connector_output',None)
            if connector and (getattr(connector,'invalid_block_ids',None) or getattr(connector,'failed_recving',None)):
                store.counters['effective_invalid_load_batches']+=1
            for identity,start,end,prompt,preemptions in spans:
                if identity not in result.req_id_to_index:
                    request=self.requests.get(identity)
                    aborted=request is None or (hasattr(request,'is_finished') and request.is_finished())
                    store.counters['effective_aborted_model_rows' if aborted else 'effective_missing_model_rows']+=1
                    continue
                store.record_forward(identity,start,end,prompt,preemptions)
            store.counters['effective_completed_batches']+=1
        return update(self,output,result,*args,**kwargs)

    Scheduler._update_after_schedule=capture
    Scheduler.update_from_output=completed
    store=get_store()
    if store is not None:store.counters['effective_observer_ready']+=1


def install_builtin_lru_timing(Policy):
    """Keep the registered builtin class and its decisions, time outer calls."""
    if getattr(Policy,'_agentkv_light_wrapped',False):return
    Policy._agentkv_light_wrapped=True
    original_init=Policy.__init__
    @functools.wraps(original_init)
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self._agentkv_light_metrics=get_store()
        self._agentkv_native_miss=False
    Policy.__init__=initialize
    for name in ('get','insert','remove','clear','mark_evictable','mark_non_evictable',
                 'touch','on_store_miss','on_request_finished','evict'):
        original=getattr(Policy,name)
        def wrap(original=original,name=name):
            @functools.wraps(original)
            def invoke(self,*args,**kwargs):
                metrics=getattr(self,'_agentkv_light_metrics',None)
                if metrics is None or (name=='touch' and self._agentkv_native_miss):
                    return original(self,*args,**kwargs)
                if name=='on_store_miss':self._agentkv_native_miss=True
                start=time.perf_counter_ns()
                try:return original(self,*args,**kwargs)
                finally:
                    end=time.perf_counter_ns()
                    if name=='on_store_miss':self._agentkv_native_miss=False
                    metrics.observe('policy.'+name,end-start,end)
            return invoke
        setattr(Policy,name,wrap())
