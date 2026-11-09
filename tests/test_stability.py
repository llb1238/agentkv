import pytest
from agentkv.analysis.stability import evaluate


def records(first=100,second=100):
    return [dict(status='success',is_entry=False,send_ns=int(t*1e9),ttft_ms=v)
            for t,v in [(610,first),(1200,first),(1600,second),(2300,second)]]


def metrics(values):
    return [dict(elapsed_s=600+i*10,values={'vllm:num_requests_waiting':v,'vllm:num_requests_running':1})
            for i,v in enumerate(values)]


def check(values,rows=None,events=(),observed=True):
    return evaluate(metrics(values),records() if rows is None else rows,origin_ns=0,
                    duration_s=2400,warmup_s=600,max_num_seqs=8,kv_events=events,kv_observed=observed)


def test_idle_queue_low_running_is_stable_without_kv_blocking():
    assert check([0]*180)['stable']


def test_queue_zero_fraction_and_p99_bound():
    assert not check([0]*140+[1]*40)['checks']['queue_distribution']
    assert not check([0]*177+[9]*3)['checks']['queue_distribution']
    assert check([0]*177+[8]*3)['checks']['queue_distribution']


def test_small_near_zero_mean_ratio_is_no_longer_a_stability_gate():
    value=check([0]*89+[1]+[0]*88+[1,1])
    assert value['checks']['queue_distribution']
    assert value['checks']['fitted_growth_30min']
    assert value['stable']


def test_positive_fitted_growth_over_one_is_unstable():
    value=check([0]*150+list(range(30)))
    assert not value['checks']['fitted_growth_30min']


def test_ttft_half_ratio_and_missing_data():
    assert check([0]*180,records(second=150))['stable']
    assert not check([0]*180,records(second=151))['checks']['ttft_half_ratio']
    assert check([0]*180,[])['status']=='indeterminate'
    assert check([0]*180,observed=False)['status']=='indeterminate'


def test_kv_blocking_only_checks_observed_failed_admissions():
    event=dict(event='kv_allocation_failure',admission=True,waiting=1,running=6,ns=700_000_000_000)
    result=check([0]*180,events=[event])
    assert not result['checks']['conditional_kv_sizing']
    assert check([0]*180,events=[dict(event,running=7)])['stable']
    assert check([0]*180,events=[dict(event,admission=False)])['stable']


def test_scheduler_observer_distinguishes_kv_failure_from_slot_queue(monkeypatch):
    from agentkv.bench import observability_plugin as plugin
    events=[];monkeypatch.setattr(plugin,'_record',lambda event,**kw:events.append(dict(event=event,**kw)))
    class Manager:
        def allocate_slots(self,request):return None
    class Scheduler:
        def __init__(self):self.max_num_running_reqs=8;self.running=[];self.manager=Manager()
        def get_request_counts(self):return len(self.running),1
        def schedule(self,request):return self.manager.allocate_slots(request)
    class Request:request_id='r';status='WAITING'
    plugin.install_scheduler_observation(Scheduler,Manager)
    scheduler=Scheduler();scheduler.schedule(Request())
    assert events[0]['event']=='kv_observer_ready'
    assert events[1]['admission'] and events[1]['waiting']==1 and events[1]['running']==0
