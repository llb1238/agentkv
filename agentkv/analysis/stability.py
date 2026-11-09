"""Preregistered E0 stability checks (design section 7.6)."""
import numpy as np

VERSION='e0-stability-v2'


def evaluate(metrics, requests, *, origin_ns, duration_s, warmup_s, max_num_seqs,
             cooldown_s=0., kv_events=(), kv_observed=False, kv_summary=None):
    end=duration_s-cooldown_s
    samples=[s for s in metrics if warmup_s<=s['elapsed_s']<=end and
             'vllm:num_requests_waiting' in s.get('values',{}) and
             'vllm:num_requests_running' in s.get('values',{})]
    checks={}; details={}
    if samples:
        times=np.array([s['elapsed_s'] for s in samples],dtype=float)
        waiting=np.array([s['values']['vllm:num_requests_waiting'] for s in samples],dtype=float)
        zero=float(np.mean(waiting==0));p99=float(np.quantile(waiting,.99))
        checks['queue_distribution']=zero>=.80 and p99<=max_num_seqs
        details['queue']=dict(samples=len(samples),zero_fraction=zero,p99=p99,maximum=float(waiting.max()))
        if len(samples)>=2 and np.ptp(times)>0:
            centered=times-times.mean();slope=float(np.dot(centered,waiting-waiting.mean())/np.dot(centered,centered))
            increase=slope*1800
            checks['fitted_growth_30min']=increase<1
            details['regression']=dict(slope_per_second=slope,fitted_increase_30min=increase)
        else:checks['fitted_growth_30min']=None
    else:
        checks.update(queue_distribution=None,fitted_growth_30min=None)
        details['queue']=dict(samples=0)
    midpoint=(warmup_s+end)/2
    halves=[[],[]]
    for row in requests:
        if row.get('status')!='success' or row.get('is_entry') or row.get('ttft_ms') is None:continue
        time_s=(row['send_ns']-origin_ns)/1e9
        if warmup_s<=time_s<=end:
            halves[int(time_s>=midpoint)].append(row['ttft_ms'])
    if all(halves) and np.median(halves[0])>0:
        early=float(np.median(halves[0]));late=float(np.median(halves[1]));ratio=late/early
        checks['ttft_half_ratio']=ratio<=1.5
        details['ttft']=dict(first_half_requests=len(halves[0]),second_half_requests=len(halves[1]),
                             first_half_median_ms=early,second_half_median_ms=late,ratio=ratio)
    else:
        checks['ttft_half_ratio']=None
        details['ttft']=dict(first_half_requests=len(halves[0]),second_half_requests=len(halves[1]))
    blocked=[e for e in kv_events if e.get('event')=='kv_allocation_failure' and e.get('admission') and
             e['waiting']>0 and warmup_s<=(e['ns']-origin_ns)/1e9<=end]
    threshold=max(0,max_num_seqs-1)
    if kv_observed:
        checks['conditional_kv_sizing']=all(e['running']>=threshold for e in blocked)
    else:checks['conditional_kv_sizing']=None
    details['kv']=dict(observed=kv_observed,blocking_events=len(blocked),running_threshold=threshold,
                       below_threshold_events=sum(e['running']<threshold for e in blocked),
                       minimum_running_when_blocked=min((e['running'] for e in blocked),default=None))
    if kv_summary is not None:
        expected=[origin_ns+round(warmup_s*1e9),origin_ns+round(end*1e9)]
        if list(kv_summary['window_ns'])!=expected:raise ValueError('KV summary window differs')
        counts={int(k):v for k,v in kv_summary['running_counts'].items()}
        kv_observed=kv_summary['observed']
        below=sum(v for k,v in counts.items() if k<threshold)
        checks['conditional_kv_sizing']=below==0 if kv_observed else None
        details['kv']=dict(observed=kv_observed,blocking_events=sum(counts.values()),running_threshold=threshold,
            below_threshold_events=below,minimum_running_when_blocked=min(counts,default=None))
    stable=all(v is True for v in checks.values())
    status='stable' if stable else 'unstable' if any(v is False for v in checks.values()) else 'indeterminate'
    return dict(version=VERSION,status=status,stable=stable,checks=checks,details=details)
