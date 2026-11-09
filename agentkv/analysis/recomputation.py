"""Token-weighted recomputation by historical and triggering prefix source.

All three categories use full blocks. The prompt tail is reported separately;
this makes the rounded partition and the original nominal partition auditable.
"""
from collections import defaultdict
import numpy as np

CATEGORIES=('unavoidable','gap_eviction','other_eviction')
VERSION='e0-prefix-source-v1'


def decompose(prompt_len, hit, history_prefix, trigger_prefix, block_size=16):
    if block_size<1 or min(prompt_len,hit,history_prefix,trigger_prefix)<0:
        raise ValueError('Invalid prefix length')
    length=prompt_len//block_size*block_size
    hit=hit//block_size*block_size
    u=history_prefix//block_size*block_size
    p=trigger_prefix//block_size*block_size
    if max(hit,u,p)>length or p>u:
        raise ValueError('Prefix exceeds prompt or trigger is not in sent history')
    anomaly=hit>u
    u=max(u,hit)
    result=dict(unavoidable=length-u,gap_eviction=max(0,p-hit),
                other_eviction=max(0,u-max(p,hit)),hit=hit,history_prefix=u,
                trigger_prefix=p,prompt_blocks_tokens=length,tail_tokens=prompt_len-length,
                hit_exceeds_history=anomaly)
    assert sum(result[k] for k in CATEGORIES)==length-hit
    return result


def common_blocks(left,right):
    n=min(len(left),len(right))
    different=np.flatnonzero(left[:n]!=right[:n])
    return int(different[0]) if len(different) else n


def classify_records(records, roots, block_ids, *, real=False, block_size=16):
    """Use actual send order, including entries in history, excluding them in sums.

    Canonical IDs are complete prefix identities, so membership of a prefix's
    deepest block in the history proves the whole prefix was previously sent.
    Instance-local sets exactly implement per-instance token permutations.
    Simultaneous sends are grouped: neither is strictly earlier than the other.
    """
    send_key='send_ns' if real else 'send_s'
    eligible=[r for r in records if r.get(send_key) is not None]
    eligible.sort(key=lambda r:(r[send_key],r['instance_id'],r['trace_req_idx']))
    history=defaultdict(set);sent=defaultdict(set);out=[]
    index=0
    while index<len(eligible):
        stop=index+1
        while stop<len(eligible) and eligible[stop][send_key]==eligible[index][send_key]:stop+=1
        group=eligible[index:stop]
        for row in group:
            instance=row['instance_id'];idx=row['trace_req_idx'];keys=block_ids[idx]
            # Entries and failed/unobserved requests still become sent history.
            if not row.get('in_window') or row.get('is_entry'):continue
            if real and row.get('status')!='success':continue
            if row.get('gpu_hit') is None or row.get('cpu_hit') is None:continue
            root=roots[row['root_id']]
            u=next((i for i,key in enumerate(keys) if int(key) not in history[instance]),len(keys))
            trigger=root.triggers[idx][0]
            p=common_blocks(keys,block_ids[trigger]) if trigger in sent[instance] else 0
            item=decompose(row['prompt_len'] if real else row['prompt_tokens'],
                           row['gpu_hit']+row['cpu_hit'],u*block_size,p*block_size,block_size)
            item.update(instance_id=instance,trace_req_idx=idx,request_id=row.get('request_id'),
                        trigger_trace_req_idx=trigger)
            if real:
                server_wait=row.get('server_queued_ms')
                item['queued_s']=None if server_wait is None else (server_wait+(row.get('queued_ms') or 0))/1000
                item['server_queued_s']=None if server_wait is None else server_wait/1000
            else:item['queued_s']=row['queued_s'];item['server_queued_s']=row['queued_s']
            out.append(item)
        for row in group:
            instance=row['instance_id'];idx=row['trace_req_idx']
            history[instance].update(map(int,block_ids[idx]));sent[instance].add(idx)
        index=stop
    total=sum(r['prompt_blocks_tokens'] for r in out)
    counts={k:sum(r[k] for r in out) for k in CATEGORIES}
    gap=[r for r in out if r['gap_eviction']>0]
    avoidable=counts['gap_eviction']+counts['other_eviction']
    observed_wait=[r['queued_s'] for r in gap if r['queued_s'] is not None]
    summary=dict(version=VERSION,requests=len(out),prompt_blocks_tokens=total,source_tokens=counts,
                 source_fractions={k:v/total if total else None for k,v in counts.items()},
                 recomputed_fraction=sum(counts.values())/total if total else None,
                 tail_tokens=sum(r['tail_tokens'] for r in out),
                 hit_exceeds_history_anomalies=sum(r['hit_exceeds_history'] for r in out),
                 gap_share_of_avoidable=counts['gap_eviction']/avoidable if avoidable else None,
                 gap_requests=len(gap),gap_queue_observed_requests=len(observed_wait),
                 gap_requests_mean_queued_s=float(np.mean(observed_wait)) if observed_wait else None)
    return summary,out
