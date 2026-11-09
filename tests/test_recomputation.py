"""Prefix decomposition and causal check of simulator two-tier losses."""
import numpy as np
import pytest
from agentkv.analysis.recomputation import decompose,classify_records
from agentkv.bench.steady_state import SessionInstance,RootSchedule
from agentkv.bench.estimate_replay import simulate_requests
from agentkv.sim.two_tier import canonical_prefix_blocks,simulate


def plan(prompts,sends=None):
    rows=[]
    for i,prompt in enumerate(prompts):
        rows.append(dict(root_id='r',chain_id='c' if sends is None else str(i),chain_turn=i if sends is None else 0,
            trace_req_idx=str(i),prompt_len=len(prompt),max_tokens=1,anchor_trace_req_idx=None,
            start_delay_s=(0 if i==0 else None) if sends is None else sends[i],gap_after_s=.2 if i<len(prompts)-1 else None,
            _tokens=np.array(prompt,dtype='<i4').tobytes()))
    times=simulate_requests(rows,.5);end=max(times.values())+.5
    schedule=[dict(root_id='r',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],
                   est_send_s=times[r['trace_req_idx']],est_duration_s=end) for r in rows]
    return rows,RootSchedule(rows,schedule)


def test_rounded_partition_tail_and_anomaly():
    result=decompose(79,16,64,48)
    assert [result[k] for k in ('unavoidable','gap_eviction','other_eviction','tail_tokens')]==[0,32,16,15]
    assert decompose(80,64,32,16)['hit_exceeds_history']
    assert decompose(80,64,32,16)['history_prefix']==64


def test_infinite_cache_has_no_eviction_categories():
    rows,root=plan([[1]*32,[1]*32+[2]*16,[1]*32+[3]*16])
    blocks,_=canonical_prefix_blocks(rows)
    summary,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,10000,20000,1,
                              duration=5,warmup=0,audit_evictions=True)
    totals,items=classify_records(records,{'r':root},blocks)
    assert totals['source_tokens']['gap_eviction']==totals['source_tokens']['other_eviction']==0
    assert totals['hit_exceeds_history_anomalies']==0
    assert summary['gap_eviction_audit']['consistency']==1


def test_interference_between_two_turns_is_all_gap_eviction():
    rows,root=plan([[1]*32,[1]*32],sends=[0,2])
    rows[1].update(chain_id='0',chain_turn=1,start_delay_s=None)
    rows[0]['gap_after_s']=1.5
    extra=dict(rows[0],trace_req_idx='2',chain_id='noise',chain_turn=0,prompt_len=64,
               start_delay_s=.75,gap_after_s=None,_tokens=np.array([2]*64,dtype='<i4').tobytes())
    rows.append(extra)
    times=simulate_requests(rows,.5);end=max(times.values())+.5
    root=RootSchedule(rows,[dict(root_id='r',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],est_send_s=times[r['trace_req_idx']],est_duration_s=end) for r in rows])
    blocks,_=canonical_prefix_blocks(rows)
    summary,records=simulate({'r':root},[SessionInstance('r',0,0,0,False)],blocks,80,64,1,
                              duration=5,warmup=0,audit_evictions=True)
    _,items=classify_records(records,{'r':root},blocks)
    second=next(r for r in items if r['trace_req_idx']=='1')
    assert second['gap_eviction']==32 and second['other_eviction']==0
    assert summary['gap_eviction_audit']['blocks']==2
    assert summary['gap_eviction_audit']['consistency']==1


def test_history_is_sent_not_completed_and_entry_counts_as_history():
    rows,root=plan([[1]*32,[1]*32],sends=[0,.1]);blocks,_=canonical_prefix_blocks(rows)
    records=[dict(instance_id=0,root_id='r',trace_req_idx=str(i),send_s=i*.1,end_s=i*.1+.5,
                  is_entry=i==0,in_window=True,prompt_tokens=32,gpu_hit=0,cpu_hit=0,queued_s=0.) for i in range(2)]
    summary,items=classify_records(records,{'r':root},blocks)
    assert len(items)==1 and items[0]['other_eviction']==32 and items[0]['unavoidable']==0


def test_simultaneous_sends_do_not_contribute_to_history():
    rows,root=plan([[1]*32,[1]*32],sends=[0,0]);blocks,_=canonical_prefix_blocks(rows)
    records=[dict(instance_id=0,root_id='r',trace_req_idx=str(i),send_s=0.,is_entry=False,in_window=True,
                  prompt_tokens=32,gpu_hit=0,cpu_hit=0,queued_s=0.) for i in range(2)]
    summary,_=classify_records(records,{'r':root},blocks)
    assert summary['source_tokens']['unavoidable']==64


def test_anchored_child_uses_parent_prefix_and_real_matches_simulated():
    rows,_=plan([[1]*32,[1]*32+[2]*16],sends=[0,.75])
    rows[1]['anchor_trace_req_idx']='0';rows[1]['start_delay_s']=.25
    times=simulate_requests(rows,.5);end=max(times.values())+.5
    root=RootSchedule(rows,[dict(root_id='r',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],est_send_s=times[r['trace_req_idx']],est_duration_s=end) for r in rows])
    blocks,_=canonical_prefix_blocks(rows)
    simulated=[dict(instance_id=0,root_id='r',trace_req_idx=str(i),send_s=times[str(i)],end_s=times[str(i)]+.5,
                    is_entry=False,in_window=True,prompt_tokens=rows[i]['prompt_len'],gpu_hit=0,cpu_hit=0,queued_s=.1) for i in range(2)]
    expected,items=classify_records(simulated,{'r':root},blocks)
    assert items[1]['gap_eviction']==32
    actual=[]
    for row in simulated:
        actual.append(dict(row,status='success',prompt_len=row['prompt_tokens'],send_ns=int(row['send_s']*1e9),queued_ms=25,server_queued_ms=75))
    obtained,_=classify_records(actual,{'r':root},blocks,real=True)
    assert obtained==expected
