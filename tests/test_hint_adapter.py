import numpy as np
import pytest
from agentkv.bench.estimate_replay import simulate_requests
from agentkv.bench.steady_state import RootSchedule,SessionInstance
from agentkv.sim.two_tier import canonical_prefix_blocks,simulate


def fixture():
    prompts=[[1,2,3],[1,2,3,4,5],[8,9,10,11,12,13],[1,2,3,4,5],[14,15,16],[1,2,3]]*4
    rows=[]
    for i,p in enumerate(prompts):
        rows.append(dict(root_id='r',chain_id='c',chain_turn=i,trace_req_idx=str(i),prompt_len=len(p),max_tokens=1,
            anchor_trace_req_idx=None,start_delay_s=0 if not i else None,gap_after_s=.1 if i<len(prompts)-1 else None,
            final=i==len(prompts)-1,hint_perfect_ms=100.,hint_noisy_ms=50.,hint_online_ms=200.,hint_online_p90_ms=300.,
            _tokens=np.array(p,dtype='<i4').tobytes()))
    times=simulate_requests(rows,.5);end=max(times.values())+.5
    root=RootSchedule(rows,[dict(root_id='r',chain_id='c',chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],est_send_s=times[r['trace_req_idx']],est_duration_s=end) for r in rows])
    blocks,_=canonical_prefix_blocks(rows,1)
    return {'r':root},[SessionInstance('r',0,0,0,False)],blocks


@pytest.mark.parametrize('cpu',[0,2,5,20])
def test_none_adapter_equals_lru_even_when_store_is_skipped(cpu):
    args=fixture()
    a,ar=simulate(*args,8,cpu,2,duration=20,warmup=0,block_size=1)
    b,br=simulate(*args,8,cpu,2,duration=20,warmup=0,block_size=1,cpu_policy='hint',hint_tier='none')
    assert a==b and ar==br


@pytest.mark.parametrize('tier',['perfect','noisy','online'])
def test_hint_does_not_change_gpu_or_scheduling(tier):
    args=fixture();_,base=simulate(*args,8,10,2,duration=20,warmup=0,block_size=1)
    result,rows=simulate(*args,8,10,2,duration=20,warmup=0,block_size=1,cpu_policy='hint',hint_tier=tier)
    assert [(r['trace_req_idx'],r['send_s'],r['gpu_hit']) for r in rows]==[(r['trace_req_idx'],r['send_s'],r['gpu_hit']) for r in base]
    assert all(r['gpu_hit']+r['cpu_hit']+r['recomputed']==r['prompt_tokens'] for r in rows)
