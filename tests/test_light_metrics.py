import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from agentkv.policy.light_metrics import Histogram, LightMetrics


def test_histogram_bounds_counts_window_and_single_exit_write(tmp_path, monkeypatch):
    h = Histogram()
    values = [0, 1, 17, 254, 255, 256, 1000, 1000000]
    for v in values:h.add(v)
    assert h.count == len(values) and h.total == sum(values)
    for q in (.5, .99):
        lo, hi = h.quantile(q)
        value = sorted(values)[max(0, __import__('math').ceil(len(values)*q)-1)]
        assert lo <= value <= hi and hi-lo <= max(1, value/128)
    store = LightMetrics(tmp_path/'metrics', pid=123)
    store.set_window(0, 1, 2)
    for t in (0, 1_000_000_000, 2_000_000_000, 3_000_000_000):
        store.observe('policy.evict', 100, t)
        store.record('request_received', ns=t, request_id=str(t))
    assert not (tmp_path/'metrics').exists(), 'No file I/O during callbacks'
    writes=[];original=Path.write_text
    monkeypatch.setattr(Path,'write_text',lambda self,*a,**kw:(writes.append(self),original(self,*a,**kw))[1])
    store.close();store.close()
    assert len(writes)==1
    output=json.loads(writes[0].read_text())
    assert output['closed'] and output['histograms']['policy.evict']['count']==4
    assert output['window_histograms']['policy.evict']['count']==2


def test_none_specialization_matches_native_and_reference_on_manager_callbacks():
    pytest.importorskip('vllm.v1.kv_offload.base', reason='Requires optional patched vLLM, CPU only')
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    from agentkv.policy.measured_policy import NoneHintAdapter
    from agentkv.policy.prefix_adapter import PrefixPolicyAdapter
    actors=[LRUCachePolicy(8),NoneHintAdapter(8),PrefixPolicyAdapter(8,clock=lambda:1.)]
    ctx=ReqContext('r',num_prompt_tokens=48)
    for i in range(3):ctx.set_offload_key_position(i,16*(i+1))
    for p in actors:
        p.on_store_miss([0,1,2],ctx)
        for k in range(3):p.insert(k,NS(ref_cnt=0,chunk_id=k))
        p.on_request_finished(([2,0],[1]),set(range(3)),set(),ctx)
        p.touch([1,0],ctx)
        p.mark_non_evictable(0);p.chunks[0].ref_cnt=1
    for n,protected in [(2,{2}),(1,set()),(1,set())]:
        result=[p.evict(n,protected) for p in actors]
        assert [(None if v is None else [k for k,_ in v]) for v in result].count(
            None if result[0] is None else [k for k,_ in result[0]])==3
    ctx.kv_hints={'unexpected':'hint'}
    with pytest.raises(ValueError,match='explicit none'):actors[1].touch([],ctx)


def test_cached_prefix_enumerated_once_and_released_at_finish():
    pytest.importorskip('vllm.v1.kv_offload.base', reason='Requires optional patched vLLM, CPU only')
    from vllm.v1.kv_offload.base import ReqContext
    from agentkv.hints import make_envelope
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    p=OptimizedPrefixAdapter(8,clock=lambda:1.)
    hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect','r/c/0','r#0')
    ctx=ReqContext('r/c/0',kv_transfer_params={'kv_hints':hint},num_prompt_tokens=32)
    for i,k in enumerate((b'a',b'b')):ctx.set_offload_key_position(k,16*(i+1))
    p.on_store_miss([b'a'],ctx)
    for k in (b'a',b'b'):p.insert(k,NS(ref_cnt=0,chunk_id=0))
    p.on_store_miss([b'b'],ctx)
    p.touch([b'a'],ctx)
    p.on_request_finished(((b'a',b'b'),),set(),set(),ctx)
    assert p.prefix_enumerations==1 and p.prefix_cache_hits==3 and not p._requests


def test_light_request_summaries_join_after_close(tmp_path):
    from agentkv.bench.replay import join_observations
    store=LightMetrics(tmp_path,pid=123)
    store.record('request_received',ns=100,request_id='server-0')
    store.record('request_scheduled',ns=300,request_id='server-0')
    store.record('prefill',ns=400,request_id='server-0',prompt_tokens=32,gpu_hit=16,cpu_hit=0,recomputed=16)
    store.close()
    records=[dict(request_id='client',server_request_id='server',prompt_len=32,send_ns=10)]
    assert join_observations(records,tmp_path)['prefills_joined']==1
    assert records[0]['gpu_hit_fraction']==.5 and records[0]['server_queued_ms']==.0002
