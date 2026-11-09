import gzip
import importlib.util
import json
from types import SimpleNamespace as NS

import pytest
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.prefix_adapter import ROOT
from agentkv.hints import make_envelope


@pytest.mark.parametrize('kind',['probe','smoke'])
def test_complete_trace_replays_mutable_refs_and_nested_callbacks(tmp_path,monkeypatch,kind):
    if importlib.util.find_spec('vllm') is None:pytest.skip('Requires patched vLLM, CPU only')
    from vllm.v1.kv_offload.base import ReqContext
    if not hasattr(ReqContext,'get_offload_prefix_keys'):pytest.skip('Requires ReqContext patch')
    from agentkv.policy.full_trace_plugin import FullLRUProbePolicy,TracedHintAwarePolicy
    from agentkv.analysis import replay_callbacks
    config=tmp_path/'policy-config.json'
    config.write_text(json.dumps(dict(format='agentkv.vllm_hint_policy.v1',hint_tier='perfect',
        freeze_sha256=sha256_file(ROOT/'docs/m5_scoring_frozen.json'))))
    monkeypatch.setenv('AGENTKV_HINT_POLICY_CONFIG',str(config))
    monkeypatch.setenv('AGENTKV_CALLBACK_TRACE_DIR',str(tmp_path/'callback-traces'))
    monkeypatch.setattr(replay_callbacks,'git_commit',lambda:'test-only')
    cls=FullLRUProbePolicy if kind=='probe' else TracedHintAwarePolicy
    actor=cls(4);keys=[b'a',b'b']
    hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect','r/c/0','r#0')
    ctx=ReqContext('r/c/0',kv_transfer_params={'kv_hints':hint},num_prompt_tokens=32)
    for i,key in enumerate(keys):ctx.set_offload_key_position(key,16*(i+1))
    actor.on_store_miss(keys[:1],ctx)
    for i,key in enumerate(keys):actor.insert(key,NS(ref_cnt=-1,chunk_id=i))
    actor.on_store_miss(keys[1:],ctx)
    for key in keys:
        actor.get(key).ref_cnt=0;actor.mark_evictable(key)
    actor.on_request_finished((keys,),set(keys),set(),ctx)
    actor.mark_non_evictable(keys[1]);actor.chunks[keys[1]].ref_cnt=1
    assert actor.evict(1,{keys[0]}) is None
    actor.get(keys[1]).ref_cnt=0;actor.mark_evictable(keys[1])
    assert len(actor.evict(1,{keys[0]}))==1
    actor.trace.close()
    (tmp_path/'trial-run.json').write_text(json.dumps(dict(vllm_commit='test-backend')))
    (tmp_path/'run.json').write_text(json.dumps(dict(summary=dict(hint_sent=1))))
    result=replay_callbacks.replay(tmp_path,tmp_path/'replayed')
    assert all(result['invariants'].values())
    assert result['candidate_adapter_decisions_identical']
    assert result['prefix_checked_requests']==1 and result['eviction_decisions']==2
    events=[json.loads(line) for line in gzip.open(next((tmp_path/'callback-traces').glob('*.gz')),'rt')]
    assert events[-1]['event']=='close'
    evicts=[e for e in events if e['event']=='begin' and e['method']=='evict']
    assert evicts[0]['args'][1] and any(value and value[0]==1 for _,value in evicts[0]['refs'])
    if kind=='smoke':assert any(e['event']=='clock' for e in events)


def test_manager_observer_records_actual_zero_to_one_and_one_to_zero(tmp_path,monkeypatch):
    if importlib.util.find_spec('vllm') is None:pytest.skip('Requires vLLM, CPU only')
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.base import ReqContext
    if not hasattr(ReqContext,'get_offload_prefix_keys'):pytest.skip('Requires ReqContext patch')
    from agentkv.bench.observability_plugin import install_chunk_ref_observation
    monkeypatch.setenv('AGENTKV_CALLBACK_TRACE_DIR',str(tmp_path/'trace'))
    install_chunk_ref_observation(CPUOffloadingManager)
    manager=CPUOffloadingManager(num_chunks=4,cache_policy='FullLRUProbePolicy',
        cache_policy_module_path='agentkv.policy.full_trace_plugin')
    key=b'block-hash'+bytes(4)
    ctx=ReqContext('r',num_prompt_tokens=16);ctx.set_offload_key_position(key,16)
    manager.on_new_request(ctx)
    assert manager.prepare_store([key],ctx) is not None
    manager.complete_store([key],ctx)
    manager.prepare_load([key],ctx)
    manager.complete_load([key],ctx)
    manager.on_request_finished(ctx)
    manager._policy.trace.close()
    events=[json.loads(line) for line in gzip.open(next((tmp_path/'trace').glob('*.gz')),'rt')]
    transitions=[(old[0] if old else None,new[0] if new else None)
                 for e in events if e['event']=='manager_refs' for _,old,new in e['transitions']]
    assert (-1,0) in transitions and (0,1) in transitions and (1,0) in transitions
