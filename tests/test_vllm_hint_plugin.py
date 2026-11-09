import hashlib
import importlib.util
import json
from types import SimpleNamespace as NS

import pytest
from agentkv.hints import make_envelope
from agentkv.policy.prefix_adapter import PrefixPolicyAdapter, ROOT
from agentkv.sim.callback_adapter import CallbackPolicyAdapter


def context(chain, keys, *, turn=0, gap=1000., final=False, hint=True):
    from vllm.v1.kv_offload.base import ReqContext
    row = dict(chain_id=chain, chain_turn=turn, final=final,
               hint_perfect_ms=None if final else gap)
    env = make_envelope(row,'perfect',f'{chain}/{turn}','r#0') if hint else None
    result = ReqContext(f'{chain}/{turn}', kv_transfer_params={'kv_hints':env} if env else None,
                        num_prompt_tokens=16*len(keys))
    for i,k in enumerate(keys):result.set_offload_key_position(k,16*(i+1))
    return result


@pytest.fixture(params=['indexed', 'chain'])
def plugin(tmp_path,monkeypatch,request):
    if importlib.util.find_spec('vllm') is None:pytest.skip('Requires vLLM, CPU only')
    from vllm.v1.kv_offload.base import ReqContext
    if not hasattr(ReqContext,'get_offload_prefix_keys'):pytest.skip('Requires approved ReqContext patch')
    from agentkv.policy import vllm_plugin
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(format='agentkv.vllm_hint_policy.v1',hint_tier='perfect',implementation=request.param,
        freeze_sha256=hashlib.sha256((ROOT/'docs/m5_scoring_frozen.json').read_bytes()).hexdigest())))
    monkeypatch.setenv('AGENTKV_HINT_POLICY_CONFIG',str(path))
    now=[100.];monkeypatch.setattr(vllm_plugin,'CLOCK',lambda:now[0])
    return vllm_plugin.HintAwareCachePolicy(8),now


def test_gpu_hit_shared_prefix_of_final_parent_uses_waiting_child_deadline(plugin):
    p,now=plugin
    shared,control=b'shared',b'control'
    p.insert(shared,NS(ref_cnt=0,chunk_id=0));p.insert(control,NS(ref_cnt=0,chunk_id=1))
    p.on_request_finished(((shared,),),set(),{shared},context('parent',[shared],final=True))
    p.on_request_finished(((control,),),set(),{control},context('later',[control],gap=60000.))
    ranks=dict(p.adapter.policy.ranks)
    # GPU-hit child has an empty CPU access list, but complete request metadata.
    p.on_request_finished((),set(),set(),context('child',[shared],gap=1000.))
    policy=p.adapter.policy;identity=('r#0','r#0/child')
    assert policy.ranks==ranks, 'Ownership supplementation must not promote recency'
    assert policy.owners[shared][identity]==16
    assert policy.chains[identity].mode=='waiting' and policy.chains[identity].deadline==101.
    assert p.evict(1,set())[0][0]==control


def test_incremental_store_ownership_depths_equal_single_batch(plugin):
    plugin_policy,now=plugin
    a=plugin_policy.adapter;b=PrefixPolicyAdapter(8,clock=lambda:now[0])
    keys=[b'head',b'tail'];ctx=context('c',keys)
    a.on_store_miss(keys[:1],ctx);a.insert(keys[0],NS(ref_cnt=-1,chunk_id=0))
    a.on_store_miss(keys[1:],ctx);a.insert(keys[1],NS(ref_cnt=-1,chunk_id=1))
    b.on_store_miss(keys,ctx)
    for i,k in enumerate(keys):b.insert(k,NS(ref_cnt=-1,chunk_id=i))
    assert a.policy.owners==b.policy.owners
    assert [a.policy.owners[k][('r#0','r#0/c')] for k in keys]==[16,32]


def test_evict_never_selects_pending_copying_or_protected_and_is_atomic(plugin):
    p,_=plugin;keys=[b'pending',b'copying',b'idle']
    for i,(k,ref) in enumerate(zip(keys,[-1,2,0])):p.insert(k,NS(ref_cnt=ref,chunk_id=i))
    assert p.evict(2,set()) is None and set(p.chunks)==set(keys)
    assert p.evict(1,{keys[2]}) is None and set(p.chunks)==set(keys)
    assert p.evict(1,set())[0][0]==keys[2]


def test_serving_and_simulator_callback_adapters_make_identical_decisions(plugin):
    p,now=plugin;sim=CallbackPolicyAdapter(8,clock=lambda:now[0])
    keys=[b'a',b'b',b'c'];ctx=context('c',keys)
    for actor in (p,sim):
        actor.on_store_miss(keys[:1],ctx)
        for i,k in enumerate(keys):actor.insert(k,NS(ref_cnt=0,chunk_id=i))
        actor.on_request_finished((keys[1:],),set(keys[1:]),set(),ctx)
    now[0]+=1
    for n,protected in [(1,{keys[0]}),(1,set()),(2,set())]:
        first=p.evict(n,protected);second=sim.evict(n,protected)
        assert (None if first is None else [k for k,_ in first])==(None if second is None else [k for k,_ in second])


def test_missing_hints_match_native_lru(plugin):
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    p,_=plugin;baseline=LRUCachePolicy(8);keys=[b'a',b'b',b'c'];ctx=context('c',keys,hint=False)
    for actor in (p,baseline):
        for i,k in enumerate(keys):actor.insert(k,NS(ref_cnt=0,chunk_id=i))
        actor.on_request_finished((keys,),set(keys),set(),ctx)
    assert [k for k,_ in p.evict(2,set())]==[k for k,_ in baseline.evict(2,set())]
    assert not p.adapter.valid_any


def test_original_recorded_final_sharing_counterexample_is_fixed(plugin):
    from vllm.v1.kv_offload.base import ReqContext
    example=json.loads((ROOT/'results/m5-vllm-key-diagnosis-20261104/counterexample.json').read_text())
    p,now=plugin;key=bytes.fromhex(example['witness']['key']);control=b'control'
    p.insert(key,NS(ref_cnt=0,chunk_id=0));p.insert(control,NS(ref_cnt=0,chunk_id=1))
    def ctx(hint):
        result=ReqContext(hint['message_id'],kv_transfer_params={'kv_hints':hint},num_prompt_tokens=16)
        result.set_offload_key_position(key,16)
        return result
    p.on_request_finished(((key,),),set(),{key},ctx(example['final_hint']))
    gap=example['waiting_hint']['actions'][0]['payload']['expected_gap_ms']
    p.on_request_finished(((control,),),set(),{control},context('later',[control],gap=gap+60000))
    p.on_request_finished((),set(),set(),ctx(example['waiting_hint']))
    assert p.evict(1,set())[0][0]==control
