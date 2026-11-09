import random
from types import SimpleNamespace as NS
import pytest
from agentkv.hints import make_envelope
from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy
from agentkv.policy.chain_policy import ChainHintAwarePolicy


@pytest.mark.parametrize('scoring', ['deadline', 'remaining', 'lru_final'])
def test_chain_queue_preserves_rank_shared_protected_fallback_and_time(scoring):
    now = [100.]
    rng = random.Random(771)
    actors = [cls(32, clock=lambda: now[0], scoring=scoring)
              for cls in (OptimizedHintAwarePolicy, ChainHintAwarePolicy)]
    pool = list(range(20)); turns = {}
    for step in range(200):
        now[0] += rng.random() * 2
        chain = str(rng.randrange(4)); turn = turns.get(chain, 0); turns[chain] = turn + 1
        keys = rng.sample(pool, rng.randrange(1, 9)); final = rng.randrange(5) == 0
        hint = make_envelope(dict(chain_id=chain, chain_turn=turn, final=final,
            hint_perfect_ms=None if final else rng.random() * 3000), 'perfect', f'{chain}/{turn}', 'r#0')
        ctx = NS(req_id=f'{chain}/{turn}', kv_hints=hint, kv_transfer_params=None,
                 get_offload_key_position={k: 16 * (i + 1) for i, k in enumerate(keys)}.get)
        for actor in actors:
            actor.on_store_miss(keys, ctx)
            for key in keys:
                if actor.get(key) is None:
                    actor.insert(key, NS(ref_cnt=0, chunk_id=key))
            actor.on_request_finished((keys,), set(keys), set(), ctx)
        if actors[0].chunks and step % 3 == 0:
            key = next(iter(actors[0].chunks))
            for actor in actors:
                actor.mark_non_evictable(key)
                actor.chunks[key].ref_cnt = 1
        protected = set(rng.sample(pool, rng.randrange(4))); n = rng.randrange(6)
        results = [a.evict(n, protected) for a in actors]
        assert [None if v is None else [k for k, _ in v] for v in results][0] == (
            None if results[1] is None else [k for k, _ in results[1]])
        assert set(actors[0].chunks) == set(actors[1].chunks)
        for actor in actors:
            for key, chunk in actor.chunks.items():
                if chunk.ref_cnt:
                    chunk.ref_cnt = 0
                    actor.mark_evictable(key)


def test_no_hint_chain_heads_keep_exact_lru_and_failed_evictions_are_atomic():
    p = ChainHintAwarePolicy(8, clock=lambda: 1.)
    for key in range(4):
        p.insert(key, NS(ref_cnt=0))
    assert p.evict(5, set()) is None
    assert p.evict(4, {2}) is None and len(p.chunks) == 4
    assert [k for k, _ in p.evict(3, {2})] == [0, 1, 3]
    assert p.evict(0, set()) == []


def test_chain_queue_retains_reference_timeout_compaction_on_reuse():
    now = [0.]
    actors = [cls(8, clock=lambda: now[0], scoring='deadline')
              for cls in (OptimizedHintAwarePolicy, ChainHintAwarePolicy)]
    def context(turn, final=False, gap=1000.):
        hint = make_envelope(dict(chain_id='a', chain_turn=turn, final=final,
            hint_perfect_ms=None if final else gap), 'perfect', str(turn), 'r#0')
        return NS(req_id=str(turn), kv_hints=hint, kv_transfer_params=None,
                  get_offload_key_position=lambda k: 16)
    unknown = NS(req_id='b', kv_hints=None, kv_transfer_params=None, instance_id='r#0', chain_id='b',
                 get_offload_key_position=lambda k: 16)
    for p in actors:
        p.on_store_miss([b'k'], unknown);p.insert(b'k', NS(ref_cnt=0))
        p.on_request_finished(((b'k',),), set(), set(), context(0))
        for _ in range(160):p.mark_evictable(b'k')
        p.on_request_finished(((b'k',),), set(), set(), context(1, final=True))
        p.evict(1, set())
        p.on_request_finished((), set(), set(), context(2, gap=100000.))
    now[0] = 4.
    for p in actors:p.on_store_miss([], context(2, gap=100000.))
    assert all(not p.chains[('r#0', 'r#0/a')].expired for p in actors)


def test_remaining_empty_finished_chains_and_shared_graph_are_reclaimed():
    p=ChainHintAwarePolicy(8,clock=lambda:1.,scoring='remaining')
    for i in range(100):
        ctx=NS(req_id=i,instance_id=str(i),chain_id='a',kv_hints=None,kv_transfer_params=None)
        p.on_store_miss([i],ctx)
        p.on_request_finished((),set(),set(),ctx)
    assert not p.chains and not p._pending_by_chain
    for name in ('a','b','c'):
        ctx=NS(req_id=name,instance_id='shared',chain_id=name,kv_hints=None,kv_transfer_params=None)
        p.on_store_miss([1,2],ctx)
        for key in (1,2):
            if p.get(key) is None:p.insert(key,NS(ref_cnt=0))
        p.on_request_finished(([1,2],),set(),set(),ctx)
    assert len(p._neighbor_groups)==1 and len(p._neighbors)==3
    assert p.evict(2,{1}) is None
    assert len(p._neighbor_groups)==1 and len(p._neighbors)==3
    assert len(p.evict(2,set()))==2
    assert not p.chains and not p._neighbors and not p._neighbor_refs and not p._neighbor_groups
    assert not p._group_counts and not p._pending_by_chain


def test_archived_truncated_gzip_replays_only_completed_callback_prefix(tmp_path):
    import gzip,json
    pytest.importorskip('vllm.v1.kv_offload.base')
    from agentkv.bench.profile_callbacks import compile_trace
    folder=tmp_path/'callback-traces';folder.mkdir()
    rows=[dict(event='open',capacity=1),dict(event='begin',call=1,parent=None,method='evict',args=[0,[]],refs=[],ns=0),
          dict(event='end',call=1,result=[]),dict(event='begin',call=2,parent=None,method='evict',args=[0,[]],refs=[],ns=1)]
    payload=('\n'.join(json.dumps(r) for r in rows)+'\n').encode()
    (folder/'partial.gz').write_bytes(gzip.compress(payload)[:-8])
    with pytest.raises(EOFError):compile_trace(tmp_path)
    steps,capacity=compile_trace(tmp_path,allow_unclosed=True)
    assert capacity==1 and len(steps)==1 and steps[0][-1]==1


def test_remaining_event_compaction_removes_stale_generations_without_finishing_active_chain():
    now=[0.]
    p=ChainHintAwarePolicy(1,clock=lambda:now[0],scoring='remaining')
    context=NS(req_id='active',instance_id='r',chain_id='a',kv_hints=None,kv_transfer_params=None)
    p.on_store_miss([],context)
    c=p.chains['r','a']
    p._events=[(1000.,i,'life',('r','a'),(c.generation-1,c.version)) for i in range(200)]
    now[0]=2.
    p.on_store_miss([],context)
    assert not p._events and p.chains['r','a'] is c and not c.finished


def test_position_fast_path_keeps_first_context_and_tracks_replaced_position_table():
    ReqContext=pytest.importorskip('vllm.v1.kv_offload.base').ReqContext
    from agentkv.policy.chain_adapter import ChainPrefixAdapter
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect','msg','root')
    first=ReqContext('r',kv_hints=hint,num_prompt_tokens=32)
    first._offload_key_positions={b'a':16,b'b':32}
    second=ReqContext('r',kv_hints=hint,num_prompt_tokens=32)
    second._offload_key_positions={b'a':16,b'b':32,b'c':48}
    actors=[cls(4,clock=lambda:1.,hint_tier='online') for cls in (OptimizedPrefixAdapter,ChainPrefixAdapter)]
    for p in actors:
        p.on_store_miss([b'a',b'b'],first)
        for k in (b'a',b'b'):p.insert(k,NS(ref_cnt=0))
        p.on_store_miss([],second)
        assert p._requests['r'][1].get_offload_key_position(b'c') is None
    first._offload_key_positions=dict(second._offload_key_positions)
    for p in actors:
        p.on_store_miss([],second)
        assert p._requests['r'][1].get_offload_key_position(b'c')==48
        p.on_request_finished(((b'a',b'b'),),set(),set(),second)
    assert actors[0].policy.owners==actors[1].policy.owners
    assert not actors[1]._position_contexts and not actors[1]._requests and not actors[1]._prefix_keysets


def test_prefix_observation_is_repeated_after_intervening_cpu_replacement():
    ReqContext=pytest.importorskip('vllm.v1.kv_offload.base').ReqContext
    from agentkv.policy.chain_adapter import ChainPrefixAdapter
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    def context(name,final):
        hint=make_envelope(dict(chain_id=name,chain_turn=0,final=final,hint_perfect_ms=None if final else 1000.),
                           'perfect',name,'root')
        c=ReqContext(name,kv_hints=hint,num_prompt_tokens=16);c._offload_key_positions={b'a':16};return c
    child=context('child',False);parent=context('parent',True)
    actors=[cls(4,clock=lambda:1.,hint_tier='online') for cls in (OptimizedPrefixAdapter,ChainPrefixAdapter)]
    for p in actors:
        p.insert(b'a',NS(ref_cnt=0))
        p.on_store_miss([],child)
        p.remove(b'a')
        p.on_store_miss([b'a'],parent);p.insert(b'a',NS(ref_cnt=0))
        p.on_request_finished(((b'a',),),set(),set(),parent)
        assert all(cid[1].endswith('/parent') for cid in p.policy.owners[b'a'])
        p.on_store_miss([],child)
        assert any(cid[1].endswith('/child') for cid in p.policy.owners[b'a'])
        p.on_request_finished((),set(),set(),child)
    assert actors[0].policy.owners==actors[1].policy.owners
    assert not actors[1]._prefix_keysets


def test_missing_cached_position_keeps_actual_callback_index_fallback():
    ReqContext=pytest.importorskip('vllm.v1.kv_offload.base').ReqContext
    from agentkv.policy.chain_adapter import ChainPrefixAdapter
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect','msg','root')
    actors=[]
    for cls in (OptimizedPrefixAdapter,ChainPrefixAdapter):
        context=ReqContext('r',kv_hints=hint,num_prompt_tokens=32)
        context._offload_key_positions={b'a':16,b'b':32}
        actor=cls(4,clock=lambda:1.,hint_tier='online')
        actor.on_store_miss([],context)
        del context._offload_key_positions[b'b']
        actor.on_store_miss([b'b'],context);actor.insert(b'b',NS(ref_cnt=0))
        actors.append(actor)
    assert actors[0].policy.owners==actors[1].policy.owners
    assert set(actors[1].policy.owners[b'b'].values())=={1}


def test_finished_only_prefix_registers_resident_keys_and_leaves_no_temporary_pending():
    ReqContext=pytest.importorskip('vllm.v1.kv_offload.base').ReqContext
    from agentkv.policy.chain_adapter import ChainPrefixAdapter
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect','msg','root')
    context=ReqContext('r',kv_hints=hint,num_prompt_tokens=32);context._offload_key_positions={b'a':16,b'b':32}
    actors=[cls(4,clock=lambda:1.,hint_tier='online') for cls in (OptimizedPrefixAdapter,ChainPrefixAdapter)]
    for p in actors:
        p.insert(b'a',NS(ref_cnt=0))
        p.on_request_finished((),set(),set(),context)
        assert not p.policy._pending and not p.policy._pending_requests
    assert actors[0].policy.owners==actors[1].policy.owners
    assert all(c.mode=='waiting' for c in actors[1].policy.chains.values())


def test_finishing_prefix_preserves_overwrite_of_another_request_pending_owner():
    ReqContext=pytest.importorskip('vllm.v1.kv_offload.base').ReqContext
    from agentkv.policy.chain_adapter import ChainPrefixAdapter
    from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
    def context(req):
        hint=make_envelope(dict(chain_id='c',chain_turn=0,final=False,hint_perfect_ms=1000.),'perfect',req,'root')
        c=ReqContext(req,kv_hints=hint,num_prompt_tokens=16);c._offload_key_positions={b'a':16};return c
    actors=[cls(4,clock=lambda:1.,hint_tier='online') for cls in (OptimizedPrefixAdapter,ChainPrefixAdapter)]
    for p in actors:
        p.on_store_miss([b'a'],context('other'))
        p.on_request_finished((),set(),set(),context('finished'))
        assert not p.policy._pending
        p.insert(b'a',NS(ref_cnt=0))
    assert actors[0].policy.owners==actors[1].policy.owners=={}


def test_batched_pending_indexes_preserve_duplicates_partial_inserts_and_overwrites():
    actors=[cls(8,clock=lambda:1.,scoring='remaining')
            for cls in (OptimizedHintAwarePolicy,ChainHintAwarePolicy)]
    def context(req):
        return NS(req_id=req,instance_id='r',chain_id='c',kv_hints=None,
                  get_offload_key_position={1:16,2:32,3:48}.get)
    for req,keys,inserted in [('a',[1,2,1],[1]),('b',[3,1,2,3],[3]),('b',[2,1,3],[])]:
        for actor in actors:
            actor.on_store_miss(keys,context(req))
            for key in inserted:actor.insert(key,NS(ref_cnt=0))
        assert actors[0]._pending==actors[1]._pending
        assert actors[0]._pending_requests==actors[1]._pending_requests
        assert actors[0].owners==actors[1].owners
        expected={k for k,v in actors[1]._pending.items() if ('r','c') in v}
        assert actors[1]._pending_by_chain.get(('r','c'),set())==expected
    for actor in actors:
        actor.on_request_finished(([1,3],),set(),set(),context('b'))
    assert actors[0]._pending==actors[1]._pending=={}
    assert not actors[1]._pending_by_chain
