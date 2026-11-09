from types import SimpleNamespace as NS
import pytest
from agentkv.bench.check_chain_scoring import tail_witness, returned_membership
from agentkv.hints import make_envelope
from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
from agentkv.policy.prefix_adapter import frozen_options


@pytest.mark.parametrize('tier', ['online', 'perfect'])
def test_strict_depth_tail_order_conflicts_with_frozen_partial_prefix_touch(tier):
    a, b = b'A' * 16, b'B' * 16
    def context(chain, gap):
        row = dict(chain_id=chain, chain_turn=0, final=False,
                   hint_online_ms=gap, hint_online_p90_ms=2 * gap, hint_perfect_ms=gap)
        return NS(req_id=chain, kv_hints=make_envelope(row, tier, chain, 'r#0'), kv_transfer_params=None,
                  num_prompt_tokens=32, get_offload_prefix_keys=lambda: (a, b),
                  get_offload_key_position={a: 16, b: 32}.get)
    first, second = context('parent', 1000), context('child', 10000)
    assert frozen_options(tier)['tail_first'] is False
    actor = OptimizedPrefixAdapter(8, clock=lambda: 100., hint_tier=tier)
    actor.on_store_miss([a, b], first)
    actor.insert(a, NS(ref_cnt=0));actor.insert(b, NS(ref_cnt=0))
    actor.on_request_finished(((a, b),), {a, b}, set(), first)
    actor.touch([b], second)
    actor.on_request_finished(((b,),), set(), {b}, second)
    witness = tail_witness(actor.policy, a, 100., set())
    assert witness['frozen_selected']['key_hex'] == a.hex()
    assert witness['eligible_deeper']['key_hex'] == b.hex()
    assert witness['identical_score_lines'] and witness['identical_owner_sets']
    assert tail_witness(actor.policy, a, 100., {b}) is None
    actor.chunks[b].ref_cnt = 1
    assert tail_witness(actor.policy, a, 100., set()) is None
    actor.chunks[b].ref_cnt = 0
    assert [k for k, _ in actor.evict(1, set())] == [a]


def test_returned_membership_distinguishes_set_change_from_order_only():
    witness = dict(call=7, frozen_selected=dict(key_hex=b'a'.hex()), eligible_deeper=dict(key_hex=b'b'.hex()))
    step = ('call', 'evict', None, None, None, None, (b'a', b'c'), 7)
    assert returned_membership([step], witness)['strict_depth_tail_changes_returned_set']
    step = (*step[:-2], (b'a', b'b'), 7)
    assert not returned_membership([step], witness)['strict_depth_tail_changes_returned_set']
