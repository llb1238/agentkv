"""Serving callback adapter for the frozen scorer, without engine imports."""
import json
from pathlib import Path
from types import SimpleNamespace

from agentkv.hints import validate_envelope
from agentkv.policy.hint_aware import HintAwarePolicy
from agentkv.policy.remaining_life import RemainingLifeTable

ROOT = Path(__file__).resolve().parents[2]


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def envelope(context):
    for value in (_get(context, 'kv_hints'), _get(_get(context, 'kv_transfer_params'), 'kv_hints')):
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json')
        try:
            validate_envelope(value)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        return value
    return None


def frozen_options(tier, *, freeze=None, table=None):
    freeze = freeze or json.loads((ROOT/'docs/m5_scoring_frozen.json').read_text())
    options = dict(freeze['selected_options'][tier])
    if options['scoring'] == 'remaining':
        options['residual_table'] = RemainingLifeTable.load(tier=tier) if table is None else table
    return options


class PrefixPolicyAdapter:
    """Add ownership observations without inventing CPU cache accesses.

    The frozen scorer already exposes a miss-observation hook that does not
    promote recency. Its exact same implementation receives the full prefix
    before the original callback is forwarded. No scoring code is changed.
    """
    def __init__(self, capacity, *, clock, hint_tier='perfect', options=None):
        self.policy = HintAwarePolicy(capacity, clock=clock, **(options or frozen_options(hint_tier)))
        self.hint_tier = hint_tier
        self.valid_any = set()
        self.valid_finished = set()
        self.prefix_checks = []

    @property
    def chunks(self):
        return self.policy.chunks

    def get(self, key):
        return self.policy.get(key)

    def insert(self, key, chunk):
        return self.policy.insert(key, chunk)

    def remove(self, key):
        return self.policy.remove(key)

    def clear(self):
        return self.policy.clear()

    def mark_evictable(self, key):
        return self.policy.mark_evictable(key)

    def mark_non_evictable(self, key):
        return self.policy.mark_non_evictable(key)

    def _observe_prefix(self, context, *, finished=False):
        hint = envelope(context)
        if hint is None:
            # Invalid/missing hints have the ordinary LRU fallback. Strip both
            # channels so a malformed direct channel cannot shadow a valid one.
            return SimpleNamespace(req_id=_get(context, 'req_id'), kv_hints=None,
                kv_transfer_params=None, get_offload_key_position=context.get_offload_key_position)
        prefix = tuple(context.get_offload_prefix_keys())
        positions = [context.get_offload_key_position(k) for k in prefix]
        boundary = context.num_prompt_tokens
        if not isinstance(boundary, int) or boundary < 0:
            raise ValueError('Invalid prompt boundary')
        if any(p is None or not 0 < p <= boundary for p in positions) or positions != sorted(positions):
            raise ValueError('Invalid full prefix positions')
        normalized = SimpleNamespace(req_id=context.req_id, kv_hints=hint, kv_transfer_params=None,
            get_offload_key_position=context.get_offload_key_position)
        self.valid_any.add(hint['message_id'])
        # Frozen on_store_miss records membership/pending ownership only;
        # original touch/finished inputs below alone decide recency.
        self.policy.on_store_miss(prefix, normalized)
        if finished:
            self.valid_finished.add(hint['message_id'])
            self.prefix_checks.append(dict(request_id=context.req_id, message_id=hint['message_id'],
                prompt_tokens=boundary, prefix_keys=len(prefix), positions=positions))
        return normalized

    def touch(self, keys, context):
        return self.policy.touch(keys, self._observe_prefix(context))

    def on_store_miss(self, keys, context):
        return self.policy.on_store_miss(keys, self._observe_prefix(context))

    def on_request_finished(self, groups, inserted, reused, context):
        return self.policy.on_request_finished(groups, inserted, reused,
            self._observe_prefix(context, finished=True))

    def evict(self, n, protected):
        return self.policy.evict(n, protected)

    def ownership(self):
        total = len(self.chunks)
        owned = sum(bool(self.policy.owners.get(k)) for k in self.chunks)
        return dict(cpu_blocks=total, owned_blocks=owned, owned_fraction=owned/total if total else 0.)
