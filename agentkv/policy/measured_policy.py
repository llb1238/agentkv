"""Identical lightweight timing envelopes for native LRU and hint policies."""
import time
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from agentkv.policy.light_metrics import get_store


class NoneHintAdapter(LRUCachePolicy):
    """Native implementation of the scorer's exact no-hint LRU degeneration.

    Explicitly selected only by hint_tier=none. Unexpected hints are rejected,
    rather than accidentally ignored. Known-hint runs use the generic scorer.
    """
    def __init__(self, capacity, *, check_hints=True):
        super().__init__(capacity)
        self.check_hints = check_hints
        self.valid_any = set()
        self.valid_finished = set()

    @property
    def policy(self):return self

    def _check(self, ctx):
        if self.check_hints and (ctx.kv_hints is not None or
                (ctx.kv_transfer_params or {}).get('kv_hints') is not None):
            raise ValueError('A hint was supplied to the explicit none specialization')

    def touch(self, keys, context):
        self._check(context)
        return super().touch(keys, context)

    def on_store_miss(self, keys, context):
        self._check(context)
        # The fixed manager supplies only keys absent from CPU storage here.
        # Native touch therefore does not change their recency either.
        return LRUCachePolicy.touch(self, keys, context)

    def on_request_finished(self, groups, inserted, reused, context):
        self._check(context)
        return super().on_request_finished(groups, inserted, reused, context)


def _forward(name):
    def invoke(self, *args):
        if self.metrics is None:
            return getattr(self.adapter, name)(*args)
        start = time.perf_counter_ns()
        try:
            return getattr(self.adapter, name)(*args)
        finally:
            end = time.perf_counter_ns()
            self.metrics.observe('policy.'+name, end-start, end)
    invoke.__name__ = name
    return invoke


class MeasuredPolicy(CachePolicy):
    @property
    def chunks(self):return self.adapter.chunks
    get = _forward('get')
    insert = _forward('insert')
    remove = _forward('remove')
    clear = _forward('clear')
    mark_evictable = _forward('mark_evictable')
    mark_non_evictable = _forward('mark_non_evictable')
    touch = _forward('touch')
    on_store_miss = _forward('on_store_miss')
    on_request_finished = _forward('on_request_finished')
    evict = _forward('evict')


class MeasuredLRUCachePolicy(MeasuredPolicy):
    def __init__(self, cache_capacity):
        super().__init__(cache_capacity)
        self.adapter = LRUCachePolicy(cache_capacity)
        self.metrics = get_store()
