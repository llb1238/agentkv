"""Native LRU and frozen HintAware policies with complete call recording."""
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from agentkv.policy.callback_trace import CallbackTraceMixin
from agentkv.policy.vllm_plugin import HintAwareCachePolicy


class FullLRUProbePolicy(CallbackTraceMixin,LRUCachePolicy):
    pass


class TracedHintAwarePolicy(CallbackTraceMixin,HintAwareCachePolicy):
    pass
