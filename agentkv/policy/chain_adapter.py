"""Existing cached prefix normalization with the chain-indexed scorer."""
from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
from agentkv.policy.chain_policy import ChainHintAwarePolicy
from agentkv.policy.prefix_adapter import frozen_options


class ChainPrefixAdapter(OptimizedPrefixAdapter):
    def __init__(self, capacity, *, clock, hint_tier='perfect', options=None):
        super().__init__(capacity, clock=clock, hint_tier=hint_tier, options=options)
        self.policy = ChainHintAwarePolicy(capacity, clock=clock, **(options or frozen_options(hint_tier)))
        self._position_contexts = {}
        self._prefix_keysets = {}

    def _position_lookup(self,context):
        # The pinned ReqContext accessor is exactly this dict.get. Other duck
        # types retain their public method; no dependency on importing vLLM.
        if (type(context).__module__ == 'vllm.v1.kv_offload.base' and type(context).__name__ == 'ReqContext'
                and isinstance(getattr(context,'_offload_key_positions',None),dict)):
            return context._offload_key_positions.get
        return super()._position_lookup(context)

    def _observe_prefix(self,context,*,finished=False):
        key=context.req_id;entry=self._requests.get(key)
        if entry is None or entry[0]!=context.num_prompt_tokens:
            self._position_contexts[key]=context
            self._prefix_keysets.pop(key,None)
        else:
            # Preserve the original first-context binding, even if subsequent
            # real contexts differ; replay may replace that context's table.
            entry[1].get_offload_key_position=self._position_lookup(self._position_contexts[key])
            entry[1]._agentkv_observed_prefix=None
        result=super()._observe_prefix(context,finished=False,finishing_prefix=finished)
        result._agentkv_finishing_prefix=False
        _,_,prefix,hint=self._requests[key]
        # Only a pure dict lookup can justify the intra-callback idempotence;
        # arbitrary duck-typed getters retain the original observation path.
        getter=result.get_offload_key_position
        if hint and type(getattr(getter,'__self__',None)) is dict and getattr(getter,'__name__',None)=='get':
            known=self._prefix_keysets.get(key)
            if known is None:
                known=frozenset(prefix);self._prefix_keysets[key]=known
            result._agentkv_observed_prefix=known
        if finished:
            if hint:self.valid_finished.add(hint['message_id'])
            self._requests.pop(key,None)
            self._position_contexts.pop(key,None)
            self._prefix_keysets.pop(key,None)
        return result

    def clear(self):
        self._position_contexts.clear()
        self._prefix_keysets.clear()
        return super().clear()
