"""Per-request prefix cache and equivalent scorer; no runtime instrumentation."""
from types import SimpleNamespace
from agentkv.policy.prefix_adapter import PrefixPolicyAdapter,envelope,frozen_options
from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy


class OptimizedPrefixAdapter(PrefixPolicyAdapter):
    def __init__(self,capacity,*,clock,hint_tier='perfect',options=None,final_only=False):
        super().__init__(capacity,clock=clock,hint_tier=hint_tier,options=options)
        self.policy=OptimizedHintAwarePolicy(capacity,clock=clock,**(options or frozen_options(hint_tier)))
        self._requests={};self.prefix_cache_hits=0;self.prefix_enumerations=0
        self.final_only=final_only

    def clear(self):
        self._requests.clear()
        return super().clear()

    def _position_lookup(self,context):
        return context.get_offload_key_position

    def _observe_prefix(self,context,*,finished=False,finishing_prefix=False):
        key=context.req_id;entry=self._requests.get(key)
        if entry is None or entry[0]!=context.num_prompt_tokens:
            hint=envelope(context)
            if hint is not None and self.final_only:
                action=hint['actions'][0];payload=dict(action['payload'])
                payload.update(expected_gap_ms=None if payload['final'] else 0.,
                               gap_p90_ms=None if payload['final'] else 0.)
                hint=dict(hint,actions=[dict(action,payload=payload)])
            position=self._position_lookup(context)
            normalized=SimpleNamespace(req_id=key,kv_hints=hint,kv_transfer_params=None,
                get_offload_key_position=position)
            prefix=tuple(context.get_offload_prefix_keys()) if hint else ()
            if hint:
                self.prefix_enumerations+=1
                if any(not 0<position(k)<=context.num_prompt_tokens for k in prefix):
                    raise ValueError('Invalid full prefix')
                self.valid_any.add(hint['message_id'])
            entry=(context.num_prompt_tokens,normalized,prefix,hint)
            self._requests[key]=entry
        else:self.prefix_cache_hits+=1
        _,normalized,prefix,hint=entry
        getter=normalized.get_offload_key_position
        pure=type(getattr(getter,'__self__',None)) is dict and getattr(getter,'__name__',None)=='get'
        normalized._agentkv_finishing_prefix=finishing_prefix and pure
        if hint:
            self.policy.on_store_miss(prefix,normalized)
            if finished:self.valid_finished.add(hint['message_id'])
        if finished:self._requests.pop(key,None)
        return normalized
