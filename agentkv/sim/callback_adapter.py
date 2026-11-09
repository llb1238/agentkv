"""CPU simulator entry for recorded serving callbacks, separate from E1 model."""
from agentkv.policy.prefix_adapter import PrefixPolicyAdapter


class CallbackPolicyAdapter(PrefixPolicyAdapter):
    """Use actual incremental/offload callback inputs, not completion writes."""
