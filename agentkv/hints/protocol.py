"""Design section 4.5 hint envelope, carried in kv_transfer_params."""

import math


def validate_envelope(envelope):
    if envelope.get("protocol_version") != 1 or not isinstance(envelope.get("message_id"), str):
        raise ValueError("Invalid hint envelope")
    actions = envelope.get("actions")
    if not isinstance(actions, list) or len(actions) != 1:
        raise ValueError("Expected one tool gap action")
    action = actions[0]
    if (action.get("action_type") != "agentkv.tool_gap" or action.get("action_version") != 1
            or not isinstance(action.get("action_id"), str)):
        raise ValueError("Invalid tool gap action")
    payload = action["payload"]
    if not isinstance(payload["final"], bool) or not isinstance(payload["turn"], int):
        raise ValueError("Invalid hint payload")
    for field in ("expected_gap_ms", "gap_p90_ms"):
        value = payload[field]
        if value is None:
            if not payload["final"]:
                raise ValueError("Non-final hint must have a gap")
        elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid numeric hint")
    return envelope


def make_envelope(request, tier, request_id, instance_name):
    if tier == "none":
        return None
    if tier not in ("perfect", "noisy", "online"):
        raise ValueError("Unknown hint tier")
    expected = request[f"hint_{tier}_ms"]
    p90 = (request["hint_online_p90_ms"] if tier == "online"
           else None if expected is None else expected * 2)
    return validate_envelope(dict(
        protocol_version=1, message_id=request_id,
        actions=[dict(action_id="gap-0", action_type="agentkv.tool_gap", action_version=1,
                      payload=dict(session_id=f"{instance_name}/{request['chain_id']}",
                                   turn=request["chain_turn"], parent_session=instance_name,
                                   tool_name=None, expected_gap_ms=expected,
                                   gap_p90_ms=p90, final=request["final"]))],
    ))
