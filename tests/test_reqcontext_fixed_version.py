"""CPU-only evidence about the unmodified, pinned vLLM context contract.

These characterize 60da613, not the proposed API. After an approved backend
change they become historical evidence and the new API needs its own tests.
"""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest

PINNED = '60da6135b0349fac50a051e9dcec2bd5b4514400'


@pytest.fixture
def backend():
    if importlib.util.find_spec('vllm') is None:
        pytest.skip('Requires the pinned vLLM checkout; no GPU is needed')
    import vllm
    root = Path(vllm.__file__).resolve().parents[1]
    if not (root/'.git').exists():
        pytest.skip('Source checkout required for pinned-version audit')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != PINNED:
        pytest.skip('Historical audit applies only to unmodified 60da613')
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestOffloadState, RequestGroupState, OffloadingConnectorScheduler)
    from vllm.v1.kv_offload.base import ReqContext
    return RequestOffloadState, RequestGroupState, OffloadingConnectorScheduler, ReqContext


def state(backend):
    State, Group, _, Context = backend
    request = NS(request_id='r',block_hashes=[b'a'*32,b'b'*32],num_prompt_tokens=32,
                 kv_transfer_params=None,kv_hints=None)
    result = object.__new__(State)
    result.req = request
    result.config = NS(kv_group_configs=(NS(group_idx=0,hashes_per_chunk=1,tokens_per_chunk=16),))
    result.req_context = Context('r')
    result.group_states = (Group(next_stored_chunk_idx=2,num_hit_chunks=2),)
    return result


def test_update_keys_includes_gpu_hit_prefix_and_is_incremental(backend):
    s = state(backend)
    s.update_offload_keys()
    assert sorted(s.req_context._offload_key_positions.values()) == [16,32]
    assert len(s.group_states[0].offload_keys) == 2
    s.update_offload_keys()
    assert len(s.group_states[0].offload_keys) == 2


def test_context_positions_can_include_decode_without_prompt_boundary(backend):
    s = state(backend)
    s.update_offload_keys()
    s.req.block_hashes.append(b'c'*32)
    s.update_offload_keys()
    assert sorted(s.req_context._offload_key_positions.values()) == [16,32,48]
    assert s.req.num_prompt_tokens == 32
    assert not hasattr(s.req_context,'num_prompt_tokens')
    assert not hasattr(s.req_context,'get_offload_prefix_keys')


def test_untracked_finished_context_has_no_prefix_positions(backend):
    s = state(backend)
    seen = []
    manager = NS(on_new_request=lambda context:seen.append(context),
                 on_request_finished=lambda context:seen.append(context))
    result = backend[2].request_finished(NS(_req_status={},manager=manager),s.req)
    assert result == (False,None)
    assert seen[0] is seen[1]
    assert len(s.req.block_hashes) == 2
    assert seen[-1]._offload_key_positions == {}
