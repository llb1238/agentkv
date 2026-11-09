import json
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.chain_trial import finalize, schedule_comparison
from agentkv.bench.e0 import write_json
from agentkv.bench.replay import OUTPUT_SCHEMA
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.light_metrics import LightMetrics


def test_scheduler_gate_uses_conservative_histogram_bounds():
    assert schedule_comparison([1000000, 1010000], [2990000, 3000000])['passed']
    assert not schedule_comparison([1000000, 1010000], [2990000, 3000001])['passed']


def test_hint_trial_finalizer_preserves_source_and_unsigned_seed(tmp_path):
    row = dict(request_id='client', server_request_id='server', prompt_len=16,
        prompt_token_ids=list(range(16)), permutation_seed=2**64-1, status='success',
        is_entry=False, in_window=True, send_ns=400*10**9, end_ns=401*10**9, ttft_ms=10.)
    pq.write_table(pa.Table.from_pylist([row], schema=OUTPUT_SCHEMA), tmp_path/'requests.parquet')
    write_json(tmp_path/'run.json', dict(git_commit='code', vllm_commit='backend', status='complete',
        summary=dict(hint_sent=1), instances=[], source={}, requests_sha256=sha256_file(tmp_path/'requests.parquet')))
    before = (tmp_path/'run.json').read_bytes()
    write_json(tmp_path/'server-manifest.json', dict(vllm_commit='backend', policy='HintAwareCachePolicy'))
    write_json(tmp_path/'policy-config.json', dict(implementation='chain', hint_tier='online'))
    loaded = dict(implementation='ChainPrefixAdapter', sha256=sha256_file(tmp_path/'policy-config.json'))
    (tmp_path/'server.log').write_text('AGENTKV_HINT_POLICY_CONFIG '+json.dumps(loaded)+'\n')
    (tmp_path/'dmon.log').write_text('recorded raw data\n')
    store = LightMetrics(tmp_path/'server-observations', pid=1)
    store.set_window(0, 300, 1200)
    store.record('request_received', ns=400*10**9, request_id='server')
    store.record('request_scheduled', ns=400*10**9+1000, request_id='server')
    store.record('hint_received', ns=400*10**9, request_id='server', message_id='client')
    store.record('prefill', ns=400*10**9, request_id='server', prompt_tokens=16, gpu_hit=0, cpu_hit=0, recomputed=16)
    store.observe('scheduler.schedule', 1000, 400*10**9)
    store.counters['valid_hint_requests_finished'] = 1
    store.close()
    result = finalize(tmp_path)
    assert result['hints_sent'] == result['hints_received'] == result['hints_consumed'] == 1
    assert pq.read_table(tmp_path/'request-metadata.parquet')['permutation_seed'].to_pylist() == [2**64-1]
    assert (tmp_path/'run.json').read_bytes() == before
