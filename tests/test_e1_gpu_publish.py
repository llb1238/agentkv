from agentkv.analysis.e1_gpu_publish import ignored_payloads, primary_values


def test_only_large_online_payloads_and_process_handles_are_omitted():
    files = ['requests.parquet', 'request-metadata.parquet', 'run.json', 'controller.lock', 'controller.pid']
    assert ignored_payloads('/staging/attempt-2', files) == ['requests.parquet', 'controller.lock', 'controller.pid']
    assert ignored_payloads('/staging/simulation/lru', files) == ['controller.lock', 'controller.pid']


def test_primary_recalculation_excludes_failed_and_outside_window_requests():
    rows = [dict(in_window=True, status='success', prompt_len=16, prefill_forward_token_work=8, ttft_ms=10),
            dict(in_window=True, status='success', prompt_len=48, prefill_forward_token_work=1, ttft_ms=30),
            dict(in_window=True, status='failed', prompt_len=16, prefill_forward_token_work=None, ttft_ms=None),
            dict(in_window=False, status='success', prompt_len=1000, prefill_forward_token_work=1000, ttft_ms=999)]
    result = primary_values(rows)
    assert result['requests'] == 2 and result['effective_recompute'] == 9 / 64
    assert result['p50'] == 20 and result['p99'] == 29.8
