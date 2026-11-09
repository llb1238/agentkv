"""Ordered decision parity and uninstrumented adapter timings on real traces."""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.e0 import write_json
from agentkv.bench.profile_callbacks import compile_trace, execute, describe
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.chain_adapter import ChainPrefixAdapter
from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
from agentkv.policy.prefix_adapter import ROOT
from agentkv.provenance import git_commit


def run(source, output, tier, implementation='chain'):
    commit = git_commit()
    backend = git_commit(repository=ROOT.parent / 'vllm')
    trial = json.loads((source / 'trial-run.json').read_text())
    baseline_dir = ROOT / 'results/m5-chain-scoring-20261106' / (tier + '-check')
    baseline = json.loads((baseline_dir / 'run.json').read_text())
    assert trial['trace_sha256'] == baseline['trace_sha256']
    assert backend == trial['vllm_commit'] == baseline['vllm_commit']
    assert json.loads((source / 'policy-config.json').read_text())['hint_tier'] == tier
    for name, digest in trial['trace_sha256'].items():
        assert sha256_file(source / 'callback-traces' / name) == digest
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    provenance = dict(git_commit=commit, vllm_commit=backend, hint_tier=tier,
        implementation=implementation, trace_sha256=trial['trace_sha256'],
        baseline_run_sha256=sha256_file(baseline_dir / 'run.json'))
    steps, capacity = compile_trace(source)
    adapter = ChainPrefixAdapter if implementation != 'v1' else OptimizedPrefixAdapter
    previous = None
    def factory(capacity, clock):
        nonlocal previous
        if implementation != 'chain-plugin':
            return adapter(capacity, clock=clock, hint_tier=tier)
        from agentkv.policy import vllm_plugin
        previous = (vllm_plugin.CLOCK, os.environ.get('AGENTKV_INSTRUMENTATION'), os.environ.get('AGENTKV_HINT_POLICY_CONFIG'))
        config = output / 'policy-config.json'
        write_json(config, dict(format='agentkv.vllm_hint_policy.v1', hint_tier=tier, implementation='chain',
            freeze_sha256=sha256_file(ROOT / 'docs/m5_scoring_frozen.json'),
            residual_table_path=str(ROOT / 'agentkv/policy/data/remaining_life_v1.json')))
        os.environ['AGENTKV_INSTRUMENTATION'] = 'off'
        os.environ['AGENTKV_HINT_POLICY_CONFIG'] = str(config)
        vllm_plugin.CLOCK = clock
        return vllm_plugin.HintAwareCachePolicy(capacity)
    try:
        summary, samples, evictions, _ = execute(steps, capacity, implementation,
            actor_factory=factory)
        assert summary['eviction_fingerprint'] == baseline['eviction_fingerprint']
        decisions = [{k: e[k] for k in ('call', 'n', 'victims_sha256', 'none')} for e in evictions]
        expected = pq.read_table(baseline_dir / 'decisions.parquet').to_pylist()
        assert decisions == expected, 'Per-call baseline ordered victims differ'
    except AssertionError as error:
        write_json(output / 'run.json', dict(provenance, status='decision_mismatch', error=str(error)))
        raise
    finally:
        if previous is not None:
            from agentkv.policy import vllm_plugin
            vllm_plugin.CLOCK = previous[0]
            for key, value in zip(('AGENTKV_INSTRUMENTATION', 'AGENTKV_HINT_POLICY_CONFIG'), previous[1:]):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    callbacks = {method: describe([r['elapsed_us'] for r in samples if r['method'] == method])
                 for method in summary['counts']}
    qualified = all(callbacks[k]['p99_us'] <= 2000 for k in ('on_request_finished', 'evict'))
    pq.write_table(pa.Table.from_pylist(samples), output / 'callbacks.parquet')
    pq.write_table(pa.Table.from_pylist(decisions), output / 'decisions.parquet')
    result = dict(provenance, status='complete', summary=summary, callbacks=callbacks,
        all_recorded_ordered_keys_identical=True, all_baseline_ordered_keys_identical=True,
        instrumentation='off; decode outside callback timing; no cProfile',
        latency_gate_2ms=qualified, callbacks_sha256=sha256_file(output / 'callbacks.parquet'),
        decisions_sha256=sha256_file(output / 'decisions.parquet'))
    write_json(output / 'run.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--hint-tier', choices=('online', 'perfect'), required=True)
    p.add_argument('--implementation', choices=('v1', 'chain', 'chain-plugin'), default='chain')
    a = p.parse_args()
    run(a.run_dir, a.output_dir, a.hint_tier, a.implementation)


if __name__ == '__main__':
    main()
