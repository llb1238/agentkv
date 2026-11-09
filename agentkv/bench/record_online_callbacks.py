"""Record a real online-policy trace for decision and CPU latency regression."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from agentkv.bench.e0 import write_json
from agentkv.bench.shared_pilot import run as serve
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.prefix_adapter import ROOT
from agentkv.provenance import git_commit


def validate_trace(paths):
    opened = closed = calls = 0
    for path in paths:
        counts = {}
        with gzip.open(path, 'rt') as f:
            for line in f:
                e = json.loads(line)
                if e['event'] == 'open':
                    opened += 1
                    assert 'TracedHintAwarePolicy' in e['policy']
                elif e['event'] == 'begin':
                    counts[e['method']] = counts.get(e['method'], 0) + 1
                    calls += 1
                elif e['event'] == 'error':
                    raise AssertionError(('Recorded policy exception', e))
                elif e['event'] == 'close':
                    assert e['counts'] == counts and e['calls'] == sum(counts.values())
                    closed += 1
    assert opened == closed == 1, 'Missing complete trace footer'
    return calls


def record(output, gpu=1, port=8017):
    commit = git_commit()
    backend = git_commit(repository=ROOT.parent / 'vllm')
    assert backend == '4749dd5961841243596559b10ba0a22c48980c45'
    if output.exists():
        raise FileExistsError(output)
    config = dict(format='agentkv.vllm_hint_policy.v1', hint_tier='online',
        freeze_sha256=sha256_file(ROOT / 'docs/m5_scoring_frozen.json'),
        residual_table_path=str(ROOT / 'agentkv/policy/data/remaining_life_v1.json'))
    config_path = output.with_name(output.name + '-policy-config.json')
    write_json(config_path, config)
    serve(output, gpu, port, duration=300, warmup=0, rate=1., population_seed=3,
        run_id='m5-online-callbacks-seed3', analyze_output=False,
        cache_policy='TracedHintAwarePolicy', cache_policy_module='agentkv.policy.full_trace_plugin',
        hint_tier='online', extra_environment=dict(AGENTKV_INSTRUMENTATION='light',
            AGENTKV_CALLBACK_TRACE_DIR=str(output / 'callback-traces'), AGENTKV_CONTEXT_TIMING='1',
            AGENTKV_HINT_POLICY_CONFIG=str(config_path), PYTHONHASHSEED='0'))
    shutil.copyfile(config_path, output / 'policy-config.json')
    traces = sorted((output / 'callback-traces').glob('*.gz'))
    calls = validate_trace(traces)
    run = json.loads((output / 'run.json').read_text())
    assert run['status'] == 'complete' and run['summary']['window_failed'] == 0
    lights = [json.loads(p.read_text()) for p in (output / 'server-observations').glob('light-*.json')]
    assert lights and all(v['closed'] for v in lights)
    finished = sum(v['counters'].get('valid_hint_requests_finished', 0) for v in lights)
    assert finished == run['summary']['hint_sent']
    columns = [n for n in pq.read_schema(output / 'requests.parquet').names if n != 'prompt_token_ids']
    pq.write_table(pq.read_table(output / 'requests.parquet', columns=columns), output / 'request-metadata.parquet')
    result = dict(git_commit=commit, vllm_commit=backend, status='complete', hint_tier='online',
        duration_s=300, warmup_s=0, arrival_rate=1., population_seed=3,
        trace_calls=calls, valid_hint_requests_finished=finished,
        trace_sha256={p.name: sha256_file(p) for p in traces},
        source_run_sha256=sha256_file(output / 'run.json'),
        request_metadata_sha256=sha256_file(output / 'request-metadata.parquet'),
        policy_config_sha256=sha256_file(output / 'policy-config.json'))
    write_json(output / 'trial-run.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, default=1)
    p.add_argument('--port', type=int, default=8017)
    a = p.parse_args()
    record(a.output_dir, a.gpu_index, a.port)


if __name__ == '__main__':
    main()
