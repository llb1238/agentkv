"""Three paired 20-minute GPU overhead gates, with persistent resumable output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.e0 import write_json
from agentkv.bench.policy_overhead import merge_histograms
from agentkv.bench.replay import join_observations, percentiles
from agentkv.bench.shared_pilot import run as serve
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.prefix_adapter import ROOT
from agentkv.provenance import git_commit


def schedule_comparison(base, interval):
    upper = (interval[1] - base[0]) / 1e6
    return dict(schedule_p99_ns=interval, delta_upper_ms=upper, passed=upper <= 2.)


def finalize(directory):
    run = json.loads((directory / 'run.json').read_text())
    server = json.loads((directory / 'server-manifest.json').read_text())
    assert server['vllm_commit'] == run['vllm_commit']
    assert sha256_file(directory / 'requests.parquet') == run['requests_sha256']
    if server['policy'] != 'lru':
        marker = 'AGENTKV_HINT_POLICY_CONFIG '
        loaded = [json.JSONDecoder().raw_decode(s.split(marker, 1)[1])[0]
                  for s in (directory / 'server.log').read_text().splitlines() if marker in s]
        assert len(loaded) == 1 and loaded[0]['implementation'] == 'ChainPrefixAdapter'
        assert loaded[0]['sha256'] == sha256_file(directory / 'policy-config.json')
    schema = pq.read_schema(directory / 'requests.parquet')
    table = pq.read_table(directory / 'requests.parquet', columns=[k for k in schema.names if k != 'prompt_token_ids'])
    rows = table.to_pylist()
    joined = join_observations(rows, directory / 'server-observations')
    sent = run['summary']['hint_sent']
    assert joined['valid_hints_received'] == sent
    assert run['status'] == 'complete' and all(r['status'] == 'success' for r in rows)
    files = sorted((directory / 'server-observations').glob('light-*.json'))
    assert files and all(json.loads(f.read_text())['closed'] for f in files)
    consumed = sum(json.loads(f.read_text())['counters'].get('valid_hint_requests_finished', 0) for f in files)
    if server['policy'] != 'lru':
        assert consumed == sent
    histograms = merge_histograms(files)
    assert histograms['scheduler.schedule']['count']
    kept = [r for r in rows if r['in_window'] and r['status'] == 'success']
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), directory / 'request-metadata.parquet')
    result = dict(git_commit=run['git_commit'], vllm_commit=run['vllm_commit'], status='complete',
        histograms=histograms, ttft_ms=percentiles([r['ttft_ms'] for r in kept]),
        window_requests=len(kept), hints_sent=sent, hints_received=joined['valid_hints_received'],
        hints_consumed=consumed if server['policy'] != 'lru' else None,
        instances_sha256=hashlib.sha256(json.dumps(run['instances'], sort_keys=True).encode()).hexdigest(),
        source=run['source'], source_run_sha256=sha256_file(directory / 'run.json'),
        raw_requests_sha256=sha256_file(directory / 'requests.parquet'),
        server_manifest_sha256=sha256_file(directory / 'server-manifest.json'),
        policy_config_sha256=sha256_file(directory / 'policy-config.json'),
        dmon_sha256=sha256_file(directory / 'dmon.log'),
        metadata_sha256=sha256_file(directory / 'request-metadata.parquet'),
        metrics_sha256={f.name: sha256_file(f) for f in files})
    write_json(directory / 'overhead-run.json', result)
    return result


def run(output, gpu=1, port=8018, *, labels=('lru','online','perfect'),duration=1200,warmup=300,seed=3):
    commit = git_commit()
    backend = git_commit(repository=ROOT.parent / 'vllm')
    assert backend == '4749dd5961841243596559b10ba0a22c48980c45'
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    if labels[0]!='lru' or len(set(labels))!=len(labels):raise ValueError('One paired LRU baseline must come first')
    for label in labels:
        directory = output / label
        record = directory / 'overhead-run.json'
        if record.exists():
            r = json.loads(record.read_text())
            assert r['git_commit'] == commit and sha256_file(directory / 'run.json') == r['source_run_sha256']
            assert sha256_file(directory / 'request-metadata.parquet') == r['metadata_sha256']
            results[label] = r
            continue
        if directory.exists():
            raise RuntimeError(f'Incomplete prior run; investigate without overwriting: {directory}')
        tier = 'perfect' if label == 'perfect' else 'online'
        config = dict(format='agentkv.vllm_hint_policy.v1', hint_tier=tier, implementation='chain',
            freeze_sha256=sha256_file(ROOT / 'docs/m5_scoring_frozen.json'),
            residual_table_path=str(ROOT / 'agentkv/policy/data/remaining_life_v1.json'))
        path = output / (label + '-policy-config.json')
        write_json(path, config)
        print('BEGIN', label, flush=True)
        serve(directory, gpu, port, duration=duration, warmup=warmup, rate=1., population_seed=seed,
            run_id=f'm5-chain-overhead-pop{seed}', analyze_output=False,
            cache_policy='lru' if label == 'lru' else 'HintAwareCachePolicy',
            cache_policy_module=None if label == 'lru' else 'agentkv.policy.vllm_plugin', hint_tier=tier,
            extra_environment=dict(AGENTKV_INSTRUMENTATION='light', AGENTKV_EFFECTIVE_PREFILL='1',
                AGENTKV_BUILTIN_LRU_TIMING='1' if label == 'lru' else '0',
                AGENTKV_HINT_POLICY_CONFIG=str(path), PYTHONHASHSEED='0'))
        shutil.copyfile(path, directory / 'policy-config.json')
        results[label] = finalize(directory)
        assert results[label]['git_commit'] == commit
        print('DONE', label, results[label]['ttft_ms'], flush=True)
    base = results['lru']['histograms']['scheduler.schedule']['p99_ns']
    comparisons = {}
    reference = None
    for label, r in results.items():
        assert r['instances_sha256'] == results['lru']['instances_sha256'] and r['source'] == results['lru']['source']
        cols = ['request_id', 'sent_prompt_sha256', 'permutation_seed', 'prompt_len', 'max_tokens']
        tokens = {row['request_id']: tuple(row[k] for k in cols[1:]) for row in
                  pq.read_table(output / label / 'request-metadata.parquet', columns=cols).to_pylist()}
        if reference is None:
            reference = tokens
        else:
            assert all(reference[k] == tokens[k] for k in reference.keys() & tokens.keys())
            reference.update(tokens)
        interval = r['histograms']['scheduler.schedule']['p99_ns']
        comparisons[label] = schedule_comparison(base, interval)
        if label == 'lru':
            comparisons[label].update(delta_upper_ms=0., passed=True)
        comparisons[label]['ttft_p50_delta_ms'] = r['ttft_ms']['p50'] - results['lru']['ttft_ms']['p50']
    result = dict(git_commit=commit, vllm_commit=backend, status='complete', results=results,
        parameters=dict(duration_s=duration,warmup_s=warmup,seed=seed,rate=1.,labels=list(labels)),
        comparisons=comparisons, passed=all(r['passed'] for r in comparisons.values()))
    write_json(output / 'run.json', result)
    print(json.dumps(dict(passed=result['passed'], comparisons=comparisons), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, default=1)
    p.add_argument('--port', type=int, default=8018)
    p.add_argument('--labels',nargs='+',choices=('lru','online','perfect'),default=['lru','online','perfect'])
    p.add_argument('--duration',type=int,default=1200);p.add_argument('--warmup',type=int,default=300)
    p.add_argument('--population-seed',type=int,default=3)
    a = p.parse_args()
    run(a.output_dir,a.gpu_index,a.port,labels=a.labels,duration=a.duration,warmup=a.warmup,seed=a.population_seed)


if __name__ == '__main__':
    main()
