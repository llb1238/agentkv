"""Preregistered paired LRU / hint+none overhead trial (20 min, warmup 5 min)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.replay import join_observations
from agentkv.bench.shared_pilot import run as serve
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.light_metrics import Histogram
from agentkv.policy.prefix_adapter import ROOT
from agentkv.provenance import git_commit, current_commit


def merge_histograms(files, field='window_histograms'):
    histograms = {}
    for path in files:
        data = json.loads(path.read_text())
        if not data['closed']:raise ValueError('Missing metrics close')
        for name, values in data[field].items():
            h = histograms.setdefault(name, Histogram())
            h.buckets.update({int(k):v for k,v in values['buckets'].items()})
            h.count += values['count'];h.total += values['total_ns']
            h.maximum = max(h.maximum, values['maximum_ns'])
            minimum = values['minimum_ns']
            if minimum is not None:h.minimum = minimum if h.minimum is None else min(h.minimum, minimum)
    return {k:v.export() for k,v in histograms.items()}


def finalize(directory):
    """Join exit-only server summaries, preserving original client run bytes."""
    run = json.loads((directory/'run.json').read_text())
    manifest = json.loads((directory/'server-manifest.json').read_text())
    if run['vllm_commit'] != manifest['vllm_commit']:raise ValueError('Backend provenance differs')
    source = directory/'requests.parquet'
    if not source.exists():
        source = directory/'request-metadata.parquet'
        previous = json.loads((directory/'overhead-run.json').read_text())
        if sha256_file(source) != previous['request_metadata_sha256']:
            raise ValueError('Compact request metadata fingerprint differs')
    columns = [n for n in pq.read_schema(source).names if n != 'prompt_token_ids']
    table = pq.read_table(source,columns=columns)
    rows = table.to_pylist()
    joined = join_observations(rows, directory/'server-observations')
    window = [r for r in rows if r['in_window']]
    success = [r for r in window if r['status']=='success']
    cache = [r for r in success if r.get('gpu_hit') is not None]
    if len(cache) != len(success):raise ValueError('Unjoined nominal sources in statistical window')
    if joined['valid_hints_received'] or run['summary']['hint_sent']:raise ValueError('Unexpected hint in none control')
    # Preserve uint64 permutation seeds; inference silently chooses int64.
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),directory/'request-metadata.parquet',compression='zstd')
    files = sorted((directory/'server-observations').glob('light-*.json'))
    histograms = merge_histograms(files)
    if not histograms.get('scheduler.schedule',{}).get('count'):raise ValueError('No scheduler window histogram')
    if not histograms.get('policy.evict',{}).get('count'):raise ValueError('No policy window histogram')
    if list((directory/'server-observations').glob('*.jsonl')):raise ValueError('Per-event server log in lightweight run')
    population = hashlib.sha256(json.dumps(run['instances'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result = dict(git_commit=run['git_commit'],vllm_commit=run['vllm_commit'],status='complete',
        population_sha256=population,source=run['source'],client_run_sha256=sha256_file(directory/'run.json'),
        server_manifest_sha256=sha256_file(directory/'server-manifest.json'),
        request_metadata_sha256=sha256_file(directory/'request-metadata.parquet'),
        light_metrics_sha256={p.name:sha256_file(p) for p in files},histograms=histograms,
        all_histograms=merge_histograms(files,'histograms'),
        source_fractions={k:sum(r[k] for r in cache)/sum(r['prompt_len'] for r in cache)
                          for k in ('gpu_hit','cpu_hit','recomputed')},
        total_sent=len(rows),window_requests=len(window),window_success=len(success),
        window_failed=len(window)-len(success),observation_join=joined,ttft_ms=run['summary']['ttft_ms'],
        instrumentation='light, in-memory aggregates; one final file per process',
        none_specialization=manifest['policy']=='HintAwareCachePolicy')
    write_json(directory/'overhead-run.json',result)
    return result


def run_one(directory, policy, gpu, port):
    commit = git_commit();backend = git_commit(repository=ROOT.parent/'vllm')
    if directory.resolve().is_relative_to(ROOT):raise ValueError('Stage formal runs outside Git worktree')
    directory.parent.mkdir(parents=True,exist_ok=True)
    free = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    if dict((int(a),float(b)) for a,b in (line.split(',') for line in free.splitlines()))[gpu]<58000:
        raise ValueError('Insufficient free VRAM for unchanged config-11')
    config_path = directory.with_name(directory.name+'-policy-config.json')
    config = dict(format='agentkv.vllm_hint_policy.v1',hint_tier='none',
                  freeze_sha256=sha256_file(ROOT/'docs/m5_scoring_frozen.json'))
    write_json(config_path,config)
    klass = 'MeasuredLRUCachePolicy' if policy=='lru' else 'HintAwareCachePolicy'
    module = 'agentkv.policy.measured_policy' if policy=='lru' else 'agentkv.policy.vllm_plugin'
    serve(directory,gpu,port,duration=1200,warmup=300,rate=1.,population_seed=3,
        run_id='m5-overhead-pop3',analyze_output=False,cache_policy=klass,cache_policy_module=module,
        hint_tier='none',extra_environment=dict(AGENTKV_INSTRUMENTATION='light',
            AGENTKV_HINT_POLICY_CONFIG=str(config_path)))
    shutil.copyfile(config_path,directory/'policy-config.json')
    result = finalize(directory)
    assert result['git_commit']==commit and result['vllm_commit']==backend
    print(json.dumps(result,indent=2),flush=True)


def compare(directory):
    a,b = [json.loads((directory/p/'overhead-run.json').read_text()) for p in ('lru','hint-none')]
    if a['population_sha256']!=b['population_sha256'] or a['source']!=b['source']:
        raise ValueError('Different populations or workload fingerprints')
    differences = {p:dict(absolute_ms=b['ttft_ms'][p]-a['ttft_ms'][p],
        relative_percent=100*(b['ttft_ms'][p]/a['ttft_ms'][p]-1)) for p in ('p50','p99')}
    result = dict(git_commit=current_commit(),status='complete',
        run_commits={p:r['git_commit'] for p,r in zip(('lru','hint-none'),(a,b))},
        population_sha256=a['population_sha256'],ttft_difference=differences,
        source_difference_pp={k:100*(b['source_fractions'][k]-a['source_fractions'][k]) for k in a['source_fractions']},
        e1_overhead_gate_passed=abs(differences['p50']['relative_percent'])<=2,
        paired_runs_sha256={p:sha256_file(directory/p/'overhead-run.json') for p in ('lru','hint-none')})
    write_json(directory/'run.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,default=1);p.add_argument('--port',type=int,default=8015)
    p.add_argument('--only',choices=('lru','hint-none'))
    p.add_argument('--analyze-only',action='store_true')
    a=p.parse_args()
    if a.analyze_only:
        for policy in ('lru','hint-none'):finalize(a.output_dir/policy)
        compare(a.output_dir)
    else:
        for policy in (a.only,) if a.only else ('lru','hint-none'):
            run_one(a.output_dir/policy,policy,a.gpu_index,a.port)
        if not a.only:compare(a.output_dir)


if __name__=='__main__':main()
