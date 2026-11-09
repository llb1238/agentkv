"""Full-argument callback probe and frozen-policy GPU smoke, config-11."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.shared_pilot import run as run_server
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.prefix_adapter import ROOT
from agentkv.provenance import git_commit


def describe(values):
    return dict(count=len(values),p50=float(np.median(values)),p99=float(np.quantile(values,.99)),
        maximum=max(values)) if values else dict(count=0)


def run(output,kind,gpu=1,port=8014):
    commit=git_commit();backend=git_commit(repository=ROOT.parent/'vllm')
    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    if dict((int(a),float(b)) for a,b in (line.split(',') for line in free.splitlines()))[gpu]<58000:
        raise ValueError('Insufficient free memory for unchanged config-11')
    config_path=output.with_name(output.name+'-policy-config.json')
    config=dict(format='agentkv.vllm_hint_policy.v1',hint_tier='perfect',
        freeze_sha256=sha256_file(ROOT/'docs/m5_scoring_frozen.json'))
    write_json(config_path,config)
    duration=300 if kind=='probe' else 600
    policy='FullLRUProbePolicy' if kind=='probe' else 'TracedHintAwarePolicy'
    run_server(output,gpu,port,duration=duration,warmup=0,rate=.7,population_seed=0,
        run_id='m5-full-'+kind+'-seed0',analyze_output=False,cache_policy=policy,
        cache_policy_module='agentkv.policy.full_trace_plugin',hint_tier='perfect',
        extra_environment=dict(AGENTKV_CALLBACK_TRACE_DIR=str(output/'callback-traces'),
            AGENTKV_CONTEXT_TIMING='1',AGENTKV_HINT_POLICY_CONFIG=str(config_path)))
    shutil.copyfile(config_path,output/'policy-config.json')
    timing=[]
    for path in sorted((output/'server-observations').glob('*.jsonl')):
        for line in path.open():
            record=json.loads(line)
            if record['event']=='create_req_context':timing.append(record['duration_us'])
    text=(output/'server.log').read_text();evictions=[];hint_counts=[]
    for line in text.splitlines():
        for marker,destination in [('AGENTKV_EVICT ',evictions),('AGENTKV_HINT_COUNTS ',hint_counts)]:
            if marker in line:
                payload=line.split(marker,1)[1]
                destination.append(json.JSONDecoder().raw_decode(payload)[0])
    result=dict(git_commit=commit,vllm_commit=backend,kind=kind,configuration=config,
        policy_config_sha256=sha256_file(output/'policy-config.json'),duration_s=duration,
        create_req_context_us=describe(timing),evictions=len(evictions),
        core_evict_us=describe([r['core_us'] for r in evictions]),
        adapter_evict_us=describe([r['adapter_us'] for r in evictions]),
        ownership=describe([r['owned_fraction'] for r in evictions]),
        valid_hint_counts=hint_counts[-1] if hint_counts else None,
        trace_sha256={p.name:sha256_file(p) for p in sorted((output/'callback-traces').glob('*.gz'))},
        source_run_sha256=sha256_file(output/'run.json'),status='complete')
    run_metadata=json.loads((output/'run.json').read_text())
    assert run_metadata['vllm_commit']==backend==run_metadata['server']['vllm_commit']
    assert run_metadata['summary']['window_failed']==0
    if kind=='smoke':
        assert result['valid_hint_counts']['valid_finished']==run_metadata['summary']['hint_sent']
    table=pq.read_table(output/'requests.parquet');pq.write_table(table.drop(['prompt_token_ids']),output/'request-metadata.parquet')
    write_json(output/'trial-run.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--kind',choices=('probe','smoke'),required=True)
    p.add_argument('--gpu-index',type=int,default=1);p.add_argument('--port',type=int,default=8014)
    a=p.parse_args();run(a.output_dir,a.kind,a.gpu_index,a.port)


if __name__=='__main__':main()
