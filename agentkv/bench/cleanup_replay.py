"""Compare every ordered eviction before/after chain garbage collection."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import ROOT,write_json
from agentkv.bench.profile_callbacks import compile_trace,execute
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.chain_adapter import ChainPrefixAdapter
from agentkv.policy.chain_policy import ChainHintAwarePolicy
from agentkv.policy.prefix_adapter import frozen_options
from agentkv.provenance import git_commit


def run(output,reference_commit):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    raw=subprocess.check_output(['git','show',reference_commit+':agentkv/policy/chain_policy.py'],cwd=ROOT)
    namespace={'__name__':'agentkv.policy._pre_cleanup'}
    exec(compile(raw,'<git:'+reference_commit+'>','exec'),namespace)
    before=namespace['ChainHintAwarePolicy']
    sources=[ROOT/'results/m5-chain-scoring-20261106/online-trace',
             ROOT/'results/m5-vllm-integration-20261104/probe',ROOT/'results/m5-vllm-integration-20261104/smoke',
             ROOT/'results/archive/m5-vllm-full-probe-20261104-unclosed']
    results=[]
    for source in sources:
        path=next((source/'callback-traces').glob('*.gz'))
        with gzip.open(path,'rt') as f:header=json.loads(next(f))
        native=header['policy']=='FullLRUProbePolicy'
        partial=source.name.endswith('unclosed')
        tier='none' if native else json.loads((source/'policy-config.json').read_text())['hint_tier']
        steps,capacity=compile_trace(source,allow_unclosed=partial)
        comparisons=[]
        for label,cls in [('before',before),('after',ChainHintAwarePolicy)]:
            def factory(capacity,clock):
                if native:
                    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
                    return LRUCachePolicy(capacity)
                actor=ChainPrefixAdapter(capacity,clock=clock,hint_tier=tier)
                actor.policy=cls(capacity,clock=clock,**frozen_options(tier))
                return actor
            try:
                summary,samples,evictions,_=execute(steps,capacity,label,actor_factory=factory)
            except AssertionError as error:
                write_json(output/'run.json',dict(git_commit=commit,status='decision_mismatch',source=str(source),implementation=label,error=str(error)))
                raise
            decisions=[{k:r[k] for k in ('call','n','victims_sha256','none')} for r in evictions]
            comparisons.append(decisions)
            folder=output/source.name/label;folder.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(decisions),folder/'decisions.parquet')
            pq.write_table(pa.Table.from_pylist(samples),folder/'callbacks.parquet')
            write_json(folder/'run.json',dict(git_commit=commit,reference_commit=reference_commit,status='complete',summary=summary,
                source=str(source.relative_to(ROOT)),source_sha256={p.name:sha256_file(p) for p in (source/'callback-traces').glob('*.gz')},
                files_sha256={p.name:sha256_file(p) for p in folder.glob('*.parquet')}))
        assert comparisons[0]==comparisons[1],'Before/after ordered victims differ'
        results.append(dict(source=str(source.relative_to(ROOT)),evictions=len(comparisons[0]),ordered_lists_identical=True,
            policy='native LRU' if native else tier,partial_capture=partial))
        print('PARITY',results[-1],flush=True)
    write_json(output/'run.json',dict(git_commit=commit,reference_commit=reference_commit,status='complete',results=results))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference-commit',default='6afbaee');a=p.parse_args();run(a.output_dir,a.reference_commit)


if __name__=='__main__':main()
