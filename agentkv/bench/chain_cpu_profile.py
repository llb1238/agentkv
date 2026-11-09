"""Separate profiling run; wall times here are never latency qualification."""
import argparse
import json
from pathlib import Path
import pstats

from agentkv.bench.e0 import ROOT,write_json
from agentkv.bench.profile_callbacks import compile_trace,execute
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.chain_adapter import ChainPrefixAdapter
from agentkv.provenance import git_commit


def run(output):
    code=git_commit()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    source=ROOT/'results/m5-chain-scoring-20261106/online-trace'
    steps,capacity=compile_trace(source)
    summary,_,_,profile=execute(steps,capacity,'chain',profile=True,
        actor_factory=lambda n,clock:ChainPrefixAdapter(n,clock=clock,hint_tier='online'))
    profile.dump_stats(str(output/'profile.pstats'))
    with (output/'profile.txt').open('w') as f:pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumulative').print_stats(100)
    summary.pop('evict',None)
    write_json(output/'run.json',dict(git_commit=code,status='complete',profiling=True,not_latency_qualification=True,
        ordered_recorded_victims_identical=True,summary=summary,source_sha256={p.name:sha256_file(p) for p in (source/'callback-traces').glob('*.gz')},
        files_sha256={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.output_dir)


if __name__=='__main__':main()
