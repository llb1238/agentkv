"""Compare archived/current replay scheduling at 125 inflight and 12k-token prompts."""

import argparse
import asyncio
import importlib.util
import json
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .estimate_replay import simulate_requests
from .replay import OUTPUT_SCHEMA, ReplayConfig, ReplayEngine, TokenStore, percentiles
from .steady_state import RootSchedule, SessionInstance
from .synth import tokenizer_vocabulary
from .trace_loader import sha256_file
from .workload import ROOT, SCHEMA
from agentkv.provenance import git_commit


def load_legacy(path):
    helper_path=path.with_name('steady_state.py')
    helper_spec=importlib.util.spec_from_file_location('agentkv.bench._archived_steady_state',helper_path)
    helper=importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name]=helper
    helper_spec.loader.exec_module(helper)
    spec = importlib.util.spec_from_file_location('agentkv.bench._archived_replay',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Bind the old relative hint import while loading immutable evidence.
    from agentkv import hints
    alias="agentkv.bench.hints"
    previous=sys.modules.get(alias)
    sys.modules[alias]=hints
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None: sys.modules.pop(alias,None)
        else: sys.modules[alias]=previous
    # Pin the helper too: current permutation optimization must not leak into
    # the archived baseline through its relative imports.
    module.InstanceTokenPermutation=helper.InstanceTokenPermutation
    return module


class SliceTokens:
    def __init__(self, store, size):
        self.store, self.size = store, size
    def get(self, idx):
        return self.store.get(idx)[:self.size]


def fixture(metadata, concurrency, turns, prompt_len, gap, stagger, service, seed):
    groups = defaultdict(list)
    # Real metadata are in original row-group order, matching TokenStore.
    for i,r in enumerate(metadata):
        if r['prompt_len']>=prompt_len:
            groups[i//128].append(r)
    rng = np.random.default_rng(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    selected=[]
    for n in range(concurrency*turns):
        group=groups[group_ids[n%len(group_ids)]]
        selected.append(group[int(rng.integers(len(group)))])
    if len({r['trace_req_idx'] for r in selected}) != len(selected):
        # Select without replacement while keeping row groups round-robin.
        for group in groups.values(): rng.shuffle(group)
        selected=[]; offset=0
        while len(selected)<concurrency*turns:
            for gid in group_ids:
                if offset<len(groups[gid]): selected.append(groups[gid][offset])
                if len(selected)==concurrency*turns: break
            offset+=1
    rows=[]
    for n, source in enumerate(selected):
        chain,turn=n%concurrency,n//concurrency
        rows.append(dict(source,root_id='probe',chain_id=f'c{chain:03d}',chain_turn=turn,
                         prompt_len=prompt_len,max_tokens=3,
                         anchor_trace_req_idx=None,start_delay_s=chain*stagger if not turn else None,
                         gap_after_s=gap if turn<turns-1 else None,final=turn==turns-1,
                         hint_perfect_ms=None,hint_noisy_ms=None,hint_online_ms=None,hint_online_p90_ms=None))
    sends=simulate_requests(rows,.5)
    end=max(sends.values())+.5
    estimate=[dict(root_id='probe',chain_id=r['chain_id'],chain_turn=r['chain_turn'],trace_req_idx=r['trace_req_idx'],
                   est_send_s=sends[r['trace_req_idx']],est_duration_s=end) for r in rows]
    return rows,RootSchedule(rows,estimate)


async def monitor(engine, stop, samples):
    planned=engine.origin_ns
    while not stop.is_set():
        await asyncio.sleep(max(0,(planned-time.monotonic_ns())/1e9))
        wake=time.monotonic_ns(); lag=max(0,wake-planned); missed=lag//10_000_000
        samples.append(dict(planned_ns=planned,wakeup_ns=wake,elapsed_s=(wake-engine.origin_ns)/1e9,
                            lag_ms=lag/1e6,missed_checks=missed))
        planned+=(missed+1)*10_000_000


async def measure(mode, args, module, metadata, rows, root, vocab, url, out, commit):
    directory=out/mode; directory.mkdir()
    store = SliceTokens((module.TokenStore if mode=='before' else TokenStore)(args.workload,metadata),args.prompt_len)
    config=ReplayConfig(output_dir=directory,base_url=url,run_id=f'probe-{mode}',duration=args.duration,
                        warmup=args.warmup,max_inflight=args.concurrency,request_timeout=30,drain_timeout=30,
                        metrics_interval=.1,allow_dirty=args.allow_dirty)
    engine=(module.ReplayEngine if mode=='before' else ReplayEngine)(config,rows,store,{'probe':root},vocab,
                                    [SessionInstance('probe',0,0,0,False)])
    builds={}
    if mode=='before':
        old=engine.payload
        def instrument(node):
            start=time.monotonic_ns(); value=old(node); end=time.monotonic_ns()
            builds[node['entry'].request['trace_req_idx']]=(start,end)
            return value
        engine.payload=instrument
    stop=asyncio.Event(); loop_samples=[]
    task=asyncio.create_task(engine.run(sha256_file(args.workload)))
    while not hasattr(engine,'origin_ns'): await asyncio.sleep(.001)
    observer=asyncio.create_task(monitor(engine,stop,loop_samples))
    records=await task
    stop.set(); await observer
    for r in records:
        if mode=='before':
            a,b=builds[r['trace_req_idx']]
            r.update(body_build_ms=(b-a)/1e6,body_ready_ns=b)
    # Diagnostics only: avoid duplicating the large token payloads.
    table=pa.Table.from_pylist(records,schema=pa.schema([f for f in OUTPUT_SCHEMA if f.name!='prompt_token_ids']))
    pq.write_table(table,directory/'timing.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(loop_samples),directory/'loop_probe.parquet',compression='zstd')
    window=[r for r in records if r['in_window']]
    gaps=[abs(r['gap_error_ms']) for r in window if r.get('gap_error_ms') is not None]
    samples=[r for r in loop_samples if args.warmup<=r['elapsed_s']<=args.duration]
    metrics=engine.metrics
    running=[s.get('values',{}).get('vllm:num_requests_running',0) for s in metrics if args.warmup<=s['elapsed_s']<=args.duration]
    summary=dict(sent=len(records),window_requests=len(window),failed=sum(r['status']!='success' for r in records),
                 peak_client_inflight=engine.peak_inflight,server_running=percentiles(running),
                 queued=engine.queue_count,chain_gap_abs_error_ms=percentiles(gaps),
                 schedule_abs_error_ms=percentiles([abs(r['schedule_error_ms']) for r in window]),
                 body_build_ms=percentiles([r['body_build_ms'] for r in window]),
                 loop_lag_ms=percentiles([r['lag_ms'] for r in samples]),
                 loop_missed_checks=sum(r['missed_checks'] for r in samples))
    info=dict(mode=mode,summary=summary,git_commit=commit,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              source=dict(workload_sha256=sha256_file(args.workload),legacy_input_sha256=sha256_file(args.legacy)),
              artifacts_sha256={p.name:sha256_file(p) for p in directory.iterdir() if p.suffix=='.parquet'})
    (directory/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(mode,json.dumps(summary),flush=True)
    return info


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,default=ROOT/'results/m1-workload-seed0-v3/workload.parquet')
    parser.add_argument('--legacy',type=Path,default=ROOT/'results/archive/m1-online-lru-seed0-final/implementation/replay.py')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=125)
    parser.add_argument('--turns',type=int,default=6)
    parser.add_argument('--prompt-len',type=int,default=12000)
    parser.add_argument('--service-time',type=float,default=4)
    parser.add_argument('--gap',type=float,default=.3)
    parser.add_argument('--stagger',type=float,default=.002)
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--warmup',type=float,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument("--allow-dirty",action="store_true",help="DEBUG ONLY: bypass clean-worktree check")
    args=parser.parse_args(argv)
    commit=git_commit(args.allow_dirty)
    if args.output_dir.exists(): raise FileExistsError('New probe directory required')
    args.output_dir.mkdir(parents=True)
    metadata=pq.read_table(args.workload,columns=[n for n in SCHEMA.names if n!='prompt_token_ids'],use_threads=False).to_pylist()
    rows,root=fixture(metadata,args.concurrency,args.turns,args.prompt_len,args.gap,args.stagger,args.service_time,args.seed)
    (args.output_dir/'fixture.json').write_text(json.dumps(rows,indent=2)+'\n')
    vocabulary,_=tokenizer_vocabulary(Path('/data/models/Qwen3-8B'))
    module=load_legacy(args.legacy)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    log=(args.output_dir/'server.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','agentkv.bench.fake_server','--port',str(port),'--service-time',str(args.service_time)],stdout=log,stderr=subprocess.STDOUT)
    async def run():
        import aiohttp
        async with aiohttp.ClientSession() as client:
            for _ in range(100):
                if process.poll() is not None: raise RuntimeError('Fake server failed')
                try:
                    async with client.get(f'http://127.0.0.1:{port}/health') as response:
                        if response.status==200: break
                except aiohttp.ClientError: pass
                await asyncio.sleep(.05)
            else: raise TimeoutError('Fake server readiness')
        before=await measure('before',args,module,metadata,rows,root,vocabulary,f'http://127.0.0.1:{port}',args.output_dir,commit)
        after=await measure('after',args,module,metadata,rows,root,vocabulary,f'http://127.0.0.1:{port}',args.output_dir,commit)
        p99=after['summary']['chain_gap_abs_error_ms']['p99']
        report=dict(before=before['summary'],after=after['summary'],
                    legacy_tail_reproduced=before['summary']['chain_gap_abs_error_ms']['p99']>=50,
                    fixed_p99_below_50ms=p99 is not None and p99<50,
                    concurrency_reached=after['summary']['peak_client_inflight']>=120,
                    fixture_sha256=sha256_file(args.output_dir/'fixture.json'))
        report['passed']=all(report[k] for k in ('legacy_tail_reproduced','fixed_p99_below_50ms','concurrency_reached')) and not before['summary']['failed'] and not after['summary']['failed']
        (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    try: asyncio.run(run())
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


if __name__=='__main__': main()
