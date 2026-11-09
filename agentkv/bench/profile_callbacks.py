"""CPU-only uninstrumented callback timings and decision fingerprints.

Decode trace records outside measured callbacks. cProfile is a separate run:
its timings are not used to judge the latency thresholds.
"""
import argparse
from collections import Counter, deque
from contextlib import nullcontext
import cProfile
import gzip
import hashlib
import json
import os
from pathlib import Path
import pstats
import time
import zlib
from types import SimpleNamespace as NS

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.prefix_adapter import PrefixPolicyAdapter, ROOT
from agentkv.provenance import git_commit


def trace_lines(path,allow_unclosed=False):
    if not allow_unclosed:
        with gzip.open(path,'rt') as source:yield from source
        return
    # Recover complete JSON lines from a known, archived truncated gzip only.
    # Never change its bytes or advertise an absent footer as a full capture.
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS);pending=b''
    with path.open('rb') as source:
        while chunk:=source.read(65536):
            pending+=decoder.decompress(chunk)
            lines=pending.split(b'\n');pending=lines.pop()
            for line in lines:
                if line:yield line.decode('utf-8')


def compile_trace(directory, *, mask_hints=False, allow_unclosed=False):
    from vllm.v1.kv_offload.base import ReqContext
    keys={};contexts={};canonical={};frames={};steps=[];capacity=None;closed=0;last_complete=0
    for path in sorted((directory/'callback-traces').glob('*.gz')):
        with nullcontext(trace_lines(path,allow_unclosed)) as source:
            for line in source:
                e=json.loads(line);kind=e['event']
                if kind=='open':capacity=e['capacity']
                elif kind=='key':keys[e['key']]=bytes.fromhex(e['hex'])
                elif kind=='context':
                    req=e['req_id']
                    if req not in canonical:
                        canonical[req]=ReqContext(req,kv_transfer_params=e['kv_transfer_params'],
                            kv_hints=e['kv_hints'],num_prompt_tokens=e['num_prompt_tokens'])
                    contexts[e['context']]=canonical[req]
                    steps.append(('context',canonical[req],[(keys[k],p) for k,p in e['positions']]))
                elif kind=='begin':e['clocks']=[];frames[e['call']]=e
                elif kind=='clock':
                    if e['call'] is not None:frames[e['call']]['clocks'].append(e['value'])
                elif kind=='manager_refs':steps.append(('refs',[(keys[k],v) for k,v in e['refs']]))
                elif kind=='close':closed+=1
                elif kind=='end':
                    start=frames.pop(e['call'])
                    if start['parent'] is not None:continue
                    method=start['method'];a=start['args']
                    if method in ('get','remove','mark_evictable','mark_non_evictable'):args=(keys[a[0]],)
                    elif method=='insert':args=(keys[a[0]],tuple(a[1]))
                    elif method in ('touch','on_store_miss'):args=([keys[k] for k in a[0]],contexts[a[1]])
                    elif method=='on_request_finished':args=(tuple(tuple(keys[k] for k in g) for g in a[0]),
                        {keys[k] for k in a[1]},{keys[k] for k in a[2]},contexts[a[3]])
                    elif method=='evict':args=(a[0],{keys[k] for k in a[1]})
                    else:args=()
                    expected=None if e['result'] is None else (
                        tuple(keys[k] for k,_ in e['result']) if method=='evict' else e['result'])
                    steps.append(('call',method,args,[(keys[k],v) for k,v in start['refs']],
                                  start['clocks'],start['ns']/1e9,expected,start['call']))
                    last_complete=len(steps)
    if allow_unclosed:
        assert closed in (0,1)
        steps=steps[:last_complete]
    else:
        assert closed==1 and not frames
    if mask_hints:
        for context in canonical.values():
            context.kv_hints=None;context.kv_transfer_params=None
    return steps,capacity


def describe(values):
    return dict(count=len(values),p50_us=float(np.median(values)),p99_us=float(np.quantile(values,.99)),
                maximum_us=float(max(values))) if values else dict(count=0)


def execute(steps,capacity,implementation,*,profile=False,profile_calls=None,actor_factory=None):
    clocks=deque();fallback=0.
    def clock():return clocks.popleft() if clocks else fallback
    if actor_factory is not None:
        actor=actor_factory(capacity,clock)
    elif implementation=='vllm':
        from agentkv.policy import vllm_plugin
        vllm_plugin.CLOCK=clock
        os.environ['AGENTKV_INSTRUMENTATION']='off'
        actor=vllm_plugin.HintAwareCachePolicy(capacity)
    elif implementation=='native':
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
        actor=LRUCachePolicy(capacity)
    elif implementation=='none':
        from agentkv.policy.measured_policy import NoneHintAdapter
        actor=NoneHintAdapter(capacity)
    elif implementation=='reference':actor=PrefixPolicyAdapter(capacity,clock=clock)
    elif implementation=='indexed':
        from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy
        actor=PrefixPolicyAdapter(capacity,clock=clock)
        actor.policy=OptimizedHintAwarePolicy(capacity,clock=clock,scoring='deadline',timeout_decay=True)
    elif implementation=='cached':
        from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
        from agentkv.policy.hint_aware import HintAwarePolicy
        actor=OptimizedPrefixAdapter(capacity,clock=clock)
        actor.policy=HintAwarePolicy(capacity,clock=clock,scoring='deadline',timeout_decay=True)
    else:
        from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
        actor=OptimizedPrefixAdapter(capacity,clock=clock)
    bridge=getattr(actor,'adapter',actor)
    core=getattr(bridge,'policy',bridge)
    profiler=cProfile.Profile() if profile else None
    samples=[];fingerprint=hashlib.sha256();evictions=[];counts=Counter()
    for step in steps:
        if step[0]=='context':
            _,ctx,positions=step
            ctx._offload_key_positions=dict(positions)
            continue
        if step[0]=='refs':
            for key,value in step[1]:
                chunk=actor.chunks.get(key)
                assert (chunk is None)==(value is None)
                if chunk is not None:chunk.ref_cnt,chunk.chunk_id=value
            continue
        _,method,args,refs,values,fallback,expected,call_id=step
        for key,value in refs:
            chunk=actor.chunks.get(key)
            assert (chunk is None)==(value is None),(method,call_id)
            if chunk is not None:chunk.ref_cnt,chunk.chunk_id=value
        clocks=deque(values)
        if method=='insert':args=(args[0],NS(ref_cnt=args[1][0],chunk_id=args[1][1]))
        callback=getattr(actor,method)
        active=profiler and (profile_calls is None or call_id in profile_calls)
        if active:profiler.enable()
        start=time.perf_counter_ns();result=callback(*args);elapsed=time.perf_counter_ns()-start
        if active:profiler.disable()
        counts[method]+=1
        if method=='evict':
            victims=None if result is None else tuple(k for k,_ in result)
            if victims!=expected:
                print('DECISION_MISMATCH',call_id,'expected',expected,'actual',victims,flush=True)
                for key in expected or ():
                    if key in actor.chunks:
                        print('REMAINING_EXPECTED',key,core._lines(key),
                            getattr(core,'_entries',{}).get(key),
                            getattr(core,'_signatures',{}).get(key),
                            [(c,vars(core.chains[c]) | {'keys':'omitted','layers':'omitted','depths':'omitted'})
                             for c in core.owners.get(key,{})],flush=True)
                        break
                raise AssertionError(('Eviction changed',call_id))
            digest=hashlib.sha256(b''.join(victims or ())).hexdigest()
            fingerprint.update(str(call_id).encode()+b':'+(b'None' if victims is None else b'|'.join(victims)))
            row=dict(call=call_id,n=args[0],elapsed_us=elapsed/1e3,victims_sha256=digest,
                     none=victims is None,**getattr(core,'work_counters',{}))
            evictions.append(row)
        if not profile:samples.append(dict(method=method,call=call_id,elapsed_us=elapsed/1e3))
        assert not clocks,('Clock sequence changed',method,call_id)
    return dict(counts=dict(counts),eviction_fingerprint=fingerprint.hexdigest(),
        prefix_enumerations=getattr(bridge,'prefix_enumerations',None),
        prefix_cache_entries=len(getattr(bridge,'_requests',{})),
        evict=describe([e['elapsed_us'] for e in evictions])),samples,evictions,profiler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=ROOT/'results/m5-vllm-integration-20261104/smoke')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--implementation',choices=('reference','optimized','indexed','cached','native','none','vllm'),default='reference')
    p.add_argument('--mask-hints',action='store_true',help='Mask hints on the recorded native-LRU path for no-hint equivalence')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--profile-calls',help='Comma-separated call IDs for longest-call diagnosis')
    a=p.parse_args()
    commit=git_commit();backend=git_commit(repository=ROOT.parent/'vllm')
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if a.implementation=='vllm':
        config_path=a.output_dir.with_name(a.output_dir.name+'-config.json')
        config_path.write_text(json.dumps(dict(format='agentkv.vllm_hint_policy.v1',hint_tier='perfect',
            freeze_sha256=sha256_file(ROOT/'docs/m5_scoring_frozen.json')))+'\n')
        os.environ['AGENTKV_HINT_POLICY_CONFIG']=str(config_path)
    steps,capacity=compile_trace(a.run_dir,mask_hints=a.mask_hints)
    ids={int(s) for s in a.profile_calls.split(',')} if a.profile_calls else None
    profiling=a.profile or ids is not None
    summary,samples,evictions,profiler=execute(steps,capacity,a.implementation,profile=profiling,profile_calls=ids)
    a.output_dir.mkdir(parents=True)
    if samples:pq.write_table(pa.Table.from_pylist(samples),a.output_dir/'callbacks.parquet')
    pq.write_table(pa.Table.from_pylist(evictions),a.output_dir/'evictions.parquet')
    if profiler:
        profiler.dump_stats(str(a.output_dir/'profile.pstats'))
        with (a.output_dir/'profile.txt').open('w') as target:
            pstats.Stats(profiler,stream=target).strip_dirs().sort_stats('cumulative').print_stats(65)
        functions=[]
        for (file,line,name),(cc,nc,tt,ct,callers) in pstats.Stats(profiler).stats.items():
            if name in ('get_offload_prefix_keys','_compact','_advance','_refresh','evict','_observe'):
                functions.append(dict(file=file,line=line,name=name,calls=nc,self_s=tt,cumulative_s=ct))
        summary['profile_functions']=functions
    write_json(a.output_dir/'run.json',dict(git_commit=commit,vllm_commit=backend,
        implementation=a.implementation,mask_hints=a.mask_hints,cprofile=profiling,profile_calls=sorted(ids) if ids else None,summary=summary,
        trace_sha256={p.name:sha256_file(p) for p in (a.run_dir/'callback-traces').glob('*.gz')}))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
