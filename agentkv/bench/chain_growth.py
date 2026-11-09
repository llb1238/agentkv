"""Read-only minute snapshots and per-call timings of the deployed chain scorer."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.e0 import ROOT,WORKLOAD,SCHEDULE,load_inputs,population_hash,write_json
from agentkv.bench.e1_simulation import _schedule_fingerprint
from agentkv.bench.steady_state import sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.chain_policy import ChainHintAwarePolicy
from agentkv.policy.prefix_adapter import frozen_options
from agentkv.provenance import git_commit
from agentkv.sim.hint_adapter import HintPolicyAdapter
from agentkv.sim.two_tier import simulate


def snapshot(policy,t):
    chains=policy.chains
    def valid(event):
        _,_,kind,key,version=event;c=chains.get(key)
        if kind=='life':return c is not None and (c.generation,c.version)==version and c.mode=='waiting'
        if kind=='chain':return c is not None and c.version==version and c.mode=='waiting' and not c.expired
        return policy._versions.get(key)==version
    ranks=[item for q in policy._rank_lists.values() for item in q]
    return dict(time_s=t,chains=len(chains),chains_with_blocks=sum(bool(c.keys) for c in chains.values()),
        empty_chains=sum(not c.keys for c in chains.values()),final_retained=sum(c.mode=='final' for c in chains.values()),
        final_empty=sum(c.mode=='final' and not c.keys for c in chains.values()),
        overdue_retained=sum(c.mode=='waiting' and t>c.deadline+c.margin for c in chains.values()),
        overdue_empty=sum(c.mode=='waiting' and t>c.deadline+c.margin and not c.keys for c in chains.values()),
        cpu_blocks=len(policy.chunks),chain_events=len(policy._events),stale_chain_events=sum(not valid(e) for e in policy._events),
        block_events=len(policy._block_events),cancelled_events=len(policy._cancelled_times),
        priority_heap_entries=sum(len(q) for q in policy._heaps.values()),rank_entries=len(ranks),
        stale_rank_entries=sum(policy.ranks.get(k)!=r for r,k in ranks),rank_views=len(policy._rank_lists),
        neighbor_chains=len(policy._neighbors),neighbor_edges=sum(map(len,policy._neighbors.values())),
        neighbor_groups=len(policy._neighbor_groups),shared_chain_views=len(policy._shared),
        pending_blocks=len(policy._pending),pending_requests=len(policy._pending_requests))


def run(output,duration=4800,rate=1.,seed=4):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Use a persistent staging directory outside Git')
    output.mkdir(parents=True)
    roots,blocks=load_inputs()
    instances=sample_population({k:r.duration_s for k,r in roots.items()},rate,seed,duration,True)
    calls=[]
    class Measured(ChainHintAwarePolicy):
        def on_request_finished(self,*args,**kwargs):
            start=time.perf_counter_ns()
            result=super().on_request_finished(*args,**kwargs)
            calls.append(dict(time_s=adapter.now,method='on_request_finished',elapsed_us=(time.perf_counter_ns()-start)/1000))
            return result
        def evict(self,*args,**kwargs):
            start=time.perf_counter_ns()
            result=super().evict(*args,**kwargs)
            calls.append(dict(time_s=adapter.now,method='evict',elapsed_us=(time.perf_counter_ns()-start)/1000))
            return result
    adapter=HintPolicyAdapter(274320//16,hint_tier='online',**frozen_options('online'))
    adapter.policy=Measured(274320//16,clock=lambda:adapter.now,**frozen_options('online'))
    samples=[];next_sample=0
    def observe(t):
        nonlocal next_sample
        while next_sample<=min(t,duration):
            item=snapshot(adapter.policy,next_sample)
            for method in ('on_request_finished','evict'):
                values=[c['elapsed_us'] for c in calls if next_sample-60<=c['time_s']<next_sample and c['method']==method]
                for name,q in [('p50',.5),('p99',.99)]:item[method+'_'+name+'_us']=float(np.quantile(values,q)) if values else None
                item[method+'_count']=len(values)
            samples.append(item);print('SNAPSHOT',item,flush=True);next_sample+=60
    summary,rows=simulate(roots,instances,blocks,182880,274320,8,duration=duration,warmup=600,cpu_policy=adapter,observer=observe)
    for name,values in [('snapshots',samples),('callbacks',calls),('requests',rows)]:
        pq.write_table(pa.Table.from_pylist(values),output/(name+'.parquet'),compression='zstd')
    trends={}
    for key in samples[0]:
        if key=='time_s':continue
        kept=[s for s in samples if s['time_s']>=600 and s.get(key) is not None]
        if not kept:continue
        v=np.asarray([s[key] for s in kept]);x=np.asarray([s['time_s'] for s in kept])
        trends[key]=dict(first=float(v[0]),last=float(v[-1]),maximum=float(max(v)),nondecreasing=bool(np.all(np.diff(v)>=0)),
            slope_per_minute=float(np.polyfit(x/60,v,1)[0]),last_20_mean=float(v[-20:].mean()),first_20_mean=float(v[:20].mean()))
    record=dict(git_commit=commit,status='complete',parameters=dict(duration_s=duration,rate=rate,seed=seed,gpu_tokens=182880,cpu_tokens=274320,slots=8,service_time_s=.5),
        source=dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE)),
        instances=[asdict(i) for i in instances],population_sha256=population_hash(instances),schedule_sha256=_schedule_fingerprint(rows),
        summary=summary,trends=trends,snapshots_do_not_advance_policy_clock=True,
        files_sha256={p.name:sha256_file(p) for p in output.glob('*.parquet')})
    write_json(output/'run.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--duration',type=int,default=4800);p.add_argument('--rate',type=float,default=1.);p.add_argument('--seed',type=int,default=4)
    a=p.parse_args();run(a.output_dir,a.duration,a.rate,a.seed)


if __name__=='__main__':main()
