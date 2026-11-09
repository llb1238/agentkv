"""Block-prefix GPU LRU simulation with pluggable CPU eviction policies.

GPU and CPU keep independent copies of completed prompt blocks. GPU evictions
never insert into CPU. Prefix lookups use contiguous GPU then contiguous CPU;
all remaining tokens are recomputed. Active GPU blocks and private output KV
are reserved until completion. Transfers are instantaneous, service is fixed.
"""

import argparse
import heapq
import json
import math
from collections import OrderedDict, defaultdict, deque
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.capacity import prefix
from agentkv.provenance import git_commit
from agentkv.bench.steady_state import RootSchedule, sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import ROOT, SCHEMA



class CPUCachePolicy(ABC):
    """CPU resident-key policy; M5 policies implement these same operations.

    on_access inserts or touches a completed block. evict removes exactly n
    eligible keys or returns None without modifying the cache. Active offers
    are protected. on_request is optional context for lookahead/hint policies.
    """
    @abstractmethod
    def on_access(self, block, *, time_s=0.): ...

    @abstractmethod
    def evict(self, n, protected=frozenset(), *, time_s=0.): ...

    @abstractmethod
    def __contains__(self, block): ...

    @abstractmethod
    def __len__(self): ...

    def on_request(self, request_id, blocks, *, time_s=0.):
        pass


    def on_finished(self, request_id, blocks, *, stored=True, time_s=0.):
        pass


class LRUPolicy(CPUCachePolicy):
    def __init__(self):
        self.resident=OrderedDict()

    def __contains__(self, block): return block in self.resident
    def __len__(self): return len(self.resident)

    def on_access(self, block, *, time_s=0.):
        self.resident[block]=None
        self.resident.move_to_end(block)

    def evict(self, n, protected=frozenset(), *, time_s=0.):
        if n<0: raise ValueError('Negative eviction count')
        victims=[]
        for key in self.resident:
            if len(victims)==n: break
            if key not in protected: victims.append(key)
        if len(victims)!=n: return None
        for key in victims: del self.resident[key]
        return victims


class FutureUses:
    """Expected deadlines of all unissued requests in sampled instance lifetimes.

    Requests consumed out of deadline order remain supported: only issued IDs
    are removed, never all deadlines <= now (queued requests still matter).
    """
    def __init__(self, events):
        self.deadlines=[]; self.request_ids={}; self.uses=defaultdict(list)
        for deadline,request_id,blocks in sorted(events,key=lambda e:(e[0],e[1])):
            number=len(self.deadlines)
            self.deadlines.append(deadline);self.request_ids[request_id]=number
            for block in blocks: self.uses[int(block)].append(number)
        self.consumed=np.zeros(len(self.deadlines),dtype=bool)
        self.positions={}

    def consume(self, request_id):
        number=self.request_ids[request_id]
        if self.consumed[number]: raise ValueError('Request consumed twice')
        self.consumed[number]=True

    def next_use(self, block):
        events=self.uses.get(block,())
        pos=self.positions.get(block,0)
        while pos<len(events) and self.consumed[events[pos]]: pos+=1
        self.positions[block]=pos
        return self.deadlines[events[pos]] if pos<len(events) else math.inf


class OraclePolicy(CPUCachePolicy):
    """Evict farthest next expected use, infinity first; larger block ID breaks ties.

    This is block Belady over expected deadlines, not a proven global optimum
    for chained prefix hits, GPU residency and request-completion writes.
    """
    def __init__(self, future):
        self.future=future
        self.resident={};self.heap=[];self.generation=0

    def __contains__(self, block): return block in self.resident
    def __len__(self): return len(self.resident)

    def _refresh(self, block):
        self.generation+=1
        value=self.future.next_use(block)
        self.resident[block]=(value,self.generation)
        heapq.heappush(self.heap,(-value,-block,self.generation))

    def _compact(self):
        if len(self.heap)>max(64,2*len(self.resident)):
            self.heap=[(-value,-block,version) for block,(value,version) in self.resident.items()]
            heapq.heapify(self.heap)

    def on_request(self, request_id, blocks, *, time_s=0.):
        self.future.consume(request_id)
        for block in blocks:
            if block in self.resident:
                value=self.future.next_use(block)
                if value!=self.resident[block][0]: self._refresh(block)
        self._compact()

    def on_access(self, block, *, time_s=0.):
        value=self.future.next_use(block)
        if block not in self.resident or self.resident[block][0]!=value:
            self._refresh(block)
        self._compact()

    def evict(self, n, protected=frozenset(), *, time_s=0.):
        if n<0: raise ValueError('Negative eviction count')
        selected=[];deferred=[]
        while self.heap and len(selected)<n:
            item=heapq.heappop(self.heap);negative,neg_key,version=item;key=-neg_key
            if self.resident.get(key)!=( -negative,version): continue
            if key in protected: deferred.append(item)
            else: selected.append(item)
        if len(selected)!=n:
            for item in selected+deferred: heapq.heappush(self.heap,item)
            return None
        for item in deferred: heapq.heappush(self.heap,item)
        victims=[-item[1] for item in selected]
        for key in victims: del self.resident[key]
        self._compact()
        return victims


def scheduled_future(roots, instances, block_ids):
    events=[]
    for ins in instances:
        root=roots[ins.root_id]
        for entry in root.entries(ins.tau_s):
            request=entry.request;idx=request['trace_req_idx']
            blocks=block_ids[idx]+(ins.instance_id<<32)
            events.append((ins.start_s+root.sends[idx]-ins.tau_s,(ins.instance_id,idx),blocks))
    return FutureUses(events)


class TwoTierLRU:
    """Fixed-size blocks, request-level completion writes and pinned GPU reuse."""
    def __init__(self, gpu_tokens, cpu_tokens, block_size=16, cpu_policy=None, audit_evictions=False):
        if block_size<1 or gpu_tokens<block_size or cpu_tokens<0:
            raise ValueError('Invalid cache capacity or block size')
        self.block_size=block_size
        self.last_loss={} if audit_evictions else None
        self.gpu_capacity=gpu_tokens//block_size
        self.cpu_capacity=cpu_tokens//block_size
        self.gpu={}  # ready key -> active refcount
        self.evictable=OrderedDict()
        self.cpu=cpu_policy if cpu_policy is not None else LRUPolicy()
        self.reserved=0
        self.gpu_evictions=self.cpu_evictions=self.cpu_store_skips=0
        self.peak_gpu_blocks=0

    def lookup(self, keys):
        g=0
        for key in keys:
            if key not in self.gpu: break
            g+=1
        c=0
        for key in keys[g:]:
            if key not in self.cpu: break
            c+=1
        return g,c

    def begin(self, keys, prompt_len, max_tokens, *, request_id=None, time_s=0.):
        """Return (GPU hit blocks, CPU hit blocks, private blocks), or queue."""
        required=math.ceil((prompt_len+max_tokens)/self.block_size)
        if required>self.gpu_capacity:
            raise ValueError('A single request exceeds GPU KV capacity')
        g,c=self.lookup(keys)
        private=required-g
        need=max(0,len(self.gpu)+self.reserved+private-self.gpu_capacity)
        pinned_now=sum(key in self.evictable for key in keys[:g])
        if need>len(self.evictable)-pinned_now:
            return None
        for key in keys[:g]:
            self.gpu[key]+=1
            self.evictable.pop(key,None)
        while need:
            key,_=self.evictable.popitem(last=False)
            del self.gpu[key]
            if self.last_loss is not None and key not in self.cpu:self.last_loss[key]=time_s
            self.gpu_evictions+=1
            need-=1
        self.reserved+=private
        self.cpu.on_request(request_id,keys,time_s=time_s)
        self.peak_gpu_blocks=max(self.peak_gpu_blocks,len(self.gpu)+self.reserved)
        return g,c,private

    def finish(self, keys, allocation, *, time_s=0., request_id=None):
        g,_,private=allocation
        self.reserved-=private
        for key in keys[:g]:
            self.gpu[key]-=1
        # Prefix heads are more recent than suffixes within a request, matching
        # the local CPU LRU policy's reversed request-key touch convention.
        for key in reversed(keys):
            self.gpu.setdefault(key,0)
            if self.last_loss is not None:self.last_loss.pop(key,None)
            if not self.gpu[key]:
                self.evictable[key]=None
                self.evictable.move_to_end(key)
        stored=False
        if len(keys)<=self.cpu_capacity:
            # Keep the whole current offer protected, as prepare_store does.
            new=[key for key in keys if key not in self.cpu]
            need=max(0,len(self.cpu)+len(new)-self.cpu_capacity)
            if need:
                victims=self.cpu.evict(need,set(keys),time_s=time_s)
                if victims is None:
                    self.cpu_store_skips+=1
                    self.cpu.on_finished(request_id,keys,stored=False,time_s=time_s)
                    return
                self.cpu_evictions+=len(victims)
                if self.last_loss is not None:
                    for victim in victims:
                        if victim not in self.gpu:self.last_loss[victim]=time_s
            for key in reversed(keys):
                self.cpu.on_access(key,time_s=time_s)
            stored=True
        elif keys:
            self.cpu_store_skips+=1
        self.cpu.on_finished(request_id,keys,stored=stored,time_s=time_s)
        if len(self.gpu)+self.reserved>self.gpu_capacity:
            raise AssertionError('GPU allocation exceeded capacity')
        if len(self.cpu)>self.cpu_capacity:
            raise AssertionError('CPU allocation exceeded capacity')


def canonical_prefix_blocks(rows, block_size=16):
    """Exact prefix identities without probabilistic hashes or cross-root sharing.

    Sort byte encodings. Each prompt can share only full prefix blocks with
    its lexicographic predecessor; allocate new global IDs for the suffix.
    Partial final blocks are never reusable, as in block prefix caching.
    """
    by_root=defaultdict(list)
    for row in rows: by_root[row['root_id']].append(row)
    output={}; next_id=1
    for rid,group in sorted(by_root.items()):
        previous=b''; previous_ids=np.empty(0,dtype=np.int64)
        for r in sorted(group,key=lambda r:r['_tokens']):
            value=r['_tokens']; n=len(value)//4//block_size
            shared=min(n,prefix(previous,value)//block_size)
            ids=np.empty(n,dtype=np.int64)
            ids[:shared]=previous_ids[:shared]
            ids[shared:]=np.arange(next_id,next_id+n-shared,dtype=np.int64)
            next_id+=n-shared
            output[r['trace_req_idx']]=ids
            previous,previous_ids=value,ids
    return output,next_id-1


def load_block_ids(path, block_size):
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128,use_threads=False):
        metadata=batch.select(['root_id','trace_req_idx']).to_pylist()
        col=batch.column(batch.schema.get_field_index('prompt_token_ids'))
        for i,r in enumerate(metadata):
            r['_tokens']=col[i].values.to_numpy().astype('<i4',copy=False).tobytes()
            rows.append(r)
    return canonical_prefix_blocks(rows,block_size)


def simulate(roots, instances, block_ids, gpu_tokens, cpu_tokens, slots,
             duration=2400., warmup=600., service_time=.5, block_size=16, cpu_policy="lru", audit_evictions=False,
             hint_tier="perfect", hint_options=None, observer=None):
    if slots<1 or not 0<=warmup<duration or service_time<=0:
        raise ValueError('Invalid simulation parameters')
    from agentkv.sim.hint_adapter import HintPolicyAdapter
    if isinstance(cpu_policy,CPUCachePolicy): policy=cpu_policy
    elif cpu_policy=="oracle": policy=OraclePolicy(scheduled_future(roots,instances,block_ids))
    elif cpu_policy=="lru": policy=LRUPolicy()
    elif cpu_policy=="hint": policy=HintPolicyAdapter(cpu_tokens//block_size,hint_tier=hint_tier,block_size=block_size,**(hint_options or {}))
    else: raise ValueError("Unknown CPU policy")
    cache=TwoTierLRU(gpu_tokens,cpu_tokens,block_size,policy,audit_evictions)
    audit_checked=audit_passed=0
    audit_failures=[]
    nodes={}; events=[]; counter=0
    def push(t,kind,key):
        nonlocal counter
        heapq.heappush(events,(t,kind,counter,key)); counter+=1
    for ins in instances:
        entries=roots[ins.root_id].entries(ins.tau_s)
        for e in entries:
            key=(ins.instance_id,e.request['trace_req_idx'])
            nodes[key]=dict(entry=e,instance=ins,children=[])
            if isinstance(policy,HintPolicyAdapter):policy.register_request(key,e.request,ins)
        for e in entries:
            key=(ins.instance_id,e.request['trace_req_idx'])
            if e.timed: push(ins.start_s+e.delay_s,1,key)
            else: nodes[ins.instance_id,e.trigger_trace_req_idx]['children'].append(key)
    queue=deque(); records=[]; active=peak=kv_blocked=0
    while events:
        t,kind,_,key=heapq.heappop(events)
        if observer is not None:observer(t)
        node=nodes[key]
        if kind==0:
            cache.finish(node.pop('keys'),node.pop('allocation'),time_s=t,request_id=key)
            node['end_s']=t
            active-=1
            for child in node['children']:
                due=t+nodes[child]['entry'].delay_s
                if due<duration: push(due,1,child)
        elif t<duration:
            node['due']=t
            queue.append(key)
        while queue and active<slots and t<duration:
            k=queue[0]; n=nodes[k]; e=n['entry']; r=e.request
            if 'keys' not in n:
                # Independent instance namespace is equivalent to token
                # permutation for prefix identity; no cross-instance hits.
                n['keys']=tuple((block_ids[r['trace_req_idx']]+(n['instance'].instance_id<<32)).tolist())
            allocation=cache.begin(n['keys'],r['prompt_len'],r['max_tokens'],request_id=k,time_s=t)
            if allocation is None:
                kv_blocked+=1
                break
            queue.popleft(); n['allocation']=allocation
            g,c,_=allocation
            record=dict(instance_id=n['instance'].instance_id,root_id=r['root_id'],trace_req_idx=r['trace_req_idx'],
                        chain_id=r['chain_id'],chain_turn=r['chain_turn'],due_s=n['due'],send_s=t,end_s=t+service_time,
                        is_entry=e.is_entry,in_window=warmup<=t<duration and not e.is_entry,
                        prompt_tokens=r['prompt_len'],gpu_hit=g*block_size,cpu_hit=c*block_size,
                        recomputed=r['prompt_len']-(g+c)*block_size,queued_s=t-n['due'])
            if audit_evictions and record['in_window']:
                from agentkv.analysis.recomputation import common_blocks
                trigger=roots[r['root_id']].triggers[r['trace_req_idx']][0]
                parent=nodes.get((n['instance'].instance_id,trigger))
                p=common_blocks(block_ids[r['trace_req_idx']],block_ids[trigger]) if parent is not None else 0
                checked=passed=0
                for position in range(g+c,p):
                    checked+=1
                    key=n['keys'][position];loss=cache.last_loss.get(key)
                    complete=parent.get('end_s')
                    good=loss is not None and complete is not None and complete-1e-9<=loss<=t+1e-9
                    passed+=int(good)
                    if not good and len(audit_failures)<20:
                        audit_failures.append(dict(instance_id=record['instance_id'],trace_req_idx=r['trace_req_idx'],
                            block=key,loss_s=loss,trigger_end_s=complete,dispatch_s=t))
                record.update(gap_audit_blocks=checked,gap_audit_passed=passed)
                audit_checked+=checked;audit_passed+=passed
            records.append(record)
            active+=1; peak=max(peak,active)
            push(t+service_time,0,k)
    if active:
        raise AssertionError('Undrained simulation')
    if observer is not None:observer(duration)
    window=[r for r in records if r['in_window']]
    total=sum(r['prompt_tokens'] for r in window)
    counts={k:sum(r[k] for r in window) for k in ('gpu_hit','cpu_hit','recomputed')}
    if sum(counts.values())!=total: raise AssertionError('Non-additive token sources')
    summary=dict(requests=len(records),window_requests=len(window),window_prompt_tokens=total,
                 source_tokens=counts,source_fractions={k:v/total if total else None for k,v in counts.items()},
                 gpu_evictions=cache.gpu_evictions,cpu_evictions=cache.cpu_evictions,
                 cpu_store_skipped_requests=cache.cpu_store_skips,peak_running=peak,
                 peak_gpu_tokens=cache.peak_gpu_blocks*block_size,kv_blocked_checks=kv_blocked,
                 queued_requests=sum(r['queued_s']>1e-9 for r in records),unsent_queued=len(queue))
    if audit_evictions:
        summary['gap_eviction_audit']=dict(blocks=audit_checked,passed=audit_passed,
            consistency=1. if not audit_checked else audit_passed/audit_checked,failures=audit_failures)
        if audit_checked!=audit_passed:
            raise AssertionError(f'Gap eviction consistency failure: {summary["gap_eviction_audit"]}')
    return summary,records


def run_capacity(capacity_dir, output_dir, gap_threshold=.10, allow_dirty=False):
    commit=git_commit(allow_dirty)
    output_dir.mkdir(parents=True,exist_ok=True)
    if (output_dir/'run.json').exists(): raise FileExistsError('New simulation directory required')
    capacity=json.loads((capacity_dir/'run.json').read_text())
    results=[]
    print('Simulating',len(capacity['candidates']),'capacity-valid configurations',flush=True)
    for budget in sorted({c['context_budget'] for c in capacity['candidates']},reverse=True):
        config=next(c for c in capacity['configurations'] if c['context_budget']==budget)
        path=Path(config['workload_path'])
        if sha256_file(path)!=config['workload_sha256']: raise ValueError('Workload SHA mismatch')
        rows=pq.read_table(path,columns=[n for n in SCHEMA.names if n!='prompt_token_ids'],use_threads=False).to_pylist()
        by_root=defaultdict(list)
        for row in rows: by_root[row['root_id']].append(row)
        schedule=pq.read_table(capacity_dir/f'schedule-{budget}.parquet').to_pylist()
        estimates=defaultdict(list)
        for row in schedule: estimates[row['root_id']].append(row)
        roots={rid:RootSchedule(rs,estimates[rid]) for rid,rs in by_root.items()}
        blocks,unique=load_block_ids(path,16)
        print('Prefix blocks',budget,unique,flush=True)
        ideal={r['trace_req_idx']:r['unlimited_hit_tokens'] for r in pq.read_table(capacity_dir/'prefix_upper.parquet').to_pylist()
               if r['context_budget']==budget}
        controls={}
        for number,candidate in enumerate(capacity['candidates']):
            if candidate['context_budget']!=budget: continue
            rate=candidate['rate_per_minute']
            cfg=next(c for c in capacity['configurations'] if c['context_budget']==budget and c['rate_per_minute']==rate)
            from agentkv.bench.steady_state import SessionInstance
            instances=[SessionInstance(**i) for i in cfg['instances']]
            control_key=(rate,candidate['max_num_seqs'])
            if control_key not in controls:
                control_summary,control_records=simulate(roots,instances,blocks,1<<50,1<<50,candidate['max_num_seqs'],
                    capacity['parameters']['duration'],capacity['parameters']['warmup'])
                control_name=f"unlimited-{budget}-{rate:g}-{candidate['max_num_seqs']}"
                control_dir=output_dir/control_name; control_dir.mkdir()
                pq.write_table(pa.Table.from_pylist(control_records),control_dir/'requests.parquet',compression='zstd')
                (control_dir/'run.json').write_text(json.dumps(dict(git_commit=commit,summary=control_summary,
                    requests_sha256=sha256_file(control_dir/'requests.parquet')),indent=2)+'\n')
                controls[control_key]=(control_summary,control_records,control_name)
            summary,records=simulate(roots,instances,blocks,candidate['gpu_tokens'],candidate['cpu_tokens'],
                       candidate['max_num_seqs'],capacity['parameters']['duration'],capacity['parameters']['warmup'])
            frac=summary['source_fractions']; hit=frac['gpu_hit']+frac['cpu_hit']
            # Stationary offered load with measured LRU miss fraction; report
            # actual window rate too, so finite-window sampling is visible.
            demand=candidate['raw_prefill_tps']*frac['recomputed']
            upper=candidate['upper_hit_rate']
            control_summary,control_records,control_name=controls[control_key]
            control_hit=1-control_summary['source_fractions']['recomputed']
            window=[r for r in records if r['in_window']]
            window_upper=sum(ideal[r['trace_req_idx']] for r in window)/sum(r['prompt_tokens'] for r in window)
            same_window={(r['instance_id'],r['trace_req_idx']) for r in window}=={
                (r['instance_id'],r['trace_req_idx']) for r in control_records if r['in_window']}
            conditions=dict(capacity=all(candidate['constraints'].values()),
                eviction_gap=upper-hit>=gap_threshold,
                controlled_eviction_gap=same_window and control_hit-hit>=gap_threshold,
                compute_22k_70pct=demand<=15400,
                nonempty=summary['window_requests']>0,all_offered_requests_sent=summary['unsent_queued']==0)
            item=dict(git_commit=commit,candidate=candidate,summary=summary,total_hit_fraction=hit,
                      upper_hit_rate=upper,upper_minus_lru=upper-hit,prefill_tokens_per_second=demand,
                      same_window_theoretical_upper=window_upper,unlimited_control=control_name,
                      unlimited_control_hit=control_hit,unlimited_minus_lru=control_hit-hit,
                      control_window_matches=same_window,
                      utilization_22k=demand/22000,utilization_34k=demand/34000,
                      measured_window_prefill_tps=summary['source_tokens']['recomputed']/(capacity['parameters']['duration']-capacity['parameters']['warmup']),
                      recommendation_checks=conditions,recommend_for_gpu_pilot=all(conditions.values()))
            name=f'config-{number:02d}'; directory=output_dir/name; directory.mkdir()
            pq.write_table(pa.Table.from_pylist(records),directory/'requests.parquet',compression='zstd')
            item['requests_sha256']=sha256_file(directory/'requests.parquet')
            (directory/'run.json').write_text(json.dumps(item,indent=2)+'\n')
            item['directory']=name; results.append(item)
            print(name,budget,rate,candidate['max_num_seqs'],candidate['cpu_gpu_ratio'],
                  'GPU/CPU/recompute',frac,'tps',demand,'pilot',item['recommend_for_gpu_pilot'],flush=True)
        del blocks,roots
    report=dict(git_commit=commit,source=dict(capacity_run_sha256=sha256_file(capacity_dir/'run.json'),
                capacity_directory=str(capacity_dir)),
                parameters=dict(block_size=16,service_time_s=.5,eviction_gap_threshold=gap_threshold,
                    duration=capacity['parameters']['duration'],warmup=capacity['parameters']['warmup']),
                conventions=dict(cpu='duplicate-write prompt mirror, independent LRU, no GPU-eviction admission; no invalidation of GPU on CPU eviction',
                    prefix='GPU contiguous full-block prefix then CPU contiguous extension; remainder recomputed; partial tail never cached',
                    timing='keys ready and CPU writes only at request completion; GPU pins + private output reservation; FCFS slot/KV queue',
                    initialization='same M/G/infinity instances and tau as capacity; empty caches at t=0; skipped history not preloaded; is_entry excluded',
                    throughput='stationary raw demand times simulated token-weighted miss fraction; actual window tps reported separately',
                    limitations='fixed .5s service; instantaneous transfers; no decode-time model or vLLM chunk-transfer/admission detail; no guarantee of shared-GPU stability'),
                results=results,recommendations=[r['directory'] for r in results if r['recommend_for_gpu_pilot']])
    (output_dir/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-dir',type=Path,default=ROOT/'results/m1-capacity-seed0-v3')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/m1-two-tier-seed0')
    parser.add_argument('--eviction-gap',type=float,default=.10)
    parser.add_argument('--allow-dirty',action='store_true',help='DEBUG ONLY: bypass clean-worktree check')
    args=parser.parse_args(argv)
    if not 0<=args.eviction_gap<=1: raise ValueError('Invalid eviction gap')
    run_capacity(args.capacity_dir,args.output_dir,args.eviction_gap,args.allow_dirty)


if __name__=='__main__': main()
