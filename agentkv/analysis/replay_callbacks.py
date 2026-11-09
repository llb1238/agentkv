"""Exact recorded-policy replay and clearly labelled LRU-state decision replay."""
import argparse
from collections import Counter, deque
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.provenance import git_commit


def preview(actor,n,protected):
    """Evaluate a candidate on an observed LRU cache path, then roll it back.

    Membership follows recorded LRU victims in this mode. This tests callbacks
    and safe decisions; it does not estimate counterfactual hit rates.
    """
    adapter=getattr(actor,'adapter',actor);core=adapter.policy
    ranks=dict(core.ranks);last=dict(core.last_access);owners=dict(core.owners);chains=dict(core.chains)
    count=core.stats['evicted']
    result=actor.evict(n,protected)
    if result:
        affected=set()
        for key,chunk in reversed(result):
            core.chunks[key]=chunk;core.ranks[key]=ranks[key];core.last_access[key]=last[key]
            for identity,depth in owners.get(key,{}).items():
                core.chains.setdefault(identity,chains[identity])
                affected.update(core._attach(key,identity,depth))
            affected.add(key)
        for key in affected:core._refresh(key,core._time)
        core.stats['evicted']=count
    return result


def replay(directory,output):
    from vllm.v1.kv_offload.base import ReqContext,TierFilter,TierMatcher,Medium,Locality
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    from agentkv.policy import vllm_plugin
    from agentkv.policy.prefix_adapter import envelope
    from agentkv.sim.callback_adapter import CallbackPolicyAdapter

    commit=git_commit();trial=json.loads((directory/'trial-run.json').read_text())
    previous=os.environ.get('AGENTKV_HINT_POLICY_CONFIG')
    os.environ['AGENTKV_HINT_POLICY_CONFIG']=str(directory/'policy-config.json')
    old_clock=vllm_plugin.CLOCK
    keys={};ids={};contexts={};frames={};clock_values=deque();fallback=0.
    def clock():return clock_values.popleft() if clock_values else fallback
    vllm_plugin.CLOCK=clock
    actors=None;reference=None;counts=Counter();decisions=[];prefix_checks=[];closed=0;seen=set();paths=[]
    try:
        for path in sorted((directory/'callback-traces').glob('*.gz')):
            paths.append(path)
            with gzip.open(path,'rt') as source:
                for line in source:
                    e=json.loads(line);kind=e['event']
                    if kind=='open':
                        assert actors is None,'Single policy instance required for this trial'
                        actors=(vllm_plugin.HintAwareCachePolicy(e['capacity']),
                                CallbackPolicyAdapter(e['capacity'],clock=clock))
                        exact='HintAware' in e['policy']
                        if not exact:reference=LRUCachePolicy(e['capacity'])
                    elif kind=='key':
                        keys[e['key']]=bytes.fromhex(e['hex']);ids[keys[e['key']]]=e['key']
                    elif kind=='context':
                        ctx=ReqContext(e['req_id'],kv_transfer_params=e['kv_transfer_params'],
                            kv_hints=e['kv_hints'],num_prompt_tokens=e['num_prompt_tokens'],
                            load_tier_filter=TierFilter(tuple(TierMatcher(
                                Medium(m) if m is not None else None,Locality(l) if l is not None else None)
                                for m,l in e['load_tier_filter'])))
                        for k,pos in e['positions']:ctx.set_offload_key_position(keys[k],pos)
                        assert ctx.get_offload_prefix_keys()==tuple(keys[k] for k in e['prefix'])
                        assert len(e['prefix'])==ctx.num_prompt_tokens//16,(ctx.req_id,'incomplete context')
                        contexts[e['context']]=ctx
                    elif kind=='clock':
                        if e['call'] is not None:frames[e['call']]['clocks'].append(e['value'])
                    elif kind=='manager_refs':
                        assert not frames,'Manager mutations must occur between root policy calls'
                        live=(*actors,reference) if reference is not None else actors
                        for actor in live:
                            for kid,value in e['refs']:
                                chunk=actor.chunks.get(keys[kid])
                                assert (chunk is None)==(value is None)
                                if chunk is not None:chunk.ref_cnt,chunk.chunk_id=value
                    elif kind=='begin':
                        counts[e['method']]+=1;e['clocks']=[];frames[e['call']]=e
                    elif kind=='error':raise AssertionError(('Recorded callback exception',e))
                    elif kind=='close':
                        assert e['counts']==dict(counts);assert e['calls']==sum(counts.values());closed+=1
                    elif kind=='end':
                        start=frames.pop(e['call'])
                        if start['parent'] is not None:continue
                        method=start['method'];args=start['args'];fallback=start['ns']/1e9
                        live=(*actors,reference) if reference is not None else actors
                        for actor in live:
                            for kid,value in start['refs']:
                                key=keys[kid];chunk=actor.chunks.get(key)
                                assert (chunk is None)==(value is None),(method,'membership',kid)
                                if chunk is not None:chunk.ref_cnt,chunk.chunk_id=value
                        def arguments():
                            if method in ('get','remove','mark_evictable','mark_non_evictable'):return [keys[args[0]]]
                            if method=='insert':return [keys[args[0]],NS(ref_cnt=args[1][0],chunk_id=args[1][1])]
                            if method in ('touch','on_store_miss'):return [[keys[k] for k in args[0]],contexts[args[1]]]
                            if method=='on_request_finished':return [tuple(tuple(keys[k] for k in g) for g in args[0]),
                                {keys[k] for k in args[1]},{keys[k] for k in args[2]},contexts[args[3]]]
                            if method=='evict':return [args[0],{keys[k] for k in args[1]}]
                            return []
                        def result_value(result):
                            if method=='get':return None if result is None else [result.ref_cnt,result.chunk_id]
                            if method=='evict':return None if result is None else [
                                [ids[k],[c.ref_cnt,c.chunk_id]] for k,c in result]
                            return None
                        results=[]
                        for actor in actors:
                            clock_values=deque(start['clocks'])
                            call_args=arguments()
                            if method=='evict':
                                adapter=getattr(actor,'adapter',actor)
                                before=set(actor.chunks);refs={k:c.ref_cnt for k,c in actor.chunks.items()}
                                ownership=adapter.ownership()
                                result=actor.evict(*call_args) if exact else preview(actor,*call_args)
                                if result is not None:
                                    assert len(result)==args[0]
                                    assert all(refs[k]==0 and k not in call_args[1] for k,_ in result)
                                elif exact:assert set(actor.chunks)==before
                                if actor is actors[0]:decisions.append(dict(ns=start['ns'],n=args[0],**ownership))
                            else:result=getattr(actor,method)(*call_args)
                            results.append(result_value(result))
                            if exact:assert not clock_values,('Unused recorded clock values',method)
                            if method=='on_request_finished':
                                ctx=call_args[3];hint=envelope(ctx)
                                if hint:
                                    payload=hint['actions'][0]['payload'];adapter=getattr(actor,'adapter',actor)
                                    identity=(payload['parent_session'] or payload['session_id'],payload['session_id'])
                                    chain=adapter.policy.chains[identity]
                                    assert chain.mode==('final' if payload['final'] else 'waiting')
                                    if not payload['final']:
                                        finished_at=start['clocks'][-1] if exact else fallback
                                        assert abs(chain.deadline-(finished_at+payload['expected_gap_ms']/1000))<1e-8
                                    for key in ctx.get_offload_prefix_keys():
                                        if key in actor.chunks:
                                            assert adapter.policy.owners[key][identity]==ctx.get_offload_key_position(key)
                                    if actor is actors[0]:
                                        seen.add(hint['message_id'])
                                        n=len(ctx.get_offload_prefix_keys())
                                        assert n==ctx.num_prompt_tokens//16,(ctx.req_id,n,ctx.num_prompt_tokens)
                                        prefix_checks.append(dict(request_id=ctx.req_id,prefix_keys=n,prompt_tokens=ctx.num_prompt_tokens))
                        assert results[0]==results[1],('Adapter decisions differ',start['call'],method)
                        if exact:assert results[0]==e['result'],('Recorded HintAware mismatch',start['call'],method)
                        else:
                            result=getattr(reference,method)(*arguments())
                            assert result_value(result)==e['result'],('Recorded LRU mismatch',start['call'],method)
                            if method=='evict' and result:
                                # Explicitly adopt recorded membership after the
                                # dry-run comparison. No insert is skipped and
                                # no unknown cache mismatch is reconciled.
                                for actor in actors:
                                    for key,_ in result:actor.remove(key)
                        for actor in live:
                            for kid,value in e['refs']:
                                chunk=actor.chunks.get(keys[kid])
                                assert (None if chunk is None else [chunk.ref_cnt,chunk.chunk_id])==value
        assert closed==1 and not frames,'Incomplete trace/footer'
        run=json.loads((directory/'run.json').read_text())
        expected=run['summary']['hint_sent']
        assert len(seen)==expected,(len(seen),expected)
        output.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(decisions),output/'ownership.parquet')
        pq.write_table(pa.Table.from_pylist(prefix_checks),output/'prefix-checks.parquet')
        summary=dict(git_commit=commit,vllm_commit=trial['vllm_commit'],
            mode='exact HintAware call/clock replay' if exact else 'exact LRU replay + candidate decision replay on recorded LRU states',
            callback_counts=dict(counts),root_sequence_complete=True,close_footers=closed,
            valid_hint_requests=len(seen),prefix_checked_requests=len(prefix_checks),eviction_decisions=len(decisions),
            invariants=dict(no_unready_pinned_or_protected_victims=True,finished_chain_state=True,
                resident_prefix_membership_and_depth=True,ownership_measured_per_eviction=True,no_exceptions=True),
            candidate_adapter_decisions_identical=True,
            limitations=[] if exact else ['Candidate dry runs restore membership before recorded LRU victims are applied; not counterfactual hit rates.'],
            trace_sha256={p.name:sha256_file(p) for p in paths})
        write_json(output/'run.json',summary)
        return summary
    finally:
        vllm_plugin.CLOCK=old_clock
        if previous is None:os.environ.pop('AGENTKV_HINT_POLICY_CONFIG',None)
        else:os.environ['AGENTKV_HINT_POLICY_CONFIG']=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();replay(a.run_dir,a.output_dir)


if __name__=='__main__':main()
