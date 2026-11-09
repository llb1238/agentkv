"""Fingerprint complete callback trials and report instrumented policy costs."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.provenance import git_commit


def describe(values):
    return dict(count=len(values),minimum=float(min(values)),p50=float(np.median(values)),
        p99=float(np.quantile(values,.99)),maximum=float(max(values))) if values else dict(count=0)


def trace_metrics(directory):
    evictions={};wall=[];batches=[];refs=Counter();closes=0
    for path in sorted((directory/'callback-traces').glob('*.gz')):
        with gzip.open(path,'rt') as source:
            for line in source:
                # Most entries are get calls: decode only the fields needed
                # for latency/transition summaries; exact replay checks all.
                if '"event":"begin"' in line and '"method":"evict"' in line:
                    e=json.loads(line);evictions[e['call']]=e['ns'];batches.append(e['args'][0])
                elif '"event":"end"' in line:
                    e=json.loads(line)
                    if e['call'] in evictions:wall.append((e['ns']-evictions.pop(e['call']))/1e3)
                elif '"event":"manager_refs"' in line:
                    e=json.loads(line)
                    for _,before,after in e['transitions']:
                        refs[f'{before[0] if before else None}->{after[0] if after else None}']+=1
                elif '"event":"close"' in line:closes+=1
    assert closes==1 and not evictions
    return dict(full_instrumented_evict_us=describe(wall),eviction_batch_blocks=describe(batches),
                manager_ref_transitions=dict(refs),close_footers=closes)


def summarize(probe,probe_replay,smoke,smoke_replay,output):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    summary=dict(git_commit=commit,trials={})
    for name,directory,replayed in [('probe',probe,probe_replay),('smoke',smoke,smoke_replay)]:
        trial=json.loads((directory/'trial-run.json').read_text())
        run=json.loads((directory/'run.json').read_text())
        replay=json.loads((replayed/'run.json').read_text())
        assert trial['vllm_commit']==run['vllm_commit']==run['server']['vllm_commit']==replay['vllm_commit']
        assert all(replay['invariants'].values()) and replay['candidate_adapter_decisions_identical']
        for filename,digest in trial['trace_sha256'].items():
            assert sha256_file(directory/'callback-traces'/filename)==digest==replay['trace_sha256'][filename]
        rows=pq.read_table(directory/'request-metadata.parquet').to_pylist()
        sent=[r for r in rows if r['send_ns'] is not None];statuses=Counter(r['status'] for r in sent)
        assert statuses.get('success',0)==len(sent)
        assert run['summary']['hint_sent']==replay['valid_hint_requests']==len(sent)
        ownership=pq.read_table(replayed/'ownership.parquet').to_pylist()
        record=dict(measurement_git_commit=trial['git_commit'],replay_git_commit=replay['git_commit'],
            vllm_commit=trial['vllm_commit'],duration_s=trial['duration_s'],sent=len(sent),statuses=dict(statuses),
            hint_sent=run['summary']['hint_sent'],hint_received=run['summary']['hint_received'],
            valid_policy_hints=replay['valid_hint_requests'],prefix_checked=replay['prefix_checked_requests'],
            recorded_callbacks=sum(replay['callback_counts'].values()),callback_counts=replay['callback_counts'],
            replay_mode=replay['mode'],invariants=replay['invariants'],
            create_req_context_us=trial['create_req_context_us'],core_evict_us=trial['core_evict_us'],
            adapter_evict_us=trial['adapter_evict_us'],ownership=describe([r['owned_fraction'] for r in ownership]),
            unknown_block_observations=sum(r['cpu_blocks']-r['owned_blocks'] for r in ownership),
            source_sha256={str(p):sha256_file(p) for p in [directory/'run.json',directory/'trial-run.json',
                directory/'request-metadata.parquet',replayed/'run.json',replayed/'ownership.parquet']},
            safety_valve_triggered=run['summary']['safety_valve_triggered'],**trace_metrics(directory))
        summary['trials'][name]=record
    summary['vllm_commit']=summary['trials']['smoke']['vllm_commit']
    summary['limits']=['Instrumented shared-GPU functional runs, no paired uninstrumented performance baseline.',
        'LRU stream candidate replay follows observed LRU membership; HintAware stream uses exact decisions and recorded clocks.',
        'Standalone real-model integration module requires unavailable Llama weights; original unit suites passed separately.']
    write_json(output/'run.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5))
    for name,replayed,color in [('probe',probe_replay,'#0072B2'),('smoke',smoke_replay,'#D55E00')]:
        rows=pq.read_table(replayed/'ownership.parquet').to_pylist();origin=rows[0]['ns']
        axes[0].plot([(r['ns']-origin)/60e9 for r in rows],[r['owned_fraction'] for r in rows],
            label=name,lw=1.,color=color)
    axes[0].set(xlabel='Minutes since first eviction',ylabel='Owned CPU block fraction',ylim=(0,1.05))
    axes[0].legend(frameon=False)
    m=summary['trials']['smoke']
    labels=['Core','Adapter','Full trace']
    fields=['core_evict_us','adapter_evict_us','full_instrumented_evict_us']
    for i,q in enumerate(['p50','p99']):
        axes[1].bar(np.arange(3)+(i-.5)*.35,[m[f][q]/1000 for f in fields],width=.35,label=q.upper())
    axes[1].set(xticks=np.arange(3),xticklabels=labels,ylabel='Evict time (ms)')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'ownership_and_cost.{ext}',dpi=300)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','probe-replay','smoke','smoke-replay','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.probe,a.probe_replay,a.smoke,a.smoke_replay,a.output_dir)


if __name__=='__main__':main()
