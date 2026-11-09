"""Fit residual-life tables on 293 held-out roots and diagnose workload hints."""
import argparse
import bisect
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import WORKLOAD,write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import ROOT,BuildConfig,noise_factor

AUDIT=ROOT/'results/m1-agentx-data-20261102/requests.parquet'
MANIFEST=WORKLOAD.with_name('manifest.json')


def interval_pairs(rows,config,prior):
    """Use only prior gaps when making each online prediction; no token input."""
    result=[]
    for cid,group in itertools.groupby(sorted(rows,key=lambda r:(r['chain_id'],r['chain_turn'])),key=lambda r:r['chain_id']):
        chain=list(group);ema=None
        if [r['chain_turn'] for r in chain]!=list(range(len(chain))):raise ValueError('Noncontiguous chain')
        for a,b in zip(chain,chain[1:]):
            # Keep operation ordering bit-identical to workload.gap_plan.
            raw=b['start_s']-(a['start_s']+a['api_time_s'])
            if raw<0 or not math.isfinite(raw):raise ValueError('Invalid serial interval')
            gap=min(raw,config.gap_cap)*config.gap_compress
            result.append(dict(root_id=a['session_id'],chain_id=cid,chain_turn=a['chain_turn'],actual_gap_s=gap,
                online_s=prior if ema is None else ema,
                noisy_s=gap*noise_factor(config.noise_seed,a['session_id'],cid,a['chain_turn'])))
            ema=gap if ema is None else config.ema_alpha*gap+(1-config.ema_alpha)*ema
    return result


def fit_tier(rows,tier,grid,bins=10):
    predictions=sorted(r[tier+'_s'] for r in rows)
    cuts=sorted({predictions[min(len(predictions)-1,len(predictions)*i//bins)] for i in range(1,bins)})
    cuts=[v for v in cuts if v>predictions[0]]
    groups=[[] for _ in range(len(cuts)+1)]
    for r in rows:groups[bisect.bisect_right(cuts,r[tier+'_s'])].append(r['actual_gap_s'])
    buckets=[]
    for values in groups:
        assert values
        values=np.array(values,dtype=float);remaining=[];survivors=[]
        for age in grid:
            live=values[values>age];survivors.append(len(live))
            remaining.append(float(np.mean(live-age)) if len(live) else None)
        buckets.append(dict(count=len(values),actual_quantiles_s=dict(zip(('p10','p50','p90','p99'),map(float,np.quantile(values,[.1,.5,.9,.99])))),
                            survivors=survivors,mean_remaining_s=remaining))
    return dict(prediction_cuts_s=cuts,buckets=buckets)


def diagnose(rows):
    positive=[r for r in rows if r['actual_gap_s']>0];zero=[r for r in rows if r['actual_gap_s']==0]
    output={}
    for tier in ('online','noisy'):
        ratios=np.array([r[tier+'_s']/r['actual_gap_s'] for r in positive])
        output[tier]=dict(intervals=len(rows),positive_actual_intervals=len(positive),zero_actual_intervals=len(zero),
            zero_actual_predicted_zero=sum(r[tier+'_s']==0 for r in zero),
            quantiles=dict(zip(('p10','p50','p90','p99'),map(float,np.quantile(ratios,[.1,.5,.9,.99])))),
            within_half_double_count=int(np.count_nonzero((ratios>=.5)&(ratios<=2))),
            within_half_double_fraction=float(np.mean((ratios>=.5)&(ratios<=2))))
    return output


def run(output):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git worktree')
    manifest=json.loads(MANIFEST.read_text());config=BuildConfig(**manifest['parameters'])
    assert sha256_file(AUDIT)==manifest['source']['requests_sha256']
    all_rows=pq.read_table(AUDIT,columns=['session_id','chain_id','chain_turn','start_s','api_time_s']).to_pylist()
    selected=set(manifest['selected_roots']);training_rows=[r for r in all_rows if r['session_id'] not in selected]
    training_ids=sorted({r['session_id'] for r in training_rows})
    assert len(training_ids)==293 and len(selected)==100 and not selected.intersection(training_ids)
    assert training_ids==manifest['conventions']['online_prior_root_ids']
    prior=manifest['conventions']['global_median_prior_s']
    pairs=interval_pairs(training_rows,config,prior)
    assert len(pairs)==manifest['conventions']['online_prior_interval_count']
    assert float(np.median([r['actual_gap_s'] for r in pairs]))==prior
    # Independent reconstruction on all selected roots validates exact online/noisy semantics.
    selected_pairs=interval_pairs([r for r in all_rows if r['session_id'] in selected],config,prior)
    by_turn={(r['root_id'],r['chain_id'],r['chain_turn']):r for r in selected_pairs}
    for r in pq.read_table(WORKLOAD,columns=['root_id','chain_id','chain_turn','final','gap_after_s','hint_online_ms','hint_noisy_ms']).to_pylist():
        if r['final']:continue
        p=by_turn[r['root_id'],r['chain_id'],r['chain_turn']]
        assert p['actual_gap_s']==r['gap_after_s'] and p['online_s']*1000==r['hint_online_ms']
        assert math.isclose(p['noisy_s']*1000,r['hint_noisy_ms'],rel_tol=1e-14,abs_tol=1e-12)
    grid=[i*config.gap_cap*config.gap_compress/120 for i in range(121)]
    model=dict(format='agentkv.remaining_life.v1',git_commit=commit,training_root_ids=training_ids,excluded_root_ids=sorted(selected),
        interval_count=len(pairs),prior_s=prior,parameters=manifest['parameters'],elapsed_grid_s=grid,
        source=dict(audit_sha256=sha256_file(AUDIT),manifest_sha256=sha256_file(MANIFEST),workload_sha256=sha256_file(WORKLOAD)),
        tiers={tier:fit_tier(pairs,tier,grid) for tier in ('online','noisy')})
    output.mkdir(parents=True)
    write_json(output/'remaining_life_v1.json',model)
    pq.write_table(pa.Table.from_pylist(pairs),output/'training_pairs.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(selected_pairs),output/'prediction_pairs.parquet',compression='zstd')
    diagnosis=diagnose(selected_pairs);write_json(output/'prediction_diagnostics.json',diagnosis)
    record=dict(git_commit=commit,status='complete',training_roots=len(training_ids),training_intervals=len(pairs),
        workload_intervals=len(selected_pairs),source=model['source'],
        artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()},prediction_diagnostics=diagnosis)
    write_json(output/'run.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)


if __name__=='__main__':main()
