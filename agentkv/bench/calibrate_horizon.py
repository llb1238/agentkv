"""Build per-hint empirical CDFs from the pinned 293-root calibration only."""
import argparse
import bisect
from collections import Counter
import itertools
from pathlib import Path

import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import ROOT,write_json
from agentkv.bench.e1_remaining import read,calibration_values
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.remaining_life import DEFAULT_TABLE
from agentkv.policy.horizon import HorizonProbabilityTable

CALIBRATION=ROOT/'results/m5-e1-remaining-config11-20261104/calibration'


def empirical(model,rows):
    tiers={}
    for tier in ('online','noisy'):
        cuts=model['tiers'][tier]['prediction_cuts_s'];groups=[Counter() for _ in range(len(cuts)+1)]
        for row in rows:groups[bisect.bisect_right(cuts,row[tier+'_s'])][row['actual_gap_s']]+=1
        buckets=[]
        for counts in groups:
            support=sorted(counts)
            buckets.append(dict(support_s=support,cumulative_counts=list(itertools.accumulate(counts[g] for g in support))))
        tiers[tier]=dict(prediction_cuts_s=cuts,buckets=buckets)
    return tiers


def run(output,calibration=CALIBRATION):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git')
    record=read(calibration/'run.json');model=read(calibration/'remaining_life_v1.json')
    assert calibration_values(model)==calibration_values(read(DEFAULT_TABLE))
    for path,sha in record['artifacts'].items():assert sha256_file(calibration/path)==sha
    rows=pq.read_table(calibration/'training_pairs.parquet').to_pylist()
    assert len(rows)==62159 and len(model['training_root_ids'])==293
    assert not set(model['training_root_ids']).intersection(model['excluded_root_ids'])
    assert {r['root_id'] for r in rows}<=set(model['training_root_ids'])
    payload=dict(format='agentkv.horizon_probability.v1',training_root_ids=model['training_root_ids'],excluded_root_ids=model['excluded_root_ids'],
        interval_count=len(rows),elapsed_grid_s=model['elapsed_grid_s'],tiers=empirical(model,rows),
        source=dict(remaining_table_sha256=sha256_file(calibration/'remaining_life_v1.json'),training_pairs_sha256=sha256_file(calibration/'training_pairs.parquet'),
                    calibration_run_sha256=sha256_file(calibration/'run.json'),**model['source']))
    for tier in ('online','noisy'):HorizonProbabilityTable(payload,tier)
    output.mkdir(parents=True);write_json(output/'horizon_probability_v1.json',payload)
    write_json(output/'run.json',dict(git_commit=commit,status='complete',training_roots=293,training_intervals=len(rows),
        source=payload['source'],table_sha256=sha256_file(output/'horizon_probability_v1.json')))
    print('Calibrated',len(rows),'intervals; SHA256',sha256_file(output/'horizon_probability_v1.json'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--calibration-dir',type=Path,default=CALIBRATION);a=p.parse_args();run(a.output_dir,a.calibration_dir)


if __name__=='__main__':main()
