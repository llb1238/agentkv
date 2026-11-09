import math
import pytest
from agentkv.policy.remaining_life import RemainingLifeTable
from agentkv.bench.calibrate_remaining_life import fit_tier,interval_pairs,diagnose
from agentkv.bench.workload import BuildConfig


def model(tier='online'):
    rows=[dict(online_s=1.,actual_gap_s=v) for v in [0.,1.,3.,5.]]
    return dict(format='agentkv.remaining_life.v1',elapsed_grid_s=[0.,1.,2.,3.,5.],tiers={tier:fit_tier(rows,'online',[0.,1.,2.,3.,5.])})


def test_empirical_residual_excludes_ended_and_zero_lifetimes():
    table=RemainingLifeTable(model())
    assert table.lookup(0,0)==(3.,1.)
    assert table.lookup(0,1)==(3.,2.)  # survivors 3 and 5, mean(2,4)
    assert table.lookup(0,2.9)==(2.,3.)
    assert table.lookup(0,3)==(2.,5.)
    assert table.lookup(0,5)==(None,math.inf)
    assert table.lookup(0,100)==(None,math.inf)
    with pytest.raises(ValueError):table.bucket(float('nan'))


def test_tied_quantile_boundaries_merge_and_all_training_pairs_covered():
    rows=[dict(online_s=i//20,actual_gap_s=i%6) for i in range(100)]
    data=fit_tier(rows,'online',[0.,1.,6.]);table=RemainingLifeTable(dict(format='agentkv.remaining_life.v1',elapsed_grid_s=[0.,1.,6.],tiers={'online':data}))
    assert len(data['buckets'])==5 and sum(b['count'] for b in data['buckets'])==100
    assert [table.bucket(i) for i in range(5)]==list(range(5))
    assert all(b['survivors'][-1]==0 for b in data['buckets'])


def test_online_prediction_uses_past_only_and_zero_gaps_are_separate():
    rows=[dict(session_id='r',chain_id='c',chain_turn=i,start_s=t,api_time_s=1.) for i,t in enumerate([0.,3.,14.,15.])]
    pairs=interval_pairs(rows,BuildConfig(),prior=7.)
    assert [r['actual_gap_s'] for r in pairs]==[2.,10.,0.]
    assert [r['online_s'] for r in pairs]==[7.,2.,6.]
    result=diagnose(pairs)
    assert result['online']['positive_actual_intervals']==2 and result['online']['zero_actual_intervals']==1
    assert result['noisy']['zero_actual_predicted_zero']==1
