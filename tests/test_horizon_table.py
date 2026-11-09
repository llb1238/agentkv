import pytest
from agentkv.bench.calibrate_horizon import empirical
from agentkv.policy.horizon import HorizonProbabilityTable


def fixture():
    rows=[dict(online_s=1.,noisy_s=10. if i<2 else 1.,actual_gap_s=g) for i,g in enumerate([0.,1.,3.,3.,5.])]
    base=dict(tiers={t:dict(prediction_cuts_s=[5.] if t=='noisy' else []) for t in ('online','noisy')})
    return dict(format='agentkv.horizon_probability.v1',elapsed_grid_s=[0.,1.,2.,3.,5.],tiers=empirical(base,rows))


def test_conditional_cdf_strict_survival_and_inclusive_horizon():
    t=HorizonProbabilityTable(fixture())
    assert t.probability(0,0,1)==.25
    assert t.probability(0,1,2)==2/3
    assert t.probability(0,3,2)==1
    assert t.probability(0,5,10)==0
    assert t.probability(0,0,0)==0
    assert t.probability(0,0,100)==1
    with pytest.raises(ValueError):t.probability(0,1,-1)


def test_hint_sources_have_separate_conditional_distributions():
    model=fixture();online=HorizonProbabilityTable(model);noisy=HorizonProbabilityTable(model,'noisy')
    assert online.probability(0,0,1)==.25
    assert noisy.probability(noisy.bucket(1),0,1)==0
    assert noisy.probability(noisy.bucket(10),0,1)==1
