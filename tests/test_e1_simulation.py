import math
from agentkv.bench.e1_simulation import paired,summarize,TIERS,COMPONENTS


def item(seed,hint,oracle=.1):
    def result(fraction):return dict(summary=dict(source_fractions=dict(recomputed=fraction)))
    cases=dict(lru=result(.3),oracle=result(oracle))
    for tier in TIERS:
        cases['hint-'+tier]=result(hint)
        for component in COMPONENTS:cases[component+'-'+tier]=result(hint+.02)
    return dict(name=str(seed),rate=.5,seed=seed,cases=cases)


def test_report_uses_paired_means_ranges_and_unclipped_oracle_capture():
    report=dict(results=[item(0,.2),item(1,.4),item(2,.05)])
    pairs,aggregates,ablations=summarize(report)
    group=[p for p in pairs if p['tier']=='perfect']
    assert math.isclose(group[1]['oracle_capture'],-.5)
    assert math.isclose(group[2]['oracle_capture'],1.25)
    stats=next(r for r in aggregates if r['tier']=='perfect')
    assert math.isclose(stats['mean_pp'],25/3)
    assert math.isclose(stats['minimum_pp'],-10) and math.isclose(stats['maximum_pp'],25)
    assert math.isclose(stats['capture_mean'],5/12)
    assert all(math.isclose(r['mean_pp'],2) for r in ablations)


def test_zero_oracle_space_is_not_a_fake_zero_capture():
    assert paired(item(0,.2,oracle=.3),'perfect')['oracle_capture'] is None
