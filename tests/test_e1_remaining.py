from agentkv.bench.e1_remaining import CASES,MAIN,paired_rows,aggregate_pairs,decision_stats


def test_remaining_matrix_controls_are_explicit_and_capture_is_not_clamped():
    assert len(CASES)==13 and len({c[0] for c in CASES})==13
    assert all(options['tail_first']==label.startswith(('tail-','legacy-')) for label,tier,options in CASES if tier)
    cases={label:dict(summary=dict(source_fractions=dict(recomputed=.3))) for label in MAIN}
    cases['lru']['summary']['source_fractions']['recomputed']=.2
    cases['oracle']['summary']['source_fractions']['recomputed']=.1
    report=dict(results=[dict(name='a',rate=.5,seed=i,cases=cases) for i in (0,1,2)])
    pairs=paired_rows(report)
    assert all(abs(p['reduction_pp']+10)<1e-12 and abs(p['oracle_capture']+1)<1e-12 for p in pairs)
    assert all(abs(r['mean_pp']+10)<1e-12 and r['populations']==3 for r in aggregate_pairs(pairs))


def test_decision_mean_and_pooled_occupancy_are_distinct():
    rows=[dict(cpu_blocks=10,overdue_blocks=5,overdue_fraction=.5),dict(cpu_blocks=100,overdue_blocks=0,overdue_fraction=0.)]
    result=decision_stats(rows)
    assert result['mean_fraction']==.25 and result['pooled_fraction']==5/110


def test_calibration_identity_allows_new_provenance_but_never_new_curves():
    import copy
    from agentkv.bench.e1_remaining import calibration_values,read
    from agentkv.policy.remaining_life import DEFAULT_TABLE
    model=read(DEFAULT_TABLE);rebuilt=copy.deepcopy(model)
    rebuilt['git_commit']='new-calibration-commit'
    rebuilt['source']['manifest_sha256']='new-manifest-metadata'
    assert calibration_values(model)==calibration_values(rebuilt)
    rebuilt['tiers']['online']['buckets'][0]['mean_remaining_s'][0]+=.001
    assert calibration_values(model)!=calibration_values(rebuilt)


def test_recorded_table_with_changed_provenance_preserves_simulated_outputs(tmp_path):
    import json
    from agentkv.bench.e1_remaining import read
    from agentkv.policy.remaining_life import DEFAULT_TABLE,RemainingLifeTable
    from agentkv.sim.hint_adapter import HintPolicyAdapter
    from agentkv.sim.two_tier import simulate
    from tests.test_hint_adapter import fixture
    model=read(DEFAULT_TABLE);model['git_commit']='new-calibration-commit'
    path=tmp_path/'recalibrated.json';path.write_text(json.dumps(model))
    for tier in ('online','noisy'):
        default=HintPolicyAdapter(10,hint_tier=tier)
        recorded=HintPolicyAdapter(10,hint_tier=tier,residual_table=RemainingLifeTable.load(path,tier))
        assert simulate(*fixture(),8,10,2,duration=20,warmup=0,block_size=1,cpu_policy=default)==simulate(
            *fixture(),8,10,2,duration=20,warmup=0,block_size=1,cpu_policy=recorded)
