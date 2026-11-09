import pytest
from agentkv.bench.e0 import RATES
from agentkv.bench.e1_selection import cases,options,select


def report():
    populations=[]
    for rate in RATES:
        for seed in (0,1,2):
            values={'lru':100,'oracle':50,'lru-final':95,'deadline-online':65,'remaining-online':70,'horizon-online':60}
            populations.append(dict(name=f'{rate}-{seed}',rate=rate,seed=seed,cases={
                k:dict(summary=dict(source_tokens=dict(recomputed=n),source_fractions=dict(recomputed=n/100))) for k,n in values.items()}))
    return dict(stage='development',results=populations)


def test_selection_rejects_one_negative_population_despite_larger_mean():
    r=report();bad=r['results'][0]['cases']['horizon-online']['summary']
    bad['source_tokens']['recomputed']=101;bad['source_fractions']['recomputed']=1.01
    choice=select(r)
    assert choice['winner']=='deadline'
    assert not next(c for c in choice['candidates'] if c['candidate']=='horizon')['eligible']


def test_selection_ties_use_fixed_order_and_held_out_cannot_select():
    r=report()
    for p in r['results']:p['cases']['horizon-online']=p['cases']['deadline-online']
    assert select(r)['winner']=='deadline'
    r['stage']='evaluation'
    with pytest.raises(ValueError):select(r)
    with pytest.raises(ValueError):cases('evaluation')
    assert len(cases('development'))==len(cases('evaluation','horizon'))==6
    assert options('perfect','remaining')['scoring']=='deadline'
    assert options('perfect','horizon')['perfect_probability']


def test_evaluation_missing_freeze_cannot_generate_population(tmp_path,monkeypatch):
    import agentkv.bench.e1_selection as module
    monkeypatch.setattr(module,'git_commit',lambda:'test-only')
    def forbidden(*args,**kwargs):raise AssertionError('Hold-out population opened before freeze')
    monkeypatch.setattr(module,'sample_population',forbidden)
    with pytest.raises(FileNotFoundError):module.run('evaluation',tmp_path/'output',development=tmp_path/'dev',freeze_file=tmp_path/'missing.json')
    assert not (tmp_path/'output').exists()
