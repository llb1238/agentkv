"""Commit-only measured runs, explicit debug warnings, and result relocation integrity."""
import json
import subprocess
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench import capacity
from agentkv.bench.trace_loader import sha256_file
from agentkv.artifacts import resolve_result_path, verify_artifact


@pytest.fixture
def repository(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','--allow-empty','-qm','init'],check=True)
    return repo


def test_clean_commit_only_and_debug_warning_even_when_clean(repository,capsys):
    clean=git_commit(repository=repository)
    assert len(clean)==40 and isinstance(clean,str)
    assert git_commit(True,repository=repository)==clean
    assert '\x1b[1;31mWARNING:' in capsys.readouterr().err
    (repository/'new').write_text('not committed')
    with pytest.raises(ValueError,match='dirty'):
        git_commit(repository=repository)
    assert git_commit(True,repository=repository)==clean
    assert 'DEBUG ONLY' in capsys.readouterr().err
    subprocess.run(['git','-C',str(repository),'add','new'],check=True)
    with pytest.raises(ValueError,match='dirty'):
        git_commit(repository=repository)


def test_capacity_refuses_dirty_before_input_or_output_access(repository,tmp_path,monkeypatch):
    (repository/'dirty').write_text('x')
    monkeypatch.setattr(capacity,'git_commit',lambda allowed:git_commit(allowed,repository=repository))
    output=tmp_path/'not-created'
    with pytest.raises(ValueError,match='dirty'):
        capacity.main(['--workloads',str(tmp_path/'missing.parquet'),'--output-dir',str(output)])
    assert not output.exists()


def test_debug_capacity_writes_commit_only_and_no_source_copies(repository,tmp_path,monkeypatch,capsys):
    (repository/'dirty').write_text('x')
    monkeypatch.setattr(capacity,'git_commit',lambda allowed:git_commit(allowed,repository=repository))
    workload=tmp_path/'input/workload.parquet';workload.parent.mkdir()
    row=dict(root_id='r',chain_id='c',chain_turn=0,trace_req_idx='q',prompt_token_ids=[1,2,3],
             prompt_len=3,max_tokens=1,anchor_trace_req_idx=None,start_delay_s=0.,gap_after_s=None,final=True)
    pq.write_table(pa.Table.from_pylist([row]),workload)
    workload.with_name('manifest.json').write_text(json.dumps({'workload':{'sha256':sha256_file(workload)},
                                                              'parameters':{'context_budget':16}}))
    model=tmp_path/'model.json';model.write_text(json.dumps({'num_hidden_layers':1,'num_key_value_heads':1,'head_dim':4}))
    output=tmp_path/'capacity'
    capacity.main(['--workloads',str(workload),'--output-dir',str(output),'--model-config',str(model),
                   '--duration','1','--warmup','0','--allow-dirty'])
    run=json.loads((output/'run.json').read_text())
    assert run['git_commit']==git_commit(True,repository=repository,warn=False)
    assert all(k not in run for k in ('git','git_worktree_dirty','git_at_end','implementation_sha256'))
    assert not list(output.rglob('*.py'))
    assert not (output/'implementation').exists()
    assert 'DEBUG ONLY' in capsys.readouterr().err


def test_archived_paths_and_explicit_readme_migration_are_verifiable():
    root=Path(__file__).resolve().parents[1]
    old=root/'results/m1-online-lru-seed0-final/run.json'
    new=resolve_result_path(old)
    assert new==root/'results/archive/m1-online-lru-seed0-final/run.json'
    if not new.exists() and (root/'RELEASE_MANIFEST.json').exists():
        pytest.skip('Historical bytes are in private evidence; relocation logic tested with synthetic fixture')
    assert new.exists()
    manifest=json.loads((root/'results/relocations.json').read_text())
    for change in manifest['presentation_changes']:
        assert verify_artifact(root/change['original_path'],change['original_sha256'])
        assert not verify_artifact(root/change['original_path'],'0'*64)


def test_results_index_covers_all_experiment_directories():
    root=Path(__file__).resolve().parents[1]/'results'
    index=(root/'INDEX.md').read_text()
    paths=[p for p in root.iterdir() if p.is_dir() and p.name!='archive']
    if (root/'archive').exists():
        paths.extend(p for p in (root/'archive').iterdir() if p.is_dir())
    for path in paths:
        assert f']({path.relative_to(root).as_posix()}/)' in index,path


def test_replay_cli_refuses_dirty_before_tokenizer_or_output(repository,tmp_path,monkeypatch):
    from agentkv.bench import replay
    (repository/'dirty').write_text('x')
    monkeypatch.setattr(replay,'git_commit',lambda allowed,**kw:git_commit(allowed,repository=repository,**kw))
    output=tmp_path/'replay-not-created'
    with pytest.raises(ValueError,match='dirty'):
        replay.main(['--output-dir',str(output),'--tokenizer',str(tmp_path/'missing-tokenizer')])
    assert not output.exists()
