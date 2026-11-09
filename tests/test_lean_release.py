import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from agentkv.analysis.final_report import summary_state
from agentkv.analysis.release import export
from agentkv.analysis.release_lean import MAX_FILE, limits, publication_status, selected
from agentkv.analysis.release_links import VLLM_URL, check_links, external_error, link_counts, redirect_links
from agentkv.artifacts import resolve_result_path, verify_artifact
from agentkv.provenance import current_commit, git_commit


def test_whitelist_exclusions_take_priority_and_5_mib_boundary():
    assert not selected('AGENTS.md', 100)
    for name in ('docs/light-a.json', 'agentkv/data.bin', 'tests/requests-a.jsonl',
                 'results/final-summary-20261108/checkpoint.json', 'results/archive/a/README.md',
                 'results/a/requests-metadata.parquet', 'results/a/per_request_nominal.csv'):
        assert not selected(name, 1), name
    assert selected('agentkv/policy/data/horizon_probability_v1.json', 5127926)
    assert selected('results/a/attempt-2/e1-analysis.json', 999999)
    assert not selected('results/kv-hints-audit-20261102/pr53423_discussion.json', 100)
    assert not selected('results/kv-hints-audit-20261102/rfc59708.json', 100)
    assert selected('results/kv-hints-audit-20261102/community-summary.md', 100)
    assert not selected('results/a/big.json', 1000001)
    limits([dict(path='table.json', bytes=MAX_FILE)])
    with pytest.raises(ValueError, match='table.json'):
        limits([dict(path='table.json', bytes=MAX_FILE + 1)])
    with pytest.raises(ValueError, match='total_bytes'):
        limits([dict(path=str(i), bytes=MAX_FILE) for i in range(20)])


def test_link_rewrite_retains_code_and_redirects_inline_reference_and_html(tmp_path):
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/EVIDENCE.md').write_text('# Evidence\n## requests\n')
    text = ('[missing](results/a/requests.parquet)\n[known](docs/EVIDENCE.md)\n'
            '[ref]: results/a/requests.parquet\n<img src="results/a/requests.parquet">\n'
            '`[example](not-a-link)`\n```md\n[example](also-not-a-link)\n```\n')
    (tmp_path/'README.md').write_text(text)
    changes = redirect_links(tmp_path, lambda _: 'requests')
    assert len(changes) == 3
    assert not check_links(tmp_path)['failures']
    assert '`[example](not-a-link)`' in (tmp_path/'README.md').read_text()


def test_backend_links_use_pinned_upstream_or_exact_patch_section(tmp_path):
    docs = tmp_path/'docs'
    (docs/'patches').mkdir(parents=True)
    (docs/'patches/vllm-patches.json').write_text(json.dumps({'patches':[{'changed_files':[
        {'path':'vllm/v1/kv_offload/base.py'}]}]}))
    (docs/'patches/README.md').write_text('<a id="file-vllm-v1-kv-offload-base-py"></a>')
    (docs/'EVIDENCE.md').write_text('private runtime evidence')
    (docs/'note.md').write_text('[public](../../vllm/vllm/v1/request.py:17)\n'
                                '[patched](../../vllm/vllm/v1/kv_offload/base.py)\n'
                                '[private](../../vllm/benchmarks/results/a/summary.json)')
    before = link_counts(tmp_path)
    changes = redirect_links(tmp_path, lambda _: 'external-inputs')
    after = link_counts(tmp_path)
    assert before['vllm_local_upstream']==before['vllm_local_patched']==before['vllm_local_evidence']==1
    assert after['vllm_upstream_permalink']==after['patch_section']==after['evidence']==1
    assert VLLM_URL+'vllm/v1/request.py#L17' in (docs/'note.md').read_text()
    assert len(changes)==3 and not check_links(tmp_path)['failures']
    assert external_error('https://github.com/vllm-project/vllm/blob/main/LICENSE')
    assert external_error(VLLM_URL+'vllm/v1/request.py#L0')
    assert external_error(VLLM_URL+'vllm/v1/request.py#L17') is None


def test_publication_guard_rejects_stale_repository_status_but_keeps_evidence_private(tmp_path):
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/PLAN.md').write_text('仓库仍为 **private**；不创建公开仓库。')
    with pytest.raises(ValueError,match='Outdated publication status'):
        publication_status(tmp_path)
    (tmp_path/'docs/PLAN.md').write_text('公开版是无历史lean快照，私有仓库保留完整历史与证据。')
    (tmp_path/'docs/EVIDENCE.md').write_text('证据包暂不公开，可应要求提供。')
    assert publication_status(tmp_path)['outdated_statements']==0


def test_snapshot_provenance_is_presentation_only(tmp_path):
    manifest = dict(format='agentkv.release_snapshot.v1', profile='lean', source_commit='b'*40)
    (tmp_path/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest))
    assert current_commit(tmp_path) == 'b'*40
    with pytest.raises(subprocess.CalledProcessError):
        git_commit(repository=tmp_path)


def test_relocated_readme_original_hash_survives_display_changes(tmp_path):
    original = b'original report'
    path = tmp_path/'results/archive/old/README.md'
    path.parent.mkdir(parents=True)
    path.write_text('relative display links')
    old = 'results/old/README.md'
    record = dict(directories={'results/old':'results/archive/old'}, presentation_changes=[
        dict(path='results/archive/old/README.md', original_path=old,
             original_sha256=hashlib.sha256(original).hexdigest(),
             current_sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    (tmp_path/'results/relocations.json').write_text(json.dumps(record))
    assert resolve_result_path(old, tmp_path) == path
    assert verify_artifact(old, hashlib.sha256(original).hexdigest(), tmp_path)
    assert not verify_artifact(old, '0'*64, tmp_path)


def test_experiment_log_file_move_has_explicit_source_and_display_hashes():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'results/relocations.json').read_text())
    item = next(r for r in data['file_moves'] if r['original_path']=='EXPERIMENT_LOG.md')
    assert resolve_result_path('EXPERIMENT_LOG.md')==root/'docs/EXPERIMENT_LOG.md'
    assert verify_artifact('EXPERIMENT_LOG.md',item['original_sha256'])


def test_summary_state_requires_individual_analysis_not_just_saved_mean(tmp_path):
    path = tmp_path/'g/lru/attempt-1/e1-analysis.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(effective_recompute=.25, ttft_ms=dict(p50=1., p99=2.))))
    saved = dict(status='complete', preregistration_commit='a', protocol_sha256='hash', rows=[
        dict(id='g/lru', rate=1., seed=3, policy='lru', status='complete', attempts=1,
             effective_recompute=.25, ttft_p50_ms=1., ttft_p99_ms=2.)])
    protocol = dict(groups=[dict(rate=1., seed=3, name='g')])
    state, inputs = summary_state(tmp_path, saved, protocol)
    assert inputs == [path] and state['runs'][0]['result']['effective_recompute'] == .25
    path.write_text(path.read_text().replace('0.25', '0.26'))
    with pytest.raises(ValueError, match='matching retained analysis'):
        summary_state(tmp_path, saved, protocol)


@pytest.mark.skipif(shutil.which('zstd') is None, reason='Evidence export requires zstd')
def test_lean_and_evidence_preserve_originals_and_have_complete_inventory(tmp_path):
    source = tmp_path/'source'
    (source/'docs').mkdir(parents=True)
    (source/'results/archive/old').mkdir(parents=True)
    (source/'results/run').mkdir()
    (source/'README.md').write_text('[old](results/archive/old/README.md)')
    (source/'results/archive/old/README.md').write_text('failed, preserved')
    (source/'results/run/light-test.json').write_text('{"raw":true}')
    (source/'results/run/e1-analysis.json').write_text('{"summary":true}')
    original_gpu = b'{"processes":"GPU-example, 123, unrelated-python, 99"}'
    (source/'results/run/analysis.json').write_bytes(original_gpu)
    (source/'AGENTS.md').write_text('Private repository instructions')
    community = source/'results/kv-hints-audit-20261102'
    community.mkdir()
    (community/'pr53423_discussion.json').write_text('[{"body":"original community words"}]')
    (community/'community-summary.md').write_text('Author summary and original-post URL')
    (source/'docs/EVIDENCE.md').write_text('# Evidence\n@@ARCHIVE@@\n@@SHA256@@\n@@CATEGORIES@@\n## archives\n')
    for name in ('DATASET.md', 'REPRODUCING.md', 'patches/README.md'):
        path = source/'docs'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture documentation')
    subprocess.run(['git','init','-q',str(source)], check=True)
    subprocess.run(['git','-C',str(source),'add','.'], check=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','fixture'], check=True)
    output, archive = tmp_path/'lean', tmp_path/'evidence.tar.zst'
    result = export(output, source, profile='lean', evidence_path=archive)
    assert not (output/'results/archive').exists() and not (output/'results/run/light-test.json').exists()
    assert (output/'results/run/e1-analysis.json').exists()
    assert not (output/'AGENTS.md').exists()
    published = next(r for r in result['files'] if r['path']=='results/run/analysis.json')
    assert published['original_sha256']==hashlib.sha256(original_gpu).hexdigest()
    assert published['published_sha256']==published['sha256'] and published['change_reasons']
    assert not (output/'results/kv-hints-audit-20261102/pr53423_discussion.json').exists()
    assert (output/'results/kv-hints-audit-20261102/community-summary.md').exists()
    assert result['profile']=='lean' and result['evidence']['files']>=4
    assert result['link_check']['failures']==[] and result['link_redirects']
    assert not (output/'.git').exists() and '@@SHA256@@' not in (output/'docs/EVIDENCE.md').read_text()
    assert result['evidence']['archive_sha256'] in (output/'docs/EVIDENCE.md').read_text()
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==result['evidence']['archive_sha256']
    process = subprocess.Popen(['zstd','-dc',str(archive)], stdout=subprocess.PIPE)
    restored = {}
    with tarfile.open(fileobj=process.stdout, mode='r|') as stream:
        for member in stream:
            restored[member.name] = stream.extractfile(member).read()
    process.stdout.close()
    assert process.wait()==0
    assert restored['README.md']==(source/'README.md').read_bytes()
    assert restored['results/run/light-test.json']==b'{"raw":true}'
    assert restored['results/run/analysis.json']==original_gpu
    assert restored['results/kv-hints-audit-20261102/pr53423_discussion.json']==b'[{"body":"original community words"}]'
    assert 'FILELIST.tsv' in restored
    for row in result['files']:
        assert hashlib.sha256((output/row['path']).read_bytes()).hexdigest()==row['sha256']
    evidence_files=[archive, archive.with_name('evidence.FILELIST.tsv'), archive.with_name('evidence.json')]
    fingerprints={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_files}
    reused=export(tmp_path/'lean-reuse',source,profile='lean',evidence_path=archive,reuse_evidence=True)
    assert reused['evidence']['reused_unchanged']
    assert fingerprints=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_files}
