"""Whitelist-based lean snapshots and byte-preserving, private evidence archives."""
from collections import defaultdict
import csv
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from agentkv.analysis.release import file_sha256, inventory, read_external_payloads
from agentkv.analysis.release_links import check_links, destination, link_counts, redirect_links
from agentkv.provenance import git_commit
from agentkv.analysis.publication_audit import audit_release
from agentkv.analysis.release_privacy import redact_tree

MAX_FILE = 5 * 1024**2  # User confirmed 5 MiB, retaining the horizon table.
MAX_TOTAL = 100_000_000
SUMMARY_MAX = 1_000_000
EXCLUDE = ('light-*.json', 'loop_lag.parquet', 'requests.parquet', 'request-metadata.parquet',
           'requests-metadata.parquet', 'callbacks.parquet', 'decisions.parquet', 'dmon.log',
           'metrics.jsonl', 'server.log', 'requests-*.jsonl', 'checkpoint.json', '*.bin',
           'per_request_nominal.csv')
ROOT_FILES = {'README.md', 'LICENSE', 'NOTICE', 'pyproject.toml', '.gitignore',
              'setup.cfg', 'setup.py', 'MANIFEST.in', 'pytest.ini',
              'requirements.txt'}
COMMUNITY_DIR = 'results/kv-hints-audit-20261102/'


def community_original(name):
    return name.startswith(COMMUNITY_DIR) and Path(name).suffix == '.json' and Path(name).name.startswith(('pr', 'rfc'))


def publication_status(root):
    expressions = (r'不创建公开仓库', r'不公开仓库', r'不改变远端私有状态',
                   r'仍(?:然)?(?:为|是).{0,20}private', r'\bstill\s+private\b',
                   r'用户.*决定公开方式', r'今后若用户批准建立公开')
    failures = []
    pages = sorted((Path(root) / 'docs').rglob('*.md'))
    for page in pages:
        for number, line in enumerate(page.read_text().splitlines(), 1):
            if any(re.search(p, line, re.I) for p in expressions):
                failures.append(dict(path=page.relative_to(root).as_posix(), line=number, text=line))
    if failures:
        raise ValueError('Outdated publication status in docs: ' + json.dumps(failures, ensure_ascii=False))
    return dict(markdown_files_checked=len(pages), outdated_statements=0,
                evidence_withheld_statement_is_current=True)


def selected(name, size):
    path = Path(name)
    if community_original(name) or name.startswith('results/archive/') or any(fnmatch.fnmatchcase(path.name, p) for p in EXCLUDE):
        return False
    if path.parts[0] in ('agentkv', 'tests', 'docs') or name in ROOT_FILES:
        return True
    if name in ('results/INDEX.md', 'results/relocations.json', COMMUNITY_DIR + 'community-summary.md') or name.startswith('results/final-summary-20261108/'):
        return True
    return (path.parts[0] == 'results' and
            (path.name == 'README.md' or path.suffix in ('.png', '.svg', '.pdf') or
             (path.suffix in ('.json', '.csv') and size <= SUMMARY_MAX)))


def limits(entries):
    oversized = [dict(path=p['path'], bytes=p['bytes']) for p in entries if p['bytes'] > MAX_FILE]
    total = sum(p['bytes'] for p in entries)
    if oversized or total > MAX_TOTAL:
        report = dict(total_bytes=total, total_limit_bytes=MAX_TOTAL, file_limit_bytes=MAX_FILE,
                      oversized_files=oversized,
                      largest_files=sorted(entries, key=lambda p: -p['bytes'])[:10])
        raise ValueError('Lean size limit exceeded; no automatic deletion: ' + json.dumps(report))


def evidence_category(name):
    path = Path(name)
    if community_original(name):
        return 'community-originals'
    if name.startswith('results/archive/'):
        return 'archives'
    if name.startswith('data/') or path.name == 'workload.parquet':
        return 'external-inputs'
    if 'callback-traces' in path.parts or path.suffix in ('.bin', '.gz') or path.name == 'callbacks.parquet':
        return 'raw-traces'
    if path.suffix == '.parquet' or path.name == 'per_request_nominal.csv' or fnmatch.fnmatchcase(path.name, 'requests-*.jsonl'):
        return 'requests'
    if path.name == 'checkpoint.json':
        return 'checkpoints'
    if path.suffix in ('.log', '.jsonl', '.pstats') or fnmatch.fnmatchcase(path.name, 'light-*.json'):
        return 'runtime-logs'
    return 'other-evidence'


def public_category(name):
    path = Path(name)
    if len(path.parts) == 1:
        return 'root-and-inventories'
    if path.parts[0] in ('agentkv', 'tests', 'docs'):
        return path.parts[0]
    if path.suffix in ('.png', '.svg', '.pdf'):
        return 'result-figures'
    if path.suffix == '.md':
        return 'result-reports'
    return 'result-summaries'


def categories(entries, classify):
    result = defaultdict(lambda: dict(files=0, bytes=0))
    for p in entries:
        row = result[classify(p['path'])]
        row['files'] += 1
        row['bytes'] += p['bytes']
    return dict(sorted(result.items()))


def tsv(entries):
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(('path', 'bytes', 'sha256'))
    for item in sorted(entries, key=lambda p: p['path']):
        writer.writerow((item['path'], item['bytes'], item['sha256']))
    return stream.getvalue().encode()


def verify_evidence(output, sidecar):
    table = Path(sidecar).read_bytes()
    expected = {r['path']: r for r in csv.DictReader(io.StringIO(table.decode()), delimiter='\t')}
    seen = set()
    process = subprocess.Popen(['zstd', '-q', '-dc', str(output)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name in seen:
                    raise ValueError('Non-regular or duplicate evidence member')
                seen.add(member.name)
                stream = archive.extractfile(member)
                if member.name == 'FILELIST.tsv':
                    assert stream.read() == table
                    continue
                item = expected[member.name]
                assert member.size == int(item['bytes'])
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                assert digest.hexdigest() == item['sha256'], member.name
        assert seen == set(expected) | {'FILELIST.tsv'}
    finally:
        process.stdout.close()
        status = process.wait()
    assert status == 0


def evidence_archive(repository, names, output, code):
    """Stream tar through zstd; archive members contain exact source bytes."""
    output = Path(output).resolve()
    sidecar = output.with_name(output.name.removesuffix('.tar.zst') + '.FILELIST.tsv')
    metadata = output.with_name(output.name.removesuffix('.tar.zst') + '.json')
    partial = output.with_name(output.name + '.partial')
    if any(p.exists() for p in (output, sidecar, metadata, partial)):
        raise FileExistsError('Use new evidence filenames; existing evidence is never overwritten')
    entries = inventory(repository, names)
    table = tsv(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['zstd', '-q', '-T2', '-3', '-o', str(partial)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            item = tarfile.TarInfo('FILELIST.tsv')
            item.size, item.mtime, item.mode = len(table), 0, 0o644
            archive.addfile(item, io.BytesIO(table))
            for entry in entries:
                path = repository / entry['path']
                item = tarfile.TarInfo(entry['path'])
                item.size, item.mtime, item.mode = entry['bytes'], 0, path.stat().st_mode & 0o777
                with path.open('rb') as stream:
                    archive.addfile(item, stream)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('zstd failed; partial file retained for diagnosis')
    except BaseException:
        process.stdin.close()
        process.wait()
        raise
    partial.rename(output)
    sidecar.write_bytes(table)
    verify_evidence(output, sidecar)
    digest = file_sha256(output)
    result = dict(format='agentkv.evidence.v1', source_commit=code, archive=output.name,
                  archive_bytes=output.stat().st_size, archive_sha256=digest,
                  files=len(entries), uncompressed_bytes=sum(p['bytes'] for p in entries),
                  filelist_sha256=hashlib.sha256(table).hexdigest(), categories=categories(entries, evidence_category),
                  contains_git_history=False, all_members_sha256_verified=True, public=False)
    metadata.write_text(json.dumps(result, indent=2) + '\n')
    return result


def reuse_evidence_archive(output):
    """Verify the original archive and sidecars without modifying any bytes."""
    output = Path(output).resolve()
    prefix = output.name.removesuffix('.tar.zst')
    sidecar = output.with_name(prefix + '.FILELIST.tsv')
    metadata = output.with_name(prefix + '.json')
    record = json.loads(metadata.read_text())
    if record['archive_bytes'] != output.stat().st_size or record['archive_sha256'] != file_sha256(output):
        raise ValueError('Existing evidence archive fingerprint changed')
    if record['filelist_sha256'] != file_sha256(sidecar):
        raise ValueError('Existing evidence filelist fingerprint changed')
    verify_evidence(output, sidecar)
    return dict(record, reused_unchanged=True)


def evidence_document(template, record):
    rows = '\n'.join(f"| {key} | {v['files']} | {v['bytes']:,} |" for key, v in record['categories'].items())
    for marker, value in {'@@ARCHIVE@@': record['archive'], '@@BYTES@@': str(record['archive_bytes']),
                           '@@SHA256@@': record['archive_sha256'], '@@SOURCE@@': record['source_commit'],
                           '@@FILELIST_SHA256@@': record['filelist_sha256'], '@@CATEGORIES@@': rows}.items():
        template = template.replace(marker, value)
    return template


def finalize(output, code, source_entries, changes, evidence, links, *, publication=None, target_counts=None, privacy=None):
    paths = [p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()]
    payloads = inventory(output, paths)
    originals = {p['path']: p['sha256'] for p in source_entries}
    privacy_changes = {p['path']: p for p in (privacy or {}).get('changed_files', [])}
    linked = {p['path'] for p in changes}
    for p in payloads:
        if p['path'] in originals and originals[p['path']] != p['sha256']:
            p['original_sha256'] = originals[p['path']]
            p['published_sha256'] = p['sha256']
            p['change_reasons'] = []
            if p['path'] in privacy_changes:
                p['change_reasons'].append(privacy_changes[p['path']]['reason'])
                privacy_changes[p['path']]['published_sha256'] = p['sha256']
            if p['path'] in linked:
                p['change_reasons'].append('Redirect publication links to public sources, patches or private evidence')
            if not p['change_reasons']:
                p['change_reasons'].append('Generate publication evidence metadata or explicit presentation relocation mapping')
    record = dict(format='agentkv.release_snapshot.v1', profile='lean', source_commit=code,
                  contains_git_history=False, external_payloads_included=False,
                  source_files=len(payloads), source_bytes=sum(p['bytes'] for p in payloads),
                  total_files=len(payloads) + 2, total_bytes=0, file_limit_bytes=MAX_FILE,
                  total_limit_bytes=MAX_TOTAL, summary_limit_bytes=SUMMARY_MAX,
                  inventory_files=['FILELIST.tsv', 'RELEASE_MANIFEST.json'], files=payloads,
                  link_redirects=changes, link_check=links, evidence=evidence,
                  publication_status_check=publication, link_target_counts=target_counts,
                  privacy_redactions=privacy, privacy_audit=audit_release(output))
    sizes = {p: 0 for p in record['inventory_files']}
    for _ in range(20):
        all_entries = payloads + [dict(path=k, bytes=v, sha256='inventory-self-reference') for k, v in sizes.items()]
        record['categories'] = categories(all_entries, public_category)
        (output / 'FILELIST.tsv').write_bytes(tsv(all_entries))
        (output / 'RELEASE_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
        current = {k: (output / k).stat().st_size for k in sizes}
        total = record['source_bytes'] + sum(current.values())
        if sizes == current and total == record['total_bytes']:
            break
        sizes, record['total_bytes'] = current, total
    else:
        raise AssertionError('Release sizes did not converge')
    limits(all_entries)
    actual = [p for p in output.rglob('*') if p.is_file()]
    assert len(actual) == record['total_files'] and sum(p.stat().st_size for p in actual) == record['total_bytes']
    assert not any(p.name == '.git' for p in output.rglob('*'))
    audit_release(output)  # Include FILELIST and RELEASE_MANIFEST in the final whole-tree gate.
    print(json.dumps({k: v for k, v in record.items() if k not in ('files', 'link_redirects')}, indent=2))
    return record


def export_lean(output, repository, *, evidence_path=None, reuse_evidence=False):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    code = git_commit(repository=repository)
    publication_status(repository)
    if output.exists():
        raise FileExistsError('Use a new lean directory')
    if output.is_relative_to(repository):
        raise ValueError('Release directory must be outside the source checkout')
    names = list(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=repository).decode().split('\0')))
    forbidden = read_external_payloads(repository)
    if any('.git' in Path(p).parts or p in forbidden for p in names):
        raise ValueError('External payload or Git history entered tracked release inputs')
    keep = [p for p in names if selected(p, (repository / p).stat().st_size)]
    source_entries = inventory(repository, keep)
    limits(source_entries)
    for name in keep:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, path)
        shutil.copymode(repository / name, path)
    privacy = redact_tree(output)
    audit_release(output)
    before_counts = link_counts(output)
    changes = redirect_links(output, evidence_category)
    for change in changes:
        original_target = destination(repository, repository / change['path'], change['target'])
        change['source_target_exists'] = original_target.exists()
        change['source_target_inside_repository'] = original_target.is_relative_to(repository)
    changed_paths = {r['path'] for r in changes}
    # Preserve the originals of every link-edited document in the evidence too.
    excluded = (set(names) - set(keep)) | changed_paths | {'docs/EVIDENCE.md'}
    unavailable = []
    for name in sorted(forbidden):
        if (repository / name).is_file():
            excluded.add(name)
        else:
            unavailable.append(name)
    relocation = output / 'results/relocations.json'
    if relocation.exists():
        data = json.loads(relocation.read_text())
        for item in data.get('presentation_changes', []) + data.get('file_moves', []):
            path = output / item['path']
            if path.exists():
                item['current_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        data['release_presentation_changes'] = changes
        relocation.write_text(json.dumps(data, indent=2) + '\n')
        excluded.add('results/relocations.json')
    template = (repository / 'docs/EVIDENCE.md').read_text()
    placeholder = dict(archive='agentkv-evidence-20261108.tar.zst', archive_bytes=0,
                       archive_sha256='0' * 64, source_commit=code, filelist_sha256='0' * 64, categories={})
    (output / 'docs/EVIDENCE.md').write_text(evidence_document(template, placeholder))
    limits(inventory(output, [p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()]))
    evidence_path = Path(evidence_path) if evidence_path else output.parent / 'agentkv-evidence-20261108.tar.zst'
    if evidence_path.resolve().is_relative_to(output) or evidence_path.resolve().is_relative_to(repository):
        raise ValueError('Evidence archive must be outside both source and lean trees')
    excluded.update(item['path'] for item in privacy['changed_files'])
    evidence = reuse_evidence_archive(evidence_path) if reuse_evidence else evidence_archive(repository, sorted(excluded), evidence_path, code)
    if reuse_evidence:
        evidence['coverage_note'] = ('Reused archive keeps its original producer commit and bytes. Newly redacted originals '
                                     'remain in the private repository; their original/published hashes are in this release manifest. '
                                     'Do not assume this older archive contains new source or document versions.')
    evidence['unavailable_external_payloads'] = unavailable
    (output / 'docs/EVIDENCE.md').write_text(evidence_document(template, evidence))
    if reuse_evidence:
        with (output / 'docs/EVIDENCE.md').open('a') as stream:
            stream.write('\n本次复用既有证据包，其字节、SHA和生产者提交保持不变。新增脱敏文件的原件继续保存在私有仓库，'
                         '原SHA、发布SHA和原因见发布清单；不宣称旧包包含本次新增的源码或文档版本。\n')
    links = check_links(output)
    if links['failures']:
        raise ValueError('Release has dead relative links: ' + json.dumps(links))
    return finalize(output, code, source_entries, changes, evidence, links,
                    publication=publication_status(output),
                    target_counts=dict(before=before_counts, after=link_counts(output)), privacy=privacy)
