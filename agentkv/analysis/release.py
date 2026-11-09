"""Build a local current-tree snapshot, excluding Git history and local payloads."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

from agentkv.provenance import ROOT, git_commit


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root, paths):
    entries = []
    for name in sorted(paths):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Release input must be a regular file: {name}')
        entries.append(dict(path=name, bytes=path.stat().st_size,
                            sha256=file_sha256(path)))
    return entries


def export_full(output, repository=ROOT):
    code = git_commit(repository=repository)
    if output.exists():
        raise FileExistsError('Use a new release directory')
    output = output.resolve()
    repository = repository.resolve()
    if output.is_relative_to(repository):
        raise ValueError('Release directory must be outside the source checkout')
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repository).decode().split('\0')
    names = [p for p in names if p]
    forbidden = read_external_payloads(repository)
    if any('.git' in Path(p).parts or p in forbidden for p in names):
        raise ValueError('Git history or external payload entered tracked release inputs')
    payloads = inventory(repository, names)
    output.mkdir(parents=True)
    for item in payloads:
        destination = output / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / item['path'], destination)
        shutil.copymode(repository / item['path'], destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f"Copied bytes differ: {item['path']}")
    record = dict(format='agentkv.release_snapshot.v1', source_commit=code,
                  contains_git_history=False, external_payloads_included=False,
                  source_files=len(payloads), source_bytes=sum(p['bytes'] for p in payloads),
                  total_files=len(payloads) + 2, total_bytes=0,
                  inventory_files=['FILELIST.tsv', 'RELEASE_MANIFEST.json'], files=payloads)
    sizes = {p: 0 for p in record['inventory_files']}
    # Inventories list themselves with sizes but no recursive self-hashes.
    # Iterate until byte counts stop changing as decimal widths stabilize.
    for _ in range(10):
        table = io.StringIO()
        writer = csv.writer(table, delimiter='\t', lineterminator='\n')
        writer.writerow(('path', 'bytes', 'sha256'))
        for item in payloads:
            writer.writerow((item['path'], item['bytes'], item['sha256']))
        for name in record['inventory_files']:
            writer.writerow((name, sizes[name], 'inventory-self-reference'))
        (output / 'FILELIST.tsv').write_text(table.getvalue())
        (output / 'RELEASE_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
        current = {name: (output / name).stat().st_size for name in sizes}
        total = record['source_bytes'] + sum(current.values())
        if current == sizes and total == record['total_bytes']:
            break
        sizes = current
        record['total_bytes'] = total
    else:
        raise AssertionError('Release inventory sizes did not converge')
    assert len(list(p for p in output.rglob('*') if p.is_file())) == record['total_files']
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) == record['total_bytes']
    assert not (output / '.git').exists()
    assert not any((output / p).exists() for p in forbidden)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, indent=2))
    return record


def export(output, repository=ROOT, *, profile='full', evidence_path=None, reuse_evidence=False):
    if profile == 'full':
        return export_full(output, repository)
    if profile != 'lean':
        raise ValueError(f'Unknown release profile: {profile}')
    from agentkv.analysis.release_lean import export_lean
    return export_lean(output, repository, evidence_path=evidence_path, reuse_evidence=reuse_evidence)


def read_external_payloads(repository):
    manifest = repository / 'docs/publication_payloads.json'
    paths = set()
    if manifest.exists():
        paths.update(r['path'] for r in json.loads(manifest.read_text())['files'])
    paths.update(('data/raw/agentx/cc-traces-weka-062126/traces.jsonl',
                  'results/m1-workload-seed0-v3/workload.parquet',
                  'results/m1-workload-seed0-v3-16k/workload.parquet'))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=('full', 'lean'), default='full')
    parser.add_argument('--evidence-file', type=Path,
                        help='lean only; defaults to agentkv-evidence-20261108.tar.zst beside output')
    parser.add_argument('--reuse-evidence', action='store_true', help='lean only; verify and reuse existing archive unchanged')
    args = parser.parse_args()
    export(args.output_dir, profile=args.profile, evidence_path=args.evidence_file, reuse_evidence=args.reuse_evidence)


if __name__ == '__main__':
    main()
