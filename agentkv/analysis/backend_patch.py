"""Export and CPU-verify the exact vLLM patches used in recorded experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

BASE = '5df63a7203cd'
TARGET = '4749dd5961841243596559b10ba0a22c48980c45'


def git(repository, *args):
    return subprocess.check_output(['git', '-C', str(repository), *args])


def export(backend, output):
    if git(backend, 'status', '--porcelain').strip():
        raise ValueError('Backend checkout must be clean; export does not modify it')
    output.mkdir(parents=True, exist_ok=True)
    parent = git(backend, 'rev-parse', TARGET + '^').decode().strip()
    entries = []
    for name, base in (('vllm-agentkv-final.patch', BASE), ('vllm-prefix-metadata.patch', parent)):
        path = output / name
        if path.exists():
            raise FileExistsError(path)
        patch = git(backend, 'diff', '--binary', '--full-index', base, TARGET)
        path.write_bytes(patch)
        paths = git(backend, 'diff', '--name-only', base, TARGET).decode().splitlines()
        checked = []
        with tempfile.TemporaryDirectory(prefix='agentkv-patch-check-') as folder:
            root = Path(folder)
            for relative in paths:
                old = subprocess.run(['git', '-C', str(backend), 'show', f'{base}:{relative}'], capture_output=True)
                if old.returncode == 0:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(old.stdout)
            subprocess.run(['git', 'apply', '--check', str(path.resolve())], cwd=root, check=True)
            subprocess.run(['git', 'apply', str(path.resolve())], cwd=root, check=True)
            for relative in paths:
                expected = git(backend, 'show', f'{TARGET}:{relative}')
                actual = (root / relative).read_bytes()
                if actual != expected:
                    raise AssertionError(relative)
                checked.append(dict(path=relative, sha256=hashlib.sha256(actual).hexdigest()))
        entries.append(dict(path=name, base_commit=git(backend, 'rev-parse', base).decode().strip(),
                            target_commit=TARGET, bytes=len(patch), sha256=hashlib.sha256(patch).hexdigest(),
                            apply_check_passed=True, all_changed_files_match_target=True, changed_files=checked))
    result = dict(format='agentkv.backend_patch_bundle.v1', patches=entries,
                  backend_worktree_modified=False, gpu_runs_started=0)
    (output / 'vllm-patches.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.backend.resolve(), args.output_dir)


if __name__ == '__main__':
    main()
