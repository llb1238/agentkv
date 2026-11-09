"""Commit-only provenance and clean-worktree checks for measured runs."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_commit(allow_dirty=False, repository=ROOT, *, warn=True):
    """Return HEAD; reject tracked/untracked changes unless debugging explicitly."""
    status = subprocess.check_output(
        ['git','status','--porcelain=v1','--untracked-files=all'],cwd=repository,text=True)
    if status and not allow_dirty:
        raise ValueError('Worktree is dirty; commit changes before running, or use --allow-dirty for DEBUG ONLY')
    if allow_dirty and warn:
        print('\033[1;31mWARNING: --allow-dirty — DEBUG ONLY / 仅用于调试；'
              '未提交代码不由 run.json 的提交号覆盖，结果不可作为正式实验。\033[0m',file=sys.stderr,flush=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()


def current_commit(repository=ROOT):
    """Read HEAD for presentation-only analysis; does not authorize measurements."""
    repository = Path(repository)
    manifest = repository / 'RELEASE_MANIFEST.json'
    if not (repository / '.git').exists() and manifest.is_file():
        import json
        import re
        record = json.loads(manifest.read_text())
        commit = record['source_commit']
        if record.get('format') != 'agentkv.release_snapshot.v1' or not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Invalid release provenance')
        return commit
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
