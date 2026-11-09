"""Resolve historical result paths and verify explicitly migrated presentation files."""
import hashlib
import json
from pathlib import Path
from .provenance import ROOT


def resolve_result_path(path, repository=ROOT):
    path=Path(path)
    relative=path.relative_to(repository) if path.is_absolute() and path.is_relative_to(repository) else path
    manifest=Path(repository)/'results/relocations.json'
    if manifest.exists():
        data=json.loads(manifest.read_text())
        relative=Path(data.get('files',{}).get(relative.as_posix(),relative))
        for original,destination in data['directories'].items():
            if relative.is_relative_to(original):
                relative=Path(destination)/relative.relative_to(original)
                break
    return relative if relative.is_absolute() else Path(repository)/relative


def verify_artifact(path, expected_sha256, repository=ROOT):
    resolved=resolve_result_path(path,repository)
    actual=hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual==expected_sha256:
        return True
    manifest=Path(repository)/'results/relocations.json'
    if not manifest.exists() or not resolved.is_relative_to(repository):
        return False
    relative=resolved.relative_to(repository).as_posix()
    for item in json.loads(manifest.read_text()).get('presentation_changes',[]):
        if item['path']==relative and expected_sha256==item['original_sha256'] and actual==item['current_sha256']:
            # Only link-edited reports are allowed through this exception.
            return resolved.name=='README.md'
    for item in json.loads(manifest.read_text()).get('file_moves',[]):
        if item['path']==relative and expected_sha256==item['original_sha256'] and actual==item['current_sha256']:
            return resolved.suffix=='.md'
    return False
