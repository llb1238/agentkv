"""Check and redirect local Markdown links in a release, without touching sources."""
import argparse
from collections import Counter
import json
import posixpath
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

VLLM_COMMIT = '5df63a7203cdb9328d824a27b682e77c7b9928be'
VLLM_URL = f'https://github.com/vllm-project/vllm/blob/{VLLM_COMMIT}/'


def targets(text, *, include_external=False):
    # Keep offsets stable while masking code, where Markdown is not rendered.
    masked = re.sub(r'(?ms)^ *(`{3,}|~{3,})[^\n]*\n.*?^ *(?:`{3,}|~{3,})[^\n]*$',
                    lambda m: re.sub(r'[^\n]', ' ', m[0]), text)
    masked = re.sub(r'`[^`\n]+`', lambda m: ' ' * len(m[0]), masked)
    expressions = (r'!?\[[^\]\n]*\]\((<[^>\n]+>|[^\s)\n]+)',
                   r'(?m)^ {0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)',
                   r'(?:href|src)=["\x27]([^"\x27]+)["\x27]')
    for expression in expressions:
        for match in re.finditer(expression, masked):
            start, end = match.span(1)
            value = text[start:end]
            if value.startswith('<') and value.endswith('>'):
                start, end, value = start + 1, end - 1, value[1:-1]
            parts = urlsplit(value)
            if include_external or (not parts.scheme and not parts.netloc and parts.path):
                yield start, end, value


def destination(root, document, target):
    path = unquote(urlsplit(target).path)
    if not path:
        return document.resolve()
    return (root / path.lstrip('/') if path.startswith('/') else document.parent / path).resolve()


def backend_target(target):
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return None
    match = re.fullmatch(r'(?:\.\./)+vllm/(.+)', unquote(parts.path))
    if not match:
        return None
    path = match[1]
    line = re.search(r':([1-9][0-9]*)(?::[0-9]+)?$', path)
    anchor = '#L' + line[1] if line else ('#' + parts.fragment if parts.fragment else '')
    path = path[:line.start()] if line else path
    if '..' in Path(path).parts:
        raise ValueError(f'Unsafe backend source path: {target}')
    return path, anchor


def patch_sections(root):
    manifest = Path(root) / 'docs/patches/vllm-patches.json'
    if not manifest.exists():
        return {}
    files = {p['path'] for patch in json.loads(manifest.read_text())['patches'] for p in patch['changed_files']}
    return {path: 'file-' + re.sub('[^a-z0-9]+', '-', path.lower()).strip('-') for path in files}


def documents(root):
    root = Path(root)
    return sorted({root / 'README.md', *root.glob('docs/**/*.md'), *root.glob('results/**/*.md')})


def target_kind(root, document, target, patches=None):
    patches = patch_sections(root) if patches is None else patches
    backend = backend_target(target)
    if backend:
        path, _ = backend
        if path in patches:
            return 'vllm_local_patched'
        return 'vllm_local_evidence' if path.startswith('benchmarks/results/') else 'vllm_local_upstream'
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return 'vllm_upstream_permalink' if target.startswith(VLLM_URL) else 'external_other'
    path = destination(root, document, target)
    if path == Path(root) / 'docs/EVIDENCE.md':
        return 'evidence'
    if path == Path(root) / 'docs/patches/README.md' and parts.fragment.startswith('file-'):
        return 'patch_section'
    return 'local_other'


def link_counts(root):
    root = Path(root).resolve()
    counts = Counter()
    patches = patch_sections(root)
    for document in documents(root):
        if document.is_file():
            counts.update(target_kind(root, document, target, patches)
                          for _, _, target in targets(document.read_text(), include_external=True))
    return dict(sorted(counts.items()))


def external_error(target):
    parts = urlsplit(target)
    if parts.scheme == 'mailto':
        return None if '@' in parts.path else 'Invalid mailto address'
    if parts.scheme not in ('https', 'http') or not parts.hostname or re.search(r'\s', target):
        return 'Invalid external URL format'
    if parts.hostname == 'github.com':
        code = re.match(r'/([^/]+/[^/]+)/(blob|tree)/([^/]+)(?:/(.*))?$', parts.path)
        if code:
            if not re.fullmatch('[0-9a-f]{40}', code[3]):
                return 'GitHub source URL must pin a full commit SHA'
            if code[1] == 'vllm-project/vllm' and code[3] != VLLM_COMMIT:
                return 'Upstream vLLM source URL must pin the registered baseline'
            if code[2] == 'blob' and not code[4]:
                return 'GitHub blob URL has no file path'
            if parts.fragment and not re.fullmatch(r'L[1-9][0-9]*(?:-L[1-9][0-9]*)?', parts.fragment):
                return 'Invalid source line anchor'
    return None


def check_links(root):
    """Check root README, all docs and retained result READMEs (superset of spec)."""
    root = Path(root).resolve()
    pages = documents(root)
    checked, external, failures = 0, 0, []
    for document in pages:
        if not document.is_file():
            continue
        for _, _, target in targets(document.read_text(), include_external=True):
            checked += 1
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                external += 1
                error = external_error(target)
                if error:
                    failures.append(dict(path=document.relative_to(root).as_posix(), target=target, error=error))
                continue
            path = destination(root, document, target)
            if not path.is_relative_to(root) or not path.exists():
                failures.append(dict(path=document.relative_to(root).as_posix(), target=target))
            elif path == root / 'docs/patches/README.md' and parts.fragment.startswith('file-'):
                if f'id="{parts.fragment}"' not in path.read_text():
                    failures.append(dict(path=document.relative_to(root).as_posix(), target=target, error='Missing patch subsection'))
    return dict(checked=checked, relative_checked=checked-external, external_checked=external,
                external_validation='format and pinned commit only; no network requests',
                failures=failures, documents=len(pages), target_counts=link_counts(root))


def redirect_links(root, category):
    root = Path(root).resolve()
    changes = []
    patches = patch_sections(root)
    index = root / 'docs/patches/upstream-source-index.json'
    record = json.loads(index.read_text()) if index.exists() else None
    if record is not None and record['upstream_commit'] != VLLM_COMMIT:
        raise ValueError('Source index has a different upstream baseline')
    known_sources = record['files'] if record is not None else None
    for document in sorted(root.rglob('*.md')):
        before = document.read_text()
        replacements = []
        for start, end, target in targets(before):
            backend = backend_target(target)
            if backend and not backend[0].startswith('benchmarks/results/'):
                source, anchor = backend
                if source in patches:
                    new = posixpath.relpath('docs/patches/README.md', document.parent.relative_to(root).as_posix()) + '#' + patches[source]
                    kind = 'patch_section'
                else:
                    if known_sources is not None and source not in known_sources:
                        raise ValueError(f'Backend source not verified at upstream baseline: {source}')
                    new, kind = VLLM_URL + source + anchor, 'vllm_upstream_permalink'
                replacements.append((start, end, new))
                changes.append(dict(path=document.relative_to(root).as_posix(), target=target,
                                    replacement=new, target_kind=kind, backend_source=source))
                continue
            path = destination(root, document, target)
            if path.is_relative_to(root) and path.exists():
                continue
            name = path.relative_to(root).as_posix() if path.is_relative_to(root) else target
            section = category(name) if path.is_relative_to(root) else 'external-inputs'
            new = posixpath.relpath('docs/EVIDENCE.md', document.parent.relative_to(root).as_posix()) + '#' + section
            replacements.append((start, end, new))
            changes.append(dict(path=document.relative_to(root).as_posix(), target=target,
                                replacement=new, evidence_section=section, target_kind='evidence'))
        for start, end, new in sorted(set(replacements), reverse=True):
            before = before[:start] + new + before[end:]
        if replacements:
            document.write_text(before)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_links(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
