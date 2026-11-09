"""Read-only prepublication checks; findings never print credential values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from agentkv.provenance import ROOT, current_commit
from agentkv.analysis.release_privacy import privacy_findings


SECRET_PATTERNS = {
    'private_key_material': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{32,}',
    'vendor_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|xox[baprs]-[A-Za-z0-9-]{25,})',
    'credential_url': rb'https?://[^\s/:]{1,80}:[^\s/@]{5,100}@',
    'literal_credential': rb'(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|authorization)["\x27]?\s*[:=]\s*["\x27](?:Bearer )?[A-Za-z0-9_+/=.-]{20,}["\x27]',
}
PRIVATE_ADDRESS = re.compile(
    rb'(?<![\w.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|'
    rb'172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3})(?![\w.])'
    rb'|(?i:(?<![\w:])(?:f[cd][0-9a-f]{2}|fe80):[0-9a-f:]{2,})'
    rb'|\b[a-zA-Z0-9][a-zA-Z0-9.-]*\.(?:internal|lan|corp)\b'
)
ABSOLUTE_DOCUMENT_PATH = re.compile(rb'/(?:home|root|data|tmp|mnt|workspace)(?:/|[\s`])')
OPAQUE_SUFFIXES = {'.gz', '.parquet', '.png', '.pdf', '.bin', '.pstats'}


def scan_text(data, path):
    findings = privacy_findings(data, path)
    for kind, expression in SECRET_PATTERNS.items():
        for match in re.finditer(expression, data):
            findings.append(dict(path=path, kind=kind, line=data.count(b'\n', 0, match.start()) + 1,
                                 value_sha256=hashlib.sha256(match.group()).hexdigest()))
    for match in PRIVATE_ADDRESS.finditer(data):
        findings.append(dict(path=path, kind='private_address',
                             line=data.count(b'\n', 0, match.start()) + 1,
                             value_sha256=hashlib.sha256(match.group()).hexdigest()))
    return findings


def audit_release(root, *, strict=True):
    """Audit every file in a publication tree, without Git or network access."""
    root = Path(root).resolve()
    findings, files, binary = [], 0, 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        files += 1
        data = path.read_bytes()
        name = path.relative_to(root).as_posix()
        findings.extend(scan_text(data, name))
        try:
            data.decode('utf-8')
        except UnicodeError:
            binary += 1
    result = dict(files_scanned=files, findings=findings, binary_files_with_byte_checks=binary,
                  scope='entire publication directory; structured checks for readable text/JSON')
    if strict and findings:
        raise ValueError('Publication privacy audit failed: ' + json.dumps(result))
    return result


def opaque(path, data):
    return Path(path).suffix in OPAQUE_SUFFIXES or b'\0' in data[:8192]


def audit(history=False):
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    findings, skipped, absolute, sizes = [], [], [], []
    for name in filter(None, files):
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        sizes.append(dict(path=name, bytes=len(data)))
        if opaque(name, data):
            skipped.append(name)
            continue
        findings.extend(scan_text(data, name))
        if name.endswith('.md') and ABSOLUTE_DOCUMENT_PATH.search(data):
            absolute.append(name)
    original_trace = 'data/raw/agentx/cc-traces-weka-062126/traces.jsonl'
    ignored = subprocess.run(['git', 'check-ignore', '-q', original_trace], cwd=ROOT).returncode == 0
    result = dict(base_commit=current_commit(), tracked_files=len(sizes),
                  tracked_bytes=sum(p['bytes'] for p in sizes),
                  findings=findings, document_absolute_paths=absolute,
                  opaque_files_not_secret_scanned=skipped,
                  largest_tracked_files=sorted(sizes, key=lambda x: -x['bytes'])[:10],
                  original_agentx_trace_tracked=original_trace in files,
                  original_agentx_trace_ignored=ignored,
                  model_weights_tracked=[f for f in files if f.endswith(('.safetensors', '.ckpt', '.pt'))],
                  large_raw_callback_files=[p for p in sizes if p['path'].endswith('.jsonl.gz') and p['bytes'] > 10 * 1024**2],
                  caveat='Pattern scan, not a proof of absence. Binary/compressed payloads require separate review.')
    if history:
        objects = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=ROOT)
        inventory = subprocess.run(['git', 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)'],
                                   input=objects, capture_output=True, check=True, cwd=ROOT).stdout
        blobs = []
        for line in inventory.decode().splitlines():
            parts = line.split(' ', 3)
            if parts[1] == 'blob':
                blobs.append(dict(sha=parts[0], bytes=int(parts[2]), path=parts[3] if len(parts) > 3 else ''))
        history_findings, omitted = [], []
        reader = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for item in blobs:
                if Path(item['path']).suffix in OPAQUE_SUFFIXES:
                    omitted.append(item['sha'])
                    continue
                reader.stdin.write((item['sha'] + '\n').encode())
                reader.stdin.flush()
                header = reader.stdout.readline().decode().split()
                data = reader.stdout.read(int(header[2]))
                assert reader.stdout.read(1) == b'\n'
                if opaque(item['path'], data):
                    omitted.append(item['sha'])
                    continue
                history_findings.extend(dict(blob=item['sha'], **f) for f in scan_text(data, item['path']))
        finally:
            reader.stdin.close()
            reader.stdout.close()
            reader.wait()
        result['history'] = dict(reachable_blobs=len(blobs), findings=history_findings,
                                 opaque_blobs_not_secret_scanned=len(omitted),
                                 blobs_over_25_mib=[x for x in blobs if x['bytes'] > 25 * 1024**2],
                                 history_was_not_rewritten=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--release-root', type=Path, help='Audit an entire lean directory and fail on any finding')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.release_root:
        if args.history:
            parser.error('--history cannot be combined with --release-root')
        result = audit_release(args.release_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(files=result['files_scanned'], findings=len(result['findings']))))
        return
    result = audit(args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(files=result['tracked_files'], findings=len(result['findings']),
                          document_absolute_paths=len(result['document_absolute_paths']),
                          large_raw_callbacks=len(result['large_raw_callback_files']),
                          history_findings=len(result.get('history', {}).get('findings', [])))))


if __name__ == '__main__':
    main()
