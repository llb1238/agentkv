"""Redact third-party machine/process identities in publication copies only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import socket

PROCESS_REDACTION = "<redacted: other user's process>"
PATH_REDACTION = "<redacted: other user's path>"
HOST_REDACTION = '<redacted: hostname>'
HOME_PREFIX = '/' + 'home' + '/'
HOME = re.compile(re.escape(HOME_PREFIX) + r'([A-Za-z0-9_.-]+)')
HOME_PATH = re.compile(re.escape(HOME_PREFIX) + r'([A-Za-z0-9_.-]+)(?:/[^\s"\x27<>]*)?')
ENV_NAMES = ('mini' + 'conda', 'ana' + 'conda')
PROCESS_MARKERS = ('sg' + 'lang', 'w' + 'zb')
ENV_PATH = re.compile(r'/(?:opt|srv|shared|scratch|data|mnt|tmp)/[^\s"\x27<>]*(?:conda|venv)[^\s"\x27<>]*', re.I)
HOST_KEYS = {'hostname', 'host_name', 'nodename', 'node_name', 'host', 'node'}
HOST_LINE = re.compile(r'(?im)^\s*(?:host\s*name|nodename)\s*:\s*([^\s"\x27<>]+)')
UNAME = re.compile(r'(?m)^Linux\s+([^\s]+)\s+[0-9][^\s]*\s')
GPU_ROW = re.compile(r'^(?P<prefix>\s*\|?\s*\d+\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+)'
                     r'(?P<pid>\d+)\s+(?P<type>[CG][A-Za-z+]*)\s+'
                     r'(?P<name>.*?)\s+(?P<memory>\d+(?:MiB|GiB|\.\.\.))(?P<end>\s*\|?\s*)$')


def project_process(name):
    # A generic engine name or a bare interpreter does not prove ownership.
    # Keep only command paths explicitly attributable to this project's tree.
    return bool(re.search(re.escape(HOME_PREFIX + 'llb/my_soft/') + r'(?:agentkv(?:-runs)?|vllm)/', str(name)))


def process_text(text, *, csv_context=False):
    output, changed = [], 0
    for line in text.splitlines(keepends=True):
        ending = '\n' if line.endswith('\n') else ''
        body = line.rstrip('\n')
        row = GPU_ROW.match(body)
        if row and not project_process(row['name']):
            output.append(row['prefix'] + PROCESS_REDACTION + ' ' + row['type'] + ' ' + row['memory'] + row['end'] + ending)
            changed += 1
            continue
        if csv_context:
            fields = next(csv.reader([body]))
            identity = (1, 2) if len(fields) == 4 else (0, 1) if len(fields) == 3 else None
            if identity and fields[identity[0]].strip().isdigit() and not project_process(fields[identity[1]]):
                fields[identity[0]] = fields[identity[1]] = PROCESS_REDACTION
                output.append(', '.join(v.strip() for v in fields) + ending)
                changed += 1
                continue
        output.append(line)
    return ''.join(output), changed


def _safe_host(value):
    return value in ('localhost', '127.0.0.1', '::1', '', HOST_REDACTION)


def transform_json(data):
    """Return a transformed value plus field-level reasons, without source writes."""
    changes = []

    def walk(value, pointer='', key=''):
        if isinstance(value, dict):
            # Structured process records, as well as CSV and nvidia-smi tables.
            if key == 'processes' and any(k in value for k in ('pid', 'process_id')):
                name = value.get('process_name', value.get('name', value.get('command', '')))
                if not project_process(name) and any(value.get(k) not in (None, PROCESS_REDACTION) for k in ('pid', 'process_id', 'process_name', 'name', 'command', 'cmdline', 'user', 'username')):
                    value = dict(value)
                    for field in ('pid', 'process_id', 'process_name', 'name', 'command', 'cmdline', 'user', 'username'):
                        if field in value:
                            value[field] = PROCESS_REDACTION
                    changes.append(dict(pointer=pointer, reason='third_party_process_identity'))
            return {k: walk(v, pointer + '/' + k.replace('~', '~0').replace('/', '~1'),
                            'processes' if key == 'processes' else k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, pointer + '/' + str(i), key) for i, v in enumerate(value)]
        if not isinstance(value, str):
            return value
        original = value
        value, count = process_text(value, csv_context=key == 'processes')
        if count:
            changes.append(dict(pointer=pointer, reason='third_party_process_identity', entries=count))
        value = HOME_PATH.sub(lambda m: m[0] if m[1] == 'llb' else PATH_REDACTION, value)
        value = ENV_PATH.sub(PATH_REDACTION, value)
        if key.lower() in HOST_KEYS and not _safe_host(value):
            value = HOST_REDACTION
        value = HOST_LINE.sub(lambda m: m[0].replace(m[1], HOST_REDACTION), value)
        value = UNAME.sub(lambda m: m[0].replace(m[1], HOST_REDACTION), value)
        # Capture an explicit actual local hostname too, without publishing it.
        hostname = socket.gethostname()
        if hostname and hostname not in ('localhost',) and hostname in value:
            value = value.replace(hostname, HOST_REDACTION)
        if value != original and not count:
            changes.append(dict(pointer=pointer, reason='third_party_path_or_hostname'))
        return value

    return walk(data), changes


def redact_tree(root):
    """Scan every readable publication file, independent of experiment names."""
    root = Path(root)
    changed = []
    scanned = 0
    json_scanned = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        scanned += 1
        original = path.read_bytes()
        try:
            text = original.decode('utf-8')
        except UnicodeError:
            continue
        is_json = path.suffix == '.json'
        data = json.loads(text) if is_json else text
        json_scanned += is_json
        transformed, changes = transform_json(data)
        if not changes:
            continue
        published = ((json.dumps(transformed, indent=2, ensure_ascii=False) + '\n') if is_json else transformed).encode()
        path.write_bytes(published)
        changed.append(dict(path=path.relative_to(root).as_posix(),
                            original_sha256=hashlib.sha256(original).hexdigest(),
                            published_sha256=hashlib.sha256(published).hexdigest(),
                            reason='Remove third-party process identities, environment paths and hostnames; retain GPU and memory values',
                            fields=changes))
    return dict(files_scanned=scanned, json_files_scanned=json_scanned, changed_files=changed)


def privacy_findings(data, path):
    """Findings contain digests, never the sensitive matched values."""
    text = data.decode('utf-8', errors='replace')
    findings = []

    def add(kind, value, pointer=None):
        findings.append(dict(path=path, kind=kind, value_sha256=hashlib.sha256(value.encode()).hexdigest(),
                             **({'pointer': pointer} if pointer is not None else {})))

    for match in HOME.finditer(text):
        if match[1] != 'llb':
            add('foreign_home', match[0])
    try:
        readable = data.decode('utf-8')
    except UnicodeError:
        return findings  # Every file gets byte/home checks; structured checks require text.
    for marker in ENV_NAMES:
        if marker in readable.casefold():
            add('foreign_environment', marker)
    for match in ENV_PATH.finditer(readable):
        add('foreign_environment', match[0])
    for marker in PROCESS_MARKERS:
        if marker in readable.casefold():
            add('foreign_process_marker', marker)
    for pattern in (HOST_LINE, UNAME):
        for match in pattern.finditer(readable):
            if not _safe_host(match[1]):
                add('hostname', match[1])
    hostname = socket.gethostname()
    if hostname and hostname != 'localhost' and hostname in readable:
        add('hostname', hostname)

    def walk(value, pointer='', key=''):
        if isinstance(value, dict):
            if key == 'processes' and any(k in value for k in ('pid', 'process_id')):
                name = value.get('process_name', value.get('name', value.get('command', '')))
                pid = value.get('pid', value.get('process_id'))
                if pid != PROCESS_REDACTION and not project_process(name):
                    add('foreign_gpu_process', str(value), pointer)
            for k, v in value.items():
                walk(v, pointer + '/' + k, 'processes' if key == 'processes' else k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, pointer + '/' + str(i), key)
        elif isinstance(value, str):
            if key.lower() in HOST_KEYS and not _safe_host(value):
                add('hostname', value, pointer)
            _, count = process_text(value, csv_context=key == 'processes')
            if count:
                add('foreign_gpu_process', value, pointer)
    if Path(path).suffix == '.json':
        try:
            walk(json.loads(readable))
        except ValueError:
            add('invalid_json', readable)
    else:
        _, count = process_text(readable)
        if count:
            add('foreign_gpu_process', readable)
    return findings
