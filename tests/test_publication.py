import pytest

from agentkv.analysis.final_report import link_inputs, reproduce
from agentkv.analysis.publication_audit import scan_text


def test_publication_findings_redact_values_and_distinguish_loopback():
    token = ('ghp_' + 'a' * 36).encode()
    address = b'.'.join([b'10', b'8', b'0', b'1'])
    findings = scan_text(token + b' ' + address + b' http://127.0.0.1:8016', 'example')
    assert {f['kind'] for f in findings} == {'vendor_token', 'private_address'}
    assert token.decode() not in str(findings)
    assert all('value_sha256' in f for f in findings)


def test_summary_links_only_required_inputs_and_does_not_mutate_sources(tmp_path):
    source = tmp_path / 'source'
    attempt = source / 'g/p/attempt-1'
    attempt.mkdir(parents=True)
    for name in ('request-metadata.parquet', 'metrics.jsonl', 'private-payload'):
        (attempt / name).write_text(name)
    output = tmp_path / 'output'
    state = dict(runs=[dict(status='complete', selected_attempt='g/p/attempt-1')])
    link_inputs(output, source, state)
    assert sorted(p.name for p in (output / 'g/p/attempt-1').iterdir()) == ['metrics.jsonl', 'request-metadata.parquet']
    assert (attempt / 'private-payload').read_text() == 'private-payload'
    with pytest.raises(FileExistsError):
        reproduce(output)
