import json
import struct
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentkv.bench import callback_probe as module


def write(path,data):path.write_text(json.dumps(data)+'\n')


def fixture(directory,*,missing=False,reversed_order=False):
    (directory/'policy-observations').mkdir();(directory/'server-observations').mkdir()
    rows=[dict(request_id=msg,server_request_id='server-'+msg,send_ns=100, end_ns=end,prompt_len=length,status='success',prompt_token_ids=[1,2]) for msg,end,length in [('a',1000,32),('b',2000,16)]]
    pq.write_table(pa.Table.from_pylist(rows),directory/'requests.parquet')
    write(directory/'run.json',dict(git_commit='test-measurement',summary=dict(hint_sent=2,hint_received=2)))
    write(directory/'server-manifest.json',dict(vllm_commit='test-backend'))
    events=[]
    for msg,start,end in [('a',300,900),('b',600,1900)]:
        for name,ns in [('request_scheduled',start),('request_finished',end)]:events.append(dict(event=name,ns=ns,request_id='server-'+msg))
    (directory/'server-observations/requests-test.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    def hint(msg):return dict(direct_present=False,nested_present=True,direct_valid=False,nested_valid=True,selected_source='nested',valid=True,message_id=msg)
    records=[dict(event='on_store_miss',ns=ns,request_id='server-a',hint=hint('a'),count=1,positions=[pos]) for ns,pos in [(500,16),(800,32)]]
    records.append(dict(event='on_request_finished',ns=900,request_id='server-a',hint=hint('a'),
        key_groups=[['k1','k2']],position_groups=[[32,16] if reversed_order else [16,32]],ref_count_groups=[[-1,0]]))
    if not missing:records.append(dict(event='on_request_finished',ns=1900,request_id='server-b',hint=hint('b'),key_groups=[['k']],position_groups=[[16]],ref_count_groups=[[0]]))
    counts={name:sum(e['event']==name for e in records) for name in ('on_store_miss','on_request_finished')}
    data=b''.join(struct.pack('<QB',e['ns'],module.METHODS.index(e['event'])) for e in records)
    (directory/'policy-observations/test.bin').write_bytes(data)
    records.append(dict(event='probe_close',ns=2000,counts=counts))
    (directory/'policy-observations/test.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in records))


@pytest.mark.parametrize('missing,reversed_order',[(False,False),(True,True)])
def test_probe_analysis_preserves_callback_semantics_and_counts(tmp_path,monkeypatch,missing,reversed_order):
    monkeypatch.setattr(module,'git_commit',lambda:'test-only')
    fixture(tmp_path,missing=missing,reversed_order=reversed_order);result=module.analyze(tmp_path)
    assert result['client_sent_with_hint']==2
    assert result['plugin_unique_valid_any']==(1 if missing else 2)
    assert result['sent_equals_plugin_any']==(not missing)
    assert result['requests_with_multiple_store_miss']==1
    assert result['requests_without_store_miss']==1
    assert result['requests_store_miss_keys_not_full_prompt']==1
    assert result['key_order']['nonmonotone_requests']==int(reversed_order)
    assert result['finished_chunk_states']['not_ready']==1
    assert result['stop_before_hint_adapter']
    assert result['timings_ms']['first_store_miss_minus_server_end_ms']['p50']==-.0004
    assert 'prompt_token_ids' not in pq.read_table(tmp_path/'request-metadata.parquet').column_names


@pytest.mark.parametrize('missing',[False,True])
def test_independent_audit_distinguishes_coverage_and_integrity(tmp_path,monkeypatch,missing):
    from agentkv.analysis import callback_probe_report as audit
    monkeypatch.setattr(module,'git_commit',lambda:'test-only')
    monkeypatch.setattr(audit,'git_commit',lambda:'test-only')
    fixture(tmp_path,missing=missing)
    module.analyze(tmp_path)
    # Reports can be regenerated from compact metadata, without prompt arrays.
    (tmp_path/'requests.parquet').unlink()
    result=audit.audit(tmp_path)
    assert result['critical_context_verification_passed']==(not missing)
    assert result['raw_context_calls_match_binary']
    assert result['scalar_shutdown_finalization_complete']
    assert result['client_requests']==2 and result['valid_hint_requests']==(1 if missing else 2)
