from agentkv.analysis.classify_probe_keys import classify, fingerprints, final_sharing_counterexample
from agentkv.hints import make_envelope
import json
import pyarrow as pa
import pyarrow.parquet as pq


def test_gpu_hit_prefix_is_not_a_failed_store_or_chunking_gap():
    assert classify(5, [48, 64, 80], 32) == ('gpu_hit_prefix_skipped_by_connector', {16, 32})
    assert classify(5, [], 80)[0] == 'gpu_hit_prefix_skipped_by_connector'
    assert classify(5, [16, 48, 64, 80], 32)[0] == 'unresolved'


def test_sub_block_empty_is_not_an_incomplete_full_block_request():
    assert classify(0, [], 0) == ('prompt_shorter_than_block', set())
    assert classify(1, [], 0)[0] == 'unresolved'


def test_prefix_fingerprints_compare_values_not_only_depth():
    a = list(range(48)); b = a[:32] + [99] * 16
    assert fingerprints(a)[:2] == fingerprints(b)[:2]
    assert fingerprints(a)[2] != fingerprints(b)[2]
    assert len(fingerprints(a + [999])) == 3


def test_omitted_shared_prefix_changes_final_priority_eviction():
    final = make_envelope(dict(chain_id='parent', chain_turn=1, final=True, hint_perfect_ms=None), 'perfect', 'a', 'r#0')
    waiting = make_envelope(dict(chain_id='child', chain_turn=0, final=False, hint_perfect_ms=1000.), 'perfect', 'b', 'r#0')
    result = final_sharing_counterexample(final, waiting, b'real-shared-key')
    assert result['recorded_omission']['waiting_chain_mode'] == 'waiting'
    assert not result['recorded_omission']['shared_registered_to_waiting_chain']
    assert result['ownership_complete_control']['shared_registered_to_waiting_chain']
    assert result['recorded_omission']['victim'] != result['ownership_complete_control']['victim']


def test_diagnosis_distinguishes_observation_from_residency(tmp_path,monkeypatch):
    from agentkv.analysis import classify_probe_keys as module
    monkeypatch.setattr(module,'git_commit',lambda:'test-commit')
    probe=tmp_path/'probe';probe.mkdir()
    (probe/'policy-observations').mkdir();(probe/'server-observations').mkdir()
    original=[dict(trace_req_idx='t',prompt_len=32,prompt_token_ids=list(range(32)))]
    workload=tmp_path/'workload.parquet';pq.write_table(pa.Table.from_pylist(original),workload)
    (probe/'run.json').write_text(json.dumps(dict(git_commit='original-commit',source=dict(workload_sha256=module.sha256_file(workload)))))
    rows=[dict(request_id=msg,root_id='r',instance_id=0,chain_id=msg,chain_turn=0,final=final,
        hint_perfect_ms=None if final else 1000.,send_ns=ns,trace_req_idx='t',prompt_len=32,permutation_seed=1)
        for msg,final,ns in [('parent',True,10),('child',False,30)]]
    pq.write_table(pa.Table.from_pylist(rows),probe/'request-metadata.parquet')
    records=[];prefills=[]
    for i,row in enumerate(rows):
        parent=i==0
        records.append(dict(event='on_request_finished',ns=20+i*20,request_id=row['request_id'],
            hint=dict(message_id=row['request_id'],session_id='r#0/'+row['chain_id'],final=row['final']),
            key_groups=[['aa','bb']] if parent else [],position_groups=[[16,32]] if parent else [],
            ref_count_groups=[[0,0]] if parent else [],insertion_only_count=2 if parent else 0,reused_count=0))
        prefills.append(dict(event='prefill',request_id=row['request_id'],gpu_hit=0 if parent else 32,
                             cpu_hit=0,recomputed=32 if parent else 0))
    (probe/'policy-observations/events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in records))
    (probe/'server-observations/requests.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in prefills))
    output=tmp_path/'diagnosis';result=module.diagnose(probe,workload,output)
    assert result['incomplete_requests']==1 and result['empty_incomplete_requests']==1
    assert result['never_observed_in_same_chain_blocks']==2
    assert result['final_sharing_witnesses'][0]['all_observed_previous_chains_final']
    row=json.loads((output/'per-request.json').read_text())[0]
    assert row['cpu_residency_of_omitted_keys']=='not_observable_from_original_scalar_log'
