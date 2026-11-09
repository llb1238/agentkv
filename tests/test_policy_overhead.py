import json
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.replay import OUTPUT_SCHEMA
from agentkv.bench.policy_overhead import finalize
from agentkv.policy.light_metrics import LightMetrics


def test_finalize_preserves_uint64_and_regenerates_from_compact_metadata(tmp_path):
    write_json(tmp_path/'run.json',dict(git_commit='code',vllm_commit='backend',instances=[],source={},
        summary=dict(hint_sent=0,ttft_ms=dict(p50=1,p99=2))))
    write_json(tmp_path/'server-manifest.json',dict(vllm_commit='backend',policy='MeasuredLRUCachePolicy'))
    row=dict(request_id='client',server_request_id='server',prompt_len=32,is_entry=False,
        in_window=True,status='success',send_ns=10,permutation_seed=2**64-1,prompt_token_ids=list(range(32)))
    pq.write_table(pa.Table.from_pylist([row],schema=OUTPUT_SCHEMA),tmp_path/'requests.parquet')
    store=LightMetrics(tmp_path/'server-observations',pid=123)
    store.set_window(0,0,2)
    for name in ('scheduler.schedule','policy.evict'):store.observe(name,100,1000)
    store.record('request_received',ns=100,request_id='server')
    store.record('request_scheduled',ns=200,request_id='server')
    store.record('prefill',ns=300,request_id='server',prompt_tokens=32,gpu_hit=16,cpu_hit=0,recomputed=16)
    store.close()
    result=finalize(tmp_path)
    table=pq.read_table(tmp_path/'request-metadata.parquet')
    assert table['permutation_seed'].to_pylist()==[2**64-1]
    assert table['gpu_hit'].to_pylist()==[16]
    assert result['source_fractions']['gpu_hit']==.5
    original=(tmp_path/'run.json').read_bytes()
    (tmp_path/'requests.parquet').unlink()
    assert finalize(tmp_path)['source_fractions']==result['source_fractions']
    assert (tmp_path/'run.json').read_bytes()==original
