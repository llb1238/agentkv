"""Independent report audit rejects a tail violation and a non-conserving label."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentkv.analysis.finalize_e0 import independent_stability,check_partition


def test_independent_check_rejects_queue_p99_despite_flat_ttft():
    samples=[dict(elapsed_s=600+10*i,values={'vllm:num_requests_waiting':v,'vllm:num_requests_running':1})
             for i,v in enumerate([0]*177+[9]*3)]
    rows=[dict(status='success',is_entry=False,send_ns=int(t*1e9),ttft_ms=50.) for t in (700,1900)]
    checks,numbers=independent_stability(rows,samples,[dict(event='kv_observer_ready')],0)
    assert not checks['queue_distribution'] and checks['ttft_half_ratio']
    assert numbers['queue_zero_fraction']>.8 and numbers['queue_p99']==9


def test_partition_audit_rejects_relabeling_even_when_totals_match(tmp_path):
    row=dict(unavoidable=16,gap_eviction=0,other_eviction=32,hit=16,history_prefix=48,
             trigger_prefix=32,prompt_blocks_tokens=64,hit_exceeds_history=False,queued_s=.1)
    pq.write_table(pa.Table.from_pylist([row]),tmp_path/'recomputation.parquet')
    summary=dict(source_tokens={k:row[k] for k in ('unavoidable','gap_eviction','other_eviction')},
                 prompt_blocks_tokens=64,requests=1)
    with pytest.raises(AssertionError):check_partition(tmp_path,summary)
