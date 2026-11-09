"""Real local streaming HTTP tests for the online replay contract."""

import asyncio
import gc
import json
import subprocess
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from aiohttp import web

from agentkv.bench.estimate_replay import write_schedule_estimate
from agentkv.bench.fake_server import FakeServer
from agentkv.hints import make_envelope
from agentkv.provenance import git_commit
from agentkv.bench.replay import ReplayConfig, acquire_slot, run_replay
from agentkv.bench.steady_state import InstanceTokenPermutation, SessionInstance, simulate_entries, RootSchedule
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.workload import SCHEMA


@pytest.fixture
def workload(tmp_path):
    rows = []
    for chain, turn, idx, anchor, delay, gap in (
        ("parent",0,"p0",None,0,.05), ("parent",1,"p1",None,None,None),
        ("child-a",0,"a0","p0",.02,None), ("child-b",0,"b0","p0",.02,None),
        ("arrival",0,"r0",None,.1,None),
    ):
        prompt = [101,102,103,104]
        final = gap is None
        rows.append(dict(root_id="r",chain_id=chain,chain_turn=turn,trace_req_idx=idx,
            is_subagent=chain.startswith("child"),is_aux=chain!="parent",chain_offset_s=99999,
            chain_offset_orig_s=0,anchor_trace_req_idx=anchor,start_delay_s=delay,
            prompt_token_ids=prompt,prompt_len=len(prompt),max_tokens=3,out_zero_orig=False,
            lcp_prev_tokens=4 if turn else 0,gap_after_s=gap,gap_truncated=False,
            hint_perfect_ms=None if final else gap*1000,
            hint_noisy_ms=None if final else gap*1200,
            hint_online_ms=None if final else gap*800,
            hint_online_p90_ms=None if final else gap*1600,final=final,
            input_len_orig=4,output_len_orig=3,t_orig_s=0,api_time_s=.5,gap_after_orig_s=gap))
    path = tmp_path / "input/workload.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),path,row_group_size=2)
    digest = sha256_file(path)
    path.with_name("manifest.json").write_text(json.dumps({"workload":{"sha256":digest}}))
    schedule = path.with_name("schedule_estimate.parquet")
    write_schedule_estimate(rows,schedule,digest)
    return path,schedule,rows


async def exercise(tmp_path, workload, tier="none", tau=0, service=.5, limit=128, instances=None, duration=1.3, fail_ids=(), request_timeout=5):
    # Discard cyclic garbage from earlier tests before starting the timed HTTP
    # fixture. A measured generation-2 collection took 64 ms inside this test
    # only in the full suite; standalone replay tests have no such backlog.
    gc.collect()
    path,schedule,requests = workload
    observations = tmp_path / "observations"
    fake = FakeServer(service, observations, fail_ids)
    runner = web.AppRunner(fake.app)
    await runner.setup()
    site = web.TCPSite(runner,"127.0.0.1",0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = ReplayConfig(output_dir=tmp_path/"run", base_url=f"http://127.0.0.1:{port}",
        duration=duration,warmup=0,cooldown=0,max_inflight=limit,hint_tier=tier,
        request_timeout=request_timeout,drain_timeout=3,metrics_interval=.1,allow_dirty=True,
        observation_dir=observations)
    instances = instances or [SessionInstance("r",0,tau,0,tau>0)]
    vocab = np.arange(100,151755,dtype=np.int32)
    try:
        # Fake HTTP tests do not authorize measurements or require a Git checkout.
        # Dirty/clean source enforcement is tested separately in real temp repos.
        with patch('agentkv.bench.replay.git_commit', return_value='f'*40):
            info,records = await run_replay(config,path,schedule,vocab,instances,server_metadata={
                "type":"fake", "service_time_s":service, "git_commit":'f'*40})
    finally:
        await runner.cleanup()
    return info,records,fake,config


@pytest.mark.parametrize("tau",[0,.25])
def test_http_event_and_timer_sends_match_offline_estimates(tmp_path,workload,tau):
    info,records,fake,config = asyncio.run(exercise(tmp_path,workload,tau=tau))
    root = RootSchedule(workload[2],pq.read_table(workload[1]).to_pylist())
    expected = simulate_entries(root.entries(tau))
    errors = [abs((r["send_ns"]-info["origin_ns"])/1e9-expected[r["trace_req_idx"]])*1000 for r in records]
    assert {r["trace_req_idx"] for r in records} == set(expected)
    assert all(r["status"]=="success" for r in records)
    assert np.quantile(errors,.99)<10,errors
    assert max(r["schedule_error_ms"] for r in records)<8
    assert info["summary"]["hint_counts_match"]
    assert isinstance(info["git_commit"],str) and len(info["git_commit"])==40
    assert "git" not in info and "git_at_end" not in info
    assert "implementation_sha256" not in info["source"]
    assert not list(config.output_dir.rglob("*.py"))
    assert all("kv_transfer_params" not in r for r in fake.requests)
    assert all(r["is_entry"] == (tau>0 and r["trace_req_idx"] in {"p1","a0","b0"}) for r in records)
    assert pq.read_table(config.output_dir/"requests.parquet").num_rows==len(records)


@pytest.mark.parametrize("tier",["perfect","noisy","online"])
def test_streamed_requests_have_correct_hint_envelope_and_permutation(tmp_path,workload,tier):
    info,records,fake,_ = asyncio.run(exercise(tmp_path,workload,tier=tier,service=.02,duration=.4,
        instances=[SessionInstance("r",0,0,0,False),SessionInstance("r",2,0,.15,False)]))
    assert len(records)==10 and all(r["status"]=="success" for r in records)
    assert info["summary"]["hint_sent"]==info["summary"]["hint_received"]==10
    vocab=np.arange(100,151755,dtype=np.int32)
    for body in fake.requests:
        instance=int(body["request_id"].split("r#")[1].split("/")[0])
        expected=InstanceTokenPermutation(vocab,"r",instance).transform([101,102,103,104]).tolist()
        assert body["prompt"]==expected and body["prompt"]!=[101,102,103,104]
        envelope=body["kv_transfer_params"]["kv_hints"]
        assert envelope["message_id"]==body["request_id"]
        payload=envelope["actions"][0]["payload"]
        request=next(r for r in workload[2] if f"/{r['chain_id']}/{r['chain_turn']}" in body["request_id"])
        assert payload["expected_gap_ms"]==request[f"hint_{tier}_ms"]
        assert payload["final"]==request["final"]
    parent=[r for r in fake.requests if "/parent/0" in r["request_id"]]
    assert parent[0]["prompt"][0]!=parent[1]["prompt"][0]


def test_inflight_limit_queues_requests_and_records_wait_time(tmp_path,workload):
    instances=[SessionInstance("r",i,0,0,False) for i in range(4)]
    info,records,fake,_=asyncio.run(exercise(tmp_path,workload,service=.04,limit=1,duration=.45,instances=instances))
    assert fake.peak==info["summary"]["peak_inflight"]==1
    assert info["summary"]["safety_valve_triggered"]
    assert any(r["was_queued"] and r["queued_ms"]>20 for r in records)
    assert all(r["send_ns"]<info["cutoff_ns"] for r in records)


def test_dirty_worktree_is_rejected_unless_explicitly_allowed(tmp_path):
    subprocess.run(["git","init","-q",str(tmp_path)],check=True)
    subprocess.run(["git","-C",str(tmp_path),"-c","user.name=Test","-c","user.email=test@example.invalid",
                    "commit","--allow-empty","-qm","test"],check=True)
    assert len(git_commit(repository=tmp_path))==40
    (tmp_path/"untracked").write_text("dirty")
    with pytest.raises(ValueError,match="allow-dirty"):
        git_commit(repository=tmp_path)
    assert len(git_commit(True,repository=tmp_path))==40


def test_none_tier_returns_no_envelope_even_for_final_request(workload):
    assert all(make_envelope(r,"none","id","instance") is None for r in workload[2])


def test_failed_trigger_blocks_dependents_and_preserves_independent_timers(tmp_path,workload):
    info,records,_,_=asyncio.run(exercise(tmp_path,workload,service=.02,duration=.3,
        fail_ids={"m1-replay/r#0/parent/0"}))
    assert {r["trace_req_idx"] for r in records}=={"p0","r0"}
    assert info["summary"]["dependency_blocked"]==3
    assert info["summary"]["window_failed"]==1
    assert next(r for r in records if r["trace_req_idx"]=="r0")["status"]=="success"


def test_cutoff_stops_new_sends_but_drains_inflight_request(tmp_path,workload):
    info,records,_,_=asyncio.run(exercise(tmp_path,workload,service=.2,duration=.05))
    assert len(records)==1 and records[0]["trace_req_idx"]=="p0"
    assert records[0]["status"]=="success"
    assert records[0]["send_ns"]<info["cutoff_ns"]<records[0]["end_ns"]


def test_simultaneous_slot_contenders_are_all_counted_as_queued():
    async def run():
        gate=asyncio.Semaphore(1)
        async def contender():
            queued,wait=await acquire_slot(gate,1)
            await asyncio.sleep(.01)
            gate.release()
            return queued,wait
        return await asyncio.gather(*(contender() for _ in range(4)))
    results=asyncio.run(run())
    assert results[0]==(False,0)
    assert all(queued and wait>5 for queued,wait in results[1:])


def test_stream_timeout_preserves_already_received_token_timestamps(tmp_path,workload):
    info,records,_,_=asyncio.run(exercise(tmp_path,workload,service=.3,duration=.35,request_timeout=.18))
    assert len(records)==2
    assert all(r['status']=='failed' for r in records)
    assert all(r['completion_tokens']>=1 and r['first_token_ns'] is not None and r['ttft_ms']>0 for r in records)
    assert info['summary']['window_failed']==2


def test_payload_build_starts_at_event_not_near_send_deadline(tmp_path,workload,monkeypatch):
    from agentkv.bench.replay import ReplayEngine
    import time
    path,schedule,rows=workload
    for r in rows:
        # Three artificially slow builds share two workers; leave enough
        # delay for two batches while still detecting the old 50ms lead.
        if r['trace_req_idx']=='p0': r['gap_after_s']=.5
        if r['trace_req_idx'] in ('a0','b0'): r['start_delay_s']=.5
    pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),path)
    path.with_name('manifest.json').write_text(json.dumps({'workload':{'sha256':sha256_file(path)}}))
    write_schedule_estimate(rows,schedule,sha256_file(path))
    original=ReplayEngine.payload
    def slow(self,node):
        time.sleep(.12)
        return original(self,node)
    monkeypatch.setattr(ReplayEngine,'payload',slow)
    info,records,_,config=asyncio.run(exercise(tmp_path,workload,service=.02,duration=.8))
    by_id={r['trace_req_idx']:r for r in records}
    parent=by_id['p0']
    for name in ('p1','a0','b0'):
        r=by_id[name]
        assert r['trigger_ns']==parent['end_ns']
        assert r['body_ready_ns']<r['due_ns']-100_000_000
        assert abs(r['schedule_error_ms'])<10
        assert r['body_build_ms']>=0
    assert (config.output_dir/'loop_lag.parquet').exists()
    assert info['summary']['event_loop_lag_ms']['n']>30
    assert info['token_store']['bytes']>0
