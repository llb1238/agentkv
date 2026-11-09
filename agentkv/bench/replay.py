"""Event-anchored, steady-state OpenAI completion replay using streaming HTTP."""

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prometheus_client.parser import text_string_to_metric_families

from agentkv.hints import make_envelope
from agentkv.provenance import git_commit
from .steady_state import InstanceTokenPermutation, RootSchedule, sample_population
from .synth import tokenizer_vocabulary
from .trace_loader import sha256_file
from .workload import ROOT, SCHEMA

EXTRA_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("policy", pa.string()), ("hint_tier", pa.string()),
    ("workload_sha256", pa.string()), ("request_id", pa.string()),
    ("server_request_id", pa.string()), ("instance_id", pa.int64()),
    ("tau_s", pa.float64()), ("instance_start_s", pa.float64()),
    ("is_entry", pa.bool_()), ("timed_trigger", pa.bool_()),
    ("trigger_trace_req_idx", pa.string()), ("due_ns", pa.int64()),
    ("send_ns", pa.int64()), ("first_token_ns", pa.int64()),
    ("last_token_ns", pa.int64()), ("end_ns", pa.int64()),
    ("ttft_ms", pa.float64()), ("tpot_ms", pa.float64()), ("e2e_ms", pa.float64()),
    ("gap_actual_s", pa.float64()), ("gap_error_ms", pa.float64()),
    ("schedule_error_ms", pa.float64()), ("in_window", pa.bool_()),
    ("queued_ms", pa.float64()), ("was_queued", pa.bool_()),
    ("server_queued_ms", pa.float64()), ("server_scheduled_ns", pa.int64()),
    ("completion_tokens", pa.int32()), ("http_status", pa.int32()),
    ("sent_prompt_sha256", pa.string()), ("permutation_seed", pa.uint64()),
    ("trigger_ns", pa.int64()), ("body_ready_ns", pa.int64()),
    ("body_build_ms", pa.float64()), ("body_prepare_wait_ms", pa.float64()),
    ("gpu_hit", pa.int64()), ("cpu_hit", pa.int64()), ("recomputed", pa.int64()),
    ("gpu_hit_fraction", pa.float64()), ("cpu_hit_fraction", pa.float64()),
    ("recomputed_fraction", pa.float64()),
    ("gpu_hit_effective", pa.int64()), ("cpu_hit_effective", pa.int64()),
    ("recomputed_effective", pa.int64()), ("replay_overlap", pa.int64()),
    ("prefill_forward_token_work", pa.int64()), ("repeated_prompt_work", pa.int64()),
    ("effective_unattributed_cache_tokens", pa.int64()),
    ("effective_ranges_complete", pa.bool_()), ("prefill_observed_preemptions", pa.int64()),
    ("status", pa.string()), ("error", pa.string()),
])
OUTPUT_SCHEMA = pa.schema(list(SCHEMA) + list(EXTRA_SCHEMA), metadata={
    b"agentkv.format": b"m1-online-v1",
    b"agentkv.prompt_token_ids": b"original workload tokens; transmitted permutation identified by hash and seed",
})


@dataclass(frozen=True)
class ReplayConfig:
    output_dir: Path
    base_url: str = "http://127.0.0.1:8012"
    model: str = "Qwen/Qwen3-8B"
    run_id: str = "m1-replay"
    policy: str = "lru"
    hint_tier: str = "none"
    arrival_rate: float = 1.0
    population_seed: int = 0
    steady_state: bool = True
    duration: float = 2400.0
    warmup: float = 600.0
    cooldown: float = 0.0
    max_inflight: int = 128
    request_timeout: float = 180.0
    drain_timeout: float = 120.0
    metrics_interval: float = 1.0
    allow_dirty: bool = False
    observation_dir: Path | None = None

    def validate(self):
        if self.hint_tier not in ("none", "perfect", "noisy", "online"):
            raise ValueError("Invalid hint tier")
        if self.max_inflight < 1 or self.population_seed < 0:
            raise ValueError("Invalid concurrency limit or seed")
        for value in (self.duration, self.request_timeout, self.drain_timeout, self.metrics_interval):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Durations and intervals must be finite and positive")
        if (not math.isfinite(self.warmup) or not math.isfinite(self.cooldown)
                or self.warmup < 0 or self.cooldown < 0 or self.warmup + self.cooldown >= self.duration):
            raise ValueError("Invalid measurement window")
        if not math.isfinite(self.arrival_rate) or self.arrival_rate <= 0:
            raise ValueError("Arrival rate must be finite and positive")


class TokenStore:
    """Decode once before the experiment clock; keep compact Arrow token buffers."""
    def __init__(self, path, requests, cache_groups=4):
        self.reader = pq.ParquetFile(path)
        self.column = self.reader.read(columns=["prompt_token_ids"], use_threads=False).column(0).combine_chunks()
        self.positions = {r["trace_req_idx"]:i for i,r in enumerate(requests)}

    def get(self, idx):
        return self.column[self.positions[idx]].values.to_numpy(zero_copy_only=False)


def parse_metrics(text, model):
    values = defaultdict(float)
    names = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
             "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in names and sample.labels.get("model_name", model) == model:
                values[sample.name] += sample.value
    return dict(values)


def percentiles(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), p50=float(np.median(values)) if values else None,
                p99=float(np.quantile(values, .99)) if values else None,
                maximum=max(values) if values else None)


async def acquire_slot(gate, timeout):
    """Take a free slot without yielding before it is reserved."""
    queued = gate.locked()
    start = time.monotonic_ns()
    if queued:
        await asyncio.wait_for(gate.acquire(), timeout=timeout)
    else:
        # Semaphore.acquire does not yield while a slot is immediately free.
        await gate.acquire()
    return queued, (time.monotonic_ns() - start) / 1e6 if queued else 0.0


def join_observations(records, directory):
    events = []
    if directory and Path(directory).exists():
        from agentkv.policy.light_metrics import read_light_events
        events.extend(read_light_events(directory))
        for path in sorted(Path(directory).glob("requests-*.jsonl")):
            for line in path.read_text().splitlines():
                events.append(json.loads(line))
    prefills = {e["request_id"]: e for e in events if e["event"] == "prefill"}
    effective = {e['request_id']:e for e in events if e['event']=='effective_prefill'}
    received={}
    scheduled={}
    for event in sorted(events,key=lambda e:e['ns']):
        if event['event']=='request_received':received.setdefault(event['request_id'],event['ns'])
        if event['event']=='request_scheduled':scheduled.setdefault(event['request_id'],event['ns'])
    joined = 0
    for record in records:
        server_id = record.get("server_request_id")
        if not server_id:
            continue
        matches = [v for k, v in prefills.items() if k == server_id or k.startswith(server_id + "-0-") or k == server_id + "-0"]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous server observation: {server_id}")
        if matches:
            values = matches[0]
            identity=values['request_id']
            if identity in received and identity in scheduled:
                record['server_queued_ms']=(scheduled[identity]-received[identity])/1e6
                record['server_scheduled_ns']=scheduled[identity]
            total = sum(values[k] for k in ("gpu_hit", "cpu_hit", "recomputed"))
            if total != record["prompt_len"] or total != values["prompt_tokens"]:
                raise ValueError(f"Non-additive nominal cache sources: {server_id}")
            for name in ("gpu_hit", "cpu_hit", "recomputed"):
                record[name] = values[name]
                record[name + "_fraction"] = values[name] / total
            if identity in effective:
                for name in ('gpu_hit_effective','cpu_hit_effective','recomputed_effective',
                             'replay_overlap','prefill_forward_token_work','repeated_prompt_work',
                             'effective_unattributed_cache_tokens','effective_ranges_complete','prefill_observed_preemptions'):
                    record[name]=effective[identity][name]
            joined += 1
    sent_ids = {r["request_id"] for r in records if r.get("send_ns") is not None}
    hints = {e["message_id"] for e in events if e["event"] == "hint_received" and e["message_id"] in sent_ids}
    return dict(prefills_joined=joined, valid_hints_received=len(hints), observation_events=len(events))


class ReplayEngine:
    def __init__(self, config, requests, tokens, roots, vocabulary, instances):
        self.config, self.requests, self.tokens, self.roots = config, requests, tokens, roots
        self.vocabulary, self.instances = vocabulary, instances
        self.records, self.completed, self.jobs, self.handles = [], {}, set(), []
        self.gate = asyncio.Semaphore(config.max_inflight)
        self.queue_count = self.hints_sent = self.peak_inflight = self.inflight = self.blocked = 0
        self.metrics, self.stop_metrics = [], asyncio.Event()
        self.loop_lags = []
        self.nodes, self.timers = {}, []

    def prepare(self):
        for instance in self.instances:
            permutation = InstanceTokenPermutation(self.vocabulary, instance.root_id, instance.instance_id)
            entries = self.roots[instance.root_id].entries(instance.tau_s)
            local = {}
            for entry in entries:
                key = (instance.instance_id, entry.request["trace_req_idx"])
                local[entry.request["trace_req_idx"]] = key
                self.nodes[key] = dict(entry=entry, instance=instance, permutation=permutation, children=[])
            for entry in entries:
                key = local[entry.request["trace_req_idx"]]
                if entry.timed:
                    self.timers.append((key, instance.start_s + entry.delay_s))
                else:
                    self.nodes[local[entry.trigger_trace_req_idx]]["children"].append(key)

    def payload(self, node):
        begin = time.monotonic_ns()
        entry, instance = node["entry"], node["instance"]
        request = entry.request
        name = f"{instance.root_id}#{instance.instance_id}"
        idx = f"{self.config.run_id}/{name}/{request['chain_id']}/{request['chain_turn']}"
        tokens = node["permutation"].transform(self.tokens.get(request["trace_req_idx"]))
        body = dict(model=self.config.model, prompt=tokens.tolist(), max_tokens=request["max_tokens"],
                    temperature=0, ignore_eos=True, stream=True, stream_options={"include_usage":True},
                    return_token_ids=True, request_id=idx)
        envelope = make_envelope(request, self.config.hint_tier, idx, name)
        if envelope is not None:
            body["kv_transfer_params"] = {"kv_hints": envelope}
        data = json.dumps(body, separators=(",", ":")).encode()
        digest = hashlib.sha256(tokens.astype("<i4").tobytes()).hexdigest()
        end = time.monotonic_ns()
        return idx, data, digest, begin, end

    def arm(self, key, due_ns, trigger_ns=None):
        if due_ns >= self.cutoff_ns:
            return
        def spawn():
            job = asyncio.create_task(self.send(key, due_ns, trigger_ns))
            self.jobs.add(job)
            job.add_done_callback(self.jobs.discard)
        trigger_ns = time.monotonic_ns() if trigger_ns is None else trigger_ns
        self.handles.append(asyncio.get_running_loop().call_at(max(time.monotonic(), trigger_ns / 1e9), spawn))

    async def send(self, key, due_ns, trigger_ns):
        node = self.nodes[key]
        request, instance, entry = node["entry"].request, node["instance"], node["entry"]
        row = dict(request, run_id=self.config.run_id, policy=self.config.policy,
                   hint_tier=self.config.hint_tier, workload_sha256=self.workload_sha,
                   instance_id=instance.instance_id, tau_s=instance.tau_s, instance_start_s=instance.start_s,
                   is_entry=entry.is_entry, timed_trigger=entry.timed,
                   trigger_trace_req_idx=entry.trigger_trace_req_idx, due_ns=due_ns, trigger_ns=trigger_ns,
                   permutation_seed=node["permutation"].seed, status="not_sent", error=None,
                   in_window=False, queued_ms=0.0, was_queued=False, completion_tokens=0)
        acquired = False
        ids, first, last, usage, done = 0, None, None, None, False
        try:
            idx, data, prompt_hash, begin, ready = await asyncio.get_running_loop().run_in_executor(self.payload_executor,self.payload,node)
            row.update(request_id=idx, sent_prompt_sha256=prompt_hash,
                       body_build_ms=(ready-begin)/1e6, body_ready_ns=ready,
                       body_prepare_wait_ms=(begin-trigger_ns)/1e6)
            await asyncio.sleep(max(0, (due_ns - time.monotonic_ns()) / 1e9))
            remaining = (self.cutoff_ns - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                return
            before_queue = time.monotonic_ns()
            row["was_queued"] = self.gate.locked()
            self.queue_count += row["was_queued"]
            _, row["queued_ms"] = await acquire_slot(self.gate, remaining)
            acquired = True
            if row["was_queued"]:
                row["queued_ms"] = (time.monotonic_ns() - before_queue) / 1e6
            if time.monotonic_ns() >= self.cutoff_ns:
                return
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            row["send_ns"] = time.monotonic_ns()
            elapsed = (row["send_ns"] - self.origin_ns) / 1e9
            row["in_window"] = self.config.warmup <= elapsed <= self.config.duration - self.config.cooldown and not entry.is_entry
            row["schedule_error_ms"] = (row["send_ns"] - due_ns) / 1e6
            previous = self.completed.get((instance.instance_id, request["chain_id"], request["chain_turn"] - 1))
            if previous is not None:
                previous["gap_actual_s"] = (row["send_ns"] - previous["end_ns"]) / 1e9
                previous["gap_error_ms"] = (previous["gap_actual_s"] - previous["gap_after_s"]) * 1000
            if self.config.hint_tier != "none":
                self.hints_sent += 1
            async with self.session.post(self.config.base_url.rstrip("/") + "/v1/completions", data=data,
                                         headers={"Content-Type":"application/json", "X-Request-Id":idx}) as response:
                row["http_status"] = response.status
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
                async for line in response.content:
                    if not line.startswith(b"data:"):
                        continue
                    value = line[5:].strip()
                    if value == b"[DONE]":
                        done = True
                        break
                    chunk = json.loads(value)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    row["server_request_id"] = chunk.get("id", row.get("server_request_id"))
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        count = len(choice.get("token_ids") or [])
                        if count or choice.get("text"):
                            stamp = time.monotonic_ns()
                            first = first or stamp
                            last = stamp
                            ids += count
            row["end_ns"] = time.monotonic_ns()
            count = usage.get("completion_tokens", ids) if usage else ids
            row.update(first_token_ns=first, last_token_ns=last, completion_tokens=count)
            if not done or first is None or count != request["max_tokens"]:
                raise RuntimeError(f"Incomplete stream: done={done}, tokens={count}/{request['max_tokens']}")
            row["ttft_ms"] = (first - row["send_ns"]) / 1e6
            row["tpot_ms"] = (last - first) / 1e6 / (count - 1) if count > 1 else None
            row["status"] = "success"
        except asyncio.CancelledError:
            row["status"], row["error"] = "cancelled", "Drain timeout or interrupted run"
        except asyncio.TimeoutError:
            row["status"], row["error"] = ("failed" if row.get("send_ns") else "not_sent"), "Request or queue timeout"
        except Exception as error:
            row["status"], row["error"] = "failed", f"{type(error).__name__}: {error}"
        finally:
            if row.get("send_ns") is not None:
                row["end_ns"] = row.get("end_ns") or time.monotonic_ns()
                count = usage.get("completion_tokens", ids) if usage else ids
                row.update(first_token_ns=first, last_token_ns=last, completion_tokens=count)
                if first is not None:
                    row["ttft_ms"] = (first - row["send_ns"]) / 1e6
                    row["tpot_ms"] = (last - first) / 1e6 / (count - 1) if count > 1 else None
                row["e2e_ms"] = (row["end_ns"] - row["send_ns"]) / 1e6
                self.inflight -= 1
                self.records.append(row)
                self.completed[instance.instance_id, request["chain_id"], request["chain_turn"]] = row
            elif row.get("was_queued"):
                row["queued_ms"] = (time.monotonic_ns() - before_queue) / 1e6
                self.unsent_queued.append(dict(request_id=row.get("request_id"), queued_ms=row["queued_ms"]))
            if acquired:
                self.gate.release()
            if row["status"] == "success":
                for child in node["children"]:
                    self.arm(child, row["end_ns"] + round(self.nodes[child]["entry"].delay_s * 1e9), row["end_ns"])
            else:
                stack = list(node["children"])
                while stack:
                    child = stack.pop()
                    self.blocked += 1
                    stack.extend(self.nodes[child]["children"])

    async def poll_metrics(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5), trust_env=False) as client:
            while not self.stop_metrics.is_set():
                stamp = time.monotonic_ns()
                sample = dict(ns=stamp, elapsed_s=(stamp - self.origin_ns) / 1e9)
                try:
                    async with client.get(self.config.base_url.rstrip("/") + "/metrics") as response:
                        response.raise_for_status()
                        sample["values"] = await asyncio.to_thread(parse_metrics, await response.text(), self.config.model)
                except Exception as error:
                    sample["error"] = str(error)
                self.metrics.append(sample)
                with (self.config.output_dir / "metrics.jsonl").open("a") as sink:
                    sink.write(json.dumps(sample) + "\n")
                try:
                    await asyncio.wait_for(self.stop_metrics.wait(), self.config.metrics_interval)
                except asyncio.TimeoutError:
                    pass

    async def monitor_loop(self):
        """Absolute 10 ms clock, reporting missed checks rather than hiding them."""
        planned = self.origin_ns
        path = self.config.output_dir / "loop_lag.parquet"
        schema = pa.schema([("planned_ns",pa.int64()),("wakeup_ns",pa.int64()),
                            ("elapsed_s",pa.float64()),("lag_ms",pa.float64()),("missed_checks",pa.int64())])
        # Compact numpy/Arrow batches avoid hundreds of thousands of Python dicts.
        batch = []
        with pq.ParquetWriter(path,schema,compression="zstd") as sink:
            while not self.stop_metrics.is_set():
                await asyncio.sleep(max(0,(planned-time.monotonic_ns())/1e9))
                wake = time.monotonic_ns()
                lag = max(0,wake-planned)
                missed = lag//10_000_000
                row = (planned,wake,(wake-self.origin_ns)/1e9,lag/1e6,missed)
                batch.append(row)
                self.loop_lags.append((row[2],row[3],missed))
                if len(batch)>=1000:
                    table = pa.Table.from_arrays([pa.array(v,type=f.type) for v,f in zip(zip(*batch),schema)],schema=schema)
                    await asyncio.to_thread(sink.write_table,table)
                    batch.clear()
                planned += (missed+1)*10_000_000
            if batch:
                sink.write_table(pa.Table.from_arrays([pa.array(v,type=f.type) for v,f in zip(zip(*batch),schema)],schema=schema))

    async def run(self, digest):
        # Bound GIL contention from list conversion and JSON encoding. These
        # workers start at the trigger, leaving the full dependency delay.
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix="replay-body") as self.payload_executor:
            return await self._run(digest)

    async def _run(self, digest):
        self.workload_sha = digest
        self.unsent_queued = []
        await asyncio.to_thread(self.prepare)
        self.origin_ns = time.monotonic_ns() + 300_000_000
        self.cutoff_ns = self.origin_ns + round(self.config.duration * 1e9)
        (self.config.output_dir/'experiment-clock.json').write_text(json.dumps(dict(
            origin_ns=self.origin_ns,warmup_s=self.config.warmup,duration_s=self.config.duration,
            cooldown_s=self.config.cooldown))+'\n')
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0), trust_env=False) as self.session:
            monitor = asyncio.create_task(self.poll_metrics())
            loop_monitor = asyncio.create_task(self.monitor_loop())
            for key, relative in self.timers:
                self.arm(key, self.origin_ns + round(relative * 1e9),
                         self.origin_ns + round(self.nodes[key]["instance"].start_s * 1e9))
            try:
                await asyncio.sleep(max(0, (self.cutoff_ns - time.monotonic_ns()) / 1e9))
            finally:
                self.cutoff_ns = min(self.cutoff_ns, time.monotonic_ns())
                for handle in self.handles:
                    handle.cancel()
                if self.jobs:
                    _, pending = await asyncio.wait(list(self.jobs), timeout=self.config.drain_timeout)
                    for job in pending:
                        job.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                self.stop_metrics.set()
                await monitor
                await loop_monitor
        return self.records


def write_records(path, records, tokens):
    with pq.ParquetWriter(path, OUTPUT_SCHEMA, compression="zstd") as writer:
        for start in range(0, len(records), 128):
            batch = [dict(r, prompt_token_ids=tokens.get(r["trace_req_idx"])) for r in records[start:start+128]]
            writer.write_table(pa.Table.from_pylist(batch, schema=OUTPUT_SCHEMA))


def summarize(records, engine, observations):
    window = [r for r in records if r["in_window"]]
    sent_window = [r for r in records if engine.config.warmup <= (r["send_ns"] - engine.origin_ns) / 1e9 <= engine.config.duration - engine.config.cooldown]
    success = [r for r in window if r["status"] == "success"]
    cache = [r for r in success if r.get("gpu_hit") is not None]
    return dict(
        total_sent=len(records), window_sent=len(sent_window),
        window_is_entry=sum(r["is_entry"] for r in sent_window),
        window_requests=len(window), window_success=len(success), window_failed=len(window)-len(success),
        ttft_ms=percentiles([r.get("ttft_ms") for r in success]),
        schedule_absolute_error_ms=percentiles([abs(r["schedule_error_ms"]) for r in window]),
        gap_absolute_error_ms=percentiles([abs(r["gap_error_ms"]) for r in window if r.get("gap_error_ms") is not None]),
        safety_valve_queued=engine.queue_count, safety_valve_triggered=engine.queue_count > 0,
        queued_ms=percentiles([r["queued_ms"] for r in records if r["was_queued"]]),
        unsent_queued=engine.unsent_queued, peak_inflight=engine.peak_inflight,
        dependency_blocked=engine.blocked, cache_source_requests=len(cache),
        nominal_fraction_per_request={k:percentiles([r[k+"_fraction"] for r in cache]) for k in ("gpu_hit", "cpu_hit", "recomputed")},
        nominal_token_weighted_fraction={k:sum(r[k] for r in cache)/sum(r["prompt_len"] for r in cache) if cache else None for k in ("gpu_hit", "cpu_hit", "recomputed")},
        hint_sent=engine.hints_sent, hint_received=observations["valid_hints_received"],
        hint_counts_match=engine.hints_sent == observations["valid_hints_received"],
        observation_join=observations,
        body_build_ms=percentiles([r.get("body_build_ms") for r in window]),
        body_prepare_wait_ms=percentiles([r.get("body_prepare_wait_ms") for r in window]),
        event_loop_lag_ms=percentiles([lag for elapsed,lag,_ in engine.loop_lags
                                     if engine.config.warmup<=elapsed<=engine.config.duration-engine.config.cooldown]),
        event_loop_missed_checks=sum(missed for _,_,missed in engine.loop_lags),
    )


async def run_replay(config, workload, schedule_path, vocabulary, instances=None, server_metadata=None):
    config.validate()
    commit = git_commit(config.allow_dirty)
    if (config.output_dir / "run.json").exists() or (config.output_dir / "requests.parquet").exists():
        raise FileExistsError("Refusing to replace an existing replay run")
    metadata_names = [name for name in SCHEMA.names if name != "prompt_token_ids"]
    requests = pq.read_table(workload, columns=metadata_names, use_threads=False).to_pylist()
    digest = sha256_file(workload)
    manifest = json.loads(Path(workload).with_name("manifest.json").read_text())
    schedule = pq.read_table(schedule_path, use_threads=False)
    if (digest != manifest["workload"]["sha256"] or schedule.schema.metadata[b"agentkv.workload_sha256"].decode() != digest
            or schedule.schema.metadata[b"agentkv.service_time_s"] != b"0.5"):
        raise ValueError("Workload/schedule provenance mismatch")
    by_root, estimates = defaultdict(list), defaultdict(list)
    for row in requests:
        by_root[row["root_id"]].append(row)
    for row in schedule.to_pylist():
        estimates[row["root_id"]].append(row)
    if set(by_root) != set(estimates):
        raise ValueError("Workload/schedule roots differ")
    roots = {rid:RootSchedule(rows, estimates[rid]) for rid, rows in sorted(by_root.items())}
    if instances is None:
        instances = sample_population({rid:r.duration_s for rid, r in roots.items()}, config.arrival_rate,
                                      config.population_seed, config.duration, config.steady_state)
    if len({i.instance_id for i in instances}) != len(instances):
        raise ValueError("Duplicate instance ID")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    info = dict(parameters={k:str(v) if isinstance(v, Path) else v for k,v in asdict(config).items()},
                git_commit=commit, source=dict(workload_sha256=digest, schedule_sha256=sha256_file(schedule_path)),
                vllm_commit=server_metadata.get('vllm_commit') if server_metadata else None,
                started_at=datetime.now(timezone.utc).isoformat(), status="running",
                M0=sum(i.initial for i in instances), instances=[asdict(i) for i in instances],
                server=server_metadata, conventions=dict(prompt_token_ids="original workload; sent token permutation recorded by seed and sha256",
                effective_cache_counts="null unless observed; nominal external hits are CPU for CPU-only offload",
                http_retry="none; failed requests block their dependent subtree"))
    destination = config.output_dir / "run.json"
    destination.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    tokens = TokenStore(workload, requests)
    engine = ReplayEngine(config, requests, tokens, roots, vocabulary, instances)
    records = await engine.run(digest)
    observations = join_observations(records, config.observation_dir)
    write_records(config.output_dir / "requests.parquet", records, tokens)
    info.update(status="complete", origin_ns=engine.origin_ns, cutoff_ns=engine.cutoff_ns,
                completed_at=datetime.now(timezone.utc).isoformat(), summary=summarize(records, engine, observations),
                requests_sha256=sha256_file(config.output_dir / "requests.parquet"))
    info["loop_lag_sha256"] = sha256_file(config.output_dir / "loop_lag.parquet")
    info["token_store"] = dict(mode="preloaded Arrow column before experiment clock", bytes=tokens.column.nbytes)
    destination.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    return info, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "results/m1-workload-seed0-v3/workload.parquet")
    parser.add_argument("--schedule", type=Path, default=ROOT / "results/m1-steady-state-seed0/schedule_estimate.parquet")
    parser.add_argument("--tokenizer", type=Path, default=Path("/data/models/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--run-id", default="m1-replay")
    parser.add_argument("--policy", default="lru")
    parser.add_argument("--hint-tier", choices=["none","perfect","noisy","online"], default="none")
    parser.add_argument("--arrival-rate", type=float, default=1)
    parser.add_argument("--population-seed", type=int, default=0)
    parser.add_argument("--steady-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration", type=float, default=2400)
    parser.add_argument("--warmup", type=float, default=600)
    parser.add_argument("--cooldown", type=float, default=0)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--drain-timeout", type=float, default=120)
    parser.add_argument("--metrics-interval", type=float, default=1)
    parser.add_argument("--observation-dir", type=Path)
    parser.add_argument("--server-manifest", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="DEBUG ONLY: bypass clean-worktree check")
    args = parser.parse_args(argv)
    # Reject dirty runs before tokenizer loading or any network activity.
    git_commit(args.allow_dirty, warn=False)
    vocabulary, vocabulary_info = tokenizer_vocabulary(args.tokenizer)
    manifest = json.loads(args.workload.with_name("manifest.json").read_text())
    if vocabulary_info["eligible_ids_sha256"] != manifest["tokenizer"]["eligible_ids_sha256"]:
        raise ValueError("Tokenizer vocabulary differs from workload")
    config = ReplayConfig(**{name:getattr(args,name) for name in ReplayConfig.__dataclass_fields__})
    server = json.loads(args.server_manifest.read_text()) if args.server_manifest else None
    if server is not None and server.get('type')!='fake' and not server.get('vllm_commit'):
        raise ValueError('Server manifest must identify the vLLM commit')
    info, _ = asyncio.run(run_replay(config, args.workload, args.schedule, vocabulary, server_metadata=server))
    print(json.dumps(info["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
