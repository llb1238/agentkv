# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np
from prometheus_client.parser import text_string_to_metric_families

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
CPU = True
BASE = f"http://127.0.0.1:{args.port}"
CLIENT = httpx.Client(base_url=BASE, trust_env=False, timeout=120)
construction = json.loads(args.fixture.read_text())
P1 = construction["turn1"]
REFERENCE_G = construction["actual_generated"]
TOOL = construction["fake_tool_suffix"]
assert len(P1) == 4096 and len(REFERENCE_G) == 18 and len(TOOL) == 43


def metrics():
    response = CLIENT.get("/metrics")
    response.raise_for_status()
    values = {}
    info = {}
    for family in text_string_to_metric_families(response.text):
        for s in family.samples:
            if (
                s.labels.get("model_name") != "Qwen/Qwen3-8B"
                or s.labels.get("engine") != "0"
                or "le" in s.labels
            ):
                continue
            extra = {
                k: v for k, v in s.labels.items() if k not in ("model_name", "engine")
            }
            key = s.name + (
                "[source=" + extra["source"] + "]" if "source" in extra else ""
            )
            values[key] = s.value
            if s.name == "vllm:cache_config_info":
                info = s.labels
    return response.text, values, info


def idle(expected_count=None):
    deadline = time.monotonic() + 20
    stable = 0
    while True:
        raw, v, info = metrics()
        done = (
            expected_count is None
            or v.get("vllm:e2e_request_latency_seconds_count", 0) >= expected_count
        )
        quiet = (
            v.get("vllm:num_requests_running", 0) == 0
            and v.get("vllm:num_requests_waiting", 0) == 0
        )
        pending = v.get("vllm:diag_cpu_pending_writes", 0) == 0
        if done and quiet and pending:
            stable += 1
            if stable >= 2:
                return raw, v, info
        else:
            stable = 0
        if time.monotonic() > deadline:
            raise RuntimeError(("idle timeout", v))
        time.sleep(0.025)


def reset():
    idle()
    deadline = time.monotonic() + 20
    while True:
        r = CLIENT.post(
            "/reset_prefix_cache", params={"reset_external": str(CPU).lower()}
        )
        r.raise_for_status()
        if r.json().get("success"):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("cache reset failed")
        time.sleep(0.05)
    return idle()


def request(name, prompt, max_tokens):
    raw, before, info = idle()
    (OUT / (name + "-before.prom")).write_text(raw)
    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "request_id": name,
    }
    ids = []
    text = []
    chunks = []
    first_token = None
    last_token = None
    first_text = None
    last_text = None
    done = None
    usage = None
    start_ns = time.monotonic_ns()
    start = start_ns / 1e9
    with CLIENT.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            content = line[5:].strip()
            at = time.monotonic_ns() / 1e9
            if content == "[DONE]":
                done = at
                break
            data = json.loads(content)
            if "error" in data:
                raise RuntimeError(data["error"])
            chunks.append({"elapsed_ms": (at - start) * 1000, "data": data})
            if data.get("usage") is not None:
                usage = data["usage"]
            for choice in data.get("choices", []):
                delta_ids = choice.get("token_ids") or []
                delta_text = choice.get("text") or ""
                if delta_ids:
                    if first_token is None:
                        first_token = at
                    last_token = at
                    ids.extend(delta_ids)
                if delta_text:
                    if first_text is None:
                        first_text = at
                    last_text = at
                    text.append(delta_text)
    end = time.monotonic_ns() / 1e9
    assert done is not None and first_text is not None and ids, (name, chunks)
    assert usage is not None and usage["prompt_tokens"] == len(prompt), (name, usage)
    assert usage["completion_tokens"] == len(ids), (name, usage, len(ids))
    raw, after, _ = idle(before.get("vllm:e2e_request_latency_seconds_count", 0) + 1)
    (OUT / (name + "-after.prom")).write_text(raw)

    def delta(key):
        return after.get(key, 0) - before.get(key, 0)

    gpu = int(delta("vllm:prefix_cache_hits_total"))
    cpu = int(delta("vllm:prompt_tokens_cached_by_source_total[source=host]"))
    computed = int(delta("vllm:request_prefill_kv_computed_tokens_sum"))
    assert gpu + cpu + computed == len(prompt), (name, gpu, cpu, computed)
    row = {
        "name": name,
        "start_ns": start_ns,
        "first_token_ns": round(first_token * 1e9),
        "first_text_ns": round(first_text * 1e9),
        "request_rate": "sequential",
        "input_tokens": len(prompt),
        "output_tokens": len(ids),
        "gpu_hit_tokens": gpu,
        "cpu_hit_tokens": cpu,
        "local_prefill_tokens": computed,
        "partition_ok": True,
        "client_ttft_text_ms": (first_text - start) * 1000,
        "client_ttft_token_ms": (first_token - start) * 1000,
        "client_visible_completion_ms": (last_text - start) * 1000,
        "client_e2e_ms": (end - start) * 1000,
        "client_tpot_ms": (last_token - first_token) * 1000 / (len(ids) - 1)
        if len(ids) > 1
        else None,
        "cpu_to_gpu_bytes": int(delta("vllm:kv_offload_load_bytes_total")),
        "cpu_evicted_chunks": int(delta("vllm:diag_cpu_evicted_chunks_total")),
        "cpu_resident_after": after.get("vllm:diag_cpu_resident_chunks"),
        "original_cpu_ready_after": after.get("vllm:diag_original_cpu_ready_chunks"),
        "original_cpu_total_after": after.get("vllm:diag_original_cpu_total_keys"),
        "generated_ids": ids,
        "generated_text": "".join(text),
        "usage": usage,
    }
    (OUT / (name + "-request.json")).write_text(
        json.dumps(payload, ensure_ascii=False) + "\n"
    )
    (OUT / (name + "-stream.json")).write_text(
        json.dumps(chunks, ensure_ascii=False) + "\n"
    )
    (OUT / (name + "-result.json")).write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "HTTP_REQUEST "
        + json.dumps(
            {
                k: v
                for k, v in row.items()
                if k not in ("generated_ids", "generated_text", "usage")
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return row, after


# Wait for the separately launched official vLLM server.
deadline = time.monotonic() + 300
while True:
    try:
        r = CLIENT.get("/health", timeout=2)
        if r.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    if time.monotonic() > deadline:
        raise RuntimeError("server readiness timeout")
    time.sleep(1)
print("HTTP_SERVER_READY", BASE, flush=True)
# Identical shape/decode warmups on each server; excluded from recorded trials.
reset()
warm, _ = request("warmup.turn1", P1, 64)
assert warm["generated_ids"] == REFERENCE_G, (warm["generated_ids"], REFERENCE_G)
P2 = P1 + warm["generated_ids"] + TOOL
assert len(P2) == 4157
request("warmup.turn2", P2, 32)
request(
    "warmup.noise",
    np.random.default_rng(100).integers(100, 100000, size=6000).tolist(),
    1,
)
results = []
settings = (
    [("gpu_quiet", 0), ("cpu_keep", 2), ("cpu_full", 6)] if CPU else [("no_cpu", 2)]
)
for trial in range(3):
    order = settings[trial % len(settings) :] + settings[: trial % len(settings)]
    for scenario, count in order:
        reset()
        name = f"trial{trial + 1}.{scenario}"
        session_start = time.perf_counter()
        first, state = request(name + ".turn1", P1, 64)
        assert first["generated_ids"] == REFERENCE_G
        p2 = P1 + first["generated_ids"] + TOOL
        assert (
            p2[: len(P1)] == P1
            and p2[len(P1) : len(P1) + len(first["generated_ids"])]
            == first["generated_ids"]
        )
        assert first["gpu_hit_tokens"] == 0 and first["cpu_hit_tokens"] == 0
        if CPU:
            assert state["vllm:diag_original_cpu_ready_chunks"] == 256, state
        evictions_start = state.get("vllm:diag_cpu_evicted_chunks_total", 0)
        progression = []
        time.sleep(0.01)
        rng = np.random.default_rng(100)
        for i in range(count):
            noise = rng.integers(100, 100000, size=6000).tolist()
            item, state = request(name + f".noise{i + 1}", noise, 1)
            progression.append(
                {
                    "noise_request": i + 1,
                    "tokens": 6000,
                    "cpu_evictions_this_request": item["cpu_evicted_chunks"],
                    "cpu_resident_chunks": state.get("vllm:diag_cpu_resident_chunks"),
                    "original_cpu_ready_chunks": state.get(
                        "vllm:diag_original_cpu_ready_chunks"
                    ),
                }
            )
        before_second = dict(state)
        second, state = request(name + ".turn2", p2, 32)
        assert second["output_tokens"] == 12, second
        second["matches_previous_ids"] = (
            second["generated_ids"] == construction["reference_second_ids"]
        )
        if scenario == "gpu_quiet":
            assert (
                second["gpu_hit_tokens"],
                second["cpu_hit_tokens"],
                second["local_prefill_tokens"],
            ) == (4112, 0, 45)
        elif scenario == "cpu_keep":
            assert (
                second["gpu_hit_tokens"],
                second["cpu_hit_tokens"],
                second["local_prefill_tokens"],
            ) == (0, 4096, 61)
            assert before_second["vllm:diag_original_cpu_ready_chunks"] == 256
            assert (
                before_second["vllm:diag_cpu_evicted_chunks_total"] - evictions_start
                == 0
            )
            assert second["cpu_to_gpu_bytes"] == 603979776
        else:
            assert (
                second["gpu_hit_tokens"],
                second["cpu_hit_tokens"],
                second["local_prefill_tokens"],
            ) == (0, 0, 4157)
            assert second["cpu_to_gpu_bytes"] == 0
            if CPU:
                assert before_second["vllm:diag_original_cpu_ready_chunks"] == 0
                assert (
                    before_second["vllm:diag_cpu_evicted_chunks_total"]
                    - evictions_start
                    > 0
                )
                assert before_second["vllm:diag_cpu_resident_chunks"] == 1695
        record = {
            "trial": trial + 1,
            "scenario": scenario,
            "turn1": first,
            "turn2": second,
            "interference_count": count,
            "interference_tokens": count * 6000,
            "cpu_evictions_before_turn2": int(
                before_second.get("vllm:diag_cpu_evicted_chunks_total", 0)
                - evictions_start
            ),
            "original_cpu_ready_before_turn2": before_second.get(
                "vllm:diag_original_cpu_ready_chunks"
            ),
            "cpu_resident_before_turn2": before_second.get(
                "vllm:diag_cpu_resident_chunks"
            ),
            "pressure_progression": progression,
            "exact_id_concat": True,
            "test_session_wall_ms_including_metrics_and_interference": (
                time.perf_counter() - session_start
            )
            * 1000,
        }
        results.append(record)
        (OUT / "sessions.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            "HTTP_SESSION "
            + json.dumps(
                {k: v for k, v in record.items() if k not in ("turn1", "turn2")},
                ensure_ascii=False,
            ),
            flush=True,
        )
CLIENT.close()
print("M0_TRACE_PASS", flush=True)
