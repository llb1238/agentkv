"""Offline contracts, plus full-corpus artifact validation callable without GPU."""

import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from agentkv.bench.estimate_replay import simulate_requests
from agentkv.bench.subset import sample_roots
from agentkv.bench.synth import Scale, Synthesizer, check_positions, common_prefix
from agentkv.bench.trace_loader import LoadedTrace, TraceRequest, load_trace, sha256_file
from agentkv.bench.workload import (
    BuildConfig, chain_start_plan, online_gap_prior, root_timeline, write_workload,
)


@pytest.fixture
def fixture_trace():
    requests = []

    def request(root, chain, turn, hashes, t, api, output=8, child=False):
        requests.append(
            TraceRequest(
                root,
                f"{root}/{chain}",
                turn,
                f"fixture/{len(requests)}",
                np.asarray(hashes, dtype=np.int64),
                t,
                api,
                len(hashes) * 64,
                output,
                child,
            )
        )

    request("a", "main", 0, [1, 2], 0, 1)
    request("a", "main", 1, [1, 2, 3], 5, 2)
    request("a", "main", 2, [1, 4], 300, 3)
    request("a", "main", 3, [1, 4, 8], 500, 1)
    request("a", "child", 0, [1, 2, 5], 10, 2, child=True)
    request("a", "child", 1, [1, 2, 5, 6], 40, 1, child=True)
    request("a", "aux", 0, [7], 0, 1, output=0)
    request("b", "main", 0, [1, 2], 0, 1)
    request("b", "main", 1, [1, 2, 3], 2, 1)
    roots = {rid: [r for r in requests if r.root_id == rid] for rid in ("a", "b")}
    chains = {r.chain_id: {} for r in requests}
    return LoadedTrace(roots, chains, {"trace_sha256": "unit-fixture"})


@pytest.fixture
def vocabulary():
    return np.arange(100, 151000, dtype=np.int32)


@pytest.fixture
def prior(tmp_path):
    # Held-out intervals [10, 40, 500] -> [5, 20, 60] at c=0.5.
    path = tmp_path / "prior-requests.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(session_id="held-out", chain_id="held-out/main", chain_turn=i,
                     start_s=float(t), api_time_s=1.0)
                for i, t in enumerate((0, 11, 52, 553))
            ]
            + [dict(session_id="a", chain_id="a/main", chain_turn=i,
                    start_s=float(i * 10000), api_time_s=1.0) for i in range(2)]
        ),
        path,
    )
    return online_gap_prior(path, ["a", "b"], BuildConfig(gap_compress=0.5))


@pytest.fixture
def built(tmp_path, fixture_trace, vocabulary, prior):
    config = BuildConfig(num_roots=2, context_budget=160, gap_compress=0.5)
    manifest = write_workload(fixture_trace, vocabulary, {}, config, tmp_path / "first", prior)
    rows = pq.read_table(tmp_path / "first/workload.parquet").to_pylist()
    return config, manifest, rows, tmp_path / "first"


def test_deterministic_bytes_and_root_sampling(
    built, tmp_path, fixture_trace, vocabulary, prior
):
    config, manifest, _, path = built
    repeat = write_workload(fixture_trace, vocabulary, {}, config, tmp_path / "second", prior)
    assert sha256_file(path / "workload.parquet") == repeat["workload"]["sha256"]
    assert repeat["workload"]["sha256"] == manifest["workload"]["sha256"]
    population = [str(i) for i in range(393)]
    chosen = sample_roots(population, 0, 100)
    assert len(chosen) == len(set(chosen)) == 100
    assert chosen == sample_roots(population[::-1], 0, 100)
    assert chosen != sample_roots(population, 1, 100)
    with pytest.raises(ValueError):
        sample_roots(population, 0, 394)


def test_positions_cover_children_and_reject_conflicting_hashes(
    fixture_trace, vocabulary
):
    requests = [r for group in fixture_trace.roots.values() for r in group]
    assert check_positions(requests) == 11
    bad = replace(requests[0], hash_ids=np.asarray([2, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="position conflict"):
        check_positions(requests + [bad])
    scale = Scale.for_root(fixture_trace.roots["a"], 160)
    synth = Synthesizer("a", scale, vocabulary)
    synth.prompt(requests[0])
    with pytest.raises(ValueError, match="position conflict"):
        synth.prompt(bad)


def test_lcp_preserves_appends_context_edits_and_child_forks(built, fixture_trace):
    _, _, rows, _ = built
    actual = {(r["chain_id"], r["chain_turn"]): r for r in rows}
    for root, requests in fixture_trace.roots.items():
        scale = Scale.for_root(requests, 160)
        for a in requests:
            for b in requests:
                expected = scale.q(common_prefix(a.hash_ids, b.hash_ids))
                left = np.asarray(
                    actual[(a.chain_id, a.chain_turn)]["prompt_token_ids"]
                )
                right = np.asarray(
                    actual[(b.chain_id, b.chain_turn)]["prompt_token_ids"]
                )
                assert common_prefix(left, right) == expected, (root, a, b)
    assert (
        actual[("a/main", 2)]["lcp_prev_tokens"] < actual[("a/main", 1)]["prompt_len"]
    )
    assert (
        actual[("a/main", 0)]["prompt_token_ids"]
        != actual[("b/main", 0)]["prompt_token_ids"]
    )


def test_budget_lengths_vocabulary_and_zero_output(built, fixture_trace, vocabulary):
    config, _, rows, _ = built
    scale_a = Scale.for_root(fixture_trace.roots["a"], config.context_budget)
    assert [scale_a.q(k) for k in range(5)] == [0, 38, 77, 116, 155]
    source = {
        (r.chain_id, r.chain_turn): r for rs in fixture_trace.roots.values() for r in rs
    }
    for row in rows:
        request = source[(row["chain_id"], row["chain_turn"])]
        scale = Scale.for_root(
            fixture_trace.roots[request.root_id], config.context_budget
        )
        assert (
            row["prompt_len"]
            == len(row["prompt_token_ids"])
            == scale.floor(request.input_len)
        )
        assert row["max_tokens"] == max(1, scale.ceil(request.output_len))
        assert row["prompt_len"] + row["max_tokens"] <= config.context_budget
        assert np.isin(row["prompt_token_ids"], vocabulary).all()
    zero = [r for r in rows if r["out_zero_orig"]]
    assert len(zero) == 1 and zero[0]["max_tokens"] == 1
    with pytest.raises(ValueError, match="at least 2"):
        Scale.for_root(fixture_trace.roots["a"], 1)


def test_gaps_cap_compression_final_and_causal_hints(built):
    _, manifest, rows, _ = built
    by_chain = defaultdict(list)
    for row in rows:
        by_chain[row["chain_id"]].append(row)
        if row["final"]:
            assert row["gap_after_s"] is None and not row["gap_truncated"]
            assert all(
                row[k] is None
                for k in (
                    "hint_perfect_ms",
                    "hint_noisy_ms",
                    "hint_online_ms",
                    "hint_online_p90_ms",
                )
            )
        else:
            assert row["gap_after_s"] >= 0
            assert row["gap_after_s"] == min(row["gap_after_orig_s"], 120) * 0.5
            assert row["hint_perfect_ms"] == 1000 * row["gap_after_s"]
            assert 0.5 <= row["hint_noisy_ms"] / row["hint_perfect_ms"] <= 2
    main = by_chain["a/main"]
    assert [r["gap_after_s"] for r in main] == [2, 60, 60, None]
    assert main[1]["hint_online_ms"] == 2000
    assert main[2]["hint_online_ms"] == 31000
    assert main[2]["hint_online_p90_ms"] == pytest.approx(54200)
    valid = [r for r in rows if not r["final"]]
    clipped = sum(r["gap_truncated"] for r in valid)
    assert manifest["statistics"]["gap_truncated_fraction"] == clipped / len(valid)
    assert (
        main[0]["hint_online_ms"]
        == manifest["conventions"]["global_median_prior_s"] * 1000
    )
    assert main[0]["hint_online_ms"] == main[0]["hint_online_p90_ms"] == 20000
    assert manifest["conventions"]["online_prior_interval_count"] == 3
    assert manifest["conventions"]["online_prior_root_ids"] == ["held-out"]


def test_prior_uses_only_held_out_roots_and_requires_intervals(tmp_path, prior):
    assert prior.median_s == 20 and prior.interval_count == 3
    with pytest.raises(ValueError, match="held-out chain interval"):
        online_gap_prior(tmp_path / "prior-requests.parquet", ["a", "held-out"], BuildConfig())


@pytest.mark.parametrize("compression", [0.0, 0.5, 1.0, 2.0])
def test_root_offsets_preserve_overlap_and_cap_three_day_idle(
    tmp_path, vocabulary, prior, compression
):
    idle = 3 * 24 * 3600
    requests = [
        TraceRequest("r", chain, turn, f"{chain}/{turn}", np.array([1]), t, api, 64, 1, False)
        for chain, turn, t, api in (
            ("main", 0, 0, 10),
            ("child", 0, 5, 20),
            ("main", 1, 25 + idle, 10),
            ("child", 1, 30 + idle, 20),
            ("aux", 0, 27 + idle, 2),
        )
    ]
    config = BuildConfig(num_roots=1, gap_cap=120, gap_compress=compression)
    mapped, duration = root_timeline(requests[::-1], config)
    assert mapped["child/0"] - mapped["main/0"] == 5 * compression
    assert mapped["child/1"] - mapped["main/1"] == 5 * compression
    assert mapped["aux/0"] - mapped["main/1"] == 2 * compression
    assert mapped["main/1"] - (5 + 20) * compression == 120 * compression
    assert duration == 170 * compression
    loaded = LoadedTrace({"r": requests}, {r.chain_id: {} for r in requests}, {})
    # The held-out fixture prior is independent of this root's timing.
    write_workload(loaded, vocabulary, {}, config, tmp_path / "compressed",
                   replace(prior, median_s=40 * compression))
    rows = pq.read_table(tmp_path / "compressed/workload.parquet").to_pylist()
    heads = {r["chain_id"]: r for r in rows if r["chain_turn"] == 0}
    assert heads["child"]["chain_offset_s"] - heads["main"]["chain_offset_s"] == 5 * compression
    assert heads["aux"]["chain_offset_s"] == 147 * compression
    assert heads["aux"]["chain_offset_orig_s"] == 27 + idle
    assert all(r["chain_offset_orig_s"] is None for r in rows if r["chain_turn"])


def test_root_timeline_merges_nested_touching_and_zero_duration_intervals(fixture_trace):
    template = fixture_trace.roots["a"][0]
    requests = [
        replace(template, trace_req_idx=str(i), t=t, api_time=api)
        for i, (t, api) in enumerate(((200, 1000), (210, 1), (1100, 0), (1200, 5), (1325, 0), (2325, 0)))
    ]
    mapped, duration = root_timeline(requests, BuildConfig())
    assert list(mapped.values()) == [200, 210, 1100, 1200, 1325, 1445]
    assert duration == 1445


def test_workload_rejects_prior_from_selected_roots(tmp_path, fixture_trace, vocabulary, prior):
    with pytest.raises(ValueError, match="overlap"):
        write_workload(fixture_trace, vocabulary, {}, BuildConfig(num_roots=2), tmp_path / "bad",
                       replace(prior, root_ids=("a",)))


@pytest.fixture
def anchor_trace(fixture_trace):
    template = fixture_trace.roots["a"][0]
    requests = [
        replace(template, root_id="r", chain_id=chain, chain_turn=turn,
                trace_req_idx=idx, t=t, api_time=api, is_subagent=child)
        for chain, turn, idx, t, api, child in (
            ("parent", 0, "p0", 0, 10, False),
            ("parent", 1, "p1", 50, 5, False),
            ("child-a", 0, "a0", 12, 3, True),
            ("child-b", 0, "b0", 12, 4, True),
            ("early", 0, "e0", 4, 1, False),
            ("later", 0, "l0", 60, 1, False),
            ("idle", 0, "i0", 3 * 86400 + 60, 1, False),
        )
    ]
    return LoadedTrace({"r": requests}, {r.chain_id: {} for r in requests}, {})


def test_parallel_subagents_share_latest_completed_parent(anchor_trace):
    plan = chain_start_plan(anchor_trace, BuildConfig())
    assert plan["child-a"] == plan["child-b"] == ("p0", 2)
    assert plan["later"] == ("p1", 5)  # Anchor may be a non-head request.
    assert plan["idle"] == ("l0", 120)
    shuffled = LoadedTrace({"r": anchor_trace.roots["r"][::-1]}, {}, {})
    assert chain_start_plan(shuffled, BuildConfig()) == plan


def test_no_anchor_uses_root_arrival_and_caps_initial_delay(anchor_trace):
    late = replace(anchor_trace.roots["r"][0], root_id="other", chain_id="other",
                   trace_req_idx="other/0", t=200)
    trace = LoadedTrace({**anchor_trace.roots, "other": [late]}, {}, {})
    plan = chain_start_plan(trace, BuildConfig(gap_compress=0.5))
    assert plan["parent"] == (None, 0)
    assert plan["early"] == (None, 2)
    assert plan["other"] == (None, 60)  # No anchor from a different root.
    assert plan["child-a"] == plan["child-b"] == ("p0", 1)


def test_anchor_dependencies_are_earlier_and_acyclic(anchor_trace):
    requests = anchor_trace.roots["r"]
    by_id = {r.trace_req_idx: r for r in requests}
    by_turn = {(r.chain_id, r.chain_turn): r for r in requests}
    plan = chain_start_plan(anchor_trace, BuildConfig())
    dependencies = {}
    for request in requests:
        if request.chain_turn:
            dependencies[request.trace_req_idx] = by_turn[
                request.chain_id, request.chain_turn - 1
            ].trace_req_idx
        else:
            anchor, _ = plan[request.chain_id]
            dependencies[request.trace_req_idx] = anchor
            if anchor is not None:
                parent = by_id[anchor]
                assert parent.root_id == request.root_id
                assert parent.chain_id != request.chain_id
                assert parent.t + parent.api_time <= request.t
                assert parent.t < request.t
    for idx in by_id:
        seen = set()
        while idx is not None:
            assert idx not in seen
            seen.add(idx)
            idx = dependencies[idx]


def test_simultaneous_zero_duration_anchors_have_deterministic_order(anchor_trace):
    template = anchor_trace.roots["r"][0]
    requests = [
        replace(template, chain_id=chain, chain_turn=turn, trace_req_idx=idx,
                t=0, api_time=0)
        for chain, turn, idx in (("a", 0, "a0"), ("a", 1, "a1"),
                                 ("b", 0, "b0"), ("c", 0, "c0"))
    ]
    loaded = LoadedTrace({"r": requests[::-1]}, {}, {})
    assert chain_start_plan(loaded, BuildConfig()) == {
        "a": (None, 0), "b": ("a0", 0), "c": ("b0", 0),
    }


def test_anchor_columns_are_written_only_on_chain_heads(
    tmp_path, anchor_trace, vocabulary, prior
):
    path = tmp_path / "anchored"
    manifest = write_workload(anchor_trace, vocabulary, {}, BuildConfig(num_roots=1), path, prior)
    rows = pq.read_table(path / "workload.parquet").to_pylist()
    heads = {r["chain_id"]: r for r in rows if r["chain_turn"] == 0}
    assert heads["child-a"]["anchor_trace_req_idx"] == "p0"
    assert heads["child-b"]["anchor_trace_req_idx"] == "p0"
    assert heads["child-a"]["start_delay_s"] == heads["child-b"]["start_delay_s"] == 2
    assert heads["early"]["anchor_trace_req_idx"] is None
    assert heads["early"]["start_delay_s"] == 4
    for row in rows:
        if row["chain_turn"]:
            assert row["anchor_trace_req_idx"] is row["start_delay_s"] is None
    assert manifest["statistics"]["chains_with_request_anchor"] == 4
    assert manifest["statistics"]["chains_anchored_at_root_arrival"] == 2


def test_loader_joins_nested_positions_without_reinferring_chains(tmp_path):
    rid = "root"
    entries = [
        {
            "type": "s",
            "t": 0.0,
            "api_time": 1.0,
            "in": 128,
            "out": 4,
            "hash_ids": [1, 2],
        },
        {
            "type": "subagent",
            "t": 2.0,
            "agent_id": "child",
            "requests": [
                {
                    "type": "n",
                    "t": 2.0,
                    "api_time": 1.0,
                    "in": 192,
                    "out": 0,
                    "hash_ids": [1, 2, 3],
                }
            ],
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {"id": rid, "block_size": 64, "hash_id_scope": "local", "requests": entries}
        )
        + "\n"
    )
    metadata = []
    for i, (container, req) in enumerate(
        (
            ("root::root", entries[0]),
            ("root::subagent:child:entry:1", entries[1]["requests"][0]),
        )
    ):
        metadata.append(
            dict(
                session_id=rid,
                container_id=container,
                original_array_index=0,
                chain_id=f"c{i}",
                chain_turn=0,
                start_s=req["t"],
                api_time_s=req["api_time"],
                input_tokens=req["in"],
                output_tokens=req["out"],
            )
        )
    pq.write_table(pa.Table.from_pylist(metadata), tmp_path / "requests.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    session_id=rid,
                    chain_id=f"c{i}",
                    request_count=1,
                    first_start_s=float(2 * i),
                )
                for i in range(2)
            ]
        ),
        tmp_path / "chains.parquet",
    )
    loaded = load_trace(
        trace,
        tmp_path / "requests.parquet",
        tmp_path / "chains.parquet",
        [rid],
        sha256_file(trace),
    )
    child = loaded.roots[rid][1]
    assert child.is_subagent and child.trace_req_idx == "1/requests/1/requests/0"
    assert child.chain_id == "c1" and child.chain_turn == 0
    assert child.hash_ids.tolist() == [1, 2, 3] and child.output_len == 0
    metadata[1]["input_tokens"] = 128
    pq.write_table(pa.Table.from_pylist(metadata), tmp_path / "requests.parquet")
    with pytest.raises(ValueError, match="Raw/audit mismatch"):
        load_trace(
            trace,
            tmp_path / "requests.parquet",
            tmp_path / "chains.parquet",
            [rid],
            sha256_file(trace),
        )


def verify_generated_workload(
    first: Path,
    second: Path,
    loaded: LoadedTrace,
    vocabulary: np.ndarray,
    pairs_per_root=128,
):
    """Validate every real row and adjacent pair, plus fixed same-root random pairs."""
    started = time.monotonic()
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (first, second)]
    hashes = [sha256_file(p / "workload.parquet") for p in (first, second)]
    assert hashes[0] == hashes[1]
    assert all(m["workload"]["sha256"] == h for m, h in zip(manifests, hashes))
    manifest = manifests[0]
    config = BuildConfig(**manifest["parameters"])
    assert set(manifest["selected_roots"]) == set(loaded.roots)
    proof = loaded.provenance["position_check"]
    assert proof["violating_occurrences"] == 0 and proof["root_sessions"] == 393
    for field in (
        "trace_sha256",
        "requests_sha256",
        "inferred_chains_sha256",
        "position_check_sha256",
    ):
        assert manifest["source"][field] == loaded.provenance[field]
    expected = {
        (r.chain_id, r.chain_turn): r for rs in loaded.roots.values() for r in rs
    }
    expected_starts = {}
    by_id = {r.trace_req_idx: r for r in expected.values()}
    for requests in loaded.roots.values():
        for head in (r for r in requests if r.chain_turn == 0):
            head_order = (head.t, head.chain_turn, head.chain_id, head.trace_req_idx)
            candidates = [
                r for r in requests
                if r.chain_id != head.chain_id and r.t + r.api_time <= head.t
                and (r.t, r.chain_turn, r.chain_id, r.trace_req_idx) < head_order
            ]
            anchor = max(candidates, default=None,
                         key=lambda r: (r.t + r.api_time, r.t, r.chain_turn,
                                        r.chain_id, r.trace_req_idx))
            raw = head.t if anchor is None else head.t - (anchor.t + anchor.api_time)
            expected_starts[head.chain_id] = (
                None if anchor is None else anchor.trace_req_idx,
                min(raw, config.gap_cap) * config.gap_compress,
            )
    scales = {
        rid: Scale.for_root(rs, config.context_budget)
        for rid, rs in loaded.roots.items()
    }
    # Independent interval-union implementation to check every chain head.
    expected_offsets, timeline_durations = {}, []
    for rid, requests in loaded.roots.items():
        merged = []
        for left, right in sorted((r.t, r.t + r.api_time) for r in requests):
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        cuts = [(right[0], max(0.0, right[0] - left[1] - config.gap_cap))
                for left, right in zip(merged, merged[1:])]
        for request in requests:
            if request.chain_turn == 0:
                removed = sum(cut for boundary, cut in cuts if boundary <= request.t)
                expected_offsets[request.chain_id] = (request.t - removed) * config.gap_compress
        timeline_durations.append((merged[-1][1] - sum(cut for _, cut in cuts)) * config.gap_compress)
    conventions = manifest["conventions"]
    assert not set(conventions["online_prior_root_ids"]) & set(loaded.roots)
    assert conventions["online_prior_interval_count"] > 0
    assert conventions["online_prior_root_count"] == len(conventions["online_prior_root_ids"])
    prior = conventions["global_median_prior_s"]
    pairs, wanted = {}, {}
    rng = np.random.Generator(np.random.PCG64(7121))
    for rid, requests in sorted(loaded.roots.items()):
        ordered = sorted(requests, key=lambda r: (r.chain_id, r.chain_turn))
        choices = (
            [
                rng.choice(len(ordered), size=2, replace=False)
                for _ in range(pairs_per_root)
            ]
            if len(ordered) > 1
            else []
        )
        pairs[rid] = [
            (
                (ordered[a].chain_id, ordered[a].chain_turn),
                (ordered[b].chain_id, ordered[b].chain_turn),
            )
            for a, b in choices
        ]
        wanted[rid] = {key for pair in pairs[rid] for key in pair}
    allowed = np.zeros(int(vocabulary.max()) + 1, dtype=bool)
    allowed[vocabulary] = True
    seen, gaps = set(), []
    clipped = adjacent_count = random_count = 0
    max_offset_error = 0.0
    current_root, previous, previous_tokens, cached = None, None, None, {}
    history, ema = [], None
    timing_rows = []

    def check_random(rid):
        for a, b in pairs[rid]:
            shared = common_prefix(expected[a].hash_ids, expected[b].hash_ids)
            assert common_prefix(cached[a], cached[b]) == scales[rid].q(shared), (
                rid,
                a,
                b,
            )
        return len(pairs[rid])

    reader = pq.ParquetFile(first / "workload.parquet")
    assert reader.schema_arrow.equals(
        pq.ParquetFile(second / "workload.parquet").schema_arrow
    )
    for batch in reader.iter_batches(batch_size=128):
        tokens = batch.column("prompt_token_ids")
        flat = tokens.values.to_numpy()
        offsets = tokens.offsets.to_numpy()
        metadata = batch.select(
            [n for n in batch.schema.names if n != "prompt_token_ids"]
        ).to_pylist()
        for i, row in enumerate(metadata):
            rid, key = row["root_id"], (row["chain_id"], row["chain_turn"])
            assert key not in seen
            seen.add(key)
            request = expected[key]
            timing_rows.append({name: row[name] for name in (
                "root_id", "chain_id", "chain_turn", "trace_req_idx",
                "anchor_trace_req_idx", "start_delay_s", "gap_after_s",
            )})
            prompt = flat[offsets[i] : offsets[i + 1]]
            if current_root != rid:
                if current_root is not None:
                    random_count += check_random(current_root)
                current_root, previous, previous_tokens, cached = rid, None, None, {}
            assert (
                len(prompt) == row["prompt_len"] == scales[rid].floor(request.input_len)
            )
            assert row["prompt_len"] + row["max_tokens"] <= config.context_budget
            assert row["max_tokens"] == max(1, scales[rid].ceil(request.output_len))
            assert row["out_zero_orig"] == (request.output_len == 0)
            assert (
                prompt.min() >= 0
                and prompt.max() < len(allowed)
                and allowed[prompt].all()
            )
            assert row["trace_req_idx"] == request.trace_req_idx
            if request.chain_turn:
                assert (
                    previous.chain_id == request.chain_id
                    and previous.chain_turn + 1 == request.chain_turn
                )
                predicted = scales[rid].q(
                    common_prefix(previous.hash_ids, request.hash_ids)
                )
                actual = common_prefix(previous_tokens, prompt)
                assert actual == predicted == row["lcp_prev_tokens"], (
                    rid,
                    key,
                    actual,
                    predicted,
                )
                adjacent_count += 1
                assert row["chain_offset_s"] is None
                assert row["chain_offset_orig_s"] is None
                assert row["anchor_trace_req_idx"] is None
                assert row["start_delay_s"] is None
            else:
                assert row["lcp_prev_tokens"] == 0
                expected_offset = expected_offsets[request.chain_id]
                max_offset_error = max(max_offset_error, abs(row["chain_offset_s"] - expected_offset))
                # Independent summation can differ at float64 rounding scale.
                assert math.isclose(row["chain_offset_s"], expected_offset, rel_tol=0, abs_tol=1e-8)
                assert row["chain_offset_orig_s"] == request.t
                assert (row["anchor_trace_req_idx"], row["start_delay_s"]) == expected_starts[request.chain_id]
                history, ema = [], None
            next_request = expected.get((request.chain_id, request.chain_turn + 1))
            assert row["final"] == (next_request is None)
            if next_request is None:
                assert row["gap_after_s"] is None and not row["gap_truncated"]
                assert all(
                    row[n] is None
                    for n in (
                        "hint_perfect_ms",
                        "hint_noisy_ms",
                        "hint_online_ms",
                        "hint_online_p90_ms",
                    )
                )
            else:
                raw = next_request.t - (request.t + request.api_time)
                gap = min(raw, config.gap_cap) * config.gap_compress
                assert math.isfinite(row["gap_after_s"]) and row["gap_after_s"] >= 0
                assert row["gap_after_s"] == gap and row["gap_after_orig_s"] == raw
                assert row["gap_truncated"] == (raw > config.gap_cap)
                assert row["hint_perfect_ms"] == gap * 1000
                assert row["hint_online_ms"] == (prior if ema is None else ema) * 1000
                p90 = prior if not history else float(np.quantile(history, 0.9, method="linear"))
                assert row["hint_online_p90_ms"] == p90 * 1000
                history.append(gap)
                ema = gap if ema is None else config.ema_alpha * gap + (1 - config.ema_alpha) * ema
                if gap:
                    assert 0.5 <= row["hint_noisy_ms"] / (gap * 1000) <= 2
                gaps.append(gap)
                clipped += row["gap_truncated"]
            if key in wanted[rid]:
                cached[key] = prompt.copy()
            previous, previous_tokens = request, prompt.copy()
    if current_root is not None:
        random_count += check_random(current_root)
    assert seen == set(expected)
    stats = manifest["statistics"]
    assert len(seen) == stats["requests"]
    assert clipped == stats["gaps_truncated"]
    assert stats["gap_truncated_fraction"] == clipped / len(gaps)
    assert float(np.median(gaps)) == stats["gap_after_s"]["median"]
    assert float(np.quantile(gaps, 0.9, method="linear")) == stats["gap_after_s"]["p90"]
    for name, value in (
        ("median", float(np.median(timeline_durations))),
        ("p90", float(np.quantile(timeline_durations, 0.9, method="linear"))),
        ("maximum", max(timeline_durations)),
    ):
        assert math.isclose(value, stats["root_timeline_duration_s"][name], rel_tol=0, abs_tol=1e-8)
    assert stats["chains_with_request_anchor"] == sum(a is not None for a, _ in expected_starts.values())
    assert stats["chains_anchored_at_root_arrival"] == sum(a is None for a, _ in expected_starts.values())
    # Independent original-event traversal checks the actual persisted DAG and
    # every simulator send time, including dependencies on interior chain turns.
    ordered_events = sorted(by_id.values(), key=lambda r: (
        r.t, r.chain_turn, r.chain_id, r.trace_req_idx
    ))
    for service in (0.2, 0.5, 1.0):
        ends, reference_sends = {}, {}
        for request in ordered_events:
            if request.chain_turn:
                previous = expected[request.chain_id, request.chain_turn - 1]
                parent = previous.trace_req_idx
                raw = request.t - (previous.t + previous.api_time)
                delay = min(raw, config.gap_cap) * config.gap_compress
            else:
                parent, delay = expected_starts[request.chain_id]
            assert parent is None or parent in ends, "Dependency must precede its request"
            send = (0.0 if parent is None else ends[parent]) + delay
            reference_sends[request.trace_req_idx] = send
            ends[request.trace_req_idx] = send + service
        actual_sends = simulate_requests(timing_rows[::-1], service)
        assert actual_sends == reference_sends
    return {
        "checks": {
            name: "passed"
            for name in (
                "determinism",
                "position_uniqueness",
                "lcp_consistency",
                "context_budget",
                "gaps",
                "root_timeline_offsets",
                "causal_online_hints",
                "event_anchors",
                "dependency_dag",
                "fixed_service_simulation",
            )
        },
        "sha256_first": hashes[0],
        "sha256_second": hashes[1],
        "position_check": proof,
        "requests_checked": len(seen),
        "chain_adjacent_pairs_checked": adjacent_count,
        "same_root_random_pairs_checked": random_count,
        "random_pair_seed": 7121,
        "eligible_token_ids_checked": True,
        "chain_offsets_checked": len(expected_offsets),
        "event_anchors_checked": len(expected_starts),
        "simulation_service_times_s": [0.2, 0.5, 1.0],
        "simulation_send_times_checked": len(expected) * 3,
        "chain_offset_tolerance_s": 1e-8,
        "chain_offset_max_abs_error_s": max_offset_error,
        "gap_truncated_fraction": clipped / len(gaps),
        "elapsed_s": time.monotonic() - started,
    }
