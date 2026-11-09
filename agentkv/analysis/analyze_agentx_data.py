# SPDX-License-Identifier: Apache-2.0
"""Audit public AgentX structure, timing, and 32K scaling without replay."""

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

DATASET_SHA256 = "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
ROOT = Path(__file__).resolve().parents[2]


def snapshot_module(path, name, injected):
    """Load the pinned data-only helper without installing the harness."""
    tree = ast.parse(path.read_text(), filename=str(path))
    removed, retained = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith("aiperf."):
            removed.append(node.module)
        else:
            retained.append(node)
    tree.body = retained
    module = types.ModuleType(name)
    module.__dict__.update(injected)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module, removed


def finite(value):
    return value is not None and math.isfinite(value)


def describe(values):
    a = np.asarray([x for x in values if finite(x)], dtype=float)
    if not len(a):
        return dict(n=0, median=None, p90=None, maximum=None, minimum=None)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        p90=float(np.quantile(a, 0.9, method="linear")),
        maximum=float(a.max()),
        minimum=float(a.min()),
    )


def lcp(a, b):
    n = min(len(a), len(b))
    if not n:
        return 0
    neq = np.asarray(a[:n], dtype=np.int64) != np.asarray(b[:n], dtype=np.int64)
    i = int(neq.argmax())
    return i if neq[i] else n


def add_coverage(trace, coverage):
    coverage["trace_keys"].update(trace.keys())
    for entry in trace["requests"]:
        if entry["type"] == "subagent":
            coverage["marker_keys"].update(entry.keys())
            coverage["marker_status"][entry.get("status")] += 1
            coverage["marker_type"][entry.get("subagent_type")] += 1
            requests = entry["requests"]
        else:
            requests = [entry]
        for request in requests:
            coverage["request_keys"].update(request.keys())
            coverage["non_null_keys"].update(
                k for k, v in request.items() if v is not None
            )
            coverage["request_type"][request["type"]] += 1
            coverage["stop"][request.get("stop", "<missing>")] += 1
            coverage["input_type"].update(request.get("input_types", []))


def process_container(
    trace,
    items,
    stream_id,
    parent_stream_id,
    agent_id,
    marker,
    detector,
    rows,
    streams,
    gaps,
    raw_gaps,
    forks,
    quality,
):
    """Infer serial chains, preserving the original request positions."""
    if not items:
        quality["empty_containers"] += 1
        return
    indexed, positions = [], {}
    for ordinal, (outer_index, request) in enumerate(items):
        start = request.t
        if marker is not None and start + 1e-6 < marker.t:
            start = marker.t + start
            quality["child_relative_t_normalized"] += 1
        request = request.model_copy(update={"t": start})
        indexed.append((outer_index, request))
        positions[outer_index] = len(rows)
        rows.append(
            dict(
                session_id=trace.id,
                container_id=stream_id,
                parent_container_id=parent_stream_id,
                agent_id=agent_id,
                recorded_ordinal=ordinal,
                original_array_index=outer_index,
                model=request.model,
                type=request.type,
                start_s=start,
                api_time_s=request.api_time,
                end_s=start + request.api_time if finite(request.api_time) else None,
                think_time_s=request.think_time,
                ttft_s=getattr(request, "ttft", None),
                input_tokens=request.input_length,
                output_tokens=request.output_length,
                hash_blocks=len(request.hash_ids),
                stop=request.stop,
                chain_id=None,
                chain_turn=None,
                previous_lcp_tokens=None,
            )
        )
        quality["input_hash_count_mismatch"] += (
            request.input_length != len(request.hash_ids) * trace.block_size
        )
        quality["empty_hash_requests"] += not bool(request.hash_ids)
        quality["negative_api_time"] += (
            finite(request.api_time) and request.api_time < 0
        )
    ordered = sorted(indexed, key=lambda pair: (pair[1].t, pair[0]))
    quality["containers_not_in_timestamp_order"] += indexed != ordered
    for (_, previous), (index, current) in zip(ordered, ordered[1:]):
        delta = (
            current.t - previous.t - previous.api_time
            if finite(previous.api_time)
            else None
        )
        raw_gaps.append(
            dict(
                session_id=trace.id,
                container_id=stream_id,
                next_array_index=index,
                idle_s=delta,
            )
        )
    detection = detector.detect_agent_chains(indexed)
    quality["detected_seams_merged"] += detection.seams_merged
    quality["unclassified_empty_hash"] += detection.unclassified_empty_hash
    live = [detection.main_index, *detection.worker_indices]
    observed = []
    for ci in live:
        chain = detection.chains[ci]
        chain_id = f"{stream_id}::inferred:{ci}"
        requests = chain.requests
        observed.extend(i for i, _ in requests)
        streams.append(
            dict(
                session_id=trace.id,
                container_id=stream_id,
                chain_id=chain_id,
                inferred=True,
                is_container_main=ci == detection.main_index,
                request_count=len(requests),
                first_start_s=requests[0][1].t,
                last_end_s=max(r.t + max(r.api_time or 0, 0) for _, r in requests),
            )
        )
        if chain.fork is not None:
            forks.append(
                dict(
                    session_id=trace.id,
                    container_id=stream_id,
                    chain_id=chain_id,
                    parent_chain=None
                    if chain.fork.parent_chain is None
                    else f"{stream_id}::inferred:{chain.fork.parent_chain}",
                    parent_request_array_index=chain.fork.fork_outer_idx,
                    prefix_tokens=chain.fork.depth * trace.block_size,
                    start_s=chain.fork.fork_time,
                )
            )
        for turn, (index, request) in enumerate(requests):
            record = rows[positions[index]]
            record["chain_id"], record["chain_turn"] = chain_id, turn
            if not turn:
                continue
            previous_index, previous = requests[turn - 1]
            record["previous_lcp_tokens"] = (
                lcp(previous.hash_ids, request.hash_ids) * trace.block_size
            )
            duration = previous.api_time
            idle = request.t - previous.t - duration if finite(duration) else None
            confirmed = (
                previous.stop == "tool_use" or "tool_result" in request.input_types
            )
            gaps.append(
                dict(
                    session_id=trace.id,
                    chain_id=chain_id,
                    previous_array_index=previous_index,
                    next_array_index=index,
                    idle_s=idle,
                    previous_api_time_s=duration,
                    start_to_start_s=request.t - previous.t,
                    next_think_time_s=request.think_time,
                    recorded_tool_signal=confirmed,
                    nonnegative=finite(idle) and idle >= -1e-6,
                )
            )
    assert sorted(observed) == sorted(i for i, _ in indexed), stream_id


def write_table(path, records):
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path.with_suffix(".parquet"), compression="zstd")
    with path.with_suffix(".csv").open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=table.column_names)
        writer.writeheader()
        writer.writerows(records)
    assert pq.read_table(path.with_suffix(".parquet")).num_rows == len(records)


def ecdf(ax, values, label):
    a = np.sort(np.asarray([x for x in values if finite(x)], dtype=float))
    if len(a):
        ax.plot(a, np.arange(1, len(a) + 1) / len(a), label=f"{label} (n={len(a):,})")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.25)


def make_figures(figures, sessions, requests, gaps):
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    products = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            p = figures / f"m1_agentx_{name}.{ext}"
            fig.savefig(p, dpi=180, bbox_inches="tight")
            products.append(str(p))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [len(sessions), sum(s["subagent_groups"] for s in sessions), len(requests)]
    bars = ax.bar(
        ["Root sessions", "Explicit subagents", "Model requests"],
        counts,
        color=["#2878b5", "#38a78e", "#e89a37"],
    )
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=5)
    ax.set_ylim(100, max(counts) * 2)
    ax.set_title("Full public AgentX corpus: count audit")
    save(fig, "corpus_counts")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ecdf(ax, [s["total_requests"] for s in sessions], "All requests / root")
    ecdf(ax, [s["main_requests"] for s in sessions], "Top-level requests / root")
    ax.set_xscale("log")
    ax.set_xlabel("Requests per root session")
    ax.set_title("A root includes all explicitly nested requests")
    ax.legend()
    save(fig, "turns_per_session")

    for field, title in [
        ("input_tokens", "Recorded input block-count lengths"),
        ("output_tokens", "Recorded output lengths"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ecdf(ax, [r[field] for r in requests], "Original")
        ecdf(ax, [r["scaled_" + field] for r in requests], "32K global scaling")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Tokens (symlog, includes zero)")
        ax.set_title(title)
        ax.legend()
        save(fig, field)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ecdf(
        ax,
        [max(g["idle_s"], 0) for g in gaps if g["nonnegative"]],
        "Inferred-chain end-to-start gap",
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Seconds (symlog)")
    ax.set_title("Observed idle gaps; not individually identified tool runtimes")
    ax.legend()
    save(fig, "idle_gaps")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    count = Counter(s["subagent_groups"] for s in sessions)
    ax.bar(list(count), list(count.values()), color="#38a78e")
    ax.set_xlabel("Explicit subagent markers per root session")
    ax.set_ylabel("Root sessions")
    ax.set_title("Explicit topology, before inferred chain splitting")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "subagents_per_session")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ecdf(
        ax,
        [
            r["previous_lcp_tokens"] / r["input_tokens"]
            for r in requests
            if r["previous_lcp_tokens"] is not None and r["input_tokens"]
        ],
        "Original previous-chain input LCP",
    )
    ecdf(
        ax,
        [
            r["scaled_gpu_lcp_tokens"] / r["scaled_input_tokens"]
            for r in requests
            if r["previous_lcp_tokens"] is not None and r["scaled_input_tokens"]
        ],
        "Scaled LCP rounded to 16-token GPU blocks",
    )
    ax.set_xlabel("Shared input prefix / current input length")
    ax.set_title("Structural prefix sharing; not measured GPU residency")
    ax.legend()
    save(fig, "prefix_fraction")
    return products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/agentx/cc-traces-weka-062126/traces.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/m1-agentx-data-20261102"
    )
    parser.add_argument("--figures", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--reserve", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source"
    models, model_imports = snapshot_module(
        source / "weka_trace_models.py",
        "agentx_audit_models",
        {"AIPerfBaseModel": BaseModel},
    )
    detector, detector_imports = snapshot_module(
        source / "weka_agent_chains.py",
        "agentx_audit_chains",
        {
            "WekaNormalRequest": models.WekaNormalRequest,
            "WekaStreamingRequest": models.WekaStreamingRequest,
        },
    )
    coverage = {
        name: Counter()
        for name in [
            "trace_keys",
            "marker_keys",
            "marker_status",
            "marker_type",
            "request_keys",
            "non_null_keys",
            "request_type",
            "stop",
            "input_type",
        ]
    }
    requests, sessions, chains, gaps, raw_gaps, forks, markers = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    quality, digest, seen_ids, examples = Counter(), hashlib.sha256(), set(), []
    with args.input.open("rb") as stream:
        for row_number, line in enumerate(stream, 1):
            digest.update(line)
            raw = json.loads(line)
            add_coverage(raw, coverage)
            trace = models.WekaTrace.model_validate(raw)
            del raw
            assert trace.id not in seen_ids, trace.id
            seen_ids.add(trace.id)
            assert trace.hash_id_scope == "local" and trace.block_size == 64
            n_req_before, n_chains_before = len(requests), len(chains)
            root_stream = f"{trace.id}::root"
            top = [(i, r) for i, r in enumerate(trace.requests) if r.type != "subagent"]
            groups = [
                (i, r) for i, r in enumerate(trace.requests) if r.type == "subagent"
            ]
            process_container(
                trace,
                top,
                root_stream,
                None,
                None,
                None,
                detector,
                requests,
                chains,
                gaps,
                raw_gaps,
                forks,
                quality,
            )
            for i, marker in groups:
                child_id = f"{trace.id}::subagent:{marker.agent_id}:entry:{i}"
                markers.append(
                    dict(
                        session_id=trace.id,
                        container_id=child_id,
                        parent_container_id=root_stream,
                        agent_id=marker.agent_id,
                        marker_array_index=i,
                        spawn_s=marker.t,
                        duration_ms=marker.duration_ms,
                        status=marker.status,
                        declared_type=marker.subagent_type,
                        request_count=len(marker.requests),
                    )
                )
                process_container(
                    trace,
                    list(enumerate(marker.requests)),
                    child_id,
                    root_stream,
                    marker.agent_id,
                    marker,
                    detector,
                    requests,
                    chains,
                    gaps,
                    raw_gaps,
                    forks,
                    quality,
                )
            current = requests[n_req_before:]
            sessions.append(
                dict(
                    session_id=trace.id,
                    main_requests=len(top),
                    subagent_groups=len(groups),
                    subagent_requests=sum(len(x.requests) for _, x in groups),
                    total_requests=len(current),
                    inferred_chains=len(chains) - n_chains_before,
                    max_input_tokens=max(r["input_tokens"] for r in current),
                    max_context_tokens=max(
                        r["input_tokens"] + r["output_tokens"] for r in current
                    ),
                    earliest_start_s=min(r["start_s"] for r in current),
                    latest_end_s=max(r["end_s"] for r in current if finite(r["end_s"])),
                )
            )
            if len(examples) < 2 and groups:
                first = top[0][1]
                examples.append(
                    dict(
                        session_id=trace.id,
                        first_request=dict(
                            t=first.t,
                            type=first.type,
                            input_tokens=first.input_length,
                            output_tokens=first.output_length,
                            hash_ids_head=first.hash_ids[:8],
                            hash_blocks=len(first.hash_ids),
                            api_time=first.api_time,
                        ),
                        subagent=dict(
                            agent_id=groups[0][1].agent_id,
                            spawn_t=groups[0][1].t,
                            duration_ms=groups[0][1].duration_ms,
                            request_count=len(groups[0][1].requests),
                        ),
                    )
                )
            if row_number % 25 == 0:
                print(
                    f"ANALYZED {row_number} roots, {len(requests)} requests", flush=True
                )
    assert digest.hexdigest() == DATASET_SHA256
    assert len(sessions) == 393 and len(requests) == 98827
    assert sum(s["main_requests"] for s in sessions) == 56798
    assert sum(s["subagent_requests"] for s in sessions) == 42029
    assert len(markers) == 1697
    assert sum(r["input_tokens"] for r in requests) == 21635381376
    assert sum(r["output_tokens"] for r in requests) == 106474498
    peak = max(r["input_tokens"] + r["output_tokens"] for r in requests)
    alpha = min(1.0, (args.context_limit - args.reserve) / peak)
    for r in requests:
        r["scaled_input_tokens"] = math.floor(r["input_tokens"] * alpha)
        r["scaled_output_tokens"] = math.ceil(r["output_tokens"] * alpha)
        prefix = r["previous_lcp_tokens"]
        r["scaled_prefix_tokens"] = (
            math.floor(prefix * alpha) if prefix is not None else None
        )
        r["scaled_gpu_lcp_tokens"] = (
            r["scaled_prefix_tokens"] // 16 * 16 if prefix is not None else None
        )
        assert (
            r["scaled_input_tokens"] + r["scaled_output_tokens"]
            <= args.context_limit - args.reserve
        )
        assert r["input_tokens"] == 0 or r["scaled_input_tokens"] > 0
        assert r["output_tokens"] == 0 or r["scaled_output_tokens"] > 0
    for fork in forks:
        fork["scaled_prefix_tokens"] = math.floor(fork["prefix_tokens"] * alpha)
        fork["scaled_gpu_lcp_tokens"] = fork["scaled_prefix_tokens"] // 16 * 16
    valid = [g["idle_s"] for g in gaps if g["nonnegative"]]
    confirmed = [
        g["idle_s"] for g in gaps if g["nonnegative"] and g["recorded_tool_signal"]
    ]
    shared = [r for r in requests if (r["previous_lcp_tokens"] or 0) > 0]
    shared_forks = [f for f in forks if f["prefix_tokens"] > 0]
    previous_by_chain = {}
    prefix_edits = append_compatible = 0
    for r in sorted(requests, key=lambda r: (r["chain_id"], r["chain_turn"])):
        previous = previous_by_chain.get(r["chain_id"])
        if previous is not None:
            if r["previous_lcp_tokens"] < previous["input_tokens"]:
                prefix_edits += 1
            else:
                append_compatible += 1
        previous_by_chain[r["chain_id"]] = r
    assert prefix_edits + append_compatible == len(gaps)
    session_peaks = {}
    for r in requests:
        sid = r["session_id"]
        session_peaks[sid] = max(
            session_peaks.get(sid, 0), r["input_tokens"] + r["output_tokens"]
        )
    root_factors = {
        sid: min(1.0, (args.context_limit - args.reserve) / p)
        for sid, p in session_peaks.items()
    }
    for r in requests:
        factor = root_factors[r["session_id"]]
        r["per_root_scaled_input_tokens"] = math.floor(r["input_tokens"] * factor)
        r["per_root_scaled_output_tokens"] = math.ceil(r["output_tokens"] * factor)
        assert (
            r["per_root_scaled_input_tokens"] + r["per_root_scaled_output_tokens"]
            <= args.context_limit - args.reserve
        )
    for session in sessions:
        session["per_root_scaling_factor"] = root_factors[session["session_id"]]
    think_differences = [
        abs(g["idle_s"] - g["next_think_time_s"])
        for g in gaps
        if finite(g["idle_s"]) and finite(g["next_think_time_s"])
    ]
    summary = dict(
        dataset_sha256=digest.hexdigest(),
        dataset_bytes=args.input.stat().st_size,
        root_sessions=len(sessions),
        main_requests=sum(s["main_requests"] for s in sessions),
        subagent_groups=len(markers),
        subagent_inner_requests=sum(s["subagent_requests"] for s in sessions),
        model_requests=len(requests),
        explicit_containers=len(sessions) + len(markers),
        inferred_chains=len(chains),
        inferred_forks=len(forks),
        sessions_with_subagents=sum(s["subagent_groups"] > 0 for s in sessions),
        chain_continuations=dict(
            previous_input_is_prefix=append_compatible,
            previous_input_is_not_prefix=prefix_edits,
            tool_output_inclusion_not_testable=True,
        ),
        recorded_think_time_comparison=dict(
            paired=len(think_differences),
            equal_within_1ms=sum(x <= 0.001 for x in think_differences),
            absolute_difference_s=describe(think_differences),
        ),
        per_root_scaling=dict(
            factor_distribution=describe(root_factors.values()),
            input_tokens=describe(r["per_root_scaled_input_tokens"] for r in requests),
            output_tokens=describe(
                r["per_root_scaled_output_tokens"] for r in requests
            ),
            changes_cross_root_length_ratios=True,
        ),
        coverage=coverage,
        quality=dict(quality),
        statistics={
            "requests_per_root": describe(s["total_requests"] for s in sessions),
            "top_level_requests_per_root": describe(
                s["main_requests"] for s in sessions
            ),
            "requests_per_inferred_chain": describe(s["request_count"] for s in chains),
            "subagents_per_root": describe(s["subagent_groups"] for s in sessions),
            "input_tokens": describe(r["input_tokens"] for r in requests),
            "output_tokens": describe(r["output_tokens"] for r in requests),
            "context_tokens": describe(
                r["input_tokens"] + r["output_tokens"] for r in requests
            ),
            "api_time_s": describe(r["api_time_s"] for r in requests),
            "inferred_chain_idle_s": describe(valid),
            "raw_container_adjacent_idle_s": describe(g["idle_s"] for g in raw_gaps),
            "recorded_think_time_s": describe(r["think_time_s"] for r in requests),
            "confirmed_tool_idle_s": describe(confirmed),
            "previous_chain_lcp_tokens": describe(
                r["previous_lcp_tokens"] for r in requests
            ),
            "scaled_input_tokens": describe(r["scaled_input_tokens"] for r in requests),
            "scaled_output_tokens": describe(
                r["scaled_output_tokens"] for r in requests
            ),
        },
        gaps=dict(
            total_inferred_pairs=len(gaps),
            valid_nonnegative=len(valid),
            negative=sum(finite(g["idle_s"]) and g["idle_s"] < -1e-6 for g in gaps),
            missing_duration=sum(g["idle_s"] is None for g in gaps),
            raw_container_pairs=len(raw_gaps),
            raw_negative=sum(
                finite(g["idle_s"]) and g["idle_s"] < -1e-6 for g in raw_gaps
            ),
            confirmed_tool_pairs=len(confirmed),
        ),
        scaling=dict(
            context_limit=args.context_limit,
            reserve=args.reserve,
            alpha=alpha,
            original_peak_context=peak,
            scaled_peak_context=max(
                r["scaled_input_tokens"] + r["scaled_output_tokens"] for r in requests
            ),
            source_hash_block_tokens=64,
            minimum_scaled_block_width=math.floor(64 * alpha),
            target_kv_block_tokens=16,
            logical_turns_and_branches_dropped=0,
            positive_shared_pairs=len(shared),
            positive_shared_pairs_lost_at_gpu_block_boundary=sum(
                r["scaled_gpu_lcp_tokens"] == 0 for r in shared
            ),
            positive_shared_forks=len(shared_forks),
            positive_shared_forks_lost_at_gpu_block_boundary=sum(
                f["scaled_gpu_lcp_tokens"] == 0 for f in shared_forks
            ),
        ),
        upstream_helper_adaptation=dict(
            models_removed_imports=model_imports,
            detector_removed_imports=detector_imports,
            source_semantics=(
                "Unmodified helper bodies; replace only AIPerf base-class/type "
                "imports with injected data types."
            ),
        ),
    )
    for name, records in [
        ("requests", requests),
        ("sessions", sessions),
        ("inferred_chains", chains),
        ("idle_gaps", gaps),
        ("raw_container_gaps", raw_gaps),
        ("inferred_forks", forks),
        ("subagents", markers),
    ]:
        write_table(args.output / name, records)
    summary["figures"] = make_figures(args.figures, sessions, requests, gaps)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "schema_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "root_sessions",
                    "model_requests",
                    "inferred_chains",
                    "gaps",
                    "scaling",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
