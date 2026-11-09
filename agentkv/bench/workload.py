"""Build deterministic offline workloads without importing or starting vLLM."""

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .subset import sample_roots
from .synth import Scale, Synthesizer, common_prefix, tokenizer_vocabulary
from .trace_loader import LoadedTrace, available_roots, load_trace, sha256_file

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "results/m1-agentx-data-20261102"
SCHEMA = pa.schema(
    [
        ("root_id", pa.string()),
        ("chain_id", pa.string()),
        ("chain_turn", pa.int32()),
        ("trace_req_idx", pa.string()),
        ("is_subagent", pa.bool_()),
        ("is_aux", pa.bool_()),
        ("chain_offset_s", pa.float64()),
        ("chain_offset_orig_s", pa.float64()),
        ("anchor_trace_req_idx", pa.string()),
        ("start_delay_s", pa.float64()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("prompt_len", pa.int32()),
        ("max_tokens", pa.int32()),
        ("out_zero_orig", pa.bool_()),
        ("lcp_prev_tokens", pa.int32()),
        ("gap_after_s", pa.float64()),
        ("gap_truncated", pa.bool_()),
        ("hint_perfect_ms", pa.float64()),
        ("hint_noisy_ms", pa.float64()),
        ("hint_online_ms", pa.float64()),
        ("hint_online_p90_ms", pa.float64()),
        ("final", pa.bool_()),
        ("input_len_orig", pa.int64()),
        ("output_len_orig", pa.int64()),
        ("t_orig_s", pa.float64()),
        ("api_time_s", pa.float64()),
        ("gap_after_orig_s", pa.float64()),
    ],
    metadata={
        b"agentkv.format": b"m1-offline-v1",
        b"agentkv.scheduling": b"event-anchored-v1",
    },
)


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    num_roots: int = 100
    context_budget: int = 32512
    gap_cap: float = 120.0
    gap_compress: float = 1.0
    noise_seed: int = 0
    ema_alpha: float = 0.5

    def validate(self):
        if self.seed < 0 or self.noise_seed < 0 or self.num_roots < 1:
            raise ValueError("Seeds must be nonnegative and num_roots positive")
        if self.context_budget < 2 or not 0 < self.ema_alpha <= 1:
            raise ValueError("Require context_budget >= 2 and 0 < ema_alpha <= 1")
        if any(
            not math.isfinite(x) or x < 0 for x in (self.gap_cap, self.gap_compress)
        ):
            raise ValueError("Gap cap and compression must be finite and nonnegative")


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "p90": None, "maximum": None}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "maximum": float(values.max()),
    }


def chain_groups(loaded: LoadedTrace):
    for rid in sorted(loaded.roots):
        ordered = sorted(loaded.roots[rid], key=lambda r: (r.chain_id, r.chain_turn))
        for cid, group in itertools.groupby(ordered, key=lambda r: r.chain_id):
            yield rid, cid, list(group)


def gap_plan(loaded: LoadedTrace, config: BuildConfig):
    plan = {}
    for _, cid, requests in chain_groups(loaded):
        gaps = []
        for previous, current in zip(requests, requests[1:]):
            raw = current.t - (previous.t + previous.api_time)
            if not math.isfinite(raw) or raw < 0:
                raise ValueError(f"Invalid serial-chain idle gap: {cid}: {raw}")
            gap = min(raw, config.gap_cap) * config.gap_compress
            if not math.isfinite(gap):
                raise ValueError("Compressed gap overflow")
            gaps.append((raw, gap, raw > config.gap_cap))
        plan[cid] = gaps
    return plan


@dataclass(frozen=True)
class GapPrior:
    median_s: float
    interval_count: int
    root_ids: tuple[str, ...]


def online_gap_prior(
    requests_path: Path, selected_roots: list[str], config: BuildConfig
) -> GapPrior:
    """Calibrate only on held-out roots, using the pinned audit's serial chains."""
    config.validate()
    selected = set(selected_roots)
    columns = ["session_id", "chain_id", "chain_turn", "start_s", "api_time_s"]
    held_out = [
        row
        for row in pq.read_table(requests_path, columns=columns).to_pylist()
        if row["session_id"] not in selected
    ]
    values = []
    ordered = sorted(held_out, key=lambda r: (r["chain_id"], r["chain_turn"]))
    for cid, group in itertools.groupby(ordered, key=lambda r: r["chain_id"]):
        chain = list(group)
        if [r["chain_turn"] for r in chain] != list(range(len(chain))):
            raise ValueError(f"Non-contiguous prior chain: {cid}")
        for previous, current in zip(chain, chain[1:]):
            raw = current["start_s"] - (previous["start_s"] + previous["api_time_s"])
            if not math.isfinite(raw) or raw < 0:
                raise ValueError(f"Invalid prior idle gap: {cid}: {raw}")
            gap = min(raw, config.gap_cap) * config.gap_compress
            if not math.isfinite(gap):
                raise ValueError("Compressed prior gap overflow")
            values.append(gap)
    if not values:
        raise ValueError("Online prior requires at least one held-out chain interval")
    return GapPrior(
        float(np.median(values)),
        len(values),
        tuple(sorted({r["session_id"] for r in held_out})),
    )


def root_timeline(requests, config: BuildConfig):
    """Map request starts and the final end, capping only idle gaps in the union.

    The origin stays at zero; any delay before the first active interval stays
    intact. Overlapping/touching intervals share a translation before applying c.
    """
    starts = {}
    merged_end = None
    removed = 0.0
    for request in sorted(requests, key=lambda r: (r.t, r.api_time)):
        start, end = request.t, request.t + request.api_time
        if not all(math.isfinite(x) and x >= 0 for x in (start, end, request.api_time)):
            raise ValueError("Invalid root timeline timing")
        if merged_end is not None:
            removed += max(0.0, start - merged_end - config.gap_cap)
        merged_end = end if merged_end is None else max(merged_end, end)
        starts[request.trace_req_idx] = (start - removed) * config.gap_compress
    duration = (
        0.0 if merged_end is None else (merged_end - removed) * config.gap_compress
    )
    if not math.isfinite(duration) or any(
        not math.isfinite(x) for x in starts.values()
    ):
        raise ValueError("Compressed root timeline overflow")
    return starts, duration


def event_order(request):
    """Extend trace time to a strict order, including zero-duration requests."""
    return (request.t, request.chain_turn, request.chain_id, request.trace_req_idx)


def chain_start_plan(loaded: LoadedTrace, config: BuildConfig):
    """Choose the latest completed external request for each chain head."""
    plan = {}
    for requests in loaded.roots.values():
        completed = sorted(
            requests, key=lambda r: (r.t + r.api_time, event_order(r))
        )
        end_times = [r.t + r.api_time for r in completed]
        for head in (r for r in requests if r.chain_turn == 0):
            anchor = None
            index = bisect_right(end_times, head.t) - 1
            while index >= 0:
                candidate = completed[index]
                if (
                    candidate.chain_id != head.chain_id
                    and event_order(candidate) < event_order(head)
                ):
                    anchor = candidate
                    break
                index -= 1
            raw = (
                head.t if anchor is None else head.t - (anchor.t + anchor.api_time)
            )
            delay = min(raw, config.gap_cap) * config.gap_compress
            if not math.isfinite(delay) or delay < 0:
                raise ValueError(f"Invalid chain start delay: {head.chain_id}")
            plan[head.chain_id] = (None if anchor is None else anchor.trace_req_idx, delay)
    return plan


def noise_factor(seed: int, root_id: str, chain_id: str, turn: int) -> float:
    key = json.dumps([seed, root_id, chain_id, turn], separators=(",", ":"))
    derived = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(derived))
    return float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))


def write_workload(
    loaded: LoadedTrace,
    vocabulary: np.ndarray,
    vocabulary_info: dict,
    config: BuildConfig,
    output: Path,
    prior: GapPrior,
) -> dict:
    config.validate()
    started = time.monotonic()
    if len(loaded.roots) != config.num_roots:
        raise ValueError("Loaded root count differs from requested subset")
    if set(prior.root_ids) & set(loaded.roots):
        raise ValueError("Online prior roots overlap the selected workload")
    if (
        prior.interval_count < 1
        or not prior.root_ids
        or not math.isfinite(prior.median_s)
        or prior.median_s < 0
    ):
        raise ValueError("Invalid held-out online prior")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "workload.parquet"
    partial = output / "workload.parquet.partial"
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Refusing to replace workload: {output}")
    plan = gap_plan(loaded, config)
    starts = chain_start_plan(loaded, config)
    timelines = {rid: root_timeline(rs, config) for rid, rs in loaded.roots.items()}
    scales = {
        rid: Scale.for_root(requests, config.context_budget)
        for rid, requests in loaded.roots.items()
    }
    rows, lengths, outputs, gaps, offsets, original_offsets = [], [], [], [], [], []
    reference_root_ends = {}
    total = aux_requests = aux_chains = child_requests = zero_outputs = truncated = 0
    root_synth = None
    current_root = None
    with pq.ParquetWriter(
        partial,
        SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        version="2.6",
        data_page_version="1.0",
        write_statistics=True,
    ) as writer:
        for rid, cid, requests in chain_groups(loaded):
            if current_root != rid:
                current_root = rid
                root_synth = Synthesizer(rid, scales[rid], vocabulary)
            is_aux = len(requests) == 1
            aux_chains += is_aux
            previous = previous_tokens = None
            history = []
            ema = None
            offset = timelines[rid][0][requests[0].trace_req_idx]
            if not math.isfinite(offset):
                raise ValueError("Compressed chain offset overflow")
            offsets.append(offset)
            original_offsets.append(requests[0].t)
            anchor, start_delay = starts[cid]
            projected_end = offset
            for i, request in enumerate(requests):
                projected_end += request.api_time
                if i < len(plan[cid]):
                    projected_end += plan[cid][i][1]
            reference_root_ends[rid] = max(
                reference_root_ends.get(rid, 0.0), projected_end
            )
            for i, request in enumerate(requests):
                if request.chain_turn != i:
                    raise ValueError(f"Non-contiguous chain: {cid}")
                scale = scales[rid]
                prompt = root_synth.prompt(request, previous, previous_tokens)
                max_tokens = max(1, scale.ceil(request.output_len))
                if len(prompt) + max_tokens > config.context_budget:
                    raise ValueError(
                        f"Output-zero promotion exceeds budget: {request.trace_req_idx}"
                    )
                final = i == len(requests) - 1
                raw = gap = perfect = noisy = online = p90 = None
                clipped = False
                if not final:
                    raw, gap, clipped = plan[cid][i]
                    perfect = gap * 1000
                    noisy = perfect * noise_factor(config.noise_seed, rid, cid, i)
                    online = (prior.median_s if ema is None else ema) * 1000
                    p90 = (
                        prior.median_s
                        if not history
                        else float(np.quantile(history, 0.9, method="linear"))
                    ) * 1000
                    gaps.append(gap)
                    truncated += clipped
                row = dict(
                    root_id=rid,
                    chain_id=cid,
                    chain_turn=i,
                    trace_req_idx=request.trace_req_idx,
                    is_subagent=request.is_subagent,
                    is_aux=is_aux,
                    chain_offset_s=offset if i == 0 else None,
                    chain_offset_orig_s=request.t if i == 0 else None,
                    anchor_trace_req_idx=anchor if i == 0 else None,
                    start_delay_s=start_delay if i == 0 else None,
                    prompt_token_ids=prompt,
                    prompt_len=len(prompt),
                    max_tokens=max_tokens,
                    out_zero_orig=request.output_len == 0,
                    lcp_prev_tokens=0
                    if previous is None
                    else scale.q(common_prefix(previous.hash_ids, request.hash_ids)),
                    gap_after_s=gap,
                    gap_truncated=clipped,
                    hint_perfect_ms=perfect,
                    hint_noisy_ms=noisy,
                    hint_online_ms=online,
                    hint_online_p90_ms=p90,
                    final=final,
                    input_len_orig=request.input_len,
                    output_len_orig=request.output_len,
                    t_orig_s=request.t,
                    api_time_s=request.api_time,
                    gap_after_orig_s=raw,
                )
                rows.append(row)
                total += 1
                child_requests += request.is_subagent
                aux_requests += is_aux
                zero_outputs += request.output_len == 0
                lengths.append(len(prompt))
                outputs.append(max_tokens)
                if not final:
                    history.append(gap)
                    ema = (
                        gap
                        if ema is None
                        else (config.ema_alpha * gap + (1 - config.ema_alpha) * ema)
                    )
                previous, previous_tokens = request, prompt
                if len(rows) == 128:
                    writer.write_table(
                        pa.Table.from_pylist(rows, schema=SCHEMA), row_group_size=128
                    )
                    rows.clear()
        if rows:
            writer.write_table(
                pa.Table.from_pylist(rows, schema=SCHEMA), row_group_size=128
            )
    partial.replace(destination)
    stats = {
        "roots": len(loaded.roots),
        "chains": len(plan),
        "requests": total,
        "aux_chains": aux_chains,
        "aux_chain_fraction": aux_chains / len(plan),
        "aux_requests": aux_requests,
        "aux_request_fraction": aux_requests / total,
        "subagent_requests": child_requests,
        "subagent_request_fraction": child_requests / total,
        "prompt_len": describe(lengths),
        "max_tokens": describe(outputs),
        "gap_after_s": describe(gaps),
        "chain_offset_s": describe(offsets),
        "chain_offset_orig_s": describe(original_offsets),
        "root_timeline_duration_s": describe([t[1] for t in timelines.values()]),
        "root_offset_reference_duration_s": describe(
            list(reference_root_ends.values())
        ),
        "start_delay_s": describe([delay for _, delay in starts.values()]),
        "chains_with_request_anchor": sum(
            anchor is not None for anchor, _ in starts.values()
        ),
        "chains_anchored_at_root_arrival": sum(
            anchor is None for anchor, _ in starts.values()
        ),
        "gaps_truncated": truncated,
        "gap_truncated_fraction": truncated / len(gaps) if gaps else 0.0,
        "out_zero_orig_requests": zero_outputs,
    }
    manifest = {
        "format": "m1-offline-v1",
        "scheduling": "event-anchored-v1",
        "parameters": asdict(config),
        "selected_roots": sorted(loaded.roots),
        "sampling": (
            "PCG64(seed), choice without replacement from sorted root "
            "IDs; output sorted"
        ),
        "source": loaded.provenance,
        "tokenizer": vocabulary_info,
        "scales": {
            rid: {**asdict(scale), "alpha": scale.numerator / scale.denominator}
            for rid, scale in sorted(scales.items())
        },
        "conventions": {
            "seed_encoding": (
                "SHA256(UTF8(root_id + decimal(hash_id)))[:8], big-endian integer"
            ),
            "scaling": "exact rational floor/ceil; no floating-point boundary rounding",
            "is_aux": "every inferred chain containing exactly one request",
            "global_median_prior_s": prior.median_s,
            "online_prior_interval_count": prior.interval_count,
            "online_prior_root_count": len(prior.root_ids),
            "online_prior_root_ids": list(prior.root_ids),
            "online_prior_population": (
                "all serial-chain intervals in unselected roots of the pinned "
                "requests audit; min(raw_gap, gap_cap) * gap_compress"
            ),
            "online_ema": (
                "first historical gap seeds EMA; subsequent alpha=ema_alpha; "
                "excludes current/future gaps"
            ),
            "final": "gap and numeric hints null; gap_truncated=false; final=true",
            "none_hint_tier": "no envelope; hints are numeric columns only",
            "perfect_noisy_p90": (
                "online stage may derive expected_gap_ms * 2; not stored redundantly"
            ),
            "gap_truncation_denominator": (
                "non-final requests only, including zero gaps"
            ),
            "offset": (
                "reference only, not used for online scheduling; "
                "union root request [t, t + api_time] intervals; cap only gaps "
                "between merged intervals at gap_cap, then mapped first t * "
                "gap_compress; preserve delay before first interval"
            ),
            "offset_orig": "uncompressed first request t; null after chain head",
            "chain_start": (
                "latest original end <= chain head t in the same root, excluding "
                "own chain; wait for anchor completion, then "
                "min(head_t - anchor_end, gap_cap) * gap_compress; "
                "if no anchor, root arrival + min(head_t, gap_cap) * gap_compress; "
                "anchor and start_delay_s stored only on chain heads"
            ),
            "anchor_ties": (
                "maximize (end, t, chain_turn, chain_id, trace_req_idx); "
                "require (t, chain_turn, chain_id, trace_req_idx) strictly before "
                "the head, so simultaneous zero-duration requests cannot cycle"
            ),
            "root_durations": (
                "reference only, not event-anchored replay duration; "
                "timeline: mapped last original end * gap_compress; offset reference: "
                "maximum over chains of offset + sum(original api_time) + "
                "sum(transformed chain gaps); estimate event-anchored replay "
                "with python -m agentkv.bench.estimate_replay"
            ),
            "special_tokens": (
                "all_special_ids plus added tokens marked special, sampled "
                "only from actual get_vocab IDs"
            ),
            "zero_output": (
                "max_tokens=max(1, ceil(alpha*out)); out_zero_orig marks promotion"
            ),
            "lcp_first_turn": 0,
            "parquet": (
                "fixed schema/order, 128-row groups, ZSTD level 3, no "
                "dictionary; no timestamps"
            ),
        },
        "statistics": stats,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "workload": {
            "file": "workload.parquet",
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        },
        "construction_elapsed_s": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-roots", type=int, default=100)
    parser.add_argument("--context-budget", type=int, default=32512)
    parser.add_argument("--gap-cap", type=float, default=120.0)
    parser.add_argument("--gap-compress", type=float, default=1.0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    parser.add_argument(
        "--trace",
        type=Path,
        default=ROOT / "data/raw/agentx/cc-traces-weka-062126/traces.jsonl",
    )
    parser.add_argument("--requests", type=Path, default=AUDIT / "requests.parquet")
    parser.add_argument(
        "--chains", type=Path, default=AUDIT / "inferred_chains.parquet"
    )
    parser.add_argument(
        "--audit-provenance", type=Path, default=AUDIT / "provenance.json"
    )
    parser.add_argument(
        "--position-check",
        type=Path,
        default=ROOT / "results/m1-position-check/result.json",
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("/data/models/Qwen3-8B"))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/m1-workload-seed0-v3"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    config = BuildConfig(
        args.seed,
        args.num_roots,
        args.context_budget,
        args.gap_cap,
        args.gap_compress,
        args.noise_seed,
        args.ema_alpha,
    )
    config.validate()
    pinned = json.loads(args.audit_provenance.read_text())
    position = json.loads(args.position_check.read_text())
    if (
        not position["passed"]
        or position["violating_occurrences"] != 0
        or position["root_sessions"] != 393
        or position["source_sha256"] != pinned["raw_sha256"]
    ):
        raise ValueError("The full-corpus position-uniqueness gate has not passed")
    selected = sample_roots(available_roots(args.requests), args.seed, args.num_roots)
    print(f"Loading {len(selected)} roots and verifying the pinned trace", flush=True)
    loaded = load_trace(
        args.trace, args.requests, args.chains, selected, pinned["raw_sha256"]
    )
    loaded.provenance.update(
        {
            "trace_revision": pinned["dataset_revision"],
            "dataset": pinned["dataset"],
            "position_check_sha256": sha256_file(args.position_check),
            "position_check": {
                k: position[k]
                for k in (
                    "root_sessions",
                    "model_requests",
                    "block_occurrences",
                    "violating_occurrences",
                )
            },
        }
    )
    vocabulary, info = tokenizer_vocabulary(args.tokenizer)
    prior = online_gap_prior(args.requests, selected, config)
    print("Synthesizing and writing workload", flush=True)
    manifest = write_workload(loaded, vocabulary, info, config, args.output_dir, prior)
    manifest["generation_elapsed_s"] = time.monotonic() - started
    manifest["implementation_sha256"] = {
        name: sha256_file(Path(__file__).parent / name)
        for name in (
            "trace_loader.py", "subset.py", "synth.py", "workload.py",
            "estimate_replay.py",
        )
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "statistics": manifest["statistics"],
                "workload": manifest["workload"],
                "generation_elapsed_s": manifest["generation_elapsed_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
