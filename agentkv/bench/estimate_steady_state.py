"""Reproduce the 40-minute steady-state initialization experiment without GPU."""

import argparse
import json
import math
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .estimate_replay import COLUMNS, SCHEDULE_SERVICE_TIME_S
from .steady_state import RootSchedule, sample_population, simulate_entries
from .trace_loader import sha256_file
from agentkv.provenance import git_commit
from .workload import ROOT, describe


def interval_counts(intervals, times):
    """Counts for half-open [start, end) active lifetimes, including idle gaps."""
    starts = np.sort([start for start, end in intervals if end > start])
    ends = np.sort([end for start, end in intervals if end > start])
    return np.searchsorted(starts, times, side="right") - np.searchsorted(ends, times, side="right")


def peak_inflight(records):
    changes = defaultdict(int)
    for row in records:
        changes[row["send_s"]] += 1
        changes[row["send_s"] + SCHEDULE_SERVICE_TIME_S] -= 1
    current = peak = 0
    for time in sorted(changes):
        current += changes[time]
        peak = max(peak, current)
    return peak


def cdf_distance(values, reference):
    if not len(values):
        return None
    grid = np.unique(np.concatenate([values, reference]))
    left = np.searchsorted(np.sort(values), grid, side="right") / len(values)
    right = np.searchsorted(np.sort(reference), grid, side="right") / len(reference)
    return float(np.abs(left - right).max())


def run_population(roots, rate, seed, initialize, duration=2400.0, warmup=600.0, cooldown=0.0):
    durations = {rid: root.duration_s for rid, root in roots.items()}
    instances = sample_population(durations, rate, seed, duration, initialize)
    root_intervals, chain_intervals, records = [], [], []
    long_threshold = float(np.quantile(list(durations.values()), 0.9, method="linear"))
    for instance in instances:
        root = roots[instance.root_id]
        root_intervals.append((instance.start_s, instance.start_s + root.duration_s - instance.tau_s))
        for start, end in root.chain_intervals:
            if end > instance.tau_s:
                chain_intervals.append((instance.start_s + max(0.0, start - instance.tau_s),
                                        instance.start_s + end - instance.tau_s))
        entries = root.entries(instance.tau_s)
        sends = simulate_entries(entries)
        for entry in entries:
            request = entry.request
            idx = request["trace_req_idx"]
            expected = root.sends[idx] - instance.tau_s
            if not math.isclose(sends[idx], expected, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Truncated replay disagrees with estimated schedule")
            send = instance.start_s + sends[idx]
            if send > duration:
                continue
            records.append(dict(
                root_id=instance.root_id, instance_id=instance.instance_id,
                chain_id=request["chain_id"], trace_req_idx=idx,
                send_s=send, end_s=send + SCHEDULE_SERVICE_TIME_S,
                is_entry=entry.is_entry, prompt_len=request["prompt_len"],
                in_window=warmup <= send <= duration - cooldown,
                root_progress=root.sends[idx] / root.duration_s,
                long_root=root.duration_s >= long_threshold,
            ))
    records.sort(key=lambda r: (r["send_s"], r["root_id"], r["instance_id"], r["trace_req_idx"]))
    window = [r for r in records if r["in_window"]]
    measured = [r for r in window if not r["is_entry"]]
    prompts = np.asarray([r["prompt_len"] for r in measured], dtype=np.int32)
    reference = np.asarray([r["prompt_len"] for root in roots.values() for r in root.requests], dtype=np.int32)
    times = np.arange(warmup, duration - cooldown + 1e-9, 10.0)
    root_counts = interval_counts(root_intervals, times)
    chain_counts = interval_counts(chain_intervals, times)
    entries = sum(r["is_entry"] for r in window)
    late = [r for r in measured if r["long_root"] and r["root_progress"] >= 0.75]
    initial = [instance for instance in instances if instance.initial]
    summary = {
        "arrival_rate_per_minute": rate,
        "steady_state_initialization": initialize,
        "M0": len(initial),
        "expected_M0": rate / 60 * float(np.mean(list(durations.values()))) if initialize else 0.0,
        "new_instances": len(instances) - len(initial),
        "total_requests": len(records),
        "total_is_entry_requests": sum(r["is_entry"] for r in records),
        "last_is_entry_send_s": max((r["send_s"] for r in records if r["is_entry"]), default=None),
        "active_roots": describe(root_counts),
        "active_chains": describe(chain_counts),
        "window_requests": len(window),
        "window_is_entry_requests": entries,
        "window_is_entry_fraction": entries / len(window) if window else 0.0,
        "window_non_entry_requests": len(measured),
        "prompt_len_non_entry": describe(prompts),
        "prompt_cdf_max_distance_from_all_requests": cdf_distance(prompts, reference),
        "peak_inflight_requests": peak_inflight(records),
        "late_requests_in_long_roots": len(late),
        "long_root_threshold_s": long_threshold,
        "request_progress": describe([r["root_progress"] for r in measured]),
    }
    samples = [dict(time_s=float(t), active_roots=int(r), active_chains=int(c))
               for t, r, c in zip(times, root_counts, chain_counts)]
    return summary, instances, records, samples, prompts


def plot_cdfs(results, reference, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharex=True, sharey=True)
    for ax, rate in zip(axes.flat, (0.5, 1.0, 2.0, 4.0)):
        for values, label, color, style in (
            (reference, "All workload requests", "black", ":"),
            (results[rate, True], "Steady initialization", "#0072B2", "-"),
            (results[rate, False], "Empty initialization", "#D55E00", "--"),
        ):
            if len(values):
                x, counts = np.unique(values, return_counts=True)
                ax.step(np.r_[0, x], np.r_[0, np.cumsum(counts) / len(values)],
                        where="post", label=label, color=color, linestyle=style)
        ax.set_title(f"Arrival rate = {rate:g} roots/min")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 32512)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("Request CDF")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "prompt_cdf.png", dpi=300)
    fig.savefig(output / "prompt_cdf.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=ROOT / "results/m1-workload-seed0-v3/workload.parquet")
    parser.add_argument("--schedule", type=Path, default=ROOT / "results/m1-steady-state-seed0/schedule_estimate.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/m1-steady-state-seed0/population-seed0")
    parser.add_argument("--population-seed", type=int, default=0)
    parser.add_argument("--allow-dirty",action="store_true",help="DEBUG ONLY: bypass clean-worktree check")
    args = parser.parse_args(argv)
    commit = git_commit(args.allow_dirty)
    digest = sha256_file(args.workload)
    manifest = json.loads(args.workload.with_name("manifest.json").read_text())
    schedule = pq.read_table(args.schedule, use_threads=False)
    if (digest != manifest["workload"]["sha256"]
            or schedule.schema.metadata[b"agentkv.workload_sha256"].decode() != digest
            or schedule.schema.metadata[b"agentkv.service_time_s"] != b"0.5"):
        raise ValueError("Schedule/workload provenance mismatch")
    by_root, estimates = defaultdict(list), defaultdict(list)
    for row in pq.read_table(args.workload, columns=COLUMNS + ["prompt_len"], use_threads=False).to_pylist():
        by_root[row["root_id"]].append(row)
    for row in schedule.to_pylist():
        estimates[row["root_id"]].append(row)
    if set(by_root) != set(estimates):
        raise ValueError("Schedule roots differ from workload")
    roots = {rid: RootSchedule(rows, estimates[rid]) for rid, rows in sorted(by_root.items())}
    reference = np.asarray([r["prompt_len"] for root in roots.values() for r in root.requests])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "population_seed": args.population_seed,
        "duration_s": 2400, "warmup_s": 600, "cooldown_s": 0,
        "service_time_s": SCHEDULE_SERVICE_TIME_S,
        "sample_interval_s": 10,
        "mean_root_duration_s": float(np.mean([r.duration_s for r in roots.values()])),
        "baseline_prompt_len": describe(reference),
        "assumptions": {
            "concurrency": "unlimited, no queueing or network overhead; no KV prefill of skipped history",
            "active_root": "instance start <= time < start + D_R - tau, including gaps and virtual completion of pre-tau requests",
            "active_chain": "first estimated send <= progress < final estimated end, including gaps; clip start at instance start",
            "statistics": "inclusive send window [600,2400]; prompt and progress distributions exclude is_entry",
            "inflight": "actual replayed requests only, [send,send+0.5); excludes skipped pre-tau requests",
            "streams": "SeedSequence([population_seed, IEEE754 rate bits]).spawn(5): initial count/root/progress, arrivals, new roots; PCG64",
            "instance_ids": "initial odd IDs, new-arrival even IDs; paired modes share identical new instances",
            "late_long_root": "original estimated send / D_R >=0.75 and D_R >= corpus root-duration P90",
            "cdf_distance": "descriptive maximum empirical CDF distance, not a significance test",
        },
        "source": {"workload_sha256": digest, "schedule_sha256": sha256_file(args.schedule)},
        "git_commit": commit,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__},
        "runs": [],
    }
    cdfs = {}
    for rate in (0.5, 1.0, 2.0, 4.0):
        paired_new = None
        for initialize in (True, False):
            summary, instances, records, samples, prompts = run_population(roots, rate, args.population_seed, initialize)
            new = [asdict(i) for i in instances if not i.initial]
            if paired_new is not None and paired_new != new:
                raise ValueError("Paired new-arrival populations differ")
            paired_new = new
            folder = args.output_dir / f"lambda-{rate:g}" / ("steady" if initialize else "empty")
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(records), folder / "requests.parquet", compression="zstd")
            (folder / "run.json").write_text(json.dumps({
                "parameters": {"population_seed": args.population_seed, "duration_s":2400,
                               "warmup_s":600, "cooldown_s":0, "service_time_s":0.5,
                               "arrival_rate_per_minute":rate, "steady_state_initialization":initialize},
                "source": report["source"], "git_commit": report["git_commit"],
                "environment": report["environment"],
                "M0": summary["M0"], "instances": [asdict(i) for i in instances],
                "assumptions": report["assumptions"], "summary": summary,
                "requests_sha256": sha256_file(folder / "requests.parquet"),
                "samples": samples,
            }, ensure_ascii=False, indent=2) + "\n")
            report["runs"].append(summary)
            cdfs[rate, initialize] = prompts
    plot_cdfs(cdfs, reference, args.output_dir)
    report["artifacts"] = {name: sha256_file(args.output_dir / name) for name in ("prompt_cdf.png", "prompt_cdf.pdf")}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"baseline_prompt_len": report["baseline_prompt_len"], "runs": report["runs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
