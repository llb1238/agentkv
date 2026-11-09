"""Estimate event-anchored replay with constant service time and no queueing."""

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

from .trace_loader import sha256_file
from .workload import ROOT, describe

COLUMNS = [
    "root_id", "chain_id", "chain_turn", "trace_req_idx",
    "anchor_trace_req_idx", "start_delay_s", "gap_after_s",
]
SCHEDULE_SERVICE_TIME_S = 0.5
SCHEDULE_SCHEMA = pa.schema([
    ("root_id", pa.string()),
    ("chain_id", pa.string()),
    ("chain_turn", pa.int32()),
    ("trace_req_idx", pa.string()),
    ("est_send_s", pa.float64()),
    ("est_duration_s", pa.float64()),
])


def simulate_requests(rows: list[dict], service_time_s: float) -> dict[str, float]:
    """Return send times relative to each root's arrival, evaluating the full DAG.

    Each request depends either on its chain predecessor or on its head's anchor.
    Root arrivals all use time zero independently; siblings run concurrently.
    File/chain ordering and reference offsets have no effect on this simulation.
    """
    if not math.isfinite(service_time_s) or service_time_s <= 0:
        raise ValueError("Service time must be finite and positive")
    by_id = {row["trace_req_idx"]: row for row in rows}
    by_turn = {(r["root_id"], r["chain_id"], r["chain_turn"]): r for r in rows}
    if len(by_id) != len(rows) or len(by_turn) != len(rows):
        raise ValueError("Duplicate request identity or chain turn")
    children = defaultdict(list)
    pending, delays = {}, {}
    ready = deque()
    for row in rows:
        idx = row["trace_req_idx"]
        if row["chain_turn"] < 0:
            raise ValueError("Negative chain turn")
        if row["chain_turn"] == 0:
            parent = row["anchor_trace_req_idx"]
            delay = row["start_delay_s"]
            if parent is not None and parent in by_id:
                if by_id[parent]["chain_id"] == row["chain_id"]:
                    raise ValueError("Anchor must belong to another chain")
        else:
            if row["anchor_trace_req_idx"] is not None or row["start_delay_s"] is not None:
                raise ValueError("Only chain heads may carry start metadata")
            previous = by_turn.get((row["root_id"], row["chain_id"], row["chain_turn"] - 1))
            if previous is None:
                raise ValueError("Missing chain predecessor")
            parent, delay = previous["trace_req_idx"], previous["gap_after_s"]
        if delay is None or not math.isfinite(delay) or delay < 0:
            raise ValueError("Missing or invalid dependency delay")
        delays[idx] = delay
        if parent is None:
            pending[idx] = 0
            ready.append(idx)
        else:
            if parent not in by_id or by_id[parent]["root_id"] != row["root_id"]:
                raise ValueError("Missing or cross-root anchor")
            pending[idx] = 1
            children[parent].append(idx)
    sends = {idx: delays[idx] for idx in ready}
    while ready:
        idx = ready.popleft()
        end = sends[idx] + service_time_s
        if not math.isfinite(end):
            raise ValueError("Replay time overflow")
        for child in children[idx]:
            sends[child] = end + delays[child]
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if len(sends) != len(rows):
        raise ValueError("Cyclic replay dependencies")
    return sends


def estimate(rows: list[dict], service_time_s: float):
    sends = simulate_requests(rows, service_time_s)
    root_ends = defaultdict(float)
    for row in rows:
        root_ends[row["root_id"]] = max(
            root_ends[row["root_id"]], sends[row["trace_req_idx"]] + service_time_s
        )
    in_window = sum(t <= 1500.0 for t in sends.values())
    return {
        "service_time_s": service_time_s,
        "root_completion_s": describe(list(root_ends.values())),
        "requests_sent_within_1500s": in_window,
        "total_requests": len(rows),
        "request_fraction_within_1500s": in_window / len(rows) if rows else 0.0,
    }, dict(root_ends)


def write_schedule_estimate(
    rows: list[dict], destination: Path, workload_sha256: str
) -> None:
    """Write a small, independently joinable schedule at fixed s=0.5 seconds.

    est_duration_s is constant within each root, repeated on its request rows.
    All times are seconds relative to that root's arrival. Workload rows and
    their on-disk source are never modified.
    """
    sends = simulate_requests(rows, SCHEDULE_SERVICE_TIME_S)
    root_ends = defaultdict(float)
    for row in rows:
        root_ends[row["root_id"]] = max(
            root_ends[row["root_id"]],
            sends[row["trace_req_idx"]] + SCHEDULE_SERVICE_TIME_S,
        )
    schedule = [
        dict(
            root_id=row["root_id"],
            chain_id=row["chain_id"],
            chain_turn=row["chain_turn"],
            trace_req_idx=row["trace_req_idx"],
            est_send_s=sends[row["trace_req_idx"]],
            est_duration_s=root_ends[row["root_id"]],
        )
        for row in sorted(rows, key=lambda r: (
            r["root_id"], r["chain_id"], r["chain_turn"]
        ))
    ]
    schema = SCHEDULE_SCHEMA.with_metadata({
        b"agentkv.format": b"m1-schedule-estimate-v1",
        b"agentkv.service_time_s": b"0.5",
        b"agentkv.workload_sha256": workload_sha256.encode("ascii"),
        b"agentkv.estimator_sha256": sha256_file(Path(__file__)).encode("ascii"),
        b"agentkv.time_origin": b"seconds relative to each root arrival",
        b"agentkv.est_duration_s": b"per-root maximum completion, repeated on request rows",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(schedule, schema=schema), destination,
        compression="zstd", compression_level=3,
        use_dictionary=["root_id", "chain_id"], version="2.6",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path,
                        default=ROOT / "results/m1-workload-seed0-v3/workload.parquet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--service-times", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    args = parser.parse_args(argv)
    rows = pq.read_table(args.workload, columns=COLUMNS).to_pylist()
    manifest = json.loads(args.workload.with_name("manifest.json").read_text())
    digest = sha256_file(args.workload)
    if digest != manifest["workload"]["sha256"]:
        raise ValueError("Workload SHA256 differs from manifest")
    report = {
        "workload_sha256": digest,
        "parameters": manifest["parameters"],
        "assumptions": {
            "service": "every request takes exactly s seconds; original api_time is not replayed",
            "concurrency": "unlimited concurrent requests; no server queue, network overhead or inflight limit",
            "schedule": "anchor completion + start_delay_s at chain heads; previous completion + gap_after_s within chains; chain_offset_s ignored",
            "window": "each root arrival is t=0; count actual request send times <= 1500 s",
            "root_completion": "maximum request completion time within each root, without truncation at 25 minutes",
            "quantiles": "equal root weights, linear interpolation; request fraction weights every request equally",
        },
        "estimates": [],
    }
    per_root = []
    for service_time in args.service_times:
        summary, ends = estimate(rows, service_time)
        report["estimates"].append(summary)
        per_root.extend(
            {"service_time_s": service_time, "root_id": rid, "completion_s": end}
            for rid, end in sorted(ends.items())
        )
    output = args.output_dir or args.workload.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "replay_estimates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (output / "root_completion_times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["service_time_s", "root_id", "completion_s"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(per_root)
    write_schedule_estimate(rows, output / "schedule_estimate.parquet", digest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
