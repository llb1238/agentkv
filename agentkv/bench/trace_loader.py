"""Join the public trace with the pinned audit's request and chain identities."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class TraceRequest:
    root_id: str
    chain_id: str
    chain_turn: int
    trace_req_idx: str
    hash_ids: np.ndarray
    t: float
    api_time: float
    input_len: int
    output_len: int
    is_subagent: bool


@dataclass
class LoadedTrace:
    roots: dict[str, list[TraceRequest]]
    chains: dict[str, dict]
    provenance: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_roots(requests_path: Path) -> list[str]:
    table = pq.read_table(requests_path, columns=["session_id"])
    return sorted(set(table.column(0).to_pylist()))


def load_trace(
    trace_path: Path,
    requests_path: Path,
    chains_path: Path,
    root_ids: list[str],
    expected_sha256: str,
) -> LoadedTrace:
    selected = set(root_ids)
    if len(selected) != len(root_ids):
        raise ValueError("Duplicate root IDs")
    columns = [
        "session_id",
        "container_id",
        "original_array_index",
        "chain_id",
        "chain_turn",
        "start_s",
        "api_time_s",
        "input_tokens",
        "output_tokens",
    ]
    audit = pq.read_table(requests_path, columns=columns).to_pylist()
    by_position = {}
    for row in audit:
        if row["session_id"] in selected:
            key = (row["container_id"], row["original_array_index"])
            if key in by_position:
                raise ValueError(f"Duplicate audit position: {key}")
            by_position[key] = row
    chain_rows = pq.read_table(chains_path).to_pylist()
    chains = {c["chain_id"]: c for c in chain_rows if c["session_id"] in selected}
    roots = {rid: [] for rid in root_ids}
    seen = set()
    digest = hashlib.sha256()
    with trace_path.open("rb") as source:
        for line_no, line in enumerate(source, 1):
            digest.update(line)
            raw = json.loads(line)
            rid = raw["id"]
            if rid not in selected:
                continue
            if roots[rid]:
                raise ValueError(f"Duplicate raw root: {rid}")
            if raw["block_size"] != 64 or raw["hash_id_scope"] != "local":
                raise ValueError("Only root-local 64-token hash blocks are supported")
            for outer, entry in enumerate(raw["requests"]):
                child = entry["type"] == "subagent"
                container = (
                    f"{rid}::subagent:{entry['agent_id']}:entry:{outer}"
                    if child
                    else f"{rid}::root"
                )
                inner_requests = entry["requests"] if child else [entry]
                for inner, request in enumerate(inner_requests):
                    index = inner if child else outer
                    key = (container, index)
                    row = by_position[key]
                    seen.add(key)
                    t = request["t"]
                    if child and t + 1e-6 < entry["t"]:
                        t += entry["t"]
                    duration = request["api_time"]
                    if not all(math.isfinite(x) for x in (t, duration)):
                        raise ValueError(f"Non-finite timing: {key}")
                    if t < 0 or duration < 0:
                        raise ValueError(f"Negative timing: {key}")
                    actual = (t, duration, request["in"], request["out"])
                    recorded = tuple(
                        row[k]
                        for k in (
                            "start_s",
                            "api_time_s",
                            "input_tokens",
                            "output_tokens",
                        )
                    )
                    if actual != recorded or request["in"] != 64 * len(
                        request["hash_ids"]
                    ):
                        raise ValueError(f"Raw/audit mismatch: {key}")
                    if request["out"] < 0 or not request["hash_ids"]:
                        raise ValueError(f"Invalid token lengths: {key}")
                    hashes = np.asarray(request["hash_ids"], dtype=np.int64)
                    roots[rid].append(
                        TraceRequest(
                            rid,
                            row["chain_id"],
                            row["chain_turn"],
                            f"{line_no}/requests/{outer}"
                            + (f"/requests/{inner}" if child else ""),
                            hashes,
                            t,
                            duration,
                            request["in"],
                            request["out"],
                            child,
                        )
                    )
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Trace SHA256 differs from the pinned audit")
    if seen != set(by_position) or any(not requests for requests in roots.values()):
        raise ValueError("Missing raw requests or roots")
    observed = {}
    for rid, requests in roots.items():
        requests.sort(key=lambda r: (r.chain_id, r.chain_turn))
        for request in requests:
            if request.chain_id not in chains:
                raise ValueError(f"Unknown chain: {request.chain_id}")
            observed.setdefault(request.chain_id, []).append(request)
    if set(observed) != set(chains):
        raise ValueError("Audit chain partition differs from raw requests")
    for cid, requests in observed.items():
        if [r.chain_turn for r in requests] != list(range(len(requests))):
            raise ValueError(f"Non-contiguous chain: {cid}")
        if len(requests) != chains[cid]["request_count"]:
            raise ValueError(f"Wrong chain length: {cid}")
        if requests[0].t != chains[cid]["first_start_s"]:
            raise ValueError(f"Wrong chain offset: {cid}")
    return LoadedTrace(
        roots,
        chains,
        {
            "trace_sha256": digest.hexdigest(),
            "requests_sha256": sha256_file(requests_path),
            "inferred_chains_sha256": sha256_file(chains_path),
            "trace_req_idx_format": (
                "one-based JSONL line / requests / zero-based index [/ "
                "requests / zero-based child index]"
            ),
        },
    )
