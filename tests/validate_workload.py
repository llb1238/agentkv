"""Check two independently generated real workloads against the pinned trace."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from agentkv.bench.trace_loader import load_trace, sha256_file
from agentkv.bench.workload import AUDIT, ROOT
from test_synth import verify_generated_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first", type=Path, default=ROOT / "results/m1-workload-seed0-v3"
    )
    parser.add_argument(
        "--second", type=Path, default=ROOT / "results/archive/m1-workload-seed0-v3-repeat"
    )
    parser.add_argument("--position-check", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.first / "manifest.json").read_text())
    pinned = json.loads((AUDIT / "provenance.json").read_text())
    position_path = args.position_check or args.first / "position-check/result.json"
    if args.position_check is None and not position_path.exists():
        position_path = ROOT / "results/m1-position-check/result.json"
    position = json.loads(position_path.read_text())
    assert position["source_sha256"] == pinned["raw_sha256"]
    loaded = load_trace(
        ROOT / "data/raw/agentx/cc-traces-weka-062126/traces.jsonl",
        AUDIT / "requests.parquet",
        AUDIT / "inferred_chains.parquet",
        manifest["selected_roots"],
        pinned["raw_sha256"],
    )
    loaded.provenance["position_check"] = {
        key: position[key]
        for key in (
            "root_sessions",
            "model_requests",
            "block_occurrences",
            "violating_occurrences",
        )
    }
    loaded.provenance["position_check_sha256"] = sha256_file(position_path)
    from agentkv.bench.synth import tokenizer_vocabulary

    vocabulary, info = tokenizer_vocabulary(Path(manifest["tokenizer"]["local_path"]))
    assert info["eligible_ids_sha256"] == manifest["tokenizer"]["eligible_ids_sha256"]
    result = verify_generated_workload(args.first, args.second, loaded, vocabulary)
    # Recompute the prior from audit rows independently of the builder helper.
    selected = set(manifest["selected_roots"])
    audit = pq.read_table(AUDIT / "requests.parquet", columns=[
        "session_id", "chain_id", "chain_turn", "start_s", "api_time_s"
    ]).to_pylist()
    held_out = sorted((r for r in audit if r["session_id"] not in selected),
                      key=lambda r: (r["chain_id"], r["chain_turn"]))
    values = []
    config = manifest["parameters"]
    for previous, current in zip(held_out, held_out[1:]):
        if previous["chain_id"] != current["chain_id"]:
            continue
        assert current["chain_turn"] == previous["chain_turn"] + 1
        # Match the trace's t - (t + api_time) arithmetic grouping exactly.
        raw = current["start_s"] - (previous["start_s"] + previous["api_time_s"])
        assert raw >= 0
        values.append(min(raw, config["gap_cap"]) * config["gap_compress"])
    prior = manifest["conventions"]
    assert prior["online_prior_interval_count"] == len(values)
    assert prior["global_median_prior_s"] == float(np.median(values))
    assert prior["online_prior_root_ids"] == sorted({r["session_id"] for r in held_out})
    result["checks"]["held_out_online_prior"] = "passed"
    result["online_prior"] = {
        "median_s": prior["global_median_prior_s"],
        "interval_count": len(values),
        "root_count": prior["online_prior_root_count"],
    }
    (args.first / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
