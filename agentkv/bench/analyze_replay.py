"""Summarize replay output and plot the server waiting queue over time."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from agentkv.analysis.stability import evaluate
from .replay import percentiles
from .trace_loader import sha256_file


def analyze(directory):
    run = json.loads((directory / "run.json").read_text())
    if run["status"] != "complete":
        raise ValueError("Replay has not completed")
    if sha256_file(directory / "requests.parquet") != run["requests_sha256"]:
        raise ValueError("Request output SHA256 mismatch")
    columns = ["request_id","instance_id","trace_req_idx","chain_id","chain_turn","was_queued","prompt_len","in_window","is_entry",
               "status","error","send_ns","end_ns","ttft_ms","gap_error_ms","schedule_error_ms",
               "queued_ms","gpu_hit","cpu_hit","recomputed","gpu_hit_fraction","cpu_hit_fraction","recomputed_fraction"]
    rows = pq.read_table(directory / "requests.parquet",columns=columns,use_threads=False).to_pylist()
    window = [r for r in rows if r["in_window"]]
    success = [r for r in window if r["status"]=="success"]
    by_turn = {(r["instance_id"],r["chain_id"],r["chain_turn"]):r for r in rows}
    unqueued_gaps = []
    for r in window:
        following = by_turn.get((r["instance_id"],r["chain_id"],r["chain_turn"]+1))
        if r["gap_error_ms"] is not None and following is not None and not following["was_queued"]:
            unqueued_gaps.append(abs(r["gap_error_ms"]))
    samples = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    p = run["parameters"]
    observed = [s for s in samples if p["warmup"]<=s["elapsed_s"]<=p["duration"]-p["cooldown"]
                and "vllm:num_requests_waiting" in s.get("values",{})]
    waiting = [s["values"]["vllm:num_requests_waiting"] for s in observed]
    times = np.asarray([s["elapsed_s"]/60 for s in observed])
    trend = float(np.polyfit(times,waiting,1)[0]) if len(times)>1 else None
    cache = [r for r in success if r["gpu_hit"] is not None]
    if any(sum(r[k] for k in ("gpu_hit","cpu_hit","recomputed"))!=r["prompt_len"] for r in cache):
        raise ValueError("Non-additive nominal per-request cache statistics")
    fraction = {k:dict(mean=float(np.mean([r[k+"_fraction"] for r in cache])) if cache else None,
                      **percentiles([r[k+"_fraction"] for r in cache])) for k in ("gpu_hit","cpu_hit","recomputed")}
    server = json.loads((directory / 'server-manifest.json').read_text()) if (directory / 'server-manifest.json').exists() else {}
    events = []
    for path in sorted((directory / 'server-observations').glob('*.jsonl')):
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    stability = evaluate(samples, rows, origin_ns=run['origin_ns'],duration_s=p['duration'],
                         warmup_s=p['warmup'],cooldown_s=p['cooldown'],
                         max_num_seqs=server.get('max_num_seqs',(run.get('server') or {}).get('max_num_seqs',8)),
                         kv_events=events,kv_observed=any(e.get('event')=='kv_observer_ready' for e in events))
    report = dict(stability=stability,run_summary=run["summary"],window_requests=len(window),success=len(success),
                  failed=len(window)-len(success),ttft_ms=percentiles([r["ttft_ms"] for r in success]),
                  server_waiting=dict(samples=len(waiting),**percentiles(waiting),
                                      fraction_positive=sum(v>0 for v in waiting)/len(waiting) if waiting else None,
                                      linear_trend_requests_per_minute=trend),
                  schedule_absolute_error_ms=percentiles([abs(r["schedule_error_ms"]) for r in window]),
                  gap_absolute_error_ms=percentiles([abs(r["gap_error_ms"]) for r in window if r["gap_error_ms"] is not None]),
                  unqueued_schedule_absolute_error_ms=percentiles([abs(r["schedule_error_ms"]) for r in window if not r["was_queued"]]),
                  unqueued_gap_absolute_error_ms=percentiles(unqueued_gaps),
                  nominal_per_request_fraction=fraction,
                  nominal_join_success=len(cache),nominal_join_missing=len(success)-len(cache),
                  failures=[dict(request_id=r["request_id"],error=r["error"]) for r in window if r["status"]!="success"],
                  source=dict(requests_sha256=run["requests_sha256"],git_commit=run.get("git_commit") or run.get("git",{}).get("commit"),server=run["server"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [s for s in samples if "vllm:num_requests_waiting" in s.get("values",{})]
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    ax.plot([s["elapsed_s"]/60 for s in good],[s["values"]["vllm:num_requests_waiting"] for s in good],
            label="Waiting",color="#D55E00",linewidth=.8)
    ax.plot([s["elapsed_s"]/60 for s in good],[s["values"].get("vllm:num_requests_running",0) for s in good],
            label="Running",color="#0072B2",linewidth=.8,alpha=.7)
    ax.axvspan(0,p["warmup"]/60,color="gray",alpha=.15,label="Warmup")
    ax.set(xlabel="Elapsed time (minutes)",ylabel="Server requests",xlim=(0,p["duration"]/60))
    ax.legend(ncol=3,frameon=False,fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / "server_queue.png",dpi=300)
    fig.savefig(directory / "server_queue.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)
    report["queue_plot_sha256"]={name:sha256_file(directory/name) for name in ("server_queue.png","server_queue.pdf")}
    (directory / "analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (directory / "per_request_nominal.csv").open("w",newline="") as sink:
        fields=["request_id","in_window","status","prompt_len","gpu_hit_fraction","cpu_hit_fraction","recomputed_fraction"]
        writer=csv.DictWriter(sink,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ("source","failures")},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    analyze(parser.parse_args().run_dir)


if __name__=="__main__":
    main()
