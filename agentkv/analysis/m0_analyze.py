# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join M0 process traces and client timestamps into additive TTFT intervals."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

STAGES = [
    "HTTP 发送→调度器收到请求",
    "调度器收到→首次调度",
    "GPU 查找及调度入口",
    "connector 查找",
    "分配、构造加载任务及拷贝描述符",
    "提交拷贝的主机调用",
    "提交返回→Worker 观察到完成",
    "Worker 完成→调度器收到上报",
    "收到上报→下一次调度恢复",
    "恢复→prefill 调度完成",
    "调度完成→Worker 执行入口",
    "Worker 执行→调度器处理首 token",
    "调度器输出→前端首 token",
    "前端首 token→客户端首文本",
]


def read_events(directory):
    events = []
    for path in sorted(directory.glob("trace-*.jsonl")):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    return sorted(events, key=lambda e: e["ns"])


def decompose(session, events, by_request):
    client = session["turn2"]
    name = client["name"]
    req_ids = [key for key in by_request if key.startswith(f"cmpl-{name}-0-")]
    assert len(req_ids) == 1, (name, req_ids)
    req_id = req_ids[0]
    req = by_request[req_id]

    def first(event, **fields):
        return next(
            e
            for e in req
            if e["event"] == event and all(e.get(k) == v for k, v in fields.items())
        )

    arrived = first("request_arrived")
    considered = first("schedule_consider")
    lookup_begin = first("connector_lookup_begin")
    lookup_end = first("connector_lookup_end")
    dispatch = next(
        e
        for e in events
        if e["ns"] >= considered["ns"]
        and e["event"] == "schedule_end"
        and req_id in e["tokens"]
    )
    execute = first("worker_execute")
    output = first("scheduler_output")
    frontend = first("frontend_first_token")
    load = session["scenario"] == "cpu_keep"
    if load:
        enqueue = first("copy_enqueue", direction="load")
        enqueued = first("copy_enqueued", direction="load")
        done = first("copy_done", direction="load")
        recv = first("scheduler_recv_done")
        resume = first("load_resume")
        middle = [enqueue, enqueued, done, recv, resume]
    else:
        middle = [dispatch] * 5
    boundaries = [
        client["start_ns"],
        arrived["ns"],
        considered["ns"],
        lookup_begin["ns"],
        lookup_end["ns"],
        *(e["ns"] for e in middle),
        dispatch["ns"],
        execute["ns"],
        output["ns"],
        frontend["ns"],
        client["first_text_ns"],
    ]
    durations_ns = [b - a for a, b in zip(boundaries, boundaries[1:])]
    assert len(durations_ns) == len(STAGES)
    assert min(durations_ns) >= 0, (name, durations_ns)
    ttft_ns = client["first_text_ns"] - client["start_ns"]
    assert sum(durations_ns) == ttft_ns
    assert abs(ttft_ns / 1e6 - client["client_ttft_text_ms"]) < 0.000002
    assert execute["num_tokens"] == client["local_prefill_tokens"]
    gpu_begin, gpu_end = first("gpu_lookup_begin"), first("gpu_lookup_end")
    row = dict(
        name=name,
        request_id=req_id,
        scenario=session["scenario"],
        trial=session["trial"],
        ttft_ms=ttft_ns / 1e6,
        scheduler_pid=considered["pid"],
        worker_pid=execute["pid"],
        frontend_pid=frontend["pid"],
        boundaries_ns=boundaries,
        stages_ms=dict(zip(STAGES, (v / 1e6 for v in durations_ns))),
        gpu_lookup_ms=(gpu_end["ns"] - gpu_begin["ns"]) / 1e6,
        connector_lookup_ms=(lookup_end["ns"] - lookup_begin["ns"]) / 1e6,
        prefill_tokens=execute["num_tokens"],
    )
    if load:
        wait = first("load_wait")
        scan_begin, scan_end = first("cpu_hash_scan_begin"), first("cpu_hash_scan_end")
        row.update(
            cuda_ms=done["cuda_ms"],
            copy_ops=enqueue["copy_ops"],
            num_bytes=done["num_bytes"],
            copy_impl=enqueue["copy_impl"],
            gbps=done["num_bytes"] / done["cuda_ms"] / 1e6,
            bulk_estimate_ms=done["num_bytes"] / 55e6,
            cpu_hash_scan_ms=(scan_end["ns"] - scan_begin["ns"]) / 1e6,
            cpu_hash_scan_hits=scan_end["result"],
            cpu_hash_scan_checked=first("cpu_hash_scan_result")["checked"],
            load_step=wait["step"],
            recv_step=recv["step"],
            resume_step=resume["step"],
            blocked_checks=sum(e["event"] == "load_blocked" for e in req),
            load_wait_ms=(resume["ns"] - wait["ns"]) / 1e6,
        )
    return row


def lifecycle(events, by_request, session):
    prefix = f"cmpl-{session['turn1']['name']}-0-"
    req_id = next(key for key in by_request if key.startswith(prefix))
    plan = next(
        e
        for e in by_request[req_id]
        if e["event"] == "store_plan" and len(e["keys"]) == 256
    )
    key = plan["keys"][0]
    start = next(e["ns"] for e in by_request[req_id] if e["event"] == "request_arrived")
    end = session["turn2"]["first_text_ns"]
    selected = [
        e
        for e in events
        if start < e["ns"] < end
        and (
            (
                e.get("key") == key
                and e["event"] in ("gpu_cache_insert", "gpu_free_zero_ref")
            )
            or (
                key in e.get("keys", ())
                and e["event"]
                in ("gpu_hash_remove", "cpu_keys_ready", "cpu_evict", "cpu_load_keys")
            )
        )
    ]
    return dict(key=key, request_id=req_id, store_plan=plan, events=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    events = read_events(root / "traces")
    by_request = defaultdict(list)
    for event in events:
        if event["request_id"]:
            by_request[event["request_id"]].append(event)
    sessions = json.loads((root / "cpu/sessions.json").read_text())
    rows = [
        decompose(s, events, by_request)
        for s in sessions
        if s["scenario"] in ("gpu_quiet", "cpu_keep")
    ]
    assert len(rows) == 6
    representatives, medians = {}, {}
    for scenario in ("gpu_quiet", "cpu_keep"):
        group = [r for r in rows if r["scenario"] == scenario]
        representatives[scenario] = sorted(group, key=lambda r: r["ttft_ms"])[1]
        medians[scenario] = dict(
            ttft_ms=median(r["ttft_ms"] for r in group),
            stages_ms={s: median(r["stages_ms"][s] for r in group) for s in STAGES},
            samples_ms=[r["ttft_ms"] for r in group],
        )
    summary = dict(
        event_count=len(events),
        requests=rows,
        representatives=representatives,
        component_medians=medians,
        extra_ttft_ms=medians["cpu_keep"]["ttft_ms"] - medians["gpu_quiet"]["ttft_ms"],
        lifecycle=lifecycle(
            events,
            by_request,
            next(
                s for s in sessions if s["trial"] == 1 and s["scenario"] == "cpu_keep"
            ),
        ),
        cpu_eviction=lifecycle(
            events,
            by_request,
            next(
                s for s in sessions if s["trial"] == 1 and s["scenario"] == "cpu_full"
            ),
        ),
    )
    (root / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (root / "ttft_per_request.csv").open("w") as sink:
        writer = csv.writer(sink)
        writer.writerow(["name", "ttft_ms", *STAGES])
        for row in rows:
            writer.writerow([row["name"], row["ttft_ms"], *row["stages_ms"].values()])
    gpu, cpu = (representatives[k] for k in ("gpu_quiet", "cpu_keep"))
    lines = [
        "| 阶段（ms） | GPU 命中 | CPU 回载 | 差值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        a, b = gpu["stages_ms"][stage], cpu["stages_ms"][stage]
        lines.append(f"| {stage} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    a, b = gpu["ttft_ms"], cpu["ttft_ms"]
    lines.append(f"| **合计 TTFT** | **{a:.3f}** | **{b:.3f}** | **{b - a:+.3f}** |")
    (root / "ttft_table.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "event_count": len(events),
                "medians": medians,
                "extra_ttft_ms": summary["extra_ttft_ms"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
