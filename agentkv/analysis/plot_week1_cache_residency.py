"""Plot the Week 1 cache-residency evidence and an unimplemented hint target."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_evidence(results: Path):
    gpu_dir = results / "two-turn-hello-small-cache-20261101-122132"
    cpu_dir = results / "two-turn-controls-http-20261101-222611"
    gpu_path = gpu_dir / "results.json"
    cpu_path = cpu_dir / "cpu/sessions.json"
    gpu_records = [
        x for x in json.loads(gpu_path.read_text()) if x["scenario"] == "pressure"
    ]
    cpu_records = [
        x for x in json.loads(cpu_path.read_text()) if x["scenario"] == "cpu_full"
    ]
    assert len(gpu_records) == len(cpu_records) == 3
    fixture = json.loads((cpu_dir / "fixture.json").read_text())
    rows = []
    for gpu, cpu in zip(gpu_records, cpu_records, strict=True):
        assert gpu["trial"] == cpu["trial"]
        trial = gpu["trial"]
        original = json.loads(
            (gpu_dir / f"trial{trial}-pressure-token-construction.json").read_text()
        )
        for key in ("turn1", "actual_generated", "fake_tool_suffix"):
            assert original[key] == fixture[key]
        gpu_counts = [gpu["session_gpu_hashes_after_turn1"]] + [
            x["session_gpu_hashes_remaining"] for x in gpu["interference"]
        ]
        cpu_counts = [cpu["turn1"]["original_cpu_ready_after"]] + [
            x["original_cpu_ready_chunks"] for x in cpu["pressure_progression"]
        ]
        assert gpu_counts == [257, 136, 0]
        assert cpu_counts == [256, 256, 256, 256, 195, 0, 0]
        for i in range(7):
            rows.append(
                {
                    "trial": trial,
                    "interference_request": i,
                    "gpu_session_blocks": gpu_counts[i] if i < 3 else 0,
                    "gpu_observation": "matched_run_measured"
                    if i < 3
                    else "inferred_no_session_reaccess",
                    "cpu_session_ready_blocks": int(cpu_counts[i]),
                    "cpu_observation": "six_request_run_measured",
                    "cpu_resident_blocks": 256
                    if i == 0
                    else int(cpu["pressure_progression"][i - 1]["cpu_resident_chunks"]),
                    "cpu_evictions_this_request": 0
                    if i == 0
                    else cpu["pressure_progression"][i - 1][
                        "cpu_evictions_this_request"
                    ],
                    "hint_retention_target_blocks": 256,
                    "hint_target_measured": False,
                }
            )
    summary = json.loads((cpu_dir / "summary.json").read_text())
    metadata = {
        "gpu_source": str(gpu_path),
        "cpu_source": str(cpu_path),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (gpu_path, cpu_path, cpu_dir / "summary.json")
        },
        "gpu_scope": (
            "0-2 measured in three matched diagnostic trials; "
            "3-6 inferred zero because the session is not accessed before turn2"
        ),
        "cpu_scope": "0-6 measured in three six-noise-request HTTP trials",
        "block_scope": (
            "GPU includes one generated block; "
            "CPU stores only 256 original prompt blocks"
        ),
        "cpu_capacity_blocks": 1695,
        "session_cpu_fraction": 256 / 1695,
        "hint_line": "retention target only; no hint policy was run",
        "cpu_kept_ttft_median_ms": summary["cpu_keep"]["client_ttft_text_ms"]["median"],
        "cpu_lost_ttft_median_ms": summary["cpu_full"]["client_ttft_text_ms"]["median"],
        "latency_scope": (
            "different measured controls; keeping 256 blocks under six-noise "
            "pressure remains a hypothesis to test"
        ),
    }
    return rows, metadata


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=project.parent / "vllm/benchmarks/results"
    )
    parser.add_argument("--output-dir", type=Path, default=project / "docs/figures")
    args = parser.parse_args()
    rows, metadata = load_evidence(args.results_root.resolve())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "week1-cache-residency"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), stem.with_suffix(".parquet"))
    assert pq.read_table(stem.with_suffix(".parquet")).to_pylist() == rows

    cjk = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if cjk.exists():
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )
    blue, orange, green = "#2475A8", "#D56C27", "#268466"
    fig = plt.figure(figsize=(11.8, 7.0), facecolor="white")
    ax = fig.add_axes([0.085, 0.31, 0.88, 0.51])
    fig.text(
        0.085,
        0.945,
        "只占 CPU 容量 15.1% 的会话，在第 5 个干扰请求后消失",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.085,
        0.886,
        "Qwen3-8B · GPU 511 可用块 · CPU 1695 块（4 GB） · 每个干扰请求 6000 tokens",
        fontsize=11,
        color="#555555",
    )
    x = list(range(7))
    cpu = [r["cpu_session_ready_blocks"] for r in rows if r["trial"] == 1]
    ax.axvspan(3.95, 6.2, color="#FDF2E8", zorder=0)
    ax.step(
        x,
        cpu,
        where="post",
        color=orange,
        linewidth=2.8,
        label="CPU：六请求实验，3 次一致",
        zorder=3,
    )
    ax.scatter(x, cpu, color=orange, marker="s", s=38, zorder=6)
    ax.step(
        [0, 1, 2],
        [257, 136, 0],
        where="post",
        color=blue,
        linewidth=2.6,
        label="GPU：同配置探针的 0–2 点",
        zorder=4,
    )
    ax.scatter([0, 1, 2], [257, 136, 0], color=blue, s=48, zorder=7)
    ax.plot([2, 6], [0, 0], linestyle=":", color=blue, linewidth=2.5, zorder=5)
    ax.plot(
        [3.5, 6],
        [256, 256],
        color=green,
        linestyle="--",
        linewidth=2.2,
        label="Hint 保留目标：尚未实现",
        zorder=4,
    )
    ax.text(4.35, 273, "保留目标：256 块", color=green, fontsize=12)
    ax.text(0.14, 273, "CPU 256 块 / 1695 块 = 15.1%", color=orange)
    ax.annotate(
        "GPU 257 块\n含 1 个生成块",
        xy=(0, 257),
        xytext=(0.3, 207),
        color=blue,
        arrowprops={"arrowstyle": "-", "color": blue},
    )
    ax.text(1.1, 145, "136", color=blue)
    ax.text(2.1, 17, "GPU 已归零", color=blue)
    for position in (4, 5):
        ax.axvline(position, color="#9C8777", linestyle=":", linewidth=1)
    ax.annotate(
        "第 4 个：开始驱逐\n256 → 195 块",
        xy=(4, 195),
        xytext=(2.8, 139),
        fontsize=11,
        color="#7C401B",
        arrowprops={"arrowstyle": "->", "color": orange},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    ax.annotate(
        "第 5 个后：原会话全部消失\n累计至第 6 个：驱逐 811 块",
        xy=(5, 0),
        xytext=(4.2, 75),
        fontsize=11,
        color="#7C401B",
        arrowprops={"arrowstyle": "->", "color": orange},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    ax.set(
        xlim=(-0.15, 6.2),
        ylim=(-13, 303),
        xlabel="已完成的干扰请求序号（0 = 第一轮结束，尚无干扰）",
        ylabel="原会话驻留的完整 KV 块数",
        xticks=x,
        yticks=[0, 64, 128, 192, 256],
    )
    ax.yaxis.grid(True, color="#E4E8EB", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=(0.53, 0.205),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    kept = metadata["cpu_kept_ttft_median_ms"]
    lost = metadata["cpu_lost_ttft_median_ms"]
    fig.text(
        0.085,
        0.145,
        f"实测第二轮 TTFT：CPU 保留对照 {kept:.1f} ms；"
        f"CPU 丢失对照 {lost:.1f} ms（各 3 次中位数）。",
        fontsize=11,
        color="#333333",
    )
    fig.text(
        0.085,
        0.098,
        "GPU 的 0–2 点来自同输入、同容量探针；3–6 点未再访问原会话，"
        "延续 0（点线表示推断）。CPU 0–6 点均为实测。",
        fontsize=9,
        color="#666666",
    )
    fig.text(
        0.085,
        0.055,
        "绿虚线仅表示保留目标；六请求压力下 Hint 策略的 TTFT 尚未实测，"
        f"{kept:.1f} ms 是 CPU 保留对照的参照值。",
        fontsize=9,
        color="#666666",
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)
    print(
        "FIGURE_EVIDENCE_PASS: 21 rows, 3 repeated curves; "
        "PNG/SVG/PDF + CSV/JSON/parquet"
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
