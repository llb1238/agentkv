"""Event-anchored replay estimates with hand-computed schedules."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.estimate_replay import estimate, main, simulate_requests, write_schedule_estimate
from agentkv.bench.trace_loader import sha256_file


def row(idx, chain, turn=0, anchor=None, delay=0.0, gap=None, root="r"):
    return dict(
        root_id=root, chain_id=chain, chain_turn=turn, trace_req_idx=idx,
        anchor_trace_req_idx=anchor, start_delay_s=delay if turn == 0 else None,
        gap_after_s=gap, chain_offset_s=9999999,
    )


@pytest.mark.parametrize("service", [0.2, 0.5, 1.0])
def test_simulation_follows_request_events_and_parallel_fanout(service):
    rows = [
        row("p0", "parent", gap=2),
        row("p1", "parent", turn=1),
        row("a0", "a", anchor="p0", delay=1, gap=3),
        row("a1", "a", turn=1),
        row("b0", "b", anchor="p0", delay=1),
        row("c0", "c", anchor="a1", delay=4),
    ]
    sends = simulate_requests(rows[::-1], service)
    assert sends["p0"] == 0
    assert sends["p1"] == pytest.approx(service + 2)
    assert sends["a0"] == sends["b0"] == pytest.approx(service + 1)
    assert sends["a1"] == pytest.approx(2 * service + 4)
    assert sends["c0"] == pytest.approx(3 * service + 8)
    summary, roots = estimate(rows, service)
    assert roots["r"] == pytest.approx(4 * service + 8)
    assert summary["root_completion_s"]["maximum"] == roots["r"]
    assert summary["requests_sent_within_1500s"] == len(rows)


def test_window_counts_actual_sends_and_includes_exact_boundary():
    rows = [row("p", "p", delay=1499.5, gap=0),
            row("next", "p", turn=1, gap=0),
            row("late", "p", turn=2),
            row("other", "other", delay=1, root="other-root")]
    summary, roots = estimate(rows, 0.5)
    assert summary["requests_sent_within_1500s"] == 3
    assert summary["request_fraction_within_1500s"] == 0.75
    assert roots == {"r": 1501, "other-root": 1.5}


def test_simulator_rejects_cycles_even_through_chain_interior():
    rows = [row("a0", "a", anchor="b0", gap=0),
            row("a1", "a", turn=1), row("b0", "b", anchor="a1")]
    with pytest.raises(ValueError, match="Cyclic"):
        simulate_requests(rows, 0.2)


@pytest.mark.parametrize("parent", ["missing", "other"])
def test_simulator_rejects_missing_or_cross_root_anchors(parent):
    rows = [row("head", "c", anchor=parent), row("other", "d", root="other-root")]
    with pytest.raises(ValueError, match="Missing or cross-root"):
        simulate_requests(rows, 0.2)


@pytest.mark.parametrize("service", [0, -1, float("nan"), float("inf")])
def test_service_time_must_be_finite_and_positive(service):
    with pytest.raises(ValueError, match="Service time"):
        simulate_requests([row("p", "p")], service)


def test_schedule_sidecar_has_fixed_service_time_per_root_and_deterministic_rows(tmp_path):
    from copy import deepcopy

    rows = [row("p0", "parent", gap=2), row("p1", "parent", turn=1),
            row("child", "child", anchor="p0", delay=1),
            row("other", "other", delay=10, root="other-root")]
    before = deepcopy(rows)
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    write_schedule_estimate(rows, first, "a" * 64)
    write_schedule_estimate(rows[::-1], second, "a" * 64)
    assert rows == before
    assert first.read_bytes() == second.read_bytes()
    table = pq.read_table(first)
    assert table.schema.metadata[b"agentkv.service_time_s"] == b"0.5"
    assert table.schema.metadata[b"agentkv.workload_sha256"] == b"a" * 64
    actual = {r["trace_req_idx"]: r for r in table.to_pylist()}
    assert actual["p0"]["est_send_s"] == 0
    assert actual["p1"]["est_send_s"] == 2.5
    assert actual["child"]["est_send_s"] == 1.5
    assert {r["est_duration_s"] for r in actual.values() if r["root_id"] == "r"} == {3}
    assert actual["other"]["est_duration_s"] == 10.5


def test_estimator_cli_adds_sidecar_without_changing_workload_or_existing_report(tmp_path):
    import json

    rows = [row("p", "p", gap=2), row("next", "p", turn=1)]
    workload = tmp_path / "input/workload.parquet"
    workload.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), workload)
    digest = sha256_file(workload)
    workload.with_name("manifest.json").write_text(json.dumps({
        "workload": {"sha256": digest}, "parameters": {"seed": 0},
    }))
    output = tmp_path / "output"
    main(["--workload", str(workload), "--output-dir", str(output),
          "--service-times", "0.2"])
    assert sha256_file(workload) == digest
    report = json.loads((output / "replay_estimates.json").read_text())
    assert len(report["estimates"]) == 1
    assert report["estimates"][0] == estimate(rows, 0.2)[0]
    schedule = pq.read_table(output / "schedule_estimate.parquet").to_pylist()
    assert {r["est_duration_s"] for r in schedule} == {3}
