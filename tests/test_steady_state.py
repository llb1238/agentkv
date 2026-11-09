"""Steady-state boundary rules, population sampling, and instance isolation."""

import hashlib

import numpy as np
import pytest

from agentkv.bench.estimate_replay import simulate_requests, write_schedule_estimate
from agentkv.bench.steady_state import (
    InstanceTokenPermutation, RootSchedule, replay_entries, sample_population,
    simulate_entries,
)
from agentkv.bench.synth import common_prefix
from agentkv.bench.estimate_steady_state import interval_counts, peak_inflight, run_population
from test_estimate_replay import row


@pytest.fixture
def root(tmp_path):
    import pyarrow.parquet as pq

    requests = [
        row("p0", "parent", gap=4), row("p1", "parent", turn=1, gap=3),
        row("p2", "parent", turn=2),
        row("child", "child", anchor="p0", delay=2),
        row("later", "later", anchor="p1", delay=1),
        row("arrival", "arrival", delay=8),
    ]
    for i, request in enumerate(requests):
        request["prompt_len"] = 100 + i
    path = tmp_path / "schedule.parquet"
    write_schedule_estimate(requests, path, "a" * 64)
    return RootSchedule(requests, pq.read_table(path).to_pylist())


def by_id(entries):
    return {e.request["trace_req_idx"]: e for e in entries}


def test_tau_zero_is_normal_replay_without_entries(root):
    entries = root.entries(0)
    assert len(entries) == len(root.requests)
    assert not any(e.is_entry for e in entries)
    assert simulate_entries(entries[::-1]) == simulate_requests(list(root.requests), 0.5)
    for entry in entries:
        parent, delay = root.triggers[entry.request["trace_req_idx"]]
        assert entry.trigger_trace_req_idx == parent
        assert entry.delay_s == delay


def test_tau_during_request_skips_it_and_waits_for_remaining_service(root):
    entries = by_id(root.entries(0.25))
    assert "p0" not in entries
    assert entries["p1"].timed and entries["p1"].is_entry
    assert entries["p1"].delay_s == 0.5 - 0.25 + 4
    assert entries["child"].timed and entries["child"].is_entry
    assert entries["child"].delay_s == 0.5 - 0.25 + 2


def test_tau_in_gap_starts_next_request_with_remaining_delay(root):
    entries = by_id(root.entries(2))
    assert entries["p1"].timed and entries["p1"].is_entry
    assert entries["p1"].delay_s == 4.5 - 2
    assert entries["p2"].trigger_trace_req_idx == "p1"
    assert not entries["p2"].is_entry


def test_future_chain_with_old_anchor_is_timed_entry(root):
    entry = by_id(root.entries(1))["child"]
    assert entry.timed and entry.is_entry
    assert entry.delay_s == 2.5 - 1


def test_future_chain_with_retained_anchor_remains_event_driven(root):
    entry = by_id(root.entries(2))["later"]
    assert entry.trigger_trace_req_idx == "p1"
    assert entry.delay_s == 1
    assert not entry.is_entry


def test_root_arrival_trigger_is_always_timed_and_not_entry(root):
    for tau in (0, 1, 7.5, 8):
        entry = by_id(root.entries(tau))["arrival"]
        assert entry.timed and not entry.is_entry
        assert entry.delay_s == 8 - tau


@pytest.mark.parametrize("extra", [0, 0.1, 1000])
def test_tau_at_or_after_duration_returns_empty(root, extra):
    assert root.entries(root.duration_s + extra) == []


def test_send_boundary_is_inclusive_and_finished_chains_disappear(root):
    assert "child" in by_id(root.entries(2.5))
    assert "child" not in by_id(root.entries(3))
    assert by_id(root.entries(4.5))["p1"].delay_s == 0


@pytest.mark.parametrize("tau", [0, 0.25, 0.5, 1, 2.5, 4.5, 8.25])
def test_truncated_event_simulation_matches_shifted_estimated_schedule(root, tau):
    sends = simulate_entries(root.entries(tau)[::-1])
    expected = {idx: send - tau for idx, send in root.sends.items() if send >= tau}
    assert sends == expected


@pytest.mark.parametrize("tau", [-1, float("nan"), float("inf")])
def test_invalid_progress_is_rejected(root, tau):
    with pytest.raises(ValueError, match="Progress"):
        root.entries(tau)


def test_invalid_schedule_cannot_silently_change_initialization(root):
    rows = [dict(root_id=r["root_id"], chain_id=r["chain_id"], chain_turn=r["chain_turn"],
                 trace_req_idx=r["trace_req_idx"], est_send_s=root.sends[r["trace_req_idx"]],
                 est_duration_s=root.duration_s) for r in root.requests]
    next(r for r in rows if r["trace_req_idx"] == "p0")["est_send_s"] += 0.25
    with pytest.raises(ValueError, match="disagree"):
        replay_entries(list(root.requests), 0, rows)


def test_population_has_duration_bias_uniform_progress_and_paired_new_arrivals():
    durations = {"short": 1, "long": 1000}
    on = sample_population(durations, 60.0, 0, 60)
    off = sample_population(durations, 60.0, 0, 60, initialize=False)
    assert on == sample_population(dict(reversed(list(durations.items()))), 60.0, 0, 60)
    assert on != sample_population(durations, 60.0, 1, 60)
    initial = [i for i in on if i.initial]
    assert len(initial) > 100
    assert sum(i.root_id == "long" for i in initial) / len(initial) > 0.95
    assert np.mean([i.tau_s / durations[i.root_id] for i in initial]) == pytest.approx(0.5, abs=0.08)
    assert [i for i in on if not i.initial] == off
    assert all(0 <= i.tau_s < durations[i.root_id] and i.instance_id % 2 == 1 for i in initial)
    assert all(i.tau_s == 0 and i.instance_id % 2 == 0 for i in off)
    assert len({i.instance_id for i in on}) == len(on)


def test_instance_permutation_is_bijective_preserves_lcp_and_isolates_instances():
    vocabulary = np.arange(100, 151755, dtype=np.int32)
    left = InstanceTokenPermutation(vocabulary, "root", 0)
    right = InstanceTokenPermutation(vocabulary, "root", 1)
    expected_seed = int.from_bytes(hashlib.sha256(b"root#0").digest()[:8], "big")
    assert left.seed == expected_seed
    assert np.array_equal(np.sort(left.transform(vocabulary)), vocabulary)
    prompts = [np.array(tokens) for tokens in ([123, 124, 125], [123, 124, 130], [123], [140, 124])]
    for a in prompts:
        assert left.transform(a)[0] != right.transform(a)[0]
        for b in prompts:
            assert common_prefix(a, b) == common_prefix(left.transform(a), left.transform(b))
    assert np.array_equal(left.permutation, InstanceTokenPermutation(vocabulary, "root", 0).permutation)
    assert left.transform([]).size == 0
    with pytest.raises(ValueError, match="eligible"):
        left.transform([99])


def test_active_counts_include_gaps_and_inflight_is_half_open():
    assert interval_counts([(0, 10), (5, 15)], [0, 5, 10, 15]).tolist() == [1, 2, 1, 0]
    assert peak_inflight([{"send_s":0}, {"send_s":0.5}]) == 1
    assert peak_inflight([{"send_s":0}, {"send_s":0.25}]) == 2


def test_population_estimate_is_reproducible_and_excludes_entries_from_prompts(root):
    args = ({"r": root}, 30.0, 0, True)
    first = run_population(*args, duration=40, warmup=0)
    second = run_population(*args, duration=40, warmup=0)
    assert first[:4] == second[:4]
    summary, _, records, _, prompts = first
    selected = [r for r in records if r["in_window"] and not r["is_entry"]]
    assert len(prompts) == len(selected) == summary["window_non_entry_requests"]
    assert summary["window_is_entry_requests"] > 0
    assert summary["window_requests"] == len(prompts) + summary["window_is_entry_requests"]


def test_dense_permutation_lookup_is_identical_to_reference_with_special_id_holes():
    vocabulary=np.array([0,1,4,8,100,151654],dtype=np.int32)
    instance=InstanceTokenPermutation(vocabulary,'equivalence',71)
    tokens=np.array([151654,4,4,0,100,1,8],dtype=np.int32)
    expected=instance.permutation[np.searchsorted(vocabulary,tokens)]
    assert np.array_equal(instance.transform(tokens),expected)
    for invalid in ([2],[-1],[151655]):
        with pytest.raises(ValueError,match='outside'):
            instance.transform(invalid)
    sparse=InstanceTokenPermutation(np.array([0,2_000_000],dtype=np.int32),'equivalence',71)
    assert np.array_equal(sparse.transform([0,2_000_000]),sparse.permutation)
