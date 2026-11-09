"""Steady-state session populations and replay plans; no serving dependencies."""

import hashlib
import math
import struct
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .estimate_replay import SCHEDULE_SERVICE_TIME_S, simulate_requests


@dataclass(frozen=True)
class ReplayEntry:
    request: dict
    delay_s: float
    trigger_trace_req_idx: str | None
    is_entry: bool

    @property
    def timed(self):
        return self.trigger_trace_req_idx is None


class RootSchedule:
    """Validated request metadata and estimated schedule for exactly one root."""

    def __init__(self, requests: list[dict], schedule: list[dict]):
        if not requests or len({r["root_id"] for r in requests}) != 1:
            raise ValueError("Expected requests from exactly one nonempty root")
        self.root_id = requests[0]["root_id"]
        self.requests = tuple(sorted(requests, key=lambda r: (r["chain_id"], r["chain_turn"])))
        self.by_id = {r["trace_req_idx"]: r for r in self.requests}
        estimates = {r["trace_req_idx"]: r for r in schedule}
        if len(self.by_id) != len(requests) or len(estimates) != len(schedule) or set(estimates) != set(self.by_id):
            raise ValueError("Schedule request identities differ from workload")
        self.sends = {}
        durations = set()
        for idx, request in self.by_id.items():
            estimate = estimates[idx]
            if any(estimate[k] != request[k] for k in ("root_id", "chain_id", "chain_turn")):
                raise ValueError("Schedule identity mismatch")
            send, duration = estimate["est_send_s"], estimate["est_duration_s"]
            if not all(math.isfinite(t) and t >= 0 for t in (send, duration)):
                raise ValueError("Invalid estimated timing")
            self.sends[idx] = send
            durations.add(duration)
        if len(durations) != 1:
            raise ValueError("Expected one duration per root")
        self.duration_s = durations.pop()
        if not math.isclose(self.duration_s, max(self.sends.values()) + SCHEDULE_SERVICE_TIME_S, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Root duration differs from last completion")
        expected = simulate_requests(requests, SCHEDULE_SERVICE_TIME_S)
        if any(not math.isclose(self.sends[k], expected[k], rel_tol=0, abs_tol=1e-8) for k in expected):
            raise ValueError("Estimated sends disagree with dependency graph")
        by_turn = {(r["chain_id"], r["chain_turn"]): r for r in self.requests}
        self.triggers = {}
        self.chain_intervals = []
        chains = defaultdict(list)
        for request in self.requests:
            idx = request["trace_req_idx"]
            if request["chain_turn"]:
                previous = by_turn[request["chain_id"], request["chain_turn"] - 1]
                self.triggers[idx] = previous["trace_req_idx"], previous["gap_after_s"]
            else:
                self.triggers[idx] = request["anchor_trace_req_idx"], request["start_delay_s"]
            chains[request["chain_id"]].append(idx)
        for chain in chains.values():
            self.chain_intervals.append((
                self.sends[chain[0]], self.sends[chain[-1]] + SCHEDULE_SERVICE_TIME_S
            ))

    def entries(self, tau: float) -> list[ReplayEntry]:
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("Progress must be finite and nonnegative")
        if tau >= self.duration_s:
            return []
        retained = {idx for idx, send in self.sends.items() if send >= tau}
        result = []
        for request in self.requests:
            idx = request["trace_req_idx"]
            if idx not in retained:
                continue
            trigger, delay = self.triggers[idx]
            event_driven = trigger in retained
            result.append(ReplayEntry(
                request=request,
                delay_s=delay if event_driven else self.sends[idx] - tau,
                trigger_trace_req_idx=trigger if event_driven else None,
                is_entry=trigger is not None and not event_driven,
            ))
        return result


def replay_entries(requests: list[dict], tau: float, schedule: list[dict]):
    """Return all retained requests, with timer/event delays and entry flags."""
    return RootSchedule(requests, schedule).entries(tau)


def simulate_entries(entries: list[ReplayEntry], service_time_s=SCHEDULE_SERVICE_TIME_S):
    if not math.isfinite(service_time_s) or service_time_s <= 0:
        raise ValueError("Service time must be finite and positive")
    by_id = {e.request["trace_req_idx"]: e for e in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate replay request")
    children = defaultdict(list)
    ready = deque()
    sends = {}
    for idx, entry in by_id.items():
        if not math.isfinite(entry.delay_s) or entry.delay_s < 0:
            raise ValueError("Invalid replay delay")
        if entry.timed:
            sends[idx] = entry.delay_s
            ready.append(idx)
        else:
            if entry.trigger_trace_req_idx not in by_id:
                raise ValueError("Missing retained trigger")
            children[entry.trigger_trace_req_idx].append(idx)
    while ready:
        idx = ready.popleft()
        for child in children[idx]:
            sends[child] = sends[idx] + service_time_s + by_id[child].delay_s
            ready.append(child)
    if len(sends) != len(entries):
        raise ValueError("Cyclic replay entry graph")
    return sends


@dataclass(frozen=True)
class SessionInstance:
    root_id: str
    instance_id: int
    tau_s: float
    start_s: float
    initial: bool


def sample_population(durations: dict[str, float], rate_per_minute: float,
                      population_seed: int, duration_s: float, initialize=True):
    """Use five independent PCG64 streams, shared between initialization modes.

    Initial IDs are odd and new-arrival IDs even, so paired runs use identical
    new instances, arrival times, and instance token permutations.
    """
    if population_seed < 0 or not math.isfinite(rate_per_minute) or rate_per_minute <= 0:
        raise ValueError("Require nonnegative seed and positive finite arrival rate")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Experiment duration must be finite and positive")
    roots = sorted(durations)
    values = np.asarray([durations[r] for r in roots], dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Root durations must be finite and positive")
    rate_bits = int.from_bytes(struct.pack("!d", rate_per_minute), "big")
    streams = np.random.SeedSequence([population_seed, rate_bits]).spawn(5)
    count_rng, root_rng, progress_rng, arrival_rng, new_root_rng = [
        np.random.Generator(np.random.PCG64(s)) for s in streams
    ]
    count = int(count_rng.poisson(rate_per_minute / 60 * values.mean())) if initialize else 0
    instances = []
    chosen = root_rng.choice(len(roots), size=count, replace=True, p=values / values.sum())
    progresses = progress_rng.random(count)
    for i, (index, u) in enumerate(zip(chosen, progresses)):
        tau = min(float(u * values[index]), float(np.nextafter(values[index], 0)))
        instances.append(SessionInstance(roots[index], 2 * i + 1, tau, 0.0, True))
    arrival, i = 0.0, 0
    while True:
        arrival += float(arrival_rng.exponential(60 / rate_per_minute))
        if arrival >= duration_s:
            break
        rid = roots[int(new_root_rng.integers(len(roots)))]
        instances.append(SessionInstance(rid, 2 * i, 0.0, arrival, False))
        i += 1
    return instances


class InstanceTokenPermutation:
    """Apply a deterministic bijection of the eligible vocabulary per instance."""

    def __init__(self, vocabulary: np.ndarray, root_id: str, instance_id: int):
        self.vocabulary = np.asarray(vocabulary)
        if (self.vocabulary.ndim != 1 or not len(self.vocabulary)
                or not np.issubdtype(self.vocabulary.dtype, np.integer)
                or (np.diff(self.vocabulary) <= 0).any() or instance_id < 0):
            raise ValueError("Require sorted unique integer vocabulary and nonnegative instance ID")
        digest = hashlib.sha256(f"{root_id}#{instance_id}".encode("utf-8")).digest()
        self.seed = int.from_bytes(digest[:8], "big")
        self.permutation = np.random.Generator(np.random.PCG64(self.seed)).permutation(self.vocabulary)
        # Token IDs are dense for model vocabularies. Direct lookup avoids
        # O(prompt_len * log(vocab)) searches and thread contention at sends.
        self.lookup = None
        if self.vocabulary[0] >= 0 and self.vocabulary[-1] <= 1_000_000:
            self.lookup = np.full(int(self.vocabulary[-1])+1,-1,dtype=np.int64)
            self.lookup[self.vocabulary] = self.permutation

    def transform(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.size == 0:
            return np.empty(tokens.shape, dtype=self.vocabulary.dtype)
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("Token IDs must be integers")
        if self.lookup is not None:
            if tokens.min()<0 or tokens.max()>=len(self.lookup):
                raise ValueError("Token outside eligible vocabulary")
            result = self.lookup[tokens]
            if (result<0).any():
                raise ValueError("Token outside eligible vocabulary")
            return result.astype(self.vocabulary.dtype,copy=False)
        positions = np.searchsorted(self.vocabulary, tokens)
        if (positions >= len(self.vocabulary)).any() or not np.array_equal(self.vocabulary[positions], tokens):
            raise ValueError("Token outside eligible vocabulary")
        return self.permutation[positions]
