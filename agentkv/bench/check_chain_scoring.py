"""Read-only witnesses that strict depth-tail ordering changes frozen decisions.

This replays the *unchanged* indexed scorer against every recorded evict result.
It is a feasibility diagnostic, not a replacement policy or a latency benchmark.
"""
import argparse
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.bench.e0 import write_json
from agentkv.bench.profile_callbacks import compile_trace, execute
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy
from agentkv.policy.prefix_adapter import ROOT, frozen_options
from agentkv.provenance import git_commit


def tail_witness(core, key, now, protected):
    owners = core.owners.get(key, {})
    if not owners:
        return None
    lines = sorted(core._lines(key))
    first = next(iter(owners))
    for other in core.chains[first].keys:
        if other == key or other in protected or core.chunks[other].ref_cnt != 0:
            continue
        other_owners = core.owners[other]
        if other_owners.keys() != owners.keys():
            continue
        if not all(other_owners[c] > depth for c, depth in owners.items()):
            continue
        if sorted(core._lines(other)) != lines:
            continue
        def block(k):
            return dict(key_hex=k.hex(), rank=core.ranks[k], ref_cnt=core.chunks[k].ref_cnt,
                protected=k in protected,
                owners=[dict(instance=c[0], chain=c[1], depth=d) for c, d in core.owners[k].items()])
        return dict(now_s=now, frozen_selected=block(key), eligible_deeper=block(other),
            identical_owner_sets=True, identical_score_lines=True,
            score_lines=[[a, b if math.isfinite(b) else str(b)] for a, b in lines],
            implication='A strict depth-tail rule must select a deeper eligible block before the recorded shallow victim.')
    return None


def returned_membership(steps, witness):
    if witness is None:
        return None
    step = next(s for s in steps if s[0] == 'call' and s[1] == 'evict' and s[-1] == witness['call'])
    returned = step[-2]
    shallow = bytes.fromhex(witness['frozen_selected']['key_hex'])
    deeper = bytes.fromhex(witness['eligible_deeper']['key_hex'])
    assert returned is not None and shallow in returned
    return dict(call=witness['call'], returned_keys=len(returned),
        selected_position=returned.index(shallow), deeper_also_returned=deeper in returned,
        strict_depth_tail_changes_returned_set=deeper not in returned)


def check(directory, output, tier):
    commit = git_commit()
    backend = git_commit(repository=ROOT.parent / 'vllm')
    trial = json.loads((directory / 'trial-run.json').read_text())
    config = json.loads((directory / 'policy-config.json').read_text())
    assert tier == config['hint_tier'] and trial['status'] == 'complete'
    assert trial['vllm_commit'] == backend
    for name, digest in trial['trace_sha256'].items():
        assert sha256_file(directory / 'callback-traces' / name) == digest
    if output.exists():
        raise FileExistsError(output)
    steps, capacity = compile_trace(directory)
    evict_calls = [s[-1] for s in steps if s[0] == 'call' and s[1] == 'evict']
    witnesses = []

    class ExaminedPolicy(OptimizedHintAwarePolicy):
        def _remove(self, key, now, **kwargs):
            if self._in_evict and not witnesses:
                value = tail_witness(self, key, now, self.examined_protected)
                if value is not None:
                    value.update(call=self.examined_call, eviction_index=self.examined_index,
                                 requested_blocks=self.examined_n,
                                 protected_blocks=len(self.examined_protected))
                    witnesses.append(value)
            return super()._remove(key, now, **kwargs)

    class ExaminedAdapter(OptimizedPrefixAdapter):
        def __init__(self, capacity, clock):
            super().__init__(capacity, clock=clock, hint_tier=tier)
            self.policy = ExaminedPolicy(capacity, clock=clock, **frozen_options(tier))
            self.examined_index = 0

        def evict(self, n, protected):
            self.policy.examined_call = evict_calls[self.examined_index]
            self.policy.examined_index = self.examined_index
            self.policy.examined_n = n
            self.policy.examined_protected = protected
            self.examined_index += 1
            before = len(witnesses)
            result = super().evict(n, protected)
            if result is None:
                # Failed exact-n eviction rolls logical removals back; no key
                # from that attempt was actually returned to the caller.
                del witnesses[before:]
            return result

    summary, _, evictions, _ = execute(steps, capacity, 'read-only-inspection',
        actor_factory=lambda capacity, clock: ExaminedAdapter(capacity, clock))
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{k: e[k] for k in ('call', 'n', 'victims_sha256', 'none')}
                                        for e in evictions]), output / 'decisions.parquet')
    result = dict(git_commit=commit, vllm_commit=backend, status='complete', hint_tier=tier,
        implementation='Unchanged indexed scorer with read-only inspection before removal',
        timing_is_performance_measurement=False, frozen_replay_all_evictions_identical=True,
        eviction_fingerprint=summary['eviction_fingerprint'], callback_counts=summary['counts'],
        evictions=len(evictions), depth_tail_incompatibility_witness=witnesses[0] if witnesses else None,
        witness_returned_membership=returned_membership(steps, witnesses[0] if witnesses else None),
        strict_depth_tail_order_compatible=False if witnesses else None,
        not_a_chain_queue_implementation=True,
        trace_sha256=trial['trace_sha256'], trial_sha256=sha256_file(directory / 'trial-run.json'),
        decisions_sha256=sha256_file(output / 'decisions.parquet'))
    write_json(output / 'run.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--hint-tier', choices=('online', 'perfect'), required=True)
    a = p.parse_args()
    check(a.run_dir, a.output_dir, a.hint_tier)


if __name__ == '__main__':
    main()
