"""Diagnose original probe omissions without inventing missing callback arguments."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from agentkv.analysis.callback_probe_report import events
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.hints import make_envelope
from agentkv.policy.hint_aware import HintAwarePolicy
from agentkv.provenance import ROOT, git_commit


def fingerprints(tokens, block_size=16):
    """Equality labels for original prefixes, not a reconstruction of vLLM hashes.

    The same instance uses the same bijection for all its requests. Therefore
    original prefix equality implies transmitted prefix equality. Instance IDs
    must remain part of the lookup key.
    """
    state = b''
    result = []
    for offset in range(0, len(tokens) // block_size * block_size, block_size):
        state = hashlib.sha256(state + struct.pack(
            '<' + 'i' * block_size, *tokens[offset:offset + block_size])).digest()
        result.append(state)
    return result


def classify(full_blocks, observed_positions, gpu_hit, block_size=16):
    missing = set(range(block_size, full_blocks * block_size + 1, block_size)) - set(observed_positions)
    if not full_blocks and not observed_positions:
        return 'prompt_shorter_than_block', missing
    if missing and missing == set(range(block_size, gpu_hit + 1, block_size)):
        return 'gpu_hit_prefix_skipped_by_connector', missing
    return 'unresolved', missing


def final_sharing_counterexample(final_hint, waiting_hint, key):
    """Minimal CPU counterexample; not a replay of unrecorded cache mutations.

    Real final/waiting identities and the omitted real key are used. A synthetic
    later-returning control chain makes the ordering difference observable.
    """
    final = final_hint['actions'][0]['payload']
    waiting = waiting_hint['actions'][0]['payload']
    assert final['final'] and not waiting['final']
    control_key = b'agentkv-diagnosis-control'
    control = json.loads(json.dumps(waiting_hint))
    control['message_id'] = 'synthetic-control'
    payload = control['actions'][0]['payload']
    payload.update(session_id='synthetic-control-chain', parent_session='synthetic-control-instance',
                   expected_gap_ms=waiting['expected_gap_ms'] + 60000,
                   gap_p90_ms=waiting['expected_gap_ms'] + 120000)

    def context(envelope, keys):
        return SimpleNamespace(req_id=envelope['message_id'], kv_hints=None,
            kv_transfer_params={'kv_hints': envelope},
            get_offload_key_position={k: 16 for k in keys}.get)

    decisions = {}
    for label, include_shared in [('recorded_omission', False), ('ownership_complete_control', True)]:
        policy = HintAwarePolicy(2, clock=lambda: 100., scoring='deadline',
                                 tail_first=False, final_first=True, timeout_decay=True)
        for k in (key, control_key):
            policy.insert(k, SimpleNamespace(ref_cnt=0))
        policy.on_request_finished(((key,),), set(), {key}, context(final_hint, [key]))
        policy.on_request_finished(((control_key,),), set(), {control_key}, context(control, [control_key]))
        groups = ((key,),) if include_shared else ()
        policy.on_request_finished(groups, set(), set(), context(waiting_hint, [key]))
        identity = (waiting['parent_session'] or waiting['session_id'], waiting['session_id'])
        chain = policy.chains[identity]
        decisions[label] = dict(waiting_chain_mode=chain.mode, deadline=chain.deadline,
            shared_registered_to_waiting_chain=identity in policy.owners.get(key, {}),
            victim='shared' if policy.evict(1, set())[0][0] == key else 'control')
    assert decisions['recorded_omission']['victim'] == 'shared'
    assert decisions['ownership_complete_control']['victim'] == 'control'
    return decisions


def diagnose(probe, workload, output):
    commit = git_commit()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    run = json.loads((probe / 'run.json').read_text())
    freeze = json.loads((ROOT/'docs/m5_scoring_frozen.json').read_text())
    assert sha256_file(ROOT/'agentkv/policy/hint_aware.py') == freeze['source_hashes']['agentkv/policy/hint_aware.py']
    assert sha256_file(workload) == run['source']['workload_sha256']
    metadata = pq.read_table(probe / 'request-metadata.parquet').to_pylist()
    clients = {r['request_id']: r for r in metadata if r['send_ns'] is not None}
    trace = {r['trace_req_idx']: r for r in pq.read_table(workload).to_pylist()}
    labels, instances, seeds = {}, {}, {}
    for msg, row in clients.items():
        original = trace[row['trace_req_idx']]
        assert original['prompt_len'] == row['prompt_len']
        instances[msg] = (row['root_id'], row['instance_id'])
        instance = instances[msg]
        assert seeds.setdefault(instance, row['permutation_seed']) == row['permutation_seed']
        labels[msg] = fingerprints(original['prompt_token_ids'])

    context_events = [e for e in events(probe / 'policy-observations')
                      if e.get('hint', {}).get('message_id') in clients]
    assert all(a['ns'] <= b['ns'] for a, b in zip(context_events, context_events[1:]))
    known = {}
    for e in context_events:
        msg = e['hint']['message_id']
        for keys, positions in zip(e.get('key_groups', ()), e.get('position_groups', ())):
            for key, pos in zip(keys, positions):
                if pos is None or pos % 16 or pos > clients[msg]['prompt_len']:
                    continue
                identity = (instances[msg], labels[msg][pos // 16 - 1])
                assert known.setdefault(identity, key) == key

    by_request = defaultdict(list)
    ever = defaultdict(set)
    for e in context_events:
        msg = e['hint']['message_id']
        positions = e.get('positions', []) + [p for g in e.get('position_groups', []) for p in g]
        e['_resolved_keys'] = [known[(instances[msg], labels[msg][p // 16 - 1])]
            for p in positions if p is not None and p % 16 == 0 and p <= clients[msg]['prompt_len']
            and (instances[msg], labels[msg][p // 16 - 1]) in known]
        by_request[msg].append(e)
        ever[e['hint']['session_id']].update(e['_resolved_keys'])

    lifecycle = {}
    for path in sorted((probe / 'server-observations').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['event'] == 'prefill':
                assert e['request_id'] not in lifecycle
                lifecycle[e['request_id']] = e

    prior = defaultdict(set)
    modes, final_messages = {}, {}
    rows, witnesses = [], []
    for e in context_events:
        h = e['hint']; msg = h['message_id']; chain = h['session_id']
        if e['event'] == 'on_request_finished':
            modes[chain] = 'final' if h['final'] else 'waiting'
            if h['final']:
                final_messages[chain] = msg
            row = clients[msg]
            positions = [p for group in e['position_groups'] for p in group]
            full = row['prompt_len'] // 16
            if len(positions) != full or not positions:
                prefill = lifecycle[e['request_id']]
                category, missing = classify(full, positions, prefill['gpu_hit'])
                keys = [known.get((instances[msg], labels[msg][p // 16 - 1])) for p in sorted(missing)]
                assert all(k is not None for k in keys), 'Do not invent unresolved keys'
                earlier = sum(k in prior[chain] for k in keys)
                eventual = sum(k in ever[chain] for k in keys)
                finished_refs = [ref for g in e['ref_count_groups'] for ref in g]
                rows.append(dict(request_id=msg, server_request_id=e['request_id'],
                    instance_id=row['instance_id'], chain_id=row['chain_id'], chain_turn=row['chain_turn'],
                    final=h['final'], category=category, prompt_len=row['prompt_len'], full_prompt_blocks=full,
                    finished_keys=len(positions), missing_blocks=len(missing), gpu_hit=prefill['gpu_hit'],
                    cpu_hit=prefill['cpu_hit'], recomputed=prefill['recomputed'],
                    missing_positions=sorted(missing), missing_keys=keys,
                    missing_keys_seen_in_prior_same_chain_callbacks=earlier,
                    missing_keys_seen_in_any_same_chain_callbacks=eventual,
                    missing_keys_never_seen_in_same_chain_callbacks=len(keys)-eventual,
                    store_miss_calls=sum(x['event']=='on_store_miss' for x in by_request[msg]),
                    insertion_only_count=e['insertion_only_count'], reused_count=e['reused_count'],
                    finalized_ready_resident=sum(ref is not None and ref >= 0 for ref in finished_refs),
                    cpu_residency_of_omitted_keys='not_observable_from_original_scalar_log',
                    registration_scope='hint-bearing callback arguments, not an LRU-owned chain registry'))
                if not h['final']:
                    for k in keys:
                        if k in prior[chain]:
                            continue
                        owners = [c for c in prior if k in prior[c]]
                        finals = [c for c in owners if modes.get(c) == 'final']
                        if finals:
                            witnesses.append(dict(waiting_request=msg, key=k,
                                final_request=final_messages[finals[0]],
                                all_observed_previous_chains_final=all(modes.get(c)=='final' for c in owners)))
                            break
        else:
            modes[chain] = 'active'
        prior[chain].update(e['_resolved_keys'])

    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), output / 'per-request.parquet')
    write_json(output / 'per-request.json', rows)
    # This is a test of forwarding semantics using real identities/keys, not a
    # claim that the original LRU trace contained HintAware eviction decisions.
    witness = next(w for w in witnesses if w['all_observed_previous_chains_final'])
    def envelope(msg):
        row = clients[msg]
        return make_envelope(row, 'perfect', msg, f"{row['root_id']}#{row['instance_id']}")
    example = dict(witness=witness, scope='minimal CPU counterexample with synthetic control chain; not full trace replay',
        final_hint=envelope(witness['final_request']), waiting_hint=envelope(witness['waiting_request']))
    example['decisions'] = final_sharing_counterexample(example['final_hint'], example['waiting_hint'], bytes.fromhex(witness['key']))
    write_json(output / 'counterexample.json', example)
    counts = Counter(r['category'] for r in rows)
    summary = dict(git_commit=commit, measurement_git_commit=run['git_commit'],
        source_sha256={str(p): sha256_file(p) for p in [workload, probe/'run.json', probe/'request-metadata.parquet',
            *sorted((probe/'policy-observations').glob('*.gz')), *sorted((probe/'policy-observations').glob('*.bin')),
            *sorted((probe/'server-observations').glob('*.jsonl'))]},
        categories=dict(counts), incomplete_requests=sum(r['missing_blocks']>0 for r in rows),
        empty_incomplete_requests=sum(r['missing_blocks']>0 and r['finished_keys']==0 for r in rows),
        empty_requests=sum(r['finished_keys']==0 for r in rows),
        omitted_blocks=sum(r['missing_blocks'] for r in rows),
        previously_observed_in_same_chain_requests=sum(r['missing_blocks']>0 and r['missing_keys_seen_in_prior_same_chain_callbacks']==r['missing_blocks'] for r in rows),
        eventually_observed_in_same_chain_requests=sum(r['missing_blocks']>0 and r['missing_keys_seen_in_any_same_chain_callbacks']==r['missing_blocks'] for r in rows),
        never_observed_in_same_chain_blocks=sum(r['missing_keys_never_seen_in_same_chain_callbacks'] for r in rows),
        final_sharing_witnesses=witnesses, stop_before_adapter=True,
        frozen_policy_sha256=sha256_file(ROOT/'agentkv/policy/hint_aware.py'),
        frozen_config_sha256=sha256_file(ROOT/'docs/m5_scoring_frozen.json'),
        correctness_risk='GPU-hit shared prefix can retain final-only ownership despite a waiting reuser; omission changes eviction order',
        limitations=['Original scalar records contain time/method only, no key, ref_cnt or protected set.',
            'on_store_miss/touch log positions but not keys; only finished log keys; hints contain a summary, not the complete envelope.',
            'Prefix keys were resolved by within-instance prefix equality against actual recorded keys; original bytes unchanged.',
            'Prior/ever callback observation is not proof of CPU residency or actual HintAware ownership. Original policy was LRU.',
            'Exact full original-sequence replay and CPU residency at omitted-key times cannot be recovered from these logs.'])
    write_json(output / 'run.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','final_sharing_witnesses')}, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-dir', type=Path, default=ROOT/'results/m5-vllm-callback-probe-20261104')
    p.add_argument('--workload', type=Path, default=ROOT/'results/m1-workload-seed0-v3-16k/workload.parquet')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    diagnose(a.probe_dir, a.workload, a.output_dir)


if __name__ == '__main__':
    main()
