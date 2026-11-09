"""CPU-only prefix capacity bounds and steady-state KV provisioning estimates."""

import argparse
import bisect
import csv
import heapq
import json
import math
import platform
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .estimate_replay import simulate_requests, write_schedule_estimate
from agentkv.provenance import git_commit
from .steady_state import RootSchedule, sample_population
from .synth import common_prefix
from .trace_loader import sha256_file
from .workload import ROOT, SCHEMA


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(p50=float(np.median(a)), p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def prefix(a, b):
    return common_prefix(np.frombuffer(a, dtype='<i4'), np.frombuffer(b, dtype='<i4'))


class PrefixSet:
    """Multiset of token sequences; exact trie node count via lexicographic neighbors."""
    def __init__(self):
        self.keys = []
        self.tokens = 0

    def add(self, value):
        i = bisect.bisect_left(self.keys, value)
        left = self.keys[i-1] if i else b''
        right = self.keys[i] if i < len(self.keys) else b''
        self.tokens += len(value)//4 - prefix(left, value) - prefix(value, right) + prefix(left, right)
        self.keys.insert(i, value)

    def remove(self, value):
        i = bisect.bisect_left(self.keys, value)
        if i == len(self.keys) or self.keys[i] != value:
            raise ValueError('Missing prefix multiset member')
        left = self.keys[i-1] if i else b''
        right = self.keys[i+1] if i+1 < len(self.keys) else b''
        self.tokens -= len(value)//4 - prefix(left, value) - prefix(value, right) + prefix(left, right)
        self.keys.pop(i)

    def best(self, value):
        i = bisect.bisect_left(self.keys, value)
        return max(prefix(self.keys[i-1], value) if i else 0,
                   prefix(value, self.keys[i]) if i < len(self.keys) else 0)


def root_profile(rows, sends):
    """Strictly earlier sends only; equal-time requests cannot hit each other."""
    ordered = sorted(rows, key=lambda r: (sends[r['trace_req_idx']], r['trace_req_idx']))
    seen = PrefixSet()
    hits = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and sends[ordered[j]['trace_req_idx']] == sends[ordered[i]['trace_req_idx']]:
            j += 1
        for r in ordered[i:j]:
            hits[r['trace_req_idx']] = seen.best(r['_tokens'])
        for r in ordered[i:j]:
            seen.add(r['_tokens'])
        i = j
    # A conservative working-set definition: distinct prompt prefixes of the
    # last sent request on every live chain. Drop a chain at its final end.
    events = []
    for r in ordered:
        t = sends[r['trace_req_idx']]
        events.append((t, 1, r['trace_req_idx'], r))
        if r['final']:
            events.append((t + .5, 0, r['trace_req_idx'], r))
    current, live, times, sizes = {}, PrefixSet(), [], []
    for t, kind, _, r in sorted(events):
        chain = r['chain_id']
        if chain in current:
            live.remove(current.pop(chain))
        if kind:
            current[chain] = r['_tokens']
            live.add(r['_tokens'])
        times.append(t)
        sizes.append(live.tokens)
    return hits, dict(times=np.asarray(times), tokens=np.asarray(sizes))


def load_profiles(path):
    by_root = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, use_threads=False):
        plain = batch.drop_columns(['prompt_token_ids']).to_pylist()
        tokens = batch.column(batch.schema.get_field_index('prompt_token_ids'))
        for i, row in enumerate(plain):
            row['_tokens'] = tokens[i].values.to_numpy().astype('<i4', copy=False).tobytes()
            by_root[row['root_id']].append(row)
    all_rows = [r for rows in by_root.values() for r in rows]
    sends = simulate_requests(all_rows, .5)
    profiles, roots, request_stats = {}, {}, []
    for rid, rows in sorted(by_root.items()):
        hits, profiles[rid] = root_profile(rows, sends)
        end = max(sends[r['trace_req_idx']] + .5 for r in rows)
        estimate = [dict(root_id=rid, chain_id=r['chain_id'], chain_turn=r['chain_turn'],
                         trace_req_idx=r['trace_req_idx'], est_send_s=sends[r['trace_req_idx']], est_duration_s=end) for r in rows]
        metadata = [{k:v for k,v in r.items() if k != '_tokens'} for r in rows]
        roots[rid] = RootSchedule(metadata, estimate)
        request_stats.extend(dict(root_id=rid, trace_req_idx=r['trace_req_idx'], prompt_len=r['prompt_len'],
                                  max_tokens=r['max_tokens'], est_send_s=sends[r['trace_req_idx']],
                                  unlimited_hit_tokens=hits[r['trace_req_idx']]) for r in rows)
    return roots, profiles, request_stats


def working_set(instances, profiles, samples):
    totals = np.zeros(len(samples), dtype=np.int64)
    for ins in instances:
        p = profiles[ins.root_id]
        progress = samples - ins.start_s + ins.tau_s
        indexes = np.searchsorted(p['times'], progress, side='right') - 1
        valid = (samples >= ins.start_s) & (indexes >= 0)
        totals[valid] += p['tokens'][indexes[valid]]
    return totals


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), q * sum(w), side='left')])


def running_kv(roots, instances, slots, warmup=600, duration=2400, service=.5, block=16):
    """FCFS slot queue, original event dependencies, constant service, full KV reservation."""
    nodes, events, queue = {}, [], deque()
    counter = 0
    def push(t, kind, key):
        nonlocal counter
        heapq.heappush(events, (t, kind, counter, key))
        counter += 1
    for ins in instances:
        for e in roots[ins.root_id].entries(ins.tau_s):
            key = ins.instance_id, e.request['trace_req_idx']
            nodes[key] = dict(entry=e, children=[])
        for e in roots[ins.root_id].entries(ins.tau_s):
            key = ins.instance_id, e.request['trace_req_idx']
            if e.timed:
                push(ins.start_s + e.delay_s, 1, key)
            else:
                nodes[ins.instance_id, e.trigger_trace_req_idx]['children'].append(key)
    active, tokens, last = 0, 0, 0.
    counts, weights, sizes = [], [], []
    while events:
        t, kind, _, key = heapq.heappop(events)
        if t > duration:
            break
        elapsed = max(0., t - max(last, warmup))
        if elapsed:
            counts.append(active); sizes.append(tokens); weights.append(elapsed)
        last = t
        node = nodes[key]
        if kind == 0:
            active -= 1; tokens -= node['kv']
            for child in node['children']:
                push(t + nodes[child]['entry'].delay_s, 1, child)
        else:
            queue.append(key)
        while queue and active < slots:
            k = queue.popleft(); n = nodes[k]; r = n['entry'].request
            n['kv'] = block * math.ceil((r['prompt_len'] + r['max_tokens']) / block)
            active += 1; tokens += n['kv']
            push(t + service, 0, k)
    if last < duration:
        counts.append(active); sizes.append(tokens); weights.append(duration - max(last, warmup))
    return dict(kv_tokens_p90=weighted_quantile(sizes, weights, .9),
                running_p90=weighted_quantile(counts, weights, .9),
                mean_running=float(np.average(counts, weights=weights)),
                assumption='FCFS slots; actual dependency completion; s=0.5; full prompt+max_tokens rounded per request to 16')


def saturated_kv(requests, slots, seed, draws=20000):
    lengths = np.array([16*math.ceil((r['prompt_len'] + r['max_tokens'])/16) for r in requests])
    rng = np.random.default_rng(np.random.SeedSequence([seed, slots]))
    sums = lengths[rng.integers(0, len(lengths), size=(draws, slots))].sum(axis=1)
    return dict(**stats(sums), draws=draws,
                assumption='all slots occupied; iid request mix (equal fixed service); conservative full prompt+output allocation, no shared blocks')


def tier_constraints(gpu, cpu, p90, working_set, kv_bytes, host_available, gpu_total_gib, reserve_gib):
    """Check inclusive-tier sizing, returning quantitative violations in tokens."""
    limits = dict(gpu_safety_min=2.5*p90, cpu_working_set_max=.8*working_set,
                  cpu_ratio_min=1.5*gpu, cpu_ratio_max=2*gpu,
                  gpu_memory_max=(gpu_total_gib-reserve_gib)*2**30/kv_bytes,
                  cpu_memory_max=.85*host_available/kv_bytes)
    checks = dict(gpu_safety=gpu>=limits['gpu_safety_min'],
                  gpu_lt_cpu=gpu<cpu, cpu_le_80pct_working_set=cpu<=limits['cpu_working_set_max'],
                  cpu_gpu_ratio_min=cpu>=limits['cpu_ratio_min'],
                  cpu_gpu_ratio_max=cpu<=limits['cpu_ratio_max'],
                  gpu_fits=gpu<=limits['gpu_memory_max'], cpu_fits=cpu<=limits['cpu_memory_max'])
    deficits = dict(gpu_safety=max(0.,limits['gpu_safety_min']-gpu),
                    gpu_lt_cpu=max(0.,gpu-cpu+16),
                    cpu_le_80pct_working_set=max(0.,cpu-limits['cpu_working_set_max']),
                    cpu_gpu_ratio_min=max(0.,limits['cpu_ratio_min']-cpu),
                    cpu_gpu_ratio_max=max(0.,cpu-limits['cpu_ratio_max']),
                    gpu_fits=max(0.,gpu-limits['gpu_memory_max']),
                    cpu_fits=max(0.,cpu-limits['cpu_memory_max']))
    return checks, {k:v for k,v in deficits.items() if not checks[k]}, limits


def memory_info():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable'):
            mem[name + '_bytes'] = int(value.split()[0])*1024
    mem['raw_meminfo'] = Path('/proc/meminfo').read_text()
    return mem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workloads', nargs='+', type=Path, default=[ROOT/'results/m1-workload-seed0-v3/workload.parquet', ROOT/'results/m1-workload-seed0-v3-16k/workload.parquet'])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/m1-capacity-seed0-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--duration', type=float, default=2400)
    parser.add_argument('--warmup', type=float, default=600)
    parser.add_argument('--gpu-total-gib', type=float, default=143771/1024)
    parser.add_argument('--non-kv-gpu-reserve-gib', type=float, default=30)
    parser.add_argument('--model-config',type=Path,default=Path('/data/models/Qwen3-8B/config.json'))
    parser.add_argument('--allow-dirty',action='store_true',help='DEBUG ONLY: bypass clean-worktree check')
    args = parser.parse_args(argv)
    commit = git_commit(args.allow_dirty)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    if (out/'run.json').exists():
        raise FileExistsError('Refusing to overwrite capacity run')
    model_path = args.model_config
    model = json.loads(model_path.read_text())
    kv_bytes = 2*model['num_hidden_layers']*model['num_key_value_heads']*model['head_dim']*2
    report = dict(git_commit=commit,
                  model_config_sha256=sha256_file(model_path), kv_bytes_per_token=kv_bytes,
                  memory=memory_info(), parameters=vars(args) | {'workloads':[str(p) for p in args.workloads], 'output_dir':str(out), 'model_config':str(model_path)},
                  service_time_s=.5, configurations=[], candidates=[], screened_configurations=[],
                  conventions=dict(upper='token-weighted max prompt LCP against strictly earlier sends in same instance, all history retained; no block rounding or causality-of-prefill penalty',
                     throughput='lambda/60 * mean(sum prompt tokens per root) * (1-hit); exact stationary mean, no initialization bias',
                     working_set='distinct prompt-prefix KV of latest sent request of every live chain, no cross-instance sharing; drops at final completion; excludes outputs and is a lower bound',
                     provisioning='2.5*max(saturated,scheduled) KV P90 <= GPU < CPU <= 0.8*working-set P50; 1.5 <= CPU/GPU <=2; CPU/GPU targets 1.5,1.75,2; host reserve 15% MemAvailable',
                     capacity_lines='22k and 34k token/s are user-supplied exclusive compute assumptions, not measured in this run',
                     limits='KV-only CPU footprint; GPU non-KV reserve assumed 30 GiB; vLLM staging/process overhead and CPU bandwidth require pilot validation'))
    curve_rows, all_hits, samples_rows, eligible = [], [], [], []
    for path in args.workloads:
        manifest = json.loads(path.with_name('manifest.json').read_text())
        digest = sha256_file(path)
        if digest != manifest['workload']['sha256']:
            raise ValueError('Workload fingerprint mismatch')
        budget = manifest['parameters']['context_budget']
        print('Loading/profile', budget, flush=True)
        roots, profiles, requests = load_profiles(path)
        schedule_path = out/f'schedule-{budget}.parquet'
        write_schedule_estimate([r for root in roots.values() for r in root.requests], schedule_path, digest)
        total = sum(r['prompt_len'] for r in requests)
        upper = sum(r['unlimited_hit_tokens'] for r in requests)/total
        mean_tokens = total/len(roots)
        all_hits.extend(dict(context_budget=budget, **r) for r in requests)
        saturated = {n:saturated_kv(requests,n,args.seed) for n in (8,16,32)}
        for rate in (.5,.7,1.):
            ins = sample_population({r:root.duration_s for r,root in roots.items()},rate,args.seed,args.duration,True)
            times = np.arange(args.warmup,args.duration,10.)
            ws = working_set(ins,profiles,times)
            base = rate/60*mean_tokens
            runs = {n:running_kv(roots,ins,n,args.warmup,args.duration) for n in (8,16,32)}
            config = dict(context_budget=budget,workload_path=str(path),workload_sha256=digest,
                          schedule_sha256=sha256_file(schedule_path), rate_per_minute=rate,
                          roots=len(roots),requests=len(requests),total_prompt_tokens=total,
                          upper_hit_rate=upper, raw_prefill_tokens_per_second=base,
                          prefill_at_half_upper_tokens_per_second=base*(1-upper/2),
                          working_set_tokens=stats(ws),running={str(k):v for k,v in runs.items()},
                          saturated_kv_tokens={str(k):v for k,v in saturated.items()},
                          M0=sum(i.initial for i in ins),instances=[vars(i) for i in ins])
            report['configurations'].append(config)
            for t, value in zip(times,ws):
                samples_rows.append(dict(context_budget=budget,rate_per_minute=rate,time_s=t,working_set_tokens=int(value)))
            for h in np.linspace(0,upper,201):
                curve_rows.append(dict(context_budget=budget,rate_per_minute=rate,hit_rate=float(h),prefill_tokens_per_second=base*(1-h)))
            for n in (8,16,32):
                p90 = max(saturated[n]['p90'], runs[n]['kv_tokens_p90'])
                gpu = 16*math.ceil(2.5*p90/16)
                for ratio in (1.5,1.75,2.):
                    cpu = 16*math.ceil(ratio*gpu/16)
                    demand = base*(1-upper/2)
                    checks, deficits, limits = tier_constraints(gpu,cpu,p90,float(np.median(ws)),kv_bytes,
                        report['memory']['MemAvailable_bytes'],args.gpu_total_gib,args.non_kv_gpu_reserve_gib)
                    item = dict(context_budget=budget,rate_per_minute=rate,max_num_seqs=n,
                                gpu_tokens=gpu,cpu_tokens=cpu,cpu_gpu_ratio=cpu/gpu,
                                gpu_kv_gib=gpu*kv_bytes/2**30,cpu_kv_gib=cpu*kv_bytes/2**30,
                                scheduled_kv_p90=runs[n]['kv_tokens_p90'],saturated_kv_p90=saturated[n]['p90'],
                                sizing_kv_p90=p90,working_set_p50=float(np.median(ws)),working_set_p90=float(np.quantile(ws,.9)),
                                working_set_gib=float(np.median(ws))*kv_bytes/2**30,
                                working_set_method='stationary population; live-chain latest-prompt prefix union, no cross-instance sharing, no output KV; s=.5, 10s samples in [600,2400), P50 used for constraint',
                                half_upper_prefill_tps=demand,raw_prefill_tps=base,upper_hit_rate=upper,
                                constraints=checks,violations_tokens=deficits,limits=limits,
                                constraint_distance=sum(v/max(gpu,1) for v in deficits.values()))
                    report['screened_configurations'].append(item)
                    if all(checks.values()):
                        eligible.append(item)
        del roots, profiles, requests
    report['candidates']=eligible
    report['all_eligible_candidates']=eligible
    report['closest_rejected']=sorted((c for c in report['screened_configurations'] if c['violations_tokens']),
                                       key=lambda c:c['constraint_distance'])[:6]
    report['recommendations']=[]
    report['recommendation_status']='pending two-tier LRU simulation; capacity screening alone does not recommend GPU pilots'
    report['gpu_pilot_status']='not run; recommend only after measured LRU gap>=10 percentage points and prefill<=15400 token/s; shared GPU stability remains empirical'
    for name, rows in [('prefill_curves.csv',curve_rows),('working_set_samples.csv',samples_rows)]:
        with (out/name).open('w',newline='') as sink:
            writer=csv.DictWriter(sink,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    pq.write_table(pa.Table.from_pylist(all_hits),out/'prefix_upper.parquet',compression='zstd')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budgets = sorted({r['context_budget'] for r in report['configurations']},reverse=True)
    fig,axes=plt.subplots(1,len(budgets),figsize=(7.1,2.8),sharey=True,squeeze=False)
    for ax,budget in zip(axes[0],budgets):
        for rate in (.5,.7,1.):
            rows=[r for r in curve_rows if r['context_budget']==budget and r['rate_per_minute']==rate]
            ax.plot([r['hit_rate']*100 for r in rows],[r['prefill_tokens_per_second']/1000 for r in rows],label=f'{rate:g} roots/min')
        ax.axhline(22,color='gray',ls='--',lw=.8,label='22k token/s')
        ax.axhline(34,color='black',ls=':',lw=.8,label='34k token/s')
        ax.axhline(15.4,color='gray',ls=':',lw=.6,label='70% of 22k')
        upper=next(r['upper_hit_rate'] for r in report['configurations'] if r['context_budget']==budget)
        ax.axvline(upper*50,color='#888888',ls='-.',lw=.7,label='Half upper bound')
        ax.set(title=f'{budget:,} token budget',xlabel='Token-weighted hit rate (%)',ylim=(0,None))
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Required prefill (k token/s)')
    axes[0,-1].legend(fontsize=6,frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out/f'prefill_capacity.{ext}',dpi=300)
    plt.close(fig)
    report['artifacts_sha256']={p.name:sha256_file(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.csv','.parquet','.png','.pdf')}
    report['environment']=dict(python=platform.python_version(),numpy=np.__version__,pyarrow=pa.__version__)
    (out/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(candidates=report['candidates'],memory=report['memory']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
