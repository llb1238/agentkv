"""Audit and publish completed E1 data; never starts a server or changes scoring."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from agentkv.analysis.e1_gpu import read, verify_result
from agentkv.bench.e1_gpu import ROOT, PROTOCOL, verify_protocol
from agentkv.bench.shared_pilot import parse_dmon
from agentkv.bench.trace_loader import sha256_file
from agentkv.provenance import current_commit


def ignored_payloads(directory, names):
    """Retain the immutable full prompts at the original persistent location."""
    return [name for name in names if name in ('controller.pid', 'controller.lock')
            or (name == 'requests.parquet' and Path(directory).name.startswith('attempt-'))]


def primary_values(rows):
    kept = [r for r in rows if r['in_window'] and r['status'] == 'success']
    denominator = sum(r['prompt_len'] for r in kept)
    times = [r['ttft_ms'] for r in kept]
    return dict(requests=len(kept), effective_recompute=(sum(r['prefill_forward_token_work'] for r in kept)
                / denominator if denominator else None),
                p50=float(np.quantile(times, .5)) if times else None,
                p99=float(np.quantile(times, .99)) if times else None)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def queue_figure(destination, state, rate=1.4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'lru': '#0072B2', 'lru-final': '#009E73', 'online': '#D55E00', 'perfect': '#CC79A7'}
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.2), sharex=True)
    for seed, ax in zip((3, 4, 5), axes):
        for label, color in colors.items():
            item = next((r for r in state['runs'] if r['rate'] == rate and r['seed'] == seed and r['policy'] == label),None)
            if item is None or item['status']!='complete':continue
            rows = [json.loads(s) for s in (destination / item['selected_attempt'] / 'metrics.jsonl').read_text().splitlines()]
            rows = [r for r in rows if 'vllm:num_requests_waiting' in r.get('values', {})]
            ax.plot([r['elapsed_s'] / 60 for r in rows],
                    [r['values']['vllm:num_requests_waiting'] for r in rows], color=color, lw=.65, label=label)
        ax.axvspan(0, 10, color='gray', alpha=.15)
        ax.set(ylabel=f'Seed {seed}: waiting')
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=4, fontsize=7)
    axes[-1].set(xlabel='Elapsed minutes', xlim=(0, 40))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(destination / (f'lambda{rate:g}_queues'.replace('.','')+f'.{ext}'), dpi=300)
    plt.close(fig)


def compare_versions(destination,state,protocol):
    """Report both versions, including unstable populations, without replacing v1."""
    if protocol.get('format')!='agentkv.e1_gpu.v2':return None
    previous=ROOT/protocol['prior_result_directory']
    assert sha256_file(previous/'run.json')==protocol['prior_result_sha256']
    old=read(previous/'summary.json');new=read(destination/'summary.json')
    prior={(r['rate'],r['seed'],r['policy']):r for r in old['rows']}
    rows=[];checks=[]
    for item in state['runs']:
        left=prior[item['rate'],item['seed'],item['policy']]
        right=next(r for r in new['rows'] if r['id']==item['id'])
        row={k:item[k] for k in ('rate','seed','policy')}
        for version,r in [('v1',left),('v2',right)]:
            for field in ('stability','completed','completed_non_entry','effective_recompute','ttft_p50_ms','ttft_p99_ms',
                          'paired_eligible','paired_recompute_pp','paired_ttft_p50_ms','paired_ttft_p99_ms'):
                row[version+'_'+field]=r.get(field)
        rows.append(row)
        before=next(i for i in read(previous/'checkpoint.json')['runs'] if i['id']==item['id'])
        columns=['request_id','sent_prompt_sha256','permutation_seed','prompt_len','max_tokens']
        def tokens(path):
            return {r['request_id']:tuple(r[c] for c in columns[1:]) for r in pq.read_table(path,columns=columns).to_pylist()}
        a=tokens(previous/before['selected_attempt']/'request-metadata.parquet')
        b=tokens(destination/item['selected_attempt']/'request-metadata.parquet')
        common=a.keys()&b.keys()
        assert common and all(a[k]==b[k] for k in common),'v1/v2 common logical tokens differ'
        checks.append(dict(id=item['id'],requests=len(common),identical=True))
    write_csv(destination/'v1_v2_populations.csv',rows)
    record=dict(prior_result_sha256=protocol['prior_result_sha256'],prior_summary_sha256=sha256_file(previous/'summary.json'),
                common_token_checks=checks,rows=rows)
    (destination/'v1_v2_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
    def fmt(v,scale=1):
        return 'NA' if not v['n'] else f"{v['mean']*scale:.3f} [{v['minimum']*scale:.3f}, {v['maximum']*scale:.3f}], n={v['n']}"
    lines=['','## E1 v1 / v2 并列','',
           'v2仅改变实现性能：链级评分、链内冻结rank。online/perfect及734/2005次基线的有序驱逐逐次相同；公式和校准不变。真实闭环完成时刻与交错可能不同，这不是同一时刻的A/B实验。所有共同请求的token在版本间逐项相同。',
           '','主指标仅稳定运行，配对还要求同人口LRU稳定；均值[最小,最大]，注明样本数，不做显著性检验。负差表示较同版本LRU更低。',
           '', '| λ | 版本 | 策略 | 稳定/3 | 重算% | TTFT P50 ms | TTFT P99 ms | 配对Δ重算pp | 配对ΔP50 ms | 配对ΔP99 ms |',
           '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for rate in (1.,1.4):
        for label in protocol['policies']:
            for version,report in [('v1',old),('v2',new)]:
                a=next(a for a in report['aggregates'] if a['rate']==rate and a['policy']==label)
                fields=[fmt(a['effective_recompute'],100),fmt(a['ttft_p50_ms']),fmt(a['ttft_p99_ms']),
                        fmt(a['paired_recompute_pp']),fmt(a['paired_ttft_p50_ms']),fmt(a['paired_ttft_p99_ms'])]
                lines.append(f"| {rate:g} | {version} | {label} | {a['stable_count']}/3 | "+' | '.join(fields)+' |')
    lines+=['','[逐人口v1/v2原值、稳定性及完成数](v1_v2_populations.csv)。不稳定项全部保留，不能用缺失值表示无差异。','',
        '复跑：`python -m agentkv.bench.e1_gpu run --protocol docs/e1_v2_preregistration.json --output-dir <新的持久目录> --gpu-index 1 --port 8016`。断点续跑使用原持久目录，不覆盖已有attempt。']
    with (destination/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    return record


def publish(source, destination, protocol_path=None):
    source, destination = source.resolve(), destination.resolve()
    assert not destination.is_relative_to(source) and source != destination
    protocol_path=PROTOCOL if protocol_path is None else protocol_path.resolve()
    code, registration, protocol = verify_protocol(protocol_path)
    state = read(source / 'checkpoint.json')
    total=sum(len(g['order']) for g in protocol['groups'])
    assert state['status'] == 'complete' and len(state['runs']) == total
    assert state['preregistration_commit'] == registration
    assert state['protocol_sha256'] == sha256_file(protocol_path)
    expected = [g['name'] + '/' + p for g in protocol['groups'] for p in g['order']]
    assert [r['id'] for r in state['runs']] == expected
    forecast = ROOT / protocol['simulation_directory']
    assert sha256_file(forecast / 'run.json') == protocol['simulation_run_sha256']
    for case in protocol['predictions']:
        assert sha256_file(forecast / case['group'] / case['policy'] / 'requests.parquet') == case['requests_sha256']

    audit, stability, categories, monitor, raw = [], [], [], [], []
    for item in state['runs']:
        assert item['status'] == 'complete'
        directory = source / item['selected_attempt']
        result = read(directory / 'e1-analysis.json')
        assert result == item['result'] and sha256_file(directory / 'e1-analysis.json') == item['analysis_sha256']
        verify_result(directory, result)
        run = read(directory / 'run.json')
        assert run['git_commit'] == state['git_commit'] == result['git_commit']
        assert run['vllm_commit'] == result['vllm_commit'] == protocol['backend_commit']
        if protocol['format']=='agentkv.e1_gpu.v1':
            assert len(item['attempts']) == 2 and item['attempts'][0]['status'] == 'lost'
            assert item['attempts'][1]['status'] == 'complete' and item['attempts'][1]['number'] == 2
        else:
            assert 1<=len(item['attempts'])<=2 and item['attempts'][-1]['status']=='complete'
            assert all(a['status'] in ('failed','interrupted') and a['kind']=='non_strategy_interruption'
                       for a in item['attempts'][:-1])
        values = primary_values(pq.read_table(directory / 'request-metadata.parquet', columns=[
            'in_window', 'status', 'prompt_len', 'prefill_forward_token_work', 'ttft_ms']).to_pylist())
        assert values['requests'] == result['window_success']
        assert math.isclose(values['effective_recompute'], result['effective_recompute'], abs_tol=1e-15)
        for key in ('p50', 'p99'):
            assert math.isclose(values[key], result['ttft_ms'][key], abs_tol=1e-10)
        assert result['hint_counts']['match']
        identity = dict(rate=item['rate'], seed=item['seed'], policy=item['policy'])
        v = result['stability']; d = v['details']
        stability.append(dict(identity, status=v['status'],
            zero_wait_fraction=d['queue']['zero_fraction'], waiting_p99=d['queue']['p99'],
            fitted_increase_30min=d['regression']['fitted_increase_30min'], ttft_half_ratio=d['ttft']['ratio'],
            kv_blocking_events=d['kv']['blocking_events'], kv_below_threshold_events=d['kv']['below_threshold_events']))
        c = result['recomputation']
        categories.append(dict(identity, stable=v['stable'],
            unavoidable=c['source_fractions']['unavoidable'], gap_eviction=c['source_fractions']['gap_eviction'],
            other_eviction=c['source_fractions']['other_eviction'],
            nominal_block_recomputed=c['recomputed_fraction'], effective_recomputed=result['effective_recompute'],
            tail_tokens=c['tail_tokens'], replay_tokens=result['replay_overlap_tokens'],
            repeated_tokens=result['repeated_prompt_work'], gap_requests=c['gap_requests'],
            gap_mean_queue_s=c['gap_requests_mean_queued_s'], window_failed=result['window_failed']))
        server = read(directory / 'server-manifest.json')
        clock = server['clock_pair']
        origin = clock['wall_time_s'] + (run['origin_ns'] - clock['monotonic_ns']) / 1e9
        samples = [r for r in parse_dmon((directory / 'dmon.log').read_text())
                   if 600 <= r['wall_time_s'] - origin <= 2400]
        assert {r['gpu'] for r in samples} == {0, 1}
        for gpu in (0, 1):
            values = [r for r in samples if r['gpu'] == gpu]
            monitor.append(dict(identity, gpu=gpu, samples=len(values),
                mean_sm_percent=float(np.mean([r['sm'] for r in values])),
                max_sm_percent=max(r['sm'] for r in values),
                mean_fb_mib=float(np.mean([r['fb_mib'] for r in values]))))
        raw.append(dict(id=item['id'], path=str(directory / 'requests.parquet'),
                        sha256=result['inputs_sha256']['raw_requests']))
        audit.append(dict(id=item['id'], analysis_sha256=item['analysis_sha256'],
                          stability=v['status'], primary_recomputed=result['effective_recompute'],
                          primary_values_recalculated=True, producer_fingerprints_verified=True))

    # Existing renderer checks common logical requests' token hashes across all policies.
    report = read(source / 'summary.json')
    assert report['status'] == 'complete' and len(report['common_token_checks']) == sum(len(g['order'])-1 for g in protocol['groups'])
    assert all(c['identical'] and c['requests'] > 0 for c in report['common_token_checks'])
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignored_payloads)
    for item in state['runs']:
        verify_result(destination / item['selected_attempt'], item['result'])
        assert not (destination / item['selected_attempt'] / 'requests.parquet').exists()
    write_csv(destination / 'stability.csv', stability)
    write_csv(destination / 'recomputation_sources.csv', categories)
    write_csv(destination / 'dmon_summary.csv', monitor)
    queue_figure(destination, state, protocol.get('queue_rate',1.4))
    comparison=compare_versions(destination,state,protocol)
    record = dict(git_commit=code, status='complete', measurement_commit=state['git_commit'],
        preregistration_commit=registration, vllm_commit=protocol['backend_commit'],
        protocol_sha256=sha256_file(protocol_path), source_checkpoint_sha256=sha256_file(source / 'checkpoint.json'),
        cpu_prediction_sha256=protocol['simulation_run_sha256'],
        frozen_runtime_sources_verified=True, runs=audit, common_token_checks=report['common_token_checks'],
        original_prompt_payloads=raw,
        version_comparison_verified=comparison is not None,
        published_sha256={str(p.relative_to(destination)): sha256_file(p)
            for p in sorted(destination.rglob('*')) if p.is_file()
            and p.name not in ('README.md', 'run.json')})
    (destination / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(git_commit=current_commit(), audited=len(audit),
                         stable=sum(r['stability'] == 'stable' for r in audit), destination=str(destination))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = p.parse_args()
    publish(args.source, args.destination, args.protocol)


if __name__ == '__main__':
    main()
