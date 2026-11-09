"""Audit README numbers and draw one two-panel summary from existing data."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from agentkv.provenance import ROOT, current_commit

LABELS = ('lru', 'online', 'perfect')
V2 = 'e1-gpu-v2-config11-20261106'
LIMIT = 'e1-gpu-stable-limit-12-20261107'
E0 = 'm1-e0-config11-20261103'
CPU = 'm5-scoring-evaluation-20261104'


def read(path):
    return json.loads(path.read_text())


def relative_percent(candidate, baseline):
    """Ratio of population means; deliberately not a mean of paired ratios."""
    return 100 * (mean(candidate) / mean(baseline) - 1)


def homepage_table(data):
    """Recompute the homepage's rounded cells from the audited population means."""
    lines = ['| hint 来源 | 有效重算比例 | TTFT P50 | TTFT P99 |', '|---|---:|---:|---:|']
    for label in LABELS:
        row = data['primary'][label]
        if label == 'lru':
            cells = ['vLLM 内置 LRU（忽略 hint）', f"{100*row['recompute']:.1f}%",
                     f"{row['ttft_p50_ms']:.1f} ms", f"{row['ttft_p99_ms']:,.0f} ms"]
        else:
            title = 'online：按会话历史预测' if label == 'online' else 'perfect：真实间隔对照'
            signed = lambda v, precision: f'{v:+.{precision}f}'.replace('-', '−')
            bold = '**' if label == 'perfect' else ''
            cells = [title, f"{100*row['recompute']:.1f}%（{bold}{signed(row['recompute_relative_percent'],1)}%{bold}）",
                     f"{signed(row['ttft_p50_ms_delta'],1)} ms",
                     f"{signed(row['ttft_p99_ms_delta'],0)} ms（{bold}{signed(row['ttft_p99_ms_relative_percent'],1)}%{bold}）"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def collect(results=ROOT / 'results'):
    v2, limit, e0 = (read(results / folder / filename) for folder, filename in
                     ((V2, 'summary.json'), (LIMIT, 'summary.json'), (E0, 'run.json')))
    primary, populations = {}, {}
    for label in LABELS:
        rows = [r for r in v2['rows'] if r['rate'] == 1 and r['policy'] == label]
        assert len(rows) == 3 and {r['seed'] for r in rows} == {3, 4, 5}
        assert all(r['stable'] for r in rows)
        populations[label] = rows
        primary[label] = dict(
            recompute=mean(r['effective_recompute'] for r in rows),
            predicted=mean(r['predicted'] for r in rows),
            recompute_min=min(r['effective_recompute'] for r in rows),
            recompute_max=max(r['effective_recompute'] for r in rows),
            ttft_p50_ms=mean(r['ttft_p50_ms'] for r in rows),
            ttft_p99_ms=mean(r['ttft_p99_ms'] for r in rows),
            prediction_error_pp=dict(
                mean=mean(100 * (r['effective_recompute'] - r['predicted']) for r in rows),
                minimum=min(100 * (r['effective_recompute'] - r['predicted']) for r in rows),
                maximum=max(100 * (r['effective_recompute'] - r['predicted']) for r in rows)),
        )
    for label, row in primary.items():
        for field, source_field in (('recompute', 'effective_recompute'), ('ttft_p50_ms', 'ttft_p50_ms'),
                                    ('ttft_p99_ms', 'ttft_p99_ms')):
            row[field + '_delta'] = row[field] - primary['lru'][field]
            row[field + '_relative_percent'] = relative_percent(
                [r[source_field] for r in populations[label]], [r[source_field] for r in populations['lru']])
    stability = []
    for rate, source in ((1., v2), (1.2, limit), (1.4, v2)):
        counts = {}
        for label in LABELS:
            rows = [r for r in source['rows'] if r['rate'] == rate and r['policy'] == label]
            assert len(rows) == 3 and {r['seed'] for r in rows} == {3, 4, 5}
            counts[label] = sum(r['stable'] for r in rows)
        stability.append(dict(rate=rate, counts=counts))
    gap_shares = []
    for rate in (.5, .7, 1.):
        rows = [r['real']['recomputation'] for r in e0['runs']
                if r['rate'] == rate and r['real']['stability']['stable']]
        gap = mean(r['source_fractions']['gap_eviction'] for r in rows)
        total = mean(r['recomputed_fraction'] for r in rows)
        gap_shares.append(dict(rate=rate, stable_populations=len(rows), gap_fraction=gap,
                               total_recompute=total, gap_share_total=gap / total))
    completion = {}
    for label in LABELS:
        rows = sorted((r for r in limit['rows'] if r['policy'] == label), key=lambda r: r['seed'])
        completion[label] = dict(mean=mean(r['completed'] for r in rows),
                                 by_seed={str(r['seed']): r['completed'] for r in rows})
    for row in completion.values():
        row['delta_vs_lru'] = row['mean'] - completion['lru']['mean']
        row['relative_percent_vs_lru'] = 100 * (row['mean'] / completion['lru']['mean'] - 1)
    cpu_rows = [read(results / CPU / f'lambda-1-seed-{seed}' / 'run.json')['cases'] for seed in (3, 4, 5)]
    cpu_values = {label: [row[label]['summary']['source_fractions']['recomputed'] for row in cpu_rows]
                  for label in ('lru', 'oracle', 'remaining-online', 'remaining-noisy', 'remaining-perfect')}
    cpu_means = {label: mean(values) for label, values in cpu_values.items()}
    cpu_capture = {}
    for label in ('remaining-online', 'remaining-noisy', 'remaining-perfect'):
        cpu_capture[label] = dict(
            mean_recompute=cpu_means[label],
            ratio_of_means=(cpu_means['lru'] - cpu_means[label]) / (cpu_means['lru'] - cpu_means['oracle']),
            original_mean_of_ratios=mean((lru - hint) / (lru - oracle) for lru, hint, oracle in
                                         zip(cpu_values['lru'], cpu_values[label], cpu_values['oracle'])))
    sources = {}
    for folder, names in ((V2, ('summary.json', 'README.md')), (LIMIT, ('summary.json', 'README.md')),
                          (E0, ('run.json', 'README.md'))):
        for name in names:
            path = results / folder / name
            sources[f'{folder}/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('README.md', *(f'lambda-1-seed-{seed}/run.json' for seed in (3, 4, 5))):
        sources[f'{CPU}/{name}'] = hashlib.sha256((results / CPU / name).read_bytes()).hexdigest()
    return dict(primary=primary, stability=stability, e0_gap_share_total=gap_shares,
                completion=completion, cpu_oracle_capture=cpu_capture, cpu_mean_recompute=cpu_means,
                sources_sha256=sources,
                relative_percent_formula='100 * (mean(policy) / mean(LRU) - 1)',
                e0_fraction_formula='mean(gap fraction) / mean(total recompute fraction); stable populations only')


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={'width_ratios': [1.15, 1]})
    actual = [100 * data['primary'][k]['recompute'] for k in LABELS]
    predicted = [100 * data['primary'][k]['predicted'] for k in LABELS]
    errors = np.array([[100 * (data['primary'][k]['recompute'] - data['primary'][k]['recompute_min']) for k in LABELS],
                       [100 * (data['primary'][k]['recompute_max'] - data['primary'][k]['recompute']) for k in LABELS]])
    left.bar(range(3), actual, width=.58, color='#0072B2', alpha=.8,
             yerr=errors, capsize=3, error_kw={'lw': .8, 'ecolor': '#444444'})
    left.scatter(range(3), predicted, marker='D', c='#D55E00', edgecolors='white', linewidths=.5, s=30, zorder=4)
    for i, value in enumerate(actual):
        top = max(100 * data['primary'][LABELS[i]]['recompute_max'], predicted[i])
        left.text(i, top + .65, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    left.set(xticks=range(3), xticklabels=['LRU', 'online', 'perfect'], ylim=(0, 32),
             ylabel='Recomputed prompt work (%)', title=r'A  Recompute at $\lambda=1$/min')
    left.set_axisbelow(True)
    left.grid(axis='y', alpha=.2)
    left.legend(handles=[Patch(facecolor='#0072B2', alpha=.8, label='Measured mean'),
                         Line2D([], [], marker='D', ls='', color='#D55E00', label='Simulation mean')],
                loc='lower left', frameon=False, fontsize=6.7)
    matrix = np.asarray([[r['counts'][k] for k in LABELS] for r in data['stability']])
    right.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=3, aspect='auto')
    for i in range(3):
        for j in range(3):
            right.text(j, i, f'{matrix[i, j]}/3', ha='center', va='center',
                       color='white' if matrix[i, j] >= 2 else '#333333', fontsize=12, fontweight='bold')
    right.set(xticks=range(3), xticklabels=['LRU', 'online', 'perfect'],
              yticks=range(3), yticklabels=['1', '1.2', '1.4'], ylabel='Root arrivals / min',
              title='B  Stable populations')
    right.set_xticks(np.arange(-.5, 3, 1), minor=True)
    right.set_yticks(np.arange(-.5, 3, 1), minor=True)
    right.grid(which='minor', color='white', linewidth=1.5)
    right.tick_params(which='minor', bottom=False, left=False)
    fig.text(.5, .018, 'Seeds 3/4/5. Bars/dots: population means; whiskers: min–max. '
             '1 and 1.4/min: E1 v2; 1.2/min: boundary study.', ha='center', fontsize=6.1)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for ext in ('png', 'pdf'):
        kwargs = dict(metadata={'CreationDate': None, 'ModDate': None}) if ext == 'pdf' else {}
        fig.savefig(output / f'summary.{ext}', dpi=300, **kwargs)
    plt.close(fig)


def generate(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data = collect()
    (output / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    plot(data, output)
    lines = ['# 最终总结图与草稿数字核验', '', '仅使用既有结果，没有启动GPU或重新选择人口。', '',
             '相对百分比按三个留出人口的策略均值除以LRU均值计算；不是逐人口百分比的平均。', '',
             '| 策略 | 实测重算% | 模拟% | 相对重算% | ΔP50 ms | ΔP99 ms | 相对P99% |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for k, r in data['primary'].items():
        lines.append(f"| {k} | {100*r['recompute']:.6f} | {100*r['predicted']:.6f} | {r['recompute_relative_percent']:+.6f} | "
                     f"{r['ttft_p50_ms_delta']:+.6f} | {r['ttft_p99_ms_delta']:+.6f} | {r['ttft_p99_ms_relative_percent']:+.6f} |")
    lines += ['', '| E0 λ | 稳定人口数 | 间隙/总重算% |', '|---:|---:|---:|']
    for r in data['e0_gap_share_total']:
        lines.append(f"| {r['rate']:g} | {r['stable_populations']} | {100*r['gap_share_total']:.6f} |")
    lines += ['', '| 策略 | v2实测−模拟均值 pp | 最小–最大 pp |', '|---|---:|---:|']
    for k, r in data['primary'].items():
        e = r['prediction_error_pp']
        lines.append(f"| {k} | {e['mean']:.6f} | {e['minimum']:.6f}–{e['maximum']:.6f} |")
    lines += ['', '| λ | LRU稳定 | online稳定 | perfect稳定 |', '|---:|---:|---:|---:|']
    for r in data['stability']:
        lines.append(f"| {r['rate']:g} | " + ' | '.join(f"{r['counts'][k]}/3" for k in LABELS) + ' |')
    lines += ['', '| λ=1.2策略 | 三人口平均完成数 | 较LRU增量 | 相对增量% |', '|---|---:|---:|---:|']
    for k, r in data['completion'].items():
        lines.append(f"| {k} | {r['mean']:.6f} | {r['delta_vs_lru']:+.6f} | {r['relative_percent_vs_lru']:+.6f} |")
    lines += ['', '| CPU λ=1策略 | Oracle捕获率：均值之比% | 原报告逐人口比值均值% |', '|---|---:|---:|']
    for k, r in data['cpu_oracle_capture'].items():
        lines.append(f"| {k} | {100*r['ratio_of_means']:.6f} | {100*r['original_mean_of_ratios']:.6f} |")
    lines += ['', '本次Oracle捕获率 = (mean(LRU) − mean(hint)) / (mean(LRU) − mean(Oracle))；原报告口径保留作对照。',
              f'[CPU原始留出报告](../{CPU}/README.md)。']
    lines += ['', 'E0使用稳定人口；λ=1只有两个稳定人口，不能写成三个。E1相对比例均使用种子3/4/5。', '',
              '校准差距、完成数及稳定次数的全部数值见[summary.json](summary.json)。', '',
              f'[E0来源](../{E0}/README.md) · [E1 v2来源](../{V2}/README.md) · [λ=1.2来源](../{LIMIT}/README.md)', '',
              '![总结图](summary.png)', '', 'PNG为300dpi，另有[PDF](summary.pdf)。', '',
              '```bash', 'python -m agentkv.analysis.summary_figure --output-dir ../agentkv-runs/summary-new', '```']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    record = dict(git_commit=current_commit(), gpu_runs_started=0, sources_sha256=data['sources_sha256'],
                  files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    generate(parser.parse_args().output_dir)


if __name__ == '__main__':
    main()
