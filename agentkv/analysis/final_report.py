"""Rebuild the closing report's tables and plots from saved results, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

from agentkv.provenance import ROOT, current_commit


CASES = (
    ('v1', 'e1-gpu-config11-20261105', 'e1_gpu_preregistration.json'),
    ('v2', 'e1-gpu-v2-config11-20261106', 'e1_v2_preregistration.json'),
    ('limit12', 'e1-gpu-stable-limit-12-20261107', 'stable_limit_12_preregistration.json'),
)


def read(path):
    return json.loads(path.read_text())


def link_inputs(output, source, state):
    """Link only the two inputs the renderer reads; never write into a source."""
    for item in state['runs']:
        if item['status'] != 'complete':
            raise ValueError('Closing report requires completed input runs')
        target = output / item['selected_attempt']
        target.mkdir(parents=True)
        for name in ('request-metadata.parquet', 'metrics.jsonl'):
            original = source / item['selected_attempt'] / name
            if not original.is_file():
                raise FileNotFoundError(original)
            (target / name).symlink_to(original.resolve())


def summary_state(source, saved, protocol):
    """Rebuild renderer inputs from per-run analyses, not from saved means."""
    runs, paths = [], []
    for row in saved['rows']:
        if row['status'] != 'complete':
            raise ValueError('Summary reconstruction requires complete analyses')
        candidates = sorted((source / row['id']).glob('attempt-*/e1-analysis.json'))
        matching = [p for p in candidates if read(p)['effective_recompute'] == row['effective_recompute']
                    and read(p)['ttft_ms']['p50'] == row['ttft_p50_ms']
                    and read(p)['ttft_ms']['p99'] == row['ttft_p99_ms']]
        if len(matching) != 1:
            raise ValueError(f'Expected one matching retained analysis: {row["id"]}')
        path = matching[0]
        group = next(g for g in protocol['groups'] if g['rate'] == row['rate'] and g['seed'] == row['seed'])
        runs.append(dict(id=row['id'], rate=row['rate'], seed=row['seed'], policy=row['policy'],
                         group=group['name'], status=row['status'], result=read(path),
                         selected_attempt=path.parent.relative_to(source).as_posix(),
                         attempts=[None] * row['attempts']))  # Only the saved count, not invented attempt metadata.
        paths.append(path)
    return dict(runs=runs, **{k: saved[k] for k in ('status', 'preregistration_commit', 'protocol_sha256')}), paths


def compare_final_summary(output):
    from agentkv.analysis.summary_figure import generate, homepage_table
    generate(output)
    saved_dir = ROOT / 'results/final-summary-20261108'
    actual, expected = read(output / 'summary.json'), read(saved_dir / 'summary.json')
    # Display links change the SHA of source READMEs in a lean snapshot.
    # Compare every other JSON field exactly, with no numerical tolerance.
    keys = set(expected) - {'sources_sha256'}
    assert set(actual) - {'sources_sha256'} == keys
    for key in keys:
        if actual[key] != expected[key]:
            raise AssertionError(f'Final summary JSON differs: {key}')
    tables = lambda path: [s for s in path.read_text().splitlines() if s.startswith('|')]
    assert tables(output / 'README.md') == tables(saved_dir / 'README.md'), 'Final README table differs'
    homepage = homepage_table(actual)
    assert homepage.splitlines() == tables(ROOT / 'README.md'), 'Homepage result table differs'
    (output.parent / 'homepage-table.md').write_text(homepage)
    return dict(json_fields_compared=sorted(keys), exact_json_match=True,
                readme_table_rows_compared=len(tables(saved_dir / 'README.md')), exact_table_match=True,
                homepage_table_exact_match=True, homepage_table_numeric_values=13,
                homepage_table_sha256=hashlib.sha256(homepage.encode()).hexdigest(),
                source_readme_hashes_excluded_from_numeric_comparison=True)


def reproduce(output, *, input_mode='auto'):
    from agentkv.analysis.e1_gpu import render, verify_result
    from agentkv.bench.e0 import render as render_e0

    if output.exists():
        raise FileExistsError('Use a new output directory; saved reports are immutable inputs')
    output = output.resolve()
    if output.is_relative_to(ROOT / 'results') or output == ROOT:
        raise ValueError('Keep regenerated reports outside the saved results tree')
    output.mkdir(parents=True)
    release = ROOT / 'RELEASE_MANIFEST.json'
    summary_only = input_mode == 'summary' or (input_mode == 'auto' and release.exists()
                                               and read(release).get('profile') == 'lean')
    inputs = {}
    e0 = ROOT / 'results/m1-e0-config11-20261103/run.json'
    inputs[str(e0.relative_to(ROOT))] = hashlib.sha256(e0.read_bytes()).hexdigest()
    destination = output / 'e0'
    destination.mkdir()
    render_e0(destination, read(e0))
    comparisons = {}
    for name, directory, registration in CASES:
        source = ROOT / 'results' / directory
        protocol = read(ROOT / 'docs' / registration)
        saved = read(source / 'summary.json')
        if summary_only:
            state, analyses = summary_state(source, saved, protocol)
        else:
            state, analyses = read(source / 'checkpoint.json'), [source / 'checkpoint.json']
            for item in state['runs']:
                verify_result(source / item['selected_attempt'], item['result'])
        for original in (*analyses, source / 'summary.json', ROOT / 'docs' / registration):
            inputs[str(original.relative_to(ROOT))] = hashlib.sha256(original.read_bytes()).hexdigest()
        destination = output / name
        destination.mkdir()
        if not summary_only:
            link_inputs(destination, source, state)
        render(destination, state, protocol, summary_only=summary_only,
               saved_common_checks=saved['common_token_checks'], saved_plots=source)
        actual, expected = read(destination / 'summary.json'), read(source / 'summary.json')
        # Ignore report-generation commit IDs; compare every numeric summary
        # and all common-token checks, without opening original prompt arrays.
        for key in ('rows', 'aggregates', 'common_token_checks'):
            if actual[key] != expected[key]:
                raise AssertionError(f'{name}: saved {key} differs from reconstruction')
        comparisons[name] = dict(runs=len(state['runs']), all_summary_values_identical=True,
                                 raw_evidence_reverified=not summary_only,
                                 common_token_checks='archived only' if summary_only else 'reverified')
    final_comparison = compare_final_summary(output / 'final-summary')
    for original in (ROOT / 'README.md', ROOT / 'results/final-summary-20261108/summary.json',
                     ROOT / 'results/final-summary-20261108/README.md'):
        inputs[str(original.relative_to(ROOT))] = hashlib.sha256(original.read_bytes()).hexdigest()
    result = dict(git_commit=current_commit(), gpu_runs_started=0, inputs_sha256=inputs,
                  comparisons=comparisons, output_uses_read_only_input_references=True,
                  input_mode='summary' if summary_only else 'full', final_summary_comparison=final_comparison)
    (output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-mode', choices=('auto', 'full', 'summary'), default='auto',
                        help='auto uses retained analyses in lean releases, raw verification otherwise')
    args = parser.parse_args()
    reproduce(args.output_dir, input_mode=args.input_mode)


if __name__ == '__main__':
    main()
