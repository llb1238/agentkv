import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from agentkv.analysis.release import export
from agentkv.analysis.summary_figure import collect, homepage_table, relative_percent


def test_relative_percent_uses_population_means_not_average_of_ratios():
    assert relative_percent([0, 8], [1, 9]) == pytest.approx(-20)
    assert relative_percent([0, 8], [1, 9]) != pytest.approx((-100 - 100 / 9) / 2)


def test_summary_uses_registered_batches_and_stable_e0_populations():
    data = collect()
    assert [r['counts'] for r in data['stability']] == [
        {'lru': 3, 'online': 3, 'perfect': 3},
        {'lru': 0, 'online': 0, 'perfect': 2},
        {'lru': 0, 'online': 0, 'perfect': 0}]
    assert [r['stable_populations'] for r in data['e0_gap_share_total']] == [3, 3, 2]
    assert data['primary']['online']['recompute_relative_percent'] == pytest.approx(-6.5660329598)
    assert all(r['prediction_error_pp']['maximum'] < 1 for r in data['primary'].values())
    capture = data['cpu_oracle_capture']['remaining-online']
    assert capture['ratio_of_means'] == pytest.approx(.18202744423772876)
    assert capture['ratio_of_means'] != pytest.approx(capture['original_mean_of_ratios'])
    table = homepage_table(data)
    assert '22.0%（−6.6%） | +1.9 ms | −121 ms（−11.3%）' in table
    assert '15.4%（**−34.8%**） | −3.1 ms | −300 ms（**−28.0%**）' in table


def test_release_has_exact_inventory_and_does_not_copy_local_payloads_or_history(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / '.gitignore').write_text('external.bin\n')
    (root / 'README.md').write_text('review')
    (root / 'external.bin').write_bytes(b'private payload')
    (root / 'docs').mkdir()
    (root / 'docs/publication_payloads.json').write_text(json.dumps({'files': [{'path': 'external.bin'}]}))
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                    'commit', '-qm', 'source'], check=True)
    destination = tmp_path / 'release'
    record = export(destination, root)
    assert not (destination / '.git').exists() and not (destination / 'external.bin').exists()
    assert (root / 'external.bin').read_bytes() == b'private payload'
    assert record['total_bytes'] == sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    assert record['total_files'] == len([p for p in destination.rglob('*') if p.is_file()])
    for entry in record['files']:
        assert hashlib.sha256((destination / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    assert 'RELEASE_MANIFEST.json' in (destination / 'FILELIST.tsv').read_text()
    with pytest.raises(FileExistsError):
        export(destination, root)
