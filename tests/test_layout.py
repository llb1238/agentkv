"""Installed namespace, CLI and repository-root checks after consolidation."""
import importlib
import subprocess
import sys
from pathlib import Path
import pytest
from agentkv.bench.workload import ROOT
from agentkv.hints import make_envelope, validate_envelope


def test_repository_root_and_plugin_namespace():
    assert ROOT==Path(__file__).resolve().parents[1]
    for name in ('bench','sim','doc','agentkv/prefetch','agentkv/connector','agentkv/tests'):
        assert not (ROOT/name).exists()
    assert make_envelope.__module__=='agentkv.hints.protocol'
    assert validate_envelope.__module__=='agentkv.hints.protocol'
    metadata=(ROOT/'agentkv/bench/plugins/agentkv_replay_observer-0.0.0.dist-info/entry_points.txt').read_text()
    assert 'agentkv.bench.observability_plugin:install' in metadata


@pytest.mark.parametrize('module',['workload','estimate_replay','estimate_steady_state','replay','capacity',
                                   'fake_server','report_two_tier','scheduler_probe'])
def test_benchmark_cli_help(module):
    result=subprocess.run([sys.executable,'-m',f'agentkv.bench.{module}','--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'usage:' in result.stdout


def test_m3_cli_and_frozen_legacy_imports():
    result=subprocess.run([sys.executable,'-m','agentkv.sim.two_tier','--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    from agentkv.bench.scheduler_probe import load_legacy
    original=ROOT/'results/archive/m1-online-lru-seed0-final/implementation/replay.py'
    if not original.exists() and (ROOT/'RELEASE_MANIFEST.json').exists():
        pytest.skip('Historical source snapshot is in the private evidence archive, not lean release')
    # Historical source bytes retain the original import spelling; loading
    # adapts their package context rather than rewriting fingerprinted files.
    legacy=load_legacy(original)
    assert legacy.ReplayEngine
