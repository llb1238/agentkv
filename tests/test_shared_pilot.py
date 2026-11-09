"""No-GPU checks of the exact pilot resource contract and monitoring columns."""
import json
from pathlib import Path
from agentkv.bench.shared_pilot import server_arguments, parse_dmon


def test_exact_pilot_kv_capacity_and_slots():
    args=server_arguments(Path('/tmp/vllm'),8012)
    assert args[args.index('--shutdown-timeout')+1]=='120'
    assert args[args.index('--num-gpu-blocks-override')+1]=='11430'
    assert args[args.index('--kv-cache-memory-bytes')+1]=='26966753280'
    assert args[args.index('--max-model-len')+1]=='16384'
    assert args[args.index('--max-num-seqs')+1]=='8'
    config=json.loads(args[args.index('--kv-transfer-config')+1])
    assert config['kv_connector_extra_config']['cpu_bytes_to_use']==40450129920
    assert config['kv_connector_extra_config']['eviction_policy']=='lru'


def test_dmon_parser_preserves_both_gpus_and_utilization_columns():
    text='''#Date Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 ccpm
 20261103 09:00:00 0 99 44 45 12 5 0 0 0 0 3201 1785 133902 133902 0
 20261103 09:00:00 1 102 45 43 80 40 0 0 0 0 3201 1785 79043 79044 0
'''
    rows=parse_dmon(text)
    assert [r['gpu'] for r in rows]==[0,1]
    assert [r['sm'] for r in rows]==[12,80]
    assert rows[1]['fb_mib']==79043
    assert rows[1]['wall_time_s']==rows[0]['wall_time_s']


def test_gpu0_uses_local_numa_node():
    args=server_arguments(Path('/tmp/vllm'),8012,gpu_index=0)
    assert '--cpunodebind=0' in args and '--preferred=0' in args


def test_graceful_server_shutdown_only_signals_api_process(monkeypatch):
    from agentkv.bench import shared_pilot
    import signal
    calls=[]
    class Process:
        pid=123;args=['owned-server'];done=False
        def send_signal(self,sig):calls.append(('api',sig))
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):self.done=True;calls.append(('wait',timeout))
    monkeypatch.setattr(shared_pilot.os,'killpg',lambda *args:calls.append(('group',args)))
    shared_pilot.stop_owned(Process(),grace_s=135,signal_group=False)
    assert calls[0]==('api',signal.SIGTERM)
    assert not any(c[0]=='group' for c in calls)
