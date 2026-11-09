"""vLLM CachePolicy wrapper with fingerprinted configuration and light timings."""
import hashlib
import json
import os
from pathlib import Path
import time

from agentkv.policy.measured_policy import MeasuredPolicy, NoneHintAdapter
from agentkv.policy.light_metrics import get_store, instrumentation_mode
from agentkv.policy.prefix_adapter import PrefixPolicyAdapter, ROOT, frozen_options
from agentkv.policy.optimized_adapter import OptimizedPrefixAdapter
from agentkv.policy.chain_adapter import ChainPrefixAdapter
from agentkv.policy.remaining_life import RemainingLifeTable

CLOCK = time.monotonic


class HintAwareCachePolicy(MeasuredPolicy):
    def __init__(self, cache_capacity):
        super().__init__(cache_capacity)
        path = Path(os.environ['AGENTKV_HINT_POLICY_CONFIG'])
        raw = path.read_bytes()
        config = json.loads(raw)
        if config['format'] != 'agentkv.vllm_hint_policy.v1':
            raise ValueError('Unknown policy config format')
        freeze_raw = (ROOT/'docs/m5_scoring_frozen.json').read_bytes()
        if hashlib.sha256(freeze_raw).hexdigest() != config['freeze_sha256']:
            raise ValueError('Scoring freeze fingerprint mismatch')
        freeze = json.loads(freeze_raw)
        policy_file = ROOT/'agentkv/policy/hint_aware.py'
        if hashlib.sha256(policy_file.read_bytes()).hexdigest() != freeze['source_hashes']['agentkv/policy/hint_aware.py']:
            raise ValueError('Frozen policy source changed')
        tier = config['hint_tier']
        core = config.get('implementation', 'indexed')
        if core not in ('indexed', 'chain'):
            raise ValueError('Unknown policy implementation')
        final_only=config.get('policy_variant')=='lru-final'
        table = None
        if tier in ('noisy', 'online') and not final_only:
            table_path = Path(config['residual_table_path'])
            if hashlib.sha256(table_path.read_bytes()).hexdigest() != freeze['remaining_sha256']:
                raise ValueError('Calibration fingerprint mismatch')
            table = RemainingLifeTable.load(table_path, tier=tier)
        if final_only:
            self.adapter=OptimizedPrefixAdapter(cache_capacity,clock=lambda:CLOCK(),hint_tier=tier,
                final_only=True,options=dict(scoring='lru_final',tail_first=False,final_first=True,timeout_decay=False))
            implementation='LRU-final-without-gap-predictions'
        elif tier == 'none':
            self.adapter = NoneHintAdapter(cache_capacity)
            implementation = 'native-lru-equivalent-none-specialization'
        else:
            adapter = (ChainPrefixAdapter if core == 'chain' else
                PrefixPolicyAdapter if instrumentation_mode() == 'full' else OptimizedPrefixAdapter)
            self.adapter = adapter(cache_capacity, clock=lambda: CLOCK(), hint_tier=tier,
                options=frozen_options(tier, freeze=freeze, table=table))
            implementation = adapter.__name__
        self.metrics = get_store()
        if tier == 'none' and instrumentation_mode() != 'full':
            # Avoid a second mode/config/counting wrapper on every call. All
            # ordinary native methods are inherited directly by NoneHintAdapter.
            self.evict = MeasuredPolicy.evict.__get__(self)
            self.on_request_finished = MeasuredPolicy.on_request_finished.__get__(self)
            if self.metrics is not None:
                self.metrics.counters['valid_hint_requests_any'] = 0
                self.metrics.counters['valid_hint_requests_finished'] = 0
        self.config = config
        self.config_sha256 = hashlib.sha256(raw).hexdigest()
        self.evictions = []  # Populated only by explicit full debug mode.
        print('AGENTKV_HINT_POLICY_CONFIG '+json.dumps(dict(path=str(path), sha256=self.config_sha256,
            config=config, implementation=implementation, instrumentation=instrumentation_mode()),
            sort_keys=True), flush=True)

    def on_request_finished(self, *args):
        result = super().on_request_finished(*args)
        if self.metrics is not None:
            self.metrics.counters['valid_hint_requests_any'] = len(self.adapter.valid_any)
            self.metrics.counters['valid_hint_requests_finished'] = len(self.adapter.valid_finished)
        if instrumentation_mode() == 'full':
            print('AGENTKV_HINT_COUNTS '+json.dumps(dict(valid_any=len(self.adapter.valid_any),
                valid_finished=len(self.adapter.valid_finished))), flush=True)
        return result

    def evict(self, n, protected):
        if instrumentation_mode() != 'full':
            return super().evict(n, protected)
        # Expensive ownership scans and per-call output are debug-only.
        start = time.perf_counter_ns()
        ownership = (self.adapter.ownership() if hasattr(self.adapter,'ownership') else
            dict(cpu_blocks=len(self.chunks),owned_blocks=0,owned_fraction=0.))
        core_start = time.perf_counter_ns()
        result = super().evict(n, protected)
        end = time.perf_counter_ns()
        record = dict(ns=time.monotonic_ns(), n=n, returned=None if result is None else len(result),
            core_us=(end-core_start)/1e3, adapter_us=(end-start)/1e3, **ownership)
        self.evictions.append(record)
        print('AGENTKV_EVICT '+json.dumps(record), flush=True)
        return result
