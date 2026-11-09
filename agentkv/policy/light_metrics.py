"""Process-local histograms and request summaries; one write at shutdown.

No callback opens a file. Full callback traces remain a separate, explicit
debug facility. Quantiles are conservative bucket upper bounds (<0.8% width).
"""
import atexit
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

_STORE = None


def instrumentation_mode():
    return os.environ.get('AGENTKV_INSTRUMENTATION',
        'full' if os.environ.get('AGENTKV_CALLBACK_TRACE_DIR') else 'light')


class Histogram:
    def __init__(self):
        self.buckets = Counter()
        self.count = self.total = 0
        self.minimum = None
        self.maximum = 0

    def add(self, value):
        value = max(0, int(value))
        shift = max(0, value.bit_length()-8)
        self.buckets[(shift << 8) | (value >> shift)] += 1
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @staticmethod
    def bounds(bucket):
        shift, mantissa = bucket >> 8, bucket & 255
        return mantissa << shift, ((mantissa+1) << shift)-1

    def quantile(self, q):
        target = max(1, math.ceil(self.count*q))
        seen = 0
        for bucket, count in sorted(self.buckets.items()):
            seen += count
            if seen >= target:
                return self.bounds(bucket)
        return None

    def export(self):
        return dict(count=self.count, total_ns=self.total, minimum_ns=self.minimum,
            maximum_ns=self.maximum, p50_ns=self.quantile(.5), p99_ns=self.quantile(.99),
            buckets={str(k):v for k,v in self.buckets.items()})


class LightMetrics:
    def __init__(self, directory, *, clock_file=None, pid=None):
        self.directory = Path(directory)
        self.pid = os.getpid() if pid is None else pid
        self.clock_file = Path(clock_file) if clock_file else None
        self.clock_checked = False
        self.window = None
        self.histograms = {}
        self.window_histograms = {}
        self.counters = Counter()
        self.requests = {}
        self.kv_failures = {}
        self.kv_blocking_running = Counter()
        self.closed = False

    def set_window(self, origin_ns, warmup_s, duration_s, cooldown_s=0):
        self.window = (origin_ns+round(warmup_s*1e9),
                       origin_ns+round((duration_s-cooldown_s)*1e9))

    def observe(self, name, duration_ns, stamp_ns=None):
        if self.closed:
            raise RuntimeError('Metrics recorded after final close')
        if name not in self.histograms:self.histograms[name] = Histogram()
        self.histograms[name].add(duration_ns)
        if self.window and self.window[0] <= stamp_ns <= self.window[1]:
            if name not in self.window_histograms:self.window_histograms[name] = Histogram()
            self.window_histograms[name].add(duration_ns)

    def record(self, event, *, ns=None, **fields):
        stamp = time.monotonic_ns() if ns is None else ns
        self.counters[event] += 1
        request_id = fields.get('request_id')
        # The client publishes its origin before sending the first request.
        # Read this small control file once, never in the policy hot path.
        if event == 'request_received' and self.clock_file and not self.clock_checked:
            if not request_id.endswith('_warmup') and self.clock_file.exists():
                self.set_window(**json.loads(self.clock_file.read_text()))
                self.clock_checked = True
        if request_id is not None:
            row = self.requests.setdefault(request_id, {})
            if event in ('request_received', 'request_scheduled', 'request_finished'):
                row.setdefault(event+'_ns', stamp)
            elif event == 'prefill':
                row['prefill'] = dict(ns=stamp, **fields)
            elif event == 'hint_received':
                row['hint_message_id'] = fields['message_id']
                row['hint_ns'] = stamp
        if event == 'kv_allocation_failure':
            if (self.window and self.window[0] <= stamp <= self.window[1]
                    and fields['admission'] and fields['waiting'] > 0):
                self.kv_blocking_running[int(fields['running'])] += 1
            row = self.kv_failures.setdefault(stamp//1_000_000_000,
                dict(count=0, admissions=0, maximum_running=0, maximum_waiting=0))
            row['count'] += 1
            row['admissions'] += bool(fields['admission'])
            row['maximum_running'] = max(row['maximum_running'], fields['running'])
            row['maximum_waiting'] = max(row['maximum_waiting'], fields['waiting'])

    def record_forward(self, request_id, start, end, prompt_tokens, preemptions=0):
        """Successful model-output prompt ranges, union plus repeated work."""
        start=max(0,start);end=min(prompt_tokens,end)
        if end<=start:return
        row=self.requests.setdefault(request_id,{})
        row['forward_prompt_tokens']=prompt_tokens
        row['prefill_observed_preemptions']=max(row.get('prefill_observed_preemptions',0),preemptions)
        row['prefill_forward_token_work']=row.get('prefill_forward_token_work',0)+end-start
        merged=[]
        for left,right in sorted(row.get('forward_ranges',[])+[[start,end]]):
            if merged and left<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],right)
            else:merged.append([left,right])
        row['forward_ranges']=merged

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = dict(format='agentkv.light_metrics.v1', pid=self.pid, closed=True,
            bucket_relative_width_bound=1/128, quantiles='[lower_ns, upper_ns]',
            window_ns=self.window, counters=dict(self.counters), requests=self.requests,
            kv_failures=self.kv_failures,
            kv_blocking_window=dict(window_ns=self.window,observed=bool(self.counters['kv_observer_ready']),
                running_counts=dict(self.kv_blocking_running)),
            histograms={k:v.export() for k,v in self.histograms.items()},
            window_histograms={k:v.export() for k,v in self.window_histograms.items()})
        (self.directory/f'light-{self.pid}.json').write_text(json.dumps(payload)+'\n')


def get_store():
    global _STORE
    directory = os.environ.get('AGENTKV_LIGHT_DIR') or os.environ.get('AGENTKV_OBSERVATION_DIR')
    if instrumentation_mode() != 'light' or not directory:
        return None
    if _STORE is None or _STORE.pid != os.getpid() or _STORE.directory != Path(directory):
        _STORE = LightMetrics(directory, clock_file=os.environ.get('AGENTKV_EXPERIMENT_CLOCK'))
        atexit.register(_STORE.close)
    return _STORE


def close_metrics():
    if _STORE is not None and _STORE.pid == os.getpid():
        _STORE.close()


def read_light_events(directory):
    """Expand aggregates only during offline joining, after server shutdown."""
    events = []
    for path in sorted(Path(directory).glob('light-*.json')):
        data = json.loads(path.read_text())
        if not data['closed']:
            raise ValueError('Unclosed lightweight metrics')
        for identity, row in data['requests'].items():
            for event in ('request_received', 'request_scheduled', 'request_finished'):
                if event+'_ns' in row:
                    events.append(dict(event=event, ns=row[event+'_ns'], request_id=identity))
            if 'prefill' in row:
                events.append(dict(event='prefill', **row['prefill']))
            if 'hint_message_id' in row:
                events.append(dict(event='hint_received', ns=row['hint_ns'], message_id=row['hint_message_id']))
            if 'prefill' in row and 'prefill_forward_token_work' in row:
                p=row['prefill'];length=p['prompt_tokens'];g=p['gpu_hit'];c=p['cpu_hit']
                if row['forward_prompt_tokens']!=length:raise ValueError('Prompt boundary changed')
                def overlap(left,right):
                    return sum(max(0,min(b,right)-max(a,left)) for a,b in row['forward_ranges'])
                unique=overlap(0,length);gpu=g-overlap(0,g);cpu=c-overlap(g,g+c)
                reclassified=length-gpu-cpu-unique
                if reclassified<0:raise ValueError('Non-additive observed prompt ranges')
                # After preemption, previously missing positions may be loaded
                # on a later admission. Forward work is still observed exactly;
                # their device attribution cannot be inferred from first-prefill
                # nominal statistics. Do not discard the actual work or guess.
                if reclassified:gpu=cpu=None
                events.append(dict(event='effective_prefill',request_id=identity,ns=p['ns'],
                    gpu_hit_effective=gpu,cpu_hit_effective=cpu,recomputed_effective=unique,
                    replay_overlap=overlap(0,g+c),prefill_forward_token_work=row['prefill_forward_token_work'],
                    repeated_prompt_work=row['prefill_forward_token_work']-unique,
                    effective_unattributed_cache_tokens=reclassified,
                    effective_ranges_complete=not reclassified or row.get('prefill_observed_preemptions',0)>0,
                    prefill_observed_preemptions=row.get('prefill_observed_preemptions',0)))
    return events
