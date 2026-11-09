import gzip
import json
import pytest
from agentkv.bench.record_online_callbacks import validate_trace


def test_trace_footer_and_callback_counts_are_required(tmp_path):
    path = tmp_path / 'trace.gz'
    events = [dict(event='open', policy='TracedHintAwarePolicy'),
              dict(event='begin', method='evict'),
              dict(event='close', counts={'evict': 1}, calls=1)]
    def write(rows):
        with gzip.open(path, 'wt') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
    write(events)
    assert validate_trace([path]) == 1
    write(events[:-1])
    with pytest.raises(AssertionError, match='footer'):
        validate_trace([path])
    write(events[:-1] + [dict(event='error', error='callback failed')])
    with pytest.raises(AssertionError, match='exception'):
        validate_trace([path])
