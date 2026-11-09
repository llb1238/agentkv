"""Empirical mean residual life lookup, using only the standard library."""
import bisect
import json
import math
from pathlib import Path

DEFAULT_TABLE = Path(__file__).with_name('data') / 'remaining_life_v1.json'


class RemainingLifeTable:
    def __init__(self, model, tier='online'):
        if model['format'] != 'agentkv.remaining_life.v1':
            raise ValueError('Unsupported residual life table')
        data = model['tiers'][tier]
        self.tier = tier
        self.cuts = tuple(data['prediction_cuts_s'])
        self.grid = tuple(model['elapsed_grid_s'])
        self.values = tuple(tuple(b['mean_remaining_s']) for b in data['buckets'])
        if not self.grid or self.grid[0] != 0 or any(not math.isfinite(x) or x < 0 for x in self.grid):
            raise ValueError('Invalid elapsed grid')
        if any(a >= b for a,b in zip(self.grid,self.grid[1:])):
            raise ValueError('Elapsed grid must increase')
        if any(not math.isfinite(x) or x < 0 for x in self.cuts) or any(a >= b for a,b in zip(self.cuts,self.cuts[1:])):
            raise ValueError('Invalid prediction cuts')
        if len(self.values) != len(self.cuts)+1 or any(len(v) != len(self.grid) for v in self.values):
            raise ValueError('Invalid lookup shape')
        if any(v is not None and (not math.isfinite(v) or v <= 0) for row in self.values for v in row):
            raise ValueError('Invalid remaining lifetime')

    @classmethod
    def load(cls, path=DEFAULT_TABLE, tier='online'):
        return cls(json.loads(Path(path).read_text()),tier)

    def bucket(self, prediction_s):
        if not math.isfinite(prediction_s) or prediction_s < 0:
            raise ValueError('Invalid prediction')
        return bisect.bisect_right(self.cuts,prediction_s)

    def lookup(self, bucket, elapsed_s):
        """Left-grid value and next change age; None means no survivors / LRU."""
        if elapsed_s < 0 or not math.isfinite(elapsed_s):
            raise ValueError('Invalid elapsed age')
        i = bisect.bisect_right(self.grid,elapsed_s)-1
        return self.values[bucket][i], self.grid[i+1] if i+1 < len(self.grid) else math.inf
