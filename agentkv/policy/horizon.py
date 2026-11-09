"""Conditional empirical return probability; standard-library runtime only."""
import bisect
import json
import math
from pathlib import Path
from agentkv.policy.remaining_life import RemainingLifeTable

DEFAULT_HORIZON_TABLE=Path(__file__).with_name('data')/'horizon_probability_v1.json'


class HorizonProbabilityTable(RemainingLifeTable):
    def __init__(self,model,tier='online'):
        if model['format']!='agentkv.horizon_probability.v1':raise ValueError('Unsupported probability table')
        data=model['tiers'][tier];self.tier=tier
        self.cuts=tuple(data['prediction_cuts_s']);self.grid=tuple(model['elapsed_grid_s'])
        self.support=tuple(tuple(b['support_s']) for b in data['buckets'])
        self.cumulative=tuple(tuple(b['cumulative_counts']) for b in data['buckets'])
        if len(self.support)!=len(self.cuts)+1:raise ValueError('Invalid bucket count')
        if not self.grid or self.grid[0]!=0 or any(a>=b for a,b in zip(self.grid,self.grid[1:])):raise ValueError('Invalid grid')
        if any(not math.isfinite(v) or v<0 for v in (*self.grid,*self.cuts)):raise ValueError('Invalid grid/cuts')
        if any(a>=b for a,b in zip(self.cuts,self.cuts[1:])):raise ValueError('Invalid cuts')
        for support,counts in zip(self.support,self.cumulative):
            if not support or len(support)!=len(counts):raise ValueError('Invalid CDF shape')
            if any(not math.isfinite(x) or x<0 for x in support) or any(a>=b for a,b in zip(support,support[1:])):raise ValueError('Invalid support')
            if any(isinstance(n,bool) or not isinstance(n,int) or n<=0 for n in counts) or any(a>=b for a,b in zip(counts,counts[1:])):raise ValueError('Invalid counts')

    @classmethod
    def load(cls,path=DEFAULT_HORIZON_TABLE,tier='online'):
        return cls(json.loads(Path(path).read_text()),tier)

    def count_at_most(self,bucket,x):
        i=bisect.bisect_right(self.support[bucket],x)
        return self.cumulative[bucket][i-1] if i else 0

    def survivors(self,bucket,elapsed):
        return self.cumulative[bucket][-1]-self.count_at_most(bucket,elapsed)

    def probability(self,bucket,elapsed,horizon):
        if any(not math.isfinite(v) or v<0 for v in (elapsed,horizon)):raise ValueError('Invalid elapsed/horizon')
        before=self.count_at_most(bucket,elapsed);live=self.cumulative[bucket][-1]-before
        return (self.count_at_most(bucket,elapsed+horizon)-before)/live if live else 0.
