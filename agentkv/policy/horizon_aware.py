"""Probability within an observed eviction horizon, without engine imports."""
import bisect
from collections import deque
import heapq
import math
import time
from agentkv.policy.hint_aware import HintAwarePolicy,_get
from agentkv.policy.horizon import HorizonProbabilityTable


class HorizonPolicy(HintAwarePolicy):
    def __init__(self,cache_capacity,*,clock=time.monotonic,tail_first=False,final_first=True,
                 timeout_decay=False,scoring='horizon',probability_table=None,perfect_probability=False,
                 horizon_initial_s=1.,horizon_window=4096,horizon_period_s=1.):
        if scoring!='horizon' or timeout_decay:raise ValueError('Horizon replaces deadline timeout scoring')
        if not math.isfinite(horizon_initial_s) or horizon_initial_s<0:raise ValueError('Invalid initial horizon')
        if not isinstance(horizon_window,int) or isinstance(horizon_window,bool) or horizon_window<1:raise ValueError('Invalid window')
        if not math.isfinite(horizon_period_s) or horizon_period_s<=0:raise ValueError('Invalid refresh period')
        self.initial_horizon=horizon_initial_s;self.horizon_window=horizon_window;self.horizon_period=horizon_period_s
        self.perfect_probability=perfect_probability
        self.probability_table=None if perfect_probability else probability_table or HorizonProbabilityTable.load()
        super().__init__(cache_capacity,clock=clock,tail_first=tail_first,final_first=final_first,scoring='lru_final',timeout_decay=False)
        self.scoring='horizon'

    def clear(self):
        super().clear()
        self.horizon_s=self.initial_horizon;self.residences=deque();self.residence_sum=0.
        self._horizon_origin=None;self._horizon_epoch=0;self._sampling_eviction=False;self._eviction_ages=[]
        self.horizon_trace=[];self._observed_hint=None
        self.stats.update(horizon_updates=0,horizon_samples_seen=0,probability_refreshes=0)

    def _hint(self,context):
        self._observed_hint=super()._hint(context)
        return self._observed_hint

    def _observe(self,keys,context,*,finished=False):
        now=super()._observe(keys,context,finished=finished);hint=self._observed_hint
        if not finished or not hint or hint['final']:return now
        if _get(context,'instance_id') is not None and _get(context,'chain_id') is not None:
            identity=(str(_get(context,'instance_id')),str(_get(context,'chain_id')))
        else:identity=(hint['parent'] or hint['session'],hint['session'])
        c=self.chains.get(identity)
        if c is None or c.request_id!=_get(context,'req_id',id(context)) or c.mode!='waiting':return now
        tag=(c.generation,c.version)
        if getattr(c,'probability_tag',None)==tag:return now
        c.probability_tag=tag
        if not self.perfect_probability:c.probability_bucket=self.probability_table.bucket(hint['gap']/1000)
        self._update_probability(identity,c,now,schedule=True)
        for key in list(c.keys):self._refresh(key,now)
        return now

    def _next_tick(self,now):
        index=math.floor((now-self._horizon_origin)/self.horizon_period)+1
        tick=self._horizon_origin+index*self.horizon_period
        while tick<=now:index+=1;tick=self._horizon_origin+index*self.horizon_period
        return tick

    def _advance(self):
        now=float(self.clock())
        if not math.isfinite(now) or now<self._time:raise ValueError('Clock must be finite and monotonic')
        self._time=now
        if self._horizon_origin is None:
            self._horizon_origin=now
            heapq.heappush(self._events,(self._next_tick(now),next(self._serial),'horizon',None,self._horizon_epoch))
        while self._events and self._events[0][0]<=now:
            _,_,kind,key,version=heapq.heappop(self._events)
            if kind=='horizon':
                if version!=self._horizon_epoch:continue
                self.horizon_s=math.fsum(self.residences)/len(self.residences) if self.residences else self.initial_horizon
                self._horizon_epoch+=1;self.stats['horizon_updates']+=1
                for identity,c in list(self.chains.items()):
                    if c.mode=='waiting':self._update_probability(identity,c,now,schedule=self.perfect_probability)
                for block in list(self.chunks):self._refresh(block,now)
                heapq.heappush(self._events,(self._next_tick(now),next(self._serial),'horizon',None,self._horizon_epoch))
            elif kind=='probability':
                c=self.chains.get(key)
                if c is None or c.mode!='waiting' or version[:2]!=(c.generation,c.version):continue
                if self.perfect_probability and version[2]!=self._horizon_epoch:continue
                self._update_probability(key,c,now,schedule=True);self.stats['probability_refreshes']+=1
                for block in list(c.keys):self._refresh(block,now)
        return now

    def _update_probability(self,identity,c,now,*,schedule):
        if self.perfect_probability:
            threshold=c.deadline-self.horizon_s
            c.return_probability=float(now>=threshold)
            if schedule and threshold>now:
                heapq.heappush(self._events,(threshold,next(self._serial),'probability',identity,(c.generation,c.version,self._horizon_epoch)))
        else:
            table=self.probability_table
            index=bisect.bisect_right(table.grid,max(0.,now-c.finished_at))-1
            while index+1<len(table.grid) and c.finished_at+table.grid[index+1]<=now:index+=1
            while index>0 and c.finished_at+table.grid[index]>now:index-=1
            elapsed=table.grid[index]
            c.return_probability=table.probability(c.probability_bucket,elapsed,self.horizon_s)
            if schedule and index+1<len(table.grid) and table.survivors(c.probability_bucket,elapsed):
                heapq.heappush(self._events,(c.finished_at+table.grid[index+1],next(self._serial),'probability',identity,(c.generation,c.version,None)))

    def _candidate(self,key):
        if not super()._candidate(key):return False
        return all(self.chains[identity].finished for identity in self.owners.get(key,{}))

    def _lines(self,key):
        age=max(0.,self._time-self.last_access[key])
        fallback=self.horizon_s/(self.horizon_s+age) if self.horizon_s+age else 1.
        probabilities=[]
        for identity in self.owners.get(key,{}):
            c=self.chains[identity]
            if c.mode=='final':probability=0.
            elif not c.finished:probability=1.  # excluded by _candidate
            elif c.mode=='waiting':probability=getattr(c,'return_probability',0.)
            else:probability=fallback
            probabilities.append(probability)
        return [(0,1-max(probabilities))] if probabilities else [(0,1-fallback)]

    def _remove(self,key,now):
        removed=super()._remove(key,now)
        if self._sampling_eviction:self._eviction_ages.append(max(0.,now-removed[3]))
        return removed

    def evict(self,n,protected):
        if n<=0 or n>len(self.chunks):return super().evict(n,protected)
        self._eviction_ages=[];self._sampling_eviction=True
        try:result=super().evict(n,protected)
        finally:self._sampling_eviction=False
        if result is not None:
            for age in self._eviction_ages:
                if len(self.residences)==self.horizon_window:self.residence_sum-=self.residences.popleft()
                self.residences.append(age);self.residence_sum+=age;self.stats['horizon_samples_seen']+=1
            # Prevent sub-ulp negative sums after rolling cancellation.
            self.residence_sum=max(0.,self.residence_sum)
        self.horizon_trace.append(dict(time_s=self._time,horizon_s=self.horizon_s,needed=n,evicted=0 if result is None else len(result),
            samples_seen=self.stats['horizon_samples_seen'],window_samples=len(self.residences),
            window_mean_s=self.residence_sum/len(self.residences) if self.residences else self.initial_horizon))
        return result

    def _compact(self):
        if sum(map(len,self._heaps.values()))>max(128,4*len(self.chunks)):
            for heap in self._heaps.values():
                heap[:]=[v for v in heap if self._versions.get(v[3])==v[2] and self._candidate(v[3])];heapq.heapify(heap)
        if len(self._events)>max(128,4*(len(self.chunks)+len(self.chains))):
            def current(event):
                _,_,kind,key,version=event
                if kind=='horizon':return version==self._horizon_epoch
                if kind!='probability':return False
                c=self.chains.get(key)
                return c is not None and c.mode=='waiting' and version[:2]==(c.generation,c.version) and (not self.perfect_probability or version[2]==self._horizon_epoch)
            self._events[:]=[e for e in self._events if current(e)];heapq.heapify(self._events)
