"""Decision-equivalent scorer with indexed priorities instead of stale heaps."""
import heapq
import math
from itertools import islice
from sortedcontainers import SortedList

from agentkv.policy.hint_aware import HintAwarePolicy


class OptimizedHintAwarePolicy(HintAwarePolicy):
    def clear(self):
        super().clear()
        self._heaps={0:SortedList(),1:SortedList(),2:SortedList()}
        self._entries={};self._signatures={};self.work_counters={}
        self._block_events=SortedList();self._block_event_entries={}
        self._cancelled_times=SortedList()
        self._in_evict=False

    def _erase(self,key):
        old=self._entries.pop(key,None)
        if old is not None:self._heaps[old[0]].remove(old[1])

    def _refresh(self,key,now):
        if self._in_evict:self.work_counters['refreshes']+=1
        previous=self._block_event_entries.pop(key,None)
        if previous is not None:
            self._block_events.remove(previous);self._cancelled_times.add(previous[0])
        version=next(self._serial);self._versions[key]=version
        candidate=self._candidate(key)
        if not candidate:
            self._erase(key)
            return
        lines=self._lines(key)
        a,b=min(lines,key=lambda v:(v[0]*now+v[1],v[0]))
        entry=(-b,self.ranks[key],key)
        # Preserve every reference event-version transition, even when the
        # priority is unchanged. Event cancellation is observable at crossings.
        old=self._entries.get(key)
        if old!=(a,entry):
            self._erase(key)
            self._heaps[a].add(entry);self._entries[key]=(a,entry)
        crossing=math.inf
        for other_a,other_b in lines:
            if other_a<a and math.isfinite(b) and math.isfinite(other_b):
                t=(other_b-b)/(a-other_a)
                if t>now:crossing=min(crossing,t)
        if math.isfinite(crossing):
            event=(crossing,next(self._serial),'block',key,version)
            self._block_events.add(event);self._block_event_entries[key]=event

    def _attach(self,key,identity,depth):
        if not self.tail_first and self.chains[identity].keys.get(key)==depth:
            return {key}
        return super()._attach(key,identity,depth)

    def _remove(self,key,now,*,indexed=True):
        if indexed:self._erase(key)
        event=self._block_event_entries.pop(key,None)
        if event is not None:
            self._block_events.remove(event);self._cancelled_times.add(event[0])
        if self.tail_first:return super()._remove(key,now)
        chunk=self.chunks.pop(key);rank=self.ranks.pop(key);last=self.last_access.pop(key)
        owners=self.owners.pop(key,{})
        self._versions.pop(key,None)
        for identity,depth in owners.items():
            c=self.chains[identity];c.keys.pop(key)
            layer=c.layers[depth];layer.remove(key)
            if not layer:del c.layers[depth]
        return key,chunk,rank,last,owners

    def _advance(self):
        now=float(self.clock())
        if not math.isfinite(now) or now<self._time:raise ValueError('Clock must be finite and monotonic')
        self._time=now
        while self._events or self._block_events:
            queue=self._events
            if self._block_events and (not queue or self._block_events[0]<queue[0]):
                queue=self._block_events
            if queue[0][0]>now:break
            if queue is self._block_events:
                _,_,kind,key,version=queue.pop(0);self._block_event_entries.pop(key,None)
            else:_,_,kind,key,version=heapq.heappop(queue)
            if self._in_evict:self.work_counters['events']+=1
            if kind=='life':
                chain=self.chains.get(key)
                if chain is None or (chain.generation,chain.version)!=version or chain.mode!='waiting':continue
                self._update_life(key,chain,now);self.stats['life_refreshes']+=1
                for block in list(chain.keys):self._refresh(block,now)
            elif kind=='chain':
                chain=self.chains.get(key)
                if chain is None or chain.version!=version or chain.mode!='waiting' or chain.expired:continue
                chain.expired=True;self.stats['timeout_events']+=1
                for block in list(chain.keys):self._refresh(block,now)
            elif self._versions.get(key)==version:
                self._signatures.pop(key,None)
                self._refresh(key,now)
        # Stale block events have no action. Keep only their timestamps so
        # reference compaction thresholds/timing remain exactly unchanged.
        expired=self._cancelled_times.bisect_right(now)
        if expired:del self._cancelled_times[:expired]
        return now

    def _compact(self):
        # The reference threshold counts stale crossover events too. Retaining
        # their timestamps preserves cancellation/chain-event semantics while
        # avoiding tens of thousands of Python validity checks at once.
        size=len(self._events)+len(self._block_events)+len(self._cancelled_times)
        if size<=max(128,4*(len(self.chunks)+len(self.chains))):return
        if self._in_evict:self.work_counters['event_sweeps']+=1
        def current(e):
            if e[2]=='block':return self._versions.get(e[3])==e[4]
            c=self.chains.get(e[3])
            if e[2]=='life':return c is not None and (c.generation,c.version)==e[4] and c.mode=='waiting'
            return c is not None and c.version==e[4] and c.mode=='waiting' and not c.expired
        self._events[:]=[e for e in self._events if current(e)];heapq.heapify(self._events)
        self._cancelled_times.clear()

    def _batch_candidates(self,a,now,protected):
        for intercept,rank,key in self._heaps[a]:
            if key not in protected and self.chunks[key].ref_cnt==0:
                yield (intercept-a*now,rank,self._versions[key],a,key)

    def evict(self,n,protected):
        if n<0:raise ValueError('Negative eviction count')
        if n==0:return []
        if n>len(self.chunks):return None
        self.work_counters=dict(events=0,refreshes=0,event_sweeps=0,ineligible=0)
        self._in_evict=True
        try:
            now=self._advance();selected=[];cursor={0:0,1:0,2:0}
            if not self.tail_first:
                # Ownership removal cannot change other candidates without the
                # optional tail constraint. Merge the three sorted score runs
                # once, instead of searching their heads for every victim.
                ordered=heapq.merge(*(self._batch_candidates(a,now,protected) for a in self._heaps))
                keys=[entry[4] for entry in islice(ordered,n)]
                groups={0:[],1:[],2:[]}
                for key in keys:
                    a,entry=self._entries.pop(key);groups[a].append(entry)
                for a,entries in groups.items():
                    if not entries:continue
                    heap=self._heaps[a]
                    if heap.bisect_right(entries[-1])==len(entries):
                        del heap[:len(entries)]
                    else:
                        for entry in entries:heap.remove(entry)
                selected=[self._remove(key,now,indexed=False) for key in keys]
            while self.tail_first and len(selected)<n:
                choices=[]
                for a,heap in self._heaps.items():
                    while cursor[a]<len(heap):
                        entry=heap[cursor[a]];_,rank,key=entry;version=self._versions[key]
                        if not self._candidate(key):
                            heap.pop(cursor[a]);self._entries.pop(key,None)
                            self.work_counters['ineligible']+=1;continue
                        if key in protected:cursor[a]+=1;continue
                        choices.append((-(a*now-entry[0]),rank,version,a));break
                if not choices:break
                *_,a=min(choices)
                key=self._heaps[a].pop(cursor[a])[2];self._entries.pop(key,None)
                selected.append(self._remove(key,now))
                if self.tail_first:cursor={0:0,1:0,2:0}
            if len(selected)!=n:
                for key,chunk,rank,last,owners in reversed(selected):
                    self.chunks[key]=chunk;self.ranks[key]=rank;self.last_access[key]=last
                    affected={key}
                    for identity,depth in owners.items():affected.update(self._attach(key,identity,depth))
                    for block in affected:self._refresh(block,now)
                result=None
            else:
                self.stats['evicted']+=n
                result=[(key,chunk) for key,chunk,*_ in selected]
                for identity in {cid for *_,owners in selected for cid in owners}:
                    c=self.chains[identity]
                    if c.finished and not c.keys and not any(identity in v for v in self._pending.values()):
                        del self.chains[identity]
            self._compact()
            return result
        finally:self._in_evict=False
