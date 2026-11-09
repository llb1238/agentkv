"""Independent, duck-typed CachePolicy for predicted Agent return times.

No serving-engine imports. Keys and chunks remain caller-owned objects; chunks
need only expose ref_cnt. Scores are piecewise affine, so clock changes use
three intercept heaps and lazy deadline/crossover events rather than stale
cached timestamps or a full cache scan at every eviction.
"""
import heapq
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from agentkv.policy.remaining_life import RemainingLifeTable

log = logging.getLogger(__name__)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


@dataclass
class _Chain:
    generation: int = 0
    known: bool = False
    mode: str = 'unknown'
    request_id: object = None
    turn: int = -1
    finished: bool = False
    deadline: float = 0.
    margin: float = 0.
    expired: bool = False
    finished_at: float = 0.
    life_bucket: int = 0
    life_index: int = 0
    life_remaining: object = None
    version: int = 0
    keys: dict = field(default_factory=dict)
    layers: dict = field(default_factory=dict)
    depths: list = field(default_factory=list)

    def frontier(self):
        while self.depths and -self.depths[0] not in self.layers:
            heapq.heappop(self.depths)
        return set(self.layers[-self.depths[0]]) if self.depths else set()


class HintAwarePolicy:
    def __init__(self, cache_capacity, *, clock=time.monotonic, tail_first=False,
                 final_first=True, timeout_decay=None, scoring='remaining', residual_table=None):
        if cache_capacity < 0:raise ValueError('Negative capacity')
        if scoring not in ('remaining','deadline','lru_final'):raise ValueError('Unknown scoring mode')
        self.scoring = scoring
        self.residual_table = (residual_table or RemainingLifeTable.load()) if scoring=='remaining' else None
        if timeout_decay is None:timeout_decay = scoring=='deadline'
        if scoring=='remaining' and timeout_decay:raise ValueError('Residual scoring replaces timeout decay')
        self.cache_capacity = cache_capacity
        self.clock = clock
        self.tail_first, self.final_first, self.timeout_decay = tail_first, final_first, timeout_decay
        self.clear()

    def clear(self):
        self.chunks = {};self.ranks = {};self.last_access = {};self.owners = {};self.chains = {}
        self._pending = {};self._pending_requests = {}
        self._versions = {};self._heaps = {0: [], 1: [], 2: []};self._events = []
        self._serial = itertools.count();self._chain_generations = itertools.count();self._rank = 0;self._time = -math.inf
        self.stats = dict(valid_hint_requests=0,invalid_hint_callbacks=0,timeout_events=0,evicted=0,life_refreshes=0)

    def get(self, key):return self.chunks.get(key)

    def _hint(self, context):
        envelope = _get(context,'kv_hints')
        try:
            if envelope is None:envelope = _get(_get(context,'kv_transfer_params'),'kv_hints')
            if envelope is None:return None
            if _get(envelope,'protocol_version') != 1:raise ValueError('protocol_version')
            actions = _get(envelope,'actions')
            if not isinstance(actions,(list,tuple)):raise ValueError('actions')
            matching = [a for a in actions if _get(a,'action_type') == 'agentkv.tool_gap']
            if not matching:return None
            if len(matching)!=1 or _get(matching[0],'action_version')!=1:raise ValueError('action_version/count')
            p = _get(matching[0],'payload')
            session, parent = _get(p,'session_id'), _get(p,'parent_session')
            turn, final = _get(p,'turn'), _get(p,'final')
            if not isinstance(session,str) or not session or (parent is not None and not isinstance(parent,str)):
                raise ValueError('session identity')
            if isinstance(turn,bool) or not isinstance(turn,int) or turn<0 or not isinstance(final,bool):raise ValueError('turn/final')
            gap, margin = _get(p,'expected_gap_ms'), _get(p,'gap_p90_ms')
            if not final and not (_number(gap) and _number(margin)):raise ValueError('gap')
            if final and any(v is not None and not _number(v) for v in (gap,margin)):raise ValueError('final gap')
            return dict(session=session,parent=parent,turn=turn,final=final,gap=gap,margin=margin)
        except (TypeError,ValueError,AttributeError):
            self.stats['invalid_hint_callbacks'] += 1
            log.warning('Invalid AgentKV hint for %s; using LRU',_get(context,'req_id'))
            return None

    def _advance(self):
        now = float(self.clock())
        if not math.isfinite(now) or now < self._time:raise ValueError('Clock must be finite and monotonic')
        self._time = now
        while self._events and self._events[0][0] <= now:
            _,_,kind,key,version = heapq.heappop(self._events)
            if kind == 'life':
                chain = self.chains.get(key)
                if chain is None or (chain.generation,chain.version)!=version or chain.mode!='waiting':continue
                self._update_life(key,chain,now)
                self.stats['life_refreshes'] += 1
                for block in list(chain.keys):self._refresh(block,now)
            elif kind == 'chain':
                chain = self.chains.get(key)
                if chain is None or chain.version!=version or chain.mode!='waiting' or chain.expired:continue
                chain.expired = True;self.stats['timeout_events'] += 1
                for block in list(chain.keys):self._refresh(block,now)
            elif self._versions.get(key)==version:
                self._refresh(key,now)
        return now

    def _update_life(self,identity,chain,now,*,initial=False):
        # Compare absolute event times. Subtracting large start times can put an
        # exact grid event just below its knot and repeatedly reschedule it.
        table = self.residual_table
        index = 0 if initial else chain.life_index+1
        while index+1 < len(table.grid) and chain.finished_at+table.grid[index+1] <= now:index += 1
        chain.life_index = index
        chain.life_remaining = table.values[chain.life_bucket][index]
        if chain.life_remaining is not None and index+1 < len(table.grid):
            event = chain.finished_at+table.grid[index+1]
            heapq.heappush(self._events,(event,next(self._serial),'life',identity,(chain.generation,chain.version)))

    def _candidate(self,key):
        if key not in self.chunks or self.chunks[key].ref_cnt != 0:return False
        if self.tail_first:
            for identity,depth in self.owners.get(key,{}).items():
                chain = self.chains[identity]
                if chain.known:
                    chain.frontier()  # clear stale depth entries
                    if chain.depths and depth != -chain.depths[0]:return False
        return True

    def _lines(self,key):
        lines = {}
        for identity in self.owners.get(key,{}):
            c = self.chains[identity]
            if c.mode == 'final':a,b = 0,math.inf
            elif self.scoring=='lru_final':a,b = 2,-self.last_access[key]
            elif c.mode == 'active':a,b = 1,0.
            elif c.mode == 'waiting':
                if self.scoring=='remaining':
                    a,b = (1,c.life_remaining) if c.life_remaining is not None else (2,-self.last_access[key])
                elif c.expired:a,b = 2,-c.deadline-2*c.margin
                else:a,b = 0,c.deadline
            else:a,b = 2,-self.last_access[key]
            lines[a] = min(lines.get(a,math.inf),b)
        return list(lines.items()) if lines else [(2,-self.last_access[key])]

    def _refresh(self,key,now):
        version = next(self._serial);self._versions[key] = version
        if not self._candidate(key):return
        lines = self._lines(key)
        a,b = min(lines,key=lambda v:(v[0]*now+v[1],v[0]))
        heapq.heappush(self._heaps[a],(-b,self.ranks[key],version,key))
        crossing = math.inf
        for other_a,other_b in lines:
            if other_a < a and math.isfinite(b) and math.isfinite(other_b):
                t = (other_b-b)/(a-other_a)
                if t > now:crossing = min(crossing,t)
        if math.isfinite(crossing):heapq.heappush(self._events,(crossing,next(self._serial),'block',key,version))

    def _attach(self,key,identity,depth):
        c = self.chains[identity];affected = c.frontier()
        previous = c.keys.get(key)
        if previous is not None and previous != depth:
            c.layers[previous].remove(key)
            if not c.layers[previous]:del c.layers[previous]
        if depth not in c.layers:
            c.layers[depth] = set();heapq.heappush(c.depths,-depth)
        c.layers[depth].add(key);c.keys[key] = depth
        self.owners.setdefault(key,{})[identity] = depth
        if len(c.depths)>max(64,2*len(c.layers)):
            c.depths = [-d for d in c.layers];heapq.heapify(c.depths)
        frontier = c.frontier()
        return (affected | frontier | {key}) if self.tail_first and c.known and frontier != affected else {key}

    def _observe(self,keys,context,*,finished=False):
        now = self._advance();hint = self._hint(context)
        req_id = _get(context,'req_id',id(context))
        if _get(context,'instance_id') is not None and _get(context,'chain_id') is not None:
            identity = (str(_get(context,'instance_id')),str(_get(context,'chain_id')))
        elif hint:
            identity = (hint['parent'] or hint['session'],hint['session'])
        else:return now
        if identity not in self.chains:self.chains[identity] = _Chain(generation=next(self._chain_generations))
        c = self.chains[identity]
        if hint and hint['turn']<c.turn:return now
        affected = set(c.keys)
        if c.request_id != req_id:
            c.request_id = req_id;c.finished = False;c.known = hint is not None
            c.mode = 'active' if hint else 'unknown';c.expired = False;c.version += 1
            if hint:c.turn = hint['turn']
        if finished and not c.finished:
            c.finished = True;c.version += 1;c.known = hint is not None
            if hint:
                self.stats['valid_hint_requests'] += 1
                if hint['final']:c.mode = 'final' if self.final_first else 'unknown'
                else:
                    c.mode = 'waiting';c.deadline = now+hint['gap']/1000;c.margin = hint['margin']/1000
                    c.expired = False;c.finished_at = now
                    if self.scoring=='remaining':
                        c.life_bucket = self.residual_table.bucket(hint['gap']/1000)
                        self._update_life(identity,c,now,initial=True)
                    if self.timeout_decay:
                        heapq.heappush(self._events,(math.nextafter(c.deadline+c.margin,math.inf),next(self._serial),'chain',identity,c.version))
            else:c.mode = 'unknown'
        position = _get(context,'get_offload_key_position')
        for i,key in enumerate(keys):
            depth = position(key) if position is not None else None
            if depth is None:depth = c.keys.get(key,i+1)
            if key in self.chunks:affected.update(self._attach(key,identity,depth))
            elif not finished:
                self._pending.setdefault(key,{})[identity] = (req_id,depth)
                self._pending_requests.setdefault(req_id,set()).add(key)
        if finished:
            for key in self._pending_requests.pop(req_id,()):
                pending = self._pending.get(key)
                if pending and pending.get(identity,(None,))[0]==req_id:
                    del pending[identity]
                    if not pending:del self._pending[key]
        for key in affected:self._refresh(key,now)
        return now

    def insert(self,key,chunk):
        now = self._advance()
        if key in self.chunks:raise ValueError('Duplicate cache insert')
        self.chunks[key] = chunk;self._rank += 1
        self.ranks[key] = self._rank;self.last_access[key] = now
        affected = {key}
        for identity,(_,depth) in self._pending.pop(key,{}).items():affected.update(self._attach(key,identity,depth))
        for block in affected:self._refresh(block,now)
        self._compact()

    def _remove(self,key,now):
        chunk = self.chunks.pop(key);rank = self.ranks.pop(key);last = self.last_access.pop(key)
        owners = self.owners.pop(key,{})
        self._versions.pop(key,None)
        affected = set()
        for identity,depth in owners.items():
            c = self.chains[identity];before = c.frontier() if self.tail_first and c.known else set()
            c.keys.pop(key)
            c.layers[depth].remove(key)
            if not c.layers[depth]:del c.layers[depth]
            after = c.frontier() if self.tail_first and c.known else set()
            if after != before:affected.update(after)
        for block in affected:self._refresh(block,now)
        return key,chunk,rank,last,owners

    def remove(self,key):self._remove(key,self._advance())

    def _recency(self,keys,now):
        for key in reversed(keys):
            if key in self.chunks:
                self._rank += 1;self.ranks[key] = self._rank;self.last_access[key] = now
                if self._candidate(key):self._refresh(key,now)
        self._compact()

    def touch(self,keys,req_context):
        keys = list(keys);now = self._observe(keys,req_context)
        self._recency(keys,now)

    def on_store_miss(self,keys,req_context):
        # Miss observation precedes insertion; it must not touch cached prefix
        # ranks prematurely in the completion-write simulator.
        self._observe(list(keys),req_context)
        self._compact()

    def on_request_finished(self,key_groups,insertion_only_keys,reused_keys,req_context):
        keys = [key for group in key_groups for key in group]
        position = _get(req_context,'get_offload_key_position')
        if position is not None:
            positions = {key:position(key) for key in keys}
            if all(v is not None for v in positions.values()):keys.sort(key=positions.__getitem__)
        now = self._observe(keys,req_context,finished=True)
        self._recency(keys,now)

    def mark_evictable(self,key):self._refresh(key,self._advance())
    def mark_non_evictable(self,key):self._refresh(key,self._advance())

    def _compact(self):
        if sum(map(len,self._heaps.values()))>max(128,4*len(self.chunks)):
            for a,heap in self._heaps.items():
                heap[:] = [v for v in heap if self._versions.get(v[3])==v[2] and self._candidate(v[3])]
                heapq.heapify(heap)
        if len(self._events)>max(128,4*(len(self.chunks)+len(self.chains))):
            def current(e):
                if e[2]=='block':return self._versions.get(e[3])==e[4]
                c = self.chains.get(e[3])
                if e[2]=='life':return c is not None and (c.generation,c.version)==e[4] and c.mode=='waiting'
                return c is not None and c.version==e[4] and c.mode=='waiting' and not c.expired
            self._events[:] = [e for e in self._events if current(e)];heapq.heapify(self._events)

    def evict(self,n,protected):
        if n<0:raise ValueError('Negative eviction count')
        if n==0:return []
        if n>len(self.chunks):return None
        now = self._advance();selected = [];deferred = []
        while len(selected)<n:
            choices = []
            for a,heap in self._heaps.items():
                while heap:
                    entry = heap[0];_,rank,version,key = entry
                    if self._versions.get(key)!=version or not self._candidate(key):heapq.heappop(heap);continue
                    if key in protected:
                        deferred.append((a,heapq.heappop(heap)));continue
                    choices.append((-(a*now-entry[0]),rank,version,a));break
            if not choices:break
            *_,a = min(choices)
            key = heapq.heappop(self._heaps[a])[3]
            selected.append(self._remove(key,now))
        if len(selected)!=n:
            # Undo logical removals with original recency and ownership. Lazy
            # heap versions can change; observable eviction order cannot.
            for key,chunk,rank,last,owners in reversed(selected):
                self.chunks[key] = chunk;self.ranks[key] = rank;self.last_access[key] = last
                affected = {key}
                for identity,depth in owners.items():affected.update(self._attach(key,identity,depth))
                for block in affected:self._refresh(block,now)
            result = None
        else:
            self.stats['evicted'] += n
            result = [(key,chunk) for key,chunk,*_ in selected]
            for identity in list({cid for *_,owners in selected for cid in owners}):
                c = self.chains[identity]
                if c.finished and not c.keys and not any(identity in v for v in self._pending.values()):del self.chains[identity]
        for a,entry in deferred:heapq.heappush(self._heaps[a],entry)
        self._compact()
        return result
