"""Chain-scored priority queue with the frozen block-rank tie ordering.

The old crossover ledger is retained solely for exact timeout-compaction
semantics. It contains no eviction priorities. Ordinary lifetime updates touch
one chain; only shared blocks can have legacy crossover bookkeeping.
"""
import heapq
import math
from agentkv.policy.optimized_policy import OptimizedHintAwarePolicy
from agentkv.policy.hint_aware import _Chain, _get


class ChainHintAwarePolicy(OptimizedHintAwarePolicy):
    def __init__(self, *args, **kwargs):
        if kwargs.get('tail_first', False):
            raise ValueError('Chain scoring preserves frozen rank ordering')
        super().__init__(*args, **kwargs)

    def clear(self):
        super().clear()
        self._rank_lists = {}
        self._rank_seen = {}
        self._rank_dirty = set()
        self._orphans = set()
        self._shared = {}
        self._owner_sets = {}
        self._shadow_cache = {}
        self._neighbors = {}
        self._neighbor_groups = set()
        self._recency_shadow = True
        self._group_counts = {}
        self._neighbor_refs = {}
        self._pending_by_chain = {}
        self._empty_candidates = set()
        self._next_event_compact = -math.inf

    def _group_add(self, group):
        count = self._group_counts.get(group, 0)
        self._group_counts[group] = count + 1
        if count:
            return
        self._neighbor_groups.add(group)
        for owner in group:
            refs = self._neighbor_refs.setdefault(owner, {})
            neighbors = self._neighbors.setdefault(owner, set())
            for other in group - {owner}:
                refs[other] = refs.get(other, 0) + 1
                neighbors.add(other)

    def _group_remove(self, group):
        count = self._group_counts[group] - 1
        if count:
            self._group_counts[group] = count
            return
        del self._group_counts[group]
        self._neighbor_groups.remove(group)
        for owner in group:
            refs = self._neighbor_refs[owner]
            for other in group - {owner}:
                refs[other] -= 1
                if not refs[other]:
                    del refs[other]
                    self._neighbors[owner].remove(other)
            if not refs:
                del self._neighbor_refs[owner]
                del self._neighbors[owner]

    def _cleanup_empty(self):
        # Deadline's legacy stale timer/version reuse is observable. Preserve
        # that path; remaining-life events carry a generation and are safe.
        if self.scoring != 'remaining':
            self._empty_candidates.clear()
            return
        if self._in_evict or not self._empty_candidates:
            return
        candidates, self._empty_candidates = self._empty_candidates, set()
        for identity in candidates:
            c = self.chains.get(identity)
            if c is not None and c.finished and not c.keys and not self._pending_by_chain.get(identity):
                del self.chains[identity]
                self._rank_lists.pop(identity, None)
                self._rank_dirty.discard(identity)
                self._shared.pop(identity, None)

    def _compact(self):
        if self.scoring != 'remaining':
            self._empty_candidates.clear()
            return super()._compact()
        self._cleanup_empty()
        # Check at most once per clock second, outside per-block hot loops.
        if self._time < self._next_event_compact:
            return
        self._next_event_compact = self._time + 1.
        def current(e):
            c = self.chains.get(e[3])
            if e[2] == 'life':
                return c is not None and (c.generation, c.version) == e[4] and c.mode == 'waiting'
            if e[2] == 'block':
                return self._versions.get(e[3]) == e[4]
            return c is not None and c.version == e[4] and c.mode == 'waiting' and not c.expired
        kept = [e for e in self._events if current(e)]
        if len(kept) * 2 < len(self._events):
            self._events[:] = kept
            heapq.heapify(self._events)
        self._cancelled_times.clear()

    def _attach(self, key, identity, depth):
        pending = self._pending_by_chain.get(identity)
        if pending is not None:
            pending.discard(key)
            if not pending:
                del self._pending_by_chain[identity]
        chain = self.chains[identity]
        if chain.keys.get(key) == depth:
            return {key}
        previous = chain.keys.get(key)
        chain.keys[key] = depth
        owners = self.owners.setdefault(key, {})
        owners[identity] = depth
        if previous is None:
            self._rank_dirty.add(identity)
            if key in self._orphans:
                self._orphans.remove(key)
                self._rank_dirty.add(None)
            if len(owners) > 1:
                group = frozenset(owners)
                old = self._owner_sets.get(key)
                if old is not None:
                    self._group_remove(old)
                self._owner_sets[key] = group
                self._group_add(group)
                for owner in owners:
                    self._shared.setdefault(owner, {})[key] = None
        return {key}

    def _needs_shadow(self, identity):
        if self.scoring != 'remaining' or self._block_event_entries:
            return True
        fallback = finite = False
        for owner in (identity, *self._neighbors.get(identity, ())):
            c = self.chains.get(owner)
            if c is None:
                continue
            fallback |= self._is_lru(c)
            finite |= c.mode == 'active' or (c.mode == 'waiting' and c.life_remaining is not None)
            if fallback and finite:
                return True
        return False

    def _observe(self, keys, context, *, finished=False):
        now = self._advance()
        self._recency_shadow = True
        hint = self._hint(context)
        req_id = _get(context, 'req_id', id(context))
        if _get(context, 'instance_id') is not None and _get(context, 'chain_id') is not None:
            identity = (str(_get(context, 'instance_id')), str(_get(context, 'chain_id')))
        elif hint:
            identity = (hint['parent'] or hint['session'], hint['session'])
        else:
            return now
        if identity not in self.chains:
            self.chains[identity] = _Chain(generation=next(self._chain_generations))
        c = self.chains[identity]
        if hint and hint['turn'] < c.turn:
            return now
        # The old affected set contains every chain block for score refresh.
        # Only shared blocks can affect its legacy crossover bookkeeping.
        if c.request_id != req_id:
            c.request_id = req_id
            c.finished = False
            c.known = hint is not None
            c.mode = 'active' if hint else 'unknown'
            c.expired = False
            c.version += 1
            if hint:
                c.turn = hint['turn']
        if finished and not c.finished:
            c.finished = True
            c.version += 1
            c.known = hint is not None
            if hint:
                self.stats['valid_hint_requests'] += 1
                if hint['final']:
                    c.mode = 'final' if self.final_first else 'unknown'
                else:
                    c.mode = 'waiting'
                    c.deadline = now + hint['gap'] / 1000
                    c.margin = hint['margin'] / 1000
                    c.expired = False
                    c.finished_at = now
                    if self.scoring == 'remaining':
                        c.life_bucket = self.residual_table.bucket(hint['gap'] / 1000)
                        self._update_life(identity, c, now, initial=True)
                    if self.timeout_decay:
                        heapq.heappush(self._events, (math.nextafter(c.deadline + c.margin, math.inf),
                            next(self._serial), 'chain', identity, c.version))
            else:
                c.mode = 'unknown'
        position = _get(context, 'get_offload_key_position')
        observed = _get(context, '_agentkv_observed_prefix')
        finishing_prefix = _get(context, '_agentkv_finishing_prefix', False)
        pending_keys = []
        for i, key in enumerate(keys):
            depth = position(key) if position is not None else None
            # The adapter has just observed these keys in this same callback;
            # no insertion/removal can occur between its two observations.
            # Missing positions retain the original caller-index fallback.
            if depth is not None and observed is not None and key in observed:
                continue
            if depth is None:
                depth = c.keys.get(key, i + 1)
            if key in self.chunks:
                if c.keys.get(key) != depth:
                    self._attach(key, identity, depth)
            elif not finished:
                # At completion, a newly absent ownership would only be added
                # here and deleted by the immediately following finished hook.
                # Existing entries (including another request's) keep the exact
                # overwrite/cleanup semantics, as do duplicate finished calls.
                if finishing_prefix and not c.finished and identity not in self._pending.get(key, ()):
                    continue
                pending = self._pending.get(key)
                if pending is None:
                    pending = self._pending[key] = {}
                pending[identity] = (req_id, depth)
                pending_keys.append(key)
        if pending_keys:
            # These indexes only support completion cleanup/empty-chain GC.
            # No callback can interleave this synchronous observation, and
            # attachment above may remove an emptied chain index: resolve it
            # once after attachment, rather than allocate two empty sets for
            # every prefix block, including already-pending ones.
            self._pending_requests.setdefault(req_id, set()).update(pending_keys)
            self._pending_by_chain.setdefault(identity, set()).update(pending_keys)
        if finished:
            for key in self._pending_requests.pop(req_id, ()):
                pending = self._pending.get(key)
                if pending and pending.get(identity, (None,))[0] == req_id:
                    del pending[identity]
                    if not pending:
                        del self._pending[key]
                    own = self._pending_by_chain.get(identity)
                    if own is not None:
                        own.discard(key)
                        if not own:
                            del self._pending_by_chain[identity]
        self._recency_shadow = self._needs_shadow(identity)
        if self._recency_shadow:
            for key in self._shared.get(identity, ()):
                self._shadow_refresh(key, now)
        if finished and not c.keys:
            self._empty_candidates.add(identity)
        return now

    def _recency(self, keys, now):
        for key in reversed(keys):
            if key not in self.chunks:
                continue
            self._rank += 1
            self.ranks[key] = self._rank
            self.last_access[key] = now
            self._rank_seen[key] = self._rank
            owners = self.owners.get(key, {})
            self._rank_dirty.update(owners or (None,))
            if self._recency_shadow and len(owners) > 1 and self.chunks[key].ref_cnt == 0:
                self._shadow_refresh(key, now)
        self._compact()

    def _refresh(self, key, now):
        rank = self.ranks.get(key)
        if rank is not None and self._rank_seen.get(key) != rank:
            self._rank_seen[key] = rank
            owners = self.owners.get(key)
            if owners:
                self._rank_dirty.update(owners)
            else:
                self._orphans.add(key)
                self._rank_dirty.add(None)
        if key in self._owner_sets:
            self._shadow_refresh(key, now)

    def _shadow_refresh(self, key, now):
        previous = self._block_event_entries.pop(key, None)
        if previous is not None:
            self._block_events.remove(previous)
            self._cancelled_times.add(previous[0])
        # A single owner (or no owner) has only one score line, so it can never
        # create a crossover. There is no block priority/version to update.
        owners = self.owners.get(key, {})
        if len(owners) < 2 or self.chunks[key].ref_cnt != 0:
            return
        group = self._owner_sets[key]
        cached = self._shadow_cache.get(group)
        if cached is None:
            fallback = any(self._is_lru(self.chains[c]) for c in group)
            cached = (fallback, None if fallback else self._crossing(self._lines(key), now))
            self._shadow_cache[group] = cached
        crossing = self._crossing(self._lines(key), now) if cached[0] else cached[1]
        if math.isfinite(crossing):
            version = next(self._serial)
            self._versions[key] = version
            event = (crossing, next(self._serial), 'block', key, version)
            self._block_events.add(event)
            self._block_event_entries[key] = event

    @staticmethod
    def _crossing(lines, now):
        a, b = min(lines, key=lambda v: (v[0] * now + v[1], v[0]))
        crossing = math.inf
        for other_a, other_b in lines:
            if other_a < a and math.isfinite(b) and math.isfinite(other_b):
                t = (other_b - b) / (a - other_a)
                if t > now:
                    crossing = min(crossing, t)
        return crossing

    def _advance(self):
        now = float(self.clock())
        if not math.isfinite(now) or now < self._time:
            raise ValueError('Clock must be finite and monotonic')
        self._time = now
        # Cached crossings are relative to the preceding callback's clock.
        # Reusing an already-due crossing would reschedule it forever.
        self._shadow_cache.clear()
        if (self.scoring == 'remaining' and not self._block_event_entries and not self._cancelled_times
                and (not self._events or self._events[0][0] > now)):
            return now
        while self._events or self._block_events:
            queue = self._events
            if self._block_events and (not queue or self._block_events[0] < queue[0]):
                queue = self._block_events
            if queue[0][0] > now:
                break
            if queue is self._block_events:
                _, _, kind, key, version = queue.pop(0)
                self._block_event_entries.pop(key, None)
            else:
                _, _, kind, key, version = heapq.heappop(queue)
            if kind == 'life':
                chain = self.chains.get(key)
                if chain is None or (chain.generation, chain.version) != version or chain.mode != 'waiting':
                    continue
                self._update_life(key, chain, now)
                self.stats['life_refreshes'] += 1
            elif kind == 'chain':
                chain = self.chains.get(key)
                if chain is None or chain.version != version or chain.mode != 'waiting' or chain.expired:
                    continue
                chain.expired = True
                self.stats['timeout_events'] += 1
            else:
                if self._versions.get(key) == version:
                    self._shadow_refresh(key, now)
                continue
            self._shadow_cache.clear()
            # Match the frozen housekeeping event counts, without refreshing
            # per-block eviction scores. Unshared blocks have no crossover.
            if self._needs_shadow(key):
                for block in self._shared.get(key, ()):
                    self._shadow_refresh(block, now)
        expired = self._cancelled_times.bisect_right(now)
        if expired:
            del self._cancelled_times[:expired]
        self._shadow_cache.clear()
        return now

    def _remove(self, key, now, *, indexed=True):
        self._rank_seen.pop(key, None)
        group = self._owner_sets.pop(key, None)
        if group is not None:
            self._group_remove(group)
        event = self._block_event_entries.pop(key, None)
        if event is not None:
            self._block_events.remove(event)
            self._cancelled_times.add(event[0])
        chunk = self.chunks.pop(key)
        rank = self.ranks.pop(key)
        last = self.last_access.pop(key)
        owners = self.owners.pop(key, {})
        self._versions.pop(key, None)
        if owners:
            self._rank_dirty.update(owners)
            for owner in owners:
                self.chains[owner].keys.pop(key)
                self._shared.get(owner, {}).pop(key, None)
                if not self._shared.get(owner):
                    self._shared.pop(owner, None)
                if not self.chains[owner].keys:
                    self._empty_candidates.add(owner)
        else:
            self._orphans.discard(key)
            self._rank_dirty.add(None)
        return key, chunk, rank, last, owners

    def _is_lru(self, chain):
        if chain.mode == 'final':
            return False
        return (self.scoring == 'lru_final' or chain.mode == 'unknown'
                or (chain.mode == 'waiting' and self.scoring == 'remaining'
                    and chain.life_remaining is None))

    def insert(self,key,chunk):
        # Ordinary inserts have at most one pending owner and cannot create a
        # score crossover. Preserve the general path for shared/deadline keys.
        if self.scoring!='remaining' or len(self._pending.get(key,()))>1:
            return super().insert(key,chunk)
        now=self._advance()
        if key in self.chunks:raise ValueError('Duplicate cache insert')
        self.chunks[key]=chunk;self._rank+=1
        self.ranks[key]=self._rank;self.last_access[key]=now;self._rank_seen[key]=self._rank
        pending=self._pending.pop(key,None)
        if pending:
            identity,(_,depth)=next(iter(pending.items()))
            self._attach(key,identity,depth)
        else:
            self._orphans.add(key);self._rank_dirty.add(None)
        if self._empty_candidates or now>=self._next_event_compact:self._compact()

    def mark_evictable(self,key):
        if self.scoring!='remaining':return super().mark_evictable(key)
        now=self._advance()
        # Marking eligibility never changes a rank; selection checks live refs.
        if key in self._owner_sets:self._shadow_refresh(key,now)

    def mark_non_evictable(self,key):
        if self.scoring!='remaining':return super().mark_non_evictable(key)
        now=self._advance()
        if key in self._owner_sets:self._shadow_refresh(key,now)

    def _scores(self, now):
        scores = {None: None}
        for identity, c in self.chains.items():
            if self._is_lru(c):
                scores[identity] = None
            elif c.mode == 'final':
                scores[identity] = math.inf
            elif c.mode == 'active':
                scores[identity] = now
            elif c.mode == 'waiting' and self.scoring == 'remaining':
                scores[identity] = now + c.life_remaining
            elif c.mode == 'waiting':
                scores[identity] = 2 * now + (-c.deadline - 2 * c.margin) if c.expired else c.deadline
            else:
                scores[identity] = None
        return scores

    def _ensure_ranks(self):
        for identity in self._rank_dirty:
            keys = self._orphans if identity is None else (
                self.chains[identity].keys if identity in self.chains else ())
            if keys:
                ordered = sorted(keys, key=self.ranks.__getitem__)
                self._rank_lists[identity] = [(self.ranks[k], k) for k in ordered]
            else:
                self._rank_lists.pop(identity, None)
        self._rank_dirty.clear()

    def evict(self, n, protected):
        if n < 0:
            raise ValueError('Negative eviction count')
        if n == 0:
            return []
        if n > len(self.chunks):
            return None
        self.work_counters = dict(events=0, refreshes=0, event_sweeps=0, ineligible=0)
        self._in_evict = True
        try:
            now = self._advance()
            self._ensure_ranks()
            scores = self._scores(now)
            heap, chosen, selected = [], set(), []
            groups = {}
            views = [(identity, queue, scores[identity]) for identity, queue in self._rank_lists.items()]
            chunks, last_access, owner_sets = self.chunks, self.last_access, self._owner_sets
            def head(ordinal, index):
                identity, queue, own = views[ordinal]
                while index < len(queue):
                    rank, key = queue[index]
                    if key in chosen or key in protected or chunks[key].ref_cnt != 0:
                        index += 1
                        continue
                    value = 2 * now - last_access[key] if own is None else own
                    group = owner_sets.get(key)
                    if group is not None:
                        cached = groups.get(group)
                        if cached is None:
                            fixed = [scores[c] for c in group if scores[c] is not None]
                            cached = (min(fixed, default=math.inf), any(scores[c] is None for c in group))
                            groups[group] = cached
                        best = min(cached[0], 2 * now - last_access[key]) if cached[1] else cached[0]
                        if best < value:
                            index += 1
                            continue
                    return (-value, rank, ordinal, key, index)
                return None
            for ordinal in range(len(views)):
                entry = head(ordinal, 0)
                if entry is not None:
                    heap.append(entry)
            heapq.heapify(heap)
            while heap and len(selected) < n:
                entry = heapq.heappop(heap)
                # Consume a sorted run while this chain remains the global
                # minimum. This is exactly the same heap merge, including all
                # rank/ordinal ties, without reinserting its head per block.
                while entry is not None:
                    _, _, ordinal, key, index = entry
                    if key not in chosen:
                        chosen.add(key)
                        selected.append(key)
                    if len(selected) == n:
                        break
                    entry = head(ordinal, index + 1)
                    if entry is not None and heap and entry > heap[0]:
                        heapq.heappush(heap, entry)
                        break
            removed = [self._remove(key, now) for key in selected]
            if len(removed) != n:
                for key, chunk, rank, last, owners in reversed(removed):
                    self.chunks[key] = chunk
                    self.ranks[key] = rank
                    self.last_access[key] = last
                    affected = {key}
                    for identity, depth in owners.items():
                        affected.update(self._attach(key, identity, depth))
                    for key in affected:
                        self._refresh(key, now)
                result = None
            else:
                self.stats['evicted'] += n
                result = [(item[0], item[1]) for item in removed]
                if self.scoring!='remaining':
                    for identity in {cid for *_, owners in removed for cid in owners}:
                        c = self.chains[identity]
                        if c.finished and not c.keys and not any(identity in v for v in self._pending.values()):
                            del self.chains[identity]
                            self._shared.pop(identity, None)
            self._compact()
            return result
        finally:
            self._in_evict = False
            self._cleanup_empty()
