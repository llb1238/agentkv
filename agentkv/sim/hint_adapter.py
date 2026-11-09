"""Completion-write simulator adapter; no vLLM dependency or device access."""
from dataclasses import dataclass
from types import SimpleNamespace
from agentkv.hints import make_envelope
from agentkv.policy.hint_aware import HintAwarePolicy
from agentkv.policy.remaining_life import RemainingLifeTable
from agentkv.policy.horizon import HorizonProbabilityTable
from agentkv.policy.horizon_aware import HorizonPolicy
from agentkv.sim.two_tier import CPUCachePolicy


@dataclass(slots=True)
class _Chunk:
    ref_cnt: int = 0
    chunk_id: int = 0


class HintPolicyAdapter(CPUCachePolicy):
    def __init__(self,capacity,*,hint_tier='perfect',block_size=16,diagnose_deadlines=False,**options):
        if hint_tier not in ('none','perfect','noisy','online','final'):raise ValueError('Unknown hint tier')
        self.deadline_decisions = [] if diagnose_deadlines else None
        scoring = options.setdefault('scoring','lru_final' if hint_tier=='final' else 'remaining' if hint_tier in ('noisy','online') else 'deadline')
        if scoring=='remaining' and options.get('residual_table') is None:
            options['residual_table'] = RemainingLifeTable.load(tier=hint_tier)
        if scoring=='horizon':
            options.setdefault('perfect_probability',hint_tier=='perfect')
            if hint_tier in ('online','noisy') and options.get('probability_table') is None:
                options['probability_table']=HorizonProbabilityTable.load(tier=hint_tier)
        self.now = 0.;self.hint_tier = hint_tier;self.block_size = block_size
        self.policy = (HorizonPolicy if scoring=='horizon' else HintAwarePolicy)(capacity,clock=lambda:self.now,**options)
        self.requests = {};self.active = {}

    def register_request(self,request_id,request,instance):self.requests[request_id] = (request,instance)
    def __contains__(self,key):return self.policy.get(key) is not None
    def __len__(self):return len(self.policy.chunks)

    def on_request(self,request_id,blocks,*,time_s=0.):
        self.now = time_s
        request,instance = self.requests.pop(request_id)
        name = f'{instance.root_id}#{instance.instance_id}'
        column = f'hint_{self.hint_tier}_ms'
        envelope = None
        if self.hint_tier=='final':
            # This baseline consumes only identity/turn/final, never predictions.
            signal = dict(chain_id=request['chain_id'],chain_turn=request['chain_turn'],
                          final=request['final'],hint_perfect_ms=None if request['final'] else 0.)
            envelope = make_envelope(signal,'perfect',str(request_id),name)
        elif self.hint_tier!='none' and column in request and (request.get(column) is not None or request['final']):
            envelope = make_envelope(request,self.hint_tier,str(request_id),name)
        positions = {key:(i+1)*self.block_size for i,key in enumerate(blocks)}
        context = SimpleNamespace(req_id=request_id,kv_hints=envelope,kv_transfer_params=None,
            instance_id=name,chain_id=request['chain_id'],get_offload_key_position=positions.get)
        self.active[request_id] = context
        # Observe ownership/return at actual admission, retaining E0's recency
        # convention: only completed writes promote the CPU LRU ranks.
        self.policy.on_store_miss(blocks,context)

    def on_access(self,key,*,time_s=0.):
        self.now = time_s
        if self.policy.get(key) is None:self.policy.insert(key,_Chunk())

    def on_finished(self,request_id,blocks,*,stored=True,time_s=0.):
        self.now = time_s
        context = self.active.pop(request_id)
        self.policy.on_request_finished((blocks if stored else (),),set(),set(),context)

    def evict(self,n,protected=frozenset(),*,time_s=0.):
        self.now = time_s
        if self.deadline_decisions is not None:
            overdue=set();waiting=overdue_chains=0
            for chain in self.policy.chains.values():
                if chain.mode=='waiting' and chain.keys:
                    waiting += 1
                    if time_s>chain.deadline:
                        overdue.update(chain.keys);overdue_chains += 1
            total=len(self.policy.chunks)
            self.deadline_decisions.append(dict(time_s=time_s,needed=n,cpu_blocks=total,overdue_blocks=len(overdue),
                overdue_fraction=len(overdue)/total if total else 0.,waiting_chains=waiting,overdue_chains=overdue_chains))
        victims = self.policy.evict(n,set(protected))
        return None if victims is None else [key for key,_ in victims]
