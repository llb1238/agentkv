"""Observation-only subclass of the installed vLLM LRU implementation.

No policy logic is copied or reimplemented: every CachePolicy method calls its
super method once, with the original ordering and classifications.
"""
import atexit
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import struct
import time

from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from agentkv.hints import validate_envelope

METHODS=('get','insert','remove','touch','on_store_miss','on_request_finished','evict','clear','mark_evictable','mark_non_evictable')
SCALAR=struct.Struct('<QB')
_LOGS=[]


def flush_logs():
    for log in _LOGS:
        if not log.closed:log.flush()


def close_logs():
    for log in _LOGS:log.close()


def checkpoint_logs(reason):
    # This is not a close footer: shutdown can still invoke policy methods.
    for log in _LOGS:
        if not log.closed:
            log.event('probe_checkpoint',time.monotonic_ns(),reason=reason,counts=dict(log.counts))


def get(value,name,default=None):
    return value.get(name,default) if isinstance(value,dict) else getattr(value,name,default)


def normalize(envelope):
    if envelope is None:return None
    return envelope.model_dump(mode='json') if hasattr(envelope,'model_dump') else envelope


def inspect_hints(context):
    direct=normalize(get(context,'kv_hints'));nested=normalize(get(get(context,'kv_transfer_params'),'kv_hints'))
    result=dict(direct_present=direct is not None,nested_present=nested is not None)
    for name,value in (('direct',direct),('nested',nested)):
        try:validate_envelope(value);valid=True
        except (AttributeError,KeyError,TypeError,ValueError):valid=False
        result[name+'_valid']=valid
    name='direct' if direct is not None else 'nested';selected=direct if direct is not None else nested
    result.update(selected_source=name if selected is not None else None,valid=result[name+'_valid'])
    if result['valid']:
        payload=selected['actions'][0]['payload']
        result.update(message_id=selected['message_id'],final=payload['final'],session_id=payload.get('session_id'),turn=payload['turn'])
    return result


class ProbeLog:
    def __init__(self,capacity):
        directory=os.environ.get('AGENTKV_POLICY_OBSERVATION_DIR')
        if not directory:raise ValueError('AGENTKV_POLICY_OBSERVATION_DIR is required for LRU probe')
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        name=f'policy-{os.getpid()}-{id(self)}'
        self.binary=(root/(name+'.bin')).open('wb',buffering=1024*1024)
        self.context=(root/(name+'.jsonl')).open('w',buffering=1024*1024)
        self.counts=Counter();self.hints=defaultdict(set);self.closed=False
        self.context.write(json.dumps(dict(event='probe_open',ns=time.monotonic_ns(),pid=os.getpid(),capacity=capacity,
            binary_record_format=SCALAR.format,binary_record_bytes=SCALAR.size,methods=METHODS))+'\n')
        self.flush();_LOGS.append(self);atexit.register(self.close)
        print('AGENTKV_LRU_PROBE '+str(root/name),flush=True)

    def mark(self,method):
        stamp=time.monotonic_ns();self.counts[method]+=1
        self.binary.write(SCALAR.pack(stamp,METHODS.index(method)))
        return stamp

    def event(self,method,stamp,*,context=None,**fields):
        record=dict(event=method,ns=stamp,**fields)
        if context is not None:
            hint=inspect_hints(context);record.update(request_id=get(context,'req_id'),hint=hint)
            if hint['valid']:self.hints[method].add(hint['message_id'])
        self.context.write(json.dumps(record,separators=(',',':'))+'\n')
        self.flush()

    def flush(self):
        self.binary.flush();self.context.flush()

    def close(self):
        if self.closed:return
        self.context.write(json.dumps(dict(event='probe_close',ns=time.monotonic_ns(),counts=dict(self.counts),
            unique_valid_hints={name:len(ids) for name,ids in self.hints.items()}))+'\n')
        self.flush();self.closed=True;self.binary.close();self.context.close()


class LRUProbePolicy(LRUCachePolicy):
    def __init__(self,cache_capacity):
        super().__init__(cache_capacity);self.probe=ProbeLog(cache_capacity)

    def __del__(self):
        try:
            if hasattr(self,'probe'):self.probe.close()
        except Exception:
            # Interpreter teardown may already have dismantled file objects.
            pass

    def get(self,key):
        self.probe.mark('get');return super().get(key)

    def insert(self,key,chunk):
        self.probe.mark('insert');return super().insert(key,chunk)

    def remove(self,key):
        self.probe.mark('remove');return super().remove(key)

    def touch(self,keys,req_context):
        stamp=self.probe.mark('touch');keys=list(keys)
        self.probe.event('touch',stamp,context=req_context,count=len(keys),positions=[req_context.get_offload_key_position(k) for k in keys])
        return super().touch(keys,req_context)

    def on_store_miss(self,keys,req_context):
        stamp=self.probe.mark('on_store_miss');keys=list(keys)
        self.probe.event('on_store_miss',stamp,context=req_context,count=len(keys),positions=[req_context.get_offload_key_position(k) for k in keys])
        return super().on_store_miss(keys,req_context)

    def on_request_finished(self,key_groups,insertion_only_keys,reused_keys,req_context):
        stamp=self.probe.mark('on_request_finished');groups=tuple(tuple(g) for g in key_groups)
        self.probe.event('on_request_finished',stamp,context=req_context,
            key_groups=[[bytes(k).hex() for k in g] for g in groups],
            position_groups=[[req_context.get_offload_key_position(k) for k in g] for g in groups],
            ref_count_groups=[[self.chunks[k].ref_cnt if k in self.chunks else None for k in g] for g in groups],
            insertion_only_count=len(insertion_only_keys),reused_count=len(reused_keys),resident_count=sum(k in self.chunks for g in groups for k in g))
        return super().on_request_finished(groups,insertion_only_keys,reused_keys,req_context)

    def evict(self,n,protected):
        stamp=self.probe.mark('evict');result=super().evict(n,protected)
        self.probe.event('evict',stamp,needed=n,protected=len(protected),returned=None if result is None else len(result))
        return result

    def clear(self):
        self.probe.mark('clear');return super().clear()

    def mark_evictable(self,key):
        self.probe.mark('mark_evictable');return super().mark_evictable(key)

    def mark_non_evictable(self,key):
        self.probe.mark('mark_non_evictable');return super().mark_non_evictable(key)
