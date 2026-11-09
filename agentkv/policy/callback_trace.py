"""Lossless policy-call arguments, returns and observable mutable chunk states."""
import atexit
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

_TRACES = []
METHODS = ('get','insert','remove','touch','on_store_miss','on_request_finished',
           'mark_evictable','mark_non_evictable','evict','clear')


def flush_traces(reason=None):
    for trace in _TRACES:
        if not trace.closed:
            if reason:trace.write(dict(event='checkpoint',ns=time.monotonic_ns(),reason=reason))
            trace.flush()


def close_traces():
    for trace in _TRACES:trace.close()


def plain(value):
    return value.model_dump(mode='json') if hasattr(value,'model_dump') else value


class CallbackTrace:
    def __init__(self, target, capacity):
        self.target = target
        directory = Path(os.environ['AGENTKV_CALLBACK_TRACE_DIR'])
        directory.mkdir(parents=True,exist_ok=True)
        self.path = directory/f'callbacks-{os.getpid()}-{id(self)}.jsonl.gz'
        self.sink = gzip.GzipFile(filename='',mode='wb',fileobj=self.path.open('wb'),mtime=0)
        self.keys = {};self.contexts = {};self.observed = {};self.stack = []
        self.counts = Counter();self.serial = 0;self.closed = False;self.last_flush = time.monotonic()
        self.write(dict(event='open',format='agentkv.callbacks.v1',capacity=capacity,
            policy=type(target).__name__,ns=time.monotonic_ns(),pid=os.getpid()))
        if hasattr(target,'adapter') and hasattr(target.adapter.policy,'clock'):
            clock=target.adapter.policy.clock
            def observed_clock():
                value=clock()
                self.write(dict(event='clock',call=self.stack[-1] if self.stack else None,value=value))
                return value
            target.adapter.policy.clock=observed_clock
        self.flush();_TRACES.append(self);atexit.register(self.close)

    def write(self, data):
        self.sink.write((json.dumps(data,separators=(',',':'))+'\n').encode())

    def key(self, key):
        if key not in self.keys:
            index = len(self.keys);self.keys[key] = index
            self.write(dict(event='key',key=index,hex=bytes(key).hex()))
        return self.keys[key]

    def chunk(self, key, chunk):
        return [self.key(key),None if chunk is None else [chunk.ref_cnt,chunk.chunk_id]]

    def context(self, context):
        data = dict(req_id=context.req_id,kv_hints=plain(context.kv_hints),
            kv_transfer_params=plain(context.kv_transfer_params),num_prompt_tokens=context.num_prompt_tokens,
            load_tier_filter=[[m.medium.value if m.medium is not None else None,
                               m.locality.value if m.locality is not None else None]
                              for m in context.load_tier_filter.matchers],
            manager_state_types=[f'{t.__module__}.{t.__qualname__}' for t in context._state],
            positions=[[self.key(k),p] for k,p in context._offload_key_positions.items()],
            prefix=[self.key(k) for k in context.get_offload_prefix_keys()])
        signature = json.dumps(data,separators=(',',':'))
        if signature not in self.contexts:
            index = len(self.contexts);self.contexts[signature] = index
            self.write(dict(event='context',context=index,**data))
        return self.contexts[signature]

    def delta(self, keys):
        updates = []
        for key in keys:
            chunk = self.target.chunks.get(key)
            value = None if chunk is None else (chunk.ref_cnt,chunk.chunk_id)
            if key not in self.observed or self.observed[key] != value:
                updates.append([self.key(key),value]);self.observed[key] = value
        return updates

    def invoke(self, method, args, callback):
        stamp = time.monotonic_ns();index = self.serial;self.serial += 1
        parent = self.stack[-1] if self.stack else None
        args = list(args);focus = []
        if method in ('touch','on_store_miss'):
            args[0] = list(args[0]);focus = args[0]
            encoded = [list(map(self.key,focus)),self.context(args[1])]
        elif method == 'on_request_finished':
            args[0] = tuple(tuple(g) for g in args[0]);args[1] = set(args[1]);args[2] = set(args[2])
            focus = [k for group in args[0] for k in group]
            encoded = [[list(map(self.key,g)) for g in args[0]],
                sorted(map(self.key,args[1])),sorted(map(self.key,args[2])),self.context(args[3])]
        elif method == 'evict':
            args[1] = set(args[1]);encoded = [args[0],sorted(map(self.key,args[1]))]
            # ref_cnt is mutated by the manager outside policy callbacks.
            # Synchronize all live chunks at every decision, not an inferred
            # transition based on mark_non_evictable (which precedes 0 -> 1).
            focus = list(self.target.chunks)
        elif method == 'insert':
            focus = [args[0]];encoded = [self.key(args[0]),[args[1].ref_cnt,args[1].chunk_id]]
        elif method == 'clear':
            focus = list(self.target.chunks);encoded = []
        else:
            focus = [args[0]];encoded = [self.key(args[0])]
        self.write(dict(event='begin',call=index,parent=parent,method=method,ns=stamp,
            args=encoded,refs=self.delta(focus)))
        self.stack.append(index);self.counts[method] += 1
        try:
            result = callback(*args)
        except BaseException as exc:
            self.write(dict(event='error',call=index,ns=time.monotonic_ns(),error=repr(exc)))
            self.flush();raise
        finally:
            self.stack.pop()
        if method == 'get':
            encoded_result = None if result is None else [result.ref_cnt,result.chunk_id]
        elif method == 'evict':
            encoded_result = None if result is None else [self.chunk(k,c) for k,c in result]
            if result:focus = [k for k,_ in result]
        else:encoded_result = None
        self.write(dict(event='end',call=index,ns=time.monotonic_ns(),result=encoded_result,
            refs=self.delta(focus),cache_blocks=len(self.target.chunks)))
        if method == 'on_request_finished' or time.monotonic()-self.last_flush >= 1:
            self.flush()
        return result

    def flush(self):
        self.sink.flush();self.last_flush = time.monotonic()

    def close(self):
        if self.closed:return
        self.write(dict(event='close',ns=time.monotonic_ns(),counts=dict(self.counts),calls=self.serial))
        self.flush();self.sink.close();self.closed = True


def traced_method(method):
    def invoke(self,*args):
        callback = getattr(super(CallbackTraceMixin,self),method)
        trace = getattr(self,'trace',None)
        return callback(*args) if trace is None else trace.invoke(method,args,callback)
    invoke.__name__ = method
    return invoke


class CallbackTraceMixin:
    def __init__(self,cache_capacity):
        super().__init__(cache_capacity)
        self.trace = CallbackTrace(self,cache_capacity)

    get=traced_method('get')
    insert=traced_method('insert')
    remove=traced_method('remove')
    touch=traced_method('touch')
    on_store_miss=traced_method('on_store_miss')
    on_request_finished=traced_method('on_request_finished')
    mark_evictable=traced_method('mark_evictable')
    mark_non_evictable=traced_method('mark_non_evictable')
    evict=traced_method('evict')
    clear=traced_method('clear')

    def __del__(self):
        try:
            if hasattr(self,'trace'):self.trace.close()
        except Exception:pass
