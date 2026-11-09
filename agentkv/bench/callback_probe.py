"""Five-minute, LRU-identical GPU callback feasibility gate; never runs hint policy."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import ROOT,write_json
from agentkv.bench.shared_pilot import run as run_server
from agentkv.bench.trace_loader import sha256_file

BACKEND_FILES=('vllm/v1/kv_offload/cpu/policies/base.py','vllm/v1/kv_offload/cpu/policies/lru.py',
    'vllm/v1/kv_offload/cpu/manager.py','vllm/v1/kv_offload/base.py',
    'vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py')
METHODS=('get','insert','remove','touch','on_store_miss','on_request_finished','evict','clear','mark_evictable','mark_non_evictable')


def read(path):return json.loads(path.read_text())


def describe(values):
    if not values:return dict(count=0)
    return dict(count=len(values),minimum=float(min(values)),p50=float(np.median(values)),p90=float(np.quantile(values,.9)),
                p99=float(np.quantile(values,.99)),maximum=float(max(values)))


def monotone(values):
    return all(v is not None for v in values) and all(a<=b for a,b in zip(values,values[1:]))


def analyze(directory):
    commit=git_commit();run=read(directory/'run.json')
    records=pq.read_table(directory/'requests.parquet').to_pylist()
    records=[r for r in records if r.get('send_ns') is not None];clients={r['request_id']:r for r in records}
    life=defaultdict(dict)
    for path in sorted((directory/'server-observations').glob('requests-*.jsonl')):
        for line in path.open():
            event=json.loads(line);name=event['event']
            if name in ('request_received','request_scheduled','request_finished','prefill'):
                life[event['request_id']].setdefault(name,event['ns'])
    methods=Counter();raw_files=[];per_message=defaultdict(lambda:defaultdict(list));context=[];closed=0
    order=Counter();refs=Counter();hint_routes=Counter();valid_ids=defaultdict(set);scalar_records=0;scalar_counts=Counter()
    import struct
    binary=struct.Struct('<QB')
    for path in sorted((directory/'policy-observations').glob('*.bin')):
        counts=Counter();previous=-1;total=0
        with path.open('rb') as source:
            while chunk:=source.read(binary.size*65536):
                assert len(chunk)%binary.size==0
                for ns,code in binary.iter_unpack(chunk):
                    assert ns>=previous and code<len(METHODS);previous=ns;counts[METHODS[code]]+=1;total+=1
        scalar_records+=total;scalar_counts.update(counts)
    for path in sorted((directory/'policy-observations').glob('*.jsonl')):
        raw_files.append(path)
        for line in path.open():
            event=json.loads(line);method=event['event']
            if method=='probe_close':closed+=1;methods.update(event['counts']);continue
            hint=event.get('hint')
            if hint:
                hint_routes.update({key:int(hint[key]) for key in ('direct_present','nested_present','direct_valid','nested_valid')})
                if hint['valid']:
                    msg=hint['message_id'];valid_ids[method].add(msg)
                    if msg in clients:per_message[msg][method].append(event)
                else:msg=None
                context.append(dict(method=method,ns=event['ns'],request_id=event['request_id'],message_id=msg,
                                    hint_valid=hint['valid'],hint_source=hint['selected_source'],key_count=event.get('count',sum(map(len,event.get('key_groups',[]))))))
            if method=='on_request_finished':
                groups=event['position_groups'];flat=[p for group in groups for p in group]
                order['requests']+=1;order['groups']+=len(groups);order['missing_positions']+=sum(p is None for p in flat)
                order['nonmonotone_requests']+=not all(monotone(group) for group in groups)
                order['nonmonotone_flattened']+=not monotone(flat)
                order['multi_group_requests']+=len(groups)>1
                for group in event['ref_count_groups']:
                    for ref in group:refs['absent' if ref is None else 'not_ready' if ref<0 else 'pinned_ready' if ref>0 else 'ready_unpinned']+=1
    if closed:assert methods==scalar_counts,('Truncated callback stream',methods,scalar_counts)
    timing=defaultdict(list);coverage=[]
    for msg,client in clients.items():
        events=per_message[msg];all_events=[e for values in events.values() for e in values]
        request_ids={e['request_id'] for e in all_events};assert len(request_ids)<=1
        request_id=next(iter(request_ids),None)
        server=life.get(request_id,{})
        if not request_id:
            sid=client.get('server_request_id')
            matches=[k for k in life if sid and (k==sid or k.startswith(sid+'-0-') or k==sid+'-0')]
            assert len(matches)<=1
            if matches:request_id=matches[0];server=life[request_id]
        misses=sorted(events['on_store_miss'],key=lambda e:e['ns']);ends=sorted(events['on_request_finished'],key=lambda e:e['ns'])
        full=client['prompt_len']//16
        row=dict(request_id=msg,server_request_id=request_id,status=client['status'],prompt_len=client['prompt_len'],full_prompt_blocks=full,
            policy_hint_seen=bool(all_events),finished_callbacks=len(ends),store_miss_callbacks=len(misses),store_miss_keys=sum(e['count'] for e in misses),
            send_ns=client['send_ns'],client_end_ns=client['end_ns'],server_start_ns=server.get('request_scheduled'),server_end_ns=server.get('request_finished'),
            first_store_miss_ns=misses[0]['ns'] if misses else None,last_store_miss_ns=misses[-1]['ns'] if misses else None,
            policy_finished_ns=ends[0]['ns'] if ends else None,finalized_keys=sum(len(g) for g in ends[0]['key_groups']) if ends else None)
        for callback in ('first_store_miss','last_store_miss','policy_finished'):
            stamp=row[callback+'_ns']
            for boundary in ('server_start','server_end','client_end'):
                target=row[boundary+'_ns']
                if stamp is not None and target is not None:timing[callback+'_minus_'+boundary+'_ms'].append((stamp-target)/1e6)
        coverage.append(row)
    client_ids=set(clients);any_ids=set.union(set(),*(v for v in valid_ids.values()))
    finished_ids=valid_ids['on_request_finished']
    summary=dict(git_commit=commit,stage='callback-feasibility',algorithm='unchanged installed LRU',
        client_sent_with_hint=run['summary']['hint_sent'],sent_rows=len(clients),successes=sum(r['status']=='success' for r in records),
        failures=sum(r['status']!='success' for r in records),frontend_hint_received=run['summary']['hint_received'],
        plugin_unique_valid_any=len(any_ids&client_ids),plugin_unique_valid_finished=len(finished_ids&client_ids),
        sent_equals_plugin_any=client_ids==any_ids,sent_equals_plugin_finished=client_ids==finished_ids,
        foreign_plugin_hints=sorted(any_ids-client_ids),missing_plugin_hints=sorted(client_ids-any_ids),
        missing_finished_hints=sorted(client_ids-finished_ids),hint_routes_callback_counts=dict(hint_routes),key_order=dict(order),finished_chunk_states=dict(refs),
        store_miss_callbacks_per_request=describe([r['store_miss_callbacks'] for r in coverage]),
        requests_with_multiple_store_miss=sum(r['store_miss_callbacks']>1 for r in coverage),
        requests_without_store_miss=sum(r['store_miss_callbacks']==0 for r in coverage),
        requests_store_miss_keys_not_full_prompt=sum(r['store_miss_keys']!=r['full_prompt_blocks'] for r in coverage),
        requests_store_before_generation_end=sum(r['first_store_miss_ns'] is not None and r['server_end_ns'] is not None and r['first_store_miss_ns']<r['server_end_ns'] for r in coverage),
        timings_ms={name:describe(values) for name,values in timing.items()},scalar_callbacks=scalar_records,scalar_callback_counts=dict(scalar_counts),
        closed_policy_logs=closed,policy_log_files=len(raw_files),source_run_sha256=sha256_file(directory/'run.json'))
    differences=[]
    if not summary['sent_equals_plugin_any'] or not summary['sent_equals_plugin_finished']:differences.append('hint/request-finalization coverage differs from one callback per request')
    if order['nonmonotone_requests'] or order['missing_positions']:differences.append('original prefix order cannot be established for every group')
    if summary['requests_with_multiple_store_miss']:differences.append('incremental on_store_miss rather than single full-prefix observation at admission')
    if summary['requests_store_miss_keys_not_full_prompt']:differences.append('on_store_miss observes new store candidates rather than every full request block')
    if summary['requests_store_before_generation_end']:differences.append('store preparation occurs during request execution rather than only at completion')
    if refs['not_ready']:differences.append('CPU chunks can still be not-ready at on_request_finished')
    if not closed or closed!=len(raw_files):differences.append('policy stream was not finalized; observation completeness not established')
    if summary['failures']:differences.append('requests failed; feasibility is not a clean success')
    summary.update(significant_differences=differences,stop_before_hint_adapter=bool(differences))
    pq.write_table(pa.Table.from_pylist(coverage),directory/'callback-coverage.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(context),directory/'callback-context.parquet',compression='zstd')
    full=pq.read_table(directory/'requests.parquet');pq.write_table(full.drop(['prompt_token_ids']),directory/'request-metadata.parquet',compression='zstd')
    write_json(directory/'probe-analysis.json',summary)
    lines=['# vLLM CachePolicy：LRU等价探针（5分钟）','',f"代码 `{commit}`；vLLM `{read(directory/'server-manifest.json')['vllm_commit']}`。",'',
        'config-11、λ=0.7、perfect、稳态初始化、人口种子0、T=300秒、预热0；用于回调可行性，不能当作稳态性能测量。GPU1显存共享，两卡dmon另存。',
        '', '## Hint与key顺序','',f"实际发送带hint请求 {summary['client_sent_with_hint']}；前端有效hint {summary['frontend_hint_received']}；插件任意回调收到的有效hint（按message_id去重）{summary['plugin_unique_valid_any']}；finished回调覆盖 {summary['plugin_unique_valid_finished']}。成功 {summary['successes']}，失败 {summary['failures']}。",'',
        f"原始key groups：{order['requests']}个finished回调，缺失位置{order['missing_positions']}，组内非单调{order['nonmonotone_requests']}，扁平化非单调{order['nonmonotone_flattened']}，多组请求{order['multi_group_requests']}。探针没有先排序再检查。",'',
        '## 回调阶段','',f"多次store_miss的请求{summary['requests_with_multiple_store_miss']}；无store_miss请求{summary['requests_without_store_miss']}；store_miss总块数不等于完整prompt整块数的请求{summary['requests_store_miss_keys_not_full_prompt']}。",'',
        f"首次store准备早于服务端生成结束的请求{summary['requests_store_before_generation_end']}。finished时块观测：{dict(refs)}。",'',
        '| 时间差（ms，前者减后者） | n | P50 | P90 | P99 | min/max |','|---|---:|---:|---:|---:|---:|']
    for name,d in summary['timings_ms'].items():
        lines.append(f"| {name} | {d['count']} | {d['p50']:.3f} | {d['p90']:.3f} | {d['p99']:.3f} | {d['minimum']:.3f}/{d['maximum']:.3f} |")
    lines+=['','开始=服务端首次SCHEDULED，结束=Scheduler._free_request首次入口（生成已完成）；客户端结束是流式HTTP关闭。时钟为同主机monotonic_ns。负数表示早于边界。',
        '', '## 门禁结论','',f"停止后续步骤2–4：{summary['stop_before_hint_adapter']}。"]
    lines.extend('- '+d for d in differences)
    lines+=['','模拟器是在admission时一次登记整个请求；完成时统一写CPU镜像并视为ready，然后finished。真实事件按上表报告，不自动重解释为模拟事件；策略实现/10分钟冒烟需用户决定后再继续。',
        '', '[分析](probe-analysis.json) · [逐请求关联](callback-coverage.parquet) · [上下文回调](callback-context.parquet) · [客户端元数据](request-metadata.parquet) · [run.json](run.json)',
        '', '原始完整requests.parquet留在本机（含原prompt数组，不纳入Git）；追踪副本request-metadata.parquet只删除该大列。原始policy JSONL另以gzip无损保存，二进制标量回调格式<QB：monotonic_ns uint64、方法编号uint8；方法顺序见各日志probe_open。所有源码仅以Git提交定位，未复制到结果目录。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def run(output,gpu_index=1,port=8014):
    commit=git_commit();backend=ROOT.parent/'vllm';backend_commit=git_commit(repository=backend)
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(row.split(',')[0]):float(row.split(',')[1]) for row in snapshot.splitlines()}
    if free[gpu_index]<58000:raise ValueError('Insufficient free MiB for unchanged config-11; do not alter matrix')
    run_server(output,gpu_index,port,duration=300,warmup=0,rate=.7,population_seed=0,run_id='m5-callback-probe-seed0',analyze_output=False,
        cache_policy='LRUProbePolicy',cache_policy_module='agentkv.policy.lru_probe',hint_tier='perfect',
        extra_environment=dict(AGENTKV_POLICY_PROBE='1',AGENTKV_POLICY_OBSERVATION_DIR=str(output/'policy-observations')))
    summary=analyze(output)
    write_json(output/'probe-run.json',dict(git_commit=commit,vllm_commit=backend_commit,status='complete',
        backend_interface_sha256={p:sha256_file(backend/p) for p in BACKEND_FILES},source_run_sha256=sha256_file(output/'run.json'),
        source=dict(workload_sha256=read(output/'run.json')['source']['workload_sha256']),stop_before_hint_adapter=summary['stop_before_hint_adapter']))
    # Preserve original bytes through reversible compression; raw copies stay local.
    for path in (output/'policy-observations').glob('*.jsonl'):
        data=path.read_bytes()
        with (path.with_suffix(path.suffix+'.gz')).open('wb') as dest:
            with gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0) as archive:archive.write(data)
        assert gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())==data
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu-index',type=int,choices=(0,1),default=1)
    p.add_argument('--port',type=int,default=8014);a=p.parse_args();run(a.output_dir,a.gpu_index,a.port)


if __name__=='__main__':main()
