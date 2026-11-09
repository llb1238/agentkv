"""Re-audit the callback probe from compact metadata and lossless logs, CPU only."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import struct

import pyarrow.parquet as pq
from agentkv.provenance import git_commit
from agentkv.bench.e0 import write_json
from agentkv.bench.callback_probe import METHODS,monotone
from agentkv.bench.trace_loader import sha256_file


def read(path):return json.loads(path.read_text())


def events(directory):
    compressed=sorted(directory.glob('*.jsonl.gz'))
    for path in compressed or sorted(directory.glob('*.jsonl')):
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as source:
            for line in source:yield json.loads(line)


def audit(directory):
    commit=git_commit();original=read(directory/'probe-analysis.json');run=read(directory/'run.json')
    assert sha256_file(directory/'run.json')==original['source_run_sha256']
    rows=pq.read_table(directory/'request-metadata.parquet').to_pylist();clients={r['request_id']:r for r in rows if r['send_ns'] is not None}
    coverage=pq.read_table(directory/'callback-coverage.parquet').to_pylist()
    assert len(clients)==len(coverage)==original['client_sent_with_hint']
    known=set(clients);hints=defaultdict(set);counts=Counter();expected=Counter();needed=Counter();refs=Counter();order=Counter()
    n_unready=n_absent=evicted=0;footer=0;capacity=None
    for e in events(directory/'policy-observations'):
        kind=e['event']
        if kind=='probe_open':capacity=e['capacity']
        if kind=='probe_close':footer+=1;continue
        if kind not in METHODS:continue
        counts[kind]+=1;expected[e['ns'],METHODS.index(kind)]+=1
        if kind=='evict' and e['returned'] is not None:evicted+=e['returned']
        hint=e.get('hint')
        if hint and hint['valid']:hints[kind].add(hint['message_id'])
        if kind=='on_request_finished':
            groups=e['position_groups'];rflat=[r for group in e['ref_count_groups'] for r in group]
            assert [len(g) for g in e['key_groups']]==[len(g) for g in groups]==[len(g) for g in e['ref_count_groups']]
            order['requests']+=1;order['groups']+=len(groups);order['missing_positions']+=sum(v is None for group in groups for v in group)
            order['nonmonotone_requests']+=not all(monotone(group) for group in groups)
            n_unready+=any(v is not None and v<0 for v in rflat);n_absent+=any(v is None for v in rflat)
            for r in rflat:refs['absent' if r is None else 'not_ready' if r<0 else 'pinned_ready' if r>0 else 'ready_unpinned']+=1
    observed=Counter();first=None;last=None;record=struct.Struct('<QB')
    for path in sorted((directory/'policy-observations').glob('*.bin')):
        previous=-1
        with path.open('rb') as source:
            while chunk:=source.read(record.size*65536):
                assert len(chunk)%record.size==0
                for ns,code in record.iter_unpack(chunk):
                    assert ns>=previous and code<len(METHODS);previous=ns
                    first=ns if first is None else min(first,ns);last=ns if last is None else max(last,ns)
                    observed[METHODS[code]]+=1
                    if expected[ns,code]:needed[ns,code]+=1
    assert needed==expected
    for name in ('on_store_miss','touch','on_request_finished','evict'):assert counts[name]==observed[name]
    assert dict(observed)==original['scalar_callback_counts'] and dict(refs)==original['finished_chunk_states']
    assert order['groups']==original['key_order']['groups'] and order['nonmonotone_requests']==original['key_order']['nonmonotone_requests']
    union=set().union(*hints.values())
    assert len(union&known)==original['plugin_unique_valid_any'] and len(hints['on_request_finished']&known)==original['plugin_unique_valid_finished']
    assert sorted(known-union)==original['missing_plugin_hints'] and sorted(known-hints['on_request_finished'])==original['missing_finished_hints']
    partial=sum(r['finalized_keys'] is not None and r['finalized_keys']!=r['full_prompt_blocks'] for r in coverage);empty=sum(r['finalized_keys']==0 for r in coverage)
    last_end=max(r['end_ns'] for r in clients.values())
    result=dict(git_commit=commit,measurement_git_commit=run['git_commit'],critical_context_verification_passed=known==union==hints['on_request_finished'],
        client_requests=len(known),valid_hint_requests=len(union),valid_finished_requests=len(hints['on_request_finished']),
        raw_context_calls_match_binary=True,observed_scalar_callbacks=sum(observed.values()),scalar_shutdown_finalization_complete=footer>0,
        scalar_first_ns=first,scalar_last_ns=last,last_client_end_ns=last_end,scalar_last_minus_last_client_end_ms=(last-last_end)/1e6,
        requests_finished_keys_not_full_prompt=partial,requests_empty_finished_keys=empty,
        requests_with_not_ready_keys_at_finished=n_unready,requests_with_absent_keys_at_finished=n_absent,
        full_prompt_blocks=sum(r['full_prompt_blocks'] for r in coverage),finished_key_observations=sum(r['finalized_keys'] or 0 for r in coverage),
        cpu_capacity=capacity,observed_evicted_keys=evicted,
        observed_inventory_at_log_end=observed['insert']-evicted-observed['remove'],
        observed_ref_transition_balance=observed['mark_evictable']-observed['insert']-observed['mark_non_evictable'],
        original_analysis_sha256=sha256_file(directory/'probe-analysis.json'),source_run_sha256=sha256_file(directory/'run.json'),
        limitations=['No probe_close footer was saved; do not claim complete scalar callback counts through process teardown.'
                     ' Critical request-context events are flushed synchronously and all sent message IDs are covered.'] if not footer else [])
    write_json(directory/'verification.json',result)
    marker='\n## 关键差异与观测边界（复核）\n'
    text=(directory/'README.md').read_text().split(marker)[0]
    text+=marker+'\n'
    text+='hint通路与顺序可用，但不能把真实回调直接当作当前模拟器事件。按用户要求停在步骤1：未实现vllm_plugin.py，未开展HintAware双适配一致性测试，也未开展10分钟GPU冒烟。\n\n'
    routes=original['hint_routes_callback_counts']
    text+=f"- direct kv_hints出现/有效：{routes['direct_present']}/{routes['direct_valid']}次；nested出现/有效：{routes['nested_present']}/{routes['nested_valid']}次（按上下文回调计数）。插件按message_id去重收到{len(union&known)}个有效hint，已发送{len(known)}个。\n"
    text+=f"- {order['groups']}个key group；原始位置非单调{order['nonmonotone_requests']}，缺失{order['missing_positions']}；{partial}个请求的finished key数不等于完整prompt整块数，其中{empty}个请求列表为空。sorted并不代表complete。\n"
    text+=f"- {n_unready}个请求在finished时仍有未ready CPU块（{refs['not_ready']}个块观测），{n_absent}个请求有已不在CPU字典中的块（{refs['absent']}个观测）。不能把finished视为全部新CPU副本同时可读；absent本身不单独证明是驱逐还是存储失败。\n\n"
    timing=original['timings_ms']
    def ms(name,quantile):
        value=timing.get(name,{}).get(quantile)
        return 'NA' if value is None else f'{value:.3f}'
    text+='| 事件 | 当前模拟器 | 此次真实观测 |\n|---|---|---|\n'
    text+=f"| 首次观察 | admission时一次给完整prefix | store_miss为新增候选；{original['requests_without_store_miss']}请求无此回调，{original['requests_with_multiple_store_miss']}请求分多次；首次较首次调度P50晚{ms('first_store_miss_minus_server_start_ms','p50')}ms |\n"
    text+=f"| CPU写入准备 | 请求结束才统一写镜像 | {original['requests_store_before_generation_end']}请求在生成结束前就准备store；首个store减生成结束P50={ms('first_store_miss_minus_server_end_ms','p50')}ms |\n"
    text+=f"| finished | 全量请求prefix且已ready | 减生成结束P50={ms('policy_finished_minus_server_end_ms','p50')}ms、P99={ms('policy_finished_minus_server_end_ms','p99')}ms；有缺块/未ready |\n"
    text+='| 缓存访问 | 完整prompt统一更新CPU recency | 实际finished可能只含已观察到的CPU访问/存储块 |\n\n'
    text+=f"本次记录到{sum(observed.values())}个二进制方法入口；关键上下文事件逐条与二进制时间点匹配。观测到的insert−evict−remove={result['observed_inventory_at_log_end']}，CPU容量={capacity}；mark_evictable−insert−mark_non_evictable={result['observed_ref_transition_balance']}。\n\n"
    if not footer:text+='这些计数自洽不能替代退出footer：EngineCore拆除时没有保存probe_close，不能断言进程拆除尾部所有标量回调都已记录。原始不完整标记保留，不补造footer。全部关键上下文回调按调用同步flush，其请求覆盖和时间点另作核验。\n\n'
    text+='建议下一步先修正模拟器的分段store、异步ready和实际回调key覆盖范围，保留冻结评分公式；校验后再决定是否需要扩展生命周期观测或调整适配方式。这是建议，未自动执行。原冻结策略的CPU模拟结果不应被当作已在真实回调下验证。\n\n'
    text+='[独立复核](verification.json) · [测试](tests.json) · [预注册](../../docs/m5_vllm_callback_probe_design.md)\n'
    (directory/'README.md').write_text(text)
    print(json.dumps(result,indent=2),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    audit(p.parse_args().run_dir)


if __name__=='__main__':main()
