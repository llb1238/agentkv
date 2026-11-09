"""Render audited CPU and GPU overhead evidence without rerunning measurements."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.bench.shared_pilot import parse_dmon
from agentkv.provenance import current_commit


def read(path):return json.loads(path.read_text())


def compare_evictions(first,second):
    a,b=[pq.read_table(p/'evictions.parquet').to_pylist() for p in (first,second)]
    assert len(a)==len(b)
    fields=('call','n','none','victims_sha256')
    assert all(tuple(x[k] for k in fields)==tuple(y[k] for k in fields) for x,y in zip(a,b))
    return len(a)


def cpu_evidence(directory):
    runs={k:read(directory/k/'run.json') for k in ('reference','optimized','vllm','none')}
    counts={k:compare_evictions(directory/'reference',directory/k) for k in ('optimized','vllm')}
    assert len({runs[k]['summary']['eviction_fingerprint'] for k in ('reference','optimized','vllm')})==1
    for k in ('optimized','vllm'):
        assert runs[k]['summary']['prefix_enumerations']==2259
        assert runs[k]['summary']['prefix_cache_entries']==0
    none_reference=read(directory/'none-reference/run.json')
    assert runs['none']['summary']['eviction_fingerprint']==none_reference['summary']['eviction_fingerprint']
    counts['none']=compare_evictions(directory/'none-reference',directory/'none')
    totals={}
    for k in ('reference','optimized','vllm'):
        table=pq.read_table(directory/k/'callbacks.parquet')
        totals[k]=table.group_by('method').aggregate([('elapsed_us','sum'),('elapsed_us','mean'),('elapsed_us','max')]).to_pylist()
    prefixes={}
    for k in ('reference-profile','optimized-profile'):
        profile=read(directory/k/'run.json')
        prefixes[k]=[f for f in profile['summary']['profile_functions'] if f['name']=='get_offload_prefix_keys']
    slow={k:sorted(pq.read_table(directory/k/'evictions.parquet').to_pylist(),key=lambda r:r['elapsed_us'],reverse=True)[:10]
          for k in ('reference','optimized','vllm')}
    stats={k:r['summary']['evict'] for k,r in runs.items()}
    return dict(eviction_checks=counts,evict=stats,prefix_profiles=prefixes,
        callback_totals=totals,slowest=slow,eviction_fingerprint=runs['vllm']['summary']['eviction_fingerprint'],
        cpu_gate_passed=stats['vllm']['p50_us']<500 and stats['vllm']['p99_us']<2000,
        source_commits={k:r['git_commit'] for k,r in runs.items()})


def histogram_text(h,q):
    low,high=h[q+'_ns']
    return f'{low/1000:.3f}–{high/1000:.3f}'


def gpu_pair(directory):
    pair=read(directory/'run.json')
    runs={k:read(directory/k/'overhead-run.json') for k in ('lru','hint-none')}
    monitors={}
    for name,r in runs.items():
        folder=directory/name
        assert sha256_file(folder/'run.json')==r['client_run_sha256']
        assert sha256_file(folder/'server-manifest.json')==r['server_manifest_sha256']
        assert sha256_file(folder/'request-metadata.parquet')==r['request_metadata_sha256']
        for file,digest in r['light_metrics_sha256'].items():
            assert sha256_file(folder/'server-observations'/file)==digest
        samples=parse_dmon((folder/'dmon.log').read_text())
        assert {s['gpu'] for s in samples}=={0,1}
        monitors[name]={str(gpu):sum(s['gpu']==gpu for s in samples) for gpu in (0,1)}
    assert runs['lru']['population_sha256']==runs['hint-none']['population_sha256']
    assert runs['lru']['source']==runs['hint-none']['source']
    hist={}
    for name in runs['lru']['histograms']:
        a,b=[r['histograms'][name] for r in runs.values()]
        def mean(h):return h['total_ns']/h['count']/1000
        hist[name]=dict(lru=a,hint_none=b,mean_difference_us=mean(b)-mean(a),
            p50_difference_us=[(b['p50_ns'][0]-a['p50_ns'][1])/1000,(b['p50_ns'][1]-a['p50_ns'][0])/1000],
            p99_difference_us=[(b['p99_ns'][0]-a['p99_ns'][1])/1000,(b['p99_ns'][1]-a['p99_ns'][0])/1000])
    request_maps={k:{r['request_id']:r for r in pq.read_table(directory/k/'request-metadata.parquet').to_pylist()} for k in runs}
    common=set(request_maps['lru']) & set(request_maps['hint-none'])
    fields=('sent_prompt_sha256','permutation_seed','prompt_len','max_tokens')
    assert all(tuple(request_maps['lru'][i][f] for f in fields)==tuple(request_maps['hint-none'][i][f] for f in fields) for i in common)
    for k in runs:
        for path in (directory/k/'server-observations').glob('light-*.json'):
            data=read(path)
            assert data['closed']
            for group in ('histograms','window_histograms'):
                assert all(sum(h['buckets'].values())==h['count'] for h in data[group].values())
    return dict(pair=pair,runs=runs,histogram_comparison=hist,common_requests=len(common),common_prompt_metadata_identical=True,dmon_samples=monitors)


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=directory/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65))
    for name,label in [('reference','Reference'),('vllm','Optimized, complete wrapper')]:
        values=np.sort(pq.read_table(directory/'cpu'/name/'evictions.parquet')['elapsed_us'].to_numpy()/1000)
        axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=label,lw=1)
    axes[0].axvline(2,color='gray',ls='--',lw=.7)
    axes[0].set(xscale='log',xlabel='Uninstrumented eviction (ms)',ylabel='CDF',ylim=(0,1))
    for name,label in [('lru','LRU'),('hint-none','Hint + none')]:
        rows=pq.read_table(directory/'gpu'/name/'request-metadata.parquet').to_pylist()
        values=np.sort([r['ttft_ms'] for r in rows if r['in_window'] and r['status']=='success'])
        axes[1].plot(values,np.arange(1,len(values)+1)/len(values),label=label,lw=1)
    axes[1].set(xscale='log',xlabel='GPU trial TTFT (ms)',ylim=(0,1))
    for ax in axes:ax.legend(frameon=False,fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figures/f'latency_cdf.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7.1,3.2),sharex=True)
    for name,label in [('lru','LRU'),('hint-none','Hint + none')]:
        samples=[json.loads(s) for s in (directory/'gpu'/name/'metrics.jsonl').read_text().splitlines()]
        samples=[s for s in samples if s.get('values')]
        for ax,metric in zip(axes,('vllm:num_requests_waiting','vllm:num_requests_running')):
            ax.plot([s['elapsed_s']/60 for s in samples],[s['values'][metric] for s in samples],lw=.65,label=label,alpha=.8)
    for ax in axes:ax.axvspan(0,5,color='gray',alpha=.15);ax.grid(alpha=.2)
    axes[0].set(ylabel='Waiting');axes[0].legend(frameon=False,ncol=2,fontsize=7)
    axes[1].set(ylabel='Running',xlabel='Elapsed minutes',xlim=(0,20))
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figures/f'queues.{ext}',dpi=300)
    plt.close(fig)


def report(directory,first_attempt):
    cpu=cpu_evidence(directory/'cpu');current=gpu_pair(directory/'gpu');old=gpu_pair(first_attempt)
    write_json(directory/'cpu/validation.json',cpu)
    plot(directory)
    a,b=[current['runs'][k] for k in ('lru','hint-none')]
    lines=['# M5：回调优化与无 hint 接入开销','',
        '协议：[测量前登记](../../docs/m5_policy_overhead_design.md)，首个预注册提交 `4e63484`。固定 config-11、λ=1、人口种子3、稳态初始化；每组20分钟，预热5分钟。GPU1显存共享，保留其他进程；两卡 dmon 原样归档。vLLM固定提交 `4749dd5961841243596559b10ba0a22c48980c45`。','',
        '## CPU：关闭运行时插桩，原顺序回放','',
        '使用完整smoke序列，JSON/gzip解析与ReqContext更新在计时外；无cProfile的独立回放判断门槛。逐次比较有序key、None和请求驱逐数，而非仅比较命中率。最终顺序复测固定CPU32、OMP/BLAS线程1；cProfile另跑，环境见cpu/environment.json。','',
        '| 实现 | evict数 | P50 ms | P99 ms | 最大 ms |','|---|---:|---:|---:|---:|']
    for k,label in [('reference','原参考适配层+核心'),('optimized','优化适配层+核心'),('vllm','完整vLLM包装层+优化适配层+核心')]:
        v=cpu['evict'][k];lines.append(f"| {label} | {v['count']} | {v['p50_us']/1000:.4f} | {v['p99_us']/1000:.4f} | {v['maximum_us']/1000:.4f} |")
    lines+=['',f"CPU门槛：**{'通过' if cpu['cpu_gate_passed'] else '未通过'}**。两种优化路径全部2005次驱逐逐次一致；最新none路径全部1128次与原生LRU路径一致。驱逐指纹 `{cpu['eviction_fingerprint']}`。完整前缀枚举2259次，对应2259个请求，结束后缓存条目为0。",'',
        '原cProfile热点为`_observe`、`_refresh`和`_attach`；慢驱逐包含清理大量陈旧优先级/事件项。优化使用有序优先级索引、增量取消交叉事件及批量合并候选。失效事件保留紧凑时间戳，严格保持原清理阈值和时机。候选优化曾在回调1626952处改变决策，被拒绝；诊断和中间测量保留在归档目录。','',
        'get_offload_prefix_keys计数/累计profile耗时：','']
    for name,rows in cpu['prefix_profiles'].items():
        for r in rows:lines.append(f"- {name}：{r['calls']}次，累计{r['cumulative_s']:.6f}秒。")
    total_cpu={k:sum(r['elapsed_us_sum'] for r in rows)/1e6 for k,rows in cpu['callback_totals'].items()}
    lines+=['','全回调累计壁钟计时（无profile）：'+', '.join(f'{k}={v:.3f}秒' for k,v in total_cpu.items())+f'。优化适配层比参考增加{100*(total_cpu["optimized"]/total_cpu["reference"]-1):.1f}%，完整包装层增加{100*(total_cpu["vllm"]/total_cpu["reference"]-1):.1f}%；当前优化降低evict长尾，但未降低全回调总耗时。','',
        '| 优化后最长调用 | 驱逐块数 | 耗时 ms | 到期事件 | 事件清理批次 | 工作 |','|---|---:|---:|---:|---:|---|']
    for r in cpu['slowest']['vllm'][:5]:
        work=f"合并选择并删除{r['n']}个块"
        if r.get('event_sweeps'):work+='；清理已取消事件时间戳'
        if r.get('events'):work+=f"；处理{r['events']}个到期事件"
        lines.append(f"| {r['call']} | {r['n']} | {r['elapsed_us']/1000:.4f} | {r.get('events',0)} | {r.get('event_sweeps',0)} | {work} |")
    lines+=['','参考与优化实现最长10次调用的完整样本见[cpu/validation.json](cpu/validation.json)的slowest字段；完整cProfile见cpu/reference-profile、cpu/optimized-profile。']
    lines+=['','cProfile的enable/disable本身有显著成本，不能用其分位数判断时延门槛。每种回调的累计无profile时间见cpu/validation.json；优化针对evict长尾，不代表所有hint回调都加速。','',
        '## GPU：同人口LRU与hint+none','',
        'none为显式、经过逐次验证的原生LRU退化专用路径；不是generic hint评分器的端到端成本。未知hint会拒绝，已知hint档位使用通用优化评分器。轻量模式仅内存计数和直方图，退出每进程写一次；完整插桩只用于显式调试。分位数为nearest-rank桶上下界，桶宽上限1/128；均值使用精确累计时间。','',
        '| 组别 | 窗口请求/成功/失败 | GPU命中 | CPU命中 | 重算 | TTFT P50 ms | TTFT P99 ms |','|---|---:|---:|---:|---:|---:|---:|']
    for k,r in current['runs'].items():
        f=r['source_fractions'];lines.append(f"| {k} | {r['window_requests']}/{r['window_success']}/{r['window_failed']} | {f['gpu_hit']:.3%} | {f['cpu_hit']:.3%} | {f['recomputed']:.3%} | {r['ttft_ms']['p50']:.4f} | {r['ttft_ms']['p99']:.4f} |")
    diff=current['pair']['ttft_difference'];gate=current['pair']['e1_overhead_gate_passed']
    lines+=['',f"Hint−LRU：P50 {diff['p50']['absolute_ms']:+.4f}ms（{diff['p50']['relative_percent']:+.3f}%）；P99 {diff['p99']['absolute_ms']:+.4f}ms（{diff['p99']['relative_percent']:+.3f}%）。2%门槛：**{'通过' if gate else '未通过；不进入E1'}**。",'',
        '缓存来源为统计窗口内成功、非is_entry请求的prompt token加权名义口径，逐请求三项严格加总为prompt长度。人口与输入指纹相同；闭环时序可能改变真实访问顺序，真实运行的比例近似相同不等于逐次真实驱逐相同。CPU固定序列另给出了逐次一致性证明。','',
        '| 回调/调度器 | LRU P50 µs | hint+none P50 µs | LRU P99 µs | hint+none P99 µs | 均值差 µs |','|---|---:|---:|---:|---:|---:|']
    for name,h in current['histogram_comparison'].items():
        x,y=h['lru'],h['hint_none'];lines.append(f"| {name} | {histogram_text(x,'p50')} | {histogram_text(y,'p50')} | {histogram_text(x,'p99')} | {histogram_text(y,'p99')} | {h['mean_difference_us']:+.3f} |")
    lines+=['','scheduler.schedule测的是CPU调度方法，不包含GPU核函数执行或异步等待。直方图按回调结束时刻落入[300,1200]秒归档；请求指标按发送时刻入窗，故回调计数不必等于窗口请求数。','',
        '## 首轮及边界','',
        f"[首轮原始产物与未通过结论](../archive/{first_attempt.name}/run.json)：两组3448请求全部成功，P50差值{old['pair']['ttft_difference']['p50']['relative_percent']:+.3f}%，未通过2%门槛。首轮数据未删除或合并到新一轮。退出汇总器uint64推断溢出已修复，原始客户端run.json未改；早期失败堆栈保留。",'',
        '首轮轻量直方图显示none动态转发提高高频get/insert等回调成本；后续直接继承原生LRU方法并移除重复包装。该局部差异不能独立解释全部TTFT变化。共享机器、单个人口种子和一次配对复测限制了因果归因；本任务没有执行E1，也没有证明带perfect/noisy/online hint时的TTFT开销。','',
        '![CPU与GPU时延CDF](figures/latency_cdf.png)','', '![服务器队列](figures/queues.png)','',
        '复现：先确保固定vLLM源码、模型和负载可用，用`python -m agentkv.bench.profile_callbacks --implementation reference --output-dir /tmp/NEW`进行CPU回放，再分别将implementation替换为`optimized`或`vllm`并使用新的输出目录；GPU用`python -m agentkv.bench.policy_overhead --output-dir /tmp/NEW --gpu-index 1 --port 8015`。正式测量拒绝脏工作区。结果图/报告可用`python -m agentkv.analysis.policy_overhead_report --result-dir results/m5-policy-overhead-20261104 --first-attempt results/archive/m5-policy-overhead-attempt1-20261104`重新生成。','',
        '大请求token数组保留本机并忽略Git；紧凑request-metadata.parquet、完整退出直方图、两卡dmon及客户端run.json纳入版本控制。未向结果目录复制实现代码。']
    baseline_drift=100*(a['ttft_ms']['p50']/old['runs']['lru']['ttft_ms']['p50']-1)
    lines+=['',f'LRU基线跨轮P50漂移：{baseline_drift:+.3f}%。本轮共同请求{current["common_requests"]}个，其发送prompt哈希、置换种子、prompt长度及输出预算逐项一致。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    artifact_paths=[p for folder in ('cpu','gpu') for p in (directory/folder).rglob('*') if p.is_file() and p.name!='requests.parquet']
    result=dict(git_commit=current_commit(),cpu=cpu,current_gpu=current,first_attempt_pair=old['pair'],
        cpu_gate_passed=cpu['cpu_gate_passed'],gpu_overhead_gate_passed=gate,e1_executed=False,
        source_artifact_sha256={str(p.relative_to(directory)):sha256_file(p) for p in artifact_paths},
        first_attempt_run_sha256=sha256_file(first_attempt/'run.json'),baseline_p50_between_attempts_percent=baseline_drift)
    write_json(directory/'run.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir',type=Path,required=True)
    p.add_argument('--first-attempt',type=Path,required=True)
    a=p.parse_args();r=report(a.result_dir,a.first_attempt)
    print(json.dumps(dict(cpu_gate=r['cpu_gate_passed'],gpu_gate=r['gpu_overhead_gate_passed']),indent=2))


if __name__=='__main__':main()
