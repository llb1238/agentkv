"""Publish state-growth diagnosis without rewriting measurement records."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq
from agentkv.bench.e0 import write_json
from agentkv.bench.trace_loader import sha256_file
from agentkv.provenance import git_commit


def publish(source,output,after_dir='after',parity_dir='parity-all',accepted_gpu='gpu'):
    code=git_commit()
    if output.exists():raise FileExistsError(output)
    a=json.loads((source/'before/run.json').read_text());b=json.loads((source/after_dir/'run.json').read_text())
    assert a['population_sha256']==b['population_sha256']
    assert pq.read_table(source/'before/requests.parquet').equals(pq.read_table(source/after_dir/'requests.parquet'))
    parity=json.loads((source/parity_dir/'run.json').read_text());assert parity['status']=='complete'
    output.mkdir(parents=True)
    # Retain raw per-call timing tables on persistent disk; ordered per-evict
    # fingerprints and original measurement manifests are published unchanged.
    parity_ignore=lambda d,n:[k for k in n if k=='callbacks.parquet']
    for src,dst in [('before','before'),(after_dir,'after')]:shutil.copytree(source/src,output/dst)
    shutil.copytree(source/parity_dir,output/'parity-all',ignore=parity_ignore)
    if after_dir!='after':shutil.copytree(source/'after',output/'cleanup-only')
    if parity_dir!='parity-all':shutil.copytree(source/'parity-all',output/'parity-cleanup-only',ignore=parity_ignore)
    for folder in source.glob('gpu*'):
        if folder.is_dir() and (folder/'run.json').exists():
            shutil.copytree(folder,output/folder.name,ignore=lambda d,n:[k for k in n if k in ('requests.parquet','controller.pid','controller.lock')])
    for folder in source.glob('profile-*'):
        if folder.is_dir():shutil.copytree(folder,output/folder.name)
    for folder in source.glob('parity-*'):
        if folder.is_dir() and folder.name not in (parity_dir,'parity-all'):
            shutil.copytree(folder,output/folder.name,ignore=parity_ignore)
    for name in ('parity','parity-recorded-policies'):
        if (source/name).exists() and not (output/name).exists():
            shutil.copytree(source/name,output/name,ignore=parity_ignore)
    for folder in source.glob('latency-*'):
        if folder.is_dir():shutil.copytree(folder,output/folder.name)
    for folder in source.glob('after-*'):
        if folder.name!=after_dir:shutil.copytree(folder,output/folder.name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series={name:pq.read_table(output/name/'snapshots.parquet').to_pylist() for name in ('before','after')}
    for values in series.values():
        for row in values:row['live_nonfinal_chains']=row['chains']-row['final_retained']
    with (output/'minute_states.csv').open('w',newline='') as f:
        rows=[dict(implementation=name,**row) for name,values in series.items() for row in values]
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    fields=['final_empty','empty_chains','neighbor_edges','chain_events','on_request_finished_p99_us','evict_p99_us']
    labels=['Empty final chains','Empty retained chains','Live sharing edges','Chain events incl. stale','Finished P99 (us)','Evict P99 (us)']
    fig,axes=plt.subplots(3,2,figsize=(7.1,6.))
    for ax,field,label in zip(axes.flat,fields,labels):
        for name,rows in series.items():
            kept=[r for r in rows if r[field] is not None]
            ax.plot([r['time_s']/60 for r in kept],[r[field] for r in kept],lw=1,label=name)
        ax.set(xlabel='Simulated minutes',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=7,frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'growth.{ext}',dpi=300)
    plt.close(fig)
    rows=[]
    for field in a['trends']:
        for label,record in [('before',a),('after',b)]:rows.append(dict(metric=field,implementation=label,**record['trends'][field]))
    with (output/'trends.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=['# 链状态增长：80分钟CPU诊断与清理','',
        'config-11、λ=1、人口种子4、online，固定服务0.5秒，80分钟。每60模拟秒只读采样，不推进策略时钟；每次finished/evict单独计时，快照扫描在计时外。全部单次耗时见callbacks.parquet，81个快照见snapshots.parquet。以下端点为10分钟和80分钟，趋势统计排除前10分钟。',
        f"清理前源`{a['git_commit']}`，清理后源`{b['git_commit']}`。人口指纹`{b['population_sha256']}`；全部逐请求字段（含CPU命中/重算）完全相同。",'',
        '| 项 | 清理前10→80分钟 | 清理后10→80分钟 | 清理后最大值 |','|---|---:|---:|---:|']
    for field in ('chains','final_empty','empty_chains','neighbor_chains','neighbor_edges','neighbor_groups','chain_events','stale_chain_events','overdue_empty'):
        x=a['trends'][field];y=b['trends'][field]
        lines.append(f"| {field} | {x['first']:g} → {x['last']:g} | {y['first']:g} → {y['last']:g} | {y['maximum']:g} |")
    lines+=['','单调增长的泄漏是空final链和历史邻接/共享分组。事件堆、实际存活链及回调时延均非单调增长，不能把此前GPU开销增长全部归因于泄漏。清理后空final链为0、邻接索引按当前实际共享块引用计数回收；未完成请求的空链仍保留，避免丢失pending归属。存活计数随人口和请求阶段波动，不能要求它们恒定或每分钟递减。',
        '存活口径为保留的非final链（chains−final_retained），见minute_states.csv；超时口径为waiting且t>deadline+margin。remaining模式不使用旧expired标志，拥有CPU块的超时链仍需评分，不能因过期直接丢掉归属。所有持久事件堆及过期条目均列在快照；驱逐中的链头堆每次重建并释放，其大小上限为rank_views，不跨回调积累。',
        'remaining评分路径每秒至多检查一次事件堆；无效事件超过一半才压缩。perfect旧deadline路径的超时版本复用/压缩时序具有可观测语义，因此保留原逻辑，不借此修正无关行为。共享关系清理不改变任何链分数或rank。',
        '', '## CPU耗时','',
        '下面是每分钟P99的前20/后20个统计点均值，单位μs；不是把这些均值当成全局P99，也不做显著性检验。',
        '', '| 回调 | 修复前 前20/后20 | 修复后 前20/后20 |','|---|---:|---:|']
    for field in ('on_request_finished_p99_us','evict_p99_us'):
        x=a['trends'][field];y=b['trends'][field]
        lines.append(f"| {field} | {x['first_20_mean']:.1f} / {x['last_20_mean']:.1f} | {y['first_20_mean']:.1f} / {y['last_20_mean']:.1f} |")
    lines+=['','历史垃圾状态的持续积累已消除，但不能据此宣布GPU的2ms目标已经满足。仅清理版本的evict分钟P99均值略高于原版；若包含后续热路径优化，该中间版原始记录保留于cleanup-only，最终CPU耗时按上表原样报告。',
        '', '## 有序驱逐一致性','', '| 原记录 | 策略 | evict次数 | 结果 |','|---|---|---:|---|']
    for r in parity['results']:
        lines.append(f"| {r['source']} | {r['policy']} | {r['evictions']} | 全部有序key一致"+('；仅完整回调前缀' if r['partial_capture'] else '')+' |')
    lines+=['','更早的policy-observations探针没有完整touch/protected/ref参数，不能作为完整调用重放数据；以上覆盖全部callbacks.v1记录。损坏归档保留原始字节，只恢复完整JSON行并截到最后一个已结束的顶层回调，不伪造footer或结束事件。',
        '初版检查器错误地用hint策略回放原LRU采集，修复前版本也在81092调用不匹配；修正为原生LRU路径。第二次遇到已知归档gzip无footer；最终使用显式部分前缀模式。两次检查器失败日志保留在本机持久目录，没有当作新旧策略差异或合格完整采集。',
        f'逐调用时延大表及原始prompt载荷保留在持久目录`{source}`；逐次驱逐决策表和生产者run.json按原字节发布，省略的大表仍可按原run.json的SHA核验。',
        '', '![状态与时延](growth.png)','','[全部趋势](trends.csv)。GPU验证及λ=1.2预注册另行记录；此CPU结果不代表GPU门槛通过。']
    for folder in sorted(output.glob('gpu*')):
        if not folder.is_dir() or not (folder/'run.json').exists():continue
        gpu=json.loads((folder/'run.json').read_text())
        lines+=['','## 同人口40分钟GPU验证','',
            f"目录`{folder.name}`；源`{gpu['git_commit']}`，LRU→online，各2400秒、预热600秒、λ=1、种子4，新服务、轻量插桩、两卡dmon。保守差值=online P99上界−LRU下界，是否≤2ms：`{gpu['passed']}`。",'']
        for label,r in gpu['results'].items():lines.append(f"- {label}: schedule P99 {r['histograms']['scheduler.schedule']['p99_ns']} ns，TTFT {r['ttft_ms']}，hint {r['hints_sent']}/{r['hints_received']}/{r['hints_consumed']}。")
        lines.append(f"online保守增量：{gpu['comparisons']['online']['delta_upper_ms']:.6f} ms。")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'run.json',dict(git_commit=code,status='complete',cpu_requests_identical=True,
        before_commit=a['git_commit'],after_commit=b['git_commit'],parity=parity['results'],
        accepted_gpu_directory=accepted_gpu,
        gpu_gate=(json.loads((output/accepted_gpu/'run.json').read_text())['passed'] if (output/accepted_gpu/'run.json').exists() else None),
        files_sha256={str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'run.json'}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--after-dir',default='after');p.add_argument('--parity-dir',default='parity-all');p.add_argument('--accepted-gpu',default='gpu')
    a=p.parse_args();publish(a.source,a.output_dir,a.after_dir,a.parity_dir,a.accepted_gpu)


if __name__=='__main__':main()
