"""Render capacity constraints, all measured LRU results and a pilot shortlist."""
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .trace_loader import sha256_file
from agentkv.provenance import current_commit


def render(capacity_dir, simulation_dir):
    capacity=json.loads((capacity_dir/'run.json').read_text())
    simulation=json.loads((simulation_dir/'run.json').read_text())
    results=simulation['results']; kv=capacity['kv_bytes_per_token']
    try:
        gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv'],text=True,stderr=subprocess.DEVNULL)
    except (FileNotFoundError,subprocess.CalledProcessError):
        gpu_text=''
    snapshot=dict(recorded_at=datetime.now(timezone.utc).isoformat(),raw_csv=gpu_text,
                  note='read-only snapshot; no GPU experiment or process changes')
    (simulation_dir/'gpu_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    rows=list(csv.DictReader(gpu_text.splitlines(),skipinitialspace=True))
    free_gib=max((float(row['memory.free [MiB]'].split()[0])/1024 for row in rows),default=0.)
    reserve=capacity['parameters']['non_kv_gpu_reserve_gib']
    qualified=[r for r in results if r['recommend_for_gpu_pilot'] and r['measured_window_prefill_tps']<=15400]
    shared_fit=[r for r in qualified if r['candidate']['gpu_kv_gib']+reserve<=free_gib]
    # Hold GPU/slots fixed while testing population size and CPU/GPU ratio.
    targets=[(16384,.7,8,1.5),(16384,1.,8,1.5),(16384,1.,8,2.)]
    shortlist=[]
    for budget,rate,slots,ratio in targets:
        match=next((r for r in shared_fit if r['candidate']['context_budget']==budget and
                    r['candidate']['rate_per_minute']==rate and r['candidate']['max_num_seqs']==slots and
                    abs(r['candidate']['cpu_gpu_ratio']-ratio)<.001),None)
        if match: shortlist.append(match)
    analysis=dict(git_commit=current_commit(),source=dict(capacity_sha256=sha256_file(capacity_dir/'run.json'),
                              simulation_sha256=sha256_file(simulation_dir/'run.json')),
                  threshold=dict(minimum_hit_gap_percentage_points=10,maximum_prefill_tps=15400,
                                 require_window_rate_also=True,non_kv_gpu_reserve_gib=reserve),
                  simulation_qualified=[r['directory'] for r in qualified],
                  current_shared_memory_fit=[r['directory'] for r in shared_fit],
                  priority_pilots=[r['directory'] for r in shortlist],gpu_snapshot_sha256=sha256_file(simulation_dir/'gpu_snapshot.json'),
                  gpu_trials_executed=False)
    (simulation_dir/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    plan=[]
    for r in shortlist:
        c=r['candidate']
        plan.append(dict(id=r['directory'],context_budget=c['context_budget'],arrival_rate=c['rate_per_minute'],
            max_num_seqs=c['max_num_seqs'],gpu_tokens=c['gpu_tokens'],cpu_tokens=c['cpu_tokens'],
            gpu_blocks=c['gpu_tokens']//16,gpu_bytes=c['gpu_tokens']*kv,cpu_bytes=c['cpu_tokens']*kv,
            working_set_p50_tokens=c['working_set_p50'],working_set_p90_tokens=c['working_set_p90'],
            duration_s=900,warmup_s=300,policy='lru',hint_tier='none',steady_state=True,population_seed=0,
            estimated_prefill_tps=r['prefill_tokens_per_second'],simulation_run_sha256=sha256_file(simulation_dir/r['directory']/'run.json'),
            status='recommended for pilot only; not executed or accepted as E0'))
    (simulation_dir/'pilot-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (capacity_dir/'screening.csv').open('w',newline='') as sink:
        fields=['context_budget','rate_per_minute','max_num_seqs','gpu_tokens','cpu_tokens','cpu_gpu_ratio','sizing_kv_p90','working_set_p50','working_set_p90','passed','violations_tokens']
        writer=csv.DictWriter(sink,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for c in capacity['screened_configurations']:
            writer.writerow({k:(all(c['constraints'].values()) if k=='passed' else json.dumps(c[k]) if k=='violations_tokens' else c[k]) for k in fields})
    lines=['# M1 两层容量筛选 v3','','32K=32512、16K=16384，沿用同一 v3 根会话、seed=0、锚点和间隔；本次只做 CPU 分析。','','## 新约束和估算口径','','容量筛选严格执行 `2.5×运行中 KV P90 ≤ GPU池 < CPU层 ≤ 0.8×工作集`，且 `1.5≤CPU/GPU≤2`；取消 GPU+CPU<工作集。GPU 最小池向上取整到 16 token，CPU/GPU 目标为 1.5/1.75/2，并向上取整到 block。GPU/CPU 独立保留副本，因此不能把两层容量相加理解为可用工作集。',
           '', '**P90 定容口径保持保守：** 每个槽位从全部请求混合等权抽样完整 prompt+输出、独立求和，20000 次固定种子；取其 P90 与固定 s=0.5 有限槽位模拟 P90 的较大值。表中列出这两者。没有把低并发 s=0.5 占用当成满槽位占用。',
           '', '**工作集方法（以下每组相同）：** 用固定 s=0.5 的无缓存反馈估算调度和同一 M/G/∞ 稳态人口；对每条尚未结束的链取最近已发送的 prompt，计算同实例不同 prompt 的前缀 trie 并集 token 数。实例之间不共享，链最后一次请求完成后移除；不包含输出 KV。40 分钟窗口、前 10 分钟预热，统计段每 10 秒采样；用 P50 作上述 0.8 倍约束，另报 P90。这是活跃上下文 footprint 口径，不是实测复用距离工作集；缓存模拟中的有限槽位排队不反向修改此估计。',
           '', '| 预算 | λ 根/分钟 | 工作集 P50 token / GiB | P90 token | 0.8×P50 token |', '|---|---:|---:|---:|---:|']
    for c in capacity['configurations']:
        ws=c['working_set_tokens']; lines.append(f"| {c['context_budget']} | {c['rate_per_minute']:g} | {ws['p50']:,.1f} / {ws['p50']*kv/2**30:.2f} | {ws['p90']:,.1f} | {ws['p50']*.8:,.1f} |")
    lines+=['','## 通过容量约束的全部配置','','| ID | 预算 / λ / 槽位 | GPU token | CPU token | CPU/GPU | 工作集 P50 / P90 token | 调度 / 定容 KV P90 |','|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        c=r['candidate']; lines.append(f"| {r['directory']} | {c['context_budget']} / {c['rate_per_minute']:g} / {c['max_num_seqs']} | {c['gpu_tokens']:,} | {c['cpu_tokens']:,} | {c['cpu_gpu_ratio']:.6f} | {c['working_set_p50']:,.1f} / {c['working_set_p90']:,.1f} | {c['scheduled_kv_p90']:,.0f} / {c['sizing_kv_p90']:,.1f} |")
    lines+=['',f"{len(capacity['screened_configurations'])} 个离散容量组合中 {len(capacity['candidates'])} 个通过。完整逐条布尔条件和 token 缺口见 [run.json](run.json)、[screening.csv](screening.csv)。这里只是容量通过，GPU 试跑还须通过缓存模拟。",'', '## 最近的不满足配置','','| 预算 / λ / 槽位 | GPU / CPU token | 工作集 P50 token | 未通过项 | token 缺口 |','|---|---:|---:|---|---:|']
    for c in capacity['closest_rejected']:
        lines.append(f"| {c['context_budget']} / {c['rate_per_minute']:g} / {c['max_num_seqs']} | {c['gpu_tokens']:,} / {c['cpu_tokens']:,} | {c['working_set_p50']:,.1f} | {'; '.join(c['violations_tokens'])} | {'; '.join(f'{v:,.1f}' for v in c['violations_tokens'].values())} |")
    lines+=['','表中列出精确 token 缺口，不因接近边界而放行。若最小 GPU、最小 CPU/GPU 比例已经导致 CPU 超过 0.8×工作集，增大 GPU 或比例无法消除该超限；结论基于上述满槽位保守 P90 口径。',
            '',f"Qwen3-8B BF16 每 token KV=147456 bytes。主机 MemAvailable={capacity['memory']['MemAvailable_bytes']/2**30:.2f}GiB；通过容量筛选的 CPU KV 最大 {max((c['cpu_kv_gib'] for c in capacity['candidates']),default=0.):.2f}GiB，在可用量 85% 以内。此为 KV 本体，进程和搬运缓冲区仍需试跑核验。",'', '[prefill 曲线 PNG](prefill_capacity.png) · [PDF](prefill_capacity.pdf) · [工作集采样](working_set_samples.csv) · [缓存模拟结果](../m1-two-tier-seed0/README.md)',
            '', '## 复现','','```bash','python -m agentkv.bench.capacity --output-dir results/new-capacity-run','python -m agentkv.sim.two_tier --capacity-dir results/new-capacity-run --output-dir results/new-two-tier-run','python -m agentkv.bench.report_two_tier --capacity-dir results/new-capacity-run --simulation-dir results/new-two-tier-run','```','','旧 v2 是旧约束的历史产物，本目录为新筛选，不覆盖旧数据。']
    (capacity_dir/'README.md').write_text('\n'.join(lines)+'\n')
    lines=['# M1 两层 LRU 模拟','','## 最小模型与核验门槛','','完整 16-token 前缀块；同实例可共享、不同实例隔离，等价于置换后的前缀身份。先查询 GPU 连续前缀，再用 CPU 连续前缀延伸；剩余 token（含不完整末块）全部重算。新块在请求完成时才就绪，GPU 运行 KV 钉住，输出 KV 只作私有预留，不缓存。槽位与 GPU 容量不足时 FCFS 排队。',
           '', 'CPU 是重复写入的 prompt 镜像层，两层独立执行 LRU；GPU 淘汰不向 CPU 插入，CPU 淘汰不使 GPU 副本失效。请求完成时按整条 prompt 更新两层 recency，前缀头比尾更近期。它不是独占 victim cache，也不包含 vLLM 具体 chunk 分组和异步搬运策略。服务固定 0.5 秒，搬运视为瞬时，不能预测真实服务时延。',
           '', '使用容量分析相同实例及 τ，缓存从空状态开始，不预填初始历史；T=40 分钟、预热 10 分钟，is_entry 请求照常加载但不计统计。同人口无限缓存对照使用相同槽位和完成时就绪规则。',
           '', '**测量前约定的推荐门槛：** 容量约束通过；模拟 LRU 命中率比全负载理论上限、同人口无限缓存对照都至少低 10 个百分点；稳态 prefill 需求≤15400 token/s。报告另确认该次窗口的实测速率也≤15400。全负载理论上限约98.095%；有限窗口的 prompt 混合不同，同窗口理论上限另记录，不能把无限缓存对照略高于全负载平均视为超出上限。',
           '', '稳态需求=`λ/60×mean_R(sum prompt_len)×模拟 token 加权重算比例`；窗口速率=`窗口重算 token/1800s`。所有源比例均按 token 加权，三者相加为 100%；不是逐请求比例的算术均值。仅计 prefill，不包含 decode、CPU 带宽和共享 GPU 干扰。',
           '', '## 全部模拟结果','','| ID | GPU / CPU / 重算 | 距全负载上限 pp | 同人口无限缓存命中 | 稳态 / 窗口 prefill token/s | 占22k / 占34k | 模拟门槛 |','|---|---:|---:|---:|---:|---:|---|']
    for r in results:
        f=r['summary']['source_fractions']; yes=r in qualified
        lines.append(f"| [{r['directory']}]({r['directory']}/run.json) | {f['gpu_hit']:.2%} / {f['cpu_hit']:.2%} / {f['recomputed']:.2%} | {r['upper_minus_lru']*100:.2f} | {r['unlimited_control_hit']:.3%} | {r['prefill_tokens_per_second']:,.1f} / {r['measured_window_prefill_tps']:,.1f} | {r['utilization_22k']:.2%} / {r['utilization_34k']:.2%} | {'通过' if yes else '不通过'} |")
    rejected=[r for r in results if r not in qualified]
    rejection_note='；'.join(f"{r['directory']}：稳态 {r['prefill_tokens_per_second']:,.1f}、窗口 {r['measured_window_prefill_tps']:,.1f} token/s，未过项 {','.join(k for k,v in r['recommendation_checks'].items() if not v)}" for r in rejected)
    if not results: rejection_note='没有容量合格项，未执行缓存模拟；见容量报告的最近失败配置。'
    elif not rejected: rejection_note='本次全部模拟项通过门槛。'
    lines+=['', '**不放行项：** '+rejection_note,
           '', f"{len(qualified)} 个配置通过模拟推荐门槛；这不是 GPU 试跑通过。全部候选的工作集、池大小、比例和估算方法见 [容量表](../m1-capacity-seed0-v3/README.md)。",'', '## 优先去共享 GPU 短跑的配置','','按当前只读显存快照，加 30GiB 非 KV 预留检查，优先固定 GPU 池和 8 槽位，对照 λ 与 CPU 大小：','','| ID | 预算 / λ / 槽位 | GPU token / GiB | CPU token / GiB | 工作集 P50 / P90 token | prefill token/s |','|---|---|---:|---:|---:|---:|']
    for r in shortlist:
        c=r['candidate']; lines.append(f"| {r['directory']} | {c['context_budget']} / {c['rate_per_minute']:g} / {c['max_num_seqs']} | {c['gpu_tokens']:,} / {c['gpu_kv_gib']:.2f} | {c['cpu_tokens']:,} / {c['cpu_kv_gib']:.2f} | {c['working_set_p50']:,.1f} / {c['working_set_p90']:,.1f} | {r['prefill_tokens_per_second']:,.1f} |")
    lines+=['',f"优先短跑配置数为 {len(shortlist)}。当前空闲显存最多约 {free_gib:.2f}GiB（若未取得显卡信息则为0，不认定可放入共享卡）；优先项 GPU KV 均约25.11GiB，加30GiB预留约55.11GiB。现场会变化，启动前仍须重新核对。其他大池即使通过模拟，也可能放不进目前共享 GPU，见 [analysis.json](analysis.json)。",'', '**本次未执行任何 GPU 试跑，未确定正式 E0。** 70% 算力线只是筛选余量，不能保证其他进程争用下稳定。按 [pilot-plan.json](pilot-plan.json) 短跑后仍需核验等待队列趋势、运行并发和实际命中。',
           '', '## 诊断与证据','','单链无限缓存与最大 LCP 上限精确一致的单元测试使用 token 粒度；实验使用16-token块。同一循环工作集大于 GPU、CPU 小于 GPU 的测试中，发生 GPU 驱逐仍没有 CPU 增益；另有真实16K人口的小CPU诊断，见 [small-cpu-diagnostic/run.json](small-cpu-diagnostic/run.json)。',
           '', '[run.json](run.json) 记录全部参数、代码提交号及输入/产物指纹；不复制实现代码。各配置保存 requests.parquet 和 run.json。无限缓存对照保存在 unlimited-* 目录。独立核验见 [validation.json](validation.json)，完整测试见 [pytest.log](pytest.log)。',
           '', '![Token sources](lru_sources.png)','','![Prefill requirement](lru_prefill.png)','','图另有 [来源 PDF](lru_sources.pdf)、[吞吐 PDF](lru_prefill.pdf)。']
    (simulation_dir/'README.md').write_text('\n'.join(lines)+'\n')
    diagnostic=simulation_dir/'small-cpu-diagnostic/run.json'
    if diagnostic.exists():
        diag=json.loads(diagnostic.read_text())
        with (simulation_dir/'README.md').open('a') as sink:
            sink.write(f"\n真实人口诊断：GPU={diag['parameters']['gpu_tokens']:,} token，CPU={diag['parameters']['cpu_tokens']:,} token，CPU/GPU=0.5；GPU / CPU / 重算为 "+' / '.join(f"{diag['summary']['source_fractions'][k]:.4%}" for k in ('gpu_hit','cpu_hit','recomputed'))+'。该诊断违反候选比例约束，只用于验证镜像层退化，不推荐试跑。\n')
    plot(simulation_dir,results)
    analysis['report_artifacts_sha256']={p.name:sha256_file(p) for p in simulation_dir.iterdir()
        if p.suffix in ('.png','.pdf') or p.name in ('README.md','pilot-plan.json')}
    analysis['screening_csv_sha256']=sha256_file(capacity_dir/'screening.csv')
    (simulation_dir/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')


def plot(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=[r['directory'].replace('config-','') for r in results]
    x=np.arange(len(results)); fig,axes=plt.subplots(2,1,figsize=(7.1,4.6),sharex=True)
    bottom=np.zeros(len(results))
    for name,color in [('gpu_hit','#0072B2'),('cpu_hit','#009E73'),('recomputed','#D55E00')]:
        values=np.array([r['summary']['source_fractions'][name]*100 for r in results])
        axes[0].bar(x,values,bottom=bottom,label=name,color=color,width=.8);bottom+=values
    axes[0].set(ylabel='Prompt tokens (%)',ylim=(0,100));axes[0].legend(ncol=3,fontsize=7,frameon=False)
    values=[r['prefill_tokens_per_second']/1000 for r in results]
    axes[1].bar(x,values,color=['#0072B2' if r['recommend_for_gpu_pilot'] else '#D55E00' for r in results])
    for y,label,style in [(15.4,'70% of 22k','--'),(22,'22k',':'),(34,'34k','-.')]:
        axes[1].axhline(y,color='gray',ls=style,lw=.8,label=label)
    axes[1].set(ylabel='Prefill (k token/s)',xlabel='Configuration ID',ylim=(0,36),xticks=x,xticklabels=labels)
    axes[1].legend(ncol=3,fontsize=7,frameon=False)
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'lru_sources.{ext}',dpi=300)
    plt.close(fig)
    # Standalone throughput plot for report/export.
    fig,ax=plt.subplots(figsize=(7.1,2.8))
    ax.bar(x,values,label='stationary demand',color='#0072B2')
    ax.plot(x,[r['measured_window_prefill_tps']/1000 for r in results],'.',color='black',label='observed window')
    for y,label,style in [(15.4,'70% of 22k','--'),(22,'22k',':'),(34,'34k','-.')]:ax.axhline(y,color='gray',ls=style,lw=.8,label=label)
    ax.set(ylim=(0,36),ylabel='Prefill (k token/s)',xlabel='Configuration ID',xticks=x,xticklabels=labels)
    ax.legend(ncol=3,fontsize=7,frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'lru_prefill.{ext}',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-dir',type=Path,default=Path('results/m1-capacity-seed0-v3'))
    parser.add_argument('--simulation-dir',type=Path,default=Path('results/m1-two-tier-seed0'))
    args=parser.parse_args();render(args.capacity_dir,args.simulation_dir)


if __name__=='__main__':main()
