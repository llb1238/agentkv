"""CPU forecasts and immutable preregistration for the lambda=1.2 boundary."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from agentkv.bench.e0 import ROOT,WORKLOAD,SCHEDULE,load_inputs,population_hash,write_json
from agentkv.bench.e1_gpu import BACKEND,FREEZE,PROTOCOL,read,registered_plan
from agentkv.bench.e1_simulation import _schedule_fingerprint
from agentkv.bench.steady_state import sample_population
from agentkv.bench.trace_loader import sha256_file
from agentkv.policy.chain_policy import ChainHintAwarePolicy
from agentkv.policy.prefix_adapter import frozen_options
from agentkv.provenance import git_commit
from agentkv.sim.hint_adapter import HintPolicyAdapter
from agentkv.sim.two_tier import simulate

SPEC=dict(format='agentkv.stable_limit.v1',policies=['lru','online','perfect'],order_seed=20261107,queue_rate=1.2)
REGISTRATION=ROOT/'docs/stable_limit_12_preregistration.json'
DOCUMENT=ROOT/'docs/stable_limit_12_preregistration.md'


def predict(output):
    commit=git_commit()
    if output.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(ROOT):raise ValueError('Stage outside Git')
    output.mkdir(parents=True);roots,blocks=load_inputs();cases=[]
    for group in registered_plan(SPEC):
        instances=sample_population({k:r.duration_s for k,r in roots.items()},1.2,group['seed'],2400,True)
        folder=output/group['name'];folder.mkdir()
        write_json(folder/'population.json',dict(instances=[asdict(i) for i in instances],population_sha256=population_hash(instances)))
        fingerprint=None
        for label in SPEC['policies']:
            policy='lru'
            if label!='lru':
                policy=HintPolicyAdapter(274320//16,hint_tier=label,**frozen_options(label))
                policy.policy=ChainHintAwarePolicy(274320//16,clock=lambda:policy.now,**frozen_options(label))
            summary,rows=simulate(roots,instances,blocks,182880,274320,8,duration=2400,warmup=600,cpu_policy=policy)
            current=_schedule_fingerprint(rows)
            if fingerprint is None:fingerprint=current
            assert current==fingerprint,'Policy changed fixed-service schedule/GPU'
            kept=[r for r in rows if r['in_window']];total=sum(r['prompt_tokens'] for r in kept)
            case=dict(git_commit=commit,status='complete',group=group['name'],rate=1.2,seed=group['seed'],policy=label,
                population_sha256=population_hash(instances),schedule_gpu_sha256=current,
                predicted_effective_recompute=sum(max(1,r['recomputed']) for r in kept)/total,summary=summary)
            path=folder/label;path.mkdir();pq.write_table(pa.Table.from_pylist(rows),path/'requests.parquet',compression='zstd')
            case['requests_sha256']=sha256_file(path/'requests.parquet');write_json(path/'run.json',case);cases.append(case)
            print('MODEL',group['name'],label,case['predicted_effective_recompute'],flush=True)
    write_json(output/'run.json',dict(git_commit=commit,status='complete',cases=cases,
        source=dict(workload_sha256=sha256_file(WORKLOAD),schedule_sha256=sha256_file(SCHEDULE),freeze_sha256=sha256_file(FREEZE))))


def register(models,gates):
    commit=git_commit()
    if REGISTRATION.exists() or DOCUMENT.exists():raise FileExistsError('Never change preregistration')
    assert git_commit(repository=ROOT.parent/'vllm')==BACKEND
    simulations=read(models/'run.json');gate=read(gates/'run.json')
    assert simulations['status']=='complete' and len(simulations['cases'])==9
    assert gate['status']=='complete' and gate['parameters']['seed']==4
    assert gate['parameters']['duration_s']==2400 and gate['parameters']['warmup_s']==600
    for case in simulations['cases']:
        path=models/case['group']/case['policy']
        assert read(path/'run.json')==case and sha256_file(path/'requests.parquet')==case['requests_sha256']
    protocol=copy.deepcopy(read(PROTOCOL))
    protocol.update(**SPEC,groups=registered_plan(SPEC),policy_implementation='chain',
        registration_source_commit=commit,implementation_freeze_commit=commit,
        source=simulations['source'],simulation_directory=str(models.resolve().relative_to(ROOT)),
        simulation_commit=simulations['git_commit'],simulation_run_sha256=sha256_file(models/'run.json'),predictions=simulations['cases'],
        cleanup_gate=dict(path=str(gates.resolve()),sha256=sha256_file(gates/'run.json'),passed=gate['passed']),
        runtime_source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'agentkv').rglob('*.py'))})
    write_json(REGISTRATION,protocol)
    lines=['# λ=1.2稳定负载上限：预注册','',
        '本文件与JSON单独提交后才启动9次正式GPU运行；正文不改，结果另存或只在末尾追加。',
        f'固定实现`{commit}`，vLLM fork `{BACKEND}`；config-11：Qwen3-8B、16K、8槽位、GPU182880/CPU274320 token。评分公式、校准与rank语义不变，空链/共享关系清理已通过全部可重放序列。',
        'λ=1.2根会话/分钟，种子3/4/5；LRU（内置）、online、perfect；T=40分钟、预热10分钟、收尾0。稳态初始化、实例置换、安全阀128、请求超时180秒、收尾等待120秒，与E1 v2一致。GPU1，每次新服务；轻量插桩、有效prefill观察器、两卡dmon。',
        'LRU传online信封但忽略hint；共同逻辑请求token哈希、置换种子、长度逐项核对。闭环完成时刻和发送集合可以不同；不修改负载或矩阵。',
        f"40分钟开销验证的2ms目标是否通过：`{gate['passed']}`；online相对LRU的保守scheduler P99增量为{gate['comparisons']['online']['delta_upper_ms']:.6f} ms。该项验收未通过时仍原样执行用户指定的独立λ=1.2矩阵，失败事实单列；TTFT包含实际策略开销，不将稳定性结果解释为开销达标。原始验证清单SHA256 `{sha256_file(gates/'run.json')}`。",
        '', '## 固定顺序','', '按人口分组，random.Random(20261107)对初始[lru,online,perfect]依次shuffle，组内连续。',
        '', '| 人口种子 | 顺序 |','|---:|---|']
    for g in protocol['groups']:lines.append(f"| {g['seed']} | {' → '.join(g['order'])} |")
    lines+=['','## 判据与主要指标','',
        '稳定性不变（a81dd86）：等待为0采样≥80%且等待P99≤8；队列OLS斜率×1800<1；后半/前半TTFT中位数≤1.5；存在KV不足阻塞且等待>0时运行数≥7。缺观察记无法判定，不默认稳定。',
        '主要指标：各策略稳定次数/3；稳定运行的有效重算率、TTFT P50/P99及相对同人口LRU的配对差值；全部运行的窗口完成请求数（全部/非is_entry）。有效重算为实际prompt前向工作量/成功非entry发送入窗prompt token总量，含replay/重复工作，不以名义值代替。完成数按结束时刻入窗，与主指标发送窗口分母分开。',
        '配对仅策略和LRU都稳定时纳入，逐人口一个差值，报告均值与最小–最大及有效配对数，不补零。各策略绝对值仅自身稳定人口；全部逐人口结果及不稳定/失败项完整列出，不进入稳定主均值。只有3个人口，不做显著性检验。',
        '报告主指标对比表、稳定人口等权TTFT CDF、全部等待队列曲线、实测/模拟差距。次要指标：三类名义重算及有效工作差额、hint传输/消费对账、轻量回调耗时、两卡dmon。',
        '', '## 九项CPU预测','',
        '固定服务0.5秒、零搬运；按原两层缓存和本次相同链核心模拟，三策略调度/GPU指纹必须相同。预测有效工作沿用每请求max(1,名义重算)，模型不预测TTFT或稳定性，不含GPU排队/重算服务时间反馈。',
        '', '| 种子 | LRU重算% | online重算% | perfect重算% |','|---:|---:|---:|---:|']
    for seed in (3,4,5):
        by={c['policy']:c for c in simulations['cases'] if c['seed']==seed}
        lines.append(f'| {seed} | '+' | '.join(f"{100*by[p]['predicted_effective_recompute']:.6f}" for p in SPEC['policies'])+' |')
    lines+=['',f"CPU源提交`{simulations['git_commit']}`，总清单SHA256 `{protocol['simulation_run_sha256']}`；各次原始请求指纹见JSON。",'',
        '## 异常与留存','',
        '默认拒绝dirty，两仓库及全部源码/输入SHA核验；flock控制器、原子checkpoint，已完成项核验后跳过。每次新目录，不覆盖产物。非策略中断记录原因并原样重跑一次，第二次失败如实报告；策略/正确性异常立即停，不以重跑掩盖。资源未就绪仅等待，不计attempt。',
        '全部原始数据写持久目录；run.json只记提交号来源，不复制源码。大prompt保留本机，Git发布元数据及原始清单/日志/指纹。',
        '', '```bash','python -m agentkv.bench.e1_gpu run --protocol docs/stable_limit_12_preregistration.json --output-dir /home/llb/my_soft/agentkv-runs/stable-limit-12-20261107 --gpu-index 1 --port 8016','```','',f'协议SHA256 `{sha256_file(REGISTRATION)}`。']
    DOCUMENT.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--output-dir',type=Path,required=True)
    a=sub.add_parser('register');a.add_argument('--models',type=Path,required=True);a.add_argument('--gates',type=Path,required=True)
    a=p.parse_args()
    if a.command=='predict':predict(a.output_dir)
    else:register(a.models,a.gates)


if __name__=='__main__':main()
