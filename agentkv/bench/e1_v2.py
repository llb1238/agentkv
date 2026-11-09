"""Register the chain implementation experiment only after both performance gates."""
import argparse
import copy
from pathlib import Path

from agentkv.bench.e0 import ROOT, write_json
from agentkv.bench.e1_gpu import BACKEND, FREEZE, PROTOCOL, plan, read
from agentkv.bench.trace_loader import sha256_file
from agentkv.provenance import git_commit

POLICIES=('lru','online','perfect')
V2_PROTOCOL=ROOT/'docs/e1_v2_preregistration.json'
V2_DOC=ROOT/'docs/e1_v2_preregistration.md'
GATES=ROOT/'results/m5-chain-implementation-20261106'


def groups():
    result=plan()
    for group in result:group['order']=[p for p in group['order'] if p in POLICIES]
    return result


def prerequisites():
    record=read(GATES/'run.json')
    assert record['status']=='cpu_and_gpu_qualified'
    for name,digest in record['files_sha256'].items():
        assert sha256_file(GATES/name)==digest,('Gate artifact changed',name)
    gpu=read(GATES/'gpu/run.json')
    assert gpu['status']=='complete' and gpu['passed']
    assert set(gpu['results'])==set(POLICIES)
    assert gpu['vllm_commit']==BACKEND
    for tier,count in [('online',734),('perfect',2005)]:
        cpu=read(GATES/f'cpu/chain-plugin-{tier}-1/run.json')
        assert cpu['status']=='complete'
        # The record includes both recorded victims and the earlier baseline.
        assert cpu['all_recorded_ordered_keys_identical'] and cpu['all_baseline_ordered_keys_identical']
        assert cpu['summary']['counts']['evict']==count
        if tier=='online':assert cpu['latency_gate_2ms']
    return record,gpu


def register():
    commit=git_commit()
    assert git_commit(repository=ROOT.parent/'vllm')==BACKEND
    if V2_PROTOCOL.exists() or V2_DOC.exists():raise FileExistsError('Never overwrite preregistration')
    gate,gpu=prerequisites()
    original=read(PROTOCOL)
    models=ROOT/original['simulation_directory']
    assert sha256_file(models/'run.json')==original['simulation_run_sha256']
    predictions=[c for c in original['predictions'] if c['policy'] in POLICIES]
    assert len(predictions)==18
    for case in predictions:
        folder=models/case['group']/case['policy']
        assert read(folder/'run.json')==case
        assert sha256_file(folder/'requests.parquet')==case['requests_sha256']
    protocol=copy.deepcopy(original)
    protocol.update(format='agentkv.e1_gpu.v2',policies=list(POLICIES),groups=groups(),policy_implementation='chain',
        registration_source_commit=commit,implementation_freeze_commit=commit,
        prior_protocol_sha256=sha256_file(PROTOCOL),prior_result_directory='results/e1-gpu-config11-20261105',
        prior_result_sha256=sha256_file(ROOT/'results/e1-gpu-config11-20261105/run.json'),
        predictions=predictions,predictions_reused_unchanged=True,
        gates=dict(directory=str(GATES.relative_to(ROOT)),run_sha256=sha256_file(GATES/'run.json'),
            gpu_run_sha256=sha256_file(GATES/'gpu/run.json'),gpu_measurement_commit=gpu['git_commit']),
        runtime_source_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((ROOT/'agentkv').rglob('*.py'))})
    write_json(V2_PROTOCOL,protocol)
    lines=['# E1 v2 GPU 预注册','',
        '本文件及配套JSON必须先单独提交，再开始任何E1 v2 GPU运行。提交后正文不改；结果另存结果目录或只在末尾追加。',
        '', '## 先决条件与冻结范围','',
        'online与perfect完整回调及原734/2005次基线，逐次evict返回的有序key列表完全相同。online关闭插桩的完整适配层finished/evict P99为1.079/1.959ms，均≤2ms。三个20分钟GPU门槛全部通过，online/perfect的scheduler.schedule P99相对LRU保守增量为1.802/1.343ms。',
        f"证据：[链级实现与门槛](../results/m5-chain-implementation-20261106/README.md)，GPU测量提交`{gpu['git_commit']}`；本次运行实现冻结于`{commit}`，所有Python源码SHA256见JSON。",
        'v2仅更换实现性能：链级评分、链内原rank，每个victim后重新入队；共享块按最佳链保护，原冻结公式/校准/事件账本不变。决策等价是同一录制时钟/回调流下的逐次证明；真实闭环的完成时间与请求交错可能变化，不宣称两次GPU运行的victim流相同。',
        '', '## 配置和共同输入','',
        f"config-11：16K、8槽位、GPU182880/CPU274320 token，Qwen3-8B，vLLM fork `{BACKEND}`。评分语义冻结`{original['semantic_freeze_commit']}`，校准文件原样保留。",
        'λ∈{1,1.4}根会话/分钟，人口种子3/4/5，T=2400秒、预热600秒、收尾0；稳态初始化，安全阀128，超时180秒、收尾等待120秒。GPU1，每个策略新服务；不占用/终止其他进程，记录两卡dmon及共享状态。',
        '策略：内置LRU、online、perfect。LRU传输online信封但忽略hint；另两者显式implementation=chain，轻量插桩、有效prefill观察器和scheduler计时相同。省略LRU+final，不更改其他矩阵项。',
        '沿用v1人口、实例置换、逻辑请求ID（e1-加组名）、workload及schedule。所有共同逻辑请求逐项核对token哈希/置换种子/长度/max_tokens；闭环实际发送集合可能不同，不把相同人口误写为完全相同的发送顺序。',
        '', '## 固定顺序','',
        '沿用random.Random(20261104)预先打乱的v1组内顺序，仅移除LRU+final，不重新抽样。每组连续，非策略中断最多原样重跑一次。',
        '', '| λ | 人口种子 | 顺序 |','|---:|---:|---|']
    for g in groups():lines.append(f"| {g['rate']:g} | {g['seed']} | {' → '.join(g['order'])} |")
    lines += ['', '## 指标、稳定性与分析','',
        '稳定性严格沿用a81dd86：统计窗零等待采样≥80%且等待P99≤8；队列OLS斜率×1800<1；后半/前半TTFT中位数≤1.5；确有KV不足阻塞且等待>0时运行数≥7。缺观察记无法判定，不默认稳定。',
        '主指标仅稳定运行：有效重算比例（实际完成前向prompt token工作量/成功非is_entry发送入窗请求的prompt token）、TTFT P50/P99。有效工作量包含末token replay与重复prefill，不能用名义比例替代。逐人口报告原值及相对同人口LRU差值，仅策略和LRU都稳定的配对进入均值/最小–最大；注明配对数。样本3个人口，不做显著性检验。',
        '全部18次报告稳定状态、窗口完成数（按结束时刻入窗，全部/非is_entry），所有不稳定与失败单列，不进入稳定均值。λ=1.4列等待曲线、稳定次数、完成数。',
        '次要指标：16-token整块名义重算三分类及间隙请求平均排队；有效工作与名义差额、尾块/replay/重复工作；hint发送/接收/消费对账；策略和scheduler轻量耗时直方图；两卡dmon。',
        '每λ一张主表，配对差值均值/范围；重算图叠加CPU预测（不稳定实测区别标识）；稳定人口等权TTFT CDF。与v1并列列出三种共同策略、逐人口原值/稳定性/完成数及相对各自同人口LRU的配对差值；不把不同时段绝对差当纯因果估计。',
        '', '## 固定CPU预测','',
        '缓存/评分语义未变，直接复用v1已登记且完成的固定0.5秒、无限并发、零搬运预测，不重新调参或筛负载。预测有效工作仍为每请求max(1,名义重算)，不预测TTFT。以下18项原值及原始parquet SHA256均锁入JSON；明确是复用，未声称重新测量。',
        '', '| λ | 种子 | LRU重算% | online重算% | perfect重算% |','|---:|---:|---:|---:|---:|']
    for g in groups():
        cases={c['policy']:c for c in predictions if c['group']==g['name']}
        lines.append(f"| {g['rate']:g} | {g['seed']} | "+' | '.join(f"{100*cases[p]['predicted_effective_recompute']:.6f}" for p in POLICIES)+' |')
    lines += ['',f"CPU总清单SHA256：`{protocol['simulation_run_sha256']}`；原模型提交`{protocol['simulation_commit']}`。",'',
        '## 运行、异常与留存','',
        '运行前工作区/后端必须干净，校验本协议已提交及全部源码/输入SHA。flock单控制器，原子checkpoint；完成项核验原始SHA后跳过，支持断点续跑，不覆盖任何attempt。',
        '非策略中断（服务崩溃或GPU被抢占等）保留原因及所有原始数据，最多重跑一次；第二次失败如实记录。策略异常、指纹或观测正确性错误立即停止，不用重跑掩盖。资源未就绪只等待，不计attempt。',
        '全部数据写持久ext4目录，原始客户端run.json不修改；大prompt本机保留，Git发布去prompt元数据、原始清单与日志/指纹，不复制实现代码。',
        '', '```bash',
        'python -m agentkv.bench.e1_gpu run --protocol docs/e1_v2_preregistration.json --output-dir /home/llb/my_soft/agentkv-runs/e1-v2-config11-20261106 --gpu-index 1 --port 8016',
        '```','',f"协议SHA256：`{sha256_file(V2_PROTOCOL)}`。"]
    V2_DOC.write_text('\n'.join(lines)+'\n')
    print(V2_PROTOCOL,sha256_file(V2_PROTOCOL),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register'])
    parser.parse_args();register()


if __name__=='__main__':main()
