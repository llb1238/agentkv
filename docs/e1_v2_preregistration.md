# E1 v2 GPU 预注册

本文件及配套JSON必须先单独提交，再开始任何E1 v2 GPU运行。提交后正文不改；结果另存结果目录或只在末尾追加。

## 先决条件与冻结范围

online与perfect完整回调及原734/2005次基线，逐次evict返回的有序key列表完全相同。online关闭插桩的完整适配层finished/evict P99为1.079/1.959ms，均≤2ms。三个20分钟GPU门槛全部通过，online/perfect的scheduler.schedule P99相对LRU保守增量为1.802/1.343ms。
证据：[链级实现与门槛](../results/m5-chain-implementation-20261106/README.md)，GPU测量提交`415f4778699b11c5a6b520986b48d351e196af3a`；本次运行实现冻结于`dd8c7fb220570745645df573ad062c19b228a4bf`，所有Python源码SHA256见JSON。
v2仅更换实现性能：链级评分、链内原rank，每个victim后重新入队；共享块按最佳链保护，原冻结公式/校准/事件账本不变。决策等价是同一录制时钟/回调流下的逐次证明；真实闭环的完成时间与请求交错可能变化，不宣称两次GPU运行的victim流相同。

## 配置和共同输入

config-11：16K、8槽位、GPU182880/CPU274320 token，Qwen3-8B，vLLM fork `4749dd5961841243596559b10ba0a22c48980c45`。评分语义冻结`23c2d6d909e87c1b5a7e1aa1bd11bbd84f0708db`，校准文件原样保留。
λ∈{1,1.4}根会话/分钟，人口种子3/4/5，T=2400秒、预热600秒、收尾0；稳态初始化，安全阀128，超时180秒、收尾等待120秒。GPU1，每个策略新服务；不占用/终止其他进程，记录两卡dmon及共享状态。
策略：内置LRU、online、perfect。LRU传输online信封但忽略hint；另两者显式implementation=chain，轻量插桩、有效prefill观察器和scheduler计时相同。省略LRU+final，不更改其他矩阵项。
沿用v1人口、实例置换、逻辑请求ID（e1-加组名）、workload及schedule。所有共同逻辑请求逐项核对token哈希/置换种子/长度/max_tokens；闭环实际发送集合可能不同，不把相同人口误写为完全相同的发送顺序。

## 固定顺序

沿用random.Random(20261104)预先打乱的v1组内顺序，仅移除LRU+final，不重新抽样。每组连续，非策略中断最多原样重跑一次。

| λ | 人口种子 | 顺序 |
|---:|---:|---|
| 1 | 3 | lru → online → perfect |
| 1 | 4 | lru → online → perfect |
| 1 | 5 | online → lru → perfect |
| 1.4 | 3 | perfect → online → lru |
| 1.4 | 4 | online → perfect → lru |
| 1.4 | 5 | lru → online → perfect |

## 指标、稳定性与分析

稳定性严格沿用a81dd86：统计窗零等待采样≥80%且等待P99≤8；队列OLS斜率×1800<1；后半/前半TTFT中位数≤1.5；确有KV不足阻塞且等待>0时运行数≥7。缺观察记无法判定，不默认稳定。
主指标仅稳定运行：有效重算比例（实际完成前向prompt token工作量/成功非is_entry发送入窗请求的prompt token）、TTFT P50/P99。有效工作量包含末token replay与重复prefill，不能用名义比例替代。逐人口报告原值及相对同人口LRU差值，仅策略和LRU都稳定的配对进入均值/最小–最大；注明配对数。样本3个人口，不做显著性检验。
全部18次报告稳定状态、窗口完成数（按结束时刻入窗，全部/非is_entry），所有不稳定与失败单列，不进入稳定均值。λ=1.4列等待曲线、稳定次数、完成数。
次要指标：16-token整块名义重算三分类及间隙请求平均排队；有效工作与名义差额、尾块/replay/重复工作；hint发送/接收/消费对账；策略和scheduler轻量耗时直方图；两卡dmon。
每λ一张主表，配对差值均值/范围；重算图叠加CPU预测（不稳定实测区别标识）；稳定人口等权TTFT CDF。与v1并列列出三种共同策略、逐人口原值/稳定性/完成数及相对各自同人口LRU的配对差值；不把不同时段绝对差当纯因果估计。

## 固定CPU预测

缓存/评分语义未变，直接复用v1已登记且完成的固定0.5秒、无限并发、零搬运预测，不重新调参或筛负载。预测有效工作仍为每请求max(1,名义重算)，不预测TTFT。以下18项原值及原始parquet SHA256均锁入JSON；明确是复用，未声称重新测量。

| λ | 种子 | LRU重算% | online重算% | perfect重算% |
|---:|---:|---:|---:|---:|
| 1 | 3 | 18.730292 | 17.402121 | 12.309402 |
| 1 | 4 | 26.241866 | 23.930944 | 17.245799 |
| 1 | 5 | 23.794708 | 22.958928 | 16.034451 |
| 1.4 | 3 | 30.670100 | 28.359201 | 19.642646 |
| 1.4 | 4 | 35.200457 | 29.780521 | 20.411496 |
| 1.4 | 5 | 35.247603 | 32.206001 | 22.010356 |

CPU总清单SHA256：`1386cec40ca060530e723d0c15845f6ad629b09bb6eadd3771ac328ae2898b7a`；原模型提交`0800a144cc18b8ef73663b33c8d1dcddb8124779`。

## 运行、异常与留存

运行前工作区/后端必须干净，校验本协议已提交及全部源码/输入SHA。flock单控制器，原子checkpoint；完成项核验原始SHA后跳过，支持断点续跑，不覆盖任何attempt。
非策略中断（服务崩溃或GPU被抢占等）保留原因及所有原始数据，最多重跑一次；第二次失败如实记录。策略异常、指纹或观测正确性错误立即停止，不用重跑掩盖。资源未就绪只等待，不计attempt。
全部数据写持久ext4目录，原始客户端run.json不修改；大prompt本机保留，Git发布去prompt元数据、原始清单与日志/指纹，不复制实现代码。

```bash
python -m agentkv.bench.e1_gpu run --protocol docs/e1_v2_preregistration.json --output-dir ../agentkv-runs/e1-v2-config11-20261106 --gpu-index 1 --port 8016
```

协议SHA256：`f1d0649a68484ffd53407d8122427c262aeefd28ce28a53c70effca84d50dfc4`。

> 收尾展示修订：仅将运行位置改为相对布局，参数、顺序和判据不变；原始展示字节可用 `git show f59d0026ac9c2d9942d58bed84d90f5dbc1dc580:docs/e1_v2_preregistration.md` 查看，JSON协议保持原字节。见[路径约定](REPRODUCING.md)。
