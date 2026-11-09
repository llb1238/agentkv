# AgentKV 项目状态与收尾计划

2026-11-09 · 实验阶段已结束。公开版 `llb1238/agentkv` 采用无历史的 lean 快照；私有仓库 `llb1238/agentkv-private` 保留完整 Git 历史与原始证据。后续工作为 CPU 复现与发布维护。 最终事实以 [REPORT](REPORT.md) 和[结果索引顶部](../results/INDEX.md)所列目录为依据；本页替代早期排期，原计划可从 Git 历史查看。

## 模块状态

| 模块 | 当前状态 | 已交付 / 边界 |
|---|---|---|
| M0 生命周期与代价测量 | 完成 | GPU/CPU/重算路径、TTFT 分解；[M0](../results/m0-lifecycle-20261102/README.md) |
| M1 负载与回放器 | 完成 | 固定 AgentX、确定性前缀、事件锚定、稳态初始化、假服务压力测试、真实闭环回放；[v3](../results/m1-workload-seed0-v3/README.md)、[稳态](../results/m1-steady-state-seed0/README.md) |
| M2 可观测性 | 完成，口径有边界 | 名义三分类、有效前向工作量、队列与 KV 阻塞、轻量直方图及两卡 dmon；[E0](../results/m1-e0-config11-20261103/README.md)、[E1 v2](../results/e1-gpu-v2-config11-20261106/README.md) |
| M3 两层模拟器 | 完成，过载预测不成立 | LRU、hint、块级未来访问 Oracle 共用接口；固定服务、零搬运，不能预测真实稳定性；[留出模拟](../results/m5-scoring-evaluation-20261104/README.md) |
| M4 Hint 协议与适配 | 完成项目内接入 | 官方信封、兼容 HTTP 通道、消费对账；完整前缀后端补丁4749dd5961独立提交；未宣称上游合入；[适配验收](../results/m5-vllm-integration-20261104/README.md) |
| M5 CPU 驱逐策略 | 完成实现与评估，性能目标未全通过 | 条件剩余期望、final 优先、链级 rank、惰性刷新和空链回收；完整序列决策一致；长时 GPU 调度增量仍超2ms；[最终诊断](../results/m5-chain-growth-20261107/README.md) |
| M6 GPU 层组合与预取 | 未执行，延期 | 未验证与其他 GPU 保留/预取策略的组合，不列收益 |
| M7 在线间隔预测 | 完成当前范围 | 链历史 EMA、独立根校准、noisy/online 对照；没有真实工具名或跨 trace 泛化评估；[选型](../results/m5-scoring-development-20261104/README.md) |

## 实验状态

| 编号 | 期望产出 | 最终状态 |
|---|---|---|
| E0 | 重算来源、同人口模型与 Oracle、稳定性 | 完成登记矩阵；不稳定项单列；[报告](../results/m1-e0-config11-20261103/README.md) |
| E1 | **主指标对比表 + TTFT 的 CDF 图** | v1 与 v2 完成；λ=1有稳定配对，λ=1.4无稳定主值；[v1](../results/e1-gpu-config11-20261105/README.md)、[v2](../results/e1-gpu-v2-config11-20261106/README.md) |
| E2 | 容量、模型、间隔等收益边界 | 只完成候选容量筛选及到达率边界；广泛 GPU 消融未执行 |
| E3 | 预测来源与误差鲁棒性 | CPU 开发/留出完成；GPU 仅 online/perfect，未完成 noisy/random 的完整 GPU 矩阵 |
| E4 | GPU/CPU 策略组合 | 未执行，延期 |
| E5 | 回调与调度开销 | CPU 一致性及局部耗时通过；长时 GPU 2ms 增量目标未达到；失败尝试保留 |
| E6 | 模拟器与实测校准 | 已报告稳定区间差距及过载失配；不宣称全区间校准通过 |
| 补充：稳定负载上限 | λ=1.2的稳定次数、完成数及稳定配对 | 完成；LRU/online各0/3、perfect 2/3，无稳定LRU配对；[报告](../results/e1-gpu-stable-limit-12-20261107/README.md) |

## 实际时间线

| 时间 | 完成事件与证据 |
|---|---|
| 2026-10-26—11-01 | 环境与 M0 生命周期/TTFT 探索，形成[M0证据](../results/m0-lifecycle-20261102/README.md) |
| 2026-11-02 | AgentX 数据审计、前缀与链结构核查；[数据报告](../results/m1-agentx-data-20261102/README.md) |
| 2026-11-03 | v3事件锚定、稳态回放、容量与两层模型、Oracle，完成E0登记矩阵；[E0](../results/m1-e0-config11-20261103/README.md) |
| 2026-11-04 | 三候选预注册、开发组选型与人口留出；完整前缀补丁、回调一致性和冒烟；[CPU留出](../results/m5-scoring-evaluation-20261104/README.md)、[适配](../results/m5-vllm-integration-20261104/README.md) |
| 2026-11-05—11-06 | E1 v1首批产物丢失后排除，持久目录完整重跑；[v1](../results/e1-gpu-config11-20261105/README.md) |
| 2026-11-06—11-07 | 链级评分保留原rank，逐次驱逐等价，通过短时门槛并完成E1 v2；正式长窗口暴露开销超标；[v2](../results/e1-gpu-v2-config11-20261106/README.md) |
| 2026-11-07—11-08 | 80分钟CPU状态增长诊断与回收、全部录制等价核验、长时开销复核；λ=1.2先预注册再完成矩阵；[诊断](../results/m5-chain-growth-20261107/README.md)、[上限](../results/e1-gpu-stable-limit-12-20261107/README.md) |
| 2026-11-08起 | 停止实验；整理REPORT/README、证据索引和许可 |
| 2026-11-09 | 公开版采用无历史lean快照；完整历史和证据保留在agentkv-private，补充CPU复现与公开源码链接 |

## 收尾验收

- [x] 结果目录保留旧版、负收益、失败与不稳定运行，不用聊天摘要代替丢失证据。
- [x] 预注册、冻结与留出分开；固定录制上的优化有逐次有序驱逐核验。
- [x] 将未通过的长时GPU开销、模拟器过载失配写入最终报告。
- [ ] 用户审阅README与REPORT；本次提交不代表审阅通过。
- [x] 公开版使用无历史 lean 快照；私有仓库保存完整历史，原始证据单独打包并按需提供。

## 复现与后续工作

当前入口为 `agentkv.bench`、`agentkv.hints`、`agentkv.sim` 和 `agentkv.analysis`。CPU汇总与图的复现见 [REPRODUCING](REPRODUCING.md)；历史正式测量要求干净源码、固定提交和输入指纹，`--allow-dirty` 只用于调试。

未执行的容量/模型消融、GPU预取组合、独立trace泛化和进一步开销优化均为后续方向，不在本次收尾中启动。公开状态、许可、原始载荷与历史检查见 [PUBLICATION_CHECK](PUBLICATION_CHECK.md)。
