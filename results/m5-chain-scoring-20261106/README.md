# Online 完整回调输入与链级优化的语义门槛

状态：第一步采集完成，第二步前置可行性检查发现约束不相容，已按用户要求停止。**尚未实现新链优先队列，没有进行2ms性能验收、三组20分钟GPU开销实验、E1 v2预注册或运行。** 此目录是有效的后续回放输入和反例证据，不是优化性能结果。

采集提交`cc77bda3f65745b3c1d79be279107369f85e07ea`；冻结驱逐回放核验提交`86814f7db31e377b6e24a46fa3cc6b09101797c7`；返回集合复核提交`acb954ff2094140d55a5051b58667c19aa8bc910`；vLLM仍为`4749dd5961841243596559b10ba0a22c48980c45`。六个实际策略/适配实现文件均与E1 v1预注册SHA256相同，未修改评分、校准或尾序开关。

## 5分钟真实 online 采集

config-11：16K、8槽位、GPU182880/CPU274320 token；λ=1根/分钟、人口种子3、稳态初始化，T=300秒、预热0。服务和两卡dmon均正常退出。

- 779个请求成功，失败0；发送hint与策略完成hint均为779。
- 完整记录875811个回调，含touch、on_store_miss、on_request_finished、get、insert、evict、mark操作等，保留参数、ReqContext、位置表、保护集合、时钟和可变ref_cnt。
- gzip EOF/footer、回调计数和hint对账通过。采集使用完整CallbackTraceMixin；策略本体仍走E1 v1的轻量/索引实现。采集开销会影响闭环时序，因此不把这5分钟的延迟用于性能验收。
- [原始客户端run.json](online-trace/run.json)、[采集清单](online-trace/trial-run.json)、[完整回调的外置清单](../../docs/publication_payloads.json)、[紧凑请求元数据](../../docs/EVIDENCE.md#requests)均保留原始字节。
- 完整prompt数组位于`../agentkv-runs/m5-chain-scoring-20261106/online-trace/requests.parquet`，SHA256见原始run.json与[总清单](run.json)，不加入Git。其他原始记录也保留在该持久ext4目录，未复制实现源码。

回调文件SHA256：`bec1b3a4e8aa8aa57fd02af28e00b40557db863b62cc355d596bf840f9d4bc11`。

## 冲突发生在同分块的链内次序

冻结的online和perfect均为`tail_first=false`。现有实现对相同预测分数使用块的recency rank打破平局，较小rank先驱逐。块的最近访问只更新实际回调涉及的部分，rank顺序不保证等于深度倒序。

只读检查在冻结实现实际删除块前查看状态，并继续执行原实现，验证每次返回结果与录制序列一致。以下两处都存在同分、完全相同链归属的更深块，两块均为ref_cnt=0、均不在protected中；深块也没有出现在该次原始返回集合中。

| 序列 | 实际调用ID | 本次返回数 | 原返回浅块深度 / rank | 仍可驱逐深块深度 / rank | 深块在原返回集合中 |
|---|---:|---:|---:|---:|---|
| 新online | 196992 | 283 | 5440 / 44903 | 5456 / 65645 | 否 |
| 旧perfect | 355435 | 509 | 7088 / 21850 | 7104 / 54090 | 否 |

online的两块都由同一个waiting链拥有，分数线均为`(a=1,b=62.88686901763237)`，即同一预计返回时刻；perfect两块的deadline均为`1670907.074524668`秒。这里没有“被更优链共享”或保护状态的差异可用来跳过深块。

严格深度尾序如果返回表中的浅块，就必须先返回仍可驱逐的深块；但冻结集合包含浅块而不包含深块。因此矛盾涉及**返回集合**，并非只改变同一集合中的排列。浅块在原有返回列表中的0起始位置分别为197和418。

完整key、所有者、分数、时间、ref_cnt和保护状态见[online反例](online-check/run.json)与[perfect反例](perfect-check/run.json)；返回集合复核分别见[online](online-check/membership-check.json)、[perfect](perfect-check/membership-check.json)。此外，[两块最小反例](minimal-order-counterexample.json)和参数化回归测试分别覆盖online、perfect：父链请求完成后，子链仅触碰共享深块，冻结rank顺序先选择浅块，严格尾序会选择深块。

## 已验证与未验证的边界

| 数据 | 完整回放回调数 | evict次数 | 与录制有序key逐次一致 | evict SHA256指纹 |
|---|---:|---:|---|---|
| 新online | 875811 | 734 | 是，未改动的冻结实现 | `01fe734e2b2ea34b13800544feb8df8de5ec44bf521e6905dde72fb3ee357200` |
| 旧perfect | 2349756 | 2005 | 是，未改动的冻结实现 | `09fc5af2fb5f632f3f98cf3ade5a9d6c70c09c27cdf256f2aa48222416216746` |

perfect使用[既有smoke完整序列](../m5-vllm-integration-20261104/smoke/)，其指纹与此前性能优化验证相同。新的online实际包含驱逐，与完美预测序列不能互相替代。

这里没有新链队列可宣称与旧实现等价；检查器只添加只读观察，不改变冻结选择。检查器的时间包含检查开销，**不是**无插桩性能结果。231项测试通过、3项历史后端测试跳过，见[测试日志](../../docs/EVIDENCE.md#runtime-logs)。

后续需要先确认规则：

1. 保留冻结的逐次返回集合与顺序，按链维护随时间变化的分数，链内继续使用既有rank次序；这才可以继续以“只改实现性能”为目标。
2. 采用严格深度尾序，承认策略语义改变，单独预注册，不能把后续差异只归因于实现性能。

## 复核入口

在干净工作区中设置项目及依赖路径，使用vLLM虚拟环境Python：

```bash
export PYTHONPATH=.:../agentkv-runs/python-tools
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
../vllm/.venv/bin/python -m agentkv.bench.check_chain_scoring --run-dir results/m5-chain-scoring-20261106/online-trace --output-dir ../agentkv-runs/NEW-online-check --hint-tier online
../vllm/.venv/bin/python -m agentkv.bench.check_chain_scoring --run-dir results/m5-vllm-integration-20261104/smoke --output-dir ../agentkv-runs/NEW-perfect-check --hint-tier perfect
```

新版本检查器会直接在run.json中写入返回集合复核。此次保留了最初完整回放的原始run.json，随后以独立membership-check.json关联其SHA256，未重写旧元数据。总清单记录所有发布数据的指纹；最小反例测试为`tests/test_chain_scoring_order.py`。

收尾分发说明：完整回调流保留本机原字节与SHA，已停止Git跟踪；需要恢复载荷才能执行完整CPU回放，见[复现说明](../../docs/REPRODUCING.md)。
