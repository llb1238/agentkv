# M5：完整前缀适配与完整回调验收协议

用户已确认后端提案，并要求共用推导、旧位置表对照、字段可赋值、原有测试、构造耗时和独立fork提交。本协议在新的GPU测量之前提交；结果另附，不修改本节矩阵。

## 来源与后端

- 后端fork：`https://github.com/llb1238/vllm.git`，分支`agentkv`，独立提交`4749dd5961841243596559b10ba0a22c48980c45`。
- 两条路径共用`_iter_offload_keys()`，从hashes与group配置产生相同的key/end-token；创建时登记全部已知hash，公开prefix接口按prompt边界排除decode。
- ReqContext保持可赋值dataclass；字段默认None，旧调用保留。缺边界而调用新枚举接口时明确报错。
- 测试直接从旧提交加载原函数，与新创建/多次更新/decode路径比较最终位置表；分别覆盖hashes_per_chunk=1/2。
- [后端单元验证](../results/m5-vllm-integration-20261104/backend-validation/README.md)：1395通过、10项按原有条件跳过。额外实模型集成因缺Llama快照未通过，单列保留，不替换模型或当作通过。
- 后续GPU启动前两仓库必须干净，客户端run.json顶层和server字段、服务清单及trial/replay记录均保存新vLLM提交。没有源码快照。

## 适配方式

`agentkv.policy.vllm_plugin.HintAwareCachePolicy`继承实际CachePolicy，构造参数仅cache_capacity。`AGENTKV_HINT_POLICY_CONFIG`给出配置文件；启动打印原JSON和SHA256，核对冻结记录、评分源码及适用校准表指纹。perfect沿用冻结deadline；noisy/online沿用冻结remaining，评分代码与表不改。

读取hint时先尝试ReqContext.kv_hints，再尝试kv_transfer_params.kv_hints，按message_id去重记录有效请求。缺hint退化LRU。

三个上下文回调先枚举完整prompt前缀，用冻结的“miss观察、不提升recency”通路补归属/深度，再将原访问列表转交原方法。完整前缀不被伪装成CPU命中或额外写入。CPU模拟入口为新增`agentkv.sim.callback_adapter`，原E1完成写入模型及冻结数据保留。

## 固定GPU矩阵

1. 5分钟完整记录：config-11，Qwen3-8B，16K、8槽位、GPU182880/CPU274320 token，native LRU等价插件，perfect，λ=0.7、人口种子0、稳态初始化、T=300秒、预热0。
2. 前一步的完整序列验证通过后，10分钟冒烟：同配置与人口规则，换冻结HintAware插件，T=600秒、预热0。此为功能/调度开销验证，不作为E1的TTFT收益测量。
3. GPU1显存/算力共享，两卡dmon；只停止自己创建的服务，不干预外部进程。空闲显存不足时停止，不改变配置。

## 日志与重放的边界

`agentkv.callbacks.v1`以gzip保存完整方法入口/出口、单调时间、参数、结果、parent-call ID。key字典保存原始bytes；上下文保存公开请求字段、完整hint与transfer参数、tier filter、位置映射、prompt边界和prefix序列。manager私有state只记录类型名，不复制owner对象；冻结策略不读取这些私有对象。

ChunkStatus的两个字段ref_cnt/chunk_id均保存。每次get/insert/mark等记录相关key状态，每次evict同步所有CPU块状态、完整protected与返回victims。额外只读包装manager的_prepare_load/complete_load/complete_store，记录调用前后的真实ref_cnt变化，覆盖mark_non_evictable先于0→1的时序。

HintAware记录核心每次clock读取；回放逐个重用原clock值，避免把回调入口时间当成内部评分时间。退出前checkpoint、析构/atexit/finally close；120秒服务shutdown timeout，135秒父进程宽限。验收要求真实close footer，checkpoint不能替代。

## CPU序列验证

- 原LRU流先完整重放到native LRU，所有返回值、成员关系、ref_cnt及footer计数一致，嵌套回调只由根调用触发一次。
- 同一条LRU流上的候选策略采用**明确标记的决策回放**：在记录的LRU缓存路径上比较两个适配层的驱逐选择，回滚候选预览，再显式应用记录的LRU victims。所有其他方法按原顺序调用，不跳过重复insert、不私自修复成员不一致。这只验回调/安全性，不是候选策略闭环命中率。
- 10分钟HintAware真实流则执行**完整的策略与时钟重放**，两个适配层的实际选择均须等于实录，成员状态不回滚或替换。

同时核验：

1. 每次evict恰好n个或None，None时无成员变动；victim的ref_cnt必须为0且不在protected。
2. 每个带hint的finished之后链为waiting或final；waiting deadline等于实际评分clock加expected_gap_ms。
3. 全部resident prefix归属和深度正确；多批store与单批结果相同的独立CPU测试通过；已构造的真实final共享key最小反例通过回归。
4. 每次驱逐采样全部CPU块中已归属的比例；未知块走冻结LRU分支。
5. 全序列无异常、返回值/状态匹配、结束计数完整；按message_id对账客户端实际hint数。

本配置为单组完整注意力、16-token块，每个上下文snapshot的prefix长度须等于floor(prompt_len/16)。短于一块返回0。该等式不推广到多组/部分边界模型。

## 开销

记录_create_req_context函数本身的P50/P99微秒耗时，日志写入放在计时之后。evict分别报告评分核心、适配层（含归属比例扫描）、完整实录回调入口至出口耗时；实录额外扫描、压缩与日志开销单列，不能将核心耗时当作全部服务开销。报告hint数、成功/失败、归属比例和异常。

## 入口

```bash
python -m agentkv.bench.policy_trial --kind probe --output-dir ../agentkv-runs/agentkv-full-probe-20261104
python -m agentkv.analysis.replay_callbacks --run-dir ../agentkv-runs/agentkv-full-probe-20261104 --output-dir ../agentkv-runs/agentkv-full-probe-replay-20261104
# 上一项通过后执行
python -m agentkv.bench.policy_trial --kind smoke --output-dir ../agentkv-runs/agentkv-full-smoke-20261104
python -m agentkv.analysis.replay_callbacks --run-dir ../agentkv-runs/agentkv-full-smoke-20261104 --output-dir ../agentkv-runs/agentkv-full-smoke-replay-20261104
```

## 采集后的关闭修复

第一段采集未通过gzip/退出footer验收，原始数据已[归档](EVIDENCE.md#archives)。只向API主进程发送正常SIGTERM，让vLLM管理子进程退出，避免进程组与API管理器的重复SIGTERM在EngineCore恢复默认处理后触发退出竞态。超时强制清理仍只针对自建进程组。没有更改负载、矩阵、评分或后端提交；后续新目录重新采集。197项测试通过。

## 有效运行结果

[最终报告](../results/m5-vllm-integration-20261104/README.md)：a736e2b/4749dd5961下的有效5分/10分采集分别1284/2259请求，全成功，hint和prefix对账，五不变量与实录完整性通过。HintAware实录的235万次回调按clock完整重放一致；每次驱逐归属100%。构造函数P50/P99=0.297/1.356ms，evict核心0.345/10.105ms、适配层2.239/11.803ms、完整插桩10.252/21.419ms。功能门禁通过；没有无插桩配对基线，不作端到端零开销结论。
