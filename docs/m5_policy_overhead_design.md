# M5：无插桩CPU优化与端到端开销测量（测量前登记）

固定输入：`results/m5-vllm-integration-20261104/smoke/callback-traces/`的完整HintAware序列；原实现与clock读取次序/实际evict key为参考。cProfile只包围适配层方法，JSON/gzip解析在计时外；无profile的独立回放用于时延门槛。逐次比较全部2005次evict的有序key和None，另保存指纹。原冻结评分实现保留为参考，优化采用等价实现，不改分数公式或校准表。

CPU目标：关闭全部运行时插桩后，适配层加核心evict P50<0.5ms、P99<2ms。记录最大值、最长调用及_compact/事件处理工作量。优化先基于profile证据；完整前缀首次按请求计算并缓存，finished清理。事件/堆维护可以分摊，但到每次选择时必须得到与参考相同的答案。

正式插桩默认轻量：仅内存直方图和计数器，退出一次写出；完整记录仅显式调试启用。直方图分辨率与误差写入产物。两种服务使用同样观察器；记录scheduler单步和所有CachePolicy回调分布、名义来源计数、请求TTFT和两卡dmon。实际每请求发送/结束时间可在内存保留到退出，为客户端请求指标关联，不逐请求写服务器日志。

GPU固定矩阵：config-11（16K、8槽位、GPU182880/CPU274320 token）、λ=1根/分钟、人口种子3、稳态初始化；native LRU与HintAware+none各20分钟，预热5分钟。顺序LRU后HintAware+none，分别新服务/新目录；不改负载。none的数学退化为LRU，任何专用快路径必须显式记录并经过逐次决策等价验证，不能把独立策略替换伪称为generic hint评分成本。

报告GPU/CPU/重算token加权比例、TTFT P50/P99的绝对与相对差、scheduler单步分布差及回调分布。TTFT P50比较定义为`(hint_none/lru-1)*100%`，同时报告绝对值；绝对偏差超过2%不进入E1，继续定位/优化。人口实例列表和输入指纹必须相同；闭环因完成时间不同可能使窗口内发送数有差异，不假称请求序列完全相同。共享GPU资源状态及重试均保留，不删不利结果。

2026-11-04追加：首轮运行提交0ca891d，两组窗口各3448请求且全部成功；TTFT P50为48.361627/50.1037605ms，差值+3.602305%，未过门槛。原始客户端run.json和请求数据保留；退出汇总器的uint64推断溢出在9925c70中修复，不修改原始测量。后续afdbf0f将none路径改为直接继承原生LRU方法，去掉动态属性转发及重复包装。再次按相同矩阵完整重跑两组，不更改参数；首轮失败与后续结果并列报告。重跑前已验证完整vLLM包装层关闭插桩时2005次驱逐逐次一致（P50约0.068ms、P99约1.50ms），新none路径1128次驱逐逐次一致。正式GPU测量之间不并行运行CPU profiling。

使用说明：正式入口`python -m agentkv.bench.policy_overhead --output-dir ../agentkv-runs/NEW --gpu-index 1 --port 8015`默认设置轻量观察目录；`AGENTKV_INSTRUMENTATION=light`为默认模式，`full`只用于调试，`off`用于关闭插桩的CPU回放。计时直方图使用nearest-rank桶边界，桶宽不超过1/128；每个进程退出写一个文件。按客户端一次性发布的`experiment-clock.json`选择统计窗口，不在策略回调中读写文件。原始客户端run.json在退出前生成，退出汇总完成后以overhead-run.json及紧凑request-metadata.parquet关联来源信息，保留原始文件字节。

结果与图由`python -m agentkv.analysis.policy_overhead_report --result-dir results/m5-policy-overhead-20261104 --first-attempt results/archive/m5-policy-overhead-attempt1-20261104`生成。原始大请求文件保留本机并由Git忽略；缺少大文件时，汇总器验证已有紧凑元数据SHA256后重建报告。结果见[完整报告](../results/m5-policy-overhead-20261104/README.md)，未执行E1。

> 收尾展示修订：仅将运行位置改为相对布局，参数、顺序和判据不变；原始展示字节可用 `git show f59d0026ac9c2d9942d58bed84d90f5dbc1dc580:docs/m5_policy_overhead_design.md` 查看，JSON协议保持原字节。见[路径约定](REPRODUCING.md)。
