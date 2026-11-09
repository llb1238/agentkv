# 真实回调 key 不完整：逐请求诊断与停止门禁

**结论：直接转发存在共享前缀的 final 归属错误风险，按用户要求停在第一步。** 未实现 vLLM HintAware 适配、完整双适配序列验证或10分钟GPU冒烟；本轮只使用CPU，没有新增GPU运行。冻结评分、校准表和模拟器保持原字节。

## 数量纠正与分类

上一报告将“40个不完整请求”和“6个空列表”写成包含关系，这一表述不准确：**40个不完整请求中只有4个空列表**；另外2个空列表的prompt分别为10和14 token，本来就没有完整的16-token块。原始元数据始终分别记录40和6，未重写。此次检查覆盖二者的并集，共42个请求。

| 缺失原因（互斥主分类） | 40个不完整请求中的数量 | 额外的空列表请求 |
|---|---:|---:|
| GPU命中前缀被CHUNK_LEVEL connector跳过（其他原因） | 40（36部分、4全部） | 0 |
| 存储空间不足导致finished少列key | 0 | 0 |
| 分块存储只登记部分、形成额外缺口 | 0 | 0 |
| prompt不满16 token | 0 | 2 |
| 尚未解释的缺失区间 | 0 | 0 |

40个请求的缺失位置均严格等于`{16, 32, …, gpu_hit}`，无中间/尾部缺口。36个部分缺失请求都另有CPU命中和非空reused集合；这说明其回调覆盖的后缀确有CPU访问，不能证明省略的GPU前缀此时也在CPU。

“已全部或部分在CPU层中”不是这40个请求key缺失的独立原因：manager对被offer的ready CPU块仍会记录到finished列表；store候选也在容量不足的early return之前登记。因此容量不足不能单独解释这类缺失。**省略前缀当时的CPU驻留状态，对40个请求均无法从原探针恢复**，不把它们误标为CPU不存在或CPU已存在。见[source-locations.json](source-locations.json)。

## 块最终是否被所属链的回调列出

以下“块”是请求–块对，重复使用同一物理key会重复计数；“列出”指hint携带该链身份的回调参数实际覆盖，不是声称原LRU建立过HintAware归属表。

| 检查 | 请求数 / 请求–块对 |
|---|---:|
| 缺失前缀已被该链更早的回调全部列出 | 35 / 40 请求 |
| 缺失前缀在整段日志中最终被该链全部列出 | 38 / 40 请求 |
| 缺失前缀始终未被该链任何回调列出 | 2 请求，620 请求–块对 |
| 全部缺失区间 | 13692 请求–块对 |

逐个请求、缺失位置和真实key见[per-request.json](../../docs/EVIDENCE.md#other-evidence)或[Parquet](../../docs/EVIDENCE.md#requests)。全量key从原始finished日志取得；以同一实例、相同原始token前缀进行等值匹配，不生成vLLM hash、不假造漏记的key。同一实例词表置换是一一映射，前缀相等性保持不变。先验仅按位置对照可能误认不同内容，本分析实际比较了前缀内容。

未覆盖的两例：

- `m5-callback-probe-seed0/b4c638cc16e89d17c18da83951a87fbc44dd#13/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:6/0`：缺失217块，整段日志未被本链列出的217块；final=False。
- `m5-callback-probe-seed0/4e687509d3c9c0afa7cc8b884d120eeb0d79#87/4e687509d3c9c0afa7cc8b884d120eeb0d79::root::inferred:10/0`：缺失403块，整段日志未被本链列出的403块；final=True。

## 为什么触发正确性停止条件

日志中有两例等待链的省略前缀此前仅被已final的其他链观察到。第一次：实例`d509a5…#55`的链`inferred:5`已final，随后`inferred:0`第37轮全命中GPU，finished为空；后者的perfect间隔约11.455秒。省略的872块后来才被该等待链的回调列出。另一例`b4c638…#13 / inferred:6 / turn 0`的217块在整段日志都没有被该等待链列出。

构造两个可驱逐、已ready的CPU块，使用第一例真实身份、真实共享key和负载中的hint，另加一个更晚返回的合成控制链。冻结的perfect策略在其他输入完全相同时：

| 输入归属 | 等待链状态 | 被选中的块 |
|---|---|---|
| 原回调省略共享前缀 | waiting，deadline已记录 | 共享块（仍只有final所有者） |
| 补全共享前缀归属的对照 | waiting，同一deadline | 更晚返回的控制块 |

这不是越界读写或ref_cnt安全错误，而是违反“共享块取最早返回时间”的策略语义；遗漏会把正在被等待链复用的块误当成final-only，不能笼统当作未归属块退化LRU。证据见[counterexample.json](counterexample.json)。**这是CPU最小反例，含合成控制链；不是宣称原LRU实跑已经发生HintAware误驱逐，也不证明当时872块全部驻留CPU。**

## 原日志不能支持完整真实序列回放

原二进制只存time/method；get、insert、mark等缺key/ref_cnt，evict缺protected和实际victim key；store_miss/touch只存位置；hint只保存摘要。不能从这些字节恢复完整CachePolicy序列，不能将猜测出的事件和完整hint当作“原样录下”的测试数据。

[witness-original-context.jsonl.gz](../../docs/EVIDENCE.md#raw-traces)只保存两组见证请求的原始上下文日志行，逐字节保留并按原顺序，限制及解压SHA256见[fixture.json](fixture.json)。它是证据子集，不是假充完整回放fixture。反例的完整hint明确由原workload行重建。第二部分的全部不变量、逐驱逐归属比例、双适配一致性仍未验收。

## 退出日志修复（独立完成，CPU验证）

- LRU探针新增幂等`__del__`关闭；已有atexit继续保留。
- opt-in观察插件在EngineCore.shutdown进入时先写checkpoint并flush，整个shutdown完成或抛异常时finally关闭日志；避免只等model_executor拆除后才flush。checkpoint不是close footer，阻塞/强杀不能伪称退出完整。
- 服务`--shutdown-timeout`改为120秒，父进程给自建服务135秒退出宽限；按不超过10秒的短wait轮询，避免45秒提前强杀。
- 相关CPU测试验证析构幂等、shutdown异常仍close、包装幂等及启动参数；未再次使用GPU验证关闭流程，原探针缺失footer的事实保留。

## 后续建议（未实施，等待用户决定）

先补齐“请求完整前缀key→实例/链”的只读观测，将归属登记与真实CPU访问/recency更新分开。不能用补全归属去虚构一次CPU命中或额外写入。仅依赖现有finished列表不足；直接读取ReqContext位置表也不能先验保证覆盖未启动offloading的全GPU命中请求，应在真实请求block_hash边界核验。评分公式不变。

重新采集时必须记录完整key、ReqContext hint/位置映射、chunk状态及ref_cnt转移、protected和返回victims，保留同一单调时钟和实序；之后才能运行要求的完整双适配回放及10分钟冒烟。

## 来源与复现

原GPU探针提交`a4a245634ec75e7db358386ad3554c4d4dfe7e7b`；本次CPU分析/测试提交`d344e7e9b1aa6d2eb27af7a0d22298724cf3f860`；vLLM `60da6135b0349fac50a051e9dcec2bd5b4514400`。正式CPU分析在干净工作区运行，原始探针产物不改。187项仓库测试通过，见[tests.json](tests.json)。

```bash
python -m agentkv.analysis.classify_probe_keys \
  --probe-dir results/m5-vllm-callback-probe-20261104 \
  --workload results/m1-workload-seed0-v3-16k/workload.parquet \
  --output-dir ../agentkv-runs/agentkv-key-diagnosis-rerun
```

本机解释器`../vllm/.venv/bin/python`，`PYTHONPATH=.:../agentkv-runs/agentkv-parquet-tools:../agentkv-runs/agentkv-week1-plot-tools`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。依赖NumPy/PyArrow及本仓库；CPU分析不导入vLLM设备运行时。

[run.json](run.json)保存代码提交与输入SHA256；[artifacts.json](artifacts.json)登记本目录数据指纹；没有复制实现源码。

## 逐请求索引

| # | 实例 / 链 / 轮次 | 完整块 | finished块 | 缺失块 | 同链更早已列出 | 同链最终已列出 | 原因 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `92e0638cb6f1c2438e8b9587408c5f9ed67b#71/92e0638cb6f1c2438e8b9587408c5f9ed67b::root::inferred:1/16` | 450 | 0 | 450 | 0 | 450 | GPU前缀跳过 |
| 2 | `002001296e8a8c38ad9d7cc436d691afc602#69/002001296e8a8c38ad9d7cc436d691afc602::subagent:subagent_001_f552fe6f:entry:48::inferred:0/1` | 91 | 3 | 88 | 88 | 88 | GPU前缀跳过 |
| 3 | `d509a5b878d6623b66a7a9b3ca72c18d3f7a#55/d509a5b878d6623b66a7a9b3ca72c18d3f7a::root::inferred:0/37` | 872 | 0 | 872 | 0 | 872 | GPU前缀跳过 |
| 4 | `b4c638cc16e89d17c18da83951a87fbc44dd#59/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:43/83` | 442 | 150 | 292 | 292 | 292 | GPU前缀跳过 |
| 5 | `002001296e8a8c38ad9d7cc436d691afc602#69/002001296e8a8c38ad9d7cc436d691afc602::subagent:subagent_001_f552fe6f:entry:48::inferred:0/2` | 95 | 7 | 88 | 88 | 88 | GPU前缀跳过 |
| 6 | `b4c638cc16e89d17c18da83951a87fbc44dd#59/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:43/84` | 443 | 23 | 420 | 420 | 420 | GPU前缀跳过 |
| 7 | `5b6e131844fbcb4a57f8cf516831dc679004#93/5b6e131844fbcb4a57f8cf516831dc679004::subagent:subagent_029_36546e2f:entry:630::inferred:0/0` | 31 | 0 | 31 | 0 | 31 | GPU前缀跳过 |
| 8 | `61c4d4d16f2ff84b31dc5970560c55d07dc7#77/61c4d4d16f2ff84b31dc5970560c55d07dc7::root::inferred:3/6` | 266 | 61 | 205 | 205 | 205 | GPU前缀跳过 |
| 9 | `002001296e8a8c38ad9d7cc436d691afc602#69/002001296e8a8c38ad9d7cc436d691afc602::subagent:subagent_003_788ad06a:entry:50::inferred:0/13` | 234 | 146 | 88 | 88 | 88 | GPU前缀跳过 |
| 10 | `43d1f727aec4d8b3f2aec42f816e7fabe193#101/43d1f727aec4d8b3f2aec42f816e7fabe193::root::inferred:0/313` | 992 | 84 | 908 | 908 | 908 | GPU前缀跳过 |
| 11 | `5b6e131844fbcb4a57f8cf516831dc679004#93/5b6e131844fbcb4a57f8cf516831dc679004::subagent:subagent_027_0da4495e:entry:615::inferred:0/1` | 34 | 3 | 31 | 31 | 31 | GPU前缀跳过 |
| 12 | `4e687509d3c9c0afa7cc8b884d120eeb0d79#87/4e687509d3c9c0afa7cc8b884d120eeb0d79::root::inferred:0/150` | 355 | 49 | 306 | 306 | 306 | GPU前缀跳过 |
| 13 | `5b6e131844fbcb4a57f8cf516831dc679004#21/5b6e131844fbcb4a57f8cf516831dc679004::root::inferred:132/53` | 307 | 29 | 278 | 278 | 278 | GPU前缀跳过 |
| 14 | `ecd3ea07309dd8a6b62b2f90fd8c60fa92b2#79/ecd3ea07309dd8a6b62b2f90fd8c60fa92b2::root::inferred:4/12` | 893 | 481 | 412 | 412 | 412 | GPU前缀跳过 |
| 15 | `61c4d4d16f2ff84b31dc5970560c55d07dc7#49/61c4d4d16f2ff84b31dc5970560c55d07dc7::root::inferred:6/96` | 962 | 437 | 525 | 525 | 525 | GPU前缀跳过 |
| 16 | `5b6e131844fbcb4a57f8cf516831dc679004#93/5b6e131844fbcb4a57f8cf516831dc679004::root::inferred:51/120` | 909 | 1 | 908 | 908 | 908 | GPU前缀跳过 |
| 17 | `ecd3ea07309dd8a6b62b2f90fd8c60fa92b2#79/ecd3ea07309dd8a6b62b2f90fd8c60fa92b2::root::inferred:4/24` | 946 | 177 | 769 | 769 | 769 | GPU前缀跳过 |
| 18 | `4e687509d3c9c0afa7cc8b884d120eeb0d79#87/4e687509d3c9c0afa7cc8b884d120eeb0d79::root::inferred:0/165` | 400 | 207 | 193 | 193 | 193 | GPU前缀跳过 |
| 19 | `5fc8495d01a2ad301e8ca7879579f2529029#25/5fc8495d01a2ad301e8ca7879579f2529029::subagent:subagent_025_b2a94c12:entry:73::inferred:27/62` | 801 | 373 | 428 | 428 | 428 | GPU前缀跳过 |
| 20 | `3e119fc3136975fe98cfc7fa73500d6574a0#89/3e119fc3136975fe98cfc7fa73500d6574a0::root::inferred:1/0` | 0 | 0 | 0 | 0 | 0 | 不足一块 |
| 21 | `5fc8495d01a2ad301e8ca7879579f2529029#25/5fc8495d01a2ad301e8ca7879579f2529029::subagent:subagent_025_b2a94c12:entry:73::inferred:27/63` | 802 | 322 | 480 | 480 | 480 | GPU前缀跳过 |
| 22 | `a40785d7169046c12bcab71edcfc4db9c2bf#63/a40785d7169046c12bcab71edcfc4db9c2bf::root::inferred:3/0` | 0 | 0 | 0 | 0 | 0 | 不足一块 |
| 23 | `b4c638cc16e89d17c18da83951a87fbc44dd#13/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:6/0` | 802 | 585 | 217 | 0 | 0 | GPU前缀跳过 |
| 24 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/569` | 132 | 1 | 131 | 131 | 131 | GPU前缀跳过 |
| 25 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/570` | 133 | 1 | 132 | 132 | 132 | GPU前缀跳过 |
| 26 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/571` | 134 | 1 | 133 | 133 | 133 | GPU前缀跳过 |
| 27 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/573` | 136 | 2 | 134 | 134 | 134 | GPU前缀跳过 |
| 28 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/574` | 138 | 2 | 136 | 136 | 136 | GPU前缀跳过 |
| 29 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/575` | 142 | 4 | 138 | 138 | 138 | GPU前缀跳过 |
| 30 | `b4c638cc16e89d17c18da83951a87fbc44dd#99/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:8/576` | 143 | 1 | 142 | 142 | 142 | GPU前缀跳过 |
| 31 | `43d1f727aec4d8b3f2aec42f816e7fabe193#109/43d1f727aec4d8b3f2aec42f816e7fabe193::root::inferred:0/540` | 623 | 542 | 81 | 81 | 81 | GPU前缀跳过 |
| 32 | `43d1f727aec4d8b3f2aec42f816e7fabe193#109/43d1f727aec4d8b3f2aec42f816e7fabe193::root::inferred:0/541` | 624 | 474 | 150 | 150 | 150 | GPU前缀跳过 |
| 33 | `3e119fc3136975fe98cfc7fa73500d6574a0#89/3e119fc3136975fe98cfc7fa73500d6574a0::subagent:subagent_004_768caaa4:entry:44::inferred:0/1` | 91 | 48 | 43 | 43 | 43 | GPU前缀跳过 |
| 34 | `31dd54e63bd3605ad9ab96003ab1ae8f39f3#1/31dd54e63bd3605ad9ab96003ab1ae8f39f3::root::inferred:35/19` | 538 | 178 | 360 | 360 | 360 | GPU前缀跳过 |
| 35 | `0b7f3ca692f15ae2651a310d9411d70794aa#103/0b7f3ca692f15ae2651a310d9411d70794aa::root::inferred:1/119` | 892 | 38 | 854 | 854 | 854 | GPU前缀跳过 |
| 36 | `31dd54e63bd3605ad9ab96003ab1ae8f39f3#3/31dd54e63bd3605ad9ab96003ab1ae8f39f3::root::inferred:35/27` | 563 | 226 | 337 | 337 | 337 | GPU前缀跳过 |
| 37 | `a1c8f806f868f22e423cd1016130ef83b740#73/a1c8f806f868f22e423cd1016130ef83b740::root::inferred:7/75` | 930 | 389 | 541 | 541 | 541 | GPU前缀跳过 |
| 38 | `b4c638cc16e89d17c18da83951a87fbc44dd#59/b4c638cc16e89d17c18da83951a87fbc44dd::root::inferred:45/43` | 490 | 177 | 313 | 313 | 313 | GPU前缀跳过 |
| 39 | `3e119fc3136975fe98cfc7fa73500d6574a0#89/3e119fc3136975fe98cfc7fa73500d6574a0::subagent:subagent_001_393d9df8:entry:40::inferred:0/13` | 142 | 99 | 43 | 43 | 43 | GPU前缀跳过 |
| 40 | `4e687509d3c9c0afa7cc8b884d120eeb0d79#87/4e687509d3c9c0afa7cc8b884d120eeb0d79::root::inferred:10/0` | 403 | 0 | 403 | 0 | 0 | GPU前缀跳过 |
| 41 | `5fc8495d01a2ad301e8ca7879579f2529029#25/5fc8495d01a2ad301e8ca7879579f2529029::subagent:subagent_025_b2a94c12:entry:73::inferred:27/79` | 811 | 39 | 772 | 772 | 772 | GPU前缀跳过 |
| 42 | `ecd3ea07309dd8a6b62b2f90fd8c60fa92b2#79/ecd3ea07309dd8a6b62b2f90fd8c60fa92b2::root::inferred:4/35` | 982 | 122 | 860 | 860 | 860 | GPU前缀跳过 |
