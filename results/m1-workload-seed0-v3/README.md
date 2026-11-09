# M1 离线负载 v3：事件锚定与回放时长估算

2026-11-03。使用与 [v2](../../docs/EVIDENCE.md#archives) 相同的 100 个根会话和默认参数，改为事件锚定链启动。实现：[workload.py](../../agentkv/bench/workload.py)、[estimate_replay.py](../../agentkv/bench/estimate_replay.py)；设计：[第 4.4、5.1、7.1 节](../../docs/m1_replayer_design.md)。

## 链启动规则

为链 C 选择同一根会话内、其他链中 `end_A ≤ t_C` 且原始结束时间最晚的请求 A，链首写入 `anchor_trace_req_idx`。在线等待 A 实际完成，再等待 `start_delay_s = min(t_C - end_A, G_cap) × c`。无请求锚点时，前者为空，启动延迟为 `min(t_C, G_cap) × c`，从根会话到达计时；非链首两列均为空。

结束时间并列时按 `(end, t, chain_turn, chain_id, trace_req_idx)` 取最大值；同刻零耗时请求按严格事件顺序 `(t, chain_turn, chain_id, trace_req_idx)` 只允许向前依赖，避免循环。子 Agent 可共享同一父请求锚点并行启动，锚点也可以是链内中间请求。

`chain_offset_s` 和 `chain_offset_orig_s` 保留对照，**仅供参考，在线调度不使用**。manifest 的 `root_timeline_duration_s`、`root_offset_reference_duration_s` 也为旧偏移口径的参考统计，不代表事件锚定完成时间。manifest 和 Parquet 元数据均标记 `event-anchored-v1`。

本次 **2,096 条链锚定请求，178 条链锚定根会话到达**；启动延迟中位数 1.433 秒、P90 40.4671 秒、最大 120 秒。

## 固定服务时间估算

每根会话独立以到达为 t=0；每个请求的回放服务时间恒定为 s，替代原始 `api_time`。沿请求依赖图计算：

```text
链首发送时间 = 锚点回放完成时间 + start_delay_s
无请求锚点的链首发送时间 = start_delay_s
链内后续发送时间 = 前一请求回放完成时间 + gap_after_s
每个请求完成时间 = 发送时间 + s
根会话完成时间 = 根内所有请求完成时间的最大值
```

保留负载中的截断、压缩后间隔；允许无限并发，不包含网络开销、服务端排队或在途请求上限。根完成时间不在 25 分钟处截断。中位数和 P90 按根等权，P90 采用线性插值。

| s（秒/请求） | 根完成时间中位数（秒） | P90（秒） | 最大值（秒） | 25 分钟内发出的请求数 | 占全部请求比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.2 | **1,581.7385** | **8,408.9569** | **62,166.3080** | **10,547 / 28,812** | **36.6063%** |
| 0.5 | **1,602.4385** | **8,489.5969** | **62,570.1080** | **10,388 / 28,812** | **36.0544%** |
| 1.0 | **1,633.3295** | **8,623.9969** | **63,243.1080** | **10,162 / 28,812** | **35.2700%** |

25 分钟占比按每个请求的**实际推算发送时间 ≤1500 秒**计数，分母为全部 28,812 个请求。这与 v2 报告中“前 25 分钟启动的链包含多少请求”的统计条件不同，不能直接比较两个比例。当前参数下，中位完成时间仍为约 26.4–27.2 分钟，长尾来自依赖路径上的链内间隔及启动延迟。

机器可读结果：[replay_estimates.json](replay_estimates.json)。100 根 ×3 种 s 的明细：[root_completion_times.csv](root_completion_times.csv)。估算日志：[replay-estimates.log](../../docs/EVIDENCE.md#runtime-logs)。

## 参数、来源与确定性

`seed=0`、`num_roots=100`、`context_budget=32512`、`gap_cap=120` 秒、`gap_compress=1.0`、`noise_seed=0`、`ema_alpha=0.5`。包含 **100 根、2,274 条链、28,812 个请求**。在线 hint 先验继续取未抽中的 293 根、62,159 个链内间隔的处理后中位数，值为 3.575 秒。

trace revision：`23f152f6f0f9399a85901b89a6458def0ef16729`；原始 trace SHA256：`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。复用固定审计中的链划分和本地 Qwen3-8B tokenizer。来源校验值、抽样根、词表、实现哈希、参数、先验和统计见 [manifest.json](manifest.json)。

两个独立进程重新读取 trace 并生成本目录和 [v3-repeat](../../docs/EVIDENCE.md#archives) 的负载，两个文件均为 **112,240,904 bytes**，SHA256 完全相同：

```text
38fe20ee8e3316124fc25cb4bb296f914cf3d8ee3cdc0fceac7d4f00854bd9e8
```

## 第 7.1 节验证

| 项目 | 结果与范围 |
| --- | --- |
| 确定性 | 两份独立生成的 Parquet 字节级 SHA256 相同 |
| 位置唯一性 | 重新扫描全部 393 根、98,827 次模型请求、338,052,834 次块出现，冲突 0 |
| LCP 一致性 | 全部 26,538 对链内相邻请求，以及每根 128 对随机同根请求，共 12,800 对，种子 7121 |
| 预算 | 全部 28,812 个请求满足 `prompt_len + max_tokens ≤32512`，并检查合法词表及特殊 token 排除 |
| 间隔 | 全部 26,538 个有效间隔非负且符合公式；截断 2,227 个，占 8.3917%，与 manifest 一致 |

新增验证独立穷举每条链的候选请求，核对全部 **2,274 个锚点及启动延迟**；检查同根、跨链、最晚完成条件、非链首空值和依赖无环。按原始事件顺序独立计算三种 s 的 **86,436 个发送时间**，与从实际 Parquet 构建依赖图的模拟器逐项完全一致。参考偏移仍独立核对，浮点绝对容差 `1e-8` 秒；确定性仍要求字节级 SHA256 相同。

证据：[validation.json](validation.json)、[validation.log](../../docs/EVIDENCE.md#runtime-logs)、[全量位置检查](position-check/result.json)、[扫描日志](../../docs/EVIDENCE.md#runtime-logs)。29 项单元测试通过，覆盖共享父锚点、无锚点、同刻零耗时、链内中间锚点、循环拒绝、25 分钟发送边界和原有离线行为，见 [unit-tests.log](../../docs/EVIDENCE.md#runtime-logs)。

## 复现

在仓库根目录运行。生成器拒绝覆盖已有负载；以下构建命令用于产物不存在时，重复构建应选择新的输出目录，并给验证脚本传相应的 `--first`、`--second`、`--position-check`。

```bash
export PYTHONPATH=../agentkv-runs/agentkv-parquet-tools:../agentkv-runs/agentkv-week1-plot-tools
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=../vllm/.venv/bin/python

python results/m1-workload-seed0-v3/position-check/check_positions.py
"$PYTHON" -m pytest tests -q
"$PYTHON" -m agentkv.bench.workload \
  --output-dir results/m1-workload-seed0-v3 \
  --position-check results/m1-workload-seed0-v3/position-check/result.json
"$PYTHON" -m agentkv.bench.workload \
  --output-dir results/archive/m1-workload-seed0-v3-repeat \
  --position-check results/m1-workload-seed0-v3/position-check/result.json
"$PYTHON" tests/validate_workload.py
"$PYTHON" -m agentkv.bench.estimate_replay \
  --workload results/m1-workload-seed0-v3/workload.parquet \
  --service-times 0.2 0.5 1.0
```

大型 Parquet 本机保留并由 `.gitignore` 排除；参数、SHA256、报告、估算与验证证据可随代码保存。依赖版本：[requirements.txt](../../docs/EVIDENCE.md#other-evidence)。两次构建墙钟耗时 88.66 秒、85.18 秒；全量位置扫描 81.50 秒。执行证据：[timing.txt](../../docs/EVIDENCE.md#other-evidence)、[repeat timing](../../docs/EVIDENCE.md#archives)、[validation-timing.txt](../../docs/EVIDENCE.md#other-evidence)、[estimation-timing.txt](../../docs/EVIDENCE.md#other-evidence)。
