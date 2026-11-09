# 实验日志

## 2026-10-26：项目初始化

### 环境记录 ENV-001

采集时间：2026-10-26 16:01:52–16:01:56 +08:00（Asia/Shanghai）。

| 项目 | 固定值 |
| --- | --- |
| 环境编号 | `ENV-001` |
| AgentKV | 0.1.0 |
| 项目目录 | `.` |
| AgentKV Git 状态 | 尚无提交，初始化文件尚未提交 |
| vLLM 目录 | `../vllm` |
| vLLM tag | `v0.32.0` |
| vLLM 分支 | `agentkv` |
| vLLM 基线 commit | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| vLLM 工作区 | 干净，HEAD 与上述 tag 一致 |
| vLLM 安装版本 | `0.32.0+precompiled`，可编辑安装 |
| vLLM 导入路径 | `../vllm/vllm/__init__.py` |
| Python | 3.12.3 |
| Python 完整版本 | `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]` |
| Python 解释器 | `../vllm/.venv/bin/python` |
| 虚拟环境 | `../vllm/.venv` |
| PyTorch | `2.13.0+cu130` |
| CUDA（nvidia-smi） | 13.0 |
| CUDA runtime（torch.version.cuda） | 13.0 |
| NVIDIA 驱动 | `580.178.04` |
| GPU | 2 × NVIDIA H200 NVL，每张 143771 MiB |
| FlashInfer Python / Cubin | `0.7.0.post1` / `0.7.0.post1` |
| FlashInfer JIT cache | `0.7.0.post1+cu130` |
| 系统 | `Linux-6.8.0-147-generic-x86_64-with-glibc2.39` |

`nvidia-smi` 中的 CUDA 版本表示驱动支持的 CUDA 版本，不代表本机
`nvcc` 工具链版本；PyTorch 使用的 CUDA runtime 单独记录。

GPU 标识：

| GPU index | GPU UUID |
| --- | --- |
| 0 | `GPU-48956eb2-4037-09c5-2e94-39989b4d2e0d` |
| 1 | `GPU-667cc66c-113c-1591-1ec1-f4762d255186` |

### nvidia-smi 原始输出

以下显存、温度和进程信息仅代表采集时的状态。

```text
Mon Oct 26 16:01:52 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200 NVL                On  |   00000000:98:00.0 Off |                    0 |
| N/A   37C    P0             70W /  600W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H200 NVL                On  |   00000000:B8:00.0 Off |                    0 |
| N/A   42C    P0             97W /  600W |   68935MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    1   N/A  N/A         <redacted: other user's process> C 34504MiB |
|    1   N/A  N/A         <redacted: other user's process> C 34324MiB |
+-----------------------------------------------------------------------------------------+
```

### 初始化状态

建立八个功能子包，将策略单测目录放在 `agentkv/tests/`。
vLLM 与 AgentKV 使用同一个虚拟环境，均以可编辑模式安装。
当前阶段仅初始化骨架，尚未运行模型或开展性能实验。

## 后续实验记录约定

每次实验结果都必须引用环境编号，并记录两个仓库运行时的实际 commit
和工作区状态。环境或依赖发生变化时，追加新的环境记录和编号，保留旧记录。
若运行时有未提交修改，保存对应 diff 并在结果记录中注明路径。

每次实验使用以下模板，填写实际值：

```text
实验编号 / 名称：
运行时间（含时区）：
环境编号：ENV-001（环境变化后使用新编号）
AgentKV commit / 工作区状态 / diff 路径：
vLLM commit / 工作区状态 / diff 路径：
GPU index / UUID / CUDA_VISIBLE_DEVICES：
运行前 nvidia-smi 输出文件：
Python 解释器：
完整运行命令 / 环境变量：
实验配置 / 模型 revision / trace 版本 / 随机种子：
结果文件 / 运行日志路径：
指标与结论：
```


## 2026-11-01：五步基线复核 BASELINE-001

用途：环境正常与后续读代码的参照，不作为策略性能实验数据。运行时间：2026-11-01T10:55:49.028318+08:00 至 2026-11-01T11:01:55.061741+08:00（Asia/Shanghai）。先完成 smoke/官方 benchmark，再做针对性的调用链与指标口径分析。

### 环境与归档

- 环境编号：`ENV-001` 复核；Python/包版本与此前环境一致。解释器：`../vllm/.venv/bin/python`。
- vLLM commit：`5df63a7203cdb9328d824a27b682e77c7b9928be`；没有 tracked source 修改，工作区只有未跟踪的 `benchmarks/results/` 结果。空的源码 diff 已归档。
- AgentKV：0.1.0，Git 仍无 HEAD/提交；14 个初始化文件未跟踪。`agentkv-untracked-before.patch` 保存追加本日志前的完整文本快照，`agentkv-untracked-manifest.json` 保存文件清单；空的 tracked diff 不代表这些文件已提交。
- GPU：物理 GPU 1，UUID `GPU-667cc66c-113c-1591-1ec1-f4762d255186`，H200 NVL；`CUDA_VISIBLE_DEVICES=1`，TP=1，BF16。GPU 上另有其他用户进程，运行前后快照在 `environment.json`。
- 模型目录：`../models/Qwen3-8B`；revision `b968826d9c46dd6066d109eabc6255188de91218`，与本地下载元数据核对一致。
- 原始证据目录：`../vllm/benchmarks/results/five-step-baseline-20261101-105549`。
- 可直接打开：[benchmark 原始结果](EVIDENCE.md#external-inputs)、[指标目录](EVIDENCE.md#external-inputs)、[前缀复用原始计数](EVIDENCE.md#external-inputs)。
- 完整命令、依赖版本、两个仓库状态：`environment.json`；nvidia-smi：`nvidia-smi-before.txt`。

### 1. CPU 卸载服务与缓存容量

复用 2026-10-27 启动且本次仍在运行的服务，API PID=69612、EngineCore PID=87261。本次没有重启服务或修改默认行为。原始启动段完整保存为 `server-startup.log`，启动脚本保存为 `server-start.sh`：

```bash
CUDA_VISIBLE_DEVICES=1 ../vllm/.venv/bin/vllm serve ../models/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.3 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":32000000000}}' \
  --port 8000
```

启动日志证据：`Creating v1 connector with name: OffloadingConnector`、`Creating offloading spec with name: CPUOffloadingSpec`、`Created mmap file ... (32.00 GB)`。服务在 `0.0.0.0:8000` 监听。

| 项目 | 值 | 证据/口径 |
| --- | ---: | --- |
| 权重显存 | 15.27 GiB | 启动日志 |
| GPU KV 池显存 | 23.58 GiB | 启动日志 |
| GPU KV blocks | 10732 | `/metrics` 的 cache_config_info.num_gpu_blocks |
| 每个 GPU block tokens | 16 | cache_config_info.block_size |
| GPU KV 容量 | 171712 tokens | 启动日志，与 blocks × block_size 一致 |
| CPU 预算 | 32000000000 bytes | 启动配置，32 GB 十进制 |
| CPU chunks | 13563 | 本配置 blocks_per_chunk=1，每 chunk 对应一个 GPU block |
| CPU 每 chunk 字节 | 2359296 | BF16 KV 布局与页对齐公式 |
| CPU 实际分配 | 31999131648 bytes | EngineCore `/proc/87261/maps` 的 offload 映射长度核对 |
| CPU 容量 | 217008 tokens | chunks × 16 |

启动日志只直接报告了 32.00 GB 和 36 个 CPU tensors，**没有直接打印 CPU chunk 数**。13563 的核对方法是：每 token KV bytes = 36 layers × K/V 两份 × 8 KV heads × 128 head_dim × 2 bytes = 147456；每 16-token chunk 为 2359296 bytes，已经满足 4096 页对齐；floor(32000000000 / 2359296) = 13563。其乘积 31999131648 与进程运行时映射完全一致。36 是 tensor/layer 数，不能写成 CPU 块数。`num_cpu_blocks=None` 也不代表没有 connector 的 CPU 缓存。原始计算、模型 config 与内存映射在 `cache-capacity.json`。

### 2. 精确 completions 请求

已向 `/v1/completions` 发送用户给出的请求：

```json
{"model":"Qwen/Qwen3-8B","prompt":"Hello, my name is","max_tokens":32}
```

HTTP 200；prompt 5 tokens、生成 32 tokens。返回可读文本，以 ` Xiaoyi, and I'm currently a student at the University of Toronto.` 开始。请求与完整响应保存在 `smoke-request.json`、`smoke-response.json`。

### 3. 本轮官方 benchmark

核心负载：random，输入 2048 tokens、输出 128 tokens、200 prompts。仅额外指定本地 tokenizer、输出路径及关闭进度条；没有修改采样、请求速率、并发或长度设置。完整执行命令在 `benchmark-command.sh`。

退出码 0，成功 200，失败 0，时长 15.36 s。实际总输入 409600、总输出 25600 tokens，与 200×2048/128 一致。默认 request_rate=inf、未限制最大并发，实测峰值并发 200；seed=0、无额外 warmups、采样沿用服务默认值。服务已有预热和历史缓存，本轮没有清空缓存。

| 指标 | 本轮 2026-11-01 | 此前 2026-10-27 | 本轮变化 |
| --- | ---: | ---: | ---: |
| 请求吞吐 | 13.02 req/s | 12.86 req/s | +1.22% |
| 输出 token 吞吐 | 1666.18 token/s | 1646.16 token/s | +1.22% |
| 输入+输出总吞吐 | 28325.08 token/s | 27984.73 token/s | +1.22% |
| 平均 TTFT | 6974.95 ms | 7172.02 ms | -2.75% |
| P99 TTFT | 14138.26 ms | 14314.35 ms | -1.23% |
| 平均 TPOT | 36.93 ms/token | 36.84 ms/token | +0.25% |
| P99 TPOT | 51.75 ms/token | 51.63 ms/token | +0.23% |

结果与此前基线接近，全部请求成功；这是单轮环境参照，不能据此把未来任何一次波动都判成环境故障。TTFT 包括 200 请求突发时的排队时间，TPOT 排除首 token。比较时保留负载、速率、并发、采样、缓存状态与共享 GPU 负载。完整结果 `benchmark.json`、输出 `benchmark.log`。

### 4. 指标目录与重复长 prompt

本次重新采集并整理 **47 个导出声明、57 个不同数值样本名字**，包含 HELP、TYPE、完整名字、中文含义、单位、标签与当前数值。完整目录在本记录末尾，并在 `metrics-catalog.md`/`.log` 与 `metrics-catalog.json` 单独保存。用户给定 grep 的等价结果是 `metrics-grep.log`；完整相关族是 `metrics-related.prom`。

连续发送同一份带唯一标记的长 prompt 两次，第二次请求体与第一次完全相同；temperature=0、max_tokens=16、关闭 thinking。没有清空现有缓存；唯一标记避免误命中旧实验。对比 before/after 快照时等待已结束请求的 prefill histogram count 更新，避免读到旧计数。

| 请求 | HTTP | 输入 tokens | 输出 tokens | 回答 | 客户端总耗时 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 200 | 4560 | 3 | `15` | 177.55 ms |
| 2 | 200 | 4560 | 3 | `15` | 44.98 ms |

| 指标 | 本轮请求前累计值 | 第一次增量 | 第二次增量 | 第二次后累计值 |
| --- | ---: | ---: | ---: | ---: |
| `vllm:prefix_cache_queries_total` | 830099 | +4560 | +4560 | 839219 |
| `vllm:prefix_cache_hits_total` | 4560 | +0 | +4544 | 9104 |
| `vllm:external_prefix_cache_queries_total` | 825539 | +4560 | +16 | 830115 |
| `vllm:external_prefix_cache_hits_total` | 0 | +0 | +16 | 16 |
| `vllm:prompt_tokens_cached_by_source_total[source=device]` | 4560 | +0 | +4544 | 9104 |
| `vllm:prompt_tokens_cached_by_source_total[source=host]` | 0 | +0 | +16 | 16 |
| `vllm:prompt_tokens_cached_total` | 4560 | +0 | +4560 | 9120 |
| `vllm:request_prefill_kv_computed_tokens_sum` | 825539 | +4560 | +0 | 830099 |

第二次 GPU 命中 4544 tokens，CPU 命中 16 tokens，两者合计 4560 tokens。稍后抓取完成传输计数：CPU→GPU 的 `kv_offload_load_bytes_total` 在本轮两请求窗口内增加 2359296 bytes，正好是一个 16-token KV chunk；GPU→CPU store 增加 672399360 bytes。说明本轮不仅有 GPU 复用，还有 CPU 回载。卸载异步上报，因此 store 字节只归属整个观察窗口，不强行拆成每个相邻请求的增量。

本次 prompt 长度恰为 16 的倍数。本地 GPU 查询会保留末 token 以重算 logits，按整块截断后报告 4544；connector 补上剩余 16 tokens，属于尾块边界案例，不能当作 GPU 大规模淘汰压力实验。现有“新计算 KV tokens”直方图第二次增加 0 是**统计口径**，不等于所有物理前向计算都为 0；完整回载后 `_update_waiting_for_remote_kv` 仍回退末 token 重算 logits。

原始输入 `long-prompt.txt`、请求 `prefix-request.json`、响应 `prefix-response-1.json`/`prefix-response-2.json`、三次快照 `metrics-before.prom`/`metrics-after-1.prom`/`metrics-after-2.prom`、完成传输快照 `metrics-settled.prom`、解析结果 `prefix-experiment.json` 均已保存。最终 `/health` HTTP 200。

### 5. 对照 PLAN.md 的完整 M2：已有数据、缺口与落盘

依据：[完整项目计划](PLAN.md#实际时间线)。用户已提供 M2 原文并将完整计划放到该路径；本轮计划快照 `PLAN-at-review.md` 与 SHA256 已归档。这一节替代只按 GPU/CPU/重算/PCIe 四项做的初步判定。当前任务完成五步基线与缺口定位，**不宣称整个 M2 已实现**。

#### M2 的 token 口径

M2 要求互斥的 prompt 来源分解：

```text
prompt_tokens = gpu_hit_tokens + cpu_hit_tokens + recompute_tokens
recompute_tokens = prompt_tokens - accepted_gpu_hit_tokens - accepted_cpu_hit_tokens
```

这里的 recompute 是本轮未被两层缓存满足、需要本地 prefill 的逻辑 prompt token（也包括首次冷计算），用作三项相加的校验。它与“所有实际执行过的前向重放 token”不同；末 token logits replay、抢占后重复 prefill、失败回载重试应作为额外诊断量记录，不能直接叠进上述剩余项，否则会与 cache hit 重复并破坏等式。要回答 E0 中“工具间隙驱逐占重算的大头”，还需要结合 M1 的 session/turn/前缀历史与 CPU 淘汰事件区分新增长度、冷启动和淘汰后的缓存丢失，现有总计数不能独自完成这个归因。

#### 完整覆盖表

| M2 要求 | 当前已有 | 仍需采集/埋点的位置 |
| --- | --- | --- |
| 每轮 TTFT | `vllm:time_to_first_token_seconds` histogram；内部 RequestStateStats；benchmark 可保存逐请求 ttfts | Prometheus 是聚合量，并发时不能从相邻快照归因给某个 request。回放器用 streaming 首 token 时间或按 request_id 导出内部事件；benchmark 以后加 `--save-detailed`。 |
| 每轮 TPOT | `vllm:request_time_per_output_token_seconds` histogram，单位秒/token，排除首 token；内部 FinishedRequestStats.mean_time_per_output_token | 同样需要逐请求记录并 join session/turn。`inter_token_latency_seconds` 的逐 token 样本不能混当每个请求的 TPOT。输出少于 2 tokens 时应明确不可估计/无观测口径。 |
| 每会话 JCT | 没有会话级指标；`e2e_request_latency_seconds` 只是一个请求，不是 JCT | M1 回放器维护 session_id、session_start、last_turn_end；采集约定为会话到达至最后一轮完成，包含会话内工具间隙与请求等待。不将当前独立请求假装为有真实工具间隙的会话。 |
| 每轮 GPU 命中 | `prefix_cache_hits_total`、`prompt_tokens_cached_by_source_total{source="device"}`；内部 PrefillStats.num_local_cached_tokens | 全局已有，生产并发场景应按 request_id 记录准入时实际接受的命中量；不要重复统计未成功准入的 lookup。 |
| 每轮 CPU 命中 | `prompt_tokens_cached_by_source_total{source="host"}`；CPU-only 时 external_prefix_cache_hits_total；内部 PrefillStats.num_external_cached_tokens | 在 `EngineCoreOutput.prefill_stats` 被消费的位置导出逐请求字段；CPU-only 可直接对应 host，多层时需 connector 的来源细分，不能把所有 external 命中当 CPU。 |
| 每轮重算及三项校验 | `request_prefill_kv_computed_tokens_sum/count` 和 `prompt_tokens_by_source_total{source="local_compute"}` 可覆盖逻辑未缓存量；内部 PrefillStats 有计算与两类缓存分解 | **不需要重复创建同义聚合总计数器**。导出 per-request 的三个来源和 prompt_tokens 后断言恒等式；补 session/turn/prefix 历史与淘汰原因归因。所有实际额外重放另记诊断计数。 |
| PCIe 字节数 | `kv_offload_store_bytes_total`、`kv_offload_load_bytes_total` 是完成的 GPU↔CPU KV payload 字节；旧 transfer_type 指标为兼容副本 | KV 传输总 payload 已有；每请求归因需 job_id 映射，真实 PCIe 线速/协议开销/其他 CUDA 拷贝需硬件 profiler 或链路计数。不能把 payload 叫作总线全部实测。 |
| 调度器单步耗时 | 当前未导出 `Scheduler.schedule()` 执行时长；请求 queue_time、lookup_sync_delay、MFU calc_duration 都不是这个量 | 在 `Scheduler.schedule()` 入口/出口用 perf_counter_ns 计时，统一覆盖 `engine/core.py:667` 与 `:725` 两处调用；记录 step_id、调度 tokens/requests。必要时另列 update_from_output 耗时，不能混入 GPU 执行时间。 |
| 策略函数自身耗时 | 当前 LRU/ARC 的 get/touch/evict 等没有独立耗时指标；lookup_sync_delay 包含其他逻辑 | 在自定义 CachePolicy 回调外围按 operation 单独计时；当前 `cpu/manager.py:324` 是 evict 调用点，接口 `cpu/policies/base.py:116`。不要把整个 prepare_store 的锁页/分配开销当成策略算法耗时。 |
| 统一 parquet 与分析目录 | 本轮已有数据已写出 5 个 parquet 文件并读回验证 | 完整 M2 的 session 与 timing 事件还需接入回放器/策略；画图脚本按计划放 `./agentkv/analysis/`，不放在采集脚本中。本轮仅环境验证，不生成优化收益图。 |

#### 本轮可恢复的 per-request 数据与恒等式

两次请求的 TTFT、TPOT、e2e 直方图各自 count 增量均严格等于 1，所以这次隔离串行实验可用 sum 的差恢复每请求服务端数据；多请求并发时不能沿用这种归因方法。TTFT/TPOT 与此前表中的客户端总耗时分列保存，没有混用。

| 请求 | prompt | GPU 命中 | CPU 命中 | M2 重算 | 服务端 TTFT(ms) | 服务端 TPOT(ms/token) | 三项和校验 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4560 | 0 | 0 | 4560 | 153.25 | 10.24 | 通过 |
| 2 | 4560 | 4544 | 16 | 0 | 29.51 | 5.78 | 通过 |

本轮没有会话 trace，所以 session_id、turn_id、session_jct_ms 保持 null；没有调度/策略计时器，相关字段也保持 null，没有用 0 冒充实测。之前说“严格实际重算仍有缺口”指的是额外物理前向诊断；按现在明确的 M2 三项和口径，逻辑剩余量已有内部字段与聚合指标，缺的是逐请求稳定导出及会话归因。

#### parquet 文件与复现

- `request_metrics.parquet`：2 行，真实 request_id、prompt/output、GPU/CPU/重算 tokens、恒等式、服务端 TTFT/TPOT/e2e、客户端 e2e；session/JCT 为 null。
- `benchmark_summary.parquet`：1 行，官方 benchmark 吞吐与 TTFT/TPOT 等摘要。
- `benchmark_requests.parquet`：200 行，原始逐请求 latency/queue_time；本轮 benchmark 没有加 --save-detailed，因此该文件逐请求 TTFT/TPOT 为 null，不能由均值反推。下次只需加保存详细结果选项，不需改变负载。
- `system_windows.parquet`：1 行，两次长 prompt 的观察窗口内 GPU→CPU 672399360 bytes、CPU→GPU 2359296 bytes；payload 总计 674758656 bytes。异步 store 不强行拆到单个相邻请求，硬件 PCIe 字节与未采集计时为 null。
- `prometheus_samples.parquet`：2388 行，四次原始快照的全部 samples，含 metric 名字、标签 JSON、value 与 snapshot 标识。
- `parquet-manifest.json`：字段、行数、文件大小、缺失字段说明与转换器版本；全部 parquet 已读回并校验行数，token 恒等式两行均通过。

parquet 转换器用 uv 安装到 `../agentkv-runs/agentkv-parquet-tools`（pyarrow 26.0.0），通过现有 `.venv/bin/python` 临时加载。没有把 pyarrow 装进服务的 `.venv`，没有改变正在运行的 ENV-001。

#### 问题 A：CPU 卸载在哪里触发

1. `Scheduler.schedule()`（`vllm/v1/core/sched/scheduler.py:591`）生成本步调度结果，随后构造 connector metadata（同文件 `:1561`）。
2. `OffloadingConnectorScheduler.build_connector_meta()`（`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:1846`）调用 `_build_store_jobs()`（`:1570`）。后者遍历本步调度和已完成请求，按本步计算后可保存的完整 chunk 构建 key，调用 `manager.prepare_store()`（`:1732`）挑选 CPU 存储位置和新存储项。`_calc_num_offloadable_tokens()`（`:663`）受默认 `offload_prompt_only=true` 限制。
3. `OffloadingConnector.wait_for_save()`（`vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:134`）调用 worker 的 `prepare_store_kv()`（`offloading/worker.py:259`）排队 store job；默认推迟到下一 engine step 的 `start_kv_transfers()`（`:246`）再 `submit_store()`（`vllm/v1/kv_offload/cpu/gpu_worker.py:854`）。
4. `SingleDirectionOffloadingHandler.transfer_async()`（同文件 `:534`）在传输流上等待计算流写完 KV，然后提交异步批量拷贝。`get_finished()`（`:706`）在 CUDA event 完成后返回真实已完成的 payload bytes/time；connector worker（`offloading/worker.py:273`）记录统计并传回 scheduler。

因此默认 CPU offload 是“随着完整 prompt KV chunk 产生而尽力写出”，**不是等 GPU KV 池满了或淘汰 GPU block 时才触发**；store_threshold=0，重复已有 CPU key 会去重，分配/驱逐由 CPU manager 管理。排队、提交、完成是三个不同时间点。

#### 问题 B：缺失指标应该在哪里统计

M2 的逻辑三来源应优先从 `EngineCoreOutput.prefill_stats`（`vllm/v1/engine/__init__.py:221`）及 `output_processor.py:734` 的消费路径按 request_id 导出；下面的“额外实际重放”用于解释物理执行开销，不替代来源恒等式。

| 目标 | 首选位置 | 记录内容与去重原则 |
| --- | --- | --- |
| 按请求的 GPU 命中 | `kv_cache_manager.py:253`/`:264` 与 scheduler 准入统计 | request_id、稳定 trace/turn id、prefix/block hash、实际接受的命中 tokens；避免把未成功准入的反复 lookup 都算作命中。 |
| CPU 查询/命中及其来源 | `cpu/manager.py:189` lookup、`:204` prepare_load；`offloading/scheduler.py:1099` get_external_cache_hit_sources、`:1146` update_state_after_alloc | MISS/HIT_PENDING/HIT、CPU chunk/key、接受的命中 token 区间、load job id；区别查询命中与真正提交/完成回载。 |
| 额外实际重放诊断（非 M2 互斥来源剩余项） | `Scheduler.schedule()` 和 `update_from_output()`（`:2026`）；完整回载回退处 `:3123`、KV load 失败重算处 `:3237` | 每次实际执行的 prefill token 区间，对照同一前缀已完成的历史；区分 cold_prefill/cache_miss_recompute/preemption_recompute/logits_replay/load_failure_recompute，排除普通 decode。累计前在执行完成处确认，避免把仅调度但未执行的工作计入。 |
| 按请求归因的传输 bytes | `OffloadingConnectorWorker.start_kv_transfers()`/`get_finished()`；`gpu_worker.py:534`/`:706` | 建立 job_id→request/prefix 的归属映射，在完成处记录 direction、bytes、duration 和结果；共享或合并 job 的分摊口径必须显式定义，不能把同一 job 对每个请求重复全额计算。 |

request/trace/prefix 身份应进入结构化日志或离线事件文件；Prometheus 保持有限的 source/reason 等标签，避免把每个 request_id/hash 变成时间序列。现有聚合计数器能复用就直接复用。本轮只定位和记录目标，没有修改 vLLM/AgentKV 的执行代码或新增埋点；已将能从本轮原始证据得到的数据转成 parquet。

### 当前指标完整目录（本轮实采）

| 导出名字 | 类型 | 含义 |
| --- | --- | --- |
| `vllm:kv_offload_total_bytes_total` | counter | 兼容旧版的累计传输字节数；transfer_type=GPU_to_CPU 为卸载，CPU_to_GPU 为回载。与新的 store/load bytes 重复，不能相加。 |
| `vllm:kv_offload_total_bytes_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_total_time_total` | counter | 兼容旧版的累计传输时间，单位秒，按 transfer_type 区分方向；与新的 store/load time 重复。 |
| `vllm:kv_offload_total_time_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_size_bucket`、`vllm:kv_offload_size_count`、`vllm:kv_offload_size_sum` | histogram | 兼容旧版的单次传输字节数分布，按 transfer_type 区分方向；与新的 store/load size 重复。 |
| `vllm:kv_offload_size_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_cpu_cache_usage_perc` | gauge | CPU KV 缓存中被活跃传输固定、暂不可回收的容量比例，0～1；不是缓存总驻留率。空闲时为 0 不表示 CPU 中没有 KV 数据。 |
| `vllm:kv_offload_cpu_cache_write_usage_perc` | gauge | CPU KV 缓存中被尚未完成的卸载写入占用的容量比例，0～1。 |
| `vllm:kv_offload_cpu_cache_read_usage_perc` | gauge | CPU KV 缓存中被正在回载的读取占用的容量比例，0～1。 |
| `vllm:kv_offload_cpu_allocation_size_bucket`、`vllm:kv_offload_cpu_allocation_size_count`、`vllm:kv_offload_cpu_allocation_size_sum` | histogram | 每次 prepare_store 请求分配的 CPU chunk 数量分布，单位 chunk，不是字节。 |
| `vllm:kv_offload_cpu_allocation_size_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_load_bytes_total` | counter | 从卸载存储回载到 GPU 的累计字节数；当前 CPUOffloadingSpec 下即 CPU→GPU。 |
| `vllm:kv_offload_load_bytes_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_load_time_total` | counter | 从卸载存储回载到 GPU 的累计传输时间，单位秒；不是请求端到端耗时。 |
| `vllm:kv_offload_load_time_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_load_size_bucket`、`vllm:kv_offload_load_size_count`、`vllm:kv_offload_load_size_sum` | histogram | 单次回载操作的传输字节数分布。 |
| `vllm:kv_offload_load_size_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_store_bytes_total` | counter | 从 GPU 写入卸载存储的累计字节数；当前配置下即 GPU→CPU。 |
| `vllm:kv_offload_store_bytes_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_store_time_total` | counter | 从 GPU 写入卸载存储的累计传输时间，单位秒。 |
| `vllm:kv_offload_store_time_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_store_size_bucket`、`vllm:kv_offload_store_size_count`、`vllm:kv_offload_store_size_sum` | histogram | 单次卸载操作的传输字节数分布。 |
| `vllm:kv_offload_store_size_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_lookup_sync_delay_seconds_bucket`、`vllm:kv_offload_lookup_sync_delay_seconds_count`、`vllm:kv_offload_lookup_sync_delay_seconds_sum` | histogram | 一次同步卸载缓存查询调用的耗时分布，单位秒。 |
| `vllm:kv_offload_lookup_sync_delay_seconds_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:kv_offload_lookup_async_delay_seconds`（仅声明） | histogram | 从查询首次被延后到后续查询得出结果或请求结束的等待时间分布，单位秒。本次只有 HELP/TYPE 声明，没有数值样本；不能将缺失解释为已测得 0。 |
| `vllm:kv_offload_allocation_failure_total`（仅声明） | counter | 卸载写入分配失败的累计次数。本次只有 HELP/TYPE 声明，没有数值样本；不能将缺失解释为已测得 0。 |
| `vllm:num_requests_kv_fetch_by_stage` | gauge | 按异步 KV 回载阶段划分的等待请求数：waiting_to_start=等待开始，in_progress=已开始未完成，completed_waiting=已完成尚未运行。 |
| `vllm:engine_sleep_state` | gauge | 引擎睡眠状态；sleep_state=awake 的值 1 表示唤醒，weights_offloaded 的值 1 表示一级睡眠卸载权重，discard_all 的值 1 表示二级睡眠。权重睡眠卸载与本次 KV CPU 卸载不同。 |
| `vllm:kv_cache_usage_perc` | gauge | GPU KV block 的使用比例，0～1。源码为 1−空闲 block 数/可用 block 总数（排除 null block）；可回收的缓存块可以位于空闲队列，空闲时为 0 不等于前缀缓存为空。 |
| `vllm:prefix_cache_queries_total` | counter | GPU 本地前缀缓存累计查询的 token 数，不是请求次数。 |
| `vllm:prefix_cache_queries_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:prefix_cache_hits_total` | counter | GPU 本地前缀缓存累计命中的 token 数，不是命中请求次数。 |
| `vllm:prefix_cache_hits_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:external_prefix_cache_queries_total` | counter | 通过 KV connector 查询的剩余 token 数，通常是 GPU 本地命中之后仍未命中的部分；当前 connector 可查询 CPU 卸载缓存。HELP 中 cross-instance 的描述不意味着本配置必须跨实例。 |
| `vllm:external_prefix_cache_queries_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:external_prefix_cache_hits_total` | counter | 通过 KV connector 获得的缓存命中 token 数；当前 CPU-only 配置下对应 CPU 缓存命中。 |
| `vllm:external_prefix_cache_hits_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:prompt_tokens_by_source_total` | counter | 按来源累计 prompt tokens；source=local_compute/local_cache_hit/external_kv_transfer。local_compute 来自 PrefillStats 的首次准入分解，不是逐步累计所有重复前向工作量。 |
| `vllm:prompt_tokens_by_source_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:prompt_tokens_cached_total` | counter | 累计使用的缓存 prompt token 数，包含本地及外部命中；在迭代统计时更新，与准入统计的刷新时刻可能不同。 |
| `vllm:prompt_tokens_cached_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:prompt_tokens_cached_by_source_total` | counter | 请求准入时按缓存来源统计命中 token 数：device=本地显存，host=主机 DRAM，disk=磁盘，p2p=其他实例，external_unspecified=未细分外部来源。各来源之和对应本地和外部前缀命中之和。 |
| `vllm:prompt_tokens_cached_by_source_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:request_prefill_kv_computed_tokens_bucket`、`vllm:request_prefill_kv_computed_tokens_count`、`vllm:request_prefill_kv_computed_tokens_sum` | histogram | 每个结束请求的 prompt_tokens − num_cached_tokens 的分布，单位 token。它不是所有实际前向计算的完整计数：不区分首次计算和历史前缀丢失后的重算，也不覆盖完整缓存命中后的末 token 重放及抢占后重复 prefill 的全部工作量。 |
| `vllm:request_prefill_kv_computed_tokens_created` | gauge | 对应指标族的创建时刻，Unix 秒；不是传输耗时、缓存容量或命中数量。 |
| `vllm:cache_config_info` | gauge | 数值固定为 1，标签携带缓存配置；本服务 enable_prefix_caching=True、block_size=16、num_gpu_blocks=10732、kv_cache_size_tokens=171712、gpu_memory_utilization=0.3。kv_offloading_size=None 不能用于判定 connector 卸载关闭；本服务通过 kv_transfer_config 设置 32 GB。 |

本轮共有 47 个导出声明（包含 created），57 个不同数值样本名字。完整标签与数值见 metrics-catalog.json。

当前未启用 `--kv-cache-metrics`，可选的 kv_block_lifetime_seconds、kv_block_idle_before_evict_seconds、kv_block_reuse_gap_seconds 没有导出；当前 store_threshold=0，stores_skipped 指标未启用。

### M2 timing 指标补充

| 导出名字 | 类型 | 含义 |
| --- | --- | --- |
| `vllm:time_to_first_token_seconds_bucket`、`vllm:time_to_first_token_seconds_count`、`vllm:time_to_first_token_seconds_sum` | histogram | 请求到首 token 的时间，单位秒 |
| `vllm:time_to_first_token_seconds_created` | gauge | 创建时刻，Unix 秒 |
| `vllm:request_time_per_output_token_seconds_bucket`、`vllm:request_time_per_output_token_seconds_count`、`vllm:request_time_per_output_token_seconds_sum` | histogram | 每请求平均输出 token 时间，排除首 token，单位秒/token |
| `vllm:request_time_per_output_token_seconds_created` | gauge | 创建时刻，Unix 秒 |
| `vllm:e2e_request_latency_seconds_bucket`、`vllm:e2e_request_latency_seconds_count`、`vllm:e2e_request_latency_seconds_sum` | histogram | 单请求端到端时延，单位秒，不是会话 JCT |
| `vllm:e2e_request_latency_seconds_created` | gauge | 创建时刻，Unix 秒 |

依据用户新提供 PLAN.md 完成上述 M2 修订的时间：2026-11-01T11:08:54.061334+08:00。


## 2026-11-01：CPU 对齐尾块是否真正复用的运行时核实

时间：2026-11-01T11:31:38.176560+08:00。结论：**整块 16 tokens 的 KV 从 CPU 搬回后，前 15 tokens 的 KV 进入 attention 上下文复用，最后 1 token 仍执行前向以取得 logits，并覆盖刚加载的末槽 KV。不是整块搬回后全部重算。**

### 方法与范围

在 GPU 0 启动独立、短生命周期的 in-process LLM，使用相同 Qwen3-8B、BF16、0.3 GPU memory utilization、32 GB CPUOffloadingConnector 和 max_num_batched_tokens=8192；CUDA graphs 保持启用。通过脚本内临时包装 scheduler/copy-handler/runner 方法记录实际执行，不修改 vLLM 文件，不重启 GPU 1 的现有服务。观测时 GPU→CPU 读取会同步，因此仅用于正确性核实，不作为性能实验。使用上一轮保存的完整 chat prompt 的 4560 个 token IDs，冷请求和重复请求均生成 `15`。

第一次诊断因 tokenizer 返回 BatchEncoding 而在输入断言处停止，尚未发送请求；错误日志保留。修正为先提取并核验 input_ids 后，完整诊断退出码 0、全部断言通过。原始服务最终健康检查 HTTP 200，诊断进程退出且 GPU 资源已释放。

### 真实事件链

```text
CPU chunk 284 -> GPU physical block 286
load completed: 2359296 bytes
36 层整页 CPU/GPU bytes 全部一致
remote resume: num_computed_tokens 4560 -> 4559
next schedule: 1 token
forward position: [4559]
attention seq_len: [4560]
block_table[logical block 284] = 286
write slot: 4591 = 286 * 16 + 15
36 层中尾块前 15 个位置的 KV 与 CPU 数据保持一致
后续生成 token 的位置为 4560、4561
```

以上位置采用从 0 开始的编号。CPU 尾块覆盖位置 4544～4559：

| token 范围 | 运行时行为 |
| --- | --- |
| 0～4543 | GPU 本地命中的前 284 块 |
| 4544～4558 | CPU 回载块的前 15 个 KV 位置，在位置 4559 的 attention 中作为已有上下文 |
| 4559 | 回载后将 computed frontier 回退到此位置，单 token 前向重新计算 Q/K/V 与后续层，产生 logits；写回 offset 15 |

`prepare_attn` 的 block table 确认了 attention 末块正是 CPU 回载的 physical block 286，而不是另一块；缓存页逐字节比较覆盖全部 36 层。FlashAttention 用这个 block table 与 seq_len=4560 读取上下文，`do_kv_cache_update` 用 slot_mapping 写当前 token，所以前 15 个位置被保留使用，加载的最后一个位置被当前前向的 KV 覆盖。这里验证的是逻辑拷贝 payload 与真实前向输入/映射，不是硬件 PCIe transaction profiling。

### 对现有指标和 M2 的修正

| 口径 | GPU | CPU | 本地计算 | 三项和 |
| --- | ---: | ---: | ---: | ---: |
| 现有准入/名义来源计数 | 4544 | 16 | 0 | 4560 |
| 此次实测有效复用/实际 prompt 前向 | 4544 | 15 | 1 | 4560 |

**两行都满足三项和等于 prompt 长度，所以恒等式本身不能证明来源口径正确。** 先前 `request_metrics.parquet` 的 cpu_hit_tokens/recompute_tokens 来自准入计数，应视为名义来源，不能解释为“实际复用了 CPU 的全部 16 tokens、没有实际计算”。`request_prefill_kv_computed_tokens` 使用 prompt_tokens-num_cached_tokens，因此此处仍报告 0；它没有扣除随后为 logits 回退的这个 token。

若 M2 要按“实际有效复用/实际 prefill 执行来源”统计，需从准入命中区间中扣除随后 replay 的重叠位置，并把它归到重算。建议保留 nominal_gpu_hit_tokens、nominal_cpu_hit_tokens、reported_recompute_tokens，同时增加 effective_gpu_reuse_tokens、effective_cpu_reuse_tokens、actual_prompt_forward_tokens、replay_reason，避免覆写原始 Prometheus 数据。对多次抢占/回放，再另记额外执行工作量，不能让同一 prompt 位置在来源分解中重复计数。

本配置每 token KV payload=147456 bytes，因此这次传输 2359296 bytes 中，2211840 bytes 对应有效复用的前 15 tokens；末 token 对应 147456 bytes 在随后前向中重新写入。该比例只描述本次尾块边界案例，不代表整体 CPU 卸载流量的浪费比例。

### 代码依据与证据

- 回退 1 token：[scheduler.py](patches/README.md#file-vllm-v1-core-sched-scheduler-py)。
- 计算实际调度 token 数：[scheduler.py](patches/README.md#file-vllm-v1-core-sched-scheduler-py)。
- positions 与 seq_lens：[input_batch.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/input_batch.py#L374)。
- block table/slot mapping：[model_runner.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py#L1556)。
- 当前 KV 写入 slot：[flash_attn.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/flash_attn.py#L1439)。
- attention 读取 cached K/V 与 block table：[flash_attn.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/flash_attn.py#L1275)。
- 指标使用 prompt-num_cached：[loggers.py](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/metrics/loggers.py#L1312)。
- 诊断目录：`../vllm/benchmarks/results/cpu-tail-block-probe-20261101-112321`。
- [运行时事件](EVIDENCE.md#external-inputs)、[断言结果](EVIDENCE.md#external-inputs)、[可复现诊断脚本](EVIDENCE.md#external-inputs)、[完整日志](EVIDENCE.md#external-inputs)。
- `token-accounting.parquet` 与 `probe-events.parquet` 保存此次核实的数据，均已读回验证；原先基线 parquet 保持原始值。


## 2026-11-01：固定速率 4 req/s 的环境延迟参照

运行时间：2026-11-01T11:44:37.316207+08:00 至 2026-11-01T11:45:40.649079+08:00；环境 `ENV-001`，vLLM commit `5df63a7203cdb9328d824a27b682e77c7b9928be`，GPU 1。继续使用原来的 Qwen3-8B 服务，gpu_memory_utilization=0.3、OffloadingConnector、32 GB CPU KV 缓存；服务与缓存配置未改，未重启/清空缓存。此为单轮环境参照，不作为策略性能实验数据。

### 命令与负载

```bash
../vllm/.venv/bin/vllm bench serve \
  --model Qwen/Qwen3-8B \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate 4 \
  --tokenizer ../models/Qwen3-8B \
  --save-result --save-detailed \
  --result-dir ../vllm/benchmarks/results/qwen3-8b-rate4-baseline-20261101-114437 \
  --result-filename result.json --disable-tqdm
```

仅改变请求到达速率，并开启保存逐请求详情；仍使用 seed=0、原默认采样、无额外 warmups、不限制最大并发。burstiness=1.0 对应 Poisson 到达，**4 req/s 是平均请求速率，不是严格每 250 ms 一个请求，也不是并发数固定为 4**。

### 结果与突发基线对照

本轮退出码 0，200 成功、0 失败，耗时 50.89 s。实际输入 409600、输出 25600 tokens，全部请求都为 2048/128；200 条逐请求详情均已保存，error 均为空。突发对照来自 2026-11-01 10:55 的五步基线。

| 指标 | 突发 request-rate=inf | 平均 request-rate=4 |
| --- | ---: | ---: |
| 请求吞吐 | 13.02 req/s | 3.93 req/s |
| 输出吞吐 | 1666.18 token/s | 503.03 token/s |
| 总 token 吞吐 | 28325.08 token/s | 8551.53 token/s |
| 平均 TTFT | 6974.95 ms | 92.04 ms |
| 中位数 TTFT | 7231.73 ms | 81.91 ms |
| P99 TTFT | 14138.26 ms | 186.11 ms |
| 平均 TPOT | 36.93 ms/token | 7.90 ms/token |
| P99 TPOT | 51.75 ms/token | 11.39 ms/token |
| 峰值并发 | 200.00 requests | 19.00 requests |

平均 TTFT 约下降 75.8 倍。输出吞吐 503.03 token/s 受到请求供给速率限制，不是引擎满载容量退化；饱和吞吐仍使用突发基线参照。有限请求数的到达过程以及末尾排空，使完成吞吐 3.93 req/s 略低于配置的平均到达速率 4。

### 排队与缓存交叉检查

- 服务端 `request_queue_time_seconds_count` 增量 200，sum 增量 0.002894809 s，平均 WAITING 0.01447 ms/请求。
- 服务端 TTFT histogram count 增量 200，平均 85.63 ms；客户端官方 benchmark 平均 TTFT 92.04 ms，两者测量边界不同，不混用。
- GPU 前缀命中增量 0，CPU/external 命中增量 0；local_compute 增量 409600 tokens。因此这轮 TTFT 并非通过缓存复用跳过随机 prompt prefill 得到。
- 本次 WAITING 很轻，适合作为低排队的固定速率延迟参照。它仍有重叠请求和批处理，不能等同于严格串行、独占 GPU 的单请求延迟；GPU 上其他进程状态已归档。
- 最终 `/health` HTTP 200。

### 数据归档

目录：`../vllm/benchmarks/results/qwen3-8b-rate4-baseline-20261101-114437`。

- [原始详细结果](EVIDENCE.md#external-inputs)、[完整 benchmark 输出](EVIDENCE.md#external-inputs)、[复现命令](EVIDENCE.md#external-inputs)。
- `requests.parquet`：200 行，逐请求 TTFT、TPOT、客户端 latency、输入/输出长度；TPOT 按 (latency−TTFT)/(output_tokens−1) 计算，其均值已与官方结果核对。
- `summary.parquet`：本轮摘要，含服务端平均 WAITING；`prometheus_samples.parquet`：运行前后全部原始 samples。
- `analysis.json`：服务端 queue/TTFT/cache delta；`environment.json`：时间、配置、版本、两个仓库状态及前后 GPU 快照。
- `vllm-working-tree.diff` 和 `agentkv-untracked-before.patch` 保存运行时源码/未跟踪项目状态。pyarrow 26.0.0 仅从 `../agentkv-runs/agentkv-parquet-tools` 加载，服务的 `.venv` 未改变。
- `parquet-manifest.json` 记录行数、字段语义与转换方法；parquet 已读回验证。

后续同时保留两个参照：突发运行用于饱和吞吐/高排队表现，4 req/s 运行用于相同负载下的低排队 TTFT/TPOT。比较时固定速率、分布、并发上限、采样和长度，不把两类吞吐直接当作同一性能指标。


## 2026-11-01：NUMA 本地锁页 H2D 带宽与 1 万 token prefill 对照

结论：在 H200 GPU 0 上，NUMA node 2 本地锁页内存的大块连续 H2D 带宽约 **55 GB/s**。1 万 token 的 1.47456 GB KV payload 直接拷贝实测 **26.745 ms**，同卡 Qwen3-8B BF16 无缓存单请求 1 万 token 的模型前向中位数 **292.489 ms**，阶段耗时比约 **10.94×**。这是搬运与计算成本的微基准对照，不是端到端卸载或策略加速结论。

### 环境与计时范围

- 测量结束：2026-11-01T12:04:19.997338+08:00；基于 `ENV-001`，依赖版本未变，vLLM commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。
- 测量配置：物理 GPU 0 / UUID `GPU-48956eb2-4037-09c5-2e94-39989b4d2e0d`，H200 NVL；`numactl --cpunodebind=2 --membind=2`。PCIe 当前速率 `32.0 GT/s PCIe`，宽度 `16`。
- GPU 0 两项测试顺序运行；GPU 1 的原有服务持续运行，最终健康检查 HTTP 200。共享主机的 GPU/NUMA/PCIe 状态保存在 environment.json，未宣称整机独占。
- 锁页分配 `is_pinned=True`，运行时 `/proc/self/numa_maps` 显示所有 524288 个页均为 N2，映射实际为 2 GiB（CUDA host allocator 向上分配），逻辑测试池约 1.47 GB。
- CUPTI 未安装，因此用 CUDA events；分配、随机填充、图捕获、完整逐字节正确性校验、模型加载/编译和预热均不计入样本。
- 每种拷贝大小测 3 轮×20 次，共 60 个 direct 与 60 个 graph 样本；形状和方法交错执行，报告中位数。Graph 每次 replay 使用至少 512 MiB、地址不同的 buffer 范围，避免反复测同一小段；其每 copy 时间是批量 replay 时间除以 copy 次数，不当作单次调用墙钟延迟。

### KV 字节数核算

```text
每 token = 36 layers × K/V 两份 × 8 KV heads × head_dim 128 × BF16 2 bytes
         = 147456 bytes = 144 KiB
单块     = 16 × 147456 = 2359296 bytes = 2.359296 MB = 2.25 MiB
1 万 token = 625 块 = 1474560000 bytes = 1.47456 GB
```

MB/GB 使用十进制，MiB/GiB 使用二进制；有效带宽为单向 payload bytes / elapsed time，不乘以 2。

### 连续 H2D 结果

下面为 direct 单次 copy 的 CUDA-event 时间；墙钟另外包含 Python/运行时提交及同步开销。

| KV 块数 | 大小 MB | CUDA 中位数 ms | 有效 GB/s | 单次墙钟中位数 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.359296 | 0.04970 | 47.47 | 0.06621 |
| 16 | 37.748736 | 0.68613 | 55.02 | 0.70278 |
| 32 | 75.497472 | 1.36536 | 55.29 | 1.38187 |
| 64 | 150.994944 | 2.72395 | 55.43 | 2.74086 |
| 128 | 301.989888 | 5.44178 | 55.49 | 5.45869 |
| 625 | 1474.560000 | 26.74502 | 55.13 | 26.76354 |

单块 graph/摊薄提交开销后的中位数约 0.04506 ms、52.36 GB/s；direct 为 0.04970 ms、47.47 GB/s，单次墙钟约 0.06621 ms。37.75～301.99 MB 的大块带宽约 55.02～55.49 GB/s。完整 1.47456 GB 的 direct P10/P90 为 26.735/26.756 ms，跨 60 个样本波动很小。

按 55 GB/s 估算：

```text
t = 1.47456 GB / 55 GB/s = 0.026810 s = 26.81 ms
```

完整大小直接实测 26.745 ms，单次调用至同步完成墙钟 26.764 ms，与换算相符。若把 625 个单块依次单独提交，不能直接假定仍达到连续大块的最佳带宽；还会增加每次提交开销。

### 无缓存 10000-token prefill 实测

在同一 GPU 0 上另起短生命周期的 in-process LLM：Qwen3-8B、BF16、TP=1、gpu_memory_utilization=0.3、max_num_seqs=1、max_num_batched_tokens=16384；**关闭 prefix caching、无 OffloadingConnector**，SamplingParams.skip_reading_prefix_cache=True，max_tokens=1、ignore_eos=True、temperature=0。固定随机种子 0 的有效 token IDs，不依赖文本重分词长度。

这与在线服务配置有意不同：测的是完整、单步的 uncached prefill，不含 CPU offload 工作；在线服务启用 CPU 卸载且 batch token 上限为 8192，会将 10000-token prompt 分块。两者不可混写成相同配置。测试中每个请求均验证 prompt_len=10000、cached_tokens=0，且只执行一个 scheduled_tokens=10000 的模型步，避免重复输入命中缓存。

模型初始化、编译、autotune 后再预热 3 请求，测 3 轮×5 请求。GPU events 从 runner forward_start 到 kv_connector.finish_forward（本实例为 no-op），排除输入准备、metadata 构建、logits/采样。墙钟从 LLM.generate 调用到一个输出 token 返回，包含调度/输入准备/模型/logits/采样等，不冒充纯 GPU 前向。

| 轮次 | GPU forward 中位数 ms | 首 token 墙钟中位数 ms |
| --- | ---: | ---: |
| 1 | 291.957 | 299.639 |
| 2 | 292.690 | 300.302 |
| 3 | 292.522 | 300.100 |

15 次合并 GPU forward 中位数 **292.489 ms**，min/max 290.583/293.658 ms；首 token 墙钟中位数 **300.100 ms**。

### 对照与适用范围

- 单看此次阶段成本：292.49 ms 计算 / 26.75 ms 连续拷贝 ≈ **10.94 倍**，阶段耗时差约 265.74 ms。
- 这是连续内存 copy，与生产 connector 的跨层/跨块 batch copy 不是同一调用形态；查找、分配、CPU/GPU chunk 布局、排队、并发带宽竞争、以及回载后末 token 重放仍需计入真实回载时延。
- 不能把这个倍数当作会话 JCT 或服务 TTFT 加速比。它给出当前 NUMA 本地链路条件下 1 万 token 的连续数据搬运量级，可作为 M6 提前量估算的初值；完整回载路径需要后续另测。

### 复现与数据

证据目录：`../vllm/benchmarks/results/pinned-h2d-prefill-20261101-115642`。

- 可复用脚本：[benchmark_pinned_h2d.py](patches/README.md#file-benchmarks-kernels-benchmark-pinned-h2d-py)。格式与 ruff 检查通过；本轮执行用的原始快照与格式整理后的最终版本分别归档，整理只涉及格式与标准库导入顺序。
- `bandwidth-command.sh`、`prefill-command.sh` 为完整命令；`prefill_probe.py` 为临时 GPU 前向计时脚本，无 vLLM 引擎文件改动。
- `bandwidth-samples.json`/`bandwidth-summary.json`：720 个样本、12 行汇总；`prefill-samples.json`/`prefill-summary.json`：15 个样本及 3 轮汇总。
- 同名含义的 `bandwidth_samples.parquet`、`bandwidth_summary.parquet`、`prefill_samples.parquet`、`comparison.parquet` 已统一落盘并读回核对行数；schema 在 parquet-manifest.json。
- `bandwidth-environment.json`、`prefill-environment.json`、`environment.json` 保存 NUMA 页分布、CPU affinity、配置、版本及 GPU 快照；`comparison.json` 保存字节公式和换算。
- `benchmark-source.patch` 保存新增且未跟踪的 benchmark 源码，vllm-working-tree.diff 保存 tracked diff；AgentKV 状态在 environment.json。此次没有提交、没有开启 PR。


## 2026-11-01：小 GPU 池的两轮 hello-world 会话与无关请求挤压

结论：**无干扰时第二轮几乎全由 GPU 前缀复用；用无关请求逐出原会话后，第二轮原始 4096-token prompt 全部由 CPU 恢复，没有整段重算。** 默认 offload_prompt_only=true 的生成内容仍需要重新计算。三组重复试验的 token 计数与逐出状态完全一致。

### 配置与范围

- 基础依赖环境 ENV-001，vLLM commit `5df63a7203cdb9328d824a27b682e77c7b9928be`；独立 GPU 0 / NUMA node 2 in-process LLM，原 GPU 1 的 8000 服务保持运行。
- Qwen3-8B、BF16、TP=1、prefix caching 开启。
- GPU KV：`num_gpu_blocks_override=512`，每块 16 tokens / 2359296 bytes，总 1207959552 bytes = **1.125 GiB**。其中 1 个 null block，实际可用 511 块 / 8176 tokens。
- CPU KV：`cpu_bytes_to_use=4000000000`，1695 chunks（按页对齐公式），27120 tokens；LRU，`offload_prompt_only=true`。
- GPU memory utilization 仍为 0.3 的总预算，但实际 KV 池由 block override 压小；没有把 0.3 当作这次 KV 池大小。
- `max_model_len=7168`、最终 `max_num_batched_tokens=6144`、`max_num_seqs=1`。4096/4157/6000-token 请求都能完整执行，不靠抢占或超长拒绝制造结果。
- 每组先 quiet、再 pressure；每个会话开始前只对**独立测试引擎**执行 reset_prefix_cache(reset_connector=True)，GPU/CPU 均冷启动。在同一个会话的两轮之间不清缓存。

### token-ID 回填与真实生成

第一轮输入通过完整 token IDs 固定为 4096 tokens，含 hello-world Python 任务和背景填充。实际生成 18 tokens（包括终止 token），内容为：

````text
```python
def hello_world():
    print("hello world")

hello_world()
```
````

假工具结果是预设的 `hello world` / `exit_code: 0` 文本，未执行模型生成的代码。工具结果、角色分隔和第二轮“确认执行结果”的续接共编码成 43 tokens。

```text
P2 = P1_token_ids + actual_generated_token_ids + fake_tool_suffix_token_ids
4157 = 4096 + 18 + 43
```

实际生成的 token IDs 直接从 RequestOutput 取出，没有把生成文本重新 tokenize。所有 6 个会话均断言 P2[:4096]==P1、其后 18 tokens==实际生成 IDs、剩余部分==假工具 suffix。parquet 读回后再次逐项核对完整拼接。第二轮实际回复：`代码成功运行，输出为 "hello world"。`。

### 无干扰：GPU 前缀复用

第一轮结束后原会话在 GPU 有 257 个完整缓存 hash 条目（4112 tokens），第二轮前全部保留。

| 第二轮来源 | tokens |
| --- | ---: |
| GPU 有效命中 | 4112 |
| CPU 命中 | 0 |
| 实际 prompt 前向计算 | 45 |
| 总和 | 4157 |

GPU 覆盖第二轮输入的 **98.92%**，覆盖历史 4114 tokens 的 **99.95%**。45 个实际计算 tokens 分成：历史尾部 2 + 新工具/续接 43。历史尾部 2 中，1 个是之前已执行但尚未形成可复用完整块的生成 token，另 1 个是上一轮最后采样出来、尚未产生 KV 的终止 token。

### 无关请求挤满 GPU：CPU 回载

两轮之间顺序发送两个互不相关的 **6000-token** 请求，每个只生成 1 token；随机种子 100，输入均与会话头部不同，无共享前缀。每个需要 375 个 prompt blocks，总 750 个块，大于 GPU 可用的 511 个块。

| 时间点 | 原会话 GPU hash 条目剩余 | GPU 缓存总条目 | 该请求触发的 GPU 逐出 |
| --- | ---: | ---: | ---: |
| 第一轮后 | 257 | 257 | — |
| 无关请求 1 后 | 136 | 511 | 121 |
| 无关请求 2 后 | **0** | **511** | 375 |

这来自运行中的 block-hash 映射与真实逐出事件，没有用 cache_usage=0 推断前缀是否驻留，也没有手工删原会话 GPU 条目。CPU 容量足够保存这批数据，因此该实验只制造 GPU 压力，不制造 CPU 层逐出。

| 第二轮来源 | tokens |
| --- | ---: |
| GPU 有效命中 | 0 |
| CPU 有效命中 | **4096** |
| 实际 prompt 前向计算 | **61** |
| 总和 | 4157 |

已完成 CPU→GPU 传输 **603979776 bytes = 603.98 MB = 256 个 KV 块**，正好等于原始 4096-token prompt 的 KV payload。61 个计算 tokens = **18 个历史生成 tokens + 43 个新工具/续接 tokens**；原始 prompt 的 4096 tokens 实际前向计算数为 0。

原因是默认 `offload_prompt_only=true` 只卸载第一轮 prompt，不卸载 decode 产生的块。18 个历史生成 tokens 中，17 个曾在上一轮 decode 中形成 KV、现在需要重算，1 个最后采样 token 原来尚未计算 KV。本轮不改变该默认值，因此准确观察到“原 prompt 走 CPU、生成尾部与新工具内容计算”的混合结果。

### 三组重复结果

| 组别 | 场景 | GPU 命中 | CPU 命中 | 实际 prompt 计算 | 其中历史 / 新工具 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | quiet | 4112 | 0 | 45 | 2 / 43 |
| 1 | pressure | 0 | 4096 | 61 | 18 / 43 |
| 2 | quiet | 4112 | 0 | 45 | 2 / 43 |
| 2 | pressure | 0 | 4096 | 61 | 18 / 43 |
| 3 | quiet | 4112 | 0 | 45 | 2 / 43 |
| 3 | pressure | 0 | 4096 | 61 | 18 / 43 |

三类来源在本次自然续接场景中互不重叠，均满足 GPU+CPU+实际 prompt 前向=4157。由于第二轮含新工具内容，不会出现上一轮那个“整个 prompt 全命中后再回退末 token”的名义/有效来源重叠；这里还用真实前向 position 区间核实了计算数量。

辅助诊断：第二轮服务端 TTFT 中位数 quiet 为 17.50 ms，pressure 为 42.66 ms。临时 wrapper 每步拷回 positions 并写事件，会引入同步；session_wall_ms 还包含干扰请求、10 ms 假工具等待和记录开销，**不作为策略延迟/JCT 性能结论**。生成 1 token 的干扰请求没有可估计的 TPOT，parquet 中置 null，并单独保留服务端原始报告的 0。

### 启动异常与最终采用配置

首次 `max_num_batched_tokens=8192` / max_model_len=7168 / 单序列 / 512 GPU blocks 的组合在启动预热时报 CUDA IMA，尚未发出会话请求；异常输出保存为 `initial-ima.log`。随后只将 batch-token 上限调为 6144，**仍保留相同 512 块池及两个 6000-token 干扰请求**，在正常异步执行下 3 组全部通过。根因没有定位到具体 faulting kernel，不能把配置调整当成已确认的 vLLM bug fix。

按 [vLLM CUDA core-dump 指引](https://vllm.ai/blog/2025-08-11-cuda-debugging) 为复跑启用只含调试状态的小型 GPU dump；成功运行没有生成 dump，未设置 CUDA_LAUNCH_BLOCKING 或忽略错误选项。失败和成功样本没有混合。

### 证据与复现

证据目录：`../vllm/benchmarks/results/two-turn-hello-small-cache-20261101-122132`。

- [完整实验脚本](EVIDENCE.md#external-inputs)、`command.sh`，完整运行日志 `probe.log`，启动配置与 512 块覆写信息均在日志中。
- `results.json` / `summary.json`：6 会话、3 组 quiet/pressure 对照；每个请求的原始结果文件和 before/after Prometheus 快照均保留。
- 每个会话的 `*-token-construction.json` 保存完整 P1、实际生成、工具 suffix、P2；`turn1-readable.txt` 只供阅读，不参与回填。
- `events.jsonl` 记录 accepted GPU/CPU 命中、实际前向区间、GPU hash 逐出和完成的传输字节，能够核查压力确实逐出了原会话。
- `requests.parquet`：18 个请求；`sessions.parquet`：6 个会话；`token_construction.parquet`：6 组完整 ID 拼接；`events.parquet`：2508 事件；`prometheus_samples.parquet`：17824 个原始 sample。均已读回核对行数，拼接恒等式与来源总和校验全部通过。
- `engine-capacity.json`、`environment.json`、`parquet-manifest.json` 保存配置、版本、字段语义与状态；tracked diff 和 AgentKV 尚未提交文件的 patch 已归档。
- 临时 GPU 0 实例已退出并释放显存；原服务最终健康检查 HTTP 200。没有修改 vLLM 引擎文件或现有运行服务。

## 2026-11-01：补齐用户延迟、无 CPU 对照和 CPU 满容量驱逐

本次修正上一轮实验的两个范围缺口：上一轮主要验证 token 来源，带同步诊断的辅助服务端延迟不能替代用户延迟；两个 6000-token 干扰请求也没有触发 CPU 层驱逐。现在通过正式 vLLM HTTP 流式服务完成四组、每组 3 次重复。**这是当前 vLLM LRU 的功能与环境基线，不是尚未实现的 AgentKV 策略收益。**

### 配置与测量方法

- vLLM commit `5df63a7203cdb9328d824a27b682e77c7b9928be`；vLLM `0.32.0+precompiled`，torch `2.13.0+cu130`，transformers `5.19.0`；解释器为项目 `.venv/bin/python`。模型 `../models/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 使用空闲 GPU 0 / H200 NVL，UUID `GPU-48956eb2-4037-09c5-2e94-39989b4d2e0d`，CPU 与主存绑定 NUMA 2。原 GPU 1 / 8000 服务保持运行。
- 两个配置先后使用同一 GPU：CPU-enabled 端口 8001；完全省略 `--kv-transfer-config` 的 no-CPU 对照端口 8002。两者均强制 `VLLM_KV_CACHE_LAYOUT=LBHNC`，保持缓存布局一致。
- 512 GPU blocks（含 1 个 null，可用 511×16=8176 tokens）；`max_model_len=7168`、`max_num_batched_tokens=6144`、`max_num_seqs=1`、`gpu_memory_utilization=0.3`。所有请求顺序发送，无并发队列。
- CPU 层 `cpu_bytes_to_use=4000000000`、`eviction_policy=lru`、`offload_prompt_only=true`；每块 2359296 bytes，实际容量 floor(4000000000/2359296)=**1695 blocks=27120 tokens**。
- 每组之前重置自身 GPU/CPU cache；等待请求完成、CPU pending writes=0 后再取快照。先预热第一轮、第二轮及 6000-token 干扰形状，预热不计入结果。CPU 三组顺序轮换，no-CPU 在独立进程中跑三次。
- `P2=P1 IDs+第一轮实际返回的 generated IDs+假工具 suffix IDs`，固定 `4157=4096+18+43`。流式 `return_token_ids` 直接回填；不对生成文本重新分词。两轮分别生成 18、12 tokens，包含终止 token；所有组输出长度一致。
- 上游 OpenAI SDK 3.37.0 支持 prompt 的整数 token 数组；`return_token_ids` 和 `request_id` 是本版本 vLLM 扩展。已按 check-api-compat 技能核对 SDK 和本地流式实现，无 API 字段或引擎源码修改。

### 用户看到的延迟

以下是**第二轮客户端指标的 3 次中位数**，单位 ms。计时从 HTTP 请求发送前开始；TTFT 为收到第一段非空文本，文本完成为收到最后一段非空文本，HTTP 完整结束包括终止 token、usage 和 SSE 结束。TPOT=(最后 token 到达−第一 token 到达)/(输出 tokens−1)，本次第一 token 与第一文本时间一致。测量覆盖本机 loopback HTTP 流式链路，不含远端网络与浏览器渲染。

| 第二轮场景 | GPU / CPU / 本地 prefill tokens | TTFT ms | 可见文本完成 ms | HTTP 完整结束 ms | TPOT ms/token |
| --- | --- | ---: | ---: | ---: | ---: |
| GPU 保留，无干扰 | 4112 / 0 / 45 | 35.27 | 87.94 | 94.50 | 5.31 |
| GPU 被逐出，CPU 保留（2×6000） | 0 / 4096 / 61 | 59.05 | 112.86 | 119.31 | 5.37 |
| GPU 被逐出，无 CPU 层（2×6000） | 0 / 0 / 4157 | 122.92 | 176.84 | 183.10 | 5.43 |
| GPU、CPU 均被逐出（6×6000） | 0 / 0 / 4157 | 121.22 | 178.59 | 185.49 | 5.77 |

| 场景 | 三次 TTFT ms（min–max） | 三次 HTTP E2E ms（min–max） |
| --- | --- | --- |
| GPU 保留，无干扰 | 34.95–35.93 | 94.14–95.23 |
| GPU 被逐出，CPU 保留（2×6000） | 59.05–60.66 | 118.49–119.64 |
| GPU 被逐出，无 CPU 层（2×6000） | 122.77–127.31 | 183.08–187.36 |
| GPU、CPU 均被逐出（6×6000） | 120.99–125.82 | 184.83–189.59 |

每轮完整流式响应、token IDs、TTFT、TPOT、文本完成时间、HTTP E2E 均单独保存。`sessions.parquet` 保留两轮各自延迟以及 `sum_two_requests_e2e_ms`。测试 session wall 也保存，但它包含刻意串行执行的 0/2/6 个干扰请求、metrics 轮询及假工具等待，**不能把它当成可比较的策略 JCT**。真正的会话 JCT 仍需使用计划中的统一工具间隙 trace 回放，不能用这次构造的逐出测试替代。

从 CPU 保住原 prompt 的场景转到无 CPU，第二轮 TTFT 从约 59 ms 增到 123 ms，HTTP 完整结束从约 119 ms 增到 183 ms。CPU 也被逐出后表现与无 CPU 对照接近，符合“没有可复用副本就全量 prefill”的行为。三次重复仅用于基线检查，不据此估计 P99。

### CPU 确实发生驱逐的证据

6 个独立 6000-token 输入提供 **36000 tokens 干扰**，加第一轮 4096-token prompt 共需 2506 个完整 CPU 块。实测过程如下（三次相同）：

| 干扰请求后 | CPU 总驻留块数 | 原会话 CPU ready 块数 | 本次实际驱逐块数 |
| --- | ---: | ---: | ---: |
| 1 | 631 | 256 | 0 |
| 2 | 1006 | 256 | 0 |
| 3 | 1381 | 256 | 0 |
| 4 | 1695 | 195 | 61 |
| 5 | 1695 | 0 | 375 |
| 6 | 1695 | 0 | 375 |

第 4 个干扰开始驱逐，第 5 个之后原会话的 256 个 prompt blocks 已全部消失。到第二轮请求前累计实际驱逐 **61+375+375=811 blocks**，恰为 2506−1695；原会话 ready blocks 为 **0/256**。相比之下，2 个干扰后 CPU 总驻留 1006 块，原会话仍为 256/256，实际驱逐数为 0。

第二轮结果：

- CPU 保留：GPU 0 + CPU **4096** + 本地 prefill **61** =4157；CPU→GPU 完成 **603979776 bytes**。61=18 历史生成+43 新工具续接，因为 prompt-only offload 不保存 decode 块。
- CPU 被填满逐出：GPU **0** + CPU **0** + 本地 prefill **4157** =4157；CPU→GPU **0 bytes**。
- 无 CPU 层：相同 2×6000 干扰即可使 GPU 命中为 0，CPU 命中也为 0，本地 prefill **4157**，CPU→GPU **0 bytes**。

“全部重算”的精确含义是第二轮全量 prefill：其中 4113 tokens 曾产生 KV 而现在要重算，另 1 个第一轮最后采样 token 尚无 KV，43 个工具/续接 tokens 为新输入。不能把 4157 全部称为历史重复工作。

计数取自每请求 Prometheus 差分：`prefix_cache_hits_total`、`prompt_tokens_cached_by_source_total{source="host"}`、`request_prefill_kv_computed_tokens_sum` 和 `kv_offload_load_bytes_total`。这次输入存在新后缀，不触发“整个 prompt 全命中后最后 token 回退”的来源重叠；54 个测量请求均通过来源和等于输入长度的校验。

新增临时 process-local 插件记录实际 `prepare_store(...).evicted_keys` 数量与原会话 CPU keys 的 ready 状态，导出 `vllm:diag_cpu_evicted_chunks_total`、`diag_cpu_resident_chunks`、`diag_original_cpu_ready_chunks`、`diag_original_cpu_total_keys`、`diag_cpu_pending_writes`。插件只访问 CPU 元数据，每次原会话最多 256 次 key 检查，通过原生 metrics 通道输出；没有 `.cpu()`、CUDA 同步或逐 token 文件写入。仍有轻量观测开销，未宣称是无观测开销的性能结果。

注意检查原会话驻留状态的时点是**第二轮之前**。全量 prefill 后 vLLM 会重新把这些新算出的块写回 CPU，因此第二轮后同一组 hash 可以再次变成 ready；这不是这次第二轮获得了 CPU 命中。

### 输出差异与校验

GPU/CPU 命中路径回复 `代码成功运行，输出为 "hello world"。`；全量计算路径回复 `代码成功运行，输出了 "hello world"。`，均 12 tokens。初次诊断的跨路径 token-ID 全等断言因此停止，该批数据完整保留于 `initial-cross-path-output-check/`，从最终统计中排除。后续校验保持实际 ID 回填、相同输出长度及语义结果；没有要求两种计算路径输出逐 ID 相等。该措辞差异本次未进一步定位，不能声称逐 token 确定性一致。

### 复现、数据与清理

证据目录：`../vllm/benchmarks/results/two-turn-controls-http-20261101-222611`。

- `client.py`：流式 HTTP 客户端、精确 ID 回填、干扰与断言；`fixture.json` 保存本地完整 token fixture，无需访问旧目录；`plugin/cache_probe_plugin.py` 为临时 CPU 观测插件。
- `environment.json`、`cpu-server-command.sh`、`no-cpu-server-command.sh` 保存完整配置；服务原始日志与 CPU/no-CPU 客户端日志均归档。
- `cpu/` 和 `no_cpu/` 保存每请求 body、原始 SSE JSON、结果及 before/after metrics，每会话保存完整 token-construction JSON。
- `requests.parquet` 54 请求，`sessions.parquet` 12 会话，`pressure_progression.parquet` 30 个压力观测点，`token_construction.parquet` 12 组完整 ID，`prometheus_samples.parquet` 50424 个原始 samples。写入后全部读回，重新验证行数、完整 token 拼接及来源和。
- `summary.json` 保存中位数、范围及全部 3 次原值；`parquet-manifest.json` 保存字段口径与 SHA256。`summarize.py` 可重建以上 parquet（pyarrow 仅位于 `../agentkv-runs/agentkv-parquet-tools`，未改变 serving venv）。
- 原始 vLLM tracked diff、AgentKV 尚未提交文件的 patch、脚本 SHA256 归档用于恢复工作区状态。
- 两个本次实验服务均已正常关闭，GPU 0 显存释放到 0 MiB；原 8000 服务健康检查 HTTP 200。无运行中实验进程，无引擎源码改动。

完整启动命令（先 CPU 实验退出，再启动 no-CPU；两者使用同样 controlled_env）：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_SERVER_DEV_MODE=1
export VLLM_PLUGINS=agentkv_cache_probe
export PYTHONPATH=../vllm/benchmarks/results/two-turn-controls-http-20261101-222611/plugin
numactl --cpunodebind=2 --membind=2 ../vllm/.venv/bin/vllm serve ../models/Qwen3-8B --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.3 --num-gpu-blocks-override 512 --max-model-len 7168 --max-num-batched-tokens 6144 --max-num-seqs 1 --enable-prefix-caching --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 4000000000, "offload_prompt_only": true, "eviction_policy": "lru"}}'
../vllm/.venv/bin/python ../vllm/benchmarks/results/two-turn-controls-http-20261101-222611/client.py --mode cpu --port 8001
# 关闭并等待本次 CPU 服务完全退出，然后执行：
numactl --cpunodebind=2 --membind=2 ../vllm/.venv/bin/vllm serve ../models/Qwen3-8B --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8002 --gpu-memory-utilization 0.3 --num-gpu-blocks-override 512 --max-model-len 7168 --max-num-batched-tokens 6144 --max-num-seqs 1 --enable-prefix-caching
../vllm/.venv/bin/python ../vllm/benchmarks/results/two-turn-controls-http-20261101-222611/client.py --mode no_cpu --port 8002
```

## 2026-11-01：Week 1 汇总与 Agent Hints 动机图

本次只分析现有离线证据，未启动服务或发送推理请求。新增 [Week 1 文档](WEEK1.md)，整合环境/模型与启动容量、smoke、突发及 rate=4 官方 benchmark、指标与 M2 缺口、对齐尾块 15+1 复用核验、锁页 H2D/prefill 微基准、两轮 token-ID 回填与四组 HTTP 延迟对照、异常记录、代码路径和下一步任务。README 新增“为什么需要 Agent Hints”、图片与文档入口。

驻留图保存在 `docs/figures/week1-cache-residency.{png,svg,pdf}`，底层 21 行 / 3 次重复数据另存 CSV/parquet；JSON 保存来源与口径，`week1-artifacts.json` 保存 14 个原始数据文件和交付文件的 SHA256、绘图库版本及验证结果。

- CPU 六请求实测曲线：256→256→256→256→195→0→0；第 4 个请求开始驱逐，第 5 个后清空；第 6 个后累计 CPU 驱逐 811 块，三次一致。
- GPU 同配置、同 token fixture 的配对探针：257→136→0；第 3–6 点由于原会话未再访问，延续零值，以点线明确标为推断。未把来自两批探针的点宣称为同一次运行的全部实测。
- 原会话 CPU 保留需求 256/1695=15.1%。图中绿虚线是未实现的 Hint 保留目标；59.05 ms 是 CPU 保留对照的 TTFT，不宣称 Hint 在六请求压力下已取得这个时延。
- 绘图脚本：`agentkv/analysis/plot_week1_cache_residency.py`。从原始 JSON 重建 PNG/SVG/PDF、CSV/JSON/parquet，并核对三次曲线和完整会话 token fixture；未修改引擎源码或 serving venv。
- 绘图环境安装在 `../agentkv-runs/agentkv-week1-plot-tools`，使用项目 `.venv/bin/python`；matplotlib 3.11.2、numpy 2.5.4、pyarrow 26.0.0、Pillow 13.0.0。复现命令见 WEEK1.md。
- 验证通过：ruff format/check；最终脚本重绘；人工检查 PNG；26 个文档本地链接存在；PNG 2360×1400、PDF/SVG 可解析；parquet 读回 21 行；每次驱逐和为 811；原 benchmark 两组各 200 成功、0 失败。


## 2026-11-02：M0 生命周期、TTFT 分解与产物归档

M0 产物统一移入 AgentKV：[三泳道生命周期与实际块轨迹](m0_lifecycle.md)、[TTFT 分解与测量口径](m0_ttft.md)、[接口问答及 M4/M5 接入点](m0_qa.md)。客户端位于 `agentkv/bench/m0_trace_client.py`，离线分析位于 `agentkv/analysis/m0_analyze.py`；完整原始日志、三次重复、CPU 驱逐对照、关闭 trace 对照、排除的诊断尝试及校验清单位于 [results/m0-lifecycle-20261102](../results/m0-lifecycle-20261102/README.md)。vLLM 运行埋点保留在相邻的固定版本源码工作树。

- 最终实测 GPU/CPU 回载 TTFT 中位数为 29.955/52.926 ms，增加 22.971 ms；六条请求按整数纳秒逐段闭合。它是新实测，不是对 Week1 历史 35.27/59.05 ms 的追溯分解。
- 真实回载为 256 逻辑块、9216 个层块描述符、603979776 bytes；CUDA event 中位数 17.207 ms，连续带宽换算的 10.981 ms 单独列出。完成上报后确实到下一调度步恢复；当前 TP=1 默认调度器和 Worker 共用 EngineCore 进程。
- M4 已确认策略的 request-aware 回调可取得 ReqContext / kv_transfer_params；evict 本身只收数量和 protected keys。保存按批次异步发生，默认 prompt-only 不保证整个生成历史已在 CPU。
- 采集阶段 CPU manager/Worker 68、scheduler 403 项测试通过，trace 开启额外 51 项通过，适用 pre-commit 检查通过。此次迁移使用原始数据重跑离线分析与校验，不重新启动 GPU 实验。

复现：在 AgentKV 根目录执行 `bash results/m0-lifecycle-20261102/run.sh ../agentkv-runs/m0-replay`；只重建分析可执行 `../vllm/.venv/bin/python agentkv/analysis/m0_analyze.py --run-dir results/m0-lifecycle-20261102`。

## 2026-11-02：优先核查 kv_hints，界定 M4 范围

[半页结论](kv_hints.md)：kv_hints 是官方 Agent/编排器 KV Hints 的类型化信封，来自已合并 PR #53423 / commit 986e217158；它不定义 session/tool-gap 等具体动作语义。固定版本 Python 引擎与 Rust gRPC 已透传，但 Python completions/chat HTTP 未声明和转交该字段；不能以请求接受未知字段判断生效。内置 CPU LRU/ARC 不读取 hints，KVCR adapter 仅转交给外部执行器。M4 复用 envelope 和既有 ReqContext，补 HTTP 适配、payload 校验及 Registry；相关 RFC #53421/#51428/#52113/#59708/#57103 和未合并草案均已查看、归档于 `results/kv-hints-audit-20261102/`。此次未编写策略代码或运行 GPU 实验。

## 2026-11-02：#62382 策略重叠简验

[核查笔记](pr62382_overlap.md)：对 open draft #62382 的 head `7b01911392a7` 查看实际源码和测试。已有会话目标/分支索引、GPU priority/TTL 保留、显式 GPU 驱逐及 ToolWaitPolicy（暂停时长/压力判断、CPU/Mooncake 下放、预测回载），不只是框架。CPU 满容量时按下次复用时间选牺牲块尚未在核查代码中发现；PR 自述冷层保留/删除不支持。建议先围绕现有受控数据扩展 E0、与该 PR 及 CPU LRU/ARC/Oracle 做对比，CPU 预测驱逐作为候选增量。当前没有完整 E0/Oracle 结果；本次为静态核验，未复跑该分支或发布评论。证据归档 `results/pr62382-overlap-20261102/evidence.json`。


## 2026-11-02：M1 AgentX 数据下载与全量分析

[数据说明](m1_agentx_data.md)结论为“部分能用”。下载带子 Agent 的 `cc-traces-weka-062126` 固定 revision，原始文件 1,847,151,435 bytes，SHA256 与 HF 元数据一致。全量 393 根会话、98,827 次模型请求、1,697 个显式子 Agent 分组，输入/输出总量与发布方一致。输入为 64-token 块哈希及计数，无原文或真实 token ID，哈希只能在根会话内比较。

- 固定上游链识别算法产生 10,130 条分析链，并非真实 Agent 数；88,697 个链内观察间隔的 P50/P90 为 3.733/67.634 秒。原始容器相邻请求有 6,409 对重叠；缺少工具标签，不能称其为逐工具耗时。
- 3,808 对链内相邻输入发生前缀改写或缩短，不满足全部轮次严格追加的假设。
- 全局比例 0.0326236，输入加输出不超过 32,512，预留 256 token。缩放后输入 P50 为 4,633；按根会话共同缩放的补充方案为 10,616，但改变跨会话长度比。只审计长度与逻辑边界，尚未合成 token 或验证目标模型模板。
- 7 组图另存 PNG/SVG/PDF，逐请求与会话表另存 parquet/CSV；大 JSONL 和可重建 CSV 已忽略提交。源码/来源快照、统计、复现命令见 [结果目录](../results/m1-agentx-data-20261102/README.md)，分析入口为 `agentkv/analysis/analyze_agentx_data.py`。
- 本次只做数据分析，未写回放器、未运行模型、未修改项目计划或 vLLM 引擎。M7 的工具名与真实工具耗时标签仍需数据源 B。


## 2026-11-02：M1 离线负载生成与五项验证

按 `docs/m1_replayer_design.md` 第 4 节和第 7.1 节实现离线部分；按本次目录要求，代码在根 `agentkv/bench/`，测试在根 `tests/`，包内薄入口保持 `python -m agentkv.bench.workload`。未启动 vLLM、未加载权重、未写在线回放。

- 先扫描全部 393 根、98,827 次请求、338,052,834 次哈希块出现，根内 hash_id 的位置违反次数为 0；source SHA256 与固定 trace 一致。证据在 `results/m1-position-check/`。
- 默认 seed=0、100 根、预算 32512、gap_cap=120、gap_compress=1.0，生成 2274 条链、28812 次请求。辅助链占 68.7335%（请求占 5.4248%），显式子 Agent 请求占 37.3386%；原输出为零的请求 17 个。
- 输入 P50/P90/max=12348/28023.9/32511；输出=26/168/6503；请求外间隔 P50/P90=4.1065/79.6507 秒，截断 2227/26538=8.3917%。全部请求满足输入加输出 ≤32512，排除 tokenizer 全部特殊 token。
- 两个独立进程生成的 parquet SHA256 均为 `0075a865d2a0bd0f71a783388a4fa18fb80dcc7e49ccda911a7f4313f075a5ac`；文件各 112071188 bytes。命令墙钟分别 74.11/76.91 秒。
- 五项真实负载验证通过：全量位置证据、字节确定性、26538 对链内相邻 +12800 对固定随机同根 LCP、全部预算和间隔。六个快速测试通过，ruff check/format 通过。EMA 默认 0.5，链首先验取所选负载的处理后间隔中位数，final 的数值 hint 为 null；详见 [完整报告](EVIDENCE.md#archives)。
