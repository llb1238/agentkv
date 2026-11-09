# M1 离线负载构建

实现统一放在 `agentkv/bench/`，测试在根 `tests/`；hint 协议放在 `agentkv/hints/protocol.py`，M3 模拟器放在 `agentkv/sim/`。所有模块通过 `python -m agentkv.…` 调用。离线生成器不启动 vLLM；在线回放、hint 和观测说明见本文件末尾。

从相邻 vLLM 目录使用已有 `.venv/bin/python`。本次只用 uv 重装 AgentKV 的 editable 包，没有安装或启动 vLLM；分析依赖使用 临时目录 中的独立目录。

```bash
# 在 vllm 目录；只安装项目代码，不改动 serving 依赖。
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e ../agentkv
# 若没有分析依赖，可用 uv 放到独立目录：
uv pip install --python .venv/bin/python --target ../agentkv-runs/agentkv-offline-tools \
  -r ../agentkv/results/archive/m1-workload-seed0/requirements.txt
export PYTHONPATH=../agentkv-runs/agentkv-offline-tools
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.venv/bin/python -m agentkv.bench.workload \
  --output-dir ../agentkv/results/m1-workload-seed0-v3
.venv/bin/python -m agentkv.bench.workload \
  --output-dir ../agentkv/results/archive/m1-workload-seed0-v3-repeat
.venv/bin/python -m pytest ../agentkv/tests -q
.venv/bin/python ../agentkv/tests/validate_workload.py
.venv/bin/python -m agentkv.bench.estimate_replay \
  --workload ../agentkv/results/m1-workload-seed0-v3/workload.parquet
```

本次实际使用的依赖路径为 `../agentkv-runs/agentkv-parquet-tools:../agentkv-runs/agentkv-week1-plot-tools`；不是上述新目录。tokenizer 从 `../models/Qwen3-8B` 本地加载，只读 tokenizer/config 文件，不加载模型权重。

生成器验证 `results/m1-position-check/result.json` 中的全量位置检查通过，核对原始 trace SHA256，关联审计的 `requests.parquet` 和 `inferred_chains.parquet`。不重新拆链，不删除辅助或子 Agent 请求。输出目录已有负载时拒绝覆盖，重新生成须使用新目录。

```text
trace_loader.py  原始请求与审计位置/链的严格关联；保留完整 hash_ids
subset.py        对排序后的全部根 ID 用 PCG64(seed) 无放回抽样
synth.py         根内统一有理数缩放；SHA256/PCG64 非特殊 token 合成
workload.py      间隔、数值 hints、固定 schema parquet、来源和统计 manifest
estimate_replay.py  按请求依赖图估算固定服务时间的事件锚定回放
steady_state.py  从进度点生成回放入口、抽样会话实例、执行实例词表置换
estimate_steady_state.py  四种到达率的稳态/空初始化对照与 CDF 图
```

约定均写入 manifest：哈希种子使用 UTF-8 `root_id + 十进制 hash_id`，前 8 字节按大端转整数；EMA 系数默认 0.5，第一条历史间隔初始化 EMA，链首使用**未被抽中根会话**的链内间隔经相同截断、压缩处理后的中位数先验，同时记录先验值、间隔数与根 ID。`final` 的 gap 和数值 hint 均为 null，第一轮 LCP 为 0；无 hint 档不需要额外列。perfect/noisy 的 P90 可在以后组装时由 expected×2 得到。当前不组装任何信封。

链启动采用事件锚定。每条链选择同根、其他链中原始结束时间不晚于链首、且结束时间最晚的请求，链首写入其 `anchor_trace_req_idx` 与 `start_delay_s = min(t_head - end_anchor, gap_cap) × gap_compress`。无请求锚点时，前者为空，延迟为 `min(t_head, gap_cap) × gap_compress`，从根会话到达开始等待；非链首两列均为空。结束时间并列与零耗时同刻事件按设计第 4.4 节的固定顺序处理，避免依赖循环。manifest 和 parquet 元数据标记 `event-anchored-v1`。

`chain_offset_s` 保留合并根活跃区间、截短区间间长空闲后的压缩偏移，`chain_offset_orig_s` 保留原始链首 `t`，**仅供参考，在线调度不使用**。manifest 的根时间映射和旧偏移推算时长也只作参考。

`python -m agentkv.bench.estimate_replay` 从负载的锚点和链内间隔构建请求依赖图，假设每次服务时间固定为 s=0.2/0.5/1.0 秒，报告各根完成时间分布及到达后 25 分钟内实际能发出的请求占比，输出 `replay_estimates.json` 和每根明细 `root_completion_times.csv`。估算忽略参考偏移及原始 `api_time`，允许无限并发，不包含网络、排队或在途上限。

同时输出独立的 `schedule_estimate.parquet`，固定按 s=0.5 秒为每个请求写入 `est_send_s`，并在同根的所有请求行重复该根的 `est_duration_s`。它只包含请求关联 ID 与这两列，不含 token；元数据记录服务时间、源负载 SHA256 和估算器实现 SHA256。即使 `--service-times` 不包含 0.5，也仍按 0.5 生成这个文件；原有 JSON 和 CSV 的字段与含义保持不变。用 `--output-dir` 可将新增估算保存到独立结果目录：

```bash
python -m agentkv.bench.estimate_replay \
  --workload results/m1-workload-seed0-v3/workload.parquet \
  --output-dir results/m1-steady-state-seed0
```

`RootSchedule(requests, schedule).entries(tau)` 返回所有 `est_send ≥ tau` 的请求及触发计划。`ReplayEntry.delay_s` 在定时模式下相对实例开始，在事件模式下相对 `trigger_trace_req_idx` 的完成；`timed` 用于区分两种模式。只有触发请求被跳过时才设 `is_entry=true`，根到达触发的链首不设该标记。

`sample_population` 用固定 `population_seed` 派生五个独立 PCG64 流，分别负责初始数量、初始根、进度、泊松到达及新根。初始数量为 Poisson(λE[D])，根按时长加权、进度均匀；新根均匀有放回抽取。初始实例用奇数编号，新实例用偶数编号，保证初始化开关对照中的新实例完全一致。`InstanceTokenPermutation` 按设计第 5.5.2 节的 SHA256 种子执行合法词表的一一置换，供在线阶段使用。

```bash
python -m agentkv.bench.estimate_steady_state --population-seed 0
python results/m1-steady-state-seed0/validate_population.py
```

估算器执行四种 λ × 初始化开关共八组对照。40 分钟窗口的前 10 分钟预热，每 10 秒采样一次活跃实例和链；每组保存实例参数、逐请求记录及 `run.json`，汇总保存 `summary.json` 和 prompt CDF 的 PNG/PDF。结果及统计口径见 [稳态初始化报告](../../results/m1-steady-state-seed0/README.md)。

原输出为零时，`max_tokens` 提升为 1，保留 `out_zero_orig`；如造成预算冲突则明确报错，不静默改变根内缩放。本次默认子集没有该冲突。根内缩放后任意哈希块若成为零宽，也会报错。

Parquet 固定按根 ID、链 ID、链内序号排列，128 行 row group，int32 token 列，ZSTD level 3，无时间戳。相同输入、参数、代码与依赖环境下进行字节级确定性验证；manifest 的耗时字段允许两次不同。完整 token 列是大型可重建产物，不提交 Git，SHA256 与参数随 manifest 保存。

`tests/test_synth.py` 覆盖五项验收及嵌套请求关联，包含上下文改写、父子共享、不同根、零输出、截断/压缩、因果历史 hint、未抽中根会话先验、参考时间压缩及事件锚点规则。`tests/test_estimate_replay.py` 用可手算的依赖图检查并发启动、链内前驱、链中间请求锚点、循环拒绝和 25 分钟发送边界。`tests/validate_workload.py` 重新读取真实 trace，检查两份独立产物的 SHA、所有请求的预算/间隔/特殊 token、全部链内相邻对和每根固定 128 对随机请求的 LCP；独立重算每条链的锚点、启动延迟、参考偏移和未抽中根会话先验，并用原始事件顺序交叉检查三种 s 下的全部发送时间。可用 `--position-check` 指定重新扫描的全量位置唯一性证据。

数据和估算报告见 [v3 负载目录](../../results/m1-workload-seed0-v3/README.md)，[v2](../../docs/EVIDENCE.md#archives) 和 [v1](../../docs/EVIDENCE.md#archives) 保留作对照。

## 在线回放与本地验证

在线依赖在 `.[replay]` 中声明。`agentkv.bench.replay` 读取原负载和固定 s=0.5 秒调度，沿 `RootSchedule` 的入口计划执行：保留的触发请求完成后等待相应间隔，跳过的触发请求则用相对 τ 的剩余时间定时触发。HTTP 使用 `/v1/completions`、token ID prompt、流式响应和 `ignore_eos=true`，不使用 chat template；token 按实例词表置换。`none` 完全省略 `kv_transfer_params`，其他档按设计第 4.5 节组装信封。

```bash
# 运行真实流式 HTTP 假服务测试，不需要 GPU。
python -m pytest tests/test_replay.py -q
# 也可单独启动可调服务时间的本地假服务。
python -m agentkv.bench.fake_server --port 8013 --service-time 0.5
# 在线客户端：时间参数单位为秒，到达速率单位为根会话/分钟。
python -m agentkv.bench.replay --base-url http://127.0.0.1:8012 \
  --output-dir results/new-online-run --run-id new-online-run \
  --policy lru --hint-tier none --arrival-rate 1 --steady-state \
  --duration 2400 --warmup 600 --cooldown 0 --max-inflight 128
python -m agentkv.bench.analyze_replay --run-dir results/new-online-run
```

回放器、容量分析及测量工具默认拒绝 dirty 工作区（含未跟踪文件），在输入读取和结果创建前检查。`run.json` 的代码来源只记 `git_commit`；不再写 dirty/status/diff/实现指纹，不向结果目录复制实现代码。`--allow-dirty` 仅调试使用，运行时打印红色加粗警告，调试结果不能用作正式实验。输入、调度和产物 SHA256 仍记录。已有运行产物拒绝覆盖。请求超过 `--max-inflight` 时排队，排队后再次检查截止时间；到 T 停止新发送，`--drain-timeout` 内排空在途请求。失败请求不重试并阻断其依赖分支。

回放器在实验时钟启动前将 token 列解码到紧凑的 Arrow 缓冲区，不将整个负载展开为 Python token 列表。事件请求在触发请求完成时开始构建请求体；定时请求在实例开始时开始构建，使用两个专用线程，预留整个等待间隔，发送截止点只发送已构建的字节。零等待或构建超过间隔时仍会迟发，并如实计入误差。置换使用与原算法完全等价的稠密查表。输出保留原负载 token，另记实际发送的 SHA256 和置换种子。每次运行写 `requests.parquet`、`run.json`、`metrics.jsonl`、`loop_lag.parquet`；分析器生成等待队列 PNG/PDF、逐请求名义来源比例 CSV 和 `analysis.json`。

每请求新增 `trigger_ns`、`body_ready_ns`、`body_build_ms`、`body_prepare_wait_ms`。构建耗时只计构建线程实际执行的墙钟时间，等待线程池另记；每 10ms 用绝对截止点检查事件循环，记录计划/唤醒时间、延迟及跨过的检查数，不用补跑一批零延迟检查隐藏停顿。`run.json` 记录统计窗口内分位数及总漏检数。

本地 vLLM 试跑用 `agentkv.bench.observability_plugin` 采集每次首 token 输出对应的名义来源统计。将仓库根与 `agentkv/bench/plugins` 加入服务端 `PYTHONPATH`，设置 `VLLM_PLUGINS=agentkv_replay_observer` 和 `AGENTKV_OBSERVATION_DIR`；客户端用 `--observation-dir` 指向同一目录。CPU-only offload 下外部来源计为 CPU；有效口径和 replay_overlap 未采集时保持 null。轻量插件不启用全量 `AGENTKV_TRACE`。

假服务证据见 [m1-replay-fake-validation](../../docs/EVIDENCE.md#archives)，真实试跑见 [m1-online-lru-seed0-final](../../docs/EVIDENCE.md#archives)。

## 离线容量与调度长尾

`agentkv.bench.capacity` 只用 CPU：按 s=0.5 调度计算同实例严格更早发送的所有请求中的最大 prompt LCP（同时发送互不命中），以 token 加权；输出 32K/16K、λ=0.5/0.7/1 的 prefill 需求曲线、实际槽位模拟 KV P90、满槽位的保守 KV P90 和分层缓存候选。16K 用原 trace 重新按根会话一致缩放，保持相同根、种子、锚点和间隔，不能直接截断 v3。

```bash
python -m agentkv.bench.workload --context-budget 16384 --output-dir results/m1-workload-seed0-v3-16k
python -m agentkv.bench.capacity --output-dir results/new-capacity-run
python -m agentkv.sim.two_tier --capacity-dir results/new-capacity-run --output-dir results/new-two-tier-run
python -m agentkv.bench.scheduler_probe --output-dir results/new-scheduler-probe
```

容量分析中“32K”为既有 v3 的 32512 token 总预算；16K 为 16384。曲线用平稳长期均值 λ/60×每根总 prompt token 的均值×(1−命中率)，不将单次有限窗口的抽样噪声混入到达负载。工作集为活跃链最新 prompt 的不同前缀之并，排除输出，属于上下文 footprint 的保守下界；按 s=0.5、40 分钟含 10 分钟预热，在统计窗口每 10 秒采样，以 P50 作筛选基准。GPU 按满槽位 P90 与调度 P90 中较大的值的 2.5 倍定容。筛选要求 `2.5×P90 ≤ GPU < CPU ≤ 0.8×工作集` 且 `1.5 ≤ CPU/GPU ≤ 2`；不再约束 GPU+CPU 小于工作集。所有通过项和最近的失败项及其约束缺口写入 run.json。

`agentkv.sim.two_tier` 用 16-token 完整前缀块、独立实例命名空间和重复写 CPU 镜像层模拟 LRU；运行中的 GPU KV 不可驱逐，输出 KV 仅作私有占用，不缓存；请求完成后才将 prompt 块标为就绪。GPU 淘汰不触发 CPU 插入。它按同一稳态实例执行事件依赖，先找 GPU 连续前缀，再从 CPU 延伸，剩余全部重算。数据搬运耗时视为零、每请求服务时间固定 0.5 秒，因此不是 vLLM 性能模拟器。单元测试用 token 粒度验证无限缓存等于最大 LCP 上限；实验使用 16-token 粒度并报告同人口无限缓存对照，分离块边界/冷启动与驱逐损失。

只有 LRU 命中率比全负载上限、同人口无限缓存对照都至少低 10 个百分点，且 token 加权重算需求 `≤15400 token/s` 的配置，才推荐 GPU 短时试跑。报告同时列出对 22k/34k 算力线的比例。该门槛是候选筛选，不保证共享 GPU 的实际稳定性；正式 E0 仍需实测队列与运行并发。

调度探针使用原 v3 不同行组的 12K prompt、125 条链、独立假服务进程，对比归档实现；保留未通过的诊断运行，不能靠换负载通过。探针服务时间 4 秒，用于维持约 125 个在途，不用于容量分析的 s=0.5。

完整结果：[新两层容量筛选](../../results/m1-capacity-seed0-v3/README.md)、[LRU 模拟与短跑清单](../../results/m1-two-tier-seed0/README.md)、[两次最终长尾对照](../../results/m1-scheduler-tail-final/README.md)。[旧容量 v2](../../docs/EVIDENCE.md#archives) 保留作历史对照。本次只推荐短跑配置，没有执行 GPU 验收。

CPU 策略通过 `agentkv.sim.two_tier.CPUCachePolicy` 的 `on_access` / `evict` 接入；内置 LRU 和按未发出请求预计到期时间维护未来访问的 Oracle。14 组合对照入口为 `python -m agentkv.bench.compare_oracle --output-dir ../agentkv-runs/new-oracle-run`。共享GPU指定试跑入口为 `python -m agentkv.bench.shared_pilot --output-dir ../agentkv-runs/new-pilot-run`；先提交代码，保持工作区干净，完成后再将数据发布到结果目录。
