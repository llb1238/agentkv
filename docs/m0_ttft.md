# M0：TTFT 分解与原始测量

本次测得 CPU 回载比 GPU 命中多 **22.971 ms**。
Week1 原始 HTTP 结果为 35.27/59.05 ms，差 23.78 ms；本次完整埋点的
29.955/52.926 ms 是新的独立实测，不能声称已追溯拆开昨天每一毫秒。
配置与 token 构造沿用 Week1，完整证据在
[实验目录](../results/m0-lifecycle-20261102)，
[代码路径与泳道](m0_lifecycle.md)、[接口问答](m0_qa.md)另列。

## 三次实测与口径

单位均为 ms。相同形状预热后，每组开始前重置 GPU/CPU cache，
组内两轮不重置；轮换 GPU、2×6000、6×6000 场景顺序。
TTFT 从客户端发送 HTTP 前的 monotonic timestamp 到首个非空流式文本，
本次首个 token IDs 与首文本同时到达。数据包含本机 loopback HTTP/前端开销。

| 场景 | trial 1 / 2 / 3 TTFT | 中位数 |
| --- | --- | ---: |
| GPU 命中 | 39.340 / 29.955 / 29.903 | 29.955 |
| CPU 回载 | 61.275 / 52.926 / 52.873 | 52.926 |

所有三次：GPU 命中为 4112/0/45，CPU 回载为 0/4096/61
（GPU / CPU / 实际 prefill tokens），总和均为 4157。
CPU 回载每次都是 256 blocks、603979776 bytes；两种路径均回复
`代码成功运行，输出为 "hello world"。`，12 个输出 tokens。
第一轮实际返回 18 个 IDs，直接回填到第二轮：4157=4096+18+43。

## 可相加的分解表

为同时保留“中位数 TTFT”和“每段相加等于 TTFT”，先按 TTFT 排序取中间
那条真实请求，再分解它。两列本次都对应 trial 2。这不是逐阶段各取中位数：
通常 `sum(median(阶段)) != median(sum(阶段))`。
逐段独立中位数也保存在 `analysis.json.component_medians`，没有偷偷调整余项。

| 阶段（ms） | GPU 命中 | CPU 回载 | 差值 |
| --- | ---: | ---: | ---: |
| HTTP 发送→调度器收到请求 | 6.679 | 6.952 | +0.273 |
| 调度器收到→首次调度 | 0.129 | 0.123 | -0.006 |
| GPU 查找及调度入口 | 0.711 | 0.110 | -0.602 |
| connector 查找 | 0.441 | 0.740 | +0.299 |
| 分配、构造加载任务及拷贝描述符 | 1.680 | 3.247 | +1.567 |
| 提交拷贝的主机调用 | 0.000 | 6.427 | +6.427 |
| 提交返回→Worker 观察到完成 | 0.000 | 11.551 | +11.551 |
| Worker 完成→调度器收到上报 | 0.000 | 1.607 | +1.607 |
| 收到上报→下一次调度恢复 | 0.000 | 1.159 | +1.159 |
| 恢复→prefill 调度完成 | 0.000 | 0.923 | +0.923 |
| 调度完成→Worker 执行入口 | 0.070 | 0.018 | -0.052 |
| Worker 执行→调度器处理首 token | 14.881 | 14.930 | +0.049 |
| 调度器输出→前端首 token | 0.941 | 0.710 | -0.230 |
| 前端首 token→客户端首文本 | 4.423 | 4.430 | +0.007 |
| **合计 TTFT** | **29.955** | **52.926** | **+22.971** |

内部使用整数纳秒校验：六条请求的所有区间非负，且逐条精确相加等于
`first_text_ns - start_ns`；表中四舍五入到 0.001 ms 时可能有末位舍入差。
GPU 路径仍会调用 connector 检查未命中的尾部，所以该行不写成“—”。

CPU 中位数请求比 GPU 多出来的 22.971 ms，可归为：

- 拷贝主机提交 6.427 + 提交返回后的完成等待 11.551 = **17.978 ms**。
- Worker 完成到调度器收到 1.607 + 下一调度步恢复 1.159 = **2.766 ms**。
- 相比 GPU 路径多出的分配/任务/描述符构造 **1.567 ms**，以及恢复后登记 GPU
  hash、完成 prefill 调度 **0.923 ms**。
- 其他区间的差值合计约 **−0.263 ms**；包含 GPU hit/miss 查询量差异、
  前端波动等。61 对 45 tokens 的执行区间在该请求中仅差 **0.049 ms**。

这组数据不支持把所有额外延迟都归给多出的 16 个 prefill tokens。

## 边界定义与旁证

主表是连续、不重叠的 wall-clock 分区：
`client start → request_arrived → schedule_consider → connector_lookup_begin
→ connector_lookup_end → copy_enqueue → copy_enqueued → copy_done
→ scheduler_recv_done → load_resume → schedule_end → worker_execute
→ scheduler_output → frontend_first_token → client first text`。
GPU 无加载路径用本次 `schedule_end` 折叠加载专用边界为零。

“分配、构造任务及描述符”覆盖 connector 查询结束到 transfer enqueue；
CPU 一列含分配 256 目的 GPU 块、load plan、派发与描述符填充；
GPU 一列在调度完成处结束。“Worker 执行→调度器首 token”包括输入准备、
GPU forward、采样和结果处理，**不是 CUDA kernel-only prefill 时间**。
“提交返回→完成”含异步完成轮询，不能把它直接改名为纯 DMA 时间。

| 回载请求 | load / 收到完成 / 恢复 step | 未完成检查次数 | WAITING 全过程 | 收到上报→恢复 | CUDA event | CPU hash 扫描 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trial 1 | 132 / 142 / 143 | 10 | 21.417 | 1.157 | 17.203 | 0.266 |
| trial 2 | 328 / 338 / 339 | 10 | 21.295 | 1.159 | 17.207 | 0.259 |
| trial 3 | 705 / 715 / 716 | 10 | 21.308 | 1.155 | 17.225 | 0.266 |

每次 `recv_step + 1 == resume_step`：完成上报后确实再等一个调度步，
该区间中位数 **1.157 ms**。
等待期间的 10 次检查是 connector-only 空计算调度，和正常 decode 步长无关。
本配置 scheduler/Worker 同 PID 1330768，故主表“上报”包括
执行器结果处理，不能直接标成进程间传输；前端 PID 为 1330220。

纯 GPU 查找的三次中位数：GPU 场景
**0.663 ms**，
CPU 场景 **0.063 ms**。
CPU `_maximal_prefix_lookup` 三次均为 257 次查询（256 HIT + 1 MISS），
其中位数 **0.266 ms**；整个 connector 查询中位数
**0.732 ms**，还含 key 构造和状态/指标处理。

真实回载是一个 job、9216 个 64 KiB 层块描述符，CUDA event 中位数
**17.207 ms**，有效 payload 带宽
**35.10 GB/s**。
Week1 连续 pinned H2D 约 55 GB/s 对这份 payload 的换算为 **10.981 ms**，
本次 event 多 **6.225 ms**，约为其
**1.57 倍**。
参考是 [Week1 带宽报告](EVIDENCE.md#external-inputs)，
不是声称重新跑了 256-block 连续带宽微基准。

主机提交区间约 6.4 ms 发生在 event 范围以内/与其重叠；event 计时包含
流中的空隙，不可与主机时间重复相加，也不能据此断言纯 DMA 本身变慢了
6 ms。两个时钟不足以拆开驱动内部活动，需要 profiler 才能继续归因。
目前已经能明确主机调用、提交后等待、完成上报和下一步恢复分别花在哪。

## 观测开销与排除记录

初版逐事件 JSON 序列化的数据中位数为 29.953/63.916 ms，差 33.963 ms，
不能作为最终低开销分解。最终版只在热路径记录对象，退出时写 JSONL；
EngineCore 1330768 共 42118 条、前端 45 条，均低于 65536 的
批量写盘阈值，因此测量期间没有触发该写盘。shutdown 显式 flush，
启动设置 `--shutdown-timeout 30`，避免默认立即终止子进程丢失内存日志。

关闭 trace 的独立进程对照（同样三次）中位数为
**30.354/52.436 ms**，差
**22.081 ms**。
最终开启 trace 的结果与之接近；三次样本和顺序运行仍不足以声称零开销，
也不能把两批中位数之差全部当成埋点的因果成本。
trial 1 的两条路径都出现较长的执行区间，全部保留，未以“异常值”为由删除。

早期诊断分别保存在 `pilot/`、`pilot-v2-check/`、`pilot-serialized/`，
缓冲版缺少 shutdown flush 的尝试保存在 `pilot-buffer-no-flush/`；
后者丢失了 EngineCore 内存日志，未用于分解。最终分析只读顶层 `traces/`
与 `cpu/sessions.json`，不混入这些尝试，也不为缺失时间戳补估算值。

## 复现与验证

[`m0_trace_client.py`](../agentkv/bench/m0_trace_client.py)复用 Week1 的
流式客户端、输入 fixture 与 CPU 诊断插件；
[`m0_analyze.py`](../agentkv/analysis/m0_analyze.py)重新连接原始事件并校验闭合。
实验目录中的 [run.sh](EVIDENCE.md#other-evidence) 接受全新结果目录，
完成启动、三次重复、正常退出与分析。

```bash
cd .
bash results/m0-lifecycle-20261102/run.sh ../agentkv-runs/m0-replay
../vllm/.venv/bin/python agentkv/analysis/m0_analyze.py \
  --run-dir results/m0-lifecycle-20261102
```

完整原始 JSONL、每请求 SSE/body/result、before/after Prometheus、token fixture、
`environment.json`、源码快照 `instrumentation.patch` 与 SHA256 manifest 均在实验目录。
`analysis.json` 包含原始请求 ID、所有边界纳秒、阶段值、逐阶段中位数及选中的
中位数请求；`ttft_per_request.csv` 可独立复核六条请求。
CPU 溢出对照的完整数据也保留在 `cpu/sessions.json`。

验证：CPU manager + connector Worker **68 passed**；connector scheduler
离线运行 **403 passed**；开启 trace 的 CPU manager **51 passed**。
未开启离线模式的初次组合测试在 167 passed 后长时间等待，人工中断，
保留 `pytest.log`；随后上述独立套件全部完成，不将中断的尝试计为通过套件。
本次改动只增加观测，不修改计算/策略行为；hello-world 功能请求的实际 IDs
回填、来源和、输出长度及内容均完成验证，不把它称为通用模型质量评测。
