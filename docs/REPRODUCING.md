# 复现与路径约定

## 路径均从仓库根目录解释

`agentkv/` 为本仓库；相邻的 `../vllm/` 为独立后端源码；`../agentkv-runs/` 为本机持久载荷与新输出位置；模型示例位置为 `../models/Qwen3-8B/`。这些是相对布局约定，不表示仓库随附后端、模型或外置原始文件。任意位置均可替换，但正式回放的输出必须在 Git 工作区之外。

```text
workspace/
  agentkv/          # 本仓库
  vllm/             # 独立的固定版本后端
  models/Qwen3-8B/  # 自行取得的模型与tokenizer
  agentkv-runs/     # 外置载荷、依赖目录及新输出
```

历史报告中的展示路径已按此约定转换；旧生产者JSON、原始日志和源码快照保留当时的字节，包括环境路径。每次展示修改的前后SHA登记于 `results/relocations.json`；原始展示文档可从收尾前提交恢复。不要用修改后的README字节冒充历史运行的原始SHA。

## 仅CPU：重建最终表和核心图

无Git历史的lean分发请使用[独立venv验证流程](RELEASE.md#无历史副本中的cpu验证)。`final_report`自动从保留的逐次分析JSON重算表格，并比较最终总结JSON和表格；原始证据的获取范围见[EVIDENCE](EVIDENCE.md)。完整检出仍使用下面的逐请求核验路径。

首页快速开始使用`--input-mode summary`，输出`homepage-table.md`并核对首页13个数字；它适用于公开lean和私有完整检出。共同token结论沿用原核验记录，不把汇总复算称为逐请求证据重查。

使用Python 3.12。新建环境后安装项目；不安装或启动vLLM也能重绘，后端相关测试会按依赖条件跳过。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[offline,replay,simulation,test]'
.venv/bin/python -m pytest tests -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m agentkv.analysis.final_report --output-dir ../agentkv-runs/report-review
```

最后一个命令读取已跟踪元数据，独立复算E1 v1/v2及λ=1.2的全部汇总和共同token核对，并重绘E0、重算图、TTFT CDF和队列图。输出目录必须不存在；它以文件链接读取原始请求元数据，不写入结果目录，不读取完整prompt数组，不启动任何GPU运行。输出 `run.json` 保存输入SHA、提交号和逐项一致性结论。结果等价以统计值核对，PDF生成时间等元数据不要求字节相同。

本次完整环境测试结果与依赖边界见各结果报告。纯CPU新环境若缺少vLLM，不能把跳过的后端适配测试称为通过；使用匹配后端环境才可复现那部分测试。

首页唯一总结图及所有新增比例可直接从现有汇总重建，无需读取大载荷或启动GPU：

```bash
python -m agentkv.analysis.summary_figure --output-dir ../agentkv-runs/summary-review-new
```

它分别使用E1 v2的λ=1/1.4批次和独立λ=1.2批次；相对百分比先取三人口均值，再求策略/LRU之比。图内误差线是人口最小–最大，不是置信区间。发布快照和最终后端补丁见[本地发布说明](RELEASE.md)。

## 重新生成负载与CPU实验

原始AgentX由固定下载脚本获取，许可和SHA见 [DATASET](DATASET.md)。tokenizer由使用者另行取得。以下仅为CPU命令，生成器拒绝覆盖已有输出：

```bash
bash results/m1-agentx-data-20261102/download.sh
python -m agentkv.bench.workload --tokenizer ../models/Qwen3-8B \
  --context-budget 16384 --output-dir ../agentkv-runs/new-workload-16k
python -m agentkv.bench.estimate_replay \
  --workload ../agentkv-runs/new-workload-16k/workload.parquet \
  --output-dir ../agentkv-runs/new-estimate
```

固定负载的上游revision、审计链表、位置验证与参数是输入的一部分，不能仅更换一个tokenizer就声称产物指纹不变。模拟器开发、冻结和留出命令见[CPU留出报告](../results/m5-scoring-evaluation-20261104/README.md)，开发与评估需新目录，不能用留出结果重新选策略。

## 外置载荷与历史GPU复现

原始trace、模型权重、完整prompt数组和大回调流不包含在当前源码分发中。[外置载荷清单](publication_payloads.json)记录本次停止跟踪的文件大小、SHA256及原Git blob，不修改本机原文件。通过合法渠道取得对应文件后，放回清单的相对路径，再按SHA校验；没有载荷时不能声称完成完整回调重放。

历史GPU协议分别为 [v1](e1_gpu_preregistration.md)、[v2](e1_v2_preregistration.md)、[λ=1.2](stable_limit_12_preregistration.md)。完整重跑需要切到记录的测量提交，恢复原始输入，并使用指定后端和模型；当前目录整理不替代冻结SHA检查。**本次收尾不执行这些协议，也不启动新的GPU运行。**

旧设计的绝对运行位置只用于解释原始来源，不是可移植复现命令。若未来另行批准重跑，应先选择新的持久目录并保持干净源码；`--allow-dirty`只能调试，不能产生正式比较结果。
