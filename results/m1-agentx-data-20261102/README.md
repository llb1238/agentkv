# M1 AgentX 数据审计：2026-11-02

结论与图：[docs/m1_agentx_data.md](../../docs/m1_agentx_data.md)。本目录只包含离线数据分析，不含回放器、策略或模型实验。

## 固定输入

- Harness：`SemiAnalysisAI/agentx-harness@e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`。
- 数据：`semianalysisai/cc-traces-weka-062126@23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl`，Apache-2.0。
- 本机原始文件：`data/raw/agentx/cc-traces-weka-062126/traces.jsonl`，1,847,151,435 bytes。
- SHA256：`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`，与 HF LFS 元数据一致。
- `provenance.json`、`hf-metadata.json`、`hf-tree.json` 保存来源和固定 revision；`source/` 保存所用原始模型、链识别辅助函数和文档快照及其许可证。

## 复现

在 AgentKV 根目录执行。依赖放在临时目录，不更改 serving 环境；Python 使用相邻 vLLM 的 `.venv`。

```bash
bash results/m1-agentx-data-20261102/download.sh
uv pip install --python ../vllm/.venv/bin/python \
  --target ../agentkv-runs/agentkv-m1-tools \
  -r results/m1-agentx-data-20261102/requirements.txt
bash results/m1-agentx-data-20261102/run.sh \
  > results/m1-agentx-data-20261102/analysis.log 2>&1
```

下载器可断点续传；已有完整文件时只校验字节数和 SHA256。当前机器最初遇到 HF 域名解析错误，实际下载通过 curl `--connect-to` 连接经 DNS-over-HTTPS 查询出的地址，仍保留 HTTPS 主机名与证书校验。普通网络无需该选项；本机仍有解析故障时，可把当时有效的地址传给 `AGENTKV_HF_CONNECT_IP` / `AGENTKV_HF_CDN_IP`，脚本不固定临时 CDN IP。

分析入口：`agentkv/analysis/analyze_agentx_data.py`。它在内存中加载固定源码的 schema 与 `detect_agent_chains`，仅替换 AIPerf 的基类/类型导入，不安装或运行 harness。它不是完整 benchmark loader：没有执行请求、合成 token、分类/过滤 preamble 或测量缓存命中。

## 结果与字段

| 文件 | 行的单位 | 关键口径 |
| --- | --- | --- |
| `sessions.parquet` | 原始根会话，393 行 | 顶层/嵌套请求数、显式子 Agent 数、按根会话缩放因子 |
| `requests.parquet` | 模型请求，98,827 行 | 根 ID、容器、原数组位置、推断链与轮次、相对开始/结束、输入/输出长度、上轮 LCP、两种缩放长度 |
| `inferred_chains.parquet` | 推断链，10,130 行 | 分析派生 ID，不是真实 Agent ID；链内串行 |
| `idle_gaps.parquet` | 同一推断链的相邻请求，88,697 行 | `idle_s=next.t-prev.t-prev.api_time`；没有逐工具标注 |
| `raw_container_gaps.parquet` | 原容器的相邻请求，96,737 行 | 保留 6,409 个负值，诊断并发串流 |
| `inferred_forks.parquet` | 算法识别的分叉候选，8,040 行 | 与显式子 Agent 边不同；其中 692 条有正的共享前缀 |
| `subagents.parquet` | 显式 subagent marker，1,697 行 | 根容器、agent_id、spawn、duration、嵌套请求数量 |
| `summary.json` | 汇总 | 全量统计、覆盖率、质量检查、缩放与前缀改写诊断 |
| `schema_examples.json` | 两条截短示例 | 真实记录的字段值；只列哈希头部 |

CSV 为同名 parquet 的便于检查副本，可由脚本重建，已忽略提交。输入的完整哈希序列留在下载的 JSONL 中，不在派生表里重复存储。图为 `docs/figures/m1_agentx_*.{png,svg,pdf}`，使用标准 matplotlib 绘制。

中位数与 P90 使用 numpy `quantile(method="linear")`，按表中的行等权，不按会话重新加权；因此轮次数 P90 可以是小数。未截断长空闲间隔，未把缺失数据补成 0。缩放预算是输入加输出 ≤32,512，另留 256 token；尚未经过目标 tokenizer/template 的 token 合成验证。

运行环境、制品校验值和验证结果见 `manifest.json`。重绘 PDF 的创建时间可能变化；判断复现以统计值、表内容及固定输入为准。原始 1.85 GB JSONL 在 `.gitignore` 中，不直接提交。
