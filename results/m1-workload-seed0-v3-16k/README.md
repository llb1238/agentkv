# v3 的 16K 派生负载

同一 seed=0、100 根会话，使用原 trace 按 M1 根会话一致缩放重新生成，预算为 16384；事件锚点和间隔未改变，不对现有 v3 做截断。

命令：`python -m agentkv.bench.workload --context-budget 16384 --output-dir results/m1-workload-seed0-v3-16k`。

负载 SHA256：`a35eaf1eafcb9de245be86ac1f847de0314b7dcf806ecf4ebc5d079b6c5cffd2`。大型 workload.parquet 按仓库规则留在本地；配置、来源和统计保存在 manifest.json。构建日志见 [容量初轮 build-16k.log](../../docs/EVIDENCE.md#archives)。预算及独立 LCP 核验见 [最终容量报告](../../docs/EVIDENCE.md#archives)。
