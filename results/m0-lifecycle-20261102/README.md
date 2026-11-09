# M0 实验证据（2026-11-02）

报告入口：[生命周期](../../docs/m0_lifecycle.md)、[TTFT](../../docs/m0_ttft.md)、[接口问答](../../docs/m0_qa.md)。

- 最终结果：GPU / CPU 回载 TTFT 中位数 29.954711 / 52.926117 ms，差 22.971406 ms。
- 原始证据：`traces/`（42163 事件）、`cpu/`（逐请求 metrics/SSE/IDs）、`fixture.json`、`server.log`、`client.log`。
- 重建分析：`../vllm/.venv/bin/python agentkv/analysis/m0_analyze.py --run-dir results/m0-lifecycle-20261102`。
- 新结果目录复现：`bash results/m0-lifecycle-20261102/run.sh ../agentkv-runs/m0-replay`，正常关闭后分析。
- `analysis.json`、`ttft_per_request.csv`、`ttft_table.md` 为派生结果；`trace-overhead-control.json` 为关闭 trace 的对照。
- `pilot*` 均为排除的诊断尝试，不进入最终统计；`pilot-buffer-no-flush` 缺少 EngineCore 日志。
- `instrumentation-measured.patch` 保存采集时的引擎修改；`instrumentation.patch` 保存最终引擎/工具修改（其后仅类型注解及分析校验调整）。
- `source-manifest.json` 保存最终源码 SHA256；`manifest.json` 保存本证据目录逐文件 SHA256。
- CPU manager/Worker 68、scheduler 403 项单元测试通过，trace 开启额外 51 项通过；`verification.json` 汇总检查。
- GPU 0 已释放，原 GPU 1 / 8000 服务健康；未提交 commit 或创建 PR。

上述命令从 AgentKV 仓库根目录运行；vLLM 引擎埋点仍位于相邻 vLLM 工作树。迁移前校验清单保存在 `manifest-before-migration.json`；原始日志、SSE、metrics、采集时 patch 未改写，历史命令中的旧路径用于保留采集来源。
