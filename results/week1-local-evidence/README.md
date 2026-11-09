# Week 1 本机原始证据索引

原始文件仍在相邻 vLLM 仓库的 `benchmarks/results/`，未移动或删除。为避免把实验日志、临时源码快照及历史 patch 混入引擎提交，Git 只保存本目录的字节数与 SHA256 清单；原始目录按本机实验输出忽略。

这些文件支撑现有 [Week 1 报告](../../docs/WEEK1.md) 和 [实验日志](../../docs/EXPERIMENT_LOG.md)。在其他机器上只克隆仓库不能恢复原始证据，需要另行复制本机目录并按 `manifest.json` 校验。可复用锁页 H2D 微基准单独保存在 `../vllm/benchmarks/kernels/benchmark_pinned_h2d.py`。
