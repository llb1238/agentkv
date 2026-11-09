# 仓库结构与结果索引维护验证

没有启动 GPU 测量，也没有重新生成性能结果。

- 独立目录迁移提交：`8eeb34f`；[layout-tests.log](../../docs/EVIDENCE.md#runtime-logs) 为 92 项通过。
- 干净工作区、提交号来源、红色调试警告、无源码副本、CLI、归档和索引的完整验证：[final-tests.log](../../docs/EVIDENCE.md#runtime-logs)，98 项通过。
- [integrity.json](integrity.json)：对照原始提交 `7d787ef`，514 个历史文件保持字节不变；17 个 README 只做登记的展示迁移；4 个当前审计辅助脚本更新入口。归档 14 个目录，索引覆盖 27 个实验目录，检查到的本地文档链接无断链。
- 迁移后重新独立审计：[两层模拟](../../docs/EVIDENCE.md#runtime-logs)、[调度压力探针](../../docs/EVIDENCE.md#runtime-logs)、[稳态人口](../../docs/EVIDENCE.md#runtime-logs)，均通过。

来源约定与每个结果的状态见 [结果索引](../INDEX.md)。旧报告字段、绝对路径与源码快照是历史证据，保留原样；新运行只记提交号，不再复制实现代码。
