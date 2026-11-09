# 原始证据包（暂不公开）

这是精简发布版的证据说明模板；`release --profile lean` 在发布副本中填入下列元数据，私有仓库模板不冒充已生成包。

**暂不公开，可应要求提供。** 原始实验输出、日志和外置输入不属于精简公开版的文件清单。证据包保留原始字节，不改写实验结果或原始路径；包内 `FILELIST.tsv` 逐文件列出路径、字节数和SHA256，旁边另附相同清单。

- 包名：`agentkv-evidence-20261109.tar.zst`
- 压缩包字节数：`1241959216`
- 压缩包SHA256：`e238ea49fe551ffa57649c56ca5b9ce65717035f18453a671e1608af26c246bd`
- 文件清单SHA256：`077170044cd6352ab6bada5d493220be43bbcdccdce5111456391c80334cc168`
- 导出所依据的私有源码提交：`fdc2e0ff43a3b9c92178c99df08d6178d26ab912`

| 内容类别 | 文件数 | 原始字节数（未压缩） |
|---|---:|---:|
| archives | 389 | 123,274,157 |
| checkpoints | 3 | 21,075,942 |
| community-originals | 13 | 175,853 |
| external-inputs | 3 | 1,997,595,494 |
| other-evidence | 107 | 7,838,692 |
| raw-traces | 24 | 303,653,937 |
| requests | 1033 | 1,180,053,329 |
| runtime-logs | 565 | 721,783,994 |

下面的小节也接收精简版中指向已排除文件的链接。链接改写只发生在发布副本，映射保存在其 `RELEASE_MANIFEST.json` 和 `results/relocations.json` 中；原版文档及原版迁移清单随证据包保留。

## raw-traces

完整回调序列、逐回调二进制记录及压缩流，用于决策一致性和性能回放。精简版只保留测试代码与报告；获取原始流前不能声称重跑完整实录。

## requests

逐请求Parquet、逐请求重算归因、决策和事件循环延迟序列。`final_report` 的精简模式使用保留的 `e1-analysis.json` 重算表格，不能替代逐请求原始正确性核验。

## runtime-logs

轻量观测JSON、服务端日志、指标采样、GPU dmon及性能剖析产物。精简版保留原图及汇总，TTFT CDF和队列图不由聚合分位数重新构造。

## checkpoints

运行检查点、attempt选择和恢复状态。精简版不具备断点续跑所需的完整运行目录。

## archives

被取代、失败或开发阶段的 `results/archive/` 原目录，包括报告、图和原始产物。排除归档不表示删除反面结果；最终报告仍说明其作用，并由证据包保留。

## other-evidence

其他未进入白名单的产物、超出汇总大小阈值的JSON/CSV、运行时源码快照，以及发布时修改链接的Markdown原件。修改后的展示文件不冒充生产者记录的原SHA。

## community-originals

`kv-hints-audit-20261102` 中保存的PR/RFC正文与社区评审原始JSON，按原字节和SHA保留。公开版提供[原帖索引与项目摘要](../results/kv-hints-audit-20261102/community-summary.md)，不转存社区原文；摘要不是参与者授权或上游实现状态的保证。

## external-inputs

外置清单中本地可取得的AgentX原始trace、32K/16K workload和完整回调流会纳入证据包；不存在的外置输入会在发布清单中单列，不能据此宣称随包提供。模型权重、独立vLLM源码检出及未登记的临时文件不打包，须按[数据来源与许可](DATASET.md)和[后端补丁](patches/README.md)另行取得。

相邻vLLM目录中的历史benchmark输出是本机实验记录，不是上游源码；其[清单](../results/week1-local-evidence/README.md)保留来源与指纹，这些外部目录不自动随本包提供。可公开的上游源码链接使用固定提交永久链接；项目修改的文件链接到对应补丁小节。

`run.json` 等生产者元数据中的 `/home/llb/...` 是**原实验环境路径**，不表示发布机器的当前路径，也不是可移植命令。私有原始字节不改写；lean仅对第三方进程身份、环境路径和主机名脱敏，原SHA、发布SHA及原因见发布清单。复现布局见[REPRODUCING](REPRODUCING.md)。

本次复用既有证据包，其字节、SHA和生产者提交保持不变。新增脱敏文件的原件继续保存在私有仓库，原SHA、发布SHA和原因见发布清单；不宣称旧包包含本次新增的源码或文档版本。
