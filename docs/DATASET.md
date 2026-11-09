# 数据与第三方来源

## AgentX 数据

本项目使用 SemiAnalysisAI 发布的 [cc-traces-weka-062126](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729)。固定 revision 为 `23f152f6f0f9399a85901b89a6458def0ef16729`。数据集卡标注 **Apache-2.0**；本仓库保留了[固定数据集卡快照](EVIDENCE.md#other-evidence)和[HF来源元数据](../results/m1-agentx-data-20261102/hf-metadata.json)，不以当前网页可能变化的内容替代当时来源。

原始 `traces.jsonl` 为 1,847,151,435 bytes，SHA256 为 `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。原始载荷不纳入当前 Git 文件树，下载器在忽略目录 `data/raw/agentx/` 中保存并校验。字节数、revision、抽样与统计依据均见[数据审计报告](../results/m1-agentx-data-20261102/README.md)。

AgentKV 的修改包括：推断链关联、根会话抽样、上下文长度缩放、确定性合成 token、间隔截断、事件锚定和稳态人口。它们是衍生处理，不是原始文本、原始token或真实工具语义的恢复。衍生表与结果记录不改变上游数据的许可或署名要求；使用者应一并保留本说明、[NOTICE](../NOTICE)及许可证。

## AgentX harness 代码

链识别与 schema 来源为 [SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness/tree/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf)，固定提交 `e4bb2f835ffedbbe6b76252ea5c93caec68c5faf`，其 [LICENSE](https://github.com/SemiAnalysisAI/agentx-harness/blob/e4bb2f835ffedbbe6b76252ea5c93caec68c5faf/LICENSE) 为 Apache-2.0。使用过的代码及文档快照位于 `results/m1-agentx-data-20261102/source/`，原始来源/文件指纹保存在相邻 provenance 清单。快照保留原始署名，不将其署名归于 AgentKV。

## vLLM、模型与实验载荷

- vLLM 是独立依赖，本仓库使用固定 fork 进行实验，版本和补丁说明见[完整前缀适配](../results/m5-vllm-integration-20261104/README.md)。vLLM 代码遵循其自身的 [Apache-2.0 LICENSE](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/LICENSE)；本仓库历史源码快照保留各自原始声明。
- Qwen3-8B 模型权重和 tokenizer 不随仓库分发。请从[模型官方页面](https://huggingface.co/Qwen/Qwen3-8B)取得，并遵守对应版本的许可和使用要求。
- 合成负载大文件、完整请求 prompt 数组和完整回调载荷不作为当前源码树的常规分发内容。结果目录保留元数据和 SHA256；外置载荷的恢复与公开边界见[复现说明](REPRODUCING.md)和[公开前检查](PUBLICATION_CHECK.md)。

AgentKV 原创代码采用 [Apache-2.0](../LICENSE)，第三方材料保留上游许可；根许可证不替代第三方许可，也不授予上游商标权。
