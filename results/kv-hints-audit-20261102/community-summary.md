# 社区讨论：原帖索引与项目结论

本页概括2026-11-02保存的核查材料，状态仅指当时快照。下表是AgentKV作者的摘要，不转贴社区正文或评审原文；当前进展以原帖为准。原始JSON及其SHA保留在私有证据包的[社区材料](../../docs/EVIDENCE.md#community-originals)中。

| 原帖 | 摘要与对本项目的结论 |
|---|---|
| [RFC #51428](https://github.com/vllm-project/vllm/issues/51428) | 路由器与引擎共同掌握会话复用信息，提出由路由器发出窄接口提示。AgentKV复用提示通道，在CPU层验证间隔信息的作用。 |
| [RFC #52113](https://github.com/vllm-project/vllm/issues/52113) | 将暂停、共享前缀、分支完成和恢复意图交给缓存管理。提示表达意图，具体保留与资源分配仍由引擎决定。 |
| [RFC #53421](https://github.com/vllm-project/vllm/issues/53421) | 定义编排器无关、带类型和版本的信封。传输协议不等于动作执行或缓存策略；本项目为间隔与final定义自己的payload。 |
| [PR #53423](https://github.com/vllm-project/vllm/pull/53423) | 加入信封和引擎请求透传。固定版本中Python HTTP入口是否消费字段仍须逐路径核查，HTTP成功不能代替hint对账。 |
| [PR #53423评审](https://github.com/vllm-project/vllm/pull/53423#discussion_r3946493789) | 动作执行方负责payload语义与校验。评审还涉及信任边界、重试生命周期、跨语言schema一致性及大整数精度；AgentKV不把开放payload当作免校验接口。 |
| [PR #53624](https://github.com/vllm-project/vllm/pull/53624) | 接入可选的KVCR次级层适配器，由外部后端解释路由与策略。它不代表vLLM内置LRU已经消费间隔hint。 |
| [RFC #57103](https://github.com/vllm-project/vllm/issues/57103) | 讨论会话生命周期、可组合策略、资源审批和实际完成契约。AgentKV的CPU驱逐评估只覆盖其中一个策略环节。 |
| [PR #62382](https://github.com/vllm-project/vllm/pull/62382) | RFC #57103的草案参考实现，不构成整个提案已被接受或合并的证据；项目据此核查重叠范围，未据草案宣称稳定能力。 |
| [RFC #59708](https://github.com/vllm-project/vllm/issues/59708) | 讨论COPY_TO、MOVE_TO、SET_PRIORITY及执行边界，允许G2+后端保有自己的语义。AgentKV不另造一套通用传输信封。 |
| [RFC #59708讨论](https://github.com/vllm-project/vllm/issues/59708#issuecomment-5993317583) | 有参与者建议先做内存生命周期、完成与引用释放契约测试，另有参与者报告分支上的编排器实现；这些反馈不等于上游功能已经落地。 |
| [PR #58198](https://github.com/vllm-project/vllm/pull/58198) | 提出不依赖新请求的CPU→GPU预取执行原语。AgentKV当前实验评估CPU驱逐，不将他人的预取测量计入自身收益。 |
| [PR #59714](https://github.com/vllm-project/vllm/pull/59714) | 讨论带claim/revision和TTL的G1保留优先级及生命周期分发，与本项目的CPU层剩余寿命打分属于不同作用层。 |

RFC #53421的评论快照为空，不能据此推断当前没有讨论。原始正文、13条PR评审记录、RFC评论以及API状态元数据均保留在证据包；公开版仅保留本页索引、项目核查摘要和原始指纹。
