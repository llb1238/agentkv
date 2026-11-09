# #62382 策略重叠简验

2026-11-02；核查 PR head `7b01911392a7`。结论：**已有具体的会话感知策略，
并非仅有框架；但尚未覆盖本项目计划的 CPU 预测驱逐规则。**
[PR #62382](https://github.com/vllm-project/vllm/pull/62382)仍为 open draft、未合并，
正文承认与 main 冲突。以下是固定源码核验，未运行该分支，作者报告的测试和性能数字未独立复现。

| 核查项 | 实际实现 | 与 AgentKV 的关系 |
| --- | --- | --- |
| 按会话保留 | `SessionTarget` 解析会话/分支；`RetentionDeclarations.retain` 支持独立 priority、TTL、snapshot/follow_growth；`RetentionPolicy` 将 importance/resume 信号变成 GPU `SetPriority` | 会话索引、TTL 保留与 M4/M5 的一部分明显重叠；是驱逐优先级偏置，不是硬 pin |
| 按会话驱逐 | `EvictPayload` 接受会话目标；执行器对解析出的 GPU key 调用 `evict_unreferenced`，区分 busy/missing/partial 等结果 | 已有真实执行路径；release/finish_task 清理声明不等于自动删除所有 KV |
| 工具间隙策略 | `ToolWaitPolicy` 接收 tool_wait、resume、resume_prediction；结合压力、预计暂停时长及观测到的存取耗时，决定 GPU→CPU/Mooncake 下放，并按预计恢复时间减 lead 提前回载 | 与间隙感知放置/预取直接重叠，不能再把这部分描述成该框架尚缺的用例 |
| CPU 满容量选谁驱逐 | 本次未发现按预测下次访问时间排列 CPU 牺牲块的策略；retain 拒绝非 GPU tier，显式 evict 也仅执行 GPU；PR 明确冷层保留/删除不支持 | 本项目“CPU 层预测版 Belady、共享前缀取最早复用时间、预测逾期衰减”仍是待验证的差异点，不能据此宣称全局新颖性 |

关键源码（固定 commit）：
[会话目标](https://github.com/zhhangBian/vllm-note/blob/7b01911392a707e5f9be8ebe97902fc92529dfa7/vllm/v1/kv_hints/native.py#L44)、
[RetentionPolicy](https://github.com/zhhangBian/vllm-note/blob/7b01911392a707e5f9be8ebe97902fc92529dfa7/vllm/v1/kv_hints/policy.py#L154)、
[ToolWaitPolicy](https://github.com/zhhangBian/vllm-note/blob/7b01911392a707e5f9be8ebe97902fc92529dfa7/vllm/v1/kv_hints/policy.py#L371)、
[GPU 驱逐执行](https://github.com/zhhangBian/vllm-note/blob/7b01911392a707e5f9be8ebe97902fc92529dfa7/vllm/v1/kv_hints/resources.py#L387)。
另核对了共享会话声明独立到期、follow-growth 不续期以及请求结束后 priority 生效的测试代码，
不只依据 README 或类名判断。

**方向建议：优先转评测与对比，CPU 预测驱逐作为候选增量。**
先用现有 2×6000 / 6×6000 基线验证该 PR 的 GPU 保留、工具等待策略在相同容量下能解决多少问题，
再与 CPU LRU/ARC 和未来的 CPU Oracle 对比，观察 CPU 命中、重算、传输量及 TTFT。
目前我们只有 Week1/M0 的受控场景，尚无完整闭环 E0 或 Oracle 结果，需补齐后再参与讨论。
暂不重造它已有的会话控制框架，也不因草案尚未合并就认定不存在重复工作。

[源码摘录与状态快照](../results/pr62382-overlap-20261102/evidence.json)已归档。
此次仅核查，未修改策略、切换 vLLM 分支或向 GitHub 发言。
