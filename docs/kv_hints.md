# kv_hints 核查：对 M4 的影响

核查日期：2026-11-02；固定 vLLM：`5df63a7203cd`（0.32.0）。

**结论：它是官方面向 Agent/编排器的 KV Hints 通道，目前实现的是通用、类型化、
带版本的信封；不是已经定义工具间隙语义并自动执行的 Agent 缓存策略。**

- **来历与边界。** `git log -S kv_hints` 定位到 `986e217158`，即
  [PR #53423](https://github.com/vllm-project/vllm/pull/53423)，2026-09-21 合并。
  [RFC #53421](https://github.com/vllm-project/vllm/issues/53421)明确用它替代经
  `SamplingParams.extra_args`、`kv_transfer_params` 间接透传的原型。
  [作者在评审中的说明](https://github.com/vllm-project/vllm/pull/53423#discussion_r3946493789)
  确认 payload 故意开放，由动作实现方校验；合并信封不代表合并动作执行。

- **谁赋值、传到哪里。** 外部编排器/调用者提供，connector 不自动生成。
  [`KvHintsEnvelope`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_hints/protocol.py)包含
  `protocol_version/message_id/actions`；每个 action 含
  `action_id/action_type/action_version/payload`。
  `AsyncLLM → InputProcessor → EngineCoreRequest → Request → _create_req_context`
  将它传到 `ReqContext.kv_hints`，默认 None。

- **现有 HTTP 实验有入口缺口。** Python 原生 `AsyncLLM.generate/add_request`、
  `LLMEngine.add_request` 及 Rust gRPC 已有参数/透传；但固定版本的 Python
  `/v1/completions`、`/v1/chat/completions` 请求模型未声明它，serving 调用也未传入。
  顶层未知字段可以被接受，却不会进入引擎，**HTTP 成功不等于 hint 生效**。
  Rust 两条 HTTP 转换也仍赋值 None。PR 描述中的 HTTP 表述不能替代代码核验。
  OpenAI SDK 3.37.0 的标准请求类型也无此字段，它属于 vLLM 扩展。

- **谁消费。** 内置
  [LRU](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/cpu/policies/lru.py)与
  [ARC](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/cpu/policies/arc.py)均不读取 `kv_hints`，
  虽然回调已收到 `ReqContext`。
  [KVCR tier adapter](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/tiering/kvcr/manager.py)会在
  `on_new_request` 转交 `submit_hint`，结束时 `discard_hint`；解释和执行由 KVCR 负责。

- **M4 应调整为复用信封，补语义与入口。** 不必重造 envelope 或引擎内部透传；
  session、工具名、预计间隙、final 等仍需自定义 action payload、校验及 Registry，
  M5 的 CPU 间隙感知策略仍须实现。保持现有 HTTP 回放时，需小型入口适配；或暂沿用
  `kv_transfer_params.agent_hint`，在 AgentKV 侧映射到同一信封/动作语义，不能假设
  顶层 `kv_hints` 已直通。后续
  [RFC #59708](https://github.com/vllm-project/vllm/issues/59708)讨论
  COPY_TO/MOVE_TO/SET_PRIORITY、TTL 与分发执行；
  [RFC #57103](https://github.com/vllm-project/vllm/issues/57103)及其
  [草案 #62382](https://github.com/vllm-project/vllm/pull/62382)覆盖会话、控制与策略组合。
  这些是待讨论/未合并实现，不能当作当前稳定能力或另起同类协议的理由。

[核查证据](../results/kv-hints-audit-20261102)保存 PR/RFC 与评审快照、搜索结果、
源码定位、入口检查和已有透传测试结果；未编写策略代码。
