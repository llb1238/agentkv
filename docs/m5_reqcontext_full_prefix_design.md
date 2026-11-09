# M5：完整前缀归属的 ReqContext 接口核查与待确认方案

状态：第一步 CPU 核查完成；需要一个小的 vLLM 元数据接口改动，按用户要求等待确认。提案未应用到 vLLM，未实现适配层、未重新采集回调、未启动 GPU。

## 固定版本与结论

核查版本为 `60da6135b0349fac50a051e9dcec2bd5b4514400`，vLLM 工作区干净。[源码 SHA256 与提案指纹](m5_reqcontext_source_audit.json)保留对应字节来源。

| 核查问题 | 固定版本中的行为 |
|---|---|
| `update_offload_keys()` 是否包括 GPU 命中前缀？ | **包括。** 按 `Request.block_hashes` 从已登记数量继续遍历；不以 GPU 命中数或 `next_stored_chunk_idx` 为起点，逐块调用 `set_offload_key_position`。 |
| 位置表是否恰好是 prompt 的完整块？ | **不保证。** 同一函数也会登记 decode 新增的完整块，未按 `num_prompt_tokens` 截断。`offload_prompt_only` 限制存储范围，没有限制此表。 |
| 三个回调能否取得这些位置？ | 常规 tracked 路径收到同一个 ReqContext，可读取私有 `_offload_key_positions` 并排序；公开接口只有“已知 key 查位置”，没有枚举方法，也没有 prompt 长度。不能仅凭 ReqContext 可靠排除 decode 块。 |
| finished 是否还有例外？ | 有。`request_finished()` 的 untracked 分支新建 ReqContext，未填位置即转发给 manager；即使 Request 有完整 block hashes，该位置表仍为空。 |

常规新请求在 `Scheduler.add_request()` 中调用 `connector.on_new_request()`，所以**不能把所有 GPU 命中请求都归为 untracked**。此前40个请求的 GPU 前缀被省略，是 manager 的“访问列表”与请求完整 key 集合不同；不证明这些前缀在私有位置表里也不存在。untracked 是需要覆盖的额外接口边界。

采用做法：**补充 ReqContext 的 prompt 边界与公开前缀枚举接口，再由 AgentKV 适配层独立登记完整前缀归属。** 本文明确该选择，不假设私有位置表长度天然等于 prompt 完整块数。

## 调用链证据

以下行号对应上述固定提交：

- `vllm/v1/request.py:225–230`：初始化 `block_hashes` 并更新请求 hash；`update_block_hashes()` 也用于生成阶段。
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:421–438`：`RequestOffloadState.update_offload_keys()` 遍历完整 `req.block_hashes` 并登记 end-token 位置。
- 同文件 `:549–562`：`_create_req_context()` 只复制 request ID、hint、transfer 参数与 tier filter，未提供 prompt 长度或填充 key。
- 同文件 `:1034–1044` 与 `vllm/v1/core/sched/scheduler.py:2663`：新请求先建立 tracked state。
- offloading scheduler `:1085`、`:1306`、`:2062`：lookup、调度状态更新和 tracked finished 路径更新完整位置表。
- 同文件 `:1877`、`:1908`、`:1922`：先更新位置，再创建 store 任务，最后通知 manager finished。
- `vllm/v1/kv_offload/cpu/manager.py:296`、`:239`、`:432`：分别转发 `on_store_miss`、`touch`、`on_request_finished`，使用原 ReqContext；finished 的 `key_groups` 只覆盖 manager 记录过的访问。
- offloading scheduler `:2050–2056`：untracked finished 新建空位置表并通知 manager。
- `vllm/v1/kv_offload/base.py:91–117`：ReqContext 的字段与仅支持按 key 查询的现有公开方法。

## CPU 验证

[固定版本测试](../tests/test_reqcontext_fixed_version.py)直接执行当前 vLLM 的 `update_offload_keys()` 和 untracked `request_finished()`，不加载模型或启动 CUDA：

1. 两个块都设为命中，`next_stored_chunk_idx=2`、`num_hit_chunks=2`；仍登记位置 `[16, 32]`，重复更新不重复追加。
2. prompt 长度32，追加一个 decode block hash 后位置表为 `[16, 32, 48]`；ReqContext 没有 `num_prompt_tokens` 或公开前缀枚举方法。
3. 相同请求已有完整 block hashes，但 untracked finished 收到的位置表为 `{}`。

这些是旧版本行为的审计测试，不是新接口的验收测试。更换后端提交后会跳过，届时须以新接口测试代替。全仓 **190 passed**，14条第三方弃用警告；[测试日志](validation/m5_reqcontext_audit_tests.log)。拟议补丁通过 Python 语法解析与 `git apply --check`，**未应用或执行拟议实现**，不把静态检查当作运行验收。

## 请求确认的具体改动

完整的[待审查补丁](patches/m5_reqcontext_full_prefix.proposed.patch)针对固定提交，**2个实现文件，新增40行、删除4行**。这里只保存补丁提案，vLLM 工作区保持干净。

### 1. `vllm/v1/kv_offload/base.py`：ReqContext 元数据

- 增加可选字段 `num_prompt_tokens: int | None = None`，保留现有构造调用。
- 增加 `get_offload_prefix_keys() -> tuple[OffloadKey, ...]`，按 `(end_token, key)` 稳定排序，只返回 `0 < end_token ≤ num_prompt_tokens` 的 key。
- 没有 prompt 边界的旧式/手工 context 调用新接口时明确报错；不静默返回可能混有 decode 的列表。
- 现有 `get_offload_key_position()`、state、hint 字段及方法保持原契约。

### 2. `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`：填充元数据

- `_create_req_context()` 接收 group 配置，记录 prompt 长度，按各组 `hashes_per_chunk` / `tokens_per_chunk` 填充当前完整 prompt hashes；不执行缓存 lookup、touch、分配或存储。
- 新请求及 untracked finished 的两个创建点均传入 group 配置，因此无需成功 offload 也能登记 GPU 命中的前缀。
- `update_offload_keys()` 同步 prompt 边界，原有增量登记逻辑保留。decode key 即使继续进入私有表，公开前缀接口也会排除它们。

默认缓存语义预期不变：不修改 manager 的访问列表、LRU 顺序、分配、搬运、ref_cnt、protected、驱逐返回值或调度时机；新增的是 key 所属请求的元数据。创建 context 时遍历已有 prompt hashes 会增加少量 CPU 工作，需实测，不能宣称零开销。

确认后，vLLM 改动与对应后端测试将作为**独立 Git commit**，不会混入 AgentKV 的策略提交。后端测试须覆盖 GPU 命中前缀、decode 排除、untracked finished、短于一块、深度排序及旧 LRU 驱逐/命中行为不变。

## 确认后的实现与验收顺序

1. 实施并单独提交上述 vLLM 元数据改动，记录新提交及构建/运行来源。
2. AgentKV 适配层在三个上下文回调中调用完整前缀接口，仅补充归属与深度；真实 CPU 访问/recency 仍由原回调参数决定。冻结评分公式和校准表保持原字节。父 final / 子 GPU 命中反例必须成为通过的回归测试。
3. config-11、perfect、λ=0.7、种子0、300秒、稳态初始化、预热0，重新采集完整回调。全部参数、hint、key、chunk/ref_cnt、protected 与返回值保留；每个请求按 prompt 边界核对深度与数量。
4. 固定模型为 Qwen3-8B、16-token 块、单个完整注意力组，所以前缀数量必须为 `floor(prompt_len/16)`。多组或部分边界 key 需要按各组单独核对，不把本次等式推广到其他模型。截断/无效/提前取消且未构成成功请求的情况单列，不能强行补造 key。
5. 真实序列验证五项不变量和每次驱逐时的归属比例。LRU 记录用于核对原行为；直接将 LRU eviction 结果固定后，不能冒称得到 HintAware 的闭环缓存结果。若候选策略改变 victim，回放工具必须明确处理由此改变的存储/引用状态，不能通过跳过重复 insert 或私自补删块消除异常。
6. 通过后继续双适配一致性与10分钟 HintAware 冒烟；记录有效 hint 对账、归属比例、evict P50/P99 和完整退出 footer。沿用已实现的120秒 shutdown timeout、135秒自建服务宽限和退出前 checkpoint；checkpoint 不能替代 close footer。

按照用户本次最后一条指示，本轮停止在改动方案确认处；没有修改 vLLM，没有运行第2–4步。
