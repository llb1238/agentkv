# M0 问答：延迟、策略接口与工具间隙

实测时间、三次原值和边界定义见 [TTFT 分解](m0_ttft.md)，完整流程与代码索引见
[生命周期](m0_lifecycle.md)。以下接口以相邻 vLLM 工作树的固定 commit 和实验快照为准。

## 异步回载会多等一个调度步吗？

会。`get_num_new_matched_tokens` 返回 `(4096, True)` 后，
[`Scheduler.schedule`](patches/README.md#file-vllm-v1-core-sched-scheduler-py) 分配目的块，
设置 `WAITING_FOR_REMOTE_KVS`，跳过本轮 forward。
[`ActiveKVConnector.no_forward`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/kv_connector.py)
在无模型计算的步骤里继续提交/轮询传输。Worker 完成后，
`Scheduler._update_from_kv_xfer_finished` 仅登记 finished ID；下一次
`_handle_blocked_waiting_request → _update_waiting_for_remote_kv` 才恢复执行。

因此“异步等待全过程”和“完成上报后多等的下一步”是两个不同区间，不能把
全部等待称为一个调度步，也不能把轮询步数乘上正常 decode 步长。
原始日志分别有 `load_wait`、`load_blocked`、`worker_recv_done`、
`scheduler_recv_done`、`load_resume`，实测步号与时差列在 TTFT 文档。

## 是一次连续拷贝，还是 256 次，还是更多次？

本场景是一个 load job，256 个逻辑 KV 块，**9216 个拷贝描述符**：
36 层 × 256 块，每个描述符 65536 bytes；
9216 × 65536 = **603979776 bytes**，与每请求 metrics 的已完成传输量一致。

[`SingleDirectionOffloadingHandler._fill_direct_ops / transfer_async`](patches/README.md#file-vllm-v1-kv-offload-cpu-gpu-worker-py)
逐层生成源/目的指针及 sizes，并对整批调用一次 `self._swap_blocks_batch`。
本次 `copy_impl` 是 `_custom_ops.swap_blocks_batch`，没有选择小页 Triton 路径。
**这不等于一次 604 MB 的连续内存复制，也不等于 Python 循环调用 256 次 memcpy。**

[`csrc/cache_kernels.cu::swap_blocks_batch`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/csrc/cache_kernels.cu)
在支持 CUDA batch API 和有效 stream 时调用 `cuMemcpyBatchAsync`；默认不限制
每批描述符数，配置 `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS` 可拆批；不满足条件
时逐描述符调用 `cudaMemcpyAsync`。本实验没有给 C++ 分支新增 profiler/日志，
因而证据直接确认的是 Python 批次与描述符数；不能仅凭 Python 函数名断言
预编译扩展实际走了多少次驱动 API。

约 11 ms 来自 Week1 连续 pinned H2D 的约 55 GB/s：
603979776 / 55e9 × 1000 = **10.981 ms**。它是同等 payload 的带宽换算，
并非对真实碎片化回载的计时。本次真实 CUDA event 结果见 TTFT 文档。

CUDA event 从 start record 到 end record，可能包含 GPU 等待主机提交命令的
间隙；它不是纯 DMA 活跃时间。`copy_enqueue → copy_enqueued` 则是主机调用
区间，两者会重叠，**不能相加**。分解表使用互不重叠的 wall-clock 边界；
CUDA event 作为旁证单列。驱动内部批处理、传输重叠的精确比例需要 CUPTI 等
进一步归因，本次没有把差值直接编造成某个内部开销。

## 调度器查找 256 个块的 hash 花了多久？

分开看 GPU 前缀查找、外部 key 构造、CPU key 扫描。
[`Request.update_block_hashes`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/request.py) /
[`get_request_block_hasher`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py)
已经根据已知 token IDs 生成链式 hash；第二次 connector 查询不是重新读取
GPU KV 或对 604 MB 的内容做 hash。

`get_num_new_matched_tokens` 包括 `RequestOffloadState.update_offload_keys`、
外部查找、状态/指标更新；内部 `_maximal_prefix_lookup` 本次做 **257 次查询**，
即原 prompt 的 256 HIT，再遇到第 257 个 key MISS 后停止。日志中的
`cpu_hash_scan_result.checked=257 / hits=256` 直接确认了这一点。
GPU 命中场景则保留 257 个完整 GPU blocks / 4112 tokens；CPU 回载场景的
GPU 查找在首个 miss 处停止。各自的实际耗时在 TTFT 文档独立列出。

后续补充：[kv_hints 专项核查](kv_hints.md)确认官方信封已有引擎内透传，
但固定版本的 Python OpenAI HTTP 入口尚未贯通，M4 仍需入口适配与动作语义。

## 自定义驱逐策略的基类有哪些方法，能拿到请求吗？

基类为 [`CachePolicy`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/cpu/policies/base.py)。
构造器 `__init__(cache_capacity: int)` 收到的是 CPU chunk 容量。

| 方法 | 输入及返回 | 能否直接拿到请求上下文 |
| --- | --- | --- |
| `get(key)` | `OffloadKey → ChunkStatus 或 None` | 否 |
| `insert(key, chunk)` | key、分配的 CPU chunk | 否 |
| `remove(key)` | 删除失败 store 等条目 | 否 |
| `touch(keys, req_context)` | 一组 key、`ReqContext` | **是** |
| `on_store_miss(keys, req_context)` | 新 store misses；默认委托 `touch` | **是** |
| `on_request_finished(key_groups, insertion_only_keys, reused_keys, req_context)` | 各 KV group 的有序 key，以及只插入/实际复用的分类；默认逐组 `touch` | **是** |
| `evict(n, protected)` | 数量、受保护 key 集合；返回恰好 n 个 `(key, chunk)` 或 None | **否** |
| `clear()` | 清空含 ghost/adaptive state 的全部状态 | 否 |
| `mark_evictable(key)` | chunk ref 从占用变为 0；默认空实现 | 否 |
| `mark_non_evictable(key)` | chunk 从 0 变为传输占用；默认空实现 | 否 |

前七项中 `on_store_miss`、`on_request_finished` 是已有默认实现的扩展点；
其余上述抽象方法必须实现。后两个标记回调也有默认空实现。
`evict` 必须跳过 pinned/pending 或 protected 的条目；失败返回 None 时不能
留下部分驱逐。`ChunkStatus.ref_cnt=-1` 是 store pending，`>=0` 为 ready，
`>0` 表示作为 load 源仍被传输引用，`chunk_id` 是物理 CPU 槽。

[`ReqContext`](patches/README.md#file-vllm-v1-kv-offload-base-py) 含 `req_id`、
`kv_transfer_params`、`kv_hints`、`load_tier_filter`，并提供
`get_state/set_state`、`get_offload_key_position/set_offload_key_position`。
它不是整个 `Request`，没有任意读取 prompt、采样器等字段的保证。
`OffloadKey` 也不只是 hash：是 `block_hash + 4-byte big-endian group_id`，
应使用已有 make/get helpers 拆装，避免混淆 KV groups。

**M4 可以在策略的 request-aware 回调中解析并保存 hint，再由 `evict` 读取
策略自己的 registry。当前接口不需要为了传 hint 给 `evict` 增加 Request 参数。**
也不能因为 `insert` 没有 request 参数，就认定所有策略回调都只有 hash。
共享前缀同一 key 对应多个请求时，hint 合并与过期仍需 M4 明确定义。

加载外部策略已有入口：
[`CachePolicyFactory`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/cpu/policies/factory.py)
支持 `register_cache_policy(name, module_path, class_name)`，也支持配置
`eviction_policy` 为类名、`cache_policy_module_path` 为模块路径直接加载。
[`CPUOffloadingSpec`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/kv_offload/cpu/spec.py)
将这两个配置传入 manager。这里确认接入点，尚未实现 AgentKV 策略。

## 在哪里同时拿到请求的块 hash 列表和 kv_transfer_params？

最早可在调度器的 `Scheduler.add_request → connector.on_new_request` 中取得
`request.block_hashes` 与 `request.kv_transfer_params`。
[`OffloadingConnectorScheduler.on_new_request / _create_req_context`](patches/README.md#file-vllm-distributed-kv-transfer-kv-connector-v1-offloading-scheduler-py)
已经把参数和 hints 复制到 `ReqContext`，并让 `RequestOffloadState` 保留原始请求。

要将 **实际 offload key、group 和 token 位置** 登记到 HintRegistry，最直接的点是
同文件的 `RequestOffloadState.update_offload_keys`：这里同时能读到
`self.req.block_hashes`、`self.req.kv_transfer_params`，并刚好执行
`make_offload_key` 和 `req_context.set_offload_key_position`。
它由 `get_num_new_matched_tokens` 以及每步 `_update_req_states` 调用，
因此能增量覆盖 decode 新形成的完整块；只在新请求到达时登记一次不够。

对于只依赖已有 hint 与当前 key 的 M4，`on_store_miss(keys, req_context)` 已经
提供足够信息，可以通过外部策略完成 registry 登记；若确实要跟踪从未进入
store 的完整请求 key，才考虑在上述 connector 状态更新点接入。
registry 位于调度器进程即可，CPU policy 不在 GPU Worker 执行。

## 卸载是块写满就发生，还是请求结束才发生？

它是**按调度批次发现可卸载的完整 chunk，异步提交**，不是逐个 token 写满
就立即发一条拷贝，也不是等到 request_finished 才保存。
`_build_store_jobs` 使用 `num_computed_tokens + 本步 scheduled_tokens` 规划
可保存范围；`offload_prompt_only=true` 再把范围封顶到 prompt 长度。
Worker 的 `wait_for_save → prepare_store_kv` 把 store 留到下一步开始，
transfer stream 等待 GPU compute stream，保证读的是已写完的 KV。
实验中的完整 prompt 256 块在第一轮尚未结束时已在 CPU 变 ready，
具体 monotonic 时间和结束事件可在生命周期文档中核对。

**工具间隙开始不等于会话所有 KV 都已在 CPU。** 本配置只保存第一轮
4096-token prompt；decode 的完整块和不足一块的尾部不保证有 CPU 副本。
第二轮 `4157=4096+18+43`，CPU 恢复 4096 tokens，计算 61：18 个历史生成
IDs（其中最后采样的 EOS 尚未形成 KV）加 43 个工具/角色续接 tokens。
GPU 命中则能复用 4112 tokens，只算 45。

M5 需要分别记录“哪些前缀块已 ready”与“工具间隙何时开始”，不能用
request_finished 当作 CPU 保存完成屏障。若需要保存生成历史，可评估
`offload_prompt_only=false`；完整 chunk、最后一个尚未产生 KV 的 token、
不足整块尾部仍要分别处理，并验证当前 spec 的 partial-tail 能力。

## 本版本 v1 connector 的两组接口

接口声明是 [`KVConnectorBase_V1`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/base.py)，
CPU offload 的委托实现在 [`OffloadingConnector`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)。

| 调度侧 | Worker 侧 |
| --- | --- |
| `on_new_request(request)` | `register_kv_caches(kv_caches)` |
| `get_num_new_matched_tokens(request, num_computed_tokens)`：返回 `(int 或 None, bool)` | `start_load_kv(forward_context, **kwargs)` |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | `wait_for_layer_load(layer_name)` |
| `build_connector_meta(scheduler_output)` | `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)` |
| `update_connector_output(connector_output)` | `wait_for_save()` |
| `request_finished(request, block_ids)` / `request_finished_all_groups(...)` | `get_finished(finished_req_ids)` / `get_transfer_results(...)` |
| `get_external_cache_hit_sources(...)`、`take_events()` 等 | `build_connector_worker_meta()`、`handle_preemptions(...)` 等 |

`None` 表示匹配量暂未确定，调度器要后续重试；非零数量配合 True 则表示
加载在调度步之间异步进行。**OffloadingConnector 的 `wait_for_layer_load` 和
`save_kv_layer` 是空实现**，本路径走整批 metadata；名字叫 `wait_for_save` 的
方法也只是延后 store 提交，不是强制 GPU 同步等待。store 完成通过
`completed_jobs` 上报，load 还通过 `finished_recving` 恢复请求。
