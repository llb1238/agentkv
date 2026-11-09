# 最终 vLLM 补丁

这里保存实际测量后端的补丁，不是早期的 `m5_reqcontext_full_prefix.proposed.patch` 提案。目标为 fork 的 `agentkv` 分支提交 `4749dd5961841243596559b10ba0a22c48980c45`；来源与逐文件指纹见[补丁清单](vllm-patches.json)。

| 文件 | 基线与用途 |
|---|---|
| [vllm-agentkv-final.patch](vllm-agentkv-final.patch) | 从上游基线 `5df63a7203cd` 恢复完整测量 fork；14个变更文件，包括默认关闭的观测、独立H2D微基准及完整前缀元数据 |
| [vllm-prefix-metadata.patch](vllm-prefix-metadata.patch) | 只包含最后一个元数据提交；基线为 `60da6135b0349fac50a051e9dcec2bd5b4514400`，2个实现文件与1个测试文件 |

元数据补丁共用key推导，为ReqContext补充prompt边界和完整前缀枚举，不改变默认缓存访问。已存档的后端单元验证为1395通过、10项原有跳过；另需Llama权重的实模型集成不计入成功，见[后端验证报告](../../results/m5-vllm-integration-20261104/backend-validation/README.md)。本次只做CPU补丁应用核验，没有重新运行GPU验证。

两份补丁均在独立临时目录通过 `git apply --check`，实际应用后所有变更文件逐字节等于目标提交；文件列表与SHA256见 [vllm-patches.json](vllm-patches.json)。原vLLM工作区保持不变。

```bash
# 在独立的vLLM检出中；这里只应用源码，不构建或启动GPU服务。
git checkout 5df63a7203cd
git apply --check ../agentkv/docs/patches/vllm-agentkv-final.patch
git apply ../agentkv/docs/patches/vllm-agentkv-final.patch
```

完整补丁与元数据单独补丁二选一，必须匹配各自基线；不要依次重复应用。原始源码的SPDX、署名及Apache-2.0声明保留。补丁只恢复代码树；不会创建原fork的提交历史，也不伪造原提交号。

## 修改文件与源码入口

下列文件相对上游基线有项目补丁，精简版源码链接指向本节，避免把上游版本当作测量版本。每个小节对应完整补丁中的同名文件；元数据子补丁只覆盖其清单中的三个文件。未修改源码使用[固定上游提交索引](upstream-source-index.json)。

<a id="file-gitignore"></a>

### `.gitignore`

本地实验产物的忽略规则。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-benchmarks-kernels-benchmark-pinned-h2d-py"></a>

### `benchmarks/kernels/benchmark_pinned_h2d.py`

独立的锁页H2D微基准；不属于上游基线文件。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-tests-v1-kv-connector-unit-offloading-connector-test-scheduler-py"></a>

### `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py`

完整前缀元数据、位置表与decode更新的回归测试。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-distributed-kv-transfer-kv-connector-v1-offloading-scheduler-py"></a>

### `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`

共用key推导并登记请求的完整前缀与位置。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-distributed-kv-transfer-kv-connector-v1-offloading-worker-py"></a>

### `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`

可选卸载搬运观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-core-block-pool-py"></a>

### `vllm/v1/core/block_pool.py`

可选块生命周期观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-core-kv-cache-manager-py"></a>

### `vllm/v1/core/kv_cache_manager.py`

可选KV分配与缓存观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-core-sched-scheduler-py"></a>

### `vllm/v1/core/sched/scheduler.py`

可选调度与生命周期观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-engine-output-processor-py"></a>

### `vllm/v1/engine/output_processor.py`

可选输出完成观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-kv-offload-base-py"></a>

### `vllm/v1/kv_offload/base.py`

ReqContext的prompt边界与完整前缀枚举接口。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-kv-offload-cpu-gpu-worker-py"></a>

### `vllm/v1/kv_offload/cpu/gpu_worker.py`

可选CPU与GPU卸载操作观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-kv-offload-cpu-manager-py"></a>

### `vllm/v1/kv_offload/cpu/manager.py`

可选CPU层状态与回调观测。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-kv-offload-trace-py"></a>

### `vllm/v1/kv_offload/trace.py`

默认关闭的观测入口。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。

<a id="file-vllm-v1-worker-gpu-worker-py"></a>

### `vllm/v1/worker/gpu_worker.py`

可选GPU worker观测接入。见[完整补丁](vllm-agentkv-final.patch)与[变更文件SHA](vllm-patches.json)。
