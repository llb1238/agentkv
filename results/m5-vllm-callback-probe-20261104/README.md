# vLLM CachePolicy：LRU等价探针（5分钟）

代码 `a4a245634ec75e7db358386ad3554c4d4dfe7e7b`；vLLM `60da6135b0349fac50a051e9dcec2bd5b4514400`。

config-11、λ=0.7、perfect、稳态初始化、人口种子0、T=300秒、预热0；用于回调可行性，不能当作稳态性能测量。GPU1显存共享，两卡dmon另存。

## Hint与key顺序

实际发送带hint请求 1292；前端有效hint 1292；插件任意回调收到的有效hint（按message_id去重）1292；finished回调覆盖 1292。成功 1292，失败 0。

原始key groups：1292个finished回调，缺失位置0，组内非单调0，扁平化非单调0，多组请求0。探针没有先排序再检查。

## 回调阶段

多次store_miss的请求111；无store_miss请求184；store_miss总块数不等于完整prompt整块数的请求1081。

首次store准备早于服务端生成结束的请求1108。finished时块观测：{'ready_unpinned': 544193, 'not_ready': 6082, 'absent': 1746}。

| 时间差（ms，前者减后者） | n | P50 | P90 | P99 | min/max |
|---|---:|---:|---:|---:|---:|
| first_store_miss_minus_server_start_ms | 1108 | 0.659 | 1.496 | 5.232 | 0.093/23.089 |
| first_store_miss_minus_server_end_ms | 1108 | -189.582 | -40.036 | -20.120 | -11949.276/-15.373 |
| first_store_miss_minus_client_end_ms | 1108 | -191.496 | -42.890 | -28.222 | -11952.390/-19.324 |
| last_store_miss_minus_server_start_ms | 1108 | 0.690 | 16.368 | 39.208 | 0.093/529.099 |
| last_store_miss_minus_server_end_ms | 1108 | -189.582 | -40.036 | -20.120 | -11949.276/-15.373 |
| last_store_miss_minus_client_end_ms | 1108 | -191.496 | -42.890 | -28.222 | -11952.390/-19.324 |
| policy_finished_minus_server_start_ms | 1292 | 153.418 | 996.673 | 4141.566 | 12.052/11953.461 |
| policy_finished_minus_server_end_ms | 1292 | 0.941 | 2.445 | 8.422 | 0.249/15.785 |
| policy_finished_minus_client_end_ms | 1292 | -1.452 | 0.306 | 3.027 | -34.895/11.700 |

开始=服务端首次SCHEDULED，结束=Scheduler._free_request首次入口（生成已完成）；客户端结束是流式HTTP关闭。时钟为同主机monotonic_ns。负数表示早于边界。

## 门禁结论

停止后续步骤2–4：True。
- incremental on_store_miss rather than single full-prefix observation at admission
- on_store_miss observes new store candidates rather than every full request block
- store preparation occurs during request execution rather than only at completion
- CPU chunks can still be not-ready at on_request_finished
- policy stream was not finalized; observation completeness not established

模拟器是在admission时一次登记整个请求；完成时统一写CPU镜像并视为ready，然后finished。真实事件按上表报告，不自动重解释为模拟事件；策略实现/10分钟冒烟需用户决定后再继续。

[分析](probe-analysis.json) · [逐请求关联](../../docs/EVIDENCE.md#requests) · [上下文回调](../../docs/EVIDENCE.md#requests) · [客户端元数据](../../docs/EVIDENCE.md#requests) · [run.json](run.json)

原始完整requests.parquet留在本机（含原prompt数组，不纳入Git）；追踪副本request-metadata.parquet只删除该大列。原始policy JSONL另以gzip无损保存，二进制标量回调格式<QB：monotonic_ns uint64、方法编号uint8；方法顺序见各日志probe_open。所有源码仅以Git提交定位，未复制到结果目录。

## 关键差异与观测边界（复核）

hint通路与顺序可用，但不能把真实回调直接当作当前模拟器事件。按用户要求停在步骤1：未实现vllm_plugin.py，未开展HintAware双适配一致性测试，也未开展10分钟GPU冒烟。

- direct kv_hints出现/有效：0/0次；nested出现/有效：3736/3736次（按上下文回调计数）。插件按message_id去重收到1292个有效hint，已发送1292个。
- 1286个key group；原始位置非单调0，缺失0；40个请求的finished key数不等于完整prompt整块数，其中6个请求列表为空。sorted并不代表complete。
- 35个请求在finished时仍有未ready CPU块（6082个块观测），4个请求有已不在CPU字典中的块（1746个观测）。不能把finished视为全部新CPU副本同时可读；absent本身不单独证明是驱逐还是存储失败。

| 事件 | 当前模拟器 | 此次真实观测 |
|---|---|---|
| 首次观察 | admission时一次给完整prefix | store_miss为新增候选；184请求无此回调，111请求分多次；首次较首次调度P50晚0.659ms |
| CPU写入准备 | 请求结束才统一写镜像 | 1108请求在生成结束前就准备store；首个store减生成结束P50=-189.582ms |
| finished | 全量请求prefix且已ready | 减生成结束P50=0.941ms、P99=8.422ms；有缺块/未ready |
| 缓存访问 | 完整prompt统一更新CPU recency | 实际finished可能只含已观察到的CPU访问/存储块 |

本次记录到1209713个二进制方法入口；关键上下文事件逐条与二进制时间点匹配。观测到的insert−evict−remove=17145，CPU容量=17145；mark_evictable−insert−mark_non_evictable=0。

这些计数自洽不能替代退出footer：EngineCore拆除时没有保存probe_close，不能断言进程拆除尾部所有标量回调都已记录。原始不完整标记保留，不补造footer。全部关键上下文回调按调用同步flush，其请求覆盖和时间点另作核验。

建议下一步先修正模拟器的分段store、异步ready和实际回调key覆盖范围，保留冻结评分公式；校验后再决定是否需要扩展生命周期观测或调整适配方式。这是建议，未自动执行。原冻结策略的CPU模拟结果不应被当作已在真实回调下验证。

[独立复核](verification.json) · [测试](tests.json) · [预注册](../../docs/m5_vllm_callback_probe_design.md)

## 复现与来源

预注册提交 `4d61054`；GPU测量提交 `a4a2456`；CPU独立复核提交 `b063dea249f39b57f0cd756b82f0a7d5b8ddb00e`。探针测量时178项测试通过，发布复核时[180项测试通过](final-tests.json)，冻结评分及校准表未改。只读vLLM工作区提交 `60da6135b0349fac50a051e9dcec2bd5b4514400`，接口文件指纹见 [probe-run.json](probe-run.json)。GPU1为显存/算力共享；本次自建服务已停止，外部进程未干预。

CPU复核（不启动服务或GPU）：

```bash
python -m agentkv.analysis.callback_probe_report --run-dir results/m5-vllm-callback-probe-20261104
```

重新测量入口如下；会启动本机GPU服务，当前停止门禁下**未再次执行**：

```bash
python -m agentkv.bench.callback_probe \
  --output-dir ../agentkv-runs/agentkv-callback-probe-rerun \
  --gpu-index 1 --port 8014
```

[artifacts.json](artifacts.json)登记产物SHA256、本机保留大文件指纹及JSONL的无损压缩对应。`verification.json`来自紧凑请求元数据、原始回调日志和二进制入口日志的独立核验；不重写初次分析或退出完整性标记。

本机以上命令使用相邻vLLM虚拟环境；测量与测试均设置 `PYTHONPATH=.:../agentkv-runs/agentkv-parquet-tools:../agentkv-runs/agentkv-week1-plot-tools`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`，解释器为 `../vllm/.venv/bin/python`。复现需安装本仓库Python依赖及清单对应vLLM版本，不能依赖另一个环境的同名vllm。
