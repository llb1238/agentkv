# M5：真实CachePolicy回调可行性验证（先于策略适配）

按用户要求，先运行5分钟LRU等价探针；本轮不提前实现HintAware vLLM适配或10分钟冒烟。如果发现明显的回调语义差异，停止后续步骤，提交并报告证据，由用户决定如何调整。

## 固定运行

当前干净vLLM工作树独立记录提交与接口文件指纹；不改vLLM源码。config-11：Qwen3-8B、16K、8槽位、GPU182880/CPU274320 token、LRU、perfect hint、λ=0.7根/分钟、人口种子0、稳态初始化、T=300秒、warmup=0（观察整个可行性窗口，不作稳态性能结论）。使用空闲显存足够的GPU1，显存共享，记录两张卡dmon，不终止或修改其他进程。新服务仅监听127.0.0.1，独占新的本地端口。

探针直接继承当前vLLM内置LRU，逐方法调用super，不改变key、顺序、pin、排序、驱逐结果。每个CachePolicy方法入口时刻用monotonic_ns记录；get/insert等高频操作写紧凑二进制，含ReqContext的回调另记JSONL。on_request_finished保存原始各组key顺序、ReqContext位置、缓存ref_cnt及insertion/reuse标记。分别记录direct kv_hints与kv_transfer_params.kv_hints的存在/有效性、选择来源和message_id；以message_id按请求去重，不能把多次回调当多次hint。记录seen-any、on_store_miss、touch、on_request_finished各自覆盖数，与客户端实际发送数核对。

复用已有只读服务观测，另记录Scheduler首次释放已完成请求的入口时刻，作为服务端生成结束边界；记录接收、首次调度、prefill完成，以及客户端send/end时刻。所有时钟为同一主机CLOCK_MONOTONIC。按请求关联时以有效hint.message_id为准，并核对服务端request_id。输出第一次/最后一次store_miss及finished相对服务器首次调度、生成结束和客户端结束的分布；同时报告回调次数、缺回调请求、key位置缺失与非单调数、finished时未ready块数。

## 与模拟器比较及停止门槛

模拟器约定：on_request在实际admission时一次观察完整请求块；所有新CPU副本在请求完成时一起写入并视为ready；写入后调用on_finished。真实接口若仅报告新增块、同请求多次增量store、在执行中插入/可用，或请求最终回调缺失/晚于其他阶段，均需明确量化，不能将它们静默映射成模拟器事件。key顺序按各组的真实位置检查，也报告扁平化顺序，不能先排序再宣称原始有序。

若实测出现hint不等量、key原始顺序异常、增量/提前store或最终未ready等明显语义差异，停止步骤2–4。时间差以实际分布报告；不通过切换hint档位、改容量或改变负载绕开。启动失败或请求失败也保留原始记录，不能当作可行性通过。

正式运行要求两个仓库均干净，run.json代码来源只记提交；不复制实现到结果目录。提交探针与测试后再测量，产物暂存Git外，发布时更新results/INDEX.md。GPU耗时只用于可行性，不据此声称新策略开销已满足调度要求。
