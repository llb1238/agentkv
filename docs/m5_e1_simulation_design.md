# M5 独立策略与 E1 模拟预注册

本节在所有 E1 正式模拟前提交；后续结果另写结果目录，不更改以下规则来追求收益。仅使用CPU，不启动、查询或占用GPU，不加载vLLM。

## 接口及信息边界

`agentkv.policy.hint_aware.HintAwarePolicy(cache_capacity, *, clock=...)` 采用固定版本vLLM CachePolicy的duck-typed接口，包含get/insert/remove/clear、touch(keys, req_context)、on_store_miss(keys, req_context)、on_request_finished(key_groups, insertion_only_keys, reused_keys, req_context)、evict(n, protected)及pin/unpin通知。不import vLLM。参考接口为相邻仓库 `vllm/v1/kv_offload/cpu/policies/base.py`。

按优先级读取req_context.kv_hints，其次kv_transfer_params.kv_hints；只消费agentkv.tool_gap v1，未知action忽略，格式错误告警并退化LRU。payload.session_id标识实例内的链，parent_session标识实例；无hint时可从上下文显式instance_id/chain_id登记归属。块深度优先使用get_offload_key_position，结束回调可用完整prefix顺序补全。块记住所观察到、尚驻留的链引用，不使用后续prompt或未来请求信息。共享块取所有拥有者的最早预测；final逻辑上不再提供活跃引用，但保留终止标记，以区分“所有引用已结束”和“没有hint”。

## 排序与鲁棒性

时钟单位为秒，使用注入时钟；hint单位毫秒。完成时令d=now+expected_gap_ms/1000，m=gap_p90_ms/1000。

每个拥有者给出的预计下次使用时间：已返回且请求正在执行为now；等待为d；final为正无穷；没有hint为 `now + (now − 块最近访问时刻)`。相同评分按逻辑LRU秩破同分，所以全部缺hint时严格等于LRU（包括同一时刻的触达顺序）。共享块取最小评分。超过d+m的等待链评分为 `d + 2×max(0, now−d−m)`，从阈值开始连续降低保留优先级；不将已过期预测时间永远当作“马上回来”。不做参数搜索。

启用尾部驱逐时，只允许驱逐每个有有效hint的拥有者链当前驻留最大深度的块；共享块须同时满足所有这些拥有者的限制。更深块受protected或pin保护时，不跨过去破坏其浅层前缀。无hint拥有者不施加此额外限制，以保证全缺失时LRU等价。去掉尾部组件时仅关闭此资格限制，保留同分LRU排序；若已有排序自然从尾部开始，消融可以无差别。

跳过protected、ref_cnt=-1以及ref_cnt>0。返回恰好n个(key,chunk)并删除，无法满足则None且驻留集合、归属和LRU顺序不变；n=0返回空列表。分段线性评分使用按斜率分开的堆及失效项，链超时和共享评分交点用时钟事件刷新；不把时变评分缓存为永久不变值。会话状态改变会刷新该链所拥有的驻留块，尾部约束以链的深度索引维护。

## 模拟适配

保持E0的GPU LRU、独立CPU重复prompt镜像、16-token整块、完成时写入、零搬运耗时、固定0.5秒服务、FCFS槽位/KV排队。新增CPU适配器负责构造标准hint信封、在开始时仅观察元数据，在完成时按现有尾到头顺序更新LRU秩并交付hint。开始时不提前提升CPU的LRU顺序，保证none与原LRU逐请求等价。所有策略看到相同块访问和人口；hint只从已发出请求的当前行读取。perfect/noisy的P90沿用在线信封规则（expected的2倍），online用hint_online_p90_ms。缺失信封不推断final；final=true的有效信封其数值字段为空仍有效。

## 矩阵与指标

config-11：16K、8槽位、GPU182880、CPU274320 token；40分钟，预热10分钟，稳态初始化。λ={0.5,0.7,1,1.4}，人口种子={0,1,2}。直接使用E0归档的12个人口并校验抽样算法重建的指纹。包含E0真实运行不稳定的人口：本次只比较固定服务模型，不应用GPU稳定性筛选。

每人口主策略5个：LRU、hint-perfect、hint-noisy、hint-online、CPU Oracle（GPU仍LRU）。另有3组件×3hint档位的单组件消融，共14策略×12人口=168次。消融分别关闭尾部资格限制、final无穷大优先、超时降级；关闭final时回到块LRU估计，其余组件保留。另做12次none适配器控制，验证与原LRU逐请求及汇总完全一致；合计180次正式模拟。无限容量等正确性控制由单元测试验证，不属于收益矩阵。

按原始prompt token加权报告重算比例，is_entry和预热排除；与E0原名义比例相同（本次不使用整块分类分母）。每个人口保存逐请求parquet、run.json、人口及输入SHA256。LRU/Oracle与E0模型逐项回归，并要求所有策略的请求身份、发送/结束时刻及GPU命中相同；违反先停止调查。CPU策略不同可导致写入跳过，须报告，不将结果较差的人口删除。

配对减少百分点=100×(LRU重算率−hint重算率)；按同λ的三个种子等权给均值、最小值、最大值。Oracle空间捕获率=(LRU−hint)/(LRU−Oracle)，分母为0时记NA，不截断负数或大于100%的值；Oracle只是本最小模型的CPU块级未来访问策略，不保证全局最优。消融影响=100×(消融重算率−完整hint重算率)，正数表示去掉组件变差。报告完整逐人口表、λ汇总、消融表及PNG/PDF。记录有效hint数、超时数、CPU写入跳过数及实际耗时；不将固定服务模型收益或运行时当作vLLM实测收益/TTFT。
