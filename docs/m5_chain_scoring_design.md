# 按链评分的等价优化：执行顺序与门槛

先用固定vLLM `4749dd5961`、现有冻结online策略录制config-11、λ=1、人口种子3、300秒、稳态初始化的真实完整回调。使用`TracedHintAwarePolicy`记录全部方法参数、key、ReqContext、时钟及ref_cnt变化，验证gzip footer、计数和hint对账。采集本身是调试插桩，不用于性能门槛。所有产物写到`../agentkv-runs/`的持久目录。

随后以新的online序列和既有perfect序列作有序key逐次一致性检查，比较None、返回个数、顺序及protected/ref_cnt契约。优先保留冻结语义，不能用更改打分、校准或输入来消除差异。现有冻结表两档均为`tail_first=false`；新要求的深度尾序是否与现有同分块recency顺序相容，必须以反例及真实调用定位。若不能完全相同，先停止并报告，不进入后续GPU实验。

一致后才评估无插桩online回放的on_request_finished、evict，二者P99均须小于2ms；profile单独运行，解码不计入回调耗时。再在同人口λ=1/种子3上按LRU、online、perfect顺序各跑1200秒，预热300秒，轻量插桩，比较TTFT P50及scheduler.schedule P99；后两者schedule P99不得超过LRU 2ms。冻结的真实决策等价性是前置门槛，失败不换配置绕过。

全部门槛通过之后，另行提交E1 v2预注册：λ=1/1.4、种子3/4/5、LRU/online/perfect，共18次，每次2400秒、预热600秒，其他设置和稳定性标准与v1一致。v1原始记录及评分冻结文档保留原字节，报告并列展示实现性能变化与逐次相同的驱逐决策证据。此文不是提前宣布v2已具备开测条件。

采集入口：`python -m agentkv.bench.record_online_callbacks --output-dir ../agentkv-runs/m5-chain-scoring-20261106/online-trace --gpu-index 1 --port 8017`。

## 2026-11-06：在语义门槛停止

5分钟online采集完成，779个hint全部对账、窗口请求无失败，完整记录875811个回调并验证footer。原冻结实现对新online的734次、旧perfect的2005次evict全部复现。

只读检查在真实online调用196992发现：返回的深度5440块与仍可驱逐的5456块属于同一条链、分数完全相同，均ref_cnt=0且不受protected保护；5456块不在原返回集合中。perfect调用355435同样出现7088/7104的情况。既有同分rank顺序与严格深度尾序会改变返回集合，不能把后者称为纯实现优化。

因此尚未实现完整链队列，未进行2ms性能门槛、三组20分钟GPU开销实验或E1 v2预注册/运行。等待确认：按链维护分数、链内保留冻结rank次序；或接受严格深度尾序带来的语义改变并另行预注册。完整证据、复现命令及指纹见[结果报告](../results/m5-chain-scoring-20261106/README.md)。

## 用户确认后：链级评分与原rank顺序

采用链级队列，排序键为链分数与该链剩余候选块最小rank；每移除一个候选后重新入队。共享块出现在全部所属链的候选列表；受更优链保护的副本在当前链视图中跳过。protected与live ref_cnt在选择时检查，失败的exact-n驱逐回滚。无hint及剩余寿命无存活样本时保留逐块LRU退化，链头推进后同时更新其分数。

时间变化只更新链评分；不再维护逐块驱逐分数堆。为保留冻结版的超时和压缩清理时序，保留只用于事件取消计数的交叉事件账本；只有共享块可能产生这类事件，普通链的寿命分段点不刷新其逐块驱逐分数。真实序列必须同时通过录制的有序key和734/2005次既有基线逐项核对，任何差异立即停止。CPU适配层回放无插桩、解析不计时，online的finished与evict P99均须≤2ms。通过后才开始三组20分钟GPU开销验证，再另行预注册18次E1 v2。

当前核心及完整vLLM包装层两份序列均已逐项相同。完整包装层online的finished/evict P99为1.079/1.959ms；其source提交为6fefbdd，检查关闭插桩、解析不计时。GPU验证固定LRU→online→perfect，config-11、λ=1、种子3、20分钟/预热5分钟，各自新服务，common逻辑请求token逐项核对、两卡dmon；使用轻量插桩及相同实际prefill观察器。histogram仅给区间，采用候选schedule P99上界减LRU下界≤2ms的保守门槛，不通过则不进入E1 v2。开发中未达标的CPU记录和已修复的事件缓存失效测试问题保留，不以重复测量择优覆盖。

## 2026-11-07：完成门槛及E1 v2

提交415f477的三组20分钟GPU门槛全部完成，online/perfect的schedule P99保守增量分别为1.802/1.343ms。随后在52ab4de单独提交[E1 v2预注册](e1_v2_preregistration.md)，完成18次40分钟运行；全部为attempt-1，无中断重跑，原始清单与指纹保留在[正式结果](../results/e1-gpu-v2-config11-20261106/README.md)。v1未改，逐人口并列及共同token核对均已发布。

λ=1九次均稳定，online/perfect平均减少有效重算1.547/8.190pp，TTFT P50配对差为+1.899/−3.125ms；λ=1.4九次均不稳定，不纳入稳定主指标。正式40分钟下λ=1的schedule P99保守增量为2.03–4.59ms，未保持短时门槛的2ms界限，所有超出值单列；不据短时或固定回调结果宣称全程开销保证。公式、校准和有序驱逐语义保持冻结；真实闭环的服务时刻与请求交错可能改变。

> 收尾展示修订：仅将运行位置改为相对布局，参数、顺序和判据不变；原始展示字节可用 `git show f59d0026ac9c2d9942d58bed84d90f5dbc1dc580:docs/m5_chain_scoring_design.md` 查看，JSON协议保持原字节。见[路径约定](REPRODUCING.md)。
