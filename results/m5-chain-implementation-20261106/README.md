# 链级实现：冻结决策一致性与性能门槛

状态：CPU一致性与online时延门槛通过，三组20分钟GPU开销验证全部完成且通过。E1 v2另行预注册。

用户确认采用链级评分、链内保留原rank顺序。每条链候选按rank排列，队列使用`(-预计下次使用时间, 最小候选rank)`；取走一个块后用下一候选重新入队。共享块保留在每个所有者的列表中，受更优所有者保护的视图跳过。protected和实时ref_cnt在选择时检查；无hint和无剩余寿命样本保持LRU退化，链头推进后重新计算其分数。

不再维护逐块驱逐分数堆；分段点只更新链状态。冻结实现的交叉事件取消账本仍保留，用于完全相同的超时与压缩清理时序，它不是驱逐优先队列。惰性缓存限定在一次回调时刻内。原评分实现、校准数据及原始E1 v1记录未重写，vLLM仍固定`4749dd5961841243596559b10ba0a22c48980c45`。vLLM配置须显式`implementation=chain`，默认indexed路径保留。

## 有序key一致性

源数据为[5分钟online完整采集](../m5-chain-scoring-20261106/online-trace/)及[原perfect完整序列](../m5-vllm-integration-20261104/smoke/)。时钟、回调、refs与protected原样回放，逐次比较录制的key列表及先前734/2005次基线，再比较总指纹；不是只比较命中率。

| 实现范围 | online 734次 | perfect 2005次 |
|---|---|---|
| 前缀适配层+链核心 | 全部顺序相同 | 全部顺序相同 |
| 完整vLLM包装层，关闭插桩 | 全部顺序相同 | 全部顺序相同 |

online指纹`01fe734e2b2ea34b13800544feb8df8de5ec44bf521e6905dde72fb3ee357200`；perfect指纹`09fc5af2fb5f632f3f98cf3ade5a9d6c70c09c27cdf256f2aa48222416216746`。各次[CPU数据](cpu/)保留原始run.json、回调时间和逐次决策parquet。核心测量源提交45bb4dd，完整包装层6fefbdd；完整SHA见各次记录。

## CPU时延

源序列解码在计时外；无cProfile，无运行时插桩，外部计时包围完整回调。单位ms，P99按原工具的numpy分位数计算。

| 范围 / 输入 | finished P99 | evict P99 | 门槛适用 |
|---|---:|---:|---|
| 适配层+核心 / online | 1.073 | 1.918 | 通过≤2ms |
| 完整vLLM包装 / online | 1.079 | 1.959 | 通过≤2ms |
| 适配层+核心 / perfect | 4.845 | 2.914 | 附带数值，CPU门槛指定online输入 |
| 完整vLLM包装 / perfect | 5.002 | 2.935 | 附带数值，GPU仍须通过同一开销门槛 |

P99门槛不是最大耗时门槛；完整online包装层evict最大值仍为3.595ms。先前未达标的各版时延及独立profile见[开发归档](../../docs/EVIDENCE.md#archives)，没有用重复测量择优覆盖。开发时发现的交叉时间缓存失效循环已修复，修复前版本未用于GPU。完整验证244通过、3项历史后端测试跳过，见[tests.log](../../docs/EVIDENCE.md#runtime-logs)。

## GPU门槛（已通过）

固定LRU→online→perfect，各自新服务、新目录，config-11、λ=1、人口种子3，1200秒、预热300秒；同人口、共同逻辑请求token核对。LRU为vLLM内置类、忽略传输hint；三者均使用相同轻量观察器，两卡dmon及退出汇总。报告scheduler.schedule P99与TTFT P50及相对同人口LRU的差。

直方图只提供分位区间，保守采用候选P99上界−LRU下界≤2ms作为通过条件。三组全部完成且通过才进入E1 v2；不能把此处CPU结果等同GPU验收。

```bash
python -m agentkv.bench.chain_trial --output-dir ../agentkv-runs/m5-chain-gpu-gate-20261106 --gpu-index 1 --port 8018
```

必须干净工作区。输出在持久ext4目录，支持跳过经指纹核验的已完成项；不覆盖不完整目录。原始客户端run.json保留，仅新增overhead-run.json与去掉prompt数组的元数据。没有向结果目录复制实现代码。

CPU复核使用`python -m agentkv.bench.chain_replay --run-dir results/m5-chain-scoring-20261106/online-trace --output-dir ../agentkv-runs/NEW-chain-online --hint-tier online --implementation chain-plugin`。perfect改为原smoke路径及对应tier。依赖环境与[原始采集复核说明](../m5-chain-scoring-20261106/README.md)一致。

### GPU 实测结果

测量提交`415f4778699b11c5a6b520986b48d351e196af3a`，各组1200秒、预热300秒均完成，无失败请求。原始[GPU记录](gpu/run.json)、两卡dmon、轻量直方图及去prompt元数据均保留，完整prompt产物在本机持久目录。

| 策略 | 窗口请求 | schedule P99区间 ms | 相对LRU保守差 ms | TTFT P50 ms | ΔP50 ms | hint发送/接收/消费 |
|---|---:|---:|---:|---:|---:|---|
| lru | 3445 | [7.766, 7.799] | +0.000 | 52.221 | +0.000 | 4244/4244/None |
| online | 3447 | [9.503, 9.568] | +1.802 | 52.338 | +0.117 | 4247/4247/4247 |
| perfect | 3462 | [9.044, 9.110] | +1.343 | 54.900 | +2.679 | 4263/4263/4263 |

三组共同人口指纹及共同请求token逐项相同；LRU不消费hint。online/perfect保守P99增量均≤2ms，满足进入E1 v2的门槛。TTFT P50差值为原值，不把通过schedule门槛解释为零开销。

E1 v2运行/报告器扩展后完整验证246通过、3项历史后端跳过，见[v2-runner-tests.log](../../docs/EVIDENCE.md#runtime-logs)。v1协议与结果原样保留；v2协议单独登记，显式选择chain，并按18项终态和12项组内共同token校验。

## 后续E1 v2

预注册`52ab4de`后18次40分钟运行全部完成，见[E1 v2及v1并列报告](../e1-gpu-v2-config11-20261106/README.md)。本目录run.json的e1_v2_registered=false是预注册之前的门槛发布状态，保留原字节，以匹配预注册中的SHA256。20分钟GPU门槛通过不保证40分钟或其他人口的scheduler增量≤2ms；正式数据的超出情况已另列。
