# 结果索引

## 最终报告依据（审阅稿）

2026-11-08 实验阶段结束，不再启动新的 GPU 运行。入口为 [REPORT](../docs/REPORT.md)；以下各目录共同支撑结论，后一个目录不自动替代前一个目录的测量。

| 结果目录 | 最终用途 |
|---|---|
| [final-summary-20261108](final-summary-20261108/README.md) | 首页唯一总结图、三人口均值之比、E0间隙/总重算与v2校准核算；仅重用已有数据 |
| [m0-lifecycle-20261102](m0-lifecycle-20261102/README.md) | 生命周期、CPU回载与重算代价的动机依据 |
| [m1-e0-config11-20261103](m1-e0-config11-20261103/README.md) | E0、名义重算来源、稳定性、同人口LRU/Oracle |
| [e1-gpu-config11-20261105](e1-gpu-config11-20261105/README.md) | E1 v1，有效重算与TTFT、完整持久目录重跑 |
| [e1-gpu-v2-config11-20261106](e1-gpu-v2-config11-20261106/README.md) | E1 v2，链级rank实现、与v1并列及正式长窗口开销 |
| [e1-gpu-stable-limit-12-20261107](e1-gpu-stable-limit-12-20261107/README.md) | λ=1.2九次上限实验、无稳定LRU配对的边界 |
| [m5-scoring-evaluation-20261104](m5-scoring-evaluation-20261104/README.md) | 冻结后的CPU人口留出、预测来源与Oracle空间 |
| [m5-scoring-development-20261104](m5-scoring-development-20261104/README.md) | 仅用于三候选的开发选型依据，不作最终留出数字 |
| [m5-chain-growth-20261107](m5-chain-growth-20261107/README.md) | 状态增长、回收、全部序列等价、未通过的最终GPU开销 |
| [m5-chain-implementation-20261106](m5-chain-implementation-20261106/README.md)、[m5-policy-overhead-20261104](m5-policy-overhead-20261104/README.md) | 短时门槛与none退化路径，不能代替长时验收 |
| [m5-vllm-integration-20261104](m5-vllm-integration-20261104/README.md)、[m5-vllm-key-diagnosis-20261104](m5-vllm-key-diagnosis-20261104/README.md) | 后端补丁、完整前缀归属和真实回调语义 |
| [m5-chain-scoring-20261106](m5-chain-scoring-20261106/README.md) | online录制输入及严格深度尾序的冻结语义反例 |
| [m1-two-tier-seed0](m1-two-tier-seed0/README.md)、[m1-oracle-seed0](m1-oracle-seed0/README.md) | 两层模型约束、CPU策略接口与Oracle定义 |
| [m1-agentx-data-20261102](m1-agentx-data-20261102/README.md)、[m1-workload-seed0-v3](m1-workload-seed0-v3/README.md)、[16K派生](m1-workload-seed0-v3-16k/README.md)、[m1-steady-state-seed0](m1-steady-state-seed0/README.md) | 固定数据、负载构造、事件锚点、稳态初始化 |

展示路径的迁移登记见 [relocations.json](relocations.json)；历史生产者元数据和源码快照保留原字节。外置原始载荷及公开前限制见[检查说明](../docs/PUBLICATION_CHECK.md)。

当前使用：**v3 32K/16K 负载 → 稳态估算 → capacity v3 → two-tier LRU**。M5当前最终模拟依据为独立种子3/4/5的冻结后评估，种子0/1/2均为开发数据。E0 config-11 的12次矩阵已完成：λ=0.5/0.7各3次全稳定，λ=1为2/3稳定，λ=1.4为0/3稳定；统一使用设计§7.6新判据；离线筛选和假服务通过不能等同 GPU 验收。

## 当前保留的结果目录

| 目录 | 用途 | 状态 | 取代关系 / 限制 |
|---|---|---|---|
| [final-summary-20261108](final-summary-20261108/) | 总结图与草稿数字逐项核算 | 完成；实测柱/预测点、稳定次数矩阵，PNG 300dpi及PDF | 仅派生展示，不是新GPU测量；所有源报告/汇总SHA见run.json |
| [m5-chain-growth-20261107](m5-chain-growth-20261107/) | 80分钟链状态诊断、空链/共享图回收、全部录制有序驱逐复核及40分钟GPU开销验证 | CPU结果与旧版逐请求相同；256测试通过/3历史跳过；真实online finished/evict P99均≤2ms；四轮GPU增量均未达2ms，最新版+3.277ms | 修复了状态积累，GPU开销验收仍未完成；所有失败尝试单列保留，不替代既有E1数据 |
| [e1-gpu-stable-limit-12-20261107](e1-gpu-stable-limit-12-20261107/) | config-11、λ=1.2、种子3/4/5、LRU/online/perfect的稳定负载上限矩阵 | 九项CPU预测与9/9首次GPU运行完成；LRU/online各0/3稳定，perfect为2/3；117668个hint传输对账、6项token核对通过 | 与λ=1/1.4的E1 v2并列；稳定LRU配对数为0，配对主指标为NA；此前GPU开销未达标的限制保留 |
| [e1-gpu-v2-config11-20261106](e1-gpu-v2-config11-20261106/) | 链级rank实现的E1 v2，18次GPU矩阵与v1并列 | 18/18完成，无重跑；λ=1九次全稳定、λ=1.4九次全不稳定；200190个hint传输对账、12项组内及18项跨版本token核对通过 | 当前链级正式数据；v1保留作实现性能对照；λ=1 online减重算1.547pp、配对P50+1.899ms，perfect减8.190pp；40分钟scheduler增量并非均≤2ms |
| [m5-chain-implementation-20261106](m5-chain-implementation-20261106/) | 确认rank语义后的链级实现、CPU等价与时延、三组GPU开销门槛 | 两份完整包装层回放734/2005次有序key逐次相同；online finished/evict P99=1.079/1.959ms；246测试通过/3历史跳过；GPU三组20分钟门槛通过（schedule P99差+1.802/+1.343ms） | 保持冻结rank与打分；未达标开发测量保留归档；E1 v2已完成；v1保留作对照 |
| [m5-chain-scoring-20261106](m5-chain-scoring-20261106/) | 真实online完整回调输入及严格尾序反例 | 779请求/875811回调完整；734/2005次冻结驱逐复现；严格深度尾序与冻结集合冲突的历史门槛记录 | 输入及反例继续有效；用户后续确认保留rank，见[链级实现](m5-chain-implementation-20261106/)；原始记录不改 |
| [e1-gpu-config11-20261105](e1-gpu-config11-20261105/) | E1正式GPU矩阵及24组预注册CPU预测 | 24/24重跑完成，λ=1全部12次稳定、λ=1.4全部12次不稳定；227测试通过/3历史跳过；全部原始指纹与18项共同token核对通过 | 首次GPU数据丢失的摘要全部排除，现使用持久目录attempt-2；online在λ=1平均减重算1.386pp但TTFT P50增加51.8ms；perfect减重算8.154pp；不稳定项单列、无显著性检验 |
| [kv-hints-audit-20261102](kv-hints-audit-20261102/) | 官方 hint 通道、HTTP 与源码核查 | 完成；静态审计基线；公开版提供[社区摘要](kv-hints-audit-20261102/community-summary.md) | 原始PR/RFC JSON保留在私有证据包；供 M4 参考 |
| [m0-lifecycle-20261102](m0-lifecycle-20261102/) | 生命周期、GPU/CPU/重算 TTFT 与源码观测 | 完成；有效 M0 基线 | 无取代；不代表 E0 正式负载 |
| [m1-agentx-data-20261102](m1-agentx-data-20261102/) | AgentX 原始数据审计、链推断与 provenance | 完成；当前输入依据 | 后续负载依赖本目录，不应归档 |
| [m1-position-check](m1-position-check/) | 全量 hash 位置一致性检查 | 通过；当前生成前置校验 | 后续 workload manifest 引用其结果 |
| [m1-workload-seed0-v3](m1-workload-seed0-v3/) | 32K（32512）事件锚定负载 | 通过；当前 32K 输入 | 取代 workload v1、v2；独立重复证据已归档 |
| [m1-workload-seed0-v3-16k](m1-workload-seed0-v3-16k/) | 同根同种子的 16K 派生负载 | 通过；当前 16K 输入 | 与 32K 并列，不取代 32K |
| [m1-steady-state-seed0](m1-steady-state-seed0/) | s=0.5 调度 sidecar、稳态人口与初始化对照 | 完成；当前离线依据 | 供回放和容量分析使用 |
| [m1-capacity-seed0-v3](m1-capacity-seed0-v3/) | GPU<CPU、1.5–2 比例和 0.8 工作集的新容量筛选 | 完成；54 组合/16 容量合格 | 取代 capacity 初轮、v2；推荐还须经过两层模拟 |
| [m1-two-tier-seed0](m1-two-tier-seed0/) | 两层 LRU、无限缓存对照、吞吐及短跑清单 | 完成；16 模拟/14 达门槛；非 E0 | 模拟优先清单 config-08/11/13；本目录仅为离线模型；config-11的正式E0见新目录 |
| [m1-scheduler-tail-final](m1-scheduler-tail-final/) | 125 在途、12K prompt、两次严格冻结基线的长尾验证 | 通过；当前调度修复证据 | 取代 tail 三次诊断和旧假服务调度证据；不替代真实 GPU 容量验收 |
| [pr62382-overlap-20261102](pr62382-overlap-20261102/) | #62382 策略重叠与接入点核查 | 完成；静态核查 | 无取代；不是运行对比结果 |
| [week1-local-evidence](week1-local-evidence/) | 相邻 vLLM 仓库原始证据的文件清单与 SHA256 | 有效；外部本机证据索引 | 支撑 Week1；仅克隆本仓库不能恢复外部数据 |
| [m1-oracle-seed0](m1-oracle-seed0/) | 14配置CPU LRU/Oracle成对比较与排序 | 完成；目标8.944pp、最大9.290pp | 使用当前policy接口；不覆盖旧LRU基线 |
| [m1-shared-pilot-seed0](m1-shared-pilot-seed0/) | config-11共享GPU、20分/预热5分、两卡dmon | 完成；3906成功/0失败；旧判据的队列比值条件未过 | 指定GPU/CPU池及LRU/none；运行中位数2，未通过E0验收 |
| [m1-e0-config11-20261103](m1-e0-config11-20261103/) | config-11、4个λ×3人口、40分/预热10分；同人口LRU/Oracle与重算分解 | 完成；80258请求全成功；8稳定/4不稳定；逐块一致率100% | GPU1显存/算力共享；GPU0空闲9.705GB未达门槛；不稳定运行单列且不入主曲线 |
| [m5-e1-simulation-config11-20261104](m5-e1-simulation-config11-20261104/) | config-11初轮deadline对照；4个λ×3人口、单组件消融及none控制 | 完成；180次纯CPU模拟，142测试通过，1816230逐请求记录复核通过 | 与E0并列；不取代真实GPU数据；online在λ≥1平均劣于LRU；固定0.5秒服务，无TTFT证据 |
| [m5-e1-remaining-config11-20261104](m5-e1-remaining-config11-20261104/) | 293根剩余寿命校准及早期开发对照；预测误差/过deadline占用、LRU+final、新旧打分及尾部控制 | 完成；156次纯CPU模拟，156测试通过，1574066条记录复核；online 12/12正收益 | 与初轮deadline对照并列；默认改为尾部关闭、noisy/online剩余寿命；部分档位弱于旧打分；不替代真实GPU/TTFT证据 |
| [m5-scoring-development-20261104](m5-scoring-development-20261104/) | 三候选online、LRU/终止信号/Oracle；种子0/1/2选型 | 完成；72模拟、173测试；726492条记录复核；remaining胜出并单独冻结 | 仅开发数据，不是最终数字；horizon合格但捕获率低于remaining；含293根CDF校准和H轨迹 |
| [m5-scoring-evaluation-20261104](m5-scoring-evaluation-20261104/) | 冻结remaining后，种子3/4/5；六策略×四个λ×三种子 | 完成；72模拟、700710条记录复核；online 12/12优于LRU | 当前M5最终模拟依据，仅人口留出、同一100根trace；非真实GPU/TTFT；开发数字不混入主表 |
| [m5-vllm-callback-probe-20261104](m5-vllm-callback-probe-20261104/) | config-11、perfect、λ=0.7、5分钟LRU等价回调探针；两卡dmon | 1292成功/0失败，hint 1292=1292，key原始顺序通过；真实store/ready语义与模拟器不同，按门禁停止步骤2–4 | 有效可行性诊断；非HintAware GPU收益/开销验收；退出footer缺失限制保留 |
| [m5-vllm-key-diagnosis-20261104](m5-vllm-key-diagnosis-20261104/) | 原40个不完整请求及6个空列表的逐项归因、冻结策略共享final CPU反例 | 40个均为GPU前缀跳过；其中4空，另2空为不足一块；187测试通过 | 纠正上一报告包含关系；共享final归属风险触发停止，尚未执行适配或GPU冒烟；原日志不足以完整回放 |
| [m5-vllm-integration-20261104](m5-vllm-integration-20261104/) | 新ReqContext后端、完整前缀归属、5分采集与10分冒烟及完整时钟回放 | 后端1395通过/10跳过；AgentKV197通过/3历史跳过；1284/2259请求全成功、hint与prefix对账、五不变量通过，归属100% | 后端4749dd5961，实测a736e2b；perfect/λ0.7/共享GPU功能证据，非零开销或正式E1；额外模型集成缺权重 |
| [m5-policy-overhead-20261104](m5-policy-overhead-20261104/) | 完整回调CPU优化、默认轻量插桩、config-11同人口LRU/none端到端开销复测 | CPU包装层P50/P99=0.068/1.499ms，2005次逐次一致；none1128次一致；GPU两组3445/3450请求全成功，P50差−1.168%通过2%门槛；211测试通过/3历史跳过 | 提交0ffd5d9，后端4749dd5961；取代完整插桩的性能数字，不取代完整回调功能证据；none专用LRU退化路径，非带hint开销或E1；首轮未过和基线漂移保留 |
| [repository-maintenance-20261103](repository-maintenance-20261103/) | 本次目录迁移、来源规则、索引及链接校验日志 | 维护验证；无性能测量 | layout-tests.log 对应独立迁移提交；最终验证另记 |

## 归档目录

归档保留原始测量数据、run.json、源码快照及失败日志；不会将中断或未通过结果混入当前推荐。

新增归档：[M5首轮开销未过门槛](../docs/EVIDENCE.md#archives)保留P50差+3.602%的原始两组GPU结果及汇总错误；[M5早期CPU诊断](../docs/EVIDENCE.md#archives)保留被拒绝优化、旧计时与profile，不用于最终门槛，后续由主结果的干净提交顺序复测取代。两者均为首次入库，未迁移已有报告链接；原始元数据未重写。

| 目录 | 用途 | 状态 | 后续依据 |
|---|---|---|---|
| [m5-chain-implementation-development-20261106](../docs/EVIDENCE.md#archives) | 链级实现的早期CPU时延与独立profile | 保留未达2ms门槛及开发中事件缓存失效记录；不作为GPU/E1收益依据 | [当前链级实现](m5-chain-implementation-20261106/) |
| [e1-gpu-config11-20261105-lost-on-reboot](../docs/EVIDENCE.md#archives) | 首次E1 GPU矩阵的基础设施中断记录 | 原始GPU产物及checkpoint丢失；对话监控摘要不作为测量，未伪造原始文件 | [E1持久目录恢复](e1-gpu-config11-20261105/)；CPU预测仍有效 |
| [m5-vllm-full-probe-20261104-unclosed](../docs/EVIDENCE.md#archives) | 首次完整参数5分钟采集 | 排除：gzip EOF，退出footer未验证 | [m5-vllm-integration-20261104](m5-vllm-integration-20261104/)；原始损坏流保留，未补造结束标记 |
| [m1-capacity-seed0](../docs/EVIDENCE.md#archives) | 固定 CPU=4×GPU 的初轮容量探索 | 被取代 | [m1-capacity-seed0-v3](m1-capacity-seed0-v3/) |
| [m1-capacity-seed0-v2](../docs/EVIDENCE.md#archives) | GPU+CPU<工作集 的旧筛选 | 被取代 | [m1-capacity-seed0-v3](m1-capacity-seed0-v3/) |
| [m1-online-lru-seed0](../docs/EVIDENCE.md#archives) | 首段共享 GPU 在线运行 | 中断；无完整请求产物 | [archive/m1-online-lru-seed0-final](../docs/EVIDENCE.md#archives) |
| [m1-online-lru-seed0-rerun](../docs/EVIDENCE.md#archives) | 安全阀/超时边界修复前运行 | 排除；诊断中断 | [archive/m1-online-lru-seed0-final](../docs/EVIDENCE.md#archives) |
| [m1-online-lru-seed0-final](../docs/EVIDENCE.md#archives) | 完整 40 分钟旧回放器试跑 | 保留诊断；调度精度未验收、持续排队；非 E0 | [m1-scheduler-tail-final](m1-scheduler-tail-final/) |
| [m1-replay-fake-validation](../docs/EVIDENCE.md#archives) | 旧版小负载假服务与 hint 验证 | 历史验证；被当前测试及压力验证取代 | [m1-scheduler-tail-final](m1-scheduler-tail-final/) |
| [m1-scheduler-tail-seed0](../docs/EVIDENCE.md#archives) | 首次压力探针 | 失败：uint64 产物 schema | [m1-scheduler-tail-final](m1-scheduler-tail-final/) |
| [m1-scheduler-tail-seed0-v2](../docs/EVIDENCE.md#archives) | 仅提前构建和预载 token 的探针 | 未达标：P99>50ms | [m1-scheduler-tail-final](m1-scheduler-tail-final/) |
| [m1-scheduler-tail-seed0-v3](../docs/EVIDENCE.md#archives) | 线程限制/查表优化的中间探针 | 排除：基线导入了新辅助代码 | [m1-scheduler-tail-final](m1-scheduler-tail-final/) |
| [m1-workload-seed0](../docs/EVIDENCE.md#archives) | 原始绝对链启动偏移负载 | 被取代 | [m1-workload-seed0-v3](m1-workload-seed0-v3/) |
| [m1-workload-seed0-repeat](../docs/EVIDENCE.md#archives) | v1 独立确定性复现 | 归档对照；v1 已被取代 | [m1-workload-seed0-v3](m1-workload-seed0-v3/) |
| [m1-workload-seed0-v2](../docs/EVIDENCE.md#archives) | 根会话时间线压缩后的绝对偏移负载 | 被事件锚定取代 | [m1-workload-seed0-v3](m1-workload-seed0-v3/) |
| [m1-workload-seed0-v2-repeat](../docs/EVIDENCE.md#archives) | v2 独立确定性复现 | 归档对照；v2 已被取代 | [m1-workload-seed0-v3](m1-workload-seed0-v3/) |
| [m1-workload-seed0-v3-repeat](../docs/EVIDENCE.md#archives) | v3 独立确定性复现 | 通过；证据已汇总入主报告后归档 | [m1-workload-seed0-v3](m1-workload-seed0-v3/) |

## 实验内部子目录

- `m1-two-tier-seed0/config-00..15/`：16 个容量合格组合的逐请求输出；03/04 未通过吞吐门槛，其他 14 个通过模拟门槛。优先共享 GPU 短跑清单为 08/11/13；11已完成20分钟诊断及本次E0，08/13尚无实跑。
- `m1-two-tier-seed0/unlimited-*/`：六组同人口无限缓存对照，已完成；`small-cpu-diagnostic/` 是 CPU/GPU=0.5 的负面对照，不是候选。
- `m1-scheduler-tail-final/run1/`、`run2/`：两次独立调用；各自 `before/` 是故意保留的旧版基线，`after/` 是通过 P99<50ms 的版本。
- `m1-steady-state-seed0/population-seed0/`：λ 与初始化开关的子实验；其他 trace、figures、server-observations、输入 fixture 等目录是所属实验的数据分组，不是额外的 E0 结果。

- `m1-e0-config11-20261103/lambda-*/models/{lru,oracle}/`：12个人口的成对模型；`real/`：完整实跑及去除重复prompt数组的元数据；4次不稳定实跑保留在矩阵内，单列诊断，不算程序失败或稳定样本。原始大请求文件本机保留且有SHA256，紧凑元数据、两卡dmon、事件、判据和图均纳入版本管理。

## 版本与归档规则

1. 从本次维护起，回放器、容量分析及测量工具默认要求干净工作区；代码版本在新 run.json 中只保存 `git_commit`，不保存状态清单、diff 或源码副本。`--allow-dirty` 仅调试使用，打印醒目警告，不能作为正式实验来源。输入和结果文件的 SHA256 仍保留。
2. 历史 run.json 与源码快照不重写：其中的旧 import、绝对路径、Git dirty 信息是当时执行的证据，不是当前命令入口。需要原始实现时可从其记录的 Git 提交或已存历史快照查阅。以后不再增加源码快照。
3. 当前命令统一为 `python -m agentkv.bench.…`、`python -m agentkv.sim.two_tier`。当前审计脚本已更新命名空间；归档内记录的启动脚本只代表当时的命令。`m1-two-tier-seed0/small-cpu-diagnostic/run.py` 是带自指纹的原始诊断源码，仍按原字节保留，不是当前入口。
4. [relocations.json](relocations.json) 给出旧目录到归档目录的映射。报告 README 仅更新路径和链接时，记录原始/当前 SHA256；原始字节可从记录的原始提交恢复。当前审计使用 `agentkv.artifacts.verify_artifact` 只认可明确登记的 README 展示迁移，不放宽数据或源码的指纹检查。
5. 新增、失败、被取代的实验必须更新本索引；只把失败、被取代或已汇总的复现对照移入 archive，不移动当前输入及有效基线。
