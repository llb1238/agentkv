# 发布布局：无历史lean快照与私有完整证据

公开版 `llb1238/agentkv` 采用无Git历史的lean快照；私有仓库 `llb1238/agentkv-private` 保留完整历史与原始证据。发布工具在本地生成文件，不操作GitHub仓库的创建或可见性，也不启动GPU。

## full 与 lean

`--profile full` 是默认值，保持原来的完整当前文件树导出。`--profile lean` 使用白名单：源码、测试、docs、根配置、结果索引和迁移表、最终总结目录，以及各结果的README、图和不超过1,000,000字节的JSON/CSV汇总。明确的排除项优先，整个`results/archive/`、逐请求数据、light观测、日志和检查点进入证据包。

lean 单文件上限按用户确认的 **5 MiB = 5,242,880字节**；`horizon_probability_v1.json` 保留。总量硬上限为100,000,000字节，目标低于50,000,000字节；超限列出文件并停止，不自动删除或改写校准表。清单自身也计入上限。若目标未达到、硬上限仍满足，报告实际大小。

```bash
python -m agentkv.analysis.release --profile lean \
  --output-dir ../agentkv-runs/releases/agentkv-lean-20261109-privacy \
  --evidence-file ../agentkv-runs/releases/agentkv-evidence-20261109.tar.zst --reuse-evidence
```

需要系统`zstd`。目标目录必须不存在；默认新建证据包时不得覆盖旧包，`--reuse-evidence`则只读核验并复用指定旧包；证据包不进源码仓库。它包含lean排除的当前跟踪文件、外置清单中本机可得的原始载荷，以及发布时链接改写的文档原件。旁边的FILELIST与包内同名清单逐文件列出字节/SHA；内容分类、压缩包SHA及获取约定写入发布副本的[证据说明](EVIDENCE.md)。未登记的临时文件、模型权重和Git对象不打包。

所有链接改写只发生在lean副本；`results/relocations.json`和发布清单记录映射，私有原始JSON/CSV字节不修改。lean中的第三方进程身份、环境路径和主机名会脱敏，GPU编号、显存及实验指标保留；每个修改文件记录原SHA、发布SHA和原因。链接检查覆盖README、docs及保留结果README：

```bash
python -m agentkv.analysis.release_links --root ../agentkv-runs/releases/agentkv-lean-20261109
```

### 无历史副本中的CPU验证

复制lean目录到新的验证目录，再建立独立venv，不借用vLLM或原项目的PYTHONPATH：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e '.[cpu]'
.venv/bin/python -m pytest tests -q
.venv/bin/python -m agentkv.analysis.final_report --output-dir ../lean-report-new
```

精简模式从每次运行的`e1-analysis.json`重算51次运行、负载样本汇总、配对差值与表格，并逐项等值比较保存的JSON。还重建最终总结图数据，比较全部非来源SHA字段和README全部表格行。共同token校验沿用存档结论，不宣称重查了已外置的请求；TTFT CDF/队列保留原图。只读汇总可从发布清单获取源码提交号，正式回放和容量测量仍要求真实、干净的Git检出。

`final_report`同时生成`homepage-table.md`，逐格核对首页的13个数字。README的三条CPU命令已在全新venv执行验证。构建器检查docs中的发布状态；源码链接改写前后的目标类别及数量写入发布清单。上游vLLM链接固定完整提交号，fork改动文件指向补丁小节；外部链接只检查格式、提交号和行号格式，不发网络请求。

社区PR/RFC原文JSON单独进入证据包，公开版保留[原帖索引与项目摘要](../results/kv-hints-audit-20261102/community-summary.md)。[实验日志](EXPERIMENT_LOG.md)已统一放入docs；历史manifest的旧路径按原字节保存，由`results/relocations.json`登记迁移关系。

无vLLM环境会明确跳过后端集成测试；缺少归档原件时跳过两项历史实物检查，保留合成fixture的迁移验证。不把这些跳过算成后端通过。

## 目录组成

默认审阅位置为仓库相邻的 `../agentkv-runs/releases/agentkv-20261109-review-v2/`。它从提交后的当前Git文件树复制，不复制 `.git/`、未跟踪文件、模型、原始AgentX、完整prompt数组或[外置清单](publication_payloads.json)中的大回调流。

发布目录包含当前代码、README/REPORT、已保存结果元数据与图、Apache-2.0及署名说明，并附[最终vLLM补丁](patches/README.md)。完整补丁恢复测量fork；单独元数据补丁涉及2个实现文件及1个测试文件，两个补丁按各自基线使用。

| 发布目录根文件 | 含义 |
|---|---|
| `FILELIST.tsv` | 每个文件的相对路径、字节数与SHA256；两个清单自身只列大小，避免递归自哈希 |
| `RELEASE_MANIFEST.json` | 源提交、所有源码树文件指纹、文件数、精确总字节数、无Git历史/无外置载荷标志 |

复制后逐文件校验字节，文件数与总大小包含这两个清单。统计以实际生成的清单为准，不以磁盘分配块大小代替文件字节总量。原工作区与外置载荷保持不变。

## 重建命令

在干净的源码Git检出中执行，目标目录必须不存在：

```bash
python -m agentkv.analysis.release \
  --output-dir ../agentkv-runs/releases/agentkv-20261109-review-v2
```

这是一份无历史的普通文件目录，不伪造原始实验提交，也不包含完整vLLM仓库。公开版从lean快照建立独立发布历史；私有Git对象和原始证据继续保存在agentkv-private。

## 证据和隐私边界

历史生产者元数据和日志保留原始环境路径、提交号与SHA，不在分发时改写统计或伪造旧字节。本次“干净”指当前文件树快照、无Git历史和无外置载荷，不代表对每个历史二进制做过内容脱敏；既有检查范围见[公开前检查](PUBLICATION_CHECK.md)。若需要环境元数据脱敏，另建分发副本并保留指纹映射。

已保存的元数据足以核对统计和图。部分旧补丁/源码快照及原始载荷的来源指向私有历史提交；缺少外置文件时，不能宣称完成全部原始回调或GPU端到端复现。发布后的新Git克隆可执行[CPU汇总命令](REPRODUCING.md)，原始正式测量仍须使用记录的冻结输入和后端版本。

## 第三方信息与只读证据

lean构建对整个发布文件树检查其他账户路径、外部环境路径、主机名及无法确认属于本项目的GPU进程。可确认属于本项目的命令路径保留，其余进程PID、名称和路径替换为统一占位符。CSV与嵌套nvidia-smi文本均扫描；Markdown中的历史GPU进程表也处理。AGENTS.md不进入lean。

`RELEASE_MANIFEST.json`的每个改动文件含`original_sha256`、`published_sha256`、`change_reasons`，`privacy_redactions`记录字段指针和原因。最终构建门禁包含清单自身，命中任何未清除的第三方信息即失败。

复用旧证据包时，其压缩包、侧清单、元数据与生产者提交均保持原字节；新脱敏原件仍在私有仓库，不宣称旧包收录了新增的源码/文档版本。创建新证据包的默认模式则收录脱敏前原件；两种模式都不把发布脱敏应用于私有原件或旧证据。

```bash
python -m agentkv.analysis.publication_audit \
  --release-root ../agentkv-runs/releases/agentkv-lean-20261109-privacy \
  --output ../agentkv-runs/releases/privacy-audit-new.json
```
