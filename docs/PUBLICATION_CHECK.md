# 公开快照与私有证据检查

公开版 `llb1238/agentkv` 是无Git历史的lean快照；私有仓库 `llb1238/agentkv-private` 保留完整历史与原始证据。公开快照提供代码、方法、图和汇总，原始载荷另存证据包。以下保留2026-11-08的审计结果，并说明当前分发边界。

## 检查范围与结果

| 项目 | 状态与依据 |
|---|---|
| 密钥、令牌、带凭据URL | 对当前跟踪的可读文本和可达Git历史文本做模式扫描；未发现候选。定位信息只保存路径、行号、类型与摘要，不打印候选值 |
| 内网地址 | 扫描私网IPv4、链路本地地址、ULA/link-local IPv6和常见内网域名后缀；loopback和本地监听地址不当成泄密 |
| 文档本机绝对路径 | 转为仓库根目录下的相对布局；命令中的新输出默认位于相邻持久目录。历史展示修订登记在[relocations.json](../results/relocations.json) |
| 原始数据与模型 | 原始AgentX JSONL、正式大workload、完整在线prompt数组、模型权重不纳入当前源码分发；小型单元测试fixture不属于原始数据集 |
| 原始大回调流 | 停止跟踪5个大载荷，本机原字节未删除。大小、SHA256、原Git blob和提交见[外置清单](publication_payloads.json) |
| 结果可核验性 | 原始生产者JSON、日志、源码快照不改字节；展示README有独立前后SHA映射，不能冒充原始SHA |
| 第三方身份 | lean全目录检查其他账户/环境路径、主机名和无法确认属于本项目的GPU进程；只改发布副本并保留GPU/显存数值，原SHA、发布SHA与原因见RELEASE_MANIFEST；命中残留则构建失败 |
| 许可 | 根目录Apache-2.0；[NOTICE](../NOTICE)保留第三方署名；[DATASET](DATASET.md)说明固定数据来源、许可和衍生处理 |

机器检查记录见 [publication_audit.json](publication_audit.json)，CPU重绘及测试验证见 [closure_validation.json](closure_validation.json)。上述扫描是模式检查，不是“证明不存在秘密”：二进制、压缩载荷不做秘密内容扫描，其排除项数量在机器记录中明确列出；潜在凭据也可能不符合已知模式。

## 当前文件树与Git历史必须分开判断

当前跟踪文件树已排除原始大回调流，保留的最大文件为实验观测元数据，大小和完整清单以机器记录为准。仓库仍保留较多Parquet、轻量指标及日志，不将它称为轻量代码包。

**Git历史仍含这些大载荷，历史日志与生产者元数据仍含原实验环境路径。** 把文件加入忽略规则不会清除旧Git对象。当前没有改写历史、没有force-push，也没有改写签名/指纹所对应的原始实验字节，因此不能声明整个历史已经可直接公开。

公开版从lean文件树建立独立历史；私有历史保持原样。若另做环境元数据脱敏，应保留单独副本和SHA映射，不能覆盖原始证据。

lean快照的生成方式及清单字段见[RELEASE](RELEASE.md)，原始证据的内容及获取方式见[EVIDENCE](EVIDENCE.md)。lean构建同时检查文档发布状态、相对链接及固定提交的公开源码链接。

## 复查命令（只读、CPU）

```bash
python -m agentkv.analysis.publication_audit --history \
  --output ../agentkv-runs/publication-audit.json
python -m agentkv.analysis.final_report \
  --output-dir ../agentkv-runs/report-review-new
```

扫描器只读取Git文件/对象；结果输出在调用者指定的新位置。旧原始载荷的合法获取与恢复要求见 [REPRODUCING](REPRODUCING.md)。本次文档中的相对路径不是对原始JSON字段的重写，也不是新的GPU复现结果。
