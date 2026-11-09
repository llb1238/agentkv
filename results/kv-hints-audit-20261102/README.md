# kv_hints 专项核查（2026-11-02）

结论见 [半页笔记](../../docs/kv_hints.md)。此次只查源码、历史及 GitHub 讨论，没有编写策略或运行 GPU 实验。

- `local-audit.json` / `local-source-search.txt`：固定 commit、引入 commit、源码 SHA256、Python HTTP 请求模型/转参检查及 OpenAI SDK 3.37.0 的类型核对。
- `local-audit.py`：本地核查脚本，在 vLLM 根目录用 `.venv/bin/python ../agentkv/results/kv-hints-audit-20261102/local-audit.py` 重跑；本机目录按当前相邻仓库布局固定。
- `pr53423.json`、`pr53423_discussion.json`：引入 PR 的状态、正文及人工评审讨论。原始讨论只保存在私有证据包，公开版使用[原帖与项目摘要](community-summary.md)。
- `search_queries.json` / `search_results.json`：GitHub issue 查询和返回的相关 RFC 索引。
- `rfc*.json`：#53421、#51428、#52113、#59708、#57103 的正文及相关讨论快照，原文仅保留在私有证据包。
- `pr*.json`：#53423/#53624 已合并；#59714/#58198/#62382 在核查时仍为 open draft。RFC 正文中的旧状态以 PR API 元数据为准。
- `pytest.log`：现有 Request 与 ReqContext 两个透传测试通过（2 passed）。
- `pytest-kvcr-optional.log`：初次组合包含 KVCR 测试，但环境未安装可选 kvcr，文件由 importorskip 跳过，指定节点因此无 collector；不是完整套件通过记录。未为这项源码核查安装 KVCR。
- `manifest.json`：本目录原始证据文件及半页笔记的 SHA256；它描述历史原件，不要求公开快照携带这些原文。

公开版提供[社区讨论摘要](community-summary.md)，PR/RFC原始JSON进入[证据包](../../docs/EVIDENCE.md#community-originals)，不以摘要字节冒充原始SHA。

GitHub 数据通过已连接的 GitHub 工具只读获取，未评论、发 issue 或创建 PR。第三方提出的 RFC/草案是讨论中的方向，不是主分支已支持的功能承诺。
