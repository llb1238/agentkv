# vLLM完整前缀元数据：后端验证

`4749dd5961841243596559b10ba0a22c48980c45`（fork的`agentkv`分支，已推送）。

共用key推导函数；原位置表与新实现跨创建、重复调度、decode的对照通过；ReqContext字段赋值通过。connector scheduler、CPU manager、worker、tiering等所列原有单元范围：**1395 passed / 10 skipped / 0 failed**。跳过项原有条件保留，详见JUnit。

另行尝试的实模型集成模块`test_offloading_connector.py`需要缺失的Llama权重，未通过，不能计入验收成功；初次在线尝试还遇到镜像TLS故障。原始失败日志一并保留。未改测试配置或替换该模型绕过失败。

[run.json](run.json) · [完整单元日志](../../../docs/EVIDENCE.md#runtime-logs) · [JUnit](../../../docs/EVIDENCE.md#other-evidence) · [旧函数对照](../../../docs/EVIDENCE.md#runtime-logs) · [pre-commit](../../../docs/EVIDENCE.md#runtime-logs)
