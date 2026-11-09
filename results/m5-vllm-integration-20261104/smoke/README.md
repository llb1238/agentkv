# smoke：原始采集

配置与源码见[run.json](run.json)、[服务端清单](server-manifest.json)、[trial-run.json](trial-run.json)。完整回调gzip保存原字节，可从新ReqContext复原参数、状态和内部clock。requests.parquet的prompt数组本机保留，Git追踪仅删该列的request-metadata.parquet。

[总报告](../README.md) · [CPU回放](../smoke-replay/run.json)
