# ObservabilityConfig（observability.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ObservabilityConfig

源码：`vllm/config/observability.py`（约 159 行）。`ObservabilityConfig` 描述可观测性面：隐藏 metrics 退役、OTLP traces 端点、详细 trace 模块、KV cache 驻留指标采样、cudagraph 指标、layerwise NVTX、MFU、JIT monitor、迭代日志等。它是 `VllmConfig.observability_config`，被 `vllm/v1/metrics/`、`vllm/tracing/`、`EngineCore` 消费。

## 是什么

`@config` 装饰（`observability.py:17`）。`DetailedTraceModules = Literal["model","worker","all"]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `show_hidden_metrics_for_version` | `None` | 启用自该版本起隐藏的过期 Prometheus metrics（迁移逃逸口） |
| `otlp_traces_endpoint` | `None` | OpenTelemetry traces 目标 URL |
| `collect_detailed_traces` | `None` | `["model","worker","all"]`，详细 trace 模块（须配 `otlp_traces_endpoint`） |
| `kv_cache_metrics` | `False` | KV cache 驻留指标（lifetime/idle/reuse gap，采样，须 `--disable-log-stats` 未设） |
| `kv_cache_metrics_sample` | `0.01` | KV cache 指标采样率 (0,1]，默认 1% block |
| `cudagraph_metrics` | `False` | cudagraph 指标（padded/unpadded token 数、dispatch 模式频率） |
| `enable_layerwise_nvtx_tracing` | `False` | layerwise NVTX（与 cudagraph 不兼容） |
| `enable_mfu_metrics` | `False` | Model FLOPs Utilization 指标 |
| `enable_mm_processor_stats` | `False` | 多模态 processor 计时统计（仅内部/benchmark，非 CLI） |
| `enable_logging_iteration_details` | `False` | 迭代详情日志（context/generation 请求数与 token 数 + CPU 耗时） |
| `jit_monitor_mode` | `"warn"` | warmup 后 JIT 编译事件处理：`warn`/`error` |
| `jit_monitor_verbose` | `False` | 每次受监 JIT 编译打运行时详情（开销大，调试用） |

`cached_property`：`show_hidden_metrics`（按 `version._prev_minor_version_was` 判定）、`collect_model_forward_time`（`"model"`/`"all"` in `collect_detailed_traces`）、`collect_model_execute_time`（`"worker"`/`"all"`）。

校验器：`_validate_show_hidden_metrics_for_version`（`packaging.parse` 校验版本字符串）、`_validate_otlp_traces_endpoint`（检查 `is_tracing_available`，不可用 raise 含原始 traceback）、`_validate_collect_detailed_traces`（兼容逗号串→list）、`_validate_tracing_config`（`collect_detailed_traces` 须配 `otlp_traces_endpoint`）。

`compute_hash`：空 factors——可观测性不影响编译图形状。

## 为什么

- **metrics 退役治理**：`show_hidden_metrics_for_version` 让用户在迁移期临时启用已隐藏的旧 metrics，按版本号判定（`_prev_minor_version_was`），给 deprecation 缓冲。
- **traces 分级**：`otlp_traces_endpoint` 开 OTLP 导出；`collect_detailed_traces` 控制"昂贵且阻塞"的细粒度 trace（model forward / worker execute 时间），仅在 traces 开启时才允许。
- **KV 驻留指标采样**：`kv_cache_metrics` 采样 block 生命周期（驻留/idle/reuse gap），采样率 `kv_cache_metrics_sample` 控制开销。与 `KVEventsConfig`（push 事件流）互补——指标是 pull 采样，事件是 push 流。
- **cudagraph 指标**：`cudagraph_metrics` 观察实际 batch 落入哪个捕获尺寸、padded/unpadded 分布，调优 `cudagraph_capture_sizes`。
- **JIT monitor**：warmup 后的 JIT 编译影响性能，`jit_monitor_mode="error"` 让热路径 JIT 直接报错防回归；`jit_monitor_verbose` 调试用。
- **layerwise NVTX**：逐层/逐模块 NVTX range（输入输出形状），与 cudagraph 不兼容（cudagraph 捕获后 NVTX 失效）。

## 怎么做

- **OTLP traces**：`--otlp-traces-endpoint http://otel:4317 --collect-detailed-traces model`。
- **隐藏 metrics**：`--show-hidden-metrics-for-version 0.7`。
- **KV 指标**：`--kv-cache-metrics --kv-cache-metrics-sample 0.05`（5% 采样）。
- **cudagraph 指标**：`--cudagraph-metrics`。
- **MFU**：`--enable-mfu-metrics`。
- **JIT 严格**：`--jit-monitor-mode error`。
- **迭代日志**：`--enable-logging-iteration-details`。

## 与其它模块/系统配合

- **metrics 子系统（[`16-observability/`](../16-observability/README.md) 与 `vllm/v1/metrics/`）**：`show_hidden_metrics`/`kv_cache_metrics`/`cudagraph_metrics`/`enable_mfu_metrics` 驱动 Prometheus 指标注册与采集。
- **tracing（`vllm/tracing/`）**：`otlp_traces_endpoint`/`collect_detailed_traces` 驱动 OTLP span 导出；`is_tracing_available` 检查依赖。
- **EngineCore（[`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md)）**：`enable_logging_iteration_details` 在每步打迭代日志；`collect_model_forward_time`/`collect_model_execute_time` 在 EngineCore/Worker 计时。
- **KVEventsConfig（[kv-events-config.md](kv-events-config.md)）**：`kv_cache_metrics`（pull 采样）与 `kv_events_config`（push 事件）互补观测 KV block 生命周期。
- **ProfilerConfig（[profiler-config.md](profiler-config.md)）**：`enable_layerwise_nvtx_tracing` 与 `profiler="torch"` 都是性能分析手段，但 NVTX 是轻量标记，profiler 是全量捕获。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`observability_config.compute_hash()`（空）仍进顶层哈希（`VllmConfig.compute_hash` 中无条件调用 `self.observability_config.compute_hash()`），保持聚合完整性。

## 历史版本演进

- **v0.5/v0.6（v0）**：`show_hidden_metrics_for_version`/`otlp_traces_endpoint`/`collect_detailed_traces` 已存在；v0 metrics 体系。
- **v0.7（v1 落地）**：v1 metrics 重写；`collect_model_forward_time`/`collect_model_execute_time` cached_property；`_validate_tracing_config` 校验。
- **v0.8**：`kv_cache_metrics`/`kv_cache_metrics_sample`；`cudagraph_metrics`；`enable_mfu_metrics`。
- **v0.9**：`enable_layerwise_nvtx_tracing`；`enable_mm_processor_stats`（内部）；`jit_monitor_mode`/`jit_monitor_verbose`（warmup 后 JIT 监控）。
- **v0.10–main**：`enable_logging_iteration_details`；`_prev_minor_version_was` 版本判定维护；与 MRv2 的 cudagraph dispatch 指标协同。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [kv-events-config.md](kv-events-config.md) — KV 事件流（与本配置的采样指标互补）。
- [profiler-config.md](profiler-config.md) — profiler 与 NVTX 的关系。
- [vllm-config.md](vllm-config.md) — `observability_config.compute_hash()` 进顶层哈希。
- [../16-observability/README.md](../16-observability/README.md) — metrics/tracing 子系统消费方。
