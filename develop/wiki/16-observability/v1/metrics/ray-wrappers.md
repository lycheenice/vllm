[← Wiki 首页](../../README.md) > [可观测](../../README.md) > v1/metrics/ray-wrappers

# Ray Wrappers（Ray Serve metrics 适配层）

> 源码：`vllm/v1/metrics/ray_wrappers.py`（216 行）

## 是什么

`ray_wrappers.py` 让 vLLM 在 **Ray Serve** 部署下用 `ray.util.metrics` 替代 `prometheus_client` 输出同一套 vLLM 指标。它通过"同 API 包装类"实现：包装 `ray.util.metrics.Gauge/Counter/Histogram`，让 `PrometheusStatLogger` 不改一行代码就能切换后端。

核心组件：

| 类 | 行号 | 角色 |
|---|---|---|
| `RayPrometheusMetric` (base) | `ray_wrappers.py:31` | 共用骨架：自动加 `ReplicaId` tag、`_get_tag_keys` 把 labelnames 加 `ReplicaId` 末尾、`labels()` 返回带标签的 clone、`_get_sanitized_opentelemetry_name` 把 `:` 替换为 `_`（OTel 命名规范不允许冒号） |
| `_get_replica_id()` | `ray_wrappers.py:21` | 从 `ray_serve.get_replica_context().replica_id.unique_id()` 取当前 replica ID；非 Serve 上下文返回 `None` |
| `RayGaugeWrapper` | `ray_wrappers.py:85` | 包装 `ray.util.metrics.Gauge`；`set(v)`/`set_to_current_time()`（Ray 没原生后者，用 `time.time()` 替）；忽略 `multiprocess_mode`（Ray 按 WorkerId 自治） |
| `RayCounterWrapper` | `ray_wrappers.py:119` | 包装 `ray.util.metrics.Counter`；`inc(v)` 在 `v==0` 时 noop（Ray Counter 不接受 0） |
| `RayHistogramWrapper` | `ray_wrappers.py:144` | 包装 `ray.util.metrics.Histogram`；`buckets` 映射为 Ray 的 `boundaries` |
| `RaySpecDecodingProm` | `ray_wrappers.py:171` | `SpecDecodingProm` 子类，`_counter_cls = RayCounterWrapper` |
| `RayKVConnectorProm` | `ray_wrappers.py:181` | `KVConnectorProm` 子类，三 cls 全换 Ray 版 |
| `RayPerfMetricsProm` | `ray_wrappers.py:193` | `PerfMetricsProm` 子类，Counter 换 Ray 版 |
| `RayPrometheusStatLogger` | `ray_wrappers.py:203` | `PrometheusStatLogger` 子类，五个 `_cls` 全换，`_unregister_vllm_metrics` no-op |

`RayPrometheusStatLogger` 通过 `_gauge_cls`/`_counter_cls`/`_histogram_cls`/`_spec_decoding_cls`/`_kv_connector_cls`/`_perf_metrics_cls` 类属性注入替代 cls——`PrometheusStatLogger.__init__` 中所有 `self._gauge_cls(...)`/`self._counter_cls(...)` 调用走多态自动应用 Ray 后端。

## 为什么

- **Ray Serve 自带 OTel/exporter 栈**：Ray Serve 已有 metrics 聚合到 dashboard / OTel collector 的能力，重复用 `prometheus_client` 会产生两套 metric pipeline 冲突。
- **API 兼容性**：vLLM 的 `PrometheusStatLogger` 用 `prometheus_client` API（`labels(*lvs)` / `inc(v)` / `set(v)` / `observe(v)`），Ray `util.metrics` API 略不同（`set(v, tags=...)`）——包装类适配之，避免改 logger 代码。
- **ReplicaId 自动 tag**：Ray Serve 部署有多个 replica，必须按 replica 区分指标。包装类自动把 `ReplicaId` 加为最后一个 tag，所有指标都带它——用户在 dashboard 可按 replica 切片。
- **OTel 命名规范**：OTel metric 名禁止 `:`，vLLM 全用 `vllm:xxx`，故 `_get_sanitized_opentelemetry_name` 把 `:` 替换为 `_`（`vllm:xxx` → `vllm_xxx`）。参考 OpenTelemetry C++ metadata validator 与 Ray metric.cc。
- **多进程模式忽略**：Ray 按 WorkerId 自治，`multiprocess_mode="mostrecent"/"sum"` 不适用；包装类 `del multiprocess_mode`，让聚合由可观测性后端（Prometheus/Grafana）自行处理。
- **Counter 0 noop**：`prometheus_client.Counter.inc(0)` 是合法的 noop，但 Ray Counter 不接受 0，故显式 short-circuit。
- **统一 logger 子类化**：用户通过 `STAT_LOGGER_PLUGINS_GROUP` 注册 `RayPrometheusStatLogger`，由 [loggers.py](loggers.md) 的 `StatLoggerManager` 检测到 `isinstance(x, PrometheusStatLogger)` 即跳过默认 Prom logger，实现后端替换。

## 怎么做

**Ray Serve 部署模式**（vLLM 已在 Ray Serve integration 中默认启用 RayPrometheusStatLogger，具体注入点在 `vllm/entrypoints/openai/api_server.py` 或 Ray Serve 部署模板内，待核实）：

```python
# 用户在 Ray Serve 部署 vLLM 时通常会azel
from vllm.entrypoints.openai.api_server import build_app  # 待核实
# Ray Serve 自动用 RayPrometheusStatLogger，无需手动配置
```

**手动注册插件**（若想强制 Ray 后端而不走 Ray Serve 自动检测）：

```toml
# pyproject.toml
[project.entry-points.vllm.stat_loggers_plugins]
ray = "vllm.v1.metrics.ray_wrappers:RayPrometheusStatLogger"
```

启动时 `load_stat_logger_plugin_factories()` 发现 `RayPrometheusStatLogger`（继承 `PrometheusStatLogger` 而后者继承 `AggregateStatLoggerBase`），构造为 global_stat_logger，因 `isinstance(x, PrometheusStatLogger)` 为 True 故 `custom_prometheus_logger=True`，默认 Prom logger 不再构造。

**自定义副本标签**：包装类把 `ReplicaId` 自动从 `ray.serve.get_replica_context()` 取，非 Serve 上下文为空字符串——故非 Serve 环境用 Ray 后端指标会全部带空 replica tag（不推荐此场景）。

## 与其它模块/系统配合

- **[loggers.py](loggers.md)**：`StatLoggerManager` 通过插件机制接入；`PrometheusStatLogger` 的类属性注入是多态点。
- **[prometheus.md](prometheus.md)**：`_unregister_vllm_metrics` no-op，因 Ray 不走 `prometheus_client` registry。
- **`06-sampling-decoding/speculative-decoding/metrics.md`**：`RaySpecDecodingProm` 是 `SpecDecodingProm` 的 Ray 后端版。
- **KV connector metrics**（`vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`）：`RayKVConnectorProm` 同理。
- **[perf.md](perf.md)**：`RayPerfMetricsProm` 同理。
- **13-entrypoints**：Ray Serve entrypoint 配合注入。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 已有 `vllm/engine/ray_utils.py` 或类似 Ray metrics 适配（待核实），用 `ray.util.metrics` 包装；早期无 `ReplicaId` tag。
- **v0.7（v1 落地）**：迁到 `vllm/v1/metrics/ray_wrappers.py`；引入 `ReplicaId` tag；`OTel sanitize` 命名处理。
- **v0.8**：`RayKVConnectorProm`/`RayPerfMetricsProm`/`RaySpecDecodingProm` 子类随父类新增字段同步增。
- **v0.9–main**：稳定；`labels()` 返回 clone 防共享 tag 状态。具体版本归属（待核实）。

[← 返回可观测首页](../../README.md)

## 参见

- [loggers.md](loggers.md) — 插件加载机制与 logger 多态。
- [prometheus.md](prometheus.md) — registry 在 Ray 模式下不参与。
- [perf.md](perf.md) / `ray_wrappers.md` 中的 `RayPerfMetricsProm`。
