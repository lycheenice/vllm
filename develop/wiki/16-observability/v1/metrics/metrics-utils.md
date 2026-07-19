[← Wiki 首页](../../README.md) > [可观测](../../README.md) > v1/metrics/utils

# Metrics Utils（per-engine label 工具）

> 源码：`vllm/v1/metrics/utils.py`（19 行）

## 是什么

`utils.py` 是 metrics 子系统的极小公共工具，仅两个导出：

| 名 | 行号 | 角色 |
|---|---|---|
| `PromMetric` | `utils.py:8` | `TypeAlias = Gauge | Counter | Histogram`（来自 `prometheus_client`） |
| `create_metric_per_engine(metric, per_engine_labelvalues)` | `utils.py:11` | 对每个 `engine_idx` 调 `metric.labels(*labelvalues[idx])`，返回 `dict[int, PromMetric]`；让 `PrometheusStatLogger` 用统一接口拿 per-engine 子 metric |

实现：

```python
def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]:
    """Create a labeled metric child for each engine index."""
    return {
        idx: metric.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
```

## 为什么

- **消除重复模板**：`PrometheusStatLogger` 注册了 ~30 个指标，每个都需要按 `engine_idx` 拿带 `model_name`/`engine` label 的子 metric。无此函数则每个指标都需写 4 行字典推导；此函数把模板抽成一行。

  ```python
  # 没有此函数的写法（~30 处重复）
  self.gauge_kv_cache_usage = {
      idx: gauge_kv_cache_usage.labels(model_name, str(idx))
      for idx in engine_indexes
  }
  # 有了之后
  self.gauge_kv_cache_usage = create_metric_per_engine(
      gauge_kv_cache_usage, per_engine_labelvalues
  )
  ```

- **统一 label 语义**：所有 vLLM 指标共享 `labelnames = ["model_name", "engine"]` 这两个 label，构造 `per_engine_labelvalues = {idx: [model_name, str(idx)] for idx in engine_indexes}` 后传给此函数即可——保证不同指标 label 顺序、命名一致。
- **多 label 子类的复用**：`vllm:num_requests_waiting_by_reason` 多一个 `reason` label，`vllm:prompt_tokens_by_source` 多 `source`——这些场景通过先构造 `per_engine_labelvalues_with_reason = {idx: labelvalues + [reason] ...}` 再调 `create_metric_per_engine` 复用同一函数（见 `loggers.py:489-496`）。
- **TypeAlias 简化签名**：`PromMetric` 让函数签名 `dict[int, PromMetric]` 而非 `dict[int, Gauge | Counter | Histogram]`，更短。

## 怎么做

**调用模板**（在 `PrometheusStatLogger.__init__` 与 `*_cls` 子类中通用）：

```python
labelnames = ["model_name", "engine"]
model_name = vllm_config.model_config.served_model_name
per_engine_labelvalues = {idx: [model_name, str(idx)] for idx in engine_indexes}

gauge = self._gauge_cls(
    name="vllm:num_requests_running",
    documentation="...",
    multiprocess_mode="mostrecent",
    labelnames=labelnames,
)
self.gauge_scheduler_running = create_metric_per_engine(gauge, per_engine_labelvalues)

# 多 label 场景
labelnames_extra = labelnames + ["reason"]
gauge_reason = self._gauge_cls(name="vllm:num_requests_waiting_by_reason",
                                labelnames=labelnames_extra, ...)
per_engine_labelvalues_reason = {
    idx: lvs + ["capacity"] for idx, lvs in per_engine_labelvalues.items()
}
self.gauge_waiting_by_reason["capacity"] = create_metric_per_engine(
    gauge_reason, per_engine_labelvalues_reason
)
```

**自定义 stat logger 复用**：用户实现自己的 `PrometheusStatLogger` 子类时同样调此函数，确保 label 顺序一致。

## 与其它模块/系统配合

- **[loggers.py](loggers.md)**：`PrometheusStatLogger.__init__` 中 ~30 处调用；`labelnames = ["model_name", "engine"]` 是隐式约定。
- **[ray_wrappers.md](ray-wrappers.md)**：Ray 包装类同样接 `labelnames` 与 `per_engine_labelvalues`，但额外注入 `ReplicaId`——`create_metric_per_engine` 在 Ray 路径下返回的是 `RayGaugeWrapper` 等（因 `metric.labels(*lvs)` 走包装类的 `labels()` 方法）。
- **[perf.md](perf.md)**：`PerfMetricsProm`/`PerfMetricsLogging` 同样用此函数。
- **`06-sampling-decoding/speculative-decoding/metrics.md`**、KV connector metrics：所有 `*Prom` 子组件都按此模式构造 per-engine metric。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 把 per-engine 展开逻辑内联在 `PrometheusStatLogger.__init__`（无独立工具函数）。
- **v0.7（v1 落地）**：抽出 `vllm/v1/metrics/utils.py::create_metric_per_engine`；引入 `PromMetric` TypeAlias。
- **v0.8–main**：稳定，函数签名未变；调用点随 metric 增加而增多。具体版本归属（待核实）。

[← 返回可观测首页](../../README.md)

## 参见

- [loggers.md](loggers.md) — 主要消费者。
- [ray-wrappers.md](ray-wrappers.md) — Ray 后端下的多态复用。
