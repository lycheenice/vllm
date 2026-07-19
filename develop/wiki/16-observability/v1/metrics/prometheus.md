[← Wiki 首页](../../README.md) > [可观测](../../README.md) > v1/metrics/prometheus

# Prometheus（registry / 多进程管理）

> 源码：`vllm/v1/metrics/prometheus.py`（82 行）

## 是什么

`prometheus.py` 是 vLLM 与 `prometheus_client` 库之间的薄封装，集中处理三件 registry 层面的事：

| 函数 | 行号 | 角色 |
|---|---|---|
| `setup_multiprocess_prometheus()` | `prometheus.py:17` | 若 `PROMETHEUS_MULTIPROC_DIR` 未设则创建 `tempfile.TemporaryDirectory` 并写环境变量；进程退出自动清理 |
| `get_prometheus_registry()` | `prometheus.py:39` | 决定返回全局 `REGISTRY` 还是新建 `CollectorRegistry` + `MultiProcessCollector`，依据是 `PROMETHEUS_MULTIPROC_DIR` 是否存在 |
| `unregister_vllm_metrics()` | `prometheus.py:55` | 遍历 `REGISTRY._collector_to_names`，把 `_name.startswith("vllm:")` 的 collector 反注册——给测试与 CI/CD 多次启动留干净状态 |
| `shutdown_prometheus()` | `prometheus.py:71` | 多进程模式下 `multiprocess.mark_process_dead(pid, path)`，让 collector 不再读取已死进程的文件 |

模块级全局 `_prometheus_multiproc_dir: tempfile.TemporaryDirectory | None` 持有临时目录引用，避免被 GC。

## 为什么

- **多进程模式必需**：当 uvicorn `--workers > 1` 或 `api_server_count > 1` 时，每个 worker 进程都有自己的 `prometheus_client` registry，但 `/metrics` 端点只能由一个进程暴露。`PROMETHEUS_MULTIPROC_DIR` 让各 worker 把指标写文件，主进程 `MultiProcessCollector` 聚合读出——这是 `prometheus_client` 官方多进程方案。
- **临时目录自动管理**：vLLM 不强求用户预设 `PROMETHEUS_MULTIPROC_DIR`，自己 `TemporaryDirectory` 创建并在解释器退出时自动清理；若用户已设则 warn "目录须在 vLLM 重启间清空，否则指标错乱"。
- **unregister 解决重复注册**：测试场景或 Jupyter 里多次构造 `PrometheusStatLogger` 会触发 `Duplicated timeseries` 错误。`unregister_vllm_metrics()` 在 `PrometheusStatLogger.__init__` 中调用（`loggers.py:422`），把前一轮的 vllm collector 全清，让新 logger 干净注册。
- **进程死亡标记**：`shutdown_prometheus()` 让多进程 collector 不再统计已死 worker，避免 `/metrics` 出现 stale 数据——vLLM 在 `EngineCore` 退出时调用（待核实具体调用点）。
- **不为 Ray Serve 用**：Ray 部署下 `RayPrometheusStatLogger` 替代默认 Prom logger，`_unregister_vllm_metrics` 是 no-op（见 [ray_wrappers.md](ray-wrappers.md)），因 Ray 自己管理 metrics lifecycle。

## 怎么做

**HTTP 暴露**：`/metrics` 端点由 13-entrypoints/serve 的 FastAPI app 通过 `prometheus_client.make_asgi_app()` 挂载（registry 来自 `get_prometheus_registry()`）。（具体挂载代码位置待核实）

**判断是否多进程模式**：

```python
from vllm.v1.metrics.prometheus import get_prometheus_registry
registry = get_prometheus_registry()
# 若 PROMETHEUS_MULTIPROC_DIR 已设，registry 是新的 CollectorRegistry + MultiProcessCollector
# 否则就是 prometheus_client.REGISTRY 全局
```

**测试 fixtures 调用**：

```python
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
unregister_vllm_metrics()  # 在构造 PrometheusStatLogger 前调
```

`PrometheusStatLogger.__init__` 内部已自动调用，故用户构造 logger 前无需手动调。

## 与其它模块/系统配合

- **[loggers.py](loggers.md)**：`PrometheusStatLogger.__init__` 调 `unregister_vllm_metrics()`；`_gauge_cls` 等用 `multiprocess_mode` 参数（`"mostrecent"`/`"sum"`）适配多进程聚合。
- **[ray_wrappers.md](ray-wrappers.md)**：`RayPrometheusStatLogger._unregister_vllm_metrics()` no-op。
- **[reader.py](reader.md)**：`get_metrics_snapshot()` 直接走全局 `REGISTRY.collect()`，故仅读非多进程 registry；多进程模式下 reader 行为（待核实）。
- **`13-entrypoints/serve/...`**：API server 启动时调 `setup_multiprocess_prometheus()`；`/metrics` ASGI app 绑 `get_prometheus_registry()`。
- **`prometheus_client` 库**：`REGISTRY`、`CollectorRegistry`、`multiprocess.MultiProcessCollector`、`multiprocess.mark_process_dead` 均为库原生 API。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 `vllm/engine/llm_engine.py` 内联 prometheus 注册；多进程模式简单支持，`setup_multiprocess_prometheus` 在 `vllm/engine/arg_utils.py` 启动流程中调（待核实）。
- **v0.7（v1 落地）**：抽出 `vllm/v1/metrics/prometheus.py` 模块；`unregister_vllm_metrics` 由 `PrometheusStatLogger.__init__` 调用。
- **v0.8–main**：`shutdown_prometheus` + `mark_process_dead`；`get_prometheus_registry` 抽象。具体版本归属（待核实）。

[← 返回可观测首页](../../README.md)

## 参见

- [loggers.md](loggers.md) — `PrometheusStatLogger` 调用本模块。
- [reader.md](reader.md) — 反向读取 registry 的快照。
- [ray-wrappers.md](ray-wrappers.md) — Ray Serve 替代后端的 no-op unregister。
