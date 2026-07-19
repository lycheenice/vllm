[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > instrumentator

# instrumentator/（Prometheus /metrics /health /version）

> `serve/instrumentator/` 暴露可观测端点：Prometheus `/metrics`、`/health`（就绪/存活）、`/version`、`/server_load`（服务端负载指标），并挂 FastAPI instrumentator 采集请求级指标。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `register_instrumentator_api_routers` | `vllm/entrypoints/serve/instrumentator/__init__.py:7` | 聚合注册 |
| `metrics.py: _patch_instrumentator_route_walk` | `vllm/entrypoints/serve/instrumentator/metrics.py:17` | 修补 instrumentator 路由遍历 |
| `metrics.py: PrometheusResponse` / `attach_router` | `:52/56` | `/metrics` 端点（多进程兼容） |
| `health.py: health` | `vllm/entrypoints/serve/instrumentator/health.py:23` | `/health` 就绪探针 |
| `basic.py: get_server_load_metrics`/`show_version` | `vllm/entrypoints/serve/instrumentator/basic.py:31/54` | `/server_load`、`/version` |
| `offline_docs.py` | — | 离线 docs 静态资源（待核实） |

`health`（`health.py:23`）经 `engine_client` 查询引擎状态返 200/503，K8s liveness/readiness 可用。`show_version` 返回 vLLM 版本与模型名。`get_server_load_metrics` 在 `enable_server_load_tracking` 时返回当前 `server_load_metrics`。

## 为什么

- **多进程 Prometheus**：`setup_multiprocess_prometheus`（`cli/serve.py:269`）在多 api-server 模式下设 `prometheus_multiproc_dir`，`PrometheusResponse` 聚合各 worker 指标，避免单进程视角缺失。
- **K8s 原生探针**：`/health` 让 K8s 不依赖 OpenAI 端点判断就绪，扩缩容更准。
- **服务端负载可见**：`/server_load` 暴露 `load_aware_call` 用的负载值，便于客户端做退避。
- **统一注册**：`register_instrumentator_api_routers` 一次性挂所有可观测端点，`build_app` 第一步调用。

## 怎么做

```bash
curl http://localhost:8000/metrics        # Prometheus 文本
curl http://localhost:8000/health         # 200/503
curl http://localhost:8000/version        # 版本
curl http://localhost:8000/server_load    # 负载值（需 enable_server_load_tracking）
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| register_instrumentator_api_routers | `vllm/entrypoints/serve/instrumentator/__init__.py:7` |
| PrometheusResponse | `vllm/entrypoints/serve/instrumentator/metrics.py:52` |
| metrics attach_router | `vllm/entrypoints/serve/instrumentator/metrics.py:56` |
| health | `vllm/entrypoints/serve/instrumentator/health.py:23` |
| server_load | `vllm/entrypoints/serve/instrumentator/basic.py:31` |
| version | `vllm/entrypoints/serve/instrumentator/basic.py:54` |

## 与其它模块/系统配合

- [可观测-metrics](../../16-observability/README.md)：vLLM 全部 `vllm:*` 指标的 HTTP 暴露口。
- [utils.md](utils.md)：`server_load_metrics` 与 `load_aware_call` 配合。
- [openai/api-server.md](../openai/api-server.md)：`build_app` 第一步挂载。
- [cli/serve-cmd.md](../cli/serve-cmd.md)：多 api-server 时 `setup_multiprocess_prometheus`。

## 历史版本演进

- **v0.7（/metrics + /health）**：基础可观测端点。
- **v0.9（serve/instrumentator 子包）**：抽到独立子包；多进程 Prometheus 聚合。
- **v0.10（/version + /server_load）**：版本与负载端点；`enable_server_load_tracking`。
- **main**：`_patch_instrumentator_route_walk` 修路由遍历（待核实 issue）；offline docs 静态资源。

## 参见

- [← 返回 serve/ 首页](README.md)
- [可观测-metrics](../../16-observability/README.md)
- [utils.md](utils.md)
