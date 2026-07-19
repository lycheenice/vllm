[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > elastic_ep

# elastic_ep/（弹性专家并行扩缩容）

> `serve/elastic_ep/` 实现弹性专家并行（Elastic EP）的 server 侧门控：扩缩容期间用 `ScalingMiddleware` 拒绝新请求，并暴露 `/scale_elastic_ep`、`/is_scaling_elastic_ep` 端点触发/查询扩缩容状态。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `get_scaling_elastic_ep` / `set_scaling_elastic_ep` | `vllm/entrypoints/serve/elastic_ep/middleware.py:13/17` | 全局扩缩容状态读写 |
| `ScalingMiddleware` | `vllm/entrypoints/serve/elastic_ep/middleware.py:22` | HTTP middleware：扩缩容中返 503 |
| `engine_client` 依赖 | `vllm/entrypoints/serve/elastic_ep/api_router.py:25` | 从 request 取 engine_client |
| `scale_elastic_ep` | `vllm/entrypoints/serve/elastic_ep/api_router.py:42` | `POST /scale_elastic_ep` → `engine_client.collective_rpc("scale_elastic_ep")` |
| `is_scaling_elastic_ep` | `vllm/entrypoints/serve/elastic_ep/api_router.py:91` | `GET /is_scaling_elastic_ep` 状态查询 |
| `attach_router` | `:95` | 注册路由 |

`ScalingMiddleware`（`:22`）在 `build_app` 必挂（`api_server.py:263`）：每个请求前检查 `get_scaling_elastic_ep()`，若正在扩缩容则直接 503，避免在权重/路由瞬变期产生错误输出。

## 为什么

- **扩缩容期间隔离**：弹性 EP 改变专家切分与权重布局，期间推理结果不确定，middleware 让客户端收到明确 503 而非错误 token。
- **外部编排**：`/scale_elastic_ep` 让 K8s/operator 触发扩缩容，`/is_scaling_elastic_ep` 让就绪探针在扩缩容期摘流。
- **与 Elastic EP 引擎配合**：`collective_rpc("scale_elastic_ep")` 广播到 EngineCore，触发权重重排（见 [07-distributed](../../07-distributed/README.md)）。
- **仅单 API server**：`cli/serve.py:131` 在 `enable_elastic_ep` 时 cap `api_server_count=1`，因为扩缩容状态是进程内全局，多 API server 难同步。

## 怎么做

```bash
curl -X POST http://localhost:8000/scale_elastic_ep -d '{...}'   # 触发
curl http://localhost:8000/is_scaling_elastic_ep                  # 查询
# middleware 在扩缩容期返 503
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| get/set_scaling_elastic_ep | `vllm/entrypoints/serve/elastic_ep/middleware.py:13/17` |
| ScalingMiddleware | `vllm/entrypoints/serve/elastic_ep/middleware.py:22` |
| scale_elastic_ep | `vllm/entrypoints/serve/elastic_ep/api_router.py:42` |
| is_scaling_elastic_ep | `vllm/entrypoints/serve/elastic_ep/api_router.py:91` |
| attach_router | `vllm/entrypoints/serve/elastic_ep/api_router.py:95` |
| build_app 必挂 | `vllm/entrypoints/openai/api_server.py:263` |

## 与其它模块/系统配合

- [07-distributed](../../07-distributed/README.md)：Elastic EP 引擎层、EPLB、权重迁移。
- [openai/api-server.md](../openai/api-server.md)：`build_app` 挂 middleware + 路由（仅 generate task）。
- [cli/serve-cmd.md](../cli/serve-cmd.md)：`enable_elastic_ep` cap `api_server_count=1`。
- [可观测-metrics](../../16-observability/README.md)：扩缩容状态指标（待核实）。

## 历史版本演进

- **v0.10.x（Elastic EP 引入）**：`ScalingMiddleware` + `/scale_elastic_ep`/`/is_scaling_elastic_ep`；与 `enable_elastic_ep` 配合；单 API server 限制。
- **main**：扩缩容 RPC 参数协商；与 weight transfer（[dev.md](dev.md) rlhf）协同（待核实）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [07-distributed](../../07-distributed/README.md)
- [openai/api-server.md](../openai/api-server.md)
