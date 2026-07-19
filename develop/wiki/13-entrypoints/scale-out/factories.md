[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Scale-out](README.md) > factories

# scale_out/factories.py（scale-out 装配）

> `factories.py` 把 render/derender/token-in-out 路由与 state 装配集中：`init_render_state`（render-only）、`init_scale_out_state`（推理节点）、`register_scale_out_api_routers`（按 task 挂路由）。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `init_render_state` | `vllm/entrypoints/scale_out/factories.py:19` | render-only server state（无 engine） |
| `init_scale_out_state` | `:39` | 推理节点 state（render + derender + tokens） |
| `register_scale_out_api_routers` | `:61` | 按 `supported_tasks`（含 `generate`/`render`）挂三段路由 |

`register_scale_out_api_routers`（`:61`）按是否含 `render`/`generate` 决定挂 render 端点、token-in/out 端点、derender 端点。`init_scale_out_state`（`:39`）在推理节点构造 `ServingRender`/`ServingTokens`/`ServingDerender`（render/derender 用于"单体模式"内联预处理/后处理）；`init_render_state`（`:19`）在纯 render server 仅构造 `ServingRender`。

## 为什么

- **单/拆分双形态**：单体 server 与拆分部署复用同一 factories，仅 state init 不同；serving 类相同。
- **按 task 守卫**：`render` task 走 render-only 装配，`generate` task 走完整装配，互不污染。

## 怎么做

见 [README](README.md) 部署形态。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| init_render_state | `vllm/entrypoints/scale_out/factories.py:19` |
| init_scale_out_state | `vllm/entrypoints/scale_out/factories.py:39` |
| register_scale_out_api_routers | `vllm/entrypoints/scale_out/factories.py:61` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`build_app`/`init_render_app_state` 调用。
- [render.md](render.md)/[token-in-token-out.md](token-in-token-out.md)/[derender.md](derender.md)：装配的 serving 类。

## 历史版本演进

- **v0.10（引入）**：三段 factories。
- **main**：render-only state 复用。

## 参见

- [← 返回 Scale-out 首页](README.md)
- [render.md](render.md)
