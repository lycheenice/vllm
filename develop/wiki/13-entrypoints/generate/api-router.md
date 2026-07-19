[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Generate](README.md) > api_router

# generate/api_router.py（生成路由注册 + state init）

> `generate/api_router.py` 把生成类 serving 的路由（chat/completion/responses/generative_scoring/scale-out）统一挂到 app，并提供 `init_generate_state` 实例化各 serving 对象。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `register_generate_api_routers` | `vllm/entrypoints/generate/api_router.py:19` | 挂 chat/completion/responses/generative_scoring 子路由 |
| `init_generate_state` | `:49` | 实例化 `OpenAIServingChat`/`OpenAIServingCompletion`/`OpenAIServingResponses`/`ServingGenerativeScoring` 挂 state |

`register_generate_api_routers`（`:19`）顺序调各子包 `attach_router(app)`：chat_completion → completion → responses → generative_scoring（待核实顺序与 scale_out 是否在此）。

`init_generate_state`（`:49`）从 `engine_client`/`state`/`args`/`request_logger`/`supported_tasks` 构造：

- `OpenAIServingChat`（含 `OpenAIServingChatBatch`）
- `OpenAIServingCompletion`
- `OpenAIServingResponses`
- `ServingGenerativeScoring`

并挂 `state.openai_serving_chat`/`state.openai_serving_completion`/`state.openai_serving_responses` 等。各 handler 的 `__init__` 参数（renderer/parser/usage flags）在此透传。

## 为什么

- **一处装配**：`build_app`/`init_app_state` 只调 `register_generate_api_routers`/`init_generate_state` 两个函数，生成子包内部组装顺序自管。
- **按 task 守卫**：仅在 `"generate" in supported_tasks` 时挂载（`api_server.py:203`），非生成模型零路由。
- **scale-out 联动**：`init_generate_state` 后紧跟 `init_scale_out_state`（`api_server.py:404`），让 render/derender 端点可用。

## 怎么做

见 [generate/README](README.md)。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| register_generate_api_routers | `vllm/entrypoints/generate/api_router.py:19` |
| init_generate_state | `vllm/entrypoints/generate/api_router.py:49` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`build_app`/`init_app_state` 调用。
- [base-serves.md](base-serves.md)：构造的 serving 类基类。
- [scale-out/README.md](../scale-out/README.md)：紧随的 `init_scale_out_state`。

## 历史版本演进

- **v0.9（抽 generate/api_router）**：从 api_server 抽出。
- **v0.10（responses + generative_scoring）**：扩展构造。
- **main**：`OpenAIServingChatBatch` 构造。

## 参见

- [← 返回 Generate 首页](README.md)
- [base-serves.md](base-serves.md)
