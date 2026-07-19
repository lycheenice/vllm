[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > factories

# pooling/factories.py（注册与 state 初始化）

> `factories.py` 是 pooling 子系统的"装配器"：创建各 task 的 io_processor、按 supported_tasks 挂路由、初始化 state、提供 task→invocation 类型映射。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `init_pooling_io_processors` | `vllm/entrypoints/pooling/factories.py:38` | 实例化 embed/classify/score/pooling 的 io_processor，返回映射 |
| `register_pooling_api_routers` | `:104` | 按 supported_tasks 调各 task 的 `attach_router` |
| `init_pooling_state` | `:137` | 构造各 `Serving*` 对象挂到 `app.state` |
| `get_pooling_invocation_types` | `:214` | 任务名 → 引擎 invocation 类型（`embed`/`classify`/`score`/`pooling`） |

`init_pooling_state`（`:137`）从 `engine_client` + `args` + `request_logger` + `supported_tasks` 出发，构造 `ServingEmbedding`/`ServingClassification`/`ServingScores`/`ServingPooling` 并挂 `state.serving_*`；`init_pooling_io_processors`（`:38`）按 `model_config` 与各 task 协议构造 io_processor 供 serving 使用。

`register_pooling_api_routers`（`:104`）遍历 supported_tasks，对每个 pooling task 调对应 `attach_router(app, ...)`；`build_app` 在 `any(task in POOLING_TASKS ...)` 时调用（`api_server.py:228`）。

## 为什么

- **装配集中**：把"哪些 task → 哪些 serving → 哪些路由 → 哪些 io_processor"集中在 factories，`build_app`/`init_app_state` 各调一次，避免散落。
- **task 子集按需挂**：模型可能只支持 embed 不支持 classify，按 `supported_tasks` 精准挂载，无 404 噪音。
- **invocation 类型解耦**：`get_pooling_invocation_types` 让引擎侧（`AsyncLLM.pooling`/`encode`）按统一 invocation 类型分派，serving 与引擎解耦。

## 怎么做

`init_app_state`（`api_server.py:415`）调 `init_pooling_state`；`build_app`（`:228`）调 `register_pooling_api_routers`。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| init_pooling_io_processors | `vllm/entrypoints/pooling/factories.py:38` |
| register_pooling_api_routers | `vllm/entrypoints/pooling/factories.py:104` |
| init_pooling_state | `vllm/entrypoints/pooling/factories.py:137` |
| get_pooling_invocation_types | `vllm/entrypoints/pooling/factories.py:214` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：调用方。
- [base.md](base.md)：构造的 serving 类基类。
- [多模态](../../11-multimodal/README.md)：io_processor 注入 MM 处理。

## 历史版本演进

- **v0.9（重构引入）**：按 task 拆 factories。
- **v0.11/main**：`get_pooling_invocation_types`；多模态 io_processor。

## 参见

- [← 返回 Pooling 首页](README.md)
- [base.md](base.md)
- [openai/api-server.md](../openai/api-server.md)
