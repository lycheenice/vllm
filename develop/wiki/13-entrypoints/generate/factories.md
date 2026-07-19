[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Generate](README.md) > factories

# generate/factories.py（invocation 类型映射）

> `generate/factories.py` 仅提供 `get_generate_invocation_types`，把生成任务名映射到引擎侧 invocation 类型，供 `AsyncLLM` 分派生成/beam_search。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `get_generate_invocation_types` | `vllm/entrypoints/generate/factories.py:16` | 任务→invocation 类型映射 |

与 [pooling/factories.md](../pooling/factories.md) 对称，但生成子系统的路由/state init 在 `api_router.py`，factories 仅留映射函数（历史原因，待核实是否将合并）。

## 为什么

- **任务/引擎解耦**：serving 层只报任务类型，引擎按 invocation 类型分派 generate/beam_search，互不感知。

## 怎么做

`AsyncLLM`/调度器按 `get_generate_invocation_types` 返回值选生成路径。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| get_generate_invocation_types | `vllm/entrypoints/generate/factories.py:16` |

## 与其它模块/系统配合

- [api-router.md](api-router.md)：生成路由。
- [引擎核心-AsyncLLM](../../01-engine-core/async-llm-frontend.md)：invocation 分派。

## 历史版本演进

- **v0.10（引入）**：与 pooling factories 对称化。
- **main**：维持薄函数（待核实是否会扩展）。

## 参见

- [← 返回 Generate 首页](README.md)
- [api-router.md](api-router.md)
