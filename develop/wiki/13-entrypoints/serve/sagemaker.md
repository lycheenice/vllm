[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > sagemaker

# sagemaker/（SageMaker 兼容适配）

> `serve/sagemaker/` 让 vLLM 被 AWS SageMaker JumpStart 直接托管：注册 SageMaker 期望的 `/invocations`、`/ping` 等标准路由，并在 app 装配期 `sagemaker_standards_bootstrap` 给整个 app 套上 SageMaker 风格包装。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `attach_router` | `vllm/entrypoints/serve/sagemaker/api_router.py:30` | 按 `supported_tasks`/`model_config` 注册 SageMaker 路由 |
| `sagemaker_standards_bootstrap` | `vllm/entrypoints/serve/sagemaker/api_router.py:101` | app 级包装：把 `/v1/...` 在 SageMaker 模式下映射到 `/invocations` 等 |

`attach_router`（`:30`）按 task 类型（generate/pooling/transcription）把对应 serving 对象接到 SageMaker invocation 路径；`sagemaker_standards_bootstrap`（`:101`）在 `build_app` 末尾调用（`api_server.py:293`），根据 `VLLM_SAGEMAKER_*`（待核实具体 env）决定是否启用包装。

## 为什么

- **零改部署**：SageMaker 容器期望固定路径（`/ping`、`/invocations`），vLLM 原生用 `/v1/*`；本包做路径与响应格式适配，让 vLLM 镜像直接被 SageMaker 拉起。
- **按 task 分派**：不同 SageMaker 模型类型（LLM/embedding/transcription）映射到不同 vLLM task，避免一端点承载多语义。
- **非侵入**：bootstrap 在 `build_app` 末尾包装，未启用 SageMaker 模式时近乎 no-op。

## 怎么做

通常由 SageMaker 部署脚本设 `VLLM_SAGEMAKER_*` env 后 `vllm serve`；客户端打 `/invocations`。详细字段（待核实）。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| attach_router | `vllm/entrypoints/serve/sagemaker/api_router.py:30` |
| sagemaker_standards_bootstrap | `vllm/entrypoints/serve/sagemaker/api_router.py:101` |
| build_app 调用点 | `vllm/entrypoints/openai/api_server.py:293` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`build_app` 挂 router + bootstrap。
- [pooling/README.md](../pooling/README.md)：embedding on SageMaker。
- [speech-to-text/README.md](../speech-to-text/README.md)：transcription on SageMaker。

## 历史版本演进

- **v0.7（SageMaker 适配）**：`/invocations`/`/ping` 路由 + bootstrap。
- **v0.9（serve/sagemaker 子包）**：抽到独立子包，按 task 分派。
- **v0.11/main**：transcription/translation on SageMaker 支持（待核实）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [openai/api-server.md](../openai/api-server.md)
