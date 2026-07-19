[← Wiki 首页](../README.md) > [API 入口](../README.md) > serve/

# serve/（通用 serving 工具集）

> `vllm/entrypoints/serve/` 是所有 HTTP 入口共享的"公共基建"：serving 基类、middleware（认证/CORS/X-RequestId/Scaling）、错误处理、Prometheus instrumentator、health、tokenize、LoRA HTTP 接口、profile、SageMaker 适配、Elastic EP、dev 实验端点。`openai/api_server.build_app` 第一步就调 `register_vllm_serve_api_routers` 挂这些路由。

## 是什么

| 子包/文件 | 核心 | 职责 |
|---|---|---|
| `__init__.py` | `register_vllm_serve_api_routers` / `register_vllm_dev_api_routers`（`vllm/entrypoints/serve/__init__.py:11/35`） | 路由聚合 |
| `engine/serving.py` | `BaseServing`（`vllm/entrypoints/serve/engine/serving.py:29`） | 所有 serving 类基类：`_check_model`、`create_error_response` 等 |
| `utils/` | `api_utils.py`/`server_utils.py`/`request_logger.py`/`ssl.py`/`constants.py`/`error_response.py`/`fingerprint.py`/`orca_metrics.py`/`tool_calls_utils.py` | server 工具 |
| `instrumentator/` | `metrics.py`/`health.py`/`basic.py`/`offline_docs.py` | Prometheus `/metrics`、`/health`、`/version`、`/server_load` |
| `tokenize/` | `ServingTokenization`（`vllm/entrypoints/serve/tokenize/serving.py:32`） | `/v1/tokenize`、`/v1/detokenize` |
| `lora/` | `api_router.attach_router` | `/v1/load_lora_adapter`、`/v1/unload_lora_adapter` |
| `profile/` | `start_profile`/`stop_profile` | `/start_profile`、`/stop_profile` |
| `sagemaker/` | `sagemaker_standards_bootstrap`（`vllm/entrypoints/serve/sagemaker/api_router.py:101`） | SageMaker JumpStart 兼容路由 |
| `elastic_ep/` | `ScalingMiddleware`（`vllm/entrypoints/serve/elastic_ep/middleware.py:22`） | 弹性 EP 扩缩容状态门控 |
| `dev/` | cache/rlhf/rpc/server_info/sleep | 实验端点（仅 `VLLM_SERVER_DEV_MODE`） |

`register_vllm_serve_api_routers`（`:11`）顺序：instrumentator → lora → profile → tokenize。`register_vllm_dev_api_routers`（`:35`）警告安全风险后挂 cache/rlhf/rpc/server_info/sleep。

`BaseServing`（`engine/serving.py:29`）持有 `models`/`model_config`/`request_logger`，提供 `_check_model`（`:40`，含 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 动态 LoRA）、`_is_model_supported`（`:70`，`VLLM_SKIP_MODEL_NAME_VALIDATION` 可跳过）、`create_error_response`（`:77`）。各具体 serving 类（`GenerateBaseServing`/`PoolingBaseServing`/`SpeechToTextBaseServing`）继承它。

## 为什么

- **公共路由集中**：health/metrics/tokenize/lora/profile 是任何 task 都需要的，集中到 `serve/` 一次挂载，避免每个 task 包重复。
- **middleware 标准化**：`AuthenticationMiddleware`（`server_utils.py:45`，校验 Bearer）、`XRequestIdMiddleware`（`:96`，透传/生成 X-Request-Id）、`ScalingMiddleware`（弹性 EP 扩缩容期拒请求）统一在 server 装配期挂入，serving 类无感。
- **错误响应统一**：`create_error_response`（`BaseServing`）与 `error_response.py` 工厂产出 OpenAI 风格 `ErrorResponse`，异常 handler（`server_utils`）把 `EngineGenerateError`/`GenerationError`/`HTTPException`/`RequestValidationError` 全部转成该格式。
- **dev 端点隔离**：rlhf/cache/sleep/rpc 等实验/危险端点在 `VLLM_SERVER_DEV_MODE` 才挂，并 logger.warning 提醒，防止生产误开。
- **SageMaker 适配**：`sagemaker_standards_bootstrap` 给 app 加 SageMaker JumpStart 期望的标准路由/响应包装，让 vLLM 可被 SageMaker 直接托管。

## 怎么做

### `build_app` 挂载顺序（节选）

```
build_app(args, supported_tasks)
  ├─ register_vllm_serve_api_routers(app)   # serve/__init__.py:11
  │    ├─ register_instrumentator_api_routers  # /metrics /health /version /server_load
  │    ├─ attach_lora_router                   # /v1/load_lora_adapter /v1/unload_lora_adapter
  │    ├─ attach_profile_router                # /start_profile /stop_profile
  │    └─ attach_tokenize_router               # /v1/tokenize /v1/detokenize
  ├─ register_models_api_router               # /v1/models
  ├─ register_sagemaker_api_router
  ├─ if VLLM_SERVER_DEV_MODE: register_vllm_dev_api_routers
  ├─ if "generate": register_generate_api_routers + elastic_ep
  ├─ if "generate"|"render": register_scale_out_api_routers
  ├─ if "transcription"|"realtime": register_speech_to_text_api_routers
  └─ if pooling tasks: register_pooling_api_routers
```

### `BaseServing._check_model`（engine/serving.py:40）

```
_is_model_supported(request.model)? → None
request.model in models.lora_requests? → None
VLLM_ALLOW_RUNTIME_LORA_UPDATING and resolve_lora(model)? → None/LoRARequest
else → 404 NotFound
```

### 关键文件导航

| 关注点 | 位置 |
|---|---|
| 路由聚合 | `vllm/entrypoints/serve/__init__.py:11` |
| dev 路由 | `vllm/entrypoints/serve/__init__.py:35` |
| BaseServing | `vllm/entrypoints/serve/engine/serving.py:29` |
| _check_model | `vllm/entrypoints/serve/engine/serving.py:40` |
| 异常 handler | `vllm/entrypoints/serve/utils/server_utils.py:328` 起 |
| lifespan | `vllm/entrypoints/serve/utils/server_utils.py:535` |
| AuthenticationMiddleware | `vllm/entrypoints/serve/utils/server_utils.py:45` |
| instrumentator 集合 | `vllm/entrypoints/serve/instrumentator/__init__.py:7` |
| tokenize serving | `vllm/entrypoints/serve/tokenize/serving.py:32` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`build_app` 顺序挂载本包。
- [openai/models.md](../openai/models.md)：`OpenAIServingModels` 与 `serve/lora` 路由配合。
- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing` → `BaseServing`。
- [pooling/base.md](../pooling/base.md)：`PoolingBaseServing` → `BaseServing`。
- [scale-out/README.md](../scale-out/README.md)：render-only state 复用 `ServingTokenization` 等。
- [可观测-metrics](../../16-observability/README.md)：`instrumentator/metrics` 暴露 Prometheus。
- [07-distributed](../../07-distributed/README.md)：`elastic_ep` + `dev/rlhf` weight transfer 关联 EP/EPLB。

## 历史版本演进

- **v0.5–v0.6（内联）**：server_utils/api_utils 内联在 api_server。
- **v0.9（serve/ 抽离）**：把公共工具移到 `serve/`；`register_vllm_serve_api_routers` 聚合；`CLISubcommand` 化。
- **v0.10（Elastic EP + SageMaker + dev rlhf）**：`elastic_ep/`、`sagemaker/`、`dev/rlhf`（pause/resume/weight transfer）落地。
- **v0.10.x（dev sleep/cache）**：sleep/wake、reset_prefix_cache/reset_mm_cache/reset_encoder_cache 实验端点。
- **v0.11/main**：`VLLM_ALLOW_RUNTIME_LORA_UPDATING` 与 `VLLM_SKIP_MODEL_NAME_VALIDATION`；`XRequestIdMiddleware`、`enable_server_load_tracking`；`dev/rpc` collective_rpc 透传。

## 模块导航

| 页 | 主题 |
|---|---|
| [utils.md](utils.md) | `serve/utils/` 工具集 |
| [tokenize.md](tokenize.md) | `/v1/tokenize`/`/v1/detokenize` |
| [lora-serves.md](lora-serves.md) | `/v1/load_lora_adapter`/`unload` |
| [profile.md](profile.md) | `/start_profile`/`/stop_profile` |
| [sagemaker.md](sagemaker.md) | SageMaker 适配 |
| [elastic-ep-serve.md](elastic-ep-serve.md) | `ScalingMiddleware` + scale 端点 |
| [instrumentator.md](instrumentator.md) | Prometheus/health/version |
| [engine-serve.md](engine-serve.md) | `BaseServing` 基类 |
| [dev.md](dev.md) | 实验端点 |

## 参见

- [← 返回 API 入口首页](../README.md)
- [openai/api-server.md](../openai/api-server.md)
- [可观测-metrics](../../16-observability/README.md)
