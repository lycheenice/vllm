[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > utils

# serve/utils/（server 工具集）

> `serve/utils/` 是 server 侧的横切工具集合：请求取消/负载感知调用、max_tokens 计算、非默认参数日志、LoRA modules 解析、SSL 证书刷新、请求日志器、异常→OpenAI 错误响应、uvicorn lifespan/log config、Authentication/XRequestId/Scaling middleware、SSE 解码与响应日志。

## 是什么

| 文件 | 核心 | 职责 |
|---|---|---|
| `api_utils.py` | `listen_for_disconnect`（`vllm/entrypoints/serve/utils/api_utils.py:37`）/`with_cancellation`（`:52`）/`load_aware_call`（`:101`）/`cli_env_setup`（`:149`）/`get_max_tokens`（`:170`）/`should_include_usage`（`:276`）/`process_lora_modules`（`:291`）/`log_version_and_model`（`:317`）/`validate_json_request`（`:342`） | 请求生命周期工具 |
| `server_utils.py` | `AuthenticationMiddleware`（`vllm/entrypoints/serve/utils/server_utils.py:45`）/`XRequestIdMiddleware`（`:96`）/`load_log_config`/`get_uvicorn_log_config`（`:140`）/`SSEDecoder`（`:203`）/`log_response`（`:310`）/异常 handler 集合/`lifespan`（`:535`） | middleware + lifespan + 异常 |
| `request_logger.py` | `RequestLogger` | 请求/响应日志（限长） |
| `ssl.py` | `SSLCertRefresher` | 证书周期刷新（被 [launcher.md](../launcher.md) 用） |
| `constants.py` | `H11_MAX_*_DEFAULT` | h11 默认值 |
| `error_response.py` | `create_error_response` | 构造 `ErrorResponse` |
| `fingerprint.py` | — | server fingerprint（待核实） |
| `orca_metrics.py` | — | orca 风格 metrics（待核实） |
| `tool_calls_utils.py` | `maybe_filter_parallel_tool_calls` | tool call 过滤 |

`with_cancellation`（`:52`）包装 handler，客户端断开时 `Request.disconnect()` 触发 `asyncio.CancelledError`，让引擎请求被取消；`load_aware_call`（`:101`）在 `enable_server_load_tracking` 时按 `server_load_metrics` 决定 503 还是放行/递减。

`process_lora_modules`（`:291`）：把 CLI `--lora-modules`（已由 `LoRAParserAction` 解析为 `LoRAModulePath` 列表）合并 `default_mm_loras`（来自 `lora_config.default_mm_loras`），供 `OpenAIServingModels` 加载。

异常 handler（`server_utils.py:328` 起）：`exception_handler`（兜底 `Exception`→500）、`http_exception_handler`、`validation_exception_handler`（422，清洗 loc）、`engine_error_handler`（`EngineGenerateError`/`EngineDeadError`→503）、`generation_error_handler`（`GenerationError`→OpenAI 风格）。

`lifespan`（`:535`）：FastAPI lifespan，起后台 stats 日志 task（`engine_client.do_log_stats`），与 [launcher.md](../launcher.md) 的 `serve_http` 协作。

## 为什么

- **取消传播**：HTTP 客户端断连→request 任务 cancel→`AsyncLLM` 请求 abort，避免空跑占 KV cache；`with_cancellation` 把这套语义统一封装。
- **负载感知**：`load_aware_call` 让 handler 在 server 过载时直接 503，配合 `enable_server_load_tracking` 做软背压。
- **统一错误体**：所有异常最终都成 OpenAI `ErrorResponse`，客户端解析一致；`validation_exception_handler` 还清洗 Pydantic loc 去掉 `body.` 前缀。
- **日志可配**：`get_uvicorn_log_config` 支持日志文件 + access log 过滤；`RequestLogger` 限长避免巨型 prompt 撑爆日志。
- **middleware 可组合**：`AuthenticationMiddleware`/`XRequestIdMiddleware`/`ScalingMiddleware` 都可独立开关，`build_app` 按参数挂。

## 怎么做

### 常用工具用法（serving 层）

```python
@with_cancellation
async def handler(request):
    await listen_for_disconnect(request)  # 在 StreamingResponse 内
    max_tokens = get_max_tokens(request, model_config, ...)
    ...
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| listen_for_disconnect | `vllm/entrypoints/serve/utils/api_utils.py:37` |
| with_cancellation | `vllm/entrypoints/serve/utils/api_utils.py:52` |
| load_aware_call | `vllm/entrypoints/serve/utils/api_utils.py:101` |
| cli_env_setup | `vllm/entrypoints/serve/utils/api_utils.py:149` |
| get_max_tokens | `vllm/entrypoints/serve/utils/api_utils.py:170` |
| should_include_usage | `vllm/entrypoints/serve/utils/api_utils.py:276` |
| process_lora_modules | `vllm/entrypoints/serve/utils/api_utils.py:291` |
| validate_json_request | `vllm/entrypoints/serve/utils/api_utils.py:342` |
| AuthenticationMiddleware | `vllm/entrypoints/serve/utils/server_utils.py:45` |
| XRequestIdMiddleware | `vllm/entrypoints/serve/utils/server_utils.py:96` |
| get_uvicorn_log_config | `vllm/entrypoints/serve/utils/server_utils.py:140` |
| log_response | `vllm/entrypoints/serve/utils/server_utils.py:310` |
| engine_error_handler | `vllm/entrypoints/serve/utils/server_utils.py:328` |
| validation_exception_handler | `vllm/entrypoints/serve/utils/server_utils.py:488` |
| lifespan | `vllm/entrypoints/serve/utils/server_utils.py:535` |
| SSLCertRefresher | `vllm/entrypoints/serve/utils/ssl.py`（待核实行号） |

## 与其它模块/系统配合

- [launcher.md](../launcher.md)：`serve_http` 用 `SSLCertRefresher`、`get_uvicorn_log_config`、`app.state.server`。
- [openai/api-server.md](../openai/api-server.md)：`build_app` 挂 middleware、exception handler、lifespan。
- [engine-serve.md](engine-serve.md)：`BaseServing.create_error_response` 与 `error_response.py` 一致。
- [可观测-metrics](../../16-observability/README.md)：`lifespan` stats、`log_response`、`server_load_metrics`。
- [LoRA](../../12-lora/README.md)：`process_lora_modules`、`default_mm_loras`。

## 历史版本演进

- **v0.5–v0.6（内联）**： utilities 内联 api_server。
- **v0.9（抽 serve/utils）**：拆 `api_utils`/`server_utils`/`request_logger`/`ssl`；`with_cancellation`/`load_aware_call`/`XRequestIdMiddleware`。
- **v0.10（h11 + SSL 刷新 + server_load）**：`h11_max_*` 常量；`SSLCertRefresher`；`enable_server_load_tracking`。
- **v0.11/main**：`validate_json_request`；`log_response` SSE 解码；`VLLM_DEBUG_LOG_API_SERVER_RESPONSE` 警告；`orca_metrics`/`fingerprint`（待核实稳定状态）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [launcher.md](../launcher.md)
- [openai/api-server.md](../openai/api-server.md)
- [可观测-metrics](../../16-observability/README.md)
