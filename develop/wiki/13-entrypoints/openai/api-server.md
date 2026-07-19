[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > api_server

# api_server.py（FastAPI app 与 server 骨架）

> `vllm/entrypoints/openai/api_server.py` 是 vLLM HTTP 服务的"主板"：构建 FastAPI app、按任务挂路由、初始化 app state、绑定 socket、调 `serve_http` 起 uvicorn。它同时提供 render-only（无 GPU）入口。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `build_async_engine_client` | `vllm/entrypoints/openai/api_server.py:78` | async ctx mgr：forkserver 预导入 + 转 `build_async_engine_client_from_engine_args` |
| `build_async_engine_client_from_engine_args` | `:109` | 真正构造 `AsyncLLM`，透传 `client_config`（count/index），`reset_mm_cache`，yield |
| `build_app` | `:157` | 拼 FastAPI：路由挂载 + middleware + exception handler |
| `init_app_state` | `:297` | 把 serving 对象实例化到 `app.state` |
| `init_render_app_state` | `:424` | 无引擎的 render-only state |
| `create_server_socket` / `create_server_unix_socket` | `:512` / `:530` | TCP/UDS socket |
| `validate_api_server_args` | `:536` | 校验 tool/reasoning parser 名 |
| `setup_server` | `:555` | 日志 + 插件导入 + 校验 + 绑端口 + set_ulimit |
| `build_and_serve` | `:592` | build_app + init_app_state + serve_http |
| `build_and_serve_renderer` | `:640` | render-only 版 |
| `run_server` | `:685` | 单 worker 入口 |
| `run_server_worker` | `:701` | 单 API server worker：引擎 ctx + build_and_serve |

`build_app`（`:157`）挂载顺序见 [README](README.md)；异常 handler 覆盖 `HTTPException`/`RequestValidationError`/`EngineGenerateError`/`EngineDeadError`/`GenerationError`/`VLLMValidationError`/`VLLMUnprocessableEntityError`/`Exception`；middleware 含 CORS、`AuthenticationMiddleware`、`XRequestIdMiddleware`、`ScalingMiddleware`、可选 `log_response`、用户 `--middleware` 列表。

`init_app_state`（`:297`）实例化：`OpenAIServingModels`（+`init_static_loras`）、`OnlineRenderer`、`OnlineDerenderer`、`ServingTokenization`，并按 supported_tasks 调对应 `init_*_state`；`state.engine_client`/`vllm_config`/`args`/`enable_server_load_tracking`/`server_load_metrics` 就位。

`init_render_app_state`（`:424`）：用 `renderer_from_config(vllm_config)` 构造 renderer，`OpenAIModelRegistry`（无 LoRA），`state.engine_client=None`、`state.log_stats=False`，专供 `vllm launch render`。

## 为什么

- **路由按任务动态挂载**：`supported_tasks` 由 `engine_client.get_supported_tasks()` 决定（`:609`），未支持的 task 路由不挂，避免无效依赖与 404 噪音。
- **engine ctx + serve ctx 解耦**：`build_async_engine_client` 用 `async with` 管理 `AsyncLLM` 生命周期，`build_and_serve` 在其内 `await serve_http`；`run_server_worker` 先 await `build_and_serve` 拿 `shutdown_task`，再在 ctx 外 await shutdown_task（`:719` 注释：context 退出后再 await 保证 backend 清理）。
- **render-only 分流**：`build_and_serve_renderer` 不构造 `AsyncLLM`，仅建 app + render state + serve_http，使预处理/后处理可独立 scale-out 到 CPU 节点（见 [scale-out/README](../scale-out/README.md)）。
- **绑定先于引擎**：`setup_server` 在引擎前 `bind`，规避与 Ray 启动竞态（注释链接 `vllm-project/vllm#8204`）。
- **forkserver 预导入**：当 `VLLM_WORKER_MULTIPROC_METHOD=forkserver` 时预导入 `vllm.v1.engine.async_llm`（`:89`），降低 forkserver 子进程冷启动。
- **SSL/UDS/IPv6 全覆盖**：`create_server_socket` 检测 IPv6 用 `AF_INET6`；`--uds` 走 Unix domain socket；`listen_address` 拼装考虑 IPv6 方括号与 SSL。

## 怎么做

### 单 worker 启动链

```
run_server(args)                                # :685
  └─ setup_server(args, reuse_port=False)       # :555  → (listen_address, sock)
  └─ run_server_worker(listen_address, sock, args)  # :701
       └─ async with build_async_engine_client(args, client_config):  # :78
            └─ build_and_serve(engine_client, listen_address, sock, args, **kw)  # :592
                 ├─ get_uvicorn_log_config
                 ├─ supported_tasks = engine_client.get_supported_tasks()
                 ├─ app = build_app(args, supported_tasks, model_config)   # :157
                 ├─ await init_app_state(engine_client, app.state, args, supported_tasks)  # :297
                 └─ return await serve_http(app, sock=sock, ...)   # launcher.serve_http
       └─ await shutdown_task
       └─ sock.close()
```

### render-only 启动链

`run_launch_fastapi`（`cli/launch.py:112`）→ `build_and_serve_renderer`（`:640`）：清掉 `model_config.quantization`、设 `VLLM_CPU_KVCACHE_SPACE=0`，`build_app(args, ("render",))` + `init_render_app_state`。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| engine client ctx | `vllm/entrypoints/openai/api_server.py:78` |
| AsyncLLM 构造 | `vllm/entrypoints/openai/api_server.py:136` |
| reset_mm_cache | `vllm/entrypoints/openai/api_server.py:149` |
| build_app | `vllm/entrypoints/openai/api_server.py:157` |
| 路由注册块 | `vllm/entrypoints/openai/api_server.py:182` 起 |
| middleware 块 | `vllm/entrypoints/openai/api_server.py:234` 起 |
| init_app_state | `vllm/entrypoints/openai/api_server.py:297` |
| OnlineRenderer 构造 | `vllm/entrypoints/openai/api_server.py:357` |
| init_render_app_state | `vllm/entrypoints/openai/api_server.py:424` |
| create_server_socket | `vllm/entrypoints/openai/api_server.py:512` |
| validate_api_server_args | `vllm/entrypoints/openai/api_server.py:536` |
| setup_server | `vllm/entrypoints/openai/api_server.py:555` |
| build_and_serve | `vllm/entrypoints/openai/api_server.py:592` |
| run_server_worker | `vllm/entrypoints/openai/api_server.py:701` |
| `__main__` | `vllm/entrypoints/openai/api_server.py:726` |

## 与其它模块/系统配合

- [launcher.md](../launcher.md)：`serve_http` 起 uvicorn + drain。
- [cli-args.md](cli-args.md)：`make_arg_parser`/`validate_parsed_serve_args`。
- [dp-supervisor.md](dp-supervisor.md)：多端口 LB 时 supervisor spawn 子进程跑 `run_server_worker`。
- [models.md](models.md)：`init_app_state` 构造 `OpenAIServingModels`。
- [engine-protocol.md](engine-protocol.md)：exception handler 引用 `GenerationError` 等。
- [serve/README.md](../serve/README.md)：`register_vllm_serve_api_routers` + middleware。
- [scale-out/render.md](../scale-out/render.md)：`build_and_serve_renderer` 与 render state。
- [引擎核心-AsyncLLM 前端](../../01-engine-core/async-llm-frontend.md)：`build_async_engine_client_from_engine_args`。
- [可观测-metrics](../../16-observability/README.md)：`lifespan` 起后台 stats，`instrumentator` 挂 metrics。

## 历史版本演进

- **v0.5–v0.6（单体）**：`build_app` 内联所有路由；engine 用 `AsyncLLMEngine` 同进程。
- **v0.7–v0.8（V1 + ctx）**：改成 `build_async_engine_client` async ctx + `AsyncLLM`；`init_app_state` 抽出；按 task 挂载生成/pooling 路由。
- **v0.9（serve/ 抽离 + multi api server）**：公共路由移到 `serve/`；`run_server_worker` 支持 `client_config`（count/index）；`setup_server` 抽出 socket 绑定。
- **v0.10（Responses + render-only + elastic ep）**：加 `init_render_app_state`/`build_and_serve_renderer`；`ScalingMiddleware` 必挂；`register_scale_out_api_routers` 接 "render" task。
- **v0.10.x（import tool plugin）**：`setup_server`/`run_server_worker` 调 `ToolParserManager.import_tool_parser`/`ReasoningParserManager.import_reasoning_parser`。
- **v0.11/main**：`decorate_logs("APIServer")` + `set_ulimit`；`signal.SIGTERM` 在 uvicorn 信号装好前 `_interrupt_init`（`:692`）；`sagemaker_standards_bootstrap` 必调；`VLLM_DEBUG_LOG_API_SERVER_RESPONSE` 警告敏感日志。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [launcher.md](../launcher.md)
- [cli-args.md](cli-args.md)
- [dp-supervisor.md](dp-supervisor.md)
- [serve/README.md](../serve/README.md)
