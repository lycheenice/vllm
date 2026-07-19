[← Wiki 首页](../README.md) > [API 入口](README.md) > launcher

# launcher（HTTP 启动器）

> `vllm/entrypoints/launcher.py` 是 vLLM HTTP 服务的最终启动器：在 uvicorn 之上叠加信号驱动的优雅 drain/abort、SSL 证书热刷新、引擎死亡 watchdog 与端口占用诊断。所有 `vllm serve`/`vllm launch` 单进程路径最终都汇入 `serve_http`。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `serve_http` | `vllm/entrypoints/launcher.py:26` | 主入口：配置 uvicorn、起 server_task、注册 SIGINT/SIGTERM、起 watchdog、shutdown drain |
| `watchdog_loop` | `vllm/entrypoints/launcher.py:156` | 周期（5s）轮询 `engine.errored`，必要时强制 `server.should_exit` |
| `terminate_if_errored` | `vllm/entrypoints/launcher.py:168` | 单次检查：引擎死且非 `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` 时拉闸 |

`serve_http` 流程（`vllm/entrypoints/launcher.py:26`）：

1. 打印所有路由（POST 端点 + 其他端点分开）。
2. 从 `uvicorn_kwargs` 抽 `h11_max_incomplete_event_size`/`h11_max_header_count`，缺省取 `serve/utils/constants.H11_*_DEFAULT`。
3. 构造 `uvicorn.Config`，写两个 h11 头部限制，`config.load()`。
4. `app.state.server = server`，便于其他模块访问。
5. 起两个 task：`watchdog_loop`、`server.serve(sockets=[sock])`。
6. 可选起 `SSLCertRefresher`（来自 `serve/utils/ssl.py`）做证书周期刷新。
7. `signal_handler` 在 SIGINT/SIGTERM 时置 `shutdown_event`。
8. `handle_shutdown` 等事件后，根据 `vllm_config.shutdown_timeout` 决定 `abort`（timeout=0）或 `drain`，调 `engine_client.shutdown(timeout=...)`，再 `server.should_exit=True` 并 cancel server/watchdog。
9. `server_task` 若被 `CancelledError`，调 `find_process_using_port` 诊断端口占用并 `server.shutdown()`。

## 为什么

- **可控关闭语义**：`shutdown_timeout==0` → abort（立即结束，不等请求）；`>0` → drain（停接新请求、等存量完成或超时）。该值来自 `VllmConfig.shutdown_timeout`，让 K8s 滚动升级可配置。
- **watchdog 兜底**：`StreamingResponse` 的 async generator 里抛异常不会冒泡到 uvicorn，`watchdog_loop` 每 5s 主动检查 `engine.errored and not engine.is_running`，避免僵死服务；`VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` 允许保留容器便于排查。
- **h11 头部放大**：默认 h11 对 `max_incomplete_event_size`/`max_header_count` 限制偏小，大请求体/多头部（多模态 base64）会被静默丢弃；显式提升避免 footgun。
- **端口占用诊断**：`CancelledError` 路径用 `find_process_using_port` 打印占用进程命令行，定位"端口被谁占了"。
- **SSL 热刷新**：`SSLCertRefresher` 在 cert 文件被替换后重建 SSLContext，无需重启即可轮转证书。
- **绑定早于引擎**：`setup_server`（`api_server.py:555`）在引擎构造前 `bind`，规避与 Ray 的竞态（见 `vllm-project/vllm#8204`）。

## 怎么做

### 调用关系

`build_and_serve`（`vllm/entrypoints/openai/api_server.py:592`）拿到 `engine_client`、`listen_address`、`sock`、`args` 后，把 SSL/h11/cors 等 kwargs 传给 `serve_http`，`serve_http` 返回 `shutdown_task`，由 `run_server_worker` await。

### 关键参数

| 参数 | 来源 | 作用 |
|---|---|---|
| `h11_max_incomplete_event_size` | `args.h11_max_incomplete_event_size` | h11 不完整事件最大字节 |
| `h11_max_header_count` | `args.h11_max_header_count` | h11 最大头部数 |
| `enable_ssl_refresh` | `args.enable_ssl_refresh` | 启用证书刷新 |
| `timeout_keep_alive` | `envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE` | keep-alive 超时 |
| `ssl_keyfile/certfile/ca_certs/...` | `args.*` | TLS 配置 |

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| serve_http 入口 | `vllm/entrypoints/launcher.py:26` |
| 路由打印 | `vllm/entrypoints/launcher.py:37` |
| h11 限制 | `vllm/entrypoints/launcher.py:60` |
| 信号注册 | `vllm/entrypoints/launcher.py:106` |
| shutdown 处理 | `vllm/entrypoints/launcher.py:109` |
| drain/abort 决策 | `vllm/entrypoints/launcher.py:113` |
| watchdog | `vllm/entrypoints/launcher.py:156` |
| 端口诊断 | `vllm/entrypoints/launcher.py:140` |

## 与其它模块/系统配合

- [openai/api-server.md](openai/api-server.md)：`build_and_serve`/`build_and_serve_renderer` 调 `serve_http`。
- [cli/serve-cmd.md](cli/serve-cmd.md)：`ServeSubcommand` 单进程分支调 `run_server` → `serve_http`。
- [serve/utils.md](serve/utils.md)：`server_utils.lifespan`、`get_uvicorn_log_config`、`SSLCertRefresher`、`AuthenticationMiddleware` 在此之前已挂好。
- [配置-scheduler](../10-config/scheduler-config.md)：`shutdown_timeout` 决定 drain/abort。
- [可观测-metrics](../16-observability/README.md)：`lifespan` 起后台 stats 日志任务。

## 历史版本演进

- **v0.5–v0.6（简单 uvicorn）**：直接 `uvicorn.run`，无 drain 控制；shutdown 立即断连。
- **v0.7（信号 drain）**：引入 `shutdown_event` + `engine_client.shutdown(timeout=...)`，支持 graceful drain。
- **v0.8（watchdog）**：加 `watchdog_loop`/`terminate_if_errored` 处理 StreamingResponse 异常丢失问题。
- **v0.9（h11 + SSL 刷新）**：加 `h11_max_incomplete_event_size`/`h11_max_header_count`；`SSLCertRefresher` 落地；路由打印分两类。
- **v0.10.x（端口诊断）**：`CancelledError` 分支调 `find_process_using_port` 打印占用进程。
- **main**：`decorate_logs("APIServer")` 与 `set_ulimit()` 在 `run_server` 前调用，避免 fd/ulimit 导致丢请求（`api_server.py:688`、`setup_server`）。

## 参见

- [← 返回 API 入口首页](README.md)
- [openai/api-server.md](openai/api-server.md)
- [cli/serve-cmd.md](cli/serve-cmd.md)
- [serve/utils.md](serve/utils.md)
- [可观测-metrics](../16-observability/README.md)
