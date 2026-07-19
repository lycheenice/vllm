[← Wiki 首页](../README.md) > [API 入口](README.md) > gRPC server

# gRPC server（grpc_server.py）

> `vllm/entrypoints/grpc_server.py` 提供除 HTTP 外的另一在线入口：基于 `smg-grpc-servicer` 的 gRPC 服务，背后仍用 `AsyncLLM`，支持流式生成、Kubernetes 健康探针、reflection。通过 `vllm serve --grpc` 或 `python -m vllm.entrypoints.grpc_server` 启动。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `serve_grpc` | `vllm/entrypoints/grpc_server.py:56` | 协程：构造 `AsyncLLM`、`VllmEngineServicer`、起 `grpc.aio.server` |
| `main` | `vllm/entrypoints/grpc_server.py:168` | `FlexibleArgumentParser` + `AsyncEngineArgs.add_cli_args` 解析，`uvloop.run(serve_grpc(args))` |

`serve_grpc` 流程（`:56`）：

1. `log_version_and_model` 记录版本与模型。
2. `AsyncEngineArgs.from_cli_args(args)` → `create_engine_config(usage_context=OPENAI_API_SERVER)`。
3. `AsyncLLM.from_vllm_config(..., enable_log_requests, disable_log_stats)`（`:77`）。
4. `VllmEngineServicer(async_llm, start_time)`（`:85`，来自 `smg_grpc_servicer`）。
5. `grpc.aio.server(options=...)`：关闭 send/receive 上限（`-1`）、容忍 10s keepalive ping、允许无调用 keepalive。
6. 注册 `VllmEngine` servicer、`VllmHealthServicer`（K8s）、gRPC reflection。
7. `server.add_insecure_port(f"{host}:{port}")`、`server.open()`。
8. 可选起周期 stats 日志 task（`VLLM_LOG_STATS_INTERVAL`）。
9. 注册 SIGTERM/SIGINT → `stop_event`，`await stop_event.wait()`。
10. `finally`：取消 stats task、`health_servicer.set_not_serving()`、`server.stop(grace=5.0)`、`async_llm.shutdown()`。

依赖 `smg-grpc-servicer`/`smg-grpc-proto`，未安装时抛 `ImportError`（`:35`）提示 `pip install vllm[grpc]`。

## 为什么

- **绕过 HTTP 开销**：高 QPS 内网场景下 gRPC 二进制 + HTTP/2 多路复用比 REST+SSE 更省 CPU/带宽；`max_send/receive_message_length=-1` 去掉默认 4MB 限制以放大 batch。
- **K8s 原生健康**：`VllmHealthServicer` 实现 `grpc.health.v1.Health` 标准协议，可直接作 liveness/readiness probe，无需额外 HTTP 端点。
- **reflection 调试**：开启 server reflection 后 `grpcurl` 等工具无需 proto 文件即可列出/调用方法，便于联调。
- **与 HTTP 共用引擎**：`AsyncLLM` 同时是 HTTP 入口的 backend，gRPC 仅是另一种协议包装，引擎层零改动。
- **uvloop 加速**：`main` 用 `uvloop.run`，与 HTTP 一致的事件循环策略。

## 怎么做

### 启动

```bash
vllm serve <model> --grpc --port 50051
# 或
python -m vllm.entrypoints.grpc_server --model <model> --host 0.0.0.0 --port 50051
```

`vllm serve --grpc` 分支见 `vllm/entrypoints/cli/serve.py:55`，直接 `uvloop.run(serve_grpc(args))` 后 return（跳过 HTTP 流程）。

### 关键 gRPC 选项

| 选项 | 值 | 作用 |
|---|---|---|
| `grpc.max_send_message_length` | `-1` | 取消发送上限 |
| `grpc.max_receive_message_length` | `-1` | 取消接收上限 |
| `grpc.http2.min_recv_ping_interval_without_data_ms` | `10000` | 容忍 10s keepalive ping（非流式请求无 DATA 帧） |
| `grpc.keepalive_permit_without_calls` | `True` | 允许无活动调用时 keepalive |

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| serve_grpc | `vllm/entrypoints/grpc_server.py:56` |
| AsyncLLM 构造 | `vllm/entrypoints/grpc_server.py:77` |
| server options | `vllm/entrypoints/grpc_server.py:88` |
| 健康/reflection 注册 | `vllm/entrypoints/grpc_server.py:101` |
| 信号处理 | `vllm/entrypoints/grpc_server.py:142` |
| shutdown | `vllm/entrypoints/grpc_server.py:153` |
| main / 参数解析 | `vllm/entrypoints/grpc_server.py:168` |
| 依赖缺失报错 | `vllm/entrypoints/grpc_server.py:35` |

## 与其它模块/系统配合

- [引擎核心-AsyncLLM 前端](../01-engine-core/async-llm-frontend.md)：`serve_grpc` 用同一个 `AsyncLLM.from_vllm_config`。
- [cli/serve-cmd.md](cli/serve-cmd.md)：`--grpc` 分支调 `serve_grpc`。
- [serve/utils.md](serve/utils.md)：`log_version_and_model`、`cli_env_setup` 复用。
- [配置-scheduler](../10-config/scheduler-config.md)：`shutdown_timeout` 经 `async_llm.shutdown()` 间接使用（gRPC 这里直接 `shutdown()` 无 timeout 参数，待核实是否生效）。

## 历史版本演进

- **v0.9（gRPC 引入）**：新增 `grpc_server.py`，依赖 `smg-grpc-servicer`；与 HTTP 共用 `AsyncEngineArgs`；提供 K8s health + reflection。
- **v0.10（--grpc 集成）**：`cli/serve.py` 加 `--grpc` 分支；`make_arg_parser` 注册 `--grpc` 选项（`openai/cli_args.py:373`）。
- **v0.11/main**：keepalive ping 容忍（`:95` 注释）解决非流式请求被默认 300s ping 限流断连；stats 日志 task 与 HTTP `lifespan` 行为对齐（`:127`）。

> 注：`VllmEngineServicer`/`VllmHealthServicer` 的具体 proto schema 来自外部包 `smg-grpc-proto`，本仓库不维护其 proto 定义，详细字段（待补充）。

## 参见

- [← 返回 API 入口首页](README.md)
- [cli/serve-cmd.md](cli/serve-cmd.md)
- [openai/api-server.md](openai/api-server.md)
- [引擎核心-AsyncLLM 前端](../01-engine-core/async-llm-frontend.md)
