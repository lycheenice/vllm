[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > serve

# serve 子命令（vllm serve）

> `cli/serve.py` 的 `ServeSubcommand` 是 vLLM 最常用入口：解析 serve 参数 → 按 LB 模式/`api_server_count`/headless/grpc 分派到 5 条执行分支，最终起 HTTP（或 gRPC）服务。它编排单进程、多 api-server、headless、DP supervisor 四种部署形态。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `ServeSubcommand` | `vllm/entrypoints/cli/serve.py:44` | `vllm serve` 子命令 |
| `ServeSubcommand.cmd` | `:50` | 分派逻辑 |
| `ServeSubcommand.validate` | `:150` | 调 `validate_parsed_serve_args` |
| `ServeSubcommand.subparser_init` | `:153` | 用 `make_arg_parser` 注册参数 |
| `run_headless` | `:173` | 无 API server 模式（仅引擎进程，给外部 LB/PP-TP worker 用） |
| `run_multi_api_server` | `:257` | 多 `api-server-count` 或 Rust frontend 模式 |
| `cmd_init` | `:169` | 返回 `[ServeSubcommand()]` |

`cmd`（`:50`）分派逻辑：

1. `model_tag` → `args.model`。
2. `--grpc`：`uvloop.run(serve_grpc(args))` return（见 [grpc-server.md](../grpc-server.md)）。
3. `--headless`：`api_server_count=0`。
4. 推断 LB 模式：`is_external_lb`（`data_parallel_external_lb` 或 `data_parallel_rank`）、`is_hybrid_lb`（`data_parallel_hybrid_lb` 或 `data_parallel_start_rank`）、`is_multi_port`（`data_parallel_multi_port_external_lb`）。三者互斥。
5. 默认 `api_server_count`：multi_port/external_lb/Rust frontend → 1；hybrid_lb → `data_parallel_size_local`；否则 `data_parallel_size`。Elastic EP 强制 ≤1。
6. 分派：
   - `is_multi_port` → `run_dp_supervisor(args)`（[dp-supervisor.md](../openai/dp-supervisor.md)）。
   - `api_server_count < 1` → `run_headless(args)`。
   - `api_server_count > 1 or VLLM_RUST_FRONTEND_PATH` → `run_multi_api_server(args)`。
   - 否则单进程 `uvloop.run(run_server(args))`。

`run_headless`（`:173`）：构造 `vllm_config`（`headless=True`）；若 `node_rank_within_dp>0` 起纯 worker（`MultiprocExecutor` + `start_worker_monitor`，多节点 PP/TP worker）；否则用 `CoreEngineProcManager` 起 `local_engine_count` 个引擎进程，监听 `data_parallel_master_ip:rpc_port` 供外部 API server 连。

`run_multi_api_server`（`:257`）：`setup_server(reuse_port=num>1)` 绑端口；`launch_core_engines(...)` 上下文起引擎 + coordinator；选 `RustFrontendProcessManager`（`VLLM_RUST_FRONTEND_PATH`）或 `APIServerProcessManager`（多 Python worker，每个跑 `run_server_worker`）；`wait_for_completion_or_failure`；shutdown 时按 `shutdown_timeout` 级联。

## 为什么

- **五分支覆盖全部部署**：单进程（最简）、多 api-server（单机多 GPU DP）、headless（仅引擎给外部 LB/多节点 worker）、DP supervisor（多端口外部 LB）、Rust frontend（高 QPS 前端）；CLI 一次选择，下游代码复用。
- **LB 模式互斥校验**：external/hybrid/multi-port 三模式语义重叠，启动期 raise 避免运行期行为混乱。
- **headless 解耦 engine 与 API**：headless 模式下引擎进程独立监听 RPC，外部 API server（或别的 vLLM instance）可连它，支持多节点 PP/TP 与外部 LB 混合。
- **端口绑定早于引擎**：`setup_server` 先 `bind`（multi 时 `reuse_port`），再 `launch_core_engines`，规避 Ray/引擎竞态。
- **Rust frontend 共址**：Rust frontend 是多线程单进程，`api_server_count>1` 无意义，启动期 warning 并 cap 到 1。
- **Elastic EP 限制**：弹性 EP 当前仅支持单 API server，启动期 cap 并 warning。

## 怎么做

### 单进程（最常见）

```bash
vllm serve <model> --host 0.0.0.0 --port 8000
```

→ `uvloop.run(run_server(args))` → `run_server_worker` → `build_async_engine_client` + `build_and_serve` + `serve_http`。

### 多 api-server（单机 DP）

```bash
vllm serve <model> --data-parallel-size 2 --api-server-count 2
```

→ `run_multi_api_server`：`launch_core_engines` 起 2 引擎；`APIServerProcessManager` 起 2 个 worker 共享 `SO_REUSEPORT` socket。

### headless（外部 LB / 多节点 worker）

```bash
vllm serve <model> --headless --data-parallel-size 4 --data-parallel-size-local 2
```

→ `run_headless`：在非 rank-0 节点起纯 worker；rank-0 节点起 `CoreEngineProcManager` 监听 RPC。

### 多端口外部 LB

```bash
vllm serve <model> --data-parallel-size 4 --data-parallel-size-local 2 \
  --data-parallel-multi-port-external-lb --port 8000 --data-parallel-supervisor-port 9000
```

→ `run_dp_supervisor`：supervisor 监听 9000，spawn 2 child（8000/8001）。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServeSubcommand | `vllm/entrypoints/cli/serve.py:44` |
| cmd 分派 | `vllm/entrypoints/cli/serve.py:50` |
| LB 模式推断 | `vllm/entrypoints/cli/serve.py:75` |
| api_server_count 默认 | `vllm/entrypoints/cli/serve.py:105` |
| subparser_init | `vllm/entrypoints/cli/serve.py:153` |
| run_headless | `vllm/entrypoints/cli/serve.py:173` |
| run_multi_api_server | `vllm/entrypoints/cli/serve.py:257` |
| launch_core_engines | `vllm/entrypoints/cli/serve.py:323` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`run_server`/`setup_server`/`run_server_worker`。
- [openai/dp-supervisor.md](../openai/dp-supervisor.md)：`run_dp_supervisor`。
- [openai/cli-args.md](../openai/cli-args.md)：`make_arg_parser`/`validate_parsed_serve_args`。
- [grpc-server.md](../grpc-server.md)：`--grpc` 分支。
- [引擎核心-EngineCore](../../01-engine-core/engine-core-process.md)：`CoreEngineProcManager`/`launch_core_engines`。
- [执行层-DP supervisor 关联]：`data_parallel_*` 字段决定 LB 形态。
- [可观测-metrics](../../16-observability/README.md)：`setup_multiprocess_prometheus`（多 api-server）。

## 历史版本演进

- **v0.7（serve 子命令）**：`vllm serve` 落地，仅单进程。
- **v0.8（V1 引擎 + multi api-server）**：`run_multi_api_server` + `APIServerProcessManager`；`reuse_port` 共享 socket。
- **v0.9（headless）**：`--headless` + `CoreEngineProcManager`，支持外部 LB 与多节点 PP/TP worker。
- **v0.10（multi-port external LB + elastic EP）**：`run_dp_supervisor` 分支；Elastic EP cap `api_server_count=1`。
- **v0.10.x（Rust frontend）**：`VLLM_RUST_FRONTEND_PATH` 分支 + `RustFrontendProcessManager`；defer_api_server_ports 端口分配策略。
- **v0.11/main**：`VLLM_ALLOW_RUNTIME_LORA_UPDATING` 与 multi api-server 互斥校验（`:293`）；Ray DP 端口预分配分支（`:316`）；`coordinator`/`tensor_queue` 经 `launch_core_engines` 返回。

## 参见

- [← 返回 CLI 首页](README.md)
- [openai/api-server.md](../openai/api-server.md)
- [openai/dp-supervisor.md](../openai/dp-supervisor.md)
- [grpc-server.md](../grpc-server.md)
