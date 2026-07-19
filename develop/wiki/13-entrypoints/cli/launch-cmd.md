[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > launch

# launch 子命令（vllm launch render）

> `cli/launch.py` 提供 `vllm launch`，用于单独启动 vLLM 的某个"组件"。当前唯一子组件是 `render`：仅跑预处理/后处理（chat template、多模态解析、tokenize）的 CPU-only FastAPI server，不分配 KV cache、不调 GPU。配合 scale-out 多阶段拆分使用。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `LaunchSubcommandBase` | `vllm/entrypoints/cli/launch.py:30` | 组件子命令基类，提供 `add_cli_args` 默认实现 |
| `RenderSubcommand` | `:49` | `vllm launch render`：调 `run_launch_fastapi` |
| `LaunchSubcommand` | `:60` | 顶层 `vllm launch`，用嵌套 sub-subparsers 装配各组件 |
| `run_launch_fastapi` | `:112` | render server 启动协程 |
| `cmd_init` | `:108` | 返回 `[LaunchSubcommand()]` |

`LaunchSubcommand.subparser_init`（`:79`）建 `launch` subparser，加 `required=True` 子 subparser（`dest="launch_component"`），遍历 `LaunchSubcommandBase.__subclasses__()`（动态发现）注册每个组件子 subparser，`set_defaults(launch_command=cmd_cls.cmd)`。

`RenderSubcommand.cmd`（`:56`）→ `uvloop.run(run_launch_fastapi(args))`。

`run_launch_fastapi`（`:112`）：

1. SIGTERM → `_interrupt_init`（与 api_server 一致）。
2. `setup_server(args, reuse_port=False)` 绑端口。
3. `AsyncEngineArgs.from_cli_args(args)` → `create_model_config()`。
4. **清掉 quantization**：render server 不跑量化内核，`model_config.quantization = None` 跳过校验。
5. **关 KV cache**：`envs.VLLM_CPU_KVCACHE_SPACE = 0`，抑制 CPU KV 警告。
6. `VllmConfig(model_config=model_config)`（最小配置）。
7. `build_and_serve_renderer(vllm_config, listen_address, sock, args)`（`api_server.py:640`）→ `serve_http`。

## 为什么

- **预处理/推理解耦**：把 chat template/多模态解析/tokenize 这类 CPU 密集工作从 GPU 节点剥离，独立 scale-out 到廉价 CPU 节点；GPU 节点只收 `EngineInput`（token-in/token-out，见 [scale-out/token-in-token-out.md](../scale-out/token-in-token-out.md)）。
- **无引擎也能 build_app**：`init_render_app_state`（`api_server.py:424`）用 `renderer_from_config` 直接造 renderer，`state.engine_client=None`、`OpenAIModelRegistry`（只读），跳过 `AsyncLLM` 构造。
- **动态组件发现**：`LaunchSubcommandBase.__subclasses__()` 自动发现组件类，新增组件只需写子类，不改 `LaunchSubcommand`。
- **共用 serve 参数**：`add_cli_args` 默认调 `make_arg_parser`，让 render server 与主 server 共享参数集（host/port/ssl/chat_template/...），仅引擎参数被 render 忽略。

## 怎么做

### 启动 render server

```bash
vllm launch render --model <model> --host 0.0.0.0 --port 8001
```

该 server 暴露 `/v1/chat/completions` 风格的 render 端点（[scale-out/render.md](../scale-out/render.md)）和 `/v1/tokenize`，但不生成 token。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| LaunchSubcommandBase | `vllm/entrypoints/cli/launch.py:30` |
| RenderSubcommand | `vllm/entrypoints/cli/launch.py:49` |
| LaunchSubcommand | `vllm/entrypoints/cli/launch.py:60` |
| subparser_init 动态发现 | `vllm/entrypoints/cli/launch.py:92` |
| run_launch_fastapi | `vllm/entrypoints/cli/launch.py:112` |
| 清 quantization | `vllm/entrypoints/cli/launch.py:131` |
| 关 KV cache | `vllm/entrypoints/cli/launch.py:135` |
| build_and_serve_renderer 调用 | `vllm/entrypoints/cli/launch.py:138` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`setup_server`/`build_and_serve_renderer`/`init_render_app_state`。
- [scale-out/render.md](../scale-out/render.md)：`ServingRender` 是 render server 的核心 handler。
- [scale-out/token-in-token-out.md](../scale-out/token-in-token-out.md)：render 与推理节点的协同协议。
- [chat-utils.md](../chat-utils.md)：render 内部调 `parse_chat_messages`。
- [多模态](../../11-multimodal/README.md)：render 节点跑 MM processor。

## 历史版本演进

- **v0.10（launch render 引入）**：`vllm launch render` 子命令；`build_and_serve_renderer`/`init_render_app_state` 落地；清 quantization + KV cache 警告。
- **v0.10.x（scale-out 协同）**：与 `token-in-token-out` 协议配合，支持 render↔推理跨进程。
- **main**：`LaunchSubcommandBase.__subclasses__()` 动态发现；为后续更多 launch 组件（derender server？待核实）预留扩展点。

## 参见

- [← 返回 CLI 首页](README.md)
- [openai/api-server.md](../openai/api-server.md)
- [scale-out/render.md](../scale-out/render.md)
