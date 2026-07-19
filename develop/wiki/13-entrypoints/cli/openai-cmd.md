[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > openai

# openai 子命令（vllm chat / vllm complete）

> `cli/openai.py` 提供交互式命令行客户端 `vllm chat` 与 `vllm complete`：用 OpenAI Python SDK 连到一个 vLLM server，在终端流式打印推理输出与统计。它本身不跑引擎，纯客户端工具。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `_register_signal_handlers` | `vllm/entrypoints/cli/openai.py:22` | 终端 Ctrl-C 处理 |
| `_interactive_cli` | `:30` | 构造 `openai.OpenAI` client（base_url/api_key/model） |
| `_print_chat_stream` / `_print_completion_stream` | `:48` / `:81` | 流式打印 + 统计（TTFT、tokens/s） |
| `_print_metrics` | `:70` | 打印 TTFT 与吞吐 |
| `chat` | `:103` | 对话循环（多轮，带 system prompt） |
| `_add_query_options` | `:123` | 共享 `--model`/`--base-url`/`--api-key`/`--system-prompt` 等参数 |
| `ChatCommand` | `:155` | `vllm chat` 子命令 |
| `CompleteCommand` | `:237` | `vllm complete` 子命令 |
| `cmd_init` | `:311` | 返回两个子命令 |

`ChatCommand`/`CompleteCommand` 都基于 `_add_query_options` 注入连接与生成参数，dispatch 时调 `_interactive_cli` 拿 `(model_name, client)`，再进 `chat()`（多轮）或单次 completion 流。

## 为什么

- **终端快速验证**：起完 server 后用 `vllm chat` 直接试模型行为，不必写 Python 脚本；流式 TTFT/吞吐统计便于性能直觉。
- **复用 OpenAI SDK**：直接 `openai.OpenAI(base_url=..., api_key=...)`，保证与 vLLM OpenAI 兼容协议一致，也方便切到真 OpenAI 做对照。
- **多轮上下文**：`chat()` 维护本地 message 历史，每轮把历史 + 新 user 消息发回 server，模拟真实对话。
- **共享 query 选项**：chat/complete 共用 `_add_query_options`，参数集一致，差异仅在交互模式（多轮 vs 单次）。

## 怎么做

### 用法

```bash
vllm chat --model <name> --base-url http://localhost:8000 --api-key EMPTY \
  --system-prompt "你是助手"
vllm complete --model <name> --base-url http://localhost:8000 \
  --prompt "写一首关于秋天的诗"
```

参数含 `--max-tokens`/`--temperature`/`--top-p`/`--stream`/`--system-prompt` 等（`_add_query_options`，`:123`）。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| _interactive_cli | `vllm/entrypoints/cli/openai.py:30` |
| _print_chat_stream | `vllm/entrypoints/cli/openai.py:48` |
| _print_metrics | `vllm/entrypoints/cli/openai.py:70` |
| chat 多轮 | `vllm/entrypoints/cli/openai.py:103` |
| _add_query_options | `vllm/entrypoints/cli/openai.py:123` |
| ChatCommand | `vllm/entrypoints/cli/openai.py:155` |
| CompleteCommand | `vllm/entrypoints/cli/openai.py:237` |
| cmd_init | `vllm/entrypoints/cli/openai.py:311` |

## 与其它模块/系统配合

- [README.md](README.md)：被 `main` 通过 `CMD_MODULES` 注册。
- [openai/chat-completion.md](../openai/chat-completion.md)：消费 `/v1/chat/completions`。
- [openai/completion.md](../openai/completion.md)：消费 `/v1/completions`。
- [可观测-metrics](../../16-observability/README.md)：客户端打印的 TTFT 来自 server 返回的 timing。

## 历史版本演进

- **v0.7（chat/complete 引入）**：基于 OpenAI SDK 的交互客户端；TTFT/吞吐统计。
- **v0.9（共享参数）**：`_add_query_options` 抽出共享选项。
- **v0.11/main**：`--system-prompt`、流式 metrics 改进；与 server 端 per-request metrics 协同（`enable_per_request_metrics`）。

## 参见

- [← 返回 CLI 首页](README.md)
- [openai/chat-completion.md](../openai/chat-completion.md)
- [openai/completion.md](../openai/completion.md)
