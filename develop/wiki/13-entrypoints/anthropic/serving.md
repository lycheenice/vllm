[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Anthropic](README.md) > serving

# anthropic/serving.py + protocol.py + api_router.py

> Anthropic 子系统的三件套：`serving.py` 实现 `AnthropicServingMessages` handler，`protocol.py` 定义 Anthropic 协议 schema，`api_router.py` 注册路由并翻译错误。详见本目录 [README](README.md)；本页补充协议字段与流事件细节。

## 是什么

见 [README](README.md) 的"是什么"表，本页聚焦协议细节。

### 请求字段（`AnthropicMessagesRequest`，`vllm/entrypoints/anthropic/protocol.py:118`）

| 字段 | 对应 OpenAI | 说明 |
|---|---|---|
| `model` | `model` | 模型/base/LoRA 名 |
| `messages` | `messages` | Anthropic content block 风格 |
| `system` | system message | Anthropic system 独立字段 |
| `max_tokens` | `max_tokens` | 必填 |
| `temperature`/`top_p`/`top_k` | 同名 | `top_k` 为 Anthropic 特有 |
| `stop_sequences` | `stop` | |
| `tools`/`tool_choice` | `tools`/`tool_choice` | `AnthropicTool`/`AnthropicToolChoice`（`:72/91`） |
| `json_output`/`output_config` | `response_format` | `AnthropicJsonOutputFormat`（`:104`）/`AnthropicOutputConfig`（`:111`） |
| `context_management` | — | `AnthropicContextManagement`（`:227`），缓存控制 |
| `stream` | `stream` | |

### 流事件（`AnthropicStreamEvent`，`:182`）

Anthropic 流式事件序列：`message_start` → 若干 `content_block_start`/`content_block_delta`/`content_block_stop`（按 block）→ `message_delta`（含 stop_reason/usage）→ `message_stop`。`AnthropicDelta`（`:163`）承载 delta 内容（text/tool_use）。

### 错误（`AnthropicErrorResponse`，`:18`）

`AnthropicError`（`:11`）：`type`/`message` 字段，与 OpenAI `ErrorResponse.error` 不同结构。`translate_error_response`（`api_router.py:37`）按错误 `type` 映射。

## 为什么

- **字段级互译**：`system` 独立、`top_k`、`stop_sequences`、`context_management` 都是 Anthropic 独有，必须在协议层吸收；避免污染 OpenAI schema。
- **流事件分块**：Anthropic 按 content block 分事件，与 OpenAI 单一 `choices[].delta` 不同；`AnthropicServingMessages` 维护 block 状态机把 OpenAI delta 重组。
- **错误类型映射**：Anthropic SDK 按 `error.type` 分派处理，`translate_error_response` 保证 4xx/5xx 行为与原生 Anthropic 一致。

## 怎么做

见 [README](README.md) 流程图与用法。

### 关键代码导航

见 [README](README.md) 表格，协议字段位置集中在 `vllm/entrypoints/anthropic/protocol.py`。

## 与其它模块/系统配合

见 [README](README.md)。

## 历史版本演进

见 [README](README.md)。

## 参见

- [← 返回 Anthropic 首页](README.md)
- [openai/chat-completion.md](../openai/chat-completion.md)
