[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > chat_completion

# chat_completion/serving.py + batch_serving.py

> `OpenAIServingChat` 实现 `/v1/chat/completions`（OpenAI Chat Completions API），把 chat 消息渲染成 token、提交给 `AsyncLLM`、把流式/非流式输出经 parser 还原成 `ChatCompletionResponse`/`ChatCompletionStreamResponse`。`OpenAIServingChatBatch` 扩展出 `/v1/chat/completions/batch` 单请求多对话并发端点。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `OpenAIServingChat` | `vllm/entrypoints/openai/chat_completion/serving.py:106` | `/v1/chat/completions` 主 handler，继承 `GenerateBaseServing` |
| `OpenAIServingChatBatch` | `vllm/entrypoints/openai/chat_completion/batch_serving.py:36` | 继承 `OpenAIServingChat`，多对话并发 |
| `_get_mm_token_counts` | `vllm/entrypoints/openai/chat_completion/serving.py:73` | 从 `mm_placeholders` 算每模态 token 数 |
| `_make_prompt_tokens_details` | `:90` | 构造 `PromptTokenUsageInfo`（cached + multimodal） |

`OpenAIServingChat.__init__`（`:107`）持有：`engine_client`、`models`、`response_role`、`online_renderer`、`request_logger`、`chat_template`/`chat_template_content_format`、`default_chat_template_kwargs`、parser 相关（`enable_auto_tools`/`tool_parser`/`reasoning_parser`/`exclude_tools_when_tool_choice_none`）、`return_tokens_as_token_ids`、`enable_prompt_tokens_details`、`enable_force_include_usage`、`enable_per_request_metrics`、`default_sampling_params`。

主入口 `create_chat_completion`（路由在 `chat_completion/api_router.py:53`）：

1. `_check_model(request)` → 错误返回 `ErrorResponse`。
2. `engine_client.errored` → 抛 dead_error。
3. `online_renderer.validate_chat_template`（若非 harmony）。
4. `online_renderer.preprocess_chat(messages, ...)` → `(conversation, EngineInput)`。
5. 构造 `SamplingParams`（合并 default + request，`guidance`/`structured_outputs`）。
6. 流式：`generate(..., stream=True)` → `AsyncIterator[RequestOutput]` → 每 chunk `parser.parse` → `ChatCompletionStreamResponse` SSE。
7. 非流式：`generate(...)` 收集 → `ChatCompletionResponse`，含 `usage`/`prompt_tokens_details`。

`OpenAIServingChatBatch.render_batch_chat_request`（`batch_serving.py:43`）：

- 校验 model + engine 状态。
- 非 harmony：一次 `validate_chat_template`（整批共享）。
- 对每条 `request.messages` 单独 `renderer.preprocess_chat` → 收集 `(all_conversations, all_engine_prompts)`。
- 提交并发请求，`merge_async_iterators` 合并流，按 `index` 区分 choice。

## 为什么

- **继承 `GenerateBaseServing`**：共用 `_check_model`/`_request_id`/timing metrics/`clamp_prompt_logprobs`/`format_token_id_placeholder` 等工具，与 Responses/Completion 行为一致。
- **renderer 持有解析器**：`OnlineRenderer` 封装 chat template + tool/reasoning parser；serving 类只负责"协议 ↔ 引擎"映射，不直接处理 tokenization，便于 render-only server 复用。
- **prompt_tokens_details**：`_make_prompt_tokens_details`（`:90`）把 `cached_tokens`（prefix cache 命中）与 `multimodal_tokens`（每模态占位 token）汇总，让 usage 更透明。
- **tool/reasoning parser 状态机**：流式时 `parser.parse` 是增量状态机，把 token 流切成 `content`/`tool_calls`/`reasoning_content` 三通道 delta，匹配 OpenAI tool call schema。
- **batch 并发**：`OpenAIServingChatBatch` 让客户端用一次 HTTP 请求并发跑 N 个对话（各为独立 choice），降低连接/调度开销，适合评测与批量交互。
- **harmony 分流**：当 `renderer.use_harmony` 为真（Responses 风格），走 `_make_request_with_harmony` 而非 `preprocess_chat`，与 Responses API 共用 harmony 渲染路径。

## 怎么做

### 路由

`chat_completion/api_router.py:attach_router`（`:105`）注册：

- `POST /v1/chat/completions` → `create_chat_completion`（`:53`）。
- `POST /v1/chat/completions/batch` → `create_batch_chat_completion`（`:90`）。

依赖注入 `chat(request)`（`:32`）/`batch_chat(request)`（`:36`）从 `request.app.state` 取 serving 对象，缺失返回 None（404）。

### 流式与非流式

```python
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)
    if handler is None: return 404
    generator_or_resp = await handler.create_chat_completion(request, raw_request)
    if isinstance(generator_or_resp, ErrorResponse): return JSONResponse(...)
    if request.stream: return StreamingResponse(generator, media_type="text/event-stream")
    return JSONResponse(generator_or_resp)
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIServingChat | `vllm/entrypoints/openai/chat_completion/serving.py:106` |
| __init__ | `vllm/entrypoints/openai/chat_completion/serving.py:107` |
| mm token counts | `vllm/entrypoints/openai/chat_completion/serving.py:73` |
| prompt_tokens_details | `vllm/entrypoints/openai/chat_completion/serving.py:90` |
| 路由注册 | `vllm/entrypoints/openai/chat_completion/api_router.py:105` |
| create_chat_completion | `vllm/entrypoints/openai/chat_completion/api_router.py:53` |
| batch 路由 | `vllm/entrypoints/openai/chat_completion/api_router.py:90` |
| OpenAIServingChatBatch | `vllm/entrypoints/openai/chat_completion/batch_serving.py:36` |
| render_batch_chat_request | `vllm/entrypoints/openai/chat_completion/batch_serving.py:43` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing` 基类。
- [engine-protocol.md](engine-protocol.md)：`ChatCompletionRequest`/`DeltaMessage`/`ToolCall`/`UsageInfo`。
- [models.md](models.md)：`_check_model` 路由 LoRA。
- [chat-utils.md](../chat-utils.md)：`preprocess_chat` → `parse_chat_messages_async`。
- [responses.md](responses.md)：harmony 路径共享；`OpenAIServingResponses` 也继承 `GenerateBaseServing`。
- [anthropic/serving.md](../anthropic/serving.md)：`AnthropicServingMessages` 继承 `OpenAIServingChat`。
- [多模态](../../11-multimodal/README.md)：`_get_mm_token_counts`、MM placeholder。
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)：`ResponseFormat`/structured_outputs 下发。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：tool_parsers/reasoning parsers。

## 历史版本演进

- **v0.5（chat 首版）**：`OpenAIServingChat` 内联在 api_server；支持 text + image，无 tool parser。
- **v0.7（tool call parser）**：引入 `ToolParserManager` 与 `--tool-call-parser`；流式 delta 切 tool_calls。
- **v0.8（V1 + renderer）**：迁到 `AsyncLLM` + `OnlineRenderer`；serving 类瘦身。
- **v0.9（prompt_tokens_details）**：`enable_prompt_tokens_details`、cached/multimodal tokens；`return_tokens_as_token_ids`。
- **v0.10（harmony 分流 + batch）**：`use_harmony` 分支；`OpenAIServingChatBatch` + `/v1/chat/completions/batch`；`exclude_tools_when_tool_choice_none`。
- **v0.11/main**：`default_chat_template_kwargs`；`enable_force_include_usage`；per-request metrics；`merge_async_iterators` 用于 batch 流合并；reasoning parser 与 harmony 协同（待核实完整状态机）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [generate/base-serves.md](../generate/base-serves.md)
- [responses.md](responses.md)
- [anthropic/serving.md](../anthropic/serving.md)
- [chat-utils.md](../chat-utils.md)
- [多模态](../../11-multimodal/README.md)
