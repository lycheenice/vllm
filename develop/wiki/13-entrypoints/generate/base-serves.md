[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Generate](README.md) > base

# generate/base/serving.py（GenerateBaseServing）

> `GenerateBaseServing` 是所有生成类 handler（chat/completion/responses/asr/token-in-token-out）的基类。它继承 `BaseServing` 与 `BeamSearchOnlineMixin`，封装 engine 调用、parser 注入、per-request timing、logprobs 裁剪、token id 占位符等共享逻辑。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `GenerateBaseServing` | `vllm/entrypoints/generate/base/serving.py:113` | 基类 |
| `ServeContext` | `:102` | 每请求上下文（Generic[RequestT]，持 timing/parser 状态） |
| `build_per_request_timing_metrics` | `:46` | 从 `UsageInfo`/`RequestOutput` 构造 `PerRequestTimingMetrics` |
| `format_token_id_placeholder` | `:273` | 把 token id 格式化为 `token_id_N` 调试串 |
| `resolve_token_id_placeholder` | `:277` | 反向：从 `token_id_N` 解析 token id |
| `clamp_prompt_logprobs` | `:305` | 按 `request.logprobs` 裁剪 prompt logprobs |

`GenerateBaseServing.__init__` 持 `engine_client`/`models`/`request_logger`/`return_tokens_as_token_ids`；提供 `_check_model`（继承）、`_request_id` 生成、`serve_context` 管理每请求状态。

`build_per_request_time_metrics`（`:46`）在 `enable_per_request_metrics` 时从 `RequestOutput` 的 timing（TTFT、生成时长、token 数）组装 `PerRequestTimingMetrics`，附在响应 `metadata` 上。

## 为什么

- **生成共性下沉**：所有生成 handler 都要调 `engine_client.generate`、管 parser、报 timing、裁 logprobs，基类把这些统一，子类只写"协议↔RequestOutput"映射。
- **`ServeContext` 状态隔离**：每请求一份 `ServeContext`，持 parser 实例与累积状态，避免流式时跨请求污染。
- **token id 调试**：`return_tokens_as_token_ids` 模式下用 `token_id_N` 占位符表示 token，便于精确复现，`format`/`resolve` 配对处理。
- **beam search 在线钩子**：通过 `BeamSearchOnlineMixin`（ABC）让 serving 类可挂 beam search 端点，不强制实现。

## 怎么做

子类典型：

```python
class OpenAIServingChat(GenerateBaseServing):
    async def create_chat_completion(self, request, raw_request):
        err = await self._check_model(request)
        if err: return err
        engine_input = await self.online_renderer.preprocess_chat(...)
        async for output in self.engine_client.generate(..., stream=True):
            ctx = self.serve_context_for(request_id)
            delta = ctx.parser.parse(output)
            ...yield SSE
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| GenerateBaseServing | `vllm/entrypoints/generate/base/serving.py:113` |
| ServeContext | `vllm/entrypoints/generate/base/serving.py:102` |
| build_per_request_timing_metrics | `vllm/entrypoints/generate/base/serving.py:46` |
| format_token_id_placeholder | `vllm/entrypoints/generate/base/serving.py:273` |
| clamp_prompt_logprobs | `vllm/entrypoints/generate/base/serving.py:305` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing`。
- [beam-search.md](beam-search.md)：`BeamSearchOnlineMixin`。
- [openai/chat-completion.md](../openai/chat-completion.md)/[completion.md](../openai/completion.md)/[responses.md](../openai/responses.md)：子类。
- [speech-to-text/base.md](../speech-to-text/base.md)：`SpeechToTextBaseServing` 子类。
- [可观测-metrics](../../16-observability/README.md)：`PerRequestTimingMetrics`。

## 历史版本演进

- **v0.8（引入）**：`GenerateBaseServing` 抽出。
- **v0.9（继承 BaseServing）**：改两层继承。
- **v0.10（ServeContext + per-request metrics）**：`ServeContext` 泛型；`build_per_request_timing_metrics`。
- **v0.11/main**：`format_token_id_placeholder`/`resolve_token_id_placeholder`；`return_tokens_as_token_ids`。

## 参见

- [← 返回 Generate 首页](README.md)
- [beam-search.md](beam-search.md)
- [serve/engine-serve.md](../serve/engine-serve.md)
