[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > completion

# completion/serving.py（/v1/completions）

> `OpenAIServingCompletion` 实现 OpenAI Completions API（`/v1/completions`）：接受 prompt（字符串/token id 序列/多模态 prompt），渲染成 `EngineInput`，提交 `AsyncLLM.generate`，按流式或一次性返回 `CompletionResponse`/`CompletionStreamResponse`。它是 chat completion 的"裸 token"版本，不带 chat template。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `OpenAIServingCompletion` | `vllm/entrypoints/openai/completion/serving.py:55` | `/v1/completions` handler，继承 `GenerateBaseServing` |

`__init__`（`:56`）持有：`engine_client`、`models`、`online_renderer`、`request_logger`、`return_tokens_as_token_ids`、`enable_prompt_tokens_details`、`enable_force_include_usage`、`enable_per_request_metrics`、`default_sampling_params`（`model_config.get_diff_sampling_param()`）。

主入口 `create_completion`（路由 `completion/api_router.py:46`）：

1. `_check_model(request)` → 错误返回 `ErrorResponse`。
2. `engine_client.errored` → 抛 dead_error。
3. 把 `request.prompt`（`str | list[str] | list[int] | list[list[int]]`）经 `online_renderer.preprocess_completion` → `EngineInput`。
4. 构造 `SamplingParams`（`max_tokens`/`temperature`/`top_p`/`logprobs`/`echo`/`n`/`seed`/`stop`/`structured_outputs`）。
5. 流式：`generate(..., stream=True)` → 每 chunk 组装 `CompletionResponseStreamChoice`/`CompletionStreamResponse`，按 `should_include_usage` 在末尾发 usage。
6. 非流式：收集所有 `RequestOutput` → 选 `echo`/`logprobs` → `CompletionResponse` + `UsageInfo`。

## 为什么

- **裸 token 入口**：补全 API 不套 chat template，适合 code/completion 模型、自定义 prompt 模板场景；`prompt` 支持 token id 序列（`list[int]`），便于复现/精确控制。
- **与 chat 共享基类**：`GenerateBaseServing` 提供 `_check_model`/timing/logprobs clamp/`format_token_id_placeholder`，completion 与 chat 行为一致，仅渲染协议不同。
- **`return_tokens_as_token_ids`**：调试场景下把 token 以 `token_id_N` 字面返回，便于看确切 token（对齐 OpenAI `logprobs` 调试习惯）。
- **logprobs 完整复刻**：`CompletionLogProbs` 复刻 OpenAI `logprobs` 结构（`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`），`clamp_prompt_logprobs`（基类）按 `request.logprobs` 裁剪。
- **`echo` 支持**：在响应里回显 prompt token，由 `max_tokens` 调整逻辑保证不超 `max_model_len`（`serve/utils/api_utils.get_max_tokens`）。
- **stream usage 控制**：`should_include_usage`（`api_utils.py:276`）按 `stream_options.include_usage` 决定是否在流末尾发 usage chunk。

## 怎么做

### 路由

`completion/api_router.py:attach_router`（`:69`）注册 `POST /v1/completions` → `create_completion`（`:46`）。`completion(request)`（`:30`）从 `app.state` 取 serving 对象。

### 请求示例

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","prompt":"Once upon a time","max_tokens":32,"stream":true}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIServingCompletion | `vllm/entrypoints/openai/completion/serving.py:55` |
| __init__ | `vllm/entrypoints/openai/completion/serving.py:56` |
| 路由注册 | `vllm/entrypoints/openai/completion/api_router.py:69` |
| create_completion | `vllm/entrypoints/openai/completion/api_router.py:46` |
| completion 依赖注入 | `vllm/entrypoints/openai/completion/api_router.py:30` |
| logprobs 协议 | `vllm/entrypoints/openai/completion/protocol.py`（`CompletionLogProbs`，待核实行号） |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing` 基类。
- [engine-protocol.md](engine-protocol.md)：`UsageInfo`/`ErrorResponse`/`PerRequestTimingMetrics`。
- [models.md](models.md)：`_check_model`。
- [chat-completion.md](chat-completion.md)：姊妹端点，共享 renderer/parser。
- [serve/utils.md](../serve/utils.md)：`get_max_tokens`/`should_include_usage`。
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)：`structured_outputs` 下发。
- [多模态](../../11-multimodal/README.md)：`preprocess_completion` 接受 `MessagesPrompt` 含 `multi_modal_data`。

## 历史版本演进

- **v0.5（completion 首版）**：`OpenAIServingCompletion` 内联 api_server；支持 `echo`/`logprobs`/`n`。
- **v0.7（V1 + renderer）**：迁 `AsyncLLM`；`preprocess_completion` 统一处理 prompt 多形态。
- **v0.8（return_tokens_as_token_ids）**：调试字段加入。
- **v0.9（stream_options + prompt_tokens_details）**：`enable_force_include_usage`、`enable_prompt_tokens_details`；`stream_options.include_usage`。
- **v0.10（harmony 不介入）**：completion 不走 harmony；保持纯 token 路径。
- **v0.11/main**：per-request metrics；`default_sampling_params` 由 `get_diff_sampling_param` 提供；structured_outputs 字段（待核实 completion 是否完整支持 `structural_tag`）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [chat-completion.md](chat-completion.md)
- [generate/base-serves.md](../generate/base-serves.md)
- [engine-protocol.md](engine-protocol.md)
- [serve/utils.md](../serve/utils.md)
