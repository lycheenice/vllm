[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > engine/protocol

# engine/protocol.py（OpenAI 共享协议）

> `vllm/entrypoints/openai/engine/protocol.py` 定义 OpenAI 风格 API 的"协议基座"：所有 chat/completion/responses/models 协议类共享的 Pydantic 基类、错误响应、模型卡、usage、tool call、delta、响应格式、logits processor 构造等。它处于 `openai/` 子包根部，被各 serving handler 与 protocol 文件依赖。

## 是什么

| 类/函数 | 位置 | 职责 |
|---|---|---|
| `OpenAIBaseModel` | `vllm/entrypoints/openai/engine/protocol.py:28` | Pydantic 基类：`extra="allow"` + `__log_extra_fields__` 记录被忽略字段 |
| `ErrorInfo` / `ErrorResponse` | `:60` / `:67` | OpenAI 错误响应 |
| `ModelPermission` / `ModelCard` / `ModelList` | `:71` / `:86` / `:97` | `/v1/models` 响应 |
| `PromptTokenUsageInfo` / `UsageInfo` / `PerRequestTimingMetrics` | `:102` / `:111` / `:118` | token 用量与每请求计时 |
| `RequestResponseMetadata` | `:126` | 请求元数据（透传/日志） |
| `JsonSchemaResponseFormat` / `LegacyStructuralTag` / `LegacyStructuralTagResponseFormat` / `StructuralTagResponseFormat` / `ResponseFormat` | `:131` 起 | 结构化输出格式 |
| `validate_structural_tag_response_format` / `validate_structural_tag_payload` / `validate_structured_outputs_structural_tag` | `:175/208/231` | structural tag 校验 |
| `StreamOptions` | `:249` | 流式 usage 控制选项 |
| `FunctionDefinition` | `:254` | 工具函数定义 |
| `LogitsProcessorConstructor` / `get_logits_processors` | `:273` / `:284` | 由请求构造 logits processor 链 |
| `FunctionCall` / `ToolCall` / `DeltaFunctionCall` / `DeltaToolCall` / `ExtractedToolCallInformation` | `:318` 起 | tool call 表示 |
| `DeltaMessage` | `:358` | 流式增量 |
| `GenerationError` | `:372` | serving 内部异常类 |

`OpenAIBaseModel`（`:28`）核心：`model_config = ConfigDict(extra="allow")`，并在 `__log_extra_fields__`（`@model_validator(mode="wrap")`）里缓存 `field_names`，对请求出现的非字段键 `logger.debug` 提示被忽略——既严格对齐 OpenAI schema 又容错透传未知字段。

## 为什么

- **协议单一来源**：`ErrorResponse`/`UsageInfo`/`ModelList` 等被 chat/completion/responses 共用，避免每个 endpoint 各定义一份导致序列化漂移。
- **`extra="allow"` + debug 日志**：OpenAI 经常增字段，`extra="allow"` 让未知字段不报 422 而被透传/忽略；`__log_extra_fields__` 帮助发现"客户端发了但 vLLM 没用"的字段。
- **结构化输出三格式**：`ResponseFormat` 可为 `json_schema`/`structural_tag`（新）/`legacy_structural_tag`（旧），由 `validate_structural_tag_response_format` 做迁移校验，对接 [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)。
- **logits processor 构造下沉**：`get_logits_processors`（`:284`）把请求里的 `logits_processors` 字段（字符串限定名或类）解析成可调用链，统一在 protocol 层完成，serving 层只兜结果。
- **`GenerationError` 异类**：作为 `Exception` 子类（`:372`），让 `generation_error_handler`（`serve/utils/server_utils.py:371`）把它转成 OpenAI 风格错误响应而非 500 堆栈。

## 怎么做

### 定义新协议

```python
class MyRequest(OpenAIBaseModel):
    model: str
    prompt: str
    # extra 字段自动 allow
```

继承即得 `extra="allow"` 与忽略字段日志。

### 结构化输出

请求 `response_format={"type":"json_schema","json_schema":{...}}` → `ResponseFormat` 解析 → 若 `structural_tag` 则 `validate_structural_tag_response_format` → 下发到 `StructuredOutputsParams`（`SamplingParams`）。

### Tool call 流式

`DeltaMessage`（`:358`）聚合 `role`/`content`/`tool_calls`/`function_call`/`reasoning_content`，streaming handler 每个 chunk 填一份，序列化为 SSE。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIBaseModel | `vllm/entrypoints/openai/engine/protocol.py:28` |
| 忽略字段日志 | `vllm/entrypoints/openai/engine/protocol.py:35` |
| 错误响应 | `vllm/entrypoints/openai/engine/protocol.py:60` |
| 模型卡 | `vllm/entrypoints/openai/engine/protocol.py:86` |
| Usage | `vllm/entrypoints/openai/engine/protocol.py:111` |
| 结构化输出格式 | `vllm/entrypoints/openai/engine/protocol.py:131` 起 |
| structural tag 校验 | `vllm/entrypoints/openai/engine/protocol.py:175` |
| logits processor 构造 | `vllm/entrypoints/openai/engine/protocol.py:284` |
| ToolCall | `vllm/entrypoints/openai/engine/protocol.py:327` |
| DeltaMessage | `vllm/entrypoints/openai/engine/protocol.py:358` |
| GenerationError | `vllm/entrypoints/openai/engine/protocol.py:372` |

## 与其它模块/系统配合

- [chat-completion.md](chat-completion.md)：`ChatCompletionRequest` 继承 `OpenAIBaseModel`，用 `ToolCall`/`DeltaMessage`/`UsageInfo`。
- [completion.md](completion.md)：`CompletionRequest` 同源。
- [responses.md](responses.md)：`ResponsesRequest` 继承 `OpenAIBaseModel`，用 `RequestResponseMetadata`。
- [models.md](models.md)：`ModelList`/`ModelCard` 用于 `/v1/models`。
- [serve/README.md](../serve/README.md)：`server_utils.generation_error_handler` 捕获 `GenerationError`。
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)：`ResponseFormat` → `StructuredOutputsParams`/`StructuralTagConfig`。
- [可观测-metrics](../../16-observability/README.md)：`PerRequestTimingMetrics` 被日志/metrics 引用。

## 历史版本演进

- **v0.5–v0.6（单体协议）**：协议类内联在 `api_server.py`，无共享基类。
- **v0.7（OpenAIBaseModel）**：抽 `OpenAIBaseModel` 与 `extra="allow"`；移到 `openai/protocol.py`（早名）。
- **v0.8（structural tag）**：引入 `StructuralTagResponseFormat` + 校验函数，对齐 vLLM 结构化输出 v2。
- **v0.9（logits processor + per-request metrics）**：`LogitsProcessorConstructor`/`get_logits_processors`；`PerRequestTimingMetrics` 字段。
- **v0.10（Responses + harmony）**：`RequestResponseMetadata` 为 Responses 流式服务；`GenerationError` 让 serving 内部异常走 OpenAI 错误响应。
- **v0.11/main**：`legacy_structural_tag` 兼容旧格式校验；`DeltaMessage` 加 `reasoning_content` 字段；`PromptTokenUsageInfo` 加 `multimodal_tokens`（待核实是否在本文件）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [chat-completion.md](chat-completion.md)
- [responses.md](responses.md)
- [models.md](models.md)
- [采样-结构化输出](../../06-sampling-decoding/structured-output/README.md)
