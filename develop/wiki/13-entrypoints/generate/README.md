[← Wiki 首页](../README.md) > [API 入口](../README.md) > Generate

# 生成任务（generate/）

> `vllm/entrypoints/generate/` 是生成类 serving 的"基座"：定义 `GenerateBaseServing`（被 OpenAI chat/completion/responses 与 speech-to-text 继承）、beam search（离线+在线 mixin）、generative scoring 端点。它本身不直接挂路由，由 `api_router.py` 装配。

## 是什么

| 文件/类 | 位置 | 职责 |
|---|---|---|
| `base/serving.py: GenerateBaseServing` | `vllm/entrypoints/generate/base/serving.py:113` | 生成类 handler 基类（`BaseServing`+`BeamSearchOnlineMixin`） |
| `base/serving.py: ServeContext` | `:102` | 每请求上下文（Generic[RequestT]） |
| `base/serving.py: build_per_request_timing_metrics`/`clamp_prompt_logprobs`/`format_token_id_placeholder`/`resolve_token_id_placeholder` | `:46/305/273/277` | 共享工具 |
| `api_router.py: register_generate_api_routers` | `vllm/entrypoints/generate/api_router.py:19` | 挂 chat/completion/responses 路由 |
| `api_router.py: init_generate_state` | `:49` | 实例化各 serving 对象挂 state |
| `factories.py: get_generate_invocation_types` | `vllm/entrypoints/generate/factories.py:16` | 生成任务→`generate`/`beam_search` 等 invocation |
| `beam_search/offline.py: BeamSearchOfflineMixin` | `vllm/entrypoints/generate/beam_search/offline.py:55` | `LLM.beam_search`（`:58`） |
| `beam_search/online.py: BeamSearchOnlineMixin` | `vllm/entrypoints/generate/beam_search/online.py:22` | serving 侧 beam_search 钩子（ABC） |
| `generative_scoring/serving.py: ServingGenerativeScoring` | `vllm/entrypoints/generate/generative_scoring/serving.py:145` | 生成式打分端点 |

`GenerateBaseServing`（`:113`）封装：`engine_client`、`models`、`request_logger`、parser 注入点、`ServeContext` 管理 per-request 计时（`build_per_request_timing_metrics`，`:46`）、`clamp_prompt_logprobs`（`:305`，按 `request.logprobs` 裁）、`format_token_id_placeholder`/`resolve_token_id_placeholder`（token id 调试占位符）。

`init_generate_state`（`api_router.py:49`）构造 `OpenAIServingChat`/`OpenAIServingCompletion`/`OpenAIServingResponses`/`ServingGenerativeScoring` 挂 state；`register_generate_api_routers`（`:19`）按子模块 `attach_router` 挂路由。

`BeamSearchOfflineMixin.beam_search`（`:58`）实现离线 beam search：多步生成 + bitmask 约束（`_beam_search_step`：`:193`、`_init_beam_search_structured_output`：`:327`、`_build_beam_sampling_params`：`:397`）。

`ServingGenerativeScoring`（`generative_scoring/serving.py:145`）：用生成模型做"给候选答案打分"（generate-based scoring），区别于 pooling 的 cross-encoder 打分。

## 为什么

- **生成 handler 共享基类**：chat/completion/responses/asr 都是"prompt→generate→parse"，`GenerateBaseServing` 统一 engine 调用、timing、logprobs 处理，子类只补协议解析。
- **beam search 双轨**：离线 `LLM.beam_search`（同步、多步）与在线 mixin（serving 钩子）共享 `_beam_search_step` 与 bitmask structured output，但调用形态不同。
- **generative scoring 独立端点**：基于生成的打分（如"模型对每个候选答案的 logprob"）与 pooling 打分语义不同，单列端点避免混淆。
- **timing metrics 下沉**：`build_per_request_time_metrics` 在基类，所有生成 endpoint 共用 per-request 计时（需 `enable_per_request_metrics`）。

## 怎么做

`init_app_state`（`api_server.py:397`）调 `init_generate_state`；`build_app`（`:203`）调 `register_generate_api_routers`。

### 离线 beam search

```python
from vllm import LLM, BeamSearchParams
llm = LLM(model="...")
outs = llm.beam_search(["prompt"], BeamSearchParams(beam_width=4, max_tokens=64))
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| GenerateBaseServing | `vllm/entrypoints/generate/base/serving.py:113` |
| build_per_request_timing_metrics | `vllm/entrypoints/generate/base/serving.py:46` |
| clamp_prompt_logprobs | `vllm/entrypoints/generate/base/serving.py:305` |
| register_generate_api_routers | `vllm/entrypoints/generate/api_router.py:19` |
| init_generate_state | `vllm/entrypoints/generate/api_router.py:49` |
| get_generate_invocation_types | `vllm/entrypoints/generate/factories.py:16` |
| BeamSearchOfflineMixin | `vllm/entrypoints/generate/beam_search/offline.py:55` |
| beam_search | `vllm/entrypoints/generate/beam_search/offline.py:58` |
| _beam_search_step | `vllm/entrypoints/generate/beam_search/offline.py:193` |
| BeamSearchOnlineMixin | `vllm/entrypoints/generate/beam_search/online.py:22` |
| ServingGenerativeScoring | `vllm/entrypoints/generate/generative_scoring/serving.py:145` |

## 与其它模块/系统配合

- [openai/chat-completion.md](../openai/chat-completion.md)：`OpenAIServingChat` → `GenerateBaseServing`。
- [openai/completion.md](../openai/completion.md)：`OpenAIServingCompletion` → `GenerateBaseServing`。
- [openai/responses.md](../openai/responses.md)：`OpenAIServingResponses` → `GenerateBaseServing`。
- [speech-to-text/README.md](../speech-to-text/README.md)：`SpeechToTextBaseServing` → `GenerateBaseServing`。
- [scale-out/token-in-token-out.md](../scale-out/token-in-token-out.md)：`ServingTokens` → `GenerateBaseServing`。
- [llm.md](../llm.md)：`BeamSearchOfflineMixin` 注入 `LLM`。
- [采样-结构化](../../06-sampling-decoding/structured-output/README.md)：`beam_search` bitmask、`StructuredOutputsParams`。

## 历史版本演进

- **v0.5（生成基类）**：`OpenAIServingChat`/`Completion` 共享逻辑分散。
- **v0.7（beam search）**：`LLM.beam_search` 离线版。
- **v0.8（V1 + GenerateBaseServing）**：抽 `GenerateBaseServing`；beam search 在线 mixin。
- **v0.9（serve/engine 抽 BaseServing）**：`GenerateBaseServing` 改继承 `BaseServing`；timing metrics、logprobs clamp 下沉。
- **v0.10（generative scoring + responses）**：`ServingGenerativeScoring`；`OpenAIServingResponses` 继承基类。
- **v0.11/main**：`ServeContext` 泛型化；`format_token_id_placeholder`/`resolve_token_id_placeholder`；beam search structured output bitmask（`_bitmask_to_token_ids`，`:41`）。

## 模块导航

| 页 | 主题 |
|---|---|
| [api-router.md](api-router.md) | 路由注册 + state init |
| [factories.md](factories.md) | invocation 类型映射 |
| [base-serves.md](base-serves.md) | `GenerateBaseServing` |
| [beam-search.md](beam-search.md) | beam search 离线/在线 |
| [generative-scoring.md](generative-scoring.md) | 生成式打分 |

## 参见

- [← 返回 API 入口首页](../README.md)
- [openai/chat-completion.md](../openai/chat-completion.md)
- [speech-to-text/README.md](../speech-to-text/README.md)
- [采样-结构化](../../06-sampling-decoding/structured-output/README.md)
