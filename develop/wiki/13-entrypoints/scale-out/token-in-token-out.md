[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Scale-out](README.md) > token-in-token-out

# scale_out/token_in_token_out/（ServingTokens + 协议）

> `token_in_token_out/` 定义 scale-out 的"纯推理"端点与协议：`ServingTokens` 收 `GenerateRequest`（已 tokenize 的 prompt + 多模态特征），直接 `engine_client.generate`，输出 `GenerateResponse`（output_token_ids）。同时承载 `DerenderChatRequest`/`DerenderCompletionRequest` 协议（在 `protocol.py`）。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `ServingTokens` | `vllm/entrypoints/scale_out/token_in_token_out/serving.py:61` | token 级推理 handler，继承 `GenerateBaseServing` |
| `ServingTokens._create_tokens_logprobs` | `:468` | 构造 token logprobs 输出 |
| `protocol.py: PlaceholderRangeInfo`/`MultiModalFeatures` | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:28/42` | 占位区间 + 多模态特征 schema |
| `protocol.py: GenerateRequest` | `:66` | token 级请求（prompt_token_ids/mm_features/sampling_params） |
| `protocol.py: GenerateResponse`/`GenerateResponseChoice`/`GenerateStreamResponse` | `:215/176/202` | token 级响应 |
| `protocol.py: DerenderChatRequest`/`DerenderCompletionRequest` | `:239/268` | derender 输入 |
| `api_router.py: tokenization`/`generate_tokens`/`engine_client`/`generate`/`attach_router` | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:31/35/39/58/76` | 依赖注入 + `/generate` 端点 |
| `mm_serde.py: encode_mm_kwargs_item`/`decode_mm_kwargs_item` | `vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py:17/24` | 多模态 kwargs item 序列化 |

`generate`（`:58`）收 `GenerateRequest` → `ServingTokens` 把 `prompt_token_ids` + 解码后的 `mm_kwargs`（`decode_mm_kwargs_item`，`:24`）组装 `EngineInput` → `engine_client.generate` → 收割 token id → `_create_tokens_logprobs`（`:468`，若需）→ `GenerateResponse`。

## 为什么

- **跳过 renderer**：prompt 已由 render 节点 tokenize，推理节点直接用 `tokens_input` 路径，省重复 tokenize 与 chat template 计算。
- **多模态跨进程**：`mm_serde` 用 base64 把 `MultiModalKwargsItem` 序列化，跨 HTTP/进程传特征；`MultiModalFeatures`（`:42`）描述特征张量与 placeholder（`PlaceholderRangeInfo`，`:28`）。
- **token logprobs 输出**：`_create_tokens_logprobs` 让客户端可在 derender 侧重组 logprobs（与单体行为一致）。
- **统一协议**：derender 的输入协议（`DerenderChatRequest`）也在此定义，保证 render/推理/derender 三方用同一 schema 包。

## 怎么做

见 [README](README.md) 流程图。`/generate` 端点接受 `GenerateRequest`，返回 `GenerateResponse` 或 SSE 流。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingTokens | `vllm/entrypoints/scale_out/token_in_token_out/serving.py:61` |
| _create_tokens_logprobs | `vllm/entrypoints/scale_out/token_in_token_out/serving.py:468` |
| generate 端点 | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:58` |
| attach_router | `vllm/entrypoints/scale_out/token_in_token_out/api_router.py:76` |
| GenerateRequest | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:66` |
| GenerateResponse | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:215` |
| DerenderChatRequest | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:239` |
| mm_serde encode | `vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py:17` |
| mm_serde decode | `vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py:24` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing`。
- [render.md](render.md)：上游产 `GenerateRequest`。
- [derender.md](derender.md)：下游消费 `GenerateResponse`。
- [多模态](../../11-multimodal/README.md)：`MultiModalKwargsItem`、`mm_serde`。
- [引擎核心-AsyncLLM](../../01-engine-core/async-llm-frontend.md)：`tokens_input` 路径。

## 历史版本演进

- **v0.10（引入）**：`ServingTokens` + `GenerateRequest`/`GenerateResponse` 协议；`mm_serde`；token logprobs。
- **v0.11/main**：`_create_tokens_logprobs` 完善；`PlaceholderRangeInfo`/`MultiModalFeatures`；流式 `GenerateStreamResponse`（`:202`）。

## 参见

- [← 返回 Scale-out 首页](README.md)
- [render.md](render.md)
- [derender.md](derender.md)
- [多模态](../../11-multimodal/README.md)
