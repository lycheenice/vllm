[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Scale-out](README.md) > derender

# scale_out/derender/（ServingDerender）

> `ServingDerender` 实现 derender 端点：收推理节点产出的 token id + 原始请求上下文，经 `OnlineDerenderer`（detokenize + parser）还原成 OpenAI `ChatCompletionResponse`/`CompletionResponse`。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `ServingDerender` | `vllm/entrypoints/scale_out/derender/serving.py:38` | derender handler，继承 `BaseServing` |
| `ServingDerender._extract_mm_features` | `:166` | 反向还原多模态上下文（若需） |
| `derender/api_router.py: derender` | `vllm/entrypoints/scale_out/derender/api_router.py:25` | 依赖注入 |
| `derender/api_router.py: derender_chat_completion`/`derender_completion` | `:39/64` | `/derender/chat/completions`、`/derender/completions` |

`derender_chat_completion`（`:39`）收 `DerenderChatRequest`（`token_in_token_out/protocol.py:239`，含 output_token_ids + 原 messages + sampling_params）→ `ServingDerender` 构造 `OnlineDerenderer` → detokenize + parser → `ChatCompletionResponse`。

## 为什么

- **后处理外移**：detokenize/tool parser/reasoning parser 在 CPU 节点跑，GPU 节点只产 token。
- **与 render 对称**：render 把 messages→token，derender 把 token→messages/response，二者共享 `OnlineRenderer`/`OnlineDerenderer` 配置（chat_template/tool_parser/reasoning_parser）。
- **复用 parser 状态**：derender 端可重建 parser 状态机，保证流式分块与单体一致（待核实是否流式 derender）。

## 怎么做

见 [README](README.md) 流程图。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingDerender | `vllm/entrypoints/scale_out/derender/serving.py:38` |
| derender_chat_completion | `vllm/entrypoints/scale_out/derender/api_router.py:39` |
| derender_completion | `vllm/entrypoints/scale_out/derender/api_router.py:64` |
| DerenderChatRequest | `vllm/entrypoints/scale_out/token_in_token_out/protocol.py:239` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing`。
- [render.md](render.md)：对称端。
- [token-in-token-out.md](token-in-token-out.md)：产 token。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：`OnlineDerenderer`、tool/reasoning parser。

## 历史版本演进

- **v0.10（引入）**：`ServingDerender` + derender 端点；`DerenderChatRequest`/`DerenderCompletionRequest`。
- **main**：与 harmony Responses 的 derender 协同（待核实）。

## 参见

- [← 返回 Scale-out 首页](README.md)
- [render.md](render.md)
- [token-in-token-out.md](token-in-token-out.md)
