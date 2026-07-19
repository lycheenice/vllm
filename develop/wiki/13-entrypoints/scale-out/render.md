[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Scale-out](README.md) > render

# scale_out/render/（ServingRender）

> `ServingRender` 实现 render 端点：收 OpenAI chat/completion 请求，经 `OnlineRenderer` 预处理（tokenize + 多模态），输出 `GenerateRequest`（prompt_token_ids + mm_features + sampling_params），供 token-in/token-out 推理节点消费。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `ServingRender` | `vllm/entrypoints/scale_out/render/serving.py:38` | render handler，继承 `BaseServing` |
| `ServingRender._extract_mm_features` | `:212` | 从 `EngineInput` 抽多模态特征供跨进程 |
| `render/api_router.py: render` | `vllm/entrypoints/scale_out/render/api_router.py:22` | 依赖注入 |
| `render/api_router.py: render_chat_completion`/`render_completion` | `:37/62` | `/render/chat/completions`、`/render/completions` |

`render_chat_completion`（`:37`）收 `ChatCompletionRequest` → `ServingRender` 调 `online_renderer.preprocess_chat` → 组装 `GenerateRequest`（`token_in_token_out/protocol.py:66`）→ 返回 JSON（客户端再转交推理节点）。

## 为什么

- **预处理外移**：把 chat template/MM processor 这类 CPU 工作从 GPU 节点剥离，GPU 节点收 token 直跑。
- **复用 `OnlineRenderer`**：与单体 server 同一 renderer，渲染结果一致。
- **多模态特征序列化**：`_extract_mm_features`（`:212`）+ `mm_serde`（见 [token-in-token-out.md](token-in-token-out.md)）让特征可跨进程。

## 怎么做

见 [README](README.md) 流程图。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingRender | `vllm/entrypoints/scale_out/render/serving.py:38` |
| _extract_mm_features | `vllm/entrypoints/scale_out/render/serving.py:212` |
| render_chat_completion | `vllm/entrypoints/scale_out/render/api_router.py:37` |
| render_completion | `vllm/entrypoints/scale_out/render/api_router.py:62` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing`。
- [token-in-token-out.md](token-in-token-out.md)：消费 render 输出。
- [chat-utils.md](../chat-utils.md)：renderer 内部。
- [多模态](../../11-multimodal/README.md)：`_extract_mm_features`。

## 历史版本演进

- **v0.10（引入）**：`ServingRender` + render 端点。
- **v0.11/main**：`_extract_mm_features`；与 `mm_serde` 协同。

## 参见

- [← 返回 Scale-out 首页](README.md)
- [token-in-token-out.md](token-in-token-out.md)
