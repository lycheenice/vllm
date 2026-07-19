[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > tokenize

# tokenize/serving.py（/v1/tokenize & /v1/detokenize）

> `ServingTokenization` 暴露 OpenAI 兼容的 tokenize/detokenize 调试端点，让客户端在不生成的情况下获取 prompt 的 token id（或反向），用于精确控制上下文与复现。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `ServingTokenization` | `vllm/entrypoints/serve/tokenize/serving.py:32` | `/v1/tokenize`、`/v1/detokenize` handler，继承 `BaseServing` |
| `TokenizerInfo` | `vllm/entrypoints/serve/tokenize/serving.py:164` | token 边界/位置信息辅助 |

`__init__` 持 `models`/`online_renderer`/`request_logger`/`chat_template`/`chat_template_content_format`/`default_chat_template_kwargs`/`trust_request_chat_template`。

`tokenize`（路由 `tokenize/api_router.py:47`）：接受 `model` + `prompt`/`messages`（completion 或 chat 形式）→ 经 `OnlineRenderer` 渲染 → 返回 `token_ids`、`tokens`（可选 decoded 表层）、`count`。

`detokenize`（路由 `:72`）：接受 `token_ids` → tokenizer decode → 返回文本。

## 为什么

- **预算测算**：客户端在发真实请求前算 prompt token 数，估算费用/截断点；`get_max_tokens` 也依赖精确 prompt 长度。
- **多模态可见**：tokenize 返回多模态占位符 token 位置，调试图像/音频 token 占用。
- **与 chat template 一致**：复用同一 `OnlineRenderer.preprocess_chat/completion`，保证 tokenize 结果与真实推理一致。
- **render-only 可用**：render-only server 也挂此端点（`init_render_app_state` 构造 `ServingTokenization`），便于独立验证渲染。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/tokenize \
  -d '{"model":"...","prompt":"hello"}'
curl -X POST http://localhost:8000/v1/detokenize \
  -d '{"model":"...","token_ids":[1,2,3]}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingTokenization | `vllm/entrypoints/serve/tokenize/serving.py:32` |
| TokenizerInfo | `vllm/entrypoints/serve/tokenize/serving.py:164` |
| tokenize 路由 | `vllm/entrypoints/serve/tokenize/api_router.py:47` |
| detokenize 路由 | `vllm/entrypoints/serve/tokenize/api_router.py:72` |
| attach_router | `vllm/entrypoints/serve/tokenize/api_router.py:94` |

## 与其它模块/系统配合

- [engine-serve.md](engine-serve.md)：继承 `BaseServing`。
- [openai/api-server.md](../openai/api-server.md)：`init_app_state`/`init_render_app_state` 构造。
- [chat-utils.md](../chat-utils.md)：renderer.preprocess_chat 内部。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：tokenizer。

## 历史版本演进

- **v0.9（引入）**：`/v1/tokenize`、`/v1/detokenize` 实验端点。
- **v0.10（render-only）**：render server 也提供。
- **main**：`TokenizerInfo` 辅助类；多模态占位符位置返回（待核实稳定字段）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [openai/api-server.md](../openai/api-server.md)
- [chat-utils.md](../chat-utils.md)
