[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > embed

# pooling/embed/（/v1/embeddings）

> `embed/` 实现 OpenAI 兼容 `/v1/embeddings`：把文本/多模态输入经 `ServingEmbedding` 渲染、池化，返回 `EmbeddingResponse`（向量 + usage）。离线等价 `LLM.embed`。

## 是什么

| 文件 | 职责 |
|---|---|
| `serving.py: ServingEmbedding` | `/v1/embeddings` handler，继承 `PoolingServing` |
| `api_router.py: attach_router` | 注册 `POST /v1/embeddings` |
| `protocol.py: EmbeddingRequest`/`EmbeddingResponse` | OpenAI 风格 embed schema |
| `io_processor.py` | 请求→`EngineInput`、`PoolingRequestOutput`→`EmbeddingResponse` |

请求经 `_check_model` → io_processor 渲染 → `engine_client.encode`/`pool`（`get_pooling_invocation_types` 返回 `embed`）→ 收割 `PoolingRequestOutput` → io_processor 组装 `EmbeddingResponse`（含 `data[].embedding`、`usage`）。

## 为什么

- **OpenAI drop-in**：`/v1/embeddings` 是最常见 embed 端点，保持 schema 一致让客户端无改迁移。
- **多模态 embed**：io_processor 支持图像/音频输入（如 CLIP-style），用 [多模态](../../11-multimodal/README.md) processor。
- **与离线一致**：`LLM.embed` 经同一 io_processor，结果可复现。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -d '{"model":"...","input":"text"}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingEmbedding | `vllm/entrypoints/pooling/embed/serving.py`（待核实行号） |
| attach_router | `vllm/entrypoints/pooling/embed/api_router.py` |
| EmbeddingRequest | `vllm/entrypoints/pooling/embed/protocol.py`（`run_batch.py:45` import） |
| 离线 embed | `vllm/entrypoints/pooling/offline.py:199` |

## 与其它模块/系统配合

- [base.md](base.md)：基类。
- [factories.md](factories.md)：注册。
- [openai/run-batch.md](../openai/run-batch.md)：批 embed。
- [多模态](../../11-multimodal/README.md)：MM embed。

## 历史版本演进

- **v0.7（引入）**：`/v1/embeddings` + `LLM.embed`。
- **v0.9（io_processor）**：io_processor 抽象。
- **main**：多模态 embed；usage 字段（含多模态 token）。

## 参见

- [← 返回 Pooling 首页](README.md)
- [base.md](base.md)
