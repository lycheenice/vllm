[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Pooling](README.md) > scoring

# pooling/scoring/（/v1/score & /v1/rerank）

> `scoring/` 实现相关性打分与重排：`/v1/score` 接受 query-document 对返回相关性分数；`/v1/rerank` 接受 query + 候选文档列表返回按相关性排序的结果。兼容 Cohere/Jina 风格。离线等价 `LLM.score`。

## 是什么

| 文件/类 | 位置 | 职责 |
|---|---|---|
| `serving.py: ServingScores` | `vllm/entrypoints/pooling/scoring/serving.py:35` | `/v1/score`、`/v1/rerank` handler，继承 `PoolingServing` |
| `api_router.py: attach_router` | — | 注册两个端点 |
| `protocol.py: ScoreRequest`/`RerankRequest`/`ScoreResponse`/`RerankResponse` | `vllm/entrypoints/pooling/scoring/protocol.py` | schema（`run_batch.py:49` import） |
| `io_processor.py` | — | pair 构造、分数组装 |
| `typing.py`/`utils.py` | — | pair 类型与排序工具 |

`ServingScores`（`:35`）继承 `PoolingServing`：

- `/v1/score`：`ScoreRequest{text, model, query, documents}` → io_processor 把 `(query, doc)` 拼成 pair → `engine_client.encode`/`pool`（invocation=`score`）→ 收割分数 → `ScoreResponse{data[].score}`。
- `/v1/rerank`：`RerankRequest{query, documents, top_n}` → 同样打分 → 按 score 排序 → `RerankResponse{results[].index, relevance_score}`。

## 为什么

- **pair 打分复用 pool**：cross-encoder/reranker 模型本质是 pooling 模型对 pair 输入出标量，`scoring` 复用 `PoolingBaseServing` 引擎链，仅 io_processor 拼 pair。
- **双端点兼容**：`/v1/score`（原始分数）与 `/v1/rerank`（排序+top_n）覆盖主流 rerank SDK。
- **排序下沉**：`utils.py` 在 server 侧排序，减少客户端负担。
- **离线一致**：`LLM.score`（`offline.py:289`）同 io_processor。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/score \
  -d '{"model":"...","query":"q","documents":["d1","d2"]}'
curl -X POST http://localhost:8000/v1/rerank \
  -d '{"model":"...","query":"q","documents":["d1","d2"],"top_n":2}'
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| ServingScores | `vllm/entrypoints/pooling/scoring/serving.py:35` |
| attach_router | `vllm/entrypoints/pooling/scoring/api_router.py` |
| ScoreRequest/RerankRequest | `vllm/entrypoints/pooling/scoring/protocol.py` |
| io_processor | `vllm/entrypoints/pooling/scoring/io_processor.py`（待核实行号） |
| 离线 score | `vllm/entrypoints/pooling/offline.py:289` |

## 与其它模块/系统配合

- [base.md](base.md)：基类。
- [factories.md](factories.md)：注册。
- [openai/run-batch.md](../openai/run-batch.md)：批 score/rerank。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：reranker 渲染。

## 历史版本演进

- **v0.8（score/rerank 引入）**：`/v1/score`/`/v1/rerank` + `LLM.score`。
- **v0.9（io_processor）**：scoring io_processor 抽象。
- **v0.11/main**：Cohere/Jina 兼容字段；pair 长度上限校验（待核实）。

## 参见

- [← 返回 Pooling 首页](README.md)
- [base.md](base.md)
- [openai/run-batch.md](../openai/run-batch.md)
