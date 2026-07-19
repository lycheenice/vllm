[← Wiki 首页](../README.md) > [API 入口](../README.md) > Pooling

# 池化任务（pooling/）

> `vllm/entrypoints/pooling/` 实现非生成类"池化"任务：embeddings、classification、score/rerank、generic pooling。按任务类型拆子包（`pooling/`/`embed/`/`classify/`/`scoring/`），各含 `serving.py`/`api_router.py`/`protocol.py`/`io_processor.py`，由 `factories.py` 统一注册与状态初始化。离线 `LLM` 池化方法由 `offline.py` 的 `PoolingOfflineMixin` 提供。

## 是什么

| 文件/类 | 位置 | 职责 |
|---|---|---|
| `factories.py: init_pooling_io_processors` | `vllm/entrypoints/pooling/factories.py:38` | 实例化各 task 的 io_processor |
| `factories.py: register_pooling_api_routers` | `:104` | 按 supported_tasks 挂路由 |
| `factories.py: init_pooling_state` | `:137` | 把 serving 对象挂到 `app.state` |
| `factories.py: get_pooling_invocation_types` | `:214` | 任务→invocation 类型映射 |
| `base/serving.py: PoolingBaseServing` | `vllm/entrypoints/pooling/base/serving.py:37` | 池化基类（`BaseServing`+ABC） |
| `base/serving.py: PoolingServing` | `:263` | 处理 `AnyPoolingRequest` 的具体基类 |
| `pooling/serving.py: ServingPooling` | `vllm/entrypoints/pooling/pooling/serving.py:33` | `/v1/pooling` 通用端点 |
| `scoring/serving.py: ServingScores` | `vllm/entrypoints/pooling/scoring/serving.py:35` | `/v1/score`、`/v1/rerank` |
| `embed/` | — | `/v1/embeddings` |
| `classify/` | — | `/v1/classify` |
| `offline.py: PoolingOfflineMixin` | `vllm/entrypoints/pooling/offline.py:31` | `LLM.encode`/`embed`/`classify`/`score` |
| `typing.py`/`utils.py` | — | 共享类型与工具 |

pooling 任务模型 `runner_type="pooling"`，`engine_client.get_supported_tasks()` 返回 `POOLING_TASKS`；`build_app` 检测后调 `register_pooling_api_routers`（`api_server.py:228`）。`get_pooling_invocation_types`（`:214`）把请求映射到 `PoolingParams`/`embed`/`classify`/`score` 等引擎侧 invocation 类型。

`PoolingBaseServing`（`:37`）抽象池化请求→`PoolingParams`→引擎推理→`PoolingRequestOutput` 的共性；`PoolingServing`（`:263`）处理通用 `AnyPoolingRequest`，具体子类（`ServingPooling`/`ServingScores`/embed/classify）补 task 专属协议字段。

## 为什么

- **任务族共享基建**：embed/classify/score 都是"输入 → 池化层 → 向量/分数"，`PoolingBaseServing` 把请求校验、io_processor 调用、输出组装统一，子类只声明协议差异。
- **io_processor 模式**：每个 task 有独立 `io_processor.py`，封装"请求 → 引擎输入"与"引擎输出 → 响应"，便于多模态 pooling（如带图 embed）。
- **score/rerank 双语义**：`ServingScores`（`:35`）同时支持 `/v1/score`（pairwise 打分）与 `/v1/rerank`（重排），Cohere/Jina 风格兼容。
- **离线/在线同源**：`PoolingOfflineMixin` 与 serving 共用 `OnlineRenderer.preprocess_*` 与 io_processor，保证 `llm.embed` 与 HTTP `/v1/embeddings` 结果一致。
- **`/v1/pooling` 通用端点**：`ServingPooling` 提供不带 task 语义的原始池化输出，供高级用户直接取池化向量。

## 怎么做

### 启用

模型 `runner="pooling"`（或 auto 识别 pooling 模型），`build_app` 检测 pooling task 后挂 `embed`/`classify`/`score`/`rerank`/`pooling` 路由。

### 离线

```python
llm = LLM(model="BAAI/bge-m", runner="pooling")
emb = llm.embed(["文本"])           # /v1/embeddings 等价
score = llm.score([("q","d1"),("q","d2")])  # /v1/score 等价
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| factories 注册 | `vllm/entrypoints/pooling/factories.py:104` |
| factories state | `vllm/entrypoints/pooling/factories.py:137` |
| io_processor 实例化 | `vllm/entrypoints/pooling/factories.py:38` |
| invocation 类型 | `vllm/entrypoints/pooling/factories.py:214` |
| PoolingBaseServing | `vllm/entrypoints/pooling/base/serving.py:37` |
| PoolingServing | `vllm/entrypoints/pooling/base/serving.py:263` |
| ServingPooling | `vllm/entrypoints/pooling/pooling/serving.py:33` |
| ServingScores | `vllm/entrypoints/pooling/scoring/serving.py:35` |
| PoolingOfflineMixin | `vllm/entrypoints/pooling/offline.py:31` |
| encode (离线) | `vllm/entrypoints/pooling/offline.py:51` |
| embed/classify/score (离线) | `vllm/entrypoints/pooling/offline.py:199/244/289` |

## 与其它模块/系统配合

- [serve/engine-serve.md](../serve/engine-serve.md)：`PoolingBaseServing` → `BaseServing`。
- [openai/api-server.md](../openai/api-server.md)：`build_app` 按 pooling tasks 挂路由。
- [openai/run-batch.md](../openai/run-batch.md)：批处理 embed/score。
- [llm.md](../llm.md)：`PoolingOfflineMixin` 注入 `LLM`。
- [多模态](../../11-multimodal/README.md)：io_processor 处理多模态 pooling 输入。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：pooler 渲染。

## 历史版本演进

- **v0.7（embeddings）**：`/v1/embeddings` 与 `LLM.embed`，初版 pooling 支持。
- **v0.8（score/rerank）**：`/v1/score`/`/v1/rerank` + `LLM.score`。
- **v0.9（pooling 子包重构）**：按 task 拆 `pooling/`/`embed/`/`classify/`/`scoring/`；引入 io_processor；`PoolingBaseServing`/`PoolingServing` 抽象。
- **v0.10（classify + offline 整合）**：`/v1/classify` + `LLM.classify`；`PoolingOfflineMixin` 统一 encode/embed/classify/score。
- **v0.11/main**：`/v1/pooling` 通用端点；`get_pooling_invocation_types` 任务映射；多模态 pooling（图像/音频 embed）。

## 模块导航

| 页 | 主题 |
|---|---|
| [factories.md](factories.md) | 注册与 state 初始化 |
| [base.md](base.md) | `PoolingBaseServing`/`PoolingServing` |
| [embed.md](embed.md) | `/v1/embeddings` |
| [classify.md](classify.md) | `/v1/classify` |
| [pooling-serves.md](pooling-serves.md) | `/v1/pooling` 通用 |
| [scoring.md](scoring.md) | `/v1/score`/`/v1/rerank` |

## 参见

- [← 返回 API 入口首页](../README.md)
- [serve/engine-serve.md](../serve/engine-serve.md)
- [openai/api-server.md](../openai/api-server.md)
- [多模态](../../11-multimodal/README.md)
