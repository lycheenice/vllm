# 任务类型枚举（tasks）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 任务类型

本页覆盖 `vllm/tasks.py`（28 行），定义 vLLM 全局任务字面量类型。

## 是什么

`vllm/tasks.py` 用 `Literal` + `get_args` 定义任务字符串集合：

- `GenerationTask = Literal["generate", "transcription", "realtime"]`（`vllm/tasks.py:5`）；`GENERATION_TASKS` 为其元组形式。
- `PoolingTask = Literal["embed", "classify", "token_embed", "token_classify", "plugin", "embed&token_classify"]`（`:8`）；`POOLING_TASKS` 元组。
- `ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]`（`:18`）；`SCORE_TYPE_MAP` 把部分 pooling task 映射到 score type：`embed→bi-encoder`、`classify→cross-encoder`、`token_embed→late-interaction`。
- `FrontendTask = Literal["render"]`（`:25`）；`FRONTEND_TASKS` 元组。
- `SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]`（`:27`）：三者合集，作为模型支持任务的顶层类型。

## 为什么

- **类型安全的任务标识**：用 `Literal` 而非自由 str，让 mypy/IDE 在请求装配、模型注册、pooler 选用等处做穷尽性检查。
- **统一生成/池化/前端三类**：vLLM 不只做生成（embed/classify/score/`render` 都是合法任务），需要一处枚举所有可能性。
- **ScoreType 映射**：score 任务（相似度打分）是 pooling 的一种用法，但需要区分 bi-encoder（双塔，query/doc 各自 embed 后算点积）与 cross-encoder（单塔同过）、late-interaction（ColBERT 风格 token-level）；`SCORE_TYPE_MAP` 固化该映射。
- **`plugin` / `embed&token_classify`**：为 io_processor 插件与多任务模型留扩展位。

## 怎么做

```python
from vllm.tasks import PoolingTask, SCORE_TYPE_MAP
task: PoolingTask = "embed"
score_kind = SCORE_TYPE_MAP[task]   # "bi-encoder"
```

- 模型注册时声明 `supported_tasks`（生成模型用 `GENERATION_TASKS` 子集，pooling 模型用 `POOLING_TASKS` 子集）。
- 请求路径由 `ModelConfig`/pooler 决定实际 `task` 字符串，填入 `PoolingParams.task`（见 [pooling-params.md](pooling-params.md)）。

## 与其它模块/系统配合

- [模型库](../04-model-zoo/README.md)：模型注册表的 `support_*` 元数据基于这些字面量。
- [pooling-params.md](pooling-params.md)：`PoolingParams.task: PoolingTask`，`verify` 按 task 分支。
- [配置体系](../10-config/README.md)：`ModelConfig.tasks`/`supported_tasks` 校验。
- [API 入口](../13-entrypoints/README.md)：OpenAI embeddings→`embed`、classify→`classify`、score→按 `ScoreType` 路由；`render` 走 scale-out 前端（见 `13-entrypoints`）。
- [outputs.md](outputs.md)：`EmbeddingRequestOutput`/`ClassificationRequestOutput`/`ScoringRequestOutput` 对应 pooling task。
- [plugins.md](plugins.md)：`plugin` task 走 io_processor。

## 历史版本演进

- **v0.5–v0.6**：任务概念分散在 `ModelType`/`PoolingType` 等处，无统一 Literal。
- **v0.7–v0.8**：`vllm/tasks.py` 引入，集中 `GenerationTask`/`PoolingTask`/`SupportedTask`。
- **v0.9–v0.10**：加入 `ScoreType` 与 `SCORE_TYPE_MAP`，支撑 cross-encoder/late-interaction；`token_embed`/`token_classify` 随 token 级池化上线。
- **v0.11–main**：`FrontendTask = render` 随 scale-out render 入口加入；`plugin`/`embed&token_classify` 因 io_processor 与多任务模型引入（具体版本待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [pooling-params.md](pooling-params.md)
- [outputs.md](outputs.md)
- [plugins.md](plugins.md)
