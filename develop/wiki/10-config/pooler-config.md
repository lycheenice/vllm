# PoolerConfig（pooler.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > PoolerConfig

源码：`vllm/config/pooler.py`（约 172 行）。`PoolerConfig` 描述 pooling 模型（embedding/classification/reward/score）的输出聚合行为：pooling 类型、激活、维度缩减、chunked 处理、分类 affine 校准、reward step tag 等。它是 `ModelConfig.pooler_config`（内嵌于 `model_config`，**非** `VllmConfig` 顶层字段），被 pooler 层（`vllm/model_executor/layers/pooler.py`）与 pooling model runner 消费。

## 是什么

`@config` 装饰（`pooler.py:20`）。`SequencePoolingType = Literal["CLS","LAST","MEAN"]`、`TokenPoolingType = Literal["ALL","STEP"]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `task` | `None` | `PoolingTask`（embed/classify/reward/score 等） |
| `pooling_type` | `None` | 便捷字段：设则自动派生 `seq_pooling_type`/`tok_pooling_type` |
| `seq_pooling_type` | `None` | 序列 pooling：`CLS`/`LAST`/`MEAN` |
| `tok_pooling_type` | `None` | tokenwise pooling：`ALL`/`STEP` |
| `use_activation` | `None` | 是否对 pooler 输出加激活（`None`=pooler 默认，多数 True） |
| `dimensions` | `None` | matryoshka 表示降维 |
| `enable_chunked_processing` | `False` | 长输入（超 max position embeddings）分块处理 + 加权平均 |
| `max_embed_len` | `None` | embedding 最大输入长度（`None`=max_model_len） |
| `logit_mean` | `None` | 分类 affine 校准：`activation((logit-logit_mean)/logit_sigma)`（Platt scaling） |
| `logit_sigma` | `None` | affine 校准分母 |
| `step_tag_id` | `None` | reward 模型：仅返回 `step_tag_id` 对应 token 的 score |
| `returned_token_ids` | `list[int] | None` | reward：提取的 vocab 维度索引（如 `good_token`/`bad_token`） |

校验（`__post_init__`）：`logit_sigma != 0`；`pooling_type` 与 `seq_pooling_type`/`tok_pooling_type` 不可同设；`pooling_type` 按 `SEQ_POOLING_TYPES`/`TOK_POOLING_TYPES` 派生对应字段，否则 `NotImplementedError`。

方法：`get_seq_pooling_type()`/`get_tok_pooling_type()`（未设则 raise，提示应由 `ModelConfig` 解析）。

`compute_hash`：空 factors——pooler 在输出层做事（取 CLS/LAST/MEAN/激活），不改前向图形状。

> 内嵌于 `ModelConfig`，故 `VllmConfig.compute_hash` 中经 `model_config.compute_hash()` 间接调用（但 `compute_hash` 返回空，故实际无贡献）。

## 为什么

- **多任务统一**：pooling 模型涵盖 embed/classify/reward/score，各需不同聚合与后处理。`PoolerConfig` 统一描述，`task`/`pooling_type`/`seq_pooling_type`/`tok_pooling_type`/`use_activation` 让一个 pooler 层适配多任务。
- **`pooling_type` 便捷字段**：用户设 `pooling_type="CLS"` 自动派生 `seq_pooling_type`，内部代码只用 `seq/tok_pooling_type`（明确），二者不可同设防冲突。
- **分类 affine 校准**：`logit_mean`/`logit_sigma` 实现 Platt scaling，把 raw logit 校准为概率，提升分类模型可用性。
- **reward 专用**：`step_tag_id` 让 reward 模型仅返回特定 token 的 score（如步骤末）；`returned_token_ids` 提取特定 vocab 维度（如 `math-shepherd-mistral-7b-prm` 的 good/bad token）。
- **chunked 处理**：`enable_chunked_processing` 让 embedding 模型处理超长输入（超 max position embeddings），分块后加权平均，避免 CUDA error。
- **matryoshka 降维**：`dimensions` 支持 matryoshka representation learning 模型的可变维度 embedding。
- **`compute_hash` 空**：聚合/激活在输出层，不改前向图，故不进哈希（与 reasoning/structured_outputs 等输出层配置一致）。

## 怎么做

- **embedding**：`--pooling-type MEAN`（或 `CLS`/`LAST`）；matryoshka `--dimensions 768`。
- **classification**：`--pooling-type LAST --classify` + affine `--pooler-config.logit-mean 0.5 --pooler-config.logit-sigma 2.0`。
- **reward**：`--pooling-type ALL --step-tag-id 12345`。
- **chunked**：`--pooler-config.enable-chunked-processing --pooler-config.max-embed-len 8192`。

## 与其它模块/系统配合

- **Pooler 层（[`03-model-execution/layers/pooler.md`](../03-model-execution/layers/pooler.md)）**：`seq/tok_pooling_type`/`use_activation`/`dimensions`/`logit_*`/`step_tag_id`/`returned_token_ids` 驱动 `Pooler` 层实现选择。
- **ModelConfig（[model-config.md](model-config.md)）**：`pooler_config` 内嵌；`runner_type="pooling"` 触发 pooling model runner；`convert="embed"/"classify"` adapter 协同。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`runner_type=="pooling"` 默认关 `async_scheduling`（pooling 的 async 实现负优化）；`cudagraph_mode` 在 pooling 模型被强制 PIECEWISE（`__post_init`）。
- **SchedulerConfig（[scheduler-config.md](scheduler-config.md)）**：`runner_type="pooling"` 影响调度器行为。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：pooling 模型不支持 full cudagraph → PIECEWISE。

## 历史版本演进

- **v0.5/v0.6（v0）**：pooling 字段散在 `ModelConfig`；`pooling_type`/`max_embed_len`。
- **v0.7（v1 落地）**：`PoolerConfig` 独立子配置（内嵌 `model_config`）；`seq_pooling_type`/`tok_pooling_type` 分离；`task` 字段。
- **v0.8**：`logit_mean`/`logit_sigma`（Platt scaling 分类校准）；`step_tag_id`/`returned_token_ids`（reward 专用）；`enable_chunked_processing`/`max_embed_len`。
- **v0.9**：`dimensions`（matryoshka）；`pooling_type` 便捷字段派生逻辑成形。
- **v0.10–main**：`get_seq/tok_pooling_type` 未设 raise 提示；pooling 模型强制 PIECEWISE cudagraph；MRv2 pooling（`v1/worker/gpu/pool/`）。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — 内嵌于 `ModelConfig`；`runner_type`/`convert` 协同。
- [vllm-config.md](vllm-config.md) — pooling 关 async + 强制 PIECEWISE cudagraph。
- [scheduler-config.md](scheduler-config.md) — `runner_type="pooling"`。
- [../03-model-execution/layers/pooler.md](../03-model-execution/layers/pooler.md) — Pooler 层消费方。
