# 池化参数（PoolingParams）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 池化参数

本页覆盖 `vllm/pooling_params.py`（235 行），定义池化类任务（embed/classify/score/late-interaction 等）的公共入参类型 `PoolingParams` 与 `LateInteractionParams`。

## 是什么

### `PoolingParams`（`vllm/pooling_params.py:37`）

继承 `msgspec.Struct`（`omit_defaults=True`、`array_like=True`）。字段分两组：

- **公开 API**：
  - `use_activation: bool | None`：是否对 pooler 输出施加激活（None 走 pooler 默认，多数为 True）。
  - `dimensions: int | None`：matryoshka 表征的降维数（仅 embed/token_embed）。
- **内部字段（Internal use only）**：
  - `task: PoolingTask | None`：由引擎依据模型/请求设定（embed/classify/token_embed/token_classify/plugin/...，见 [tasks.md](tasks.md)）。
  - `requires_token_ids: bool`、`skip_reading_prefix_cache: bool | None`：token 级任务/前缀缓存读取控制。
  - `step_tag_id: int | None`、`returned_token_ids: list[int] | None`：STEP 池化专用。
  - `late_interaction_params: LateInteractionParams | None`：late-interaction（ColBERT 风格）评分元数据。
  - `extra_kwargs: dict[str, Any] | None`：插件扩展位。
  - `output_kind: RequestOutputKind`：强制 `FINAL_ONLY`（`__post_init__` 校验，`:230`）。

核心方法：

- `verify(model_config)`（`:89`）：按 `task` 分支——`plugin` 交由 io_processor；其余走 `_merge_default_parameters`/`_set_default_parameters`/`_verify_valid_parameters`/`_verify_step_pooling`，校验 matryoshka 维度合法性、STEP 参数仅 `tok_pooling_type==STEP` 可用、token 级任务强制 `skip_reading_prefix_cache=True`。
- `all_parameters`/`valid_parameters`（`:72`/`:76`）：声明各 task 允许的字段白名单，`_verify_valid_parameters` 据此把"该 task 不支持的字段"判为非法。
- `clone()`（`:85`）：`deepcopy`。

### `LateInteractionParams`（`vllm/pooling_params.py:17`）

`mode`（`cache_query` 缓存 query token embeds / `score_doc` 用缓存 query 给 doc 评分）、`query_key`（DP 路由 + worker 缓存查键）、`query_uses: int | None`（预期 doc 请求数，用尽后释放缓存）。

## 为什么

- **任务态字段裁剪**：同一 `PoolingParams` 类服务多种 task，但每 task 只允许部分字段；以 `valid_parameters` 白名单 + `verify()` 显式拒绝非法组合，避免用户误设无意义参数。
- **pooler_config 合并**：模型 `PoolerConfig` 提供默认（如 `step_tag_id`），`verify` 把"用户未显式给"与"pooler 默认"合并。
- **前缀缓存策略与 task 相关**：token 级任务输出长度 < n_prompt，读取前缀缓存会出错，故按 task 自动决定 `skip_reading_prefix_cache`。
- **late-interaction 跨请求**：query/doc 是两个请求，需 DP 路由 + worker 端缓存 query embedding，`LateInteractionParams` 承载该协议。

## 怎么做

```python
from vllm.pooling_params import PoolingParams
pp = PoolingParams(dimensions=768)
pp.task = "embed"
pp.verify(model_config)        # 引擎内部调用
```

- API 层一般在请求装配时设 `task`，或在 worker 按 pooler_config 推断。
- late-interaction：query 请求带 `LateInteractionParams(mode="cache_query", query_key=...)`，doc 请求带 `mode="score_doc"` 与同 key。

## 与其它模块/系统配合

- [tasks.md](tasks.md)：`PoolingTask` 定义合法 task 字面量。
- [配置体系](../10-config/README.md)：`PoolerConfig`（`pooler-config.md`）提供默认值；`ModelConfig.is_matryoshka`/`matryoshka_dimensions`/`embedding_size` 参与 dimensions 校验。
- [引擎核心](../01-engine-core/README.md)：`EngineCoreRequest` 携带 `PoolingParams`；InputProcessor 阶段调用 `verify`。
- [pooling 入口/API](../13-entrypoints/README.md)：API 层把 embed/classify/score 请求转成 `PoolingParams`。
- [outputs.md](outputs.md)：`PoolingRequestOutput`/`EmbeddingRequestOutput` 等。
- [sampling-params.md](sampling-params.md)：复用 `RequestOutputKind`。

## 历史版本演进

- **v0.5–v0.6**：`PoolingParams` 较简单，主要 `dimensions`/`use_activation`；无 task 字段。
- **v0.7–v0.8**：引入 `task` 字段与 `valid_parameters` 白名单机制；强制 `output_kind=FINAL_ONLY`。
- **v0.9–v0.10**：STEP 池化字段（`step_tag_id`/`returned_token_ids`）加入；前缀缓存读取策略按 task 自动化。
- **v0.11–main**：`LateInteractionParams` 上线支撑 late-interaction 评分；`plugin` task 与 io_processor 集成；`extra_kwargs` 作为插件扩展位（具体版本待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [sampling-params.md](sampling-params.md)
- [outputs.md](outputs.md)
- [tasks.md](tasks.md)
