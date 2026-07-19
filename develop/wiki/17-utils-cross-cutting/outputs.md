# 对外输出类型（outputs）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 对外输出类型

本页覆盖 `vllm/outputs.py`（353 行），定义 API 用户面对的请求级输出类型——与引擎内部的 `vllm/v1/outputs.py`（`ModelRunnerOutput`，见 [引擎核心·数据模型](../01-engine-core/data-model.md)）分层。

## 是什么

```
CompletionOutput                 # 单条生成输出
RequestOutput                    # 生成请求输出（含 outputs: list[CompletionOutput]）
PoolingOutput                    # 池化单结果（torch.Tensor）
PoolingRequestOutput[_O]         # 池化请求输出（泛型基类）
├── EmbeddingRequestOutput       # embed 任务
├── ClassificationRequestOutput  # classify 任务
└── ScoringRequestOutput         # score（bi/cross/late）任务
EmbeddingOutput / ClassificationOutput / ScoringOutput   # 各自的"单结果"形态
STREAM_FINISHED                  # 流式输入终结哨兵 RequestOutput
```

### `CompletionOutput`（`vllm/outputs.py:22`，dataclass）

单条 completion：`index`、`text`、`token_ids`、`cumulative_logprob`、`logprobs`（`SampleLogprobs`）、`routed_experts: np.ndarray | None`（`[seq_len,layer_num,topk]`）、`finish_reason`、`stop_reason`、`lora_request`。`finished()` 判 `finish_reason is not None`。

### `RequestOutput`（`vllm/outputs.py:85`）

生成请求对用户的总输出：`request_id`、`prompt`、`prompt_token_ids`、`prompt_logprobs`、`outputs: list[CompletionOutput]`、`finished`、`metrics: RequestStateStats | None`、`lora_request`、`encoder_prompt`/`encoder_prompt_token_ids`（encoder/decoder 模型）、`num_cached_tokens`、`kv_transfer_params`。关键方法：

- `add(next_output, aggregate)`（`:145`）：把后续 `RequestOutput` 合并进来——按 `index` 配对，`aggregate=True` 时累加 text/token_ids/logprobs 并覆盖 finish_reason；用于 streaming 续传聚合。
- `**kwargs`（`:126`）：前向兼容，新字段在不支持的老版本上仅 `warning_once` 忽略。
- `STREAM_FINISHED`（`:192`）：固定哨兵实例，标记流式输入请求终结。

### `PoolingOutput`（`:67`）

`data: torch.Tensor`；`__eq__` 逐元素比较。

### `PoolingRequestOutput[_O]`（`:204`，`Generic[_O]`）

泛型基类：`request_id`、`outputs: _O`、`prompt_token_ids`、`num_cached_tokens`、`finished`。三个子类各自带 `from_base(request_output: PoolingRequestOutput)` 静态方法，从 `PoolingOutput(torch.Tensor)` 转喻为强类型：

- `EmbeddingOutput`（`:241`）/`EmbeddingRequestOutput`（`:267`）：Tensor 须为 1-D（embedding 向量），`embedding: list[float]`，`hidden_size` 属性。
- `ClassificationOutput`（`:280`）/`ClassificationRequestOutput`（`:307`）：1-D 概率向量，`probs: list[float]`，`num_classes` 属性。
- `ScoringOutput`（`:320`）/`ScoringRequestOutput`（`:344`）：squeeze 后须为 0-D 标量，`score: float`。`from_base` 注释说明 classify 单类与 embed 标量两种来源均映射到此。

## 为什么

- **API 形态与引擎形态解耦**：引擎内部用 `ModelRunnerOutput`（张量/list）跨进程高效传输；`outputs.py` 是"给用户看的"强类型形态，由 OutputProcessor 装配。分层让两侧各自优化。
- **泛型复用**：`PoolingRequestOutput[_O]` 一次实现 request 外壳，三种任务只差"单结果"形态，靠泛型 + `from_base` 避免重复。
- **前向兼容的 `**kwargs`**：新版本加字段不会让旧 vLLM 立即崩，便于滚动升级。
- **streaming 聚合**：`add(aggregate=)` 让 streaming-input / parallel sampling 能把多段输出缝合成单条 `RequestOutput`。

## 怎么做

```python
# 引擎侧装配（简化）
ro = RequestOutput(request_id, prompt, prompt_token_ids, None,
                   [CompletionOutput(0, text, ids, None, None)], finished=True)
# pooler 路径
base = PoolingRequestOutput(req_id, PoolingOutput(t), ids, n, True)
emb = EmbeddingRequestOutput.from_base(base)
```

- API server 据 task 类型选 `RequestOutput` 或 `EmbeddingRequestOutput`/`ClassificationRequestOutput`/`ScoringRequestOutput` 返回。
- 流式：每步产出 `RequestOutput(finished=False)`，末步 `finished=True`；`add` 累积。

## 与其它模块/系统配合

- [引擎核心 · output_processor](../01-engine-core/output-processor.md)：装配 `RequestOutput`，调 `add` 聚合 streaming。
- [采样与解码](../06-sampling-decoding/README.md)：`CompletionOutput.logprobs` 形态由采样器产出。
- [API 入口](../13-entrypoints/README.md)：HTTP 响应由这些类型序列化；OpenAI embeddings → `EmbeddingRequestOutput`，classify → `ClassificationRequestOutput`，score → `ScoringRequestOutput`。
- [LoRA](../12-lora/README.md)：`CompletionOutput.lora_request` 标记来源 adapter。
- [可观测性](../16-observability/README.md)：`metrics: RequestStateStats` 透出 TTFT/排队等指标（见 [输出处理器](../01-engine-core/output-processor.md)）。
- [pooling-params.md](pooling-params.md) / [sampling-params.md](sampling-params.md)：入参侧对应物。

## 历史版本演进

- **v0.5–v0.6**：`RequestOutput`/`CompletionOutput` 已存在；pooling 仅 `PoolingOutput`，embeddings 接口较粗。
- **v0.7–v0.8**：v1 重构后 `RequestOutput` 引入 `add(aggregate=)`、`**kwargs` 前向兼容；`STREAM_FINISHED` 哨兵配合 streaming-input。
- **v0.9–v0.10**：pooling 输出族泛型化（`PoolingRequestOutput[_O]`），新增 classify/score 专用子类；`routed_experts` 字段加入（MoE 路由专家透出）。
- **v0.11–main**：`metrics` 改用 `vllm.v1.metrics.stats.RequestStateStats`；`kv_transfer_params` 配合 KV 迁移（NIXL/Mooncake）；`from_base` 的 squeeze 规则细化（待核实具体版本）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [引擎核心 · 数据模型](../01-engine-core/data-model.md)（v1 内部 `ModelRunnerOutput`）
- [sampling-params.md](sampling-params.md)、[pooling-params.md](pooling-params.md)
- [API 入口](../13-entrypoints/README.md)
