# 采样层（Sampler）与采样元数据

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 采样层

> 物理位置提示：v1 引擎把 `Sampler` 与 `SamplingMetadata` 从旧 `vllm/model_executor/layers/sampling.py`（已废弃）迁到 `vllm/v1/sample/sampler.py` 与 `vllm/v1/sample/metadata.py`。本文档统一以 `vllm/v1/sample/...` 为路径引用，但概念上仍属于层库范畴——它直接消费 `vllm/model_executor/layers/logits_processor.py` 的 `LogitsProcessor` 与 `ParallelLMHead`。

## 是什么

| 类/函数 | 文件:行 | 角色 |
|---|---|---|
| `Sampler` | `vllm/v1/sample/sampler.py:20` | `nn.Module`，模型 forward 末端"将 logits → 采样 token"层 |
| `SamplingMetadata` | `vllm/v1/sample/metadata.py:15` | dataclass，汇总每个 request 的温度/top-k/top-p/penalty/allowed/bad words/logits processor 等 |
| `TopKTopPSampler` | `vllm/v1/sample/ops/topk_topp_sampler.py` | 由 `Sampler` 持有的子模块，封装 top-k/top-p + Gumbel 采样 |
| `LogitsProcessor`（层库版） | `vllm/model_executor/layers/logits_processor.py:19` | 把 `hidden_states` 经 `lm_head` 投影成 logits 并按需 gather/裁剪/缩放 |
| `LogitsProcessors`（v1 插件版） | `vllm/v1/sample/logits_processor/__init__.py` | 一组按 argmax-invariance 分类、可热加载的 logit processor |
| `ThinkingBudgetStateHolder` | `vllm/v1/sample/thinking_budget_state.py:33` | "思考预算"状态机，对应 `</think>` 边界约束 |

## 为什么

把 sampler 抽成独立"层"是为了：

1. **解耦模型 forward 与采样**：模型实现只输出 hidden_states，最后一步 `logits = self.logits_processor(self.lm_head, hidden_states); sampled = self.sampler(logits, sampling_metadata)` 即可。这方便 cudagraph 捕获——sampler 内部的所有张量操作都是 batch-wise、shape 已知。
2. **支持 v1 引擎的 argmax/non-argmax invariant 分类**：v1 把 logit processor 分成两拨——`non_argmax_invariant`（如 `MinTokensLogitsProcessor`、`LogitBiasLogitsProcessor`，会影响 greedy）在所有采样前应用；`argmax_invariant`（如 `MinPLogitsProcessor`）只在 random 采样路径上应用。`Sampler.apply_logits_processors`（`sampler.py:371`）按顺序执行。
3. **支持 logits processor 热加载与 plugin**：`LogitsProcessors` 通过 `_load_logitsprocs_plugins / _load_logitsprocs_by_fqcns / _load_custom_logitsprocs`（`vllm/v1/sample/logits_processor/__init__.py:56/86/158`）按 `SamplingParams.logits_processors` 字段动态加载类，并维护 per-request 状态（`AdapterLogitsProcessor` 抽象在 `:234`）。
4. **支持 logprobs 多 mode**：`LogprobsMode` 三种值 `raw_logprobs` / `raw_logits` / `processed_logprobs`（含 `processed_logits`，`(待核实)` 实际枚举命名见 `vllm/config/model.py`）。`Sampler.forward` 按模式分别走 `compute_logprobs(logits)` 或 `logits.clone()`（`sampler.py:84-94`）。`gather_specific_token_logprobs` 支持 generative-scoring API 只取指定 token id 的 logprob（`:151`）。
5. **`top_tokens` 走通量优化**：`LogitsProcessor.get_top_tokens`（`logits_processor.py:106-156`）做 vocab-parallel argmax——每 rank 算本地 argmax，再 all-gather `[value, index]` 对，避免 all-gather 整个 logits。通信量从 `O(batch*vocab)` 降到 `O(batch*2*tp_size)`，对大词表场景显著降开销。

## 怎么做

### `Sampler.forward` 流水线

`sampler.py:72-149`，按 docstring 9 步：

1. **logprobs 预计算**：若 `num_logprobs` 或 `logprob_token_ids` 非空，按 `logprobs_mode` 生成 `raw_logprobs`（`compute_logprobs` 或 `logits.clone()`）。注意：用原始 logits（penalty/温度前）算 top-k logprobs，与 v0 不同（`sampler.py:80-83` 注释）。
2. **logits → float32**：`logits = logits.to(torch.float32)`。
3. **`apply_logits_processors`**（`:371-420`）：按顺序施加 `allowed_token_ids_mask`（masked_fill -inf）、`apply_bad_words`、`non_argmax_invariant` logit processors、`apply_all_penalties`、`ThinkingBudgetStateHolder.apply_to_logits`（如果 thinking budget 激活）。
4. **`sample`**（`:243-302`）：
   - 全 greedy 或全 random 提前返回。
   - `apply_temperature`（`logits.div_(temp.unsqueeze(1))`，in-place，`:227-237`）。
   - 依次施加 `argmax_invariant` logit processors。
   - `topk_topp_sampler` 做 top-k/top-p + Gumbel 采样得到 `random_sampled`。
   - 最终用 `torch.where(temp < eps, greedy, random)` 合并 greedy 与 random 请求。
5. **`gather_logprobs`**（`:308-356`）：用 `torch.topk` 取 top-N logprob；把 sampled token 的 logprob 与 top-N 拼接；用 `batched_count_greater_than`（`ops/logprobs.py`）算 sampled token 的 rank。
6. **输出封装**：`SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=...)`；`sampled` 转 int32 减少张量体积。

### `SamplingMetadata` 的字段含义

`metadata.py:15-55` 是 dataclass，主要分组：

- **温度/top-k/top-p**：`temperature`、`top_p`、`top_k` 张量；`all_greedy`、`all_random` 布尔加速快速路径。
- **随机源**：`generators: dict[int, torch.Generator]`，per-request RNG。
- **logprobs**：`max_num_logprobs: int | None`（None 表示不返回；0 表示只返回 sampled token 的 logprob）；`logprob_token_ids: dict[int, list[int]] | None`（generative-scoring API 用，按 token id 取 logprob）。
- **penalty**：`no_penalties` 标志、`prompt_token_ids`、`frequency/presence/repetition_penalties`、`output_token_ids: list[list[int]]`。
- **词汇约束**：`allowed_token_ids_mask`、`bad_words_token_ids: dict[int, list[list[int]]]`。
- **logit processors**：`logitsprocs: LogitsProcessors`。
- **spec decode**：`spec_token_ids: list[list[int]] | None`（用于 penalty 时把 spec token 算入 output_token_ids）。
- **thinking budget**：`thinking_budget_state_holder`。

由 `Worker` 在每步从 scheduler 的 per-request `SamplingParams` 聚合而成，避免模型 forward 中读 Python 对象。

### `LogitsProcessor` 层

`vllm/model_executor/layers/logits_processor.py:19-162`：

- `forward(lm_head, hidden_states, embedding_bias)`：若 `logits_as_input=True`（部分多模态/嵌入模型）直接把 hidden_states 当 logits；否则 `_get_logits` 调 `lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)`，再按 `use_all_gather` 走 `tensor_model_parallel_all_gather` 或 `tensor_model_parallel_gather`，最后 `[..., :org_vocab_size]` 去掉 padding。
- `soft_cap`：Gemma2 的 `tanh(x/cap)*cap` 软 clamp。
- `scale`：logits 整体缩放。
- `get_top_tokens`：上面提到的 vocab-parallel argmax 快速路径，专门服务 "返回 argmax token" 这类不要求完整 logits 的场景。

### `TopKTopPSampler`

`vllm/v1/sample/ops/topk_topp_sampler.py` 与 `topk_topp_triton.py`：在 `Sampler.__init__` 中实例化并持有。支持 `use_fp64_gumbel`（fp64 Gumbel max 采样）。Top-k 用 `torch.topk`，top-p 在排序后做 cumulative 截断。

### 与 cudagraph 的协同

`forward` 全程 batch-wise，被 cudagraph 捕获。`gather_logprobs` 中显式调用 `torch._dynamo.decorators.mark_unbacked(..., 0)`（`sampler.py:217-218`、`:345-346`）避免 batch_size 从 1 变 2 触发重编译。

## 与其它模块/系统配合

- [embedding.md](embedding.md)：`LogitsProcessor._get_logits` 调 `lm_head.quant_method.apply(lm_head, hidden_states, bias=...)`，本质是 `VocabParallelEmbedding`/`ParallelLMHead` 持有的权重做 GEMM；`get_top_tokens` 用 `lm_head.shard_indices.org_vocab_start_index` 把本地 index 转回全局 token_id。
- [rejection-sampler-layer.md](rejection-sampler-layer.md)：投机解码时由 `RejectionSampler` 取代 `Sampler` 直接处理 logits（但仍复用 `Sampler` 的子能力如 `TopKTopPSampler`、`apply_top_k_top_p`）。
- [pooler.md](pooler.md)：池化模型（embedding/分类）的模型 forward 不经过 `Sampler`，而是直接由 `Pooler` 收敛 hidden_states。
- [sampling-decoding #06](../../06-sampling-decoding/README.md)：本层是 `#06` 的执行侧；`#06` 还会描述请求侧参数解析（`SamplingParams`）、结构化输出（grammar）等上游链路。
- [engine-core #01](../../01-engine-core/README.md)：`Worker` 在每步从 scheduler 收到 per-request `SamplingParams`，构造 `SamplingMetadata` 后传入 `Sampler`。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`Sampler.forward` 内部允许 torch.compile;cudagraph 捕获要求 batch_size 已知；`mark_unbacked` 避免重编译。
- [model-zoo #04](../../04-model-zoo/README.md)：所有生成式模型在 `forward` 末端返回 `(hidden_states, None)`，由 `ModelRunner` 调 `compute_logits`（拿 `LogitsProcessor`）+ `Sampler` 收尾； pooling 模型直接调 `Pooler`。

## 历史版本演进

- **v0 引擎**：`Sampler` 与 `SamplingMetadata` 在 `vllm/model_executor/layers/sampling.py`；logits processor 在 `vllm/logits_processor.py`；与 v0 引擎耦合。
- **v0.6–v0.7**：`LogitsProcessor` 层引入，把"hidden_states→logits→gather→裁剪"集中化；`get_top_tokens` vocab-parallel argmax 路径加入。
- **v0.7**：v1 引擎启动，重新在新目录 `vllm/v1/sample/` 实现 `Sampler`；引入 `all_greedy/all_random` 快速路径。
- **v0.8–v0.9**：logits processor 拆分成 `non_argmax_invariant` 与 `argmax_invariant` 两组；`MinTokensLogitsProcessor`、`LogitBiasLogitsProcessor`、`MinPLogitsProcessor` 内置；`AdapterLogitsProcessor` 抽象允许插件式 logit processor。
- **v0.9–v0.10**：`logprobs_mode` 多模式（`raw_logprobs/raw_logits/processed_logprobs/processed_logits`）引入；`logprob_token_ids` 支持 generative-scoring API；`gather_specific_token_logprobs` 加入。
- **v0.10**：`ThinkingBudgetStateHolder` 引入支持思考预算（`</think>` 边界强制）；`_combine_outputs_with_spec_tokens` 在 spec decode 时把 spec token 合入 `output_token_ids` 以便 thinking budget 正确算位置。
- **v0.10末–v0.11**：`batched_count_greater_than` 改 Triton kernel，`mark_unbacked` 显式标注避免 batch_size 重编译。
- **v0.11–v0.12**：`use_fp64_gumbel` 引入，对极小概率 token 提供 fp64 Gumbel 路径；`Sampler.forward` 内部不再 `clone()` logits，改 in-place 更新以省内存（与 `RejectionSampler` 协同注释 `logits can be updated in place to save memory`）。
- **v0.12 / main**：`predict_bonus_token` 参数化在 spec decode + thinking budget 组合下生效；`TopKTopPSampler` 的 Triton 路径在 `topk_topp_triton.py` 持续优化大词表性能。

[← 返回层库首页](../README.md)

## 参见

- [rejection-sampler-layer.md](rejection-sampler-layer.md)：投机解码场景下的扩展采样层。
- [embedding.md](embedding.md)：`LogitsProcessor` 与 `ParallelLMHead` 的耦合。
- [sampling-decoding #06](../../06-sampling-decoding/README.md)：请求侧 `SamplingParams` 与结构化输出。
