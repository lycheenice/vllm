[← Wiki 首页](../README.md) > [采样与解码](../README.md) > RejectionSampler

# RejectionSampler（拒绝采样器）

> 源码：`vllm/v1/sample/rejection_sampler.py`

---

## 是什么

`RejectionSampler` 是 spec decode 的"验证器"——给定 draft 模型提议的 `draft_token_ids` 与 target 模型对每个 draft 位置计算的 `target_logits`，按 [Leviathan 2022](https://arxiv.org/abs/2211.17192) 的拒绝采样算法决定每个 draft token 是否被接受，并在第一个被拒绝的位置采一个"recovered token"补偿分布差异，最后在全部接受时附上一个 bonus token。

类签名（`vllm/v1/sample/rejection_sampler.py:37`）：

```python
class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler, spec_config: SpeculativeConfig | None = None,
                 device: torch.device | None = None): ...
    def forward(self, metadata: SpecDecodeMetadata,
                draft_probs: torch.Tensor | None,
                logits: torch.Tensor,
                sampling_metadata: SamplingMetadata) -> SamplerOutput: ...
```

输出术语（直接引自其 docstring）：
- **accepted tokens**：`draft_prob/target_prob >= uniform(0,1)` 通过的 draft token。
- **recovered tokens**：第一个拒绝位置从 `max(target_prob - draft_prob, 0)` 归一化分布采的补偿 token。
- **bonus tokens**：全部 draft 被接受后，从 target 分布单独采的"额外 token"，由外层 `Sampler` 调用产生（`predict_bonus_token=True`），保证 bonus 可走 top-k/top-p 等普通采样策略。
- **output tokens = accepted + recovered + bonus**。

被拒绝的位置在输出张量中标记为 `PLACEHOLDER_TOKEN_ID = -1`，由 `parse_output` 在 CPU 端过滤。

## 为什么

- **statistical equivalence**：拒绝采样保证"无论 draft 模型多差，输出分布始终等于 target 模型分布"——draft 越好接受率越高、加速越大，draft 越差接受率越低但分布不变。这是 spec decode 的核心安全保证。
- **kernel fusion**：把"accept/reject/recover/bonus"四个动作合并到两个 Triton kernel（greedy 与 random 各一个），避免在 Python 层循环 draft token、避免 CPU-GPU 同步。每个 request 一个 program，内层 `for pos in range(num_draft_tokens)` 静态编译。
- **支持 greedy + random 混合**：同 batch 中可同时有 `temperature=0`（greedy）与 `temperature>0` 的请求；`rejection_greedy_sample_kernel` 与 `rejection_random_sample_kernel` 互不踩踏（前者对 random 请求 early-exit，后者反之）。
- **bonus token 单独采样**：bonus 必须能使用 top-k/top-p/min_p 等约束；rejection 内核只算 raw 概率，bonus 由 `Sampler` 调用走完整流程，然后通过 `bonus_token_ids` 传入拒绝内核。
- **synthetic mode**：配置 `rejection_sample_method="synthetic"` + `synthetic_acceptance_rates` 时，跳过 draft_probs 比较、用预设的 per-position 接受率与均匀随机数比较——用于"无 drafter 也能验证 spec decode 链路"的 benchmark/调测。

## 怎么做

### forward 主流程

`RejectionSampler.forward`（行 88）共六步：

1. `bonus_logits = logits[bonus_logits_indices]`（行 129），调用 `self.sampler(..., predict_bonus_token=True, logprobs_mode_override="processed_logits"/"raw_logits")` 走完整采样管线，得到 bonus_token_ids 与 bonus logprobs。
2. `raw_target_logits = logits[target_logits_indices]`（行 148），clone 后跑 `apply_logits_processors`——这里会处理 spec_token_ids 拼接、allowed_token_ids mask、bad_words（draft 版）、min_tokens（spec_decode 版）、penalties（带 repeat_indices 展开）、thinking budget。
3. `apply_sampling_constraints`（行 510）对 target_logits 应用 temperature（按 `cu_num_draft_tokens` 展开）、top-k、top-p——这与 `Sampler.sample` 有部分重叠但需要按 token 维度展开而非 batch 维度。
4. `rejection_sample`（行 394）核心：建 `[batch, max_spec_len+1]` 输出缓冲、生成 uniform_probs（float64 防止 0.0）、按 `all_greedy`/`all_random` 决定走哪条 kernel 路径。
5. greedy/非全 greedy 分别调 `rejection_greedy_sample_kernel` 与 `rejection_random_sample_kernel`（行 456、491）；非全 greedy 还需 `sample_recovered_tokens`（行 663）算 recovered_token_ids。
6. `_get_logprobs_tensors` 把 accepted/recovered/bonus 位置的 target_logits/bonus_logits 重新组装（行 199），用 `gather_logprobs` 取 top-k 返回。

### apply_logits_processors（spec decode 版）

`RejectionSampler.apply_logits_processors`（行 285）与 `Sampler.apply_logits_processors` 看似同名但实现差异很大：

- 必须把 `[batch, ...]` 维度的"请求级"参数（penalties、allowed_token_ids、bad_words、thinking budget）按 `num_draft_tokens` **repeat_interleave** 展开到 `[num_draft_tokens_total, ...]`（repeat_indices 的用途，行 312–319）。
- `output_token_ids` 用 `_combine_outputs_with_spec_tokens` 拼接 spec_token_ids 以便 penalties 包含 spec 历史。
- `MinTokensLogitsProcessor` 走 `apply_with_spec_decode` 而非 `apply`，按 `num_draft_tokens` cumsum 后对前 `remaining` 个 draft 位置屏蔽所有 stop tokens（`vllm/v1/sample/logits_processor/builtin.py:235`）。
- bad_words 走 `apply_bad_words_with_drafts`（`ops/bad_words.py:39`），按每个 draft 位置独立计算前缀。

### rejection_greedy_sample_kernel

Triton kernel（行 715）。每个 program 一个 request：

- 若 `is_greedy_ptr == None`（全 greedy batch），所有请求都进 greedy 路径。
- 对每个 draft 位置 `pos`：
  - synthetic mode：`accepted = uniform_prob < rate && draft_token_id >= 0`；否则 `accepted = (draft_token_id == target_argmax)`。
  - 一旦 reject，剩余位置都填充 `target_argmax`（保持 token 流连续），但不再写入 output（因为 output 默认 PLACEHOLDER）。
- 全部接受：append `bonus_token_id` 到末尾。

### rejection_random_sample_kernel

Triton kernel（行 774）。每个 program 一个 request，且 `is_greedy` 为 True 时 early exit：

- 对每个 draft 位置 `pos`：
  - draft_id < 0（padding）：直接 reject。
  - synthetic mode：`accepted = uniform_prob < rate`。
  - standard mode：`draft_prob = draft_probs[pos, draft_id]`（或 NO_DRAFT_PROBS 时取 1，用于 ngram），`accepted = draft_prob > 0 && target_prob/draft_prob >= uniform_prob`。
  - accept → 写 draft_token_id；reject → 写 `recovered_token_ids[pos]`，并标记 rejected（后续位置全部 skip，由 PLACEHOLDER 表征）。
- 全部接受：append bonus。

### sample_recovered_tokens_kernel

Triton kernel（行 873）。每个 (request, pos) 一个 program：

- 当 NO_DRAFT_PROBS：prob = target_prob（mask 掉 draft_token_id 自身），等价于"无 draft 信息时直接从 target 分布中采样一个非 draft token"。
- 否则：prob = max(target_prob - draft_prob, 0)（标准 speculative decoding 的 residual 分布）。
- 用 Gumbel-max 技巧：`score = prob * inv_q`，`inv_q` 来自 `q.exponential_().reciprocal()`，每 request 一份 q（行 682）；`tl.max(... return_indices=True)` 在 BLOCK_SIZE=8192 的 tile 内取 argmax，跨 tile 归约得 recovered_id。
- `use_fp64_gumbel` 用 float64 而非 float32 跑 Gumbel，在数值敏感场景提高精度。

### parse_output（CPU 端）

`RejectionSampler.parse_output`（行 249）把 `[batch, max_spec_len+1]` 的输出张量按 PLACEHOLDER 与 vocab_size 过滤，转成 `list[list[int]]`；同时可选过滤 logprobs_tensors。

## 与其它模块/系统配合

- [sampler.md](sampler.md)：bonus token 走 `Sampler.forward` 完整流程；`logprobs_mode_override` 让 sampler 在 spec 上下文返回 raw/processed logits 以供 `_get_logprobs_tensors` 二次组装。
- [speculative-decoding/README.md](speculative-decoding/README.md)：上层 proposers 提供 `draft_token_ids` / `draft_probs`；ngram/suffix 等"无 draft 概率"的 proposer 把 `draft_probs=None`，rejection kernel 走 NO_DRAFT_PROBS 分支。
- [speculative-decoding/metrics.md](speculative-decoding/metrics.md)：scheduler 根据 rejection 输出统计 `num_accepted_tokens`、`num_draft_tokens`、`num_accepted_tokens_per_pos`，喂给 `SpecDecodingLogging`/`SpecDecodingProm`。
- [引擎核心-调度](../01-engine-core/scheduler/README.md)：调度器在请求推进时根据接受长度更新 `num_computed_tokens`；`SpecDecodeMetadata.make_dummy` 用于 prefill-only step 的占位（无 draft token）。
- [执行层-GPUModelRunner](../02-execution/worker/README.md)：持有 `rejection_sampler` 实例，在 `forward` 之后根据是否 spec decode 切换 sampler/rejection_sampler。
- [thinking-budget.md](thinking-budget.md)：`ThinkingBudgetStateHolder.apply_to_logits` 在 rejection sampler 内被第二次调用（行 340），针对 target_logits 而非 bonus_logits。
- [structured-output/README.md](structured-output/README.md)：当 spec decode + structured output 同时开启时，bitmask 必须覆盖每个 draft 位置；`apply_grammar_bitmask` 与 `apply_sampling_constraints` 串联执行。

## 历史版本演进

- **v0.6.x（V0）**：V0 的 reject sampling 在 `vllm/spec_decode/` 下，纯 Python + per-token loop；不支持 greedy/random 混合，性能很差。
- **v0.7.0**：V1 `RejectionSampler` 落地，首次引入 Triton kernel；`rejection_sample`、`apply_sampling_constraints`、`expand_batch_to_tokens`、`generate_uniform_probs` 四个 helper 同步引入。
- **v0.7.5**：`MAX_SPEC_LEN = 128` 引入；`do_not_specialize=["max_spec_len"]` 防止 max_spec_len 变化触发重编译。
- **v0.8.0**：`sample_recovered_tokens_kernel` 引入 Gumbel-max 替代 multinomial（避免同步），`NO_DRAFT_PROBS` constexpr 分支支持 ngram proposer。
- **v0.8.5**：`use_fp64_gumbel` 选项；bonus logits 与 target logits 分开处理，避免 in-place 修改原 logits 影响 bonus 采样。
- **v0.9.0**：synthetic mode 引入（`synthetic_acceptance_rates` → `unconditional_to_conditional_rates`）；greedy kernel 也开始需要 uniform_probs。
- **v0.10.0**：`_get_logprobs_tensors` 重写，按 `cu_num_sampled_tokens` 计算 accepted_logit_indices，避免被 rejected 位置污染；支持四种 logprobs_mode。
- **v0.10.5**：`thinking_budget_state_holder` 钩入 `apply_logits_processors`；spec_token_ids 与 end_count 联动。
- **v0.11.0**：`MinTokensLogitsProcessor.apply_with_spec_decode` 引入，按 draft 位置精确屏蔽 stop tokens，修复 spec decode + min_tokens 的接受率退化 bug。
- **v0.12 / main**：并行 drafting（DFlash/DSpark）让 `bonus_token_ids` 在 proposer 内部一次采完；`rejection_random_sample_kernel` 不再依赖 draft_probs 顺序，仍按 `cu_num_draft_tokens` 切分。`MAX_SPEC_LEN` 对并行 drafting 场景偏小（待核实：并行 drafting 时 max_spec_len 实际上限）。

[← 返回采样与解码](../README.md)

## 参见

- [sampler.md](sampler.md)：被 rejection sampler 内部调用跑 bonus
- [speculative-decoding/README.md](speculative-decoding/README.md)
- [speculative-decoding/metrics.md](speculative-decoding/metrics.md)
- [sampling-ops.md](sampling-ops.md)：`apply_top_k_top_p` 在 `apply_sampling_constraints` 中被复用
- [thinking-budget.md](thinking-budget.md)
