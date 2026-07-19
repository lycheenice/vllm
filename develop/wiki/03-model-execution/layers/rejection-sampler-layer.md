# 拒绝采样层（RejectionSampler）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 拒绝采样层

> 物理位置提示：`vllm/v1/sample/rejection_sampler.py`（~953 行）。同 [sampler-layer.md](sampler-layer.md) 一样，本层在 v1 引擎下从旧 `vllm/model_executor/layers/sampling.py` 迁出；概念上仍属层库范畴，与 `Sampler` 同源。

## 是什么

| 类/函数 | 文件:行 | 角色 |
|---|---|---|
| `RejectionSampler` | `rejection_sampler.py:37` | `nn.Module`，投机解码（speculative decoding）末端"读取 draft + target logits，按拒绝采样协议产出 accepted/recovered/bonus token"层 |
| `SpecDecodeMetadata` | `vllm/v1/spec_decode/metadata.py:10` | dataclass：`draft_token_ids`、`num_draft_tokens`、`cu_num_draft_tokens`、`cu_num_sampled_tokens`、`target_logits_indices`、`bonus_logits_indices`、`logits_indices` |
| `rejection_sample` | `rejection_sampler.py:394` | 主算子：批量执行 greedy 与 random 两个 Triton kernel |
| `apply_sampling_constraints` | `:510` | 对未走 spec 路径的 bonus token 位置施加 top-k/top-p 约束 |
| `expand_batch_to_tokens` | `:568` | 把 per-request 数据展开成 per-token 张量喂给 kernel |
| `rejection_greedy_sample_kernel` | `:715` | Triton kernel：greedy draft 的接受/拒绝 + bonus |
| `rejection_random_sample_kernel` | `:774` | Triton kernel：random draft 的接受/拒绝 + recovered + bonus |
| `expand_kernel` | `:850` | Triton kernel：把 per-request 的 cu_num_draft_tokens 等扩展 |
| `sample_recovered_tokens` / `sample_recovered_tokens_kernel` | `:663` / `:873` | 拒绝后从"调整后分布"重采 token |
| `generate_uniform_probs` | `:608` | 为拒绝采样生成 uniform 噪声 |
| `parse_output` | `:249` | 把 `[batch, max_spec_len+1]` 输出解析为各 request 的 accepted+bonus tokens |

`MAX_SPEC_LEN=128`（`rejection_sampler.py:34`）：单步最大 draft token 数；`PLACEHOLDER_TOKEN_ID=-1`、`GREEDY_TEMPERATURE=0` 是 kernel 内 `tl.constexpr` 常量。

## 为什么

把拒绝采样单独成层，是为了：

1. **严格按论文实现**：算法遵循 [Accelerating LLM with Speculative Sampling](https://arxiv.org/abs/2211.17192)。术语约定（注释 `:38-58`）：
   - **accepted**：draft token 与 target argmax/random 关系通过校验而保留的 token。
   - **recovered**：在拒绝位置按调整后的分布重采出的 token。
   - **bonus**：所有 draft 都接受时附加的额外 token，由上层（`Sampler.topk_topp_sampler`）按 top-k/top-p 采样传入，避免 spec 路径不支持 top-k/top-p 的问题。
   - **output** = accepted + recovered + bonus。
2. **支持多种 proposer 类型**：`RejectionSampler.__init__` 接收 `Sampler`、`SpeculativeConfig`、`device`；当 `rejection_sample_method == "synthetic"` 且 `synthetic_acceptance_rates` 非空时进入 `synthetic_mode`——跳过真实 draft 概率，按预置 conditional acceptance rates 模拟接受/拒绝，用于离线性能评测与压测。`NO_DRAFT_PROBS` 编译期常量区分 ngram proposer（无 draft 概率）与 model proposer（有 draft 概率）。
3. **Triton 单 kernel 处理整批**：因为每 request 的 draft 长度不同、含 greedy 与 random 混合，朴素 Python 实现会非常慢。`rejection_greedy_sample_kernel` / `rejection_random_sample_kernel` 以 `(batch_size,)` 做 grid，每个 program 处理一个 request 的全部 draft token，全程 in-place 写入 `[batch, max_spec_len+1]` 输出。
4. **统一处理 logprobs 三 mode**：`is_processed_logprobs_mode` / `is_logits_logprobs_mode` 从 `Sampler.logprobs_mode` 推断（`rejection_sampler.py:70-72`），在 `forward` 内分支处理 raw vs processed；`_get_logprobs_tensors`（`:199`）按模式输出。
5. **与 Sampler 子能力共享**：`RejectionSampler.__init__` 接收 `Sampler` 实例（`:60-71`），共享 `TopKTopPSampler`、`apply_top_k_top_p`、`use_fp64_gumbel`、`apply_bad_words_with_drafts`、`apply_all_penalties`，确保 spec 路径下的 logit 处理与普通采样一致。

## 怎么做

### `forward` 调用链

`rejection_sampler.py:88-198` 入口；流程：

1. `apply_logits_processors`（`:285`）：对 target logits（含 bonus 位置）施加 allowed/bad words/penalty/thinking budget；spec 解码场景下用 `apply_bad_words_with_drafts`（`ops/bad_words.py`）考虑 draft token。
2. `apply_sampling_constraints`（`:510`）：对 bonus logits 位置施加 `apply_top_k_top_p`，产出 bonus token。
3. `expand_batch_to_tokens`（`:568`）：把 `[batch, ...]` 展开成 `[num_tokens, vocab]` 喂给 kernel；同时切出"target_logits"对应 draft token 的子张量。
4. `rejection_sample(...)`（`:394`）：
   - 若 `all_greedy` 且无 synthetic：跳过 uniform 生成；`rejection_greedy_sample_kernel` 直接基于 `target_argmax` 与 `draft_token_ids` 比较。
   - 否则先生成 `uniform_probs`，调 `rejection_greedy_sample_kernel`（处理混合 batch 中 greedy 部分），再算 `target_probs = softmax(target_logits)`，调 `sample_recovered_tokens`（含 `sample_recovered_tokens_kernel` Triton）算每个拒绝位置的 recovered token，最后 `rejection_random_sample_kernel` 把 accepted/recovered/bonus 拼到 `output_token_ids`。
5. `parse_output`（`:249`）：把 `[batch, max_spec_len+1]` 输出按 placeholder 切分，挑出非 placeholder 的有效 token，组合 `[num_request, num_output_tokens]` 形态的 `SamplerOutput`。
6. `_get_logprobs_tensors`（`:199`）按 `logprobs_mode` 输出 bonus / accepted / recovered 的 logprobs（如果请求）。

### Triton kernel 形态

| Kernel | Grid | 输入 | 输出 |
|---|---|---|---|
| `rejection_greedy_sample_kernel` | `(batch_size,)` | `output_token_ids`、`cu_num_draft_tokens`、`draft_token_ids`、`target_argmax`、`bonus_token_ids`、`is_greedy`、`max_spec_len`、`uniform_probs?`、`synthetic_conditional_rates?` | in-place 写 `output_token_ids` 的前若干列（accepted/bonus） |
| `rejection_random_sample_kernel` | `(batch_size,)` | 同上 + `draft_probs?`、`target_probs`、`recovered_token_ids` | in-place 写完整行 |
| `sample_recovered_tokens_kernel` | per-token | `draft_token_ids`、`draft_probs?`、`target_probs`、`uniform_probs`、`cu_num_draft_tokens` | `recovered_token_ids` |
| `expand_kernel` | `(batch_size,)` | `cu_num_draft_tokens` 等 | 展开后的 per-token 视图 |

`(待核实)` 详细 kernel 内部循环、warp 配置与内存布局可参考 `vllm/v1/sample/rejection_sampler.py:715-953` 的源码注释；本文不展开。

### synthetic 模式

`SpeculativeConfig.rejection_sample_method = "synthetic"`（`rejection_sampler.py:74-86`）：

- `synthetic_acceptance_rates` → `unconditional_to_conditional_rates(...)` 算出 `synthetic_conditional_rates: torch.Tensor`（per-slot 条件接受率）。
- `synthetic_mode=True` 时 kernel 把接受判定从"对比 draft/target prob"替换为"uniform_prob < conditional_rate[slot]"，因此不再需要 draft 概率，也不再需要 target 的 softmax——极大降低 kernel 成本。
- 主要用于 proposer 在线训练阶段或硬件压力测试，验证 target 模型吞吐瓶颈。

### 多流 / 同步（spec pipeline 的"同步")

任务文档中提到 "mps/sync 等层"——经核查，`vllm/v1/sample/` 中并无显式名为 `MPS` 或 `sync` 的类或函数 `(待核实)`。推测指代的是 v1 spec decode 编排层面（`vllm/v1/spec_decode/` 与 `vllm/v1/engine/core.py` 中）对"draft 模型与 target 模型的多 stream / 多 process 执行 + 同步"的整体调度。"在一个 spec decode step 内，draft 前向 → RejectionSampler.apply_logits_processors → RejectionSampler.forward → target 前向" 的同步点位于 `gpu_worker.py` 与 `spec_decode_metadata.py`；`RejectionSampler` 自身只消费"双流已对齐后的 logits"。详见 [spec decode 子系统 #06](../../06-sampling-decoding/README.md)。

## 与其它模块/系统配合

- [sampler-layer.md](sampler-layer.md)：`RejectionSampler` 在 `__init__` 持有 `Sampler`，复用其 `TopKTopPSampler`、`logprobs_mode`、`use_fp64_gumbel`、`apply_top_k_top_p`、`apply_bad_words_with_drafts`、`apply_all_penalties` 等子能力。
- [embedding.md](embedding.md)：target/draft 的 logits 由 `LogitsProcessor.forward(lm_head, hidden_states)` 产出；`bonus_logits_indices` 直接索引 target logits 末段。
- [sampling-decoding #06](../../06-sampling-decoding/README.md)：spec decode 提出 draft 的五个 proposer（`eagle.py` / `medusa.py` / `ngram_proposer.py` / `ngram_proposer_gpu.py` / `custom_class_proposer.py` / `draft_model.py` / `gemma4.py` / `dflash.py` / `step3p5.py` / `suffix_decoding.py` / `llm_base_proposer.py`）都把 draft logits/token 喂给本层；`SpecDecodeMetadata.make_dummy` 用于不需要真实 draft 概率的 ngram 路径。
- [engine-core #01](../../01-engine-core/README.md)：`SpecDecodeMetadata` 由 EngineCore 在调度期构造，通过 `forward_context` 传入 Worker 的 `RejectionSampler.forward`；cudagraph 捕获时 `MAX_SPEC_LEN=128` 是静态上界。
- [compilation-ir #09](../../09-compilation-ir/README.md)：Triton kernel 内部用 `tl.constexpr` 编译期常量（`SYNTHETIC_MODE`、`NO_DRAFT_PROBS`），对应 `torch.compile` 的"特化代码路径"机制——同一段 kernel 在不同 spec 配置下生成不同 binary。
- [model-zoo #04](../../04-model-zoo/README.md)：模型代码不直接调 `RejectionSampler`；由 `ModelRunner` 在 spec decode 启用时替换普通 `Sampler.forward` 调用。

## 历史版本演进

- **v0 引擎**：拒绝采样逻辑分散在 `vllm/spec_decode/` 多个文件，与 v0 scheduler/worker 强耦合。
- **v0.6–v0.7**：v1 引擎启动后，`RejectionSampler` 重写在 `vllm/v1/sample/rejection_sampler.py`，统一 dict 输入并支持 batch 混合 greedy/random。
- **v0.8**：引入 `MAX_SPEC_LEN` 静态上界以支持 cudagraph；bonus token 不再在 sampler 内采样，改由上层传 `bonus_token_ids`。
- **v0.9**：合成模式（synthetic mode）引入，服务 spec decode acceptance rate 离线调评；`synthetic_conditional_rates` 取代真实 draft 概率。
- **v0.10**：`logprobs_mode` 多模式（raw_logprobs / raw_logits / processed_*）扩展到 spec decode；`is_processed_logprobs_mode` 与 `is_logits_logprobs_mode` 分支进入 `_get_logprobs_tensors`。
- **v0.10末–v0.11**：`apply_bad_words_with_drafts` 引入——在 spec 路径上 bad words 约束需考虑 draft token；`predict_bonus_token` + thinking budget 协同（bonus 位置也要参与 thinking budget 强制）。
- **v0.11–v0.12**：`use_fp64_gumbel` 引入，配合 `Sampler` 的 fp64 Gumbel 路径；`rejection_greedy_sample_kernel` 与 `rejection_random_sample_kernel` 拆分独立 kernel，避免合并 kernel 的 warp 占用过高问题。
- **v0.12 / main**：`MAX_SPEC_LEN` 由 64 提升到 128（`rejection_sampler.py:34`）以支持更长 draft；`ngram_proposer_gpu.py` 等 GPU ngram proposer 上线，复用 `NO_DRAFT_PROBS` 路径；`step3p5.py`、`suffix_decoding.py`、`dflash.py` 等 proposer 接入扩展 spec 提议选择面。具体每个 proposer 的接受率历史 `(待核实)`，详见 [spec decode #06](../../06-sampling-decoding/README.md)。

[← 返回层库首页](../README.md)

## 参见

- [sampler-layer.md](sampler-layer.md)：普通采样层；本层复用其子能力。
- [sampling-decoding #06](../../06-sampling-decoding/README.md)：spec decode 编排与各 proposer。
- [embedding.md](embedding.md)：`LogitsProcessor` 与 `ParallelLMHead` 产出 logits。
