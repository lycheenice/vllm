[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > SpecDecodeBaseProposer

# SpecDecodeBaseProposer（drafter 基类）

> 源码：`vllm/v1/spec_decode/llm_base_proposer.py`（1856 行）

---

## 是什么

`SpecDecodeBaseProposer` 是 EAGLE / EAGLE3 / DFlash / Gemma4 / Step3.5 MTP / Draft model / 通用 MTP 等所有"需要模型 forward 才能 propose"的 drafter 的共同基类。它的核心职责是：

- 持有 drafter 模型（`self.model`）与输入缓冲（`input_ids` / `positions` / `hidden_states` / `inputs_embeds` / `_slot_mapping_buffer`）。
- 提供 `propose(num_speculative_tokens, target_token_ids, target_positions, target_hidden_states, next_token_ids, ...)` 统一入口。
- 维护 `draft_attn_groups` 与 attention metadata build（per-group + per-layer）。
- 与 `CudagraphDispatcher` / EPLB / DP 等执行层机制衔接。
- 支持异构 vocab（`use_heterogeneous_vocab` + `VocabMapping`）与并行 drafting（`parallel_drafting`，DFlash 用）。

Ngram / Suffix / Medusa / ExtractHiddenStates / CustomClass 不继承此基类（接口兼容但独立实现 `propose`）。

## 为什么

- **统一抽象**：六七种 drafter 在"如何用 target hidden/positions 起步 → forward → 采样 → 回填到 K 个 slot"上的流程趋同；抽到基类避免重复实现。
- **CUDA graph 友好**：所有输入用预分配 pinned/GPU buffer；`set_inputs_first_pass` 与 `_update_positions_dependent_metadata` 通过 Triton kernel 把数据拷贝 + slot 重算合并，避免 Python loop。
- **padded drafter batch**：默认启用 `padded_drafter_batch`，所有请求被 pad 到 K 个 draft slot，让 CUDA graph 与 batch 实际接受长度解耦；`disable_padded_drafter_batch` 是 opt-in。
- **M-RoPE / XDRoPE 支持**：drafter 可能与 target 用不同 RoPE 类型；`_set_positions` / `_get_positions` 双 helper 处理 (3, T) / (xdim, T) 与 (T,) 三种形状。
- **多 KV cache group**：DFlash / Gemma4 / Step3.5 跨 sliding/full attention 类型，每个 draft 层可能属于不同 KV cache group，需要 per-group block_table 与 slot_mapping。基类默认只支持单 group，子类覆盖 `build_per_group_and_layer_attn_metadata` / `_get_slot_mapping` 扩展。

## 怎么做

### 关键属性

| 属性 | 含义 |
|---|---|
| `pass_hidden_states_to_model` | drafter 是否需要 target 的 hidden states 作为输入（EAGLE/MTP/DFlash True；独立 draft_model False） |
| `parallel_drafting` | 是否一次 forward 算完所有 K 个 draft（DFlash True；EAGLE 多步 False） |
| `extra_slots_per_request` | 每请求额外占用的 KV slot 数：EAGLE=1（next_token），DFlash=K（所有 draft） |
| `net_num_new_slots_per_request` | 扣除 pass_hidden_states 占用后的真正新增 slot 数 |
| `needs_extra_input_slots` | net > 0 时为 True，启用 `is_rejected_token_mask` / `is_masked_token_mask` 跟踪 |
| `constant_draft_positions` | Gemma4 专用——所有 draft step 复用首个位置 |
| `use_local_argmax_reduction` | 通过 `get_top_tokens` 避免 logit 全 vocab 计算（EAGLE3 多分支） |
| `use_heterogeneous_vocab` | draft 与 target vocab 不同，走 `VocabMapping` |
| `_enable_probabilistic_draft_probs` | standard rejection + probabilistic draft sampling，让 draft_probs 可用 |

### propose 主入口（行 502）

```python
def propose(self, num_speculative_tokens, target_token_ids, target_positions,
            target_hidden_states, next_token_ids, token_indices_to_sample,
            common_attn_metadata, sampling_metadata, mm_embed_inputs=None,
            num_rejected_tokens_gpu=None, slot_mappings=None) -> torch.Tensor:
```

关键步骤：

1. **EAGLE3 / DFlash 特殊化**（行 526）：调 `model.combine_hidden_states(target_hidden_states)` 把多个 aux hidden states 合并。
2. **set_inputs_first_pass**（行 821）：默认 EAGLE 路径——把 `target_token_ids` 左移一位，末位填 `next_token_ids`；`hidden_states` 直接拷贝；positions 不变。`needs_extra_input_slots=True`（draft_model / parallel drafting）路径走 `copy_and_expand_eagle_inputs_kernel` triton kernel 把 context+query 同时拷贝，包含 `is_rejected_token_mask` / `is_masked_token_mask` 标志。
3. **build_per_group_and_layer_attn_metadata**（行 993）：每 attn_group build 一次 metadata。
4. **build_model_inputs_first_pass**（行 962）：组装 `model_kwargs`（含 multimodal embeds）。
5. **MTP index_share step 0**（行 571）：若 `_share_mtp_indices=True`，第一步关 skip_topk 让 MTP layer 自己算 indices。
6. **first forward**（行 580）：`set_forward_context(...)` 包裹 `self.model(**model_kwargs)`；多模态 hidden_states 可能返回 tuple。
7. **采样本位置**（行 605–620）：`sample_hidden_states = last_hidden_states[token_indices_to_sample]`；调 `_sample_draft_tokens`。
8. **early exit**（行 619）：`num_speculative_tokens == 1` 或 `parallel_drafting` 直接返回 `[B, K]`。
9. **multi-pass loop**（行 682–761）：剩余 K-1 个 token，每次 update positions / slot_mapping / attn_metadata → forward → 采样。
10. **stack & return**：`torch.stack(draft_token_ids_list, dim=1)` 得 `[B, K]`；draft_probs 同理。

### set_inputs_first_pass（行 821）

默认 EAGLE 路径不做 pad（`needs_extra_input_slots=False`）：
- `self.input_ids[:num_tokens-1] = target_token_ids[1:]`：所有位置左移。
- `self.input_ids[token_indices_to_sample] = next_token_ids`：每请求最后位置填 next_token。
- `self._set_positions(num_tokens, target_positions)`：处理 M-RoPE / XDRoPE。
- `self.hidden_states[:num_tokens] = target_hidden_states`。

`needs_extra_input_slots=True`（draft_model / parallel drafting）路径走 triton kernel `copy_and_expand_eagle_inputs_kernel`（一次完成 input_ids / positions / hidden_states / masks / slot_mapping 拷贝），并调 `compute_new_slot_mapping` 重算被 reject 的位置。

### _update_positions_dependent_metadata（行 769）

每个 multi-pass iteration 调用：

- `eagle_step_update_slot_mapping_and_metadata` triton kernel（`utils.py:29`）：positions+1 + block table lookup + slot_mapping 重算 + seq_lens+=1，全部 fused 在一个 kernel 中以减少 launch overhead。
- 同步更新 `common_attn_metadata` 的 `max_seq_len` / `_seq_lens_cpu` / `_num_computed_tokens_cpu` / `seq_lens_cpu_upper_bound`。
- M-RoPE / XDRoPE 同步三轴 / 多轴。

### _sample_draft_tokens（行 468）

```python
def _sample_draft_tokens(self, hidden_states, sampling_metadata):
    if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
        return self._greedy_sample(hidden_states), None
    logits = self.model.compute_logits(hidden_states)
    if self.use_heterogeneous_vocab:
        logits = self.vocab_mapping.constrain_draft_logits(logits)
    draft_token_ids, draft_probs = self._sample_from_logits(logits, sampling_metadata)
    if self.use_heterogeneous_vocab:
        draft_token_ids = self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
    return draft_token_ids, draft_probs
```

`_greedy_sample`（行 428）默认走 `compute_logits(...).argmax(-1)`，但 `use_local_argmax_reduction=True` 时调 `model.get_top_tokens(hidden_states)`（EAGLE3 多 LM head 分支合并，避免拼成全 vocab）。

### prepare_next_token_ids_padded（行 1050）

padded drafter batch 模式的下一步输入准备：

- 用 triton kernel `eagle_prepare_next_token_padded_kernel` 一次性计算每请求的 `next_token_ids` 与 `valid_sampled_tokens_count`。
- 处理 `discard_request_mask`：被 discard 的请求走 `backup_next_token_ids`（从 `requests[req_id].get_token_id(num_tokens_no_spec-1)` 取）。
- 此函数是 cuda graph 与非 cuda graph 路径的分水岭——padded 模式下整个 prepare → forward → reject → next_prepare 都能在 cuda graph 里 capture。

### initialize_attn_backend（行 1705）

被 ModelRunner 在 `initialize_metadata_builders` 阶段调用：

- 调 `validate_same_kv_cache_group`（默认断言所有 draft 层属同一 KV group；Gemma4/Step3.5 覆盖此方法解除约束）。
- 按 `kv_cache_group_id` 找到对应 `kv_cache_spec`，为每个 draft attn backend 创建 `AttentionGroup`。
- 设置 `self.block_size` 与 `self.kv_cache_gid`。

### _determine_batch_execution_and_padding（行 1767）

DP 与 cudagraph dispatch：

- `cudagraph_dispatcher.dispatch(num_tokens, valid_modes)` 得到 cudagraph_mode 与 padded num_tokens。
- DP > 1 时调 `coordinate_batch_across_dp` 在 rank 间协商 num_tokens_padded；不支持 ubatching（`allow_microbatching=False`，TODO 注释）。

## 与其它模块/系统配合

- [../rejection-sampler.md](../rejection-sampler.md)：drafter 输出最终流入 RejectionSampler；`take_last_draft_probs` 给 rejection sampler 拿 draft_probs。
- [eagle.md](eagle.md) / [step3p5.md](step3p5.md) / [gemma4.md](gemma4.md) / [dflash.md](dflash.md) / [draft-model.md](draft-model.md) / [mtp.md](mtp.md)：子类按需覆盖。
- [vocab-mapping.md](vocab-mapping.md)：异构 vocab 时由 `_sample_draft_tokens` / `set_inputs_first_pass` 触发 token id 映射。
- [引擎核心-调度](../../01-engine-core/scheduler/README.md)：scheduler 决定 `num_speculative_tokens`（dynamic SD）并把它传给 proposer。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：在 `_prepare_inputs` 阶段调 `drafter.propose(...)`；持有 `use_aux_hidden_state_outputs` 用于决定 forward 是否收集 aux hidden states。
- [注意力后端](../../05-attention/README.md)：`AttentionGroup.create_metadata_builders` + `build_for_drafting`；EAGLE 的 draft KV cache 通过 `kv_sharing_target_layer_name` 跨模型共享（Gemma4 `_setup_gemma4_kv_sharing`）。
- [分布式-EPLB](../../07-distributed/README.md)：`set_eplb_state` 注入；`eplb_state.prepare_forward` 在每个 forward 前。
- [编译与 IR](../../09-compilation-ir/README.md)：`CudagraphDispatcher.initialize_cudagraph_keys` 在 PIECEWISE 模式下捕获 drafter forward。

## 历史版本演进

- **v0.7.0**：基类首次引入，含 `propose`、`set_inputs_first_pass`、`build_model_inputs_first_pass` 等基础方法；EAGLE 是首个子类。
- **v0.8.0**：padded drafter batch 模式落地；`prepare_inputs_padded` / `prepare_next_token_ids_padded` 加入；`_determine_batch_execution_and_padding` 整合 cudagraph。
- **v0.8.5**：MTP（DeepSeek 系）继承此基类；`model_returns_tuple` 区分 DeepSeekMTPModel（返回 `(logit_hidden, recycle_hidden)`）与其他 MTP。
- **v0.9.0**：异构 vocab + VocabMapping 引入；`_enable_probabilistic_draft_probs` 与 `compute_probs_and_sample_next_token`（行 1818，注释说明 draft_probs 当前未启用，正在管理）。
- **v0.10.0**：Dynamic SD 与 `disable_padded_drafter_batch` 选项；`_raise_if_padded_drafter_batch_disabled`、`_warn_if_multimodal`、`_raise_if_mrope` 三个 sanity check 加入。
- **v0.10.5**：`_share_mtp_indices` 字段加入（MTP layer 的 topk indices 跨步骤复用优化）；`constant_draft_positions` 字段加入（Gemma4 Q-only attention）。
- **v0.11.0**：DFlash 子类落地；`parallel_drafting` / `extra_slots_per_request` / `net_num_new_slots_per_request` / `needs_extra_input_slots` 字段族引入；`is_rejected_token_mask` / `is_masked_token_mask` 双 mask 支持 padded reject。
- **v0.12 / main**：EPLB `set_eplb_state` hook 加入；`allowed_attn_types` 列出 ROCm 上支持的多 spec drafter 注意力类型；`initial_attn_allowed_attn_types` 与 `BreakableCUDAGraphWrapper.unwrap` 衔接。

[← 返回投机解码](../README.md)

## 参见

- [eagle.md](eagle.md)
- [gemma4.md](gemma4.md)
- [step3p5.md](step3p5.md)
- [dflash.md](dflash.md)
- [draft-model.md](draft-model.md)
- [mtp.md](mtp.md)
- [vocab-mapping.md](vocab-mapping.md)
