[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > Step3p5MTPProposer

# Step3p5MTPProposer（Step3.5 MTP）

> 源码：`vllm/v1/spec_decode/step3p5.py`

---

## 是什么

`Step3p5MTPProposer` 是 Step3.5 模型的 MTP（Multi-Token Prediction）drafter，继承 `EagleProposer`，但覆盖几个关键方法以支持 Step3.5 的两个独有特性：

1. **per-MTP-layer draft-step 选择**：每个 MTP layer 在 forward 时被传入 `spec_step_idx`（0,1,...,K-1），允许 layer 内部根据 draft step 索引切换 head/behavior。
2. **多 KV cache group**：Step3.5 的多个 MTP layer 可能落在不同 KV cache group（不同 head dim / 不同 sliding 窗口），基类的"所有 draft 层同 group"断言被解除。

类签名（`vllm/v1/spec_decode/step3p5.py:24`）：

```python
class Step3p5MTPProposer(EagleProposer):
    """Step3.5 MTP proposer with per-layer draft-step selection."""
```

## 为什么

- **per-step head**：Step3.5 每个 MTP layer 对应预测位置 i，其 lm_head 与 layer weights 都按 step 索引选择；预测第 0 个 token 用 layer 0，第 1 个用 layer 1，以此类推。`compute_logits(hidden_states, spec_step_idx=...)` 在模型 forward 中按 step 取出对应 head。
- **多 KV cache group 必要性**：Step3.5 model_arch_config 中可能定义 sliding_attention 与 full_attention 两种 layer 类型在不同 MTP layer 中并存；强制单 group 会导致 attention metadata builder 拿到错误 block_table。
- **Step3.5 lm_head 不共享**：`_maybe_share_lm_head` 覆盖为空（行 154）——Step3.5 checkpoints 自带每层 lm_head，不复用 target 的。
- **draft quant config**：`_create_draft_vllm_config` 覆盖以应用 `get_draft_quant_config`，让 drafter 可以独立走不同量化配置。

## 怎么做

### propose（行 274）

完整覆盖基类 `propose`，主体逻辑相同，关键差异：

- `model_kwargs["spec_step_idx"] = 0` 显式传入 first forward（行 317）。
- 后续 multi-pass 每次：`spec_step_idx = token_index + 1`（行 393），与 loop iteration 一一对应。
- `_sample_draft_tokens_for_step(hidden_states, sampling_metadata, spec_step_idx)`（行 257）替代 `_sample_draft_tokens`，内部调 `compute_logits(hidden_states, spec_step_idx=spec_step_idx)` 而非无参数版本。

### _sample_draft_tokens_for_step（行 257）

```python
def _sample_draft_tokens_for_step(self, hidden_states, sampling_metadata, spec_step_idx):
    if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states), None
        logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
        return logits.argmax(dim=-1), None
    logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
    return self._sample_from_logits(logits, sampling_metadata)
```

`get_top_tokens` 路径不传 spec_step_idx（用模型自身的 step 推断），其他路径显式传 step。

### initialize_attn_backend（行 173）

覆盖基类：

- 不调 `validate_same_kv_cache_group`（直接 skip）。
- 对每个 draft attn layer 按 `(backend.full_cls_name(), kv_cache_group_id)` 分组创建 `AttentionGroup`，每组用对应 gid 的 `kv_cache_spec` 与 `kernel_block_size`。
- 每组调 `create_metadata_builders` 独立创建 metadata builder。

### build_per_group_and_layer_attn_metadata（行 121）

覆盖基类以支持 per-group block_table：

- 从 `_per_group_block_tables` 取每 group 的 block_table，切片到 `num_reqs`。
- 从 `_per_group_slot_mappings` 取每 group 的 slot_mapping，切片到 `num_actual_tokens`。
- 对每 group `attention_group.get_metadata_builder().build_for_drafting(common_attn_metadata=cm, draft_index=draft_index)`。
- `cm` 是 `common_attn_metadata` 的浅拷贝（`copy(common_attn_metadata)`），替换了 `block_table_tensor` 与 `slot_mapping`。

### _update_positions_dependent_metadata（行 79）

覆盖基类：

- 调 super() 算 primary gid 的 slot_mapping 与 seq_lens（+=1）。
- 把 primary gid 的 slot_mapping 存回 `_per_group_slot_mappings[kv_cache_gid]`。
- 对剩余 gid：用各自 `_per_group_block_tables` 重算 slot_id = `block_id * block_size + (new_pos % block_size)`，越界填 `PADDING_SLOT_ID`。
- 完成后每 group 的 slot_mapping 都被更新，准备下一 round attention metadata build。

### _get_slot_mapping（行 58）

覆盖基类按 group 选择 buffer：

- primary gid 用基类的 `_slot_mapping_buffer`。
- 其他 gid 用 `_per_group_slot_mapping_buffers`（懒初始化）。
- 把 `_per_group_slot_mappings[gid]` 拷贝到 buffer，padding 槽位填 PADDING_SLOT_ID。
- 返回 `{layer_name: buffer_view}` per layer。

### set_per_group_attn_metadata（行 40）

外部（GPUModelRunner 在 `_prepare_inputs` 时）调用：

```python
def set_per_group_attn_metadata(self, gid, block_table, slot_mapping):
    self._per_group_block_tables[gid] = block_table
    self._per_group_slot_mappings[gid] = slot_mapping
```

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：基类流程；Step3.5 在 attn metadata 层与 multi-pass 内部差异最大。
- [eagle.md](eagle.md)：底类；Step3.5 与 EAGLE3 在 hidden state 输入上兼容。
- [gemma4.md](gemma4.md)：另一种多 KV cache group drafter，但 Gemma4 用 `constant_draft_positions=True`，Step3.5 仍是 autoregressive。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：ModelRunner 在 `_prepare_inputs` 时为 Step3.5 调 `set_per_group_attn_metadata` 注入每 group block_table（待核实：调用位置与具体函数）。
- [注意力后端](../../05-attention/README.md)：`build_for_drafting` 接口；多 group 必须 builder 独立。
- [模型库-Step3.5](../../04-model-zoo/README.md)（待补充）：Step3.5 模型架构具体实现位置。
- [配置体系-quant](../../10-config/README.md)：`get_draft_quant_config` 让 drafter 走不同量化配置。

## 历史版本演进

- **v0.10.5**：Step3.5 模型 landfall，`Step3p5MTPProposer` 与之同步引入；覆盖 `propose` / `initialize_attn_backend` / `_get_slot_mapping` / `_update_positions_dependent_metadata`。
- **v0.11.0**：`spec_step_idx` 显式入参化（之前可能内嵌在 model forward 内部）。
- **v0.12 / main**：`_per_group_slot_mapping_buffers` 懒初始化与 PADDING_SLOT_ID 填充完善，处理 batch_size 与 input_batch_size 不同的情况。

[← 返回投机解码](../README.md)

## 参见

- [eagle.md](eagle.md)
- [gemma4.md](gemma4.md)
- [mtp.md](mtp.md)
- [llm-base-proposer.md](llm-base-proposer.md)
