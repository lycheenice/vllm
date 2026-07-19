[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > DFlashProposer

# DFlashProposer（DFlash 并行 drafting）

> 源码：`vllm/v1/spec_decode/dflash.py`

---

## 是什么

`DFlashProposer` 是 DFlash 算法的 drafter 实现，继承 `SpecDecodeBaseProposer`，并启用"并行 drafting"模式——一次 drafter forward 同时算 K 个 draft position，不再走 EAGLE 的 multi-pass。其核心机制是用 mask token 填充未确定位置，让 drafter 的 attention 把上下文 K/V 当作已知，Q 是 `[1 + K]`（bonus + mask_tokens）。

类签名（`vllm/v1/spec_decode/dflash.py:23`）：

```python
class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(vllm_config=vllm_config, device=device,
                         pass_hidden_states_to_model=True, runner=runner)
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        ...
        self.dflash_causal = self.dflash_config.get("causal", False)
```

DFlash 的关键 hf_config 字段：

- `dflash_config["mask_token_id"]`：mask token 的 id。
- `dflash_config["causal"]`：是否使用因果 attention（某些 DFlash 变体用 bidirectional）。
- `dflash_config["use_aux_hidden_state"]`：是否走 EAGLE3-style aux hidden 路径，默认 True。

## 为什么

- **并行 drafting 消除 multi-pass 开销**：EAGLE 的 K 步 forward 串行；DFlash 把 bonus + K 个 mask token 作为 query，target hidden states 作为 context K/V，一次 forward 出 K 个 draft。当 K 较大（如 8+）时延迟显著低于 EAGLE。
- **DFlash 模型架构支持**：DFlash drafter 模型在 attention 中可以处理 mask token 的 bidirectional context；`non-causal`（即 `causal=False`）路径允许 query 看 K/V 的全部上下文（包括其他 mask position），由 `dflash_config["causal"]` 控制。
- **跨注意力机制**：context K/V 来自 target hidden states 的 precompute（`precompute_and_store_context_kv`），drafter 模型本身的 KV projection 已经把 target hidden 转成 drafter dimension 的 K/V 存入 cache；drafter forward 只跑 Q 路径。
- **EAGLE3 aux hidden 兼容**：DFlash 复用 `_get_eagle3_use_aux_hidden_state_from_config` 从 `dflash_config["use_aux_hidden_state"]` 读取，走与 EAGLE3 相同的 target hidden 收集流程。
- **multimodal 兼容**：DFlash 覆盖 `_warn_if_multimodal` 为 no-op（行 92）——Qwen3.5 模型支持 multimodal prefix。

## 怎么做

### set_inputs_first_pass（行 97）

覆盖基类：

- 不旋转 target_token_ids（ drafter 不需要 target token ids，只用 hidden states）。
- 每请求构造 `1 + K` 个 query token：bonus（=next_token_ids）+ K 个 mask_token_id。
- 通过 triton kernel `copy_and_expand_dflash_inputs_kernel`（行 136）一次写入：
  - `out_input_ids`：query ids（bonus + mask）。
  - `out_context_positions` / `out_query_positions`：分离的 context 与 query positions buffer。
  - `out_context_slot_mapping` / `out_query_slot_mapping`：分别对应 cache 写入与 query attention。
  - `out_token_indices`：query 中 mask token 的位置索引（用于后续采样 draft）。
- 处理 `num_rejected_tokens_gpu`：减去 rejected 数量得到 effective context len。
- 构造新 `CommonAttentionMetadata`：`max_query_len = 1 + K`，`num_actual_tokens = batch_size * (1 + K)`，`causal` 由 `dflash_causal` 决定。

### build_model_inputs_first_pass（行 260）

覆盖基类：

- **precompute context K/V**：调 `self.model.precompute_and_store_context_kv(self._dflash_hidden_states, self._context_positions_buffer, self._context_slot_mapping_buffer)`——drafter 模型直接把 context hidden 的 K/V 投影结果写入 KV cache。
- 返回 `model_kwargs = {input_ids, positions, inputs_embeds=None}`：仅 query 路径走 forward。

### dummy_run（行 201）

覆盖基类：

- 仅一次 forward（无 multi-pass）。
- DFlash 用 context states 作为 unpadded metadata，所以 `hidden_states` 用 unpadded `num_tokens` 而非 padded `num_input_tokens`。
- max_query_tokens 较小（仅 spec tokens）。
- multimodal 不支持。

### _get_eagle3_use_aux_hidden_state_from_config（行 304）

```python
@override
def _get_eagle3_use_aux_hidden_state_from_config(self):
    return self.dflash_config.get("use_aux_hidden_state", True)
```

复用 EAGLE3 的 aux hidden 收集流程，由 GPUModelRunner 通过 `use_aux_hidden_state_outputs=True` 启用。

### build_per_group_and_layer_attn_metadata（行 287）

覆盖基类调 super，但在 `dflash_causal=False` 时 assert 每层 attention metadata 的 `causal is False`：

```python
if not self.dflash_causal:
    for layer_name, attn_metadata in per_layer.items():
        assert getattr(attn_metadata, "causal", None) is False, (
            f"Attention metadata for layer {layer_name} does not have"
            " non-causal support, which is required for DFlash."
            " Consider using a different attention backend, e.g FlashAttention."
        )
```

DFlash non-causal 需要支持 bidirectional 的 attention backend（FlashAttention 等）。

### _create_draft_vllm_config（行 76）

覆盖基类：

- 清除 `is_mm_prefix_lm`（drafter 是 text-only）。
- `use_non_causal = not self.dflash_causal`：让 attention_config 知道是否走 non-causal。

### dflash_config property（行 307）

```python
@property
def dflash_config(self):
    return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
```

直接从 hf_config 取 `dflash_config` dict，缺失时返回空 dict。

### 关键 buffer 布局

- `_context_slot_mapping_buffer` / `_context_positions_buffer`：`max_num_tokens` 大小，存 context 部分的 slot / position。
- `_slot_mapping_buffer` / `positions`：`max_query_tokens` 大小，仅 query 部分。
- `input_ids`：`max_num_tokens` 大小，但仅前 `num_query_total` 有效。
- `_dflash_hidden_states`：临时存 target hidden states 引用，`build_model_inputs_first_pass` 用。

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：基类流程；DFlash 走 `parallel_drafting=True` 早返回路径（基类行 619）。
- [eagle.md](eagle.md)：aux hidden 收集流程复用；`use_aux_hidden_state_outputs = True` 由 GPUModelRunner 设置（`gpu_model_runner.py:620`）。
- [dspark.md](dspark.md)：DSpark 是 DFlash 的 DSV4 特化，但 drafter 类在 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` 而非本目录。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：`spec_config.use_dflash()` 触发实例化；drafter 不走 `_combine_outputs_with_spec_tokens` 的 EAGLE 风格路径。
- [注意力后端](../../05-attention/README.md)：non-causal attention 需要 FlashAttention 等支持 bidirectional 的 backend。
- [模型库-DFlash 架构](../../04-model-zoo/README.md)（待补充）：`DFlashQwen3ForCausalLM` / `DFlashLagunaForCausalLM` 的 `precompute_and_store_context_kv` 方法位置。

## 历史版本演进

- **v0.11.0**：DFlashProposer landfall；`max_query_tokens` / `max_positions` 重新设计以容纳 context + query 分离；`copy_and_expand_dflash_inputs_kernel` 落地。
- **v0.11.5**：`non-causal` 支持 multiline attention backend（FlashAttention）fallback；`dflash_causal` 字段入 config。
- **v0.12 / main**：`precompute_and_store_context_kv` 接口稳定；与 EAGLE3 aux hidden path 合并以共享 `use_aux_hidden_state_outputs` infra；与 DSpark 的 DSV4 特化分家。

[← 返回投机解码](../README.md)

## 参见

- [eagle.md](eagle.md)：aux hidden 复用
- [dspark.md](dspark.md)：DFlash 的 DSV4 特化
- [llm-base-proposer.md](llm-base-proposer.md)
- [../rejection-sampler.md](../rejection-sampler.md)：并行 drafting 走 `parallel_drafting=True` 路径，rejection kernel 仍按 `cu_num_draft_tokens` 切分
