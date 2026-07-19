[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > Gemma4Proposer

# Gemma4Proposer（Gemma4 MTP）

> 源码：`vllm/v1/spec_decode/gemma4.py`

---

## 是什么

`Gemma4Proposer` 是 Google Gemma4 模型的 MTP drafter 实现继承 `SpecDecodeBaseProposer`，与 Step3.5 同样支持多 KV cache group（sliding vs full attention），但独特之处：

- **`constant_draft_positions=True`**：所有 K 个 draft step 都从 target 末位置 predict，不前进 positions。drafter 自己的 KV cache 也只用单 position。
- **跨模型 KV 共享**：每个 draft decoder layer 映射到 target 中"同型（sliding/full）的最后一个非 KV-shared 层"，复用其 KV cache。`_setup_gemma4_kv_sharing` 完成此映射。
- **`model_returns_tuple=True`**：drafter forward 返回 `(draft_hidden, backbone_hidden)`，前者用于采样本 token，后者作为下一步 hidden_states 输入。
- **centroids CUDA graph**：当模型启用 centroids masking（`masked_embedding` 存在）时，预捕获 batch size ∈ {1,2,4,8,16,32,64} 的 CUDA graph 加速 `get_top_tokens`。
- **不共享 lm_head**：Gemma4 draft 的 lm_head 维度（draft hidden）与 target backbone 维度不同，强制 drafter 保留自己的 lm_head。

类签名（`vllm/v1/spec_decode/gemma4.py:31`）：

```python
class Gemma4Proposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device,
                         pass_hidden_states_to_model=True, runner=runner)
        self.constant_draft_positions = True
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        ...
```

## 为什么

- **Gemma4 架构特性**：Gemma4 的 MTP layer 是 Q-only attention——Drafter 不需要真正的因果 attention，只查询 target 末位置的 KV。这就让 positions 与 seq_lens 在 K 步 drafting 中保持不变，简化 attention metadata build。
- **跨模型 KV 共享**：Gemma4 的 draft layer 与 target 同型的最后一层用同一 block_table / KV slot，避免在 drafter 内分配独立 KV cache。这节省显存并让 drafter forward 极轻。
- **sliding + full 双 head dim**：Gemma4 有 sliding attention（head_dim=256）与 full attention（global_head_dim=512）两种类型，每种 head dim 都需要一个 `AttentionGroup`；不能用基类的单 group 假设。
- **centroids masking**：当 `use_ordered_embeddings=True` 时，drafter 用 centroid-based nearest-neighbor 选 token；`get_top_tokens` 是个性能热点，预捕 CUDA graph 加速。
- **强制 TRITON_ATTN**：Gemma4 因 head_dim 异构强制 `TRITON_ATTN` backdrafter 端也必须继承，否则 sliding layer fallback 到 FLASH_ATTN 会因 head_dim 不匹配 fail。`_create_draft_vllm_config` 覆盖以保留 target 的 attn backend。

## 怎么做

### model_returns_tuple（行 64）

```python
def model_returns_tuple(self) -> bool:
    return True
```

drafter forward 返回 `(draft_hidden_states, backbone_hidden_states)`，与 `compute_logits(hidden_states)` 和 `hidden_states[token_indices_to_sample]` 的双路径对应。基类 `propose` 行 591–595 已处理 tuple 拆分。

### build_per_group_and_layer_attn_metadata（行 70）

覆盖基类，与 Step3.5 类似。差异：

- Gemma4 不需要 per-group slot_mapping buffer（draft 层共享 target 的 slot）。
- 仅 `block_table_tensor` 按 group 切片：`cm.block_table_tensor = self._per_group_block_tables[gid][:batch_size]`。

### _greedy_sample（行 104）

centroids CUDA graph 路径：

```python
def _greedy_sample(self, hidden_states):
    if self._centroids_sizes:
        T = hidden_states.shape[0]
        for size in self._centroids_sizes:
            if size >= T:
                self._centroids_inputs[size][:T].copy_(hidden_states)
                self._centroids_graphs[size].replay()
                return self._centroids_outputs[size][:T].clone()
        return self.model.get_top_tokens(hidden_states)
    return super()._greedy_sample(hidden_states)
```

找到 ≥ T 的最小 graph size，replay 后取 `[:T]` 切片。fallback 走原 `get_top_tokens`。

### _setup_centroids_cuda_graphs（行 115）

为 batch size ∈ {1,2,4,8,16,32,64} 各捕获一个 CUDA graph：

```python
for size in [1, 2, 4, 8, 16, 32, 64]:
    static_input = torch.zeros(size, masked_emb.hidden_size, ...)
    # warmup 3 次
    for _ in range(3):
        masked_emb.get_top_tokens(static_input, lm_head_weight)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = masked_emb.get_top_tokens(static_input, lm_head_weight)
    self._centroids_graphs[size] = g
    ...
```

`_centroids_sizes` 排序保证查找最快（升序）；`_centroids_inputs` / `_centroids_outputs` 按 size 索引。

### _create_draft_vllm_config（行 147）

```python
def _create_draft_vllm_config(self):
    base = super()._create_draft_vllm_config()
    target_backend = self.vllm_config.attention_config.backend
    if target_backend is not None:
        base = replace(base, attention_config=replace(base.attention_config, backend=target_backend))
    return base
```

基类默认把 draft 的 `attention_config.backend` 置 None 让 draft layer 自动选；Gemma4 强制继承 target 的 `TRITON_ATTN`，因为 sliding layer 在 FLASH_ATTN 下无法处理 KV-shared cache。

### _maybe_share_lm_head（行 168）

显式覆盖为 no-op，并 log "keeping draft model's own lm_head"draft hidden_size ≠ backbone hidden_size 时 sharing 会 break compute_logits 与 centroids masking。

### _setup_gemma4_kv_sharing（行 280）

在 `load_model` 末尾调：

1. 从 target `text_config.layer_types` 取每种 attention 类型的 layer index（排除最后 `num_kv_shared_layers` 个共享层）：`type_to_target_indices["full_attention"] = [0, 2, 4, ...]`，`type_to_target_indices["sliding_attention"] = [1, 3, 5, ...]`。
2. 找到 target attn layer name prefix（如 `model.layers`）。
3. 对每个 draft layer（按 index 顺序）：取其 `layer_types[draft_idx]` 决定 attention 类型，找该类型在 target 中最后一个非共享 layer index `target_idx`。
4. 设置 `draft_layer.self_attn.attn.kv_sharing_target_layer_name = f"{target_prefix}.{target_idx}.self_attn.attn"`。

### initialize_attn_backend（行 200）

覆盖基类以多 group：

- 按 `(backend.full_cls_name(), kv_cache_spec)` 分组（spec 含 head_dim / sliding 等信息，区分不同 group）。
- 每 group 独立 `AttentionGroup` 与 `create_metadata_builders`。
- `validate_same_kv_cache_group` 被覆盖为 no-op（行 195）。

### validate_same_kv_cache_group（行 195）

```python
def validate_same_kv_cache_group(self, kv_cache_config):
    """Draft layers span multiple KV cache groups (sliding + full
    attention with different head dimensions), so skip the base
    class single-group assertion."""
```

## 与其它模块/系统配合

- [llm-base-proposer.md](llm-base-proposer.md)：流程主体复用基类，差异在 attn metadata 与 sampling 路径。
- [step3p5.md](step3p5.md)：同样多 KV cache group，但 Step3.5 不用 `constant_draft_positions`，更接近 EAGLE。
- [eagle.md](eagle.md)：Gemma4 + EAGLE3 `combine_hidden_states` 在 `_get_eagle3_use_aux_hidden_state_from_config` 中被基类调用。
- [编译与 IR](../../09-compilation-ir/README.md)：centroids CUDA graph 在 `load_model` 阶段捕获；`torch.accelerator.synchronize` 与 `torch.cuda.graph` 上下文管理。
- [注意力后端](../../05-attention/README.md)：TRITON_ATTN 强制；`build_for_drafting` 调用；多 group attention metadata builder。
- [模型库-Gemma4](../../04-model-zoo/README.md)（待补充）：`Gemma4ForConditionalGeneration` / `Gemma4UnifiedForConditionalGeneration` 模型实现。
- [执行层-GPUModelRunner](../../02-execution/worker/README.md)：ModelRunner 在 `_prepare_inputs` 调 `set_per_group_block_table` 注入每 gid 的 block_table。

## 历史版本演进

- **v0.10.5**：Gemma4 模型 landfall，`Gemma4Proposer` 同步引入；`constant_draft_positions` 字段加入。
- **v0.10.5+**：centroids CUDA graph 加入，针对 `use_ordered_embeddings` 场景显著降低 `get_top_tokens` 延迟。
- **v0.11.0**：`_setup_gemma4_kv_sharing` 完善：处理 `num_kv_shared_layers`、target prefix 自动检测。
- **v0.12 / main**：`_create_draft_vllm_config` 显式保留 target attn backend，修复 sliding layer fallback bug。

[← 返回投机解码](../README.md)

## 参见

- [step3p5.md](step3p5.md)
- [llm-base-proposer.md](llm-base-proposer.md)
- [eagle.md](eagle.md)
