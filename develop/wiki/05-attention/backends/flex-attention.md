# FlexAttention backend

[← Wiki 首页](../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > FlexAttention

> 源码：`vllm/v1/attention/backends/flex_attention.py`

## 是什么

`FlexAttentionBackend`（`flex_attention.py:85`）封装 PyTorch 2.5+ 的 `torch.nn.attention.flex_attention`，通过 `BlockMask` 描述任意稀疏/因果/双向掩码。三件套：`FlexAttentionBackend`、`FlexAttentionMetadataBuilder`、`FlexAttentionImpl`，外加 `FlexAttentionMetadata`（含 `block_mask`）。

支持 `EncoderOnlyAttentionSpec`（无 paged KV cache）与标准 paged decoder 两种模式。

## 为什么

- **任意 mask**：FlexAttention 用 `score_mod` / `mask_mod` + `BlockMask` 表达任意注意力掩码，是 PrefixLM 多模态双向、文档级因果、sliding window 等复杂 mask 的统一解。
- **torch.compile 友好**：`create_block_mask` 与 `flex_attention` 都 `torch.compile`（`flex_attention.py:50-53`），可融合进 compile 图。
- **CUDA graph ALWAYS**：`_cudagraph_support = AttentionCGSupport.ALWAYS`（`flex_attention.py:851`）。
- **ViT/encoder 路径**：`EncoderOnlyAttentionSpec` 时跳过 paged KV cache，直接 ragged attention，适合视觉模型。

## 怎么做

### 能力要点

| 项 | 值 | 位置 |
|----|----|------|
| `supports_attn_type` | 全四种 | `flex_attention.py:108` |
| block sizes | `MultipleOf(1)` | `flex_attention.py:156` |
| `_cudagraph_support` | `ALWAYS` | `flex_attention.py:851` |
| KV cache 形状 | 标准 NHD/HND | `flex_attention.py:126` |
| encoder-only | `uses_paged_kv = not isinstance(kv_cache_spec, EncoderOnlyAttentionSpec)` | `flex_attention.py:1089` |

### BlockMask 构造

builder 的核心是构造 `BlockMask`：

- `_build_block_mask_direct`（`flex_attention.py:686`）—— 用 `BlockMask.from_kv_blocks` 高效构造，去重 block 索引。
- `build_block_mask`（`flex_attention.py:818`）—— 主入口。
- `physical_to_logical_mapping`（`flex_attention.py:161`）/ `unique_static_unsorted`（`flex_attention.py:269`）—— 把物理 block 映射到逻辑、去重。
- `causal_mask_mod` / `bidirectional_mask_mod`（`flex_attention.py:323/329`）—— mask 函数。
- `BlockSparsityHint`（`flex_attention.py:341`）—— 静态稀疏 hint。
- `copy_to_persistent`（`flex_attention.py:356`）—— 持久化 buffer。

### forward

`FlexAttentionImpl.forward`（`flex_attention.py:1257`）调编译后的 `flex_attention_compiled`，传入 `BlockMask`。`get_kernel_options`（`flex_attention.py:1395`）按配置选 kernel 选项。

### spec decode 与 KV sharing

`make_kv_sharing_fast_prefill_common_attn_metadata` / `create_fast_prefill_custom_backend`（来自 `utils.py`）支持 KV sharing fast prefill（KV cache 跨层共享时的快速 prefill）。

## 与其它模块/系统配合

- **utils**：`split_decodes_and_prefills`、`PerLayerParameters`、KV sharing fast prefill，见 [utils](utils.md)。
- **selector**：CUDA 非 MLA 末位优先级（fallback for 复杂 mask），见 [selector](../selector.md)。
- **ViT**：`platforms/cuda.py:get_supported_vit_attn_backends` 把 `FLEX_ATTENTION` 列为 ViT 可选 backend。
- **torch.compile**：`recompile_limit=16`（`flex_attention.py:49`）允许较多 mask 重编译。
- **PrefixLM**：`bidirectional_mask_mod` 服务多模态 PrefixLM。

## 历史版本演进

- **v0.7.x**：`FlexAttentionBackend` 引入（PyTorch 2.5 flex_attention）。
- **v0.8.x**：`BlockMask.from_kv_blocks` 直构优化；`EncoderOnlyAttentionSpec` 支持；`ALWAYS` cudagraph。
- **v0.9.x**：`BlockSparsityHint` 静态稀疏；KV sharing fast prefill；`physical_to_logical_mapping` 去重。
- **v0.10.x（当前）**：`get_kernel_options`；torch.compile 容忍 `recompile_limit=16`；多模态前缀双向 mask 完善。

---

[← 返回注意力首页](../../README.md)

## 参见

- [FlashAttention backend](flash-attn.md)
- [Triton backend](triton.md)
- [Utils](utils.md)
- [selector](../selector.md)
