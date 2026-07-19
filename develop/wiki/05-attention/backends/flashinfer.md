# FlashInfer backend

[← Wiki 首页](../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > FlashInfer

> 源码：`vllm/v1/attention/backends/flashinfer.py`

## 是什么

`FlashInferBackend` 封装 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 库，提供 NVIDIA Blackwell（SM100）优先的 prefill/decode 实现，并支持 FP8 / NVFP4 KV cache。它用三个 FlashInfer wrapper：

- `BatchPrefillWithRaggedKVCacheWrapper` —— ragged prefill。
- `BatchPrefillWithPagedKVCacheWrapper` —— paged prefill / chunked prefill。
- `BatchDecodeWithPagedKVCacheWrapper` —— paged decode。
- `MultiLevelCascadeAttentionWrapper` —— cascade。
- `trtllm_batch_context_with_kv_cache` / `trtllm_batch_decode_with_kv_cache` / `fast_decode_plan` —— Blackwell trtllm-gen 路径。

主类：`FlashInferBackend`、`FlashInferMetadataBuilder`、`FlashInferImpl`、`FlashInferMetadata`。

## 为什么

- **Blackwell 最优**：trtllm-gen kernel 在 SM100 上常比 FA 更快，且支持大 page（128/256/512/1024）。
- **NVFP4 KV cache**：FlashInfer 原生 `FP4Tensor`，vLLM 据此支持 `nvfp4` KV dtype。
- **综合能力**：cascade、FP8、大 page、split decode/prefill 一站式。
- **性能自适应**：`use_cascade_attention` 与 `split_decodes_and_prefills` 让同一 batch 内 decode/prefill 用不同 plan。

## 怎么做

### 能力要点

| 项 | 值 | 位置 |
|----|----|------|
| `supports_non_causal` | `True` | `flashinfer.py:369` |
| block sizes | 默认 `[16,32,64]`；GQA+SM100+trtllm 可用则扩到 `[128,256,512,1024]` | `flashinfer.py:343` |
| KV cache 形状 | `(num_blocks, 2, block_size, num_kv_heads, head_size)`；nvfp4 用 packed `nvfp4_kv_cache_full_dim` | `flashinfer.py:381` |
| stride_order | 随 `NHD`/`HND` layout 变（含 num_layers 维度时 blocks 优先） | `flashinfer.py:395` |
| kv dtype | auto/fp16/bf16/fp8/e4m3/e5m2/nvfp4 | `flashinfer.py:427` |
| `supports_sink` | trtllm 路径支持 sink | `flashinfer.py:453` |
| cudagraph | 动态 `get_cudagraph_support` | `flashinfer.py:851` |

### trtllm-gen 路径选择

`vllm/utils/flashinfer.py` 的 `can_use_trtllm_attention` / `use_trtllm_attention` / `force_use_trtllm_attention` 决定是否切到 trtllm-gen kernel（Blackwell GQA/MQA）。workspace buffer 由 `_get_trtllm_workspace_buffer`（`flashinfer.py:94`）按 `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` 分配。

### FP8 KV cache dequant kernel

`_trtllm_prefill_attn_kvfp8_dequant`（`flashinfer.py:103`）是内嵌 Triton kernel，把 fp8 KV cache dequant 成 mock bf16 cache 供 prefill。

### DCP

与 FlashAttention 一样接 `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce` 做跨 rank LSE 归约（`flashinfer.py:74-75` import）。

### 自适应 split

`split_decodes_and_prefills`（来自 `utils.py`）把混合 batch 拆成 decode 段与 prefill 段，分别 plan/forward。`infer_global_hyperparameters` 推断 batch 级参数。

## 与其它模块/系统配合

- **ops**：`merge_attn_states`（cascade）、`cp_lse_ag_out_rs`/`dcp_a2a_lse_reduce`（DCP），见 [ops](../ops.md)。
- **utils**：`get_kv_cache_layout`、`split_decodes_and_prefills`、`get_per_layer_parameters`、`infer_global_hyperparameters`，见 [utils](utils.md)。
- **selector**：Blackwell 非 MLA 第一优先级、MLA sparse FP8 KV 第一优先级，见 [selector](../selector.md)。
- **MLA**：`mla/flashinfer_mla.py` + `mla/flashinfer_mla_sparse*.py` 是 FlashInfer 的 MLA 变体，走 `MLACommonImpl` / `SparseMLAAttentionImpl`，见 [MLA FlashInfer](mla/flashinfer.md)。
- **量化**：NVFP4 用 `vllm.utils.torch_utils.nvfp4_kv_cache_*`；FP8 走 `is_quantized_kv_cache`。
- **环境变量**：`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`。

## 历史版本演进

- **v0.7.x**：`FlashInferBackend` 引入，prefill/decode wrapper。
- **v0.8.x**：cascade（`MultiLevelCascadeAttentionWrapper`）；FP8 KV cache；trtllm-gen prefill context。
- **v0.9.x**：trtllm-gen decode（`trtllm_batch_decode_with_kv_cache` + `fast_decode_plan`）；NVFP4 KV cache；DCP A2A；大 page 支持（128/256/512/1024）；split decode/prefill。
- **v0.10.x（当前）**：动态 block sizes 表（按 SM100+GQA 探测大 page）；`_trtllm_prefill_attn_kvfp8_dequant` Triton dequant；MLA sparse SM120 变体（在 `mla/`）。

---

[← 返回注意力首页](../../README.md)

## 参见

- [FlashAttention backend](flash-attn.md)
- [Triton backend](triton.md)
- [MLA FlashInfer](mla/flashinfer.md)
- [底层 ops](../ops.md)
- [Utils](utils.md)
