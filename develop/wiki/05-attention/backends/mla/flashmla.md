# FlashMLA backend

[← Wiki 首页](../../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > [← MLA 首页](../README.md) > FlashMLA

> 源码：`vllm/v1/attention/backends/mla/flashmla.py`、`flashmla_sparse.py`；ops：`vllm/v1/attention/ops/flashmla.py`

## 是什么

封装 DeepSeek 官方 [FlashMLA](https://github.com/deepseek-ai/FlashMLA) C++ 扩展（`vllm._flashmla_C` + `vllm._flashmla_extension_C`）的 MLA backend：

- `FlashMLABackend` / `FlashMLAImpl`（`flashmla.py:47/216`）—— dense MLA decode，支持 Hopper（SM90）与 Blackwell DC（SM100）。
- `FlashMLASparseBackend` / `FlashMLASparseImpl`（`flashmla_sparse.py:90/540`）—— DeepSeek V4 sparse MLA，继承 `SparseMLAAttentionImpl`。

继承 `MLACommonBackend`/`MLACommonImpl`，prefill 走公共 `forward_mha` + prefill backend。

## 为什么

- FlashMLA 是 DeepSeek 官方为 MLA decode 优化的 kernel，Hopper 上业界领先。
- 支持 FP8 KV cache（`fp8` / `fp8_ds_mla`，`flash_mla_with_kvcache_fp8`）。
- dense 支持 SM90 与 SM100 DC；sparse 扩展到 DeepSeek V3.2/V4。
- `FlashMLASchedMeta` + `get_mla_metadata` 计算 `num_kv_splits` / tile scheduler metadata。
- spec decode：`reshape_query_for_spec_decode` / `reshape_attn_output_for_spec_decode`（`flashmla.py:33`）。

## 怎么做

### FlashMLABackend（dense）能力

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16 | `flashmla.py:48` |
| kv dtype | auto/fp16/bf16/fp8/fp8_e4m3 | `flashmla.py:49` |
| block sizes | `[64]` | `flashmla.py:58` |
| `supports_compute_capability` | `major in [9, 10]` | `flashmla.py:82` |
| cudagraph | `UNIFORM_BATCH` | `flashmla.py:119` |
| stride_order | blocks 优先 `(1,0,2,3)` | `flashmla.py:62` |
| `supports_combination` | sparse 时探 `is_flashmla_sparse_supported` | `flashmla.py:86` |

`FlashMLAMetadata`（`flashmla.py:114`）含 `FlashMLADecodeMetadata`（`FlashMLASchedMeta`、`scheduler_metadata`、`tile_scheduler_metadata`）。

`FlashMLAImpl.forward_mqa`（`flashmla.py:266`）：

- `VLLM_BATCH_INVARIANT` 路径手工构造 tile_scheduler_metadata（单 partition），见 `flashmla.py:287-315`。
- FP8 KV 走 `flash_mla_with_kvcache_fp8`，否则 `flash_mla_with_kvcache`。
- `attn_type` 必须 DECODER。

### FlashMLASparse（DeepSeek V4）

| 项 | 值 | 位置 |
|----|----|------|
| dtype | 仅 bf16 | `flashmla_sparse.py:91` |
| kv dtype | auto/bf16/`fp8_ds_mla`/`fp8` | `flashmla_sparse.py:92` |
| block sizes | `[64]` | `flashmla_sparse.py:100` |
| `is_mla()` / `is_sparse()` | True | `flashmla_sparse.py:121/125` |
| head sizes | `[576]`（512 NoPE + 64 RoPE，DeepSeek V3.2） | `flashmla_sparse.py:116` |
| cudagraph | `UNIFORM_BATCH` | `flashmla_sparse.py:232` |

FP8 sparse 有两种模式（`flashmla_sparse.py:53-65` 注释）：

1. **Mixed batch**：FP8 decode kernel 同时处理 prefill+decode（低 head 数/TP 时用，省 BF16 prefill 的 head padding）。
2. **Separate**：BF16 prefill kernel + FP8 decode kernel（高 head 数用）。

`FlashMLASparseImpl.forward_mqa`（`flashmla_sparse.py:857`）调 `flash_mla_sparse_fwd`，用 indexer 选出的 top-k 页。`get_prefill_workspace_size`（`flashmla_sparse.py:221`）算 prefill workspace。

### ops/flashmla.py 探测

`_is_flashmla_available`（`ops/flashmla.py:33`）检查 `vllm._flashmla_C` + `vllm._flashmla_extension_C` 编译成功；`is_flashmla_dense_supported`（Hopper only）/ `is_flashmla_sparse_supported`（Hopper+Blackwell DC）。

## 与其它模块/系统配合

- **ops**：`ops/flashmla.py`，见 [ops](../../ops.md)。
- **indexer**：sparse 用 `mla/indexer.py` 选页，见 [MLA 总览](../README.md)。
- **selector**：Hopper MLA 第 2 优先级（FLA MLA 后）；Blackwell sparse BF16 + 高 head 数第一，见 [selector](../../selector.md)。
- **workspace manager**：sparse 用 `current_workspace_manager`，见 [执行层-Worker](../../../02-execution/worker/README.md)。
- **VLLM_BATCH_INVARIANT**：dense 走单 partition tile scheduler 路径。

## 历史版本演进

- **v0.8.x**：`FlashMLABackend` 引入（DeepSeek 官方 FlashMLA C++），Hopper dense decode。
- **v0.9.x**：FP8 KV cache（`flash_mla_with_kvcache_fp8`）；`FlashMLASparseBackend`（DeepSeek V4，`flash_mla_sparse_fwd`）；mixed vs separate prefill/decode 模式；`VLLM_BATCH_INVARIANT` 路径。
- **v0.10.x（当前）**：`_flashmla_extension_C` 探测；sparse head_size 576（DeepSeek V3.2 layout）；prefill workspace；stride_order blocks 优先。

---

[← 返回 MLA 首页](../README.md)

## 参见

- [MLA 总览](../README.md)
- [MLA FlashAttn](flashattn.md)
- [MLA FlashInfer](flashinfer.md)
- [底层 ops](../../ops.md)
- [模型库-DeepSeek](../../../04-model-zoo/architecture-families/deepseek.md)
