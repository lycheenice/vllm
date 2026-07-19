# FlashAttn MLA backend

[← Wiki 首页](../../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > [← MLA 首页](../README.md) > FlashAttn MLA

> 源码：`vllm/v1/attention/backends/mla/flashattn_mla.py`、`flashattn_mla_sparse.py`、`prefill/flash_attn.py`

## 是什么

基于 FlashAttention 的 MLA backend，覆盖 Hopper（SM90）dense + sparse 与 prefill：

- `FlashAttnMLABackend` / `FlashAttnMLAImpl`（`flashattn_mla.py:43/264`）—— Hopper dense MLA，用 `vllm_flash_attn.flash_attn_varlen_func`。
- `FlashAttnMLASparseBackend` / `FlashAttnMLASparseImpl`（`flashattn_mla_sparse.py:32/192`）—— DeepSeek V4 sparse MLA（Hopper），继承 `SparseMLAAttentionImpl`。
- `FlashAttnPrefillBackend`（`prefill/flash_attn.py:40`）—— MLA prefill 专用，Hopper/Blackwell 默认最高优先级，含 FA4 CuTeDSL warmup。

## 为什么

- FlashAttention 3/4 在 Hopper/Blackwell 的 MLA prefill 与 dense decode 上是开源最快之一。
- `FlashAttnMLABackend` 在 Hopper MLA 优先级**第一**（`platforms/cuda.py:136`）。
- `supports_combination` 委托 `fa_utils.flash_attn_supports_mla()` gating。
- FA4 prefill（CuTeDSL）支持 `supports_quant_output` 融合 FP8/NVFP4 输出。
- sparse 版给 DeepSeek V4 Hopper 路径，与 FlashInfer/FlashMLA sparse 互补。

## 怎么做

### FlashAttnMLABackend 能力

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16 | `flashattn_mla.py:44` |
| kv dtype | auto/fp16/bf16（不支持 fp8） | `flashattn_mla.py:45` |
| block sizes | `MultipleOf(16)` | `flashattn_mla.py:52` |
| `supports_compute_capability` | `major == 9`（仅 Hopper） | `flashattn_mla.py:80` |
| `supports_batch_invariance` | `True` | `flashattn_mla.py:68` |
| cudagraph | `UNIFORM_BATCH` | `flashattn_mla.py:116` |
| stride_order | blocks 优先 `(1,0,2,3)` | `flashattn_mla.py:56` |
| `supports_combination` | `flash_attn_supports_mla()` gating | `flashattn_mla.py:84` |

`FlashAttnMLAImpl` 用 `MLACommonImpl.forward_mha` 走 prefill backend，`forward_mqa`（`flashattn_mla.py:318`）调 `flash_attn_varlen_func`（带 `get_scheduler_metadata`）。

### FlashAttnMLASparse（DeepSeek V4）

| 项 | 值 | 位置 |
|----|----|------|
| `get_name` | `FLASH_ATTN_MLA_SPARSE` | `flashattn_mla_sparse.py` |
| `is_sparse()` | True | — |
| cudagraph | `UNIFORM_BATCH` | `flashattn_mla_sparse.py:135` |
| 继承 | `SparseMLAAttentionImpl`（仅 `forward_mqa`） | `flashattn_mla_sparse.py:192` |

`FlashAttnMLASparseImpl.forward_mqa`（`flashattn_mla_sparse.py:237`）用 indexer 选出的 top-k 页构造 ragged KV，调 FA varlen。

### FlashAttnPrefillBackend

`prefill/flash_attn.py:40`：

- `is_available` 检 `is_flash_attn_varlen_func_available()`。
- 支持 `supports_quant_output`（FA4 fused FP8/NVFP4 写出）。
- FA4 CuTeDSL warmup：`iter_fa4_mla_prefill_compile_requests` + `FA4MLAPrefillCompileContext`（`prefill/flash_attn.py:18-22`）。
- `prefill/selector.py`：Hopper 只选它；Blackwell 第一优先级。

## 与其它模块/系统配合

- **fa_utils**：`flash_attn_supports_mla`、`get_flash_attn_version`，见 [Utils](../utils.md)。
- **ops**：间接用 `flash_attn_varlen_func`；sparse 用 `mla/sparse_utils.py`、`mla/indexer.py`。
- **selector**：Hopper MLA 第一优先级；sparse 表中 `FLASH_ATTN_MLA_SPARSE`，见 [selector](../../selector.md)。
- **MLA 公共层**：见 [MLA 总览](../README.md)。
- **FA4 warmup**：`vllm.model_executor.warmup.cutedsl_warmup` + `fa4_cutedsl_config`。

## 历史版本演进

- **v0.8.x**：`FlashAttnMLABackend` 引入（Hopper MLA），复用 `MLACommonImpl`。
- **v0.9.x**：`FlashAttnMLASparseBackend`（DeepSeek V4）；FA3 prefill；`flash_attn_supports_mla` gating。
- **v0.10.x（当前）**：FA4 CuTeDSL prefill warmup（Blackwell）；`prefill/flash_attn.py` 独立化；`supports_quant_output` fused FP8/NVFP4 写出。

---

[← 返回 MLA 首页](../README.md)

## 参见

- [FlashAttention 标准 backend](../flash-attn.md)
- [MLA 总览](../README.md)
- [MLA FlashInfer](flashinfer.md)
- [Utils](../utils.md)
