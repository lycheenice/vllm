# AITER MLA backend

[← Wiki 首页](../../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > [← MLA 首页](../README.md) > AITER MLA

> 源码：`vllm/v1/attention/backends/mla/rocm_aiter_mla.py`、`aiter_triton_mla.py`、`rocm_aiter_mla_sparse.py`、`prefill/aiter_flash_attn.py`

## 是什么

ROCm 平台的 MLA backend 全家桶，基于 AMD [AITER](https://github.com/ROCm/aiter) 库：

- `AiterMLABackend` / `AiterMLAImpl`（`rocm_aiter_mla.py:54/685`）—— dense MLA decode（AITER kernel）+ prefill（flash_attn / gfx950 FP8 MLA）。
- `AiterTritonMLABackend` / `AiterTritonMLAImpl`（`aiter_triton_mla.py:6/16`）—— 继承 `AiterMLABackend`，prefill 改用 aiter Triton MHA（`aiter.ops.triton.mha.flash_attn_varlen_func`）。
- `ROCMAiterMLASparseBackend` / `ROCMAiterMLASparseImpl`（`rocm_aiter_mla_sparse.py:265/628`）—— DeepSeek V4 sparse MLA decode。
- `AiterFlashAttnPrefillBackend`（`prefill/aiter_flash_attn.py:23`）—— MLA prefill 专用，调 `aiter.flash_attn_varlen_func`，原生支持 qk≠v head dim。

## 为什么

- ROCm（MI300X gfx942 / MI325 gfx950）无 NVIDIA FlashAttention/CUTLASS，必须用 AMD AITER kernel。
- AITER MLA decode kernel 内部 page_size=1，wrapper 把 block 层索引展平成 per-token（`rocm_aiter_mla.py:71-75` 注释），故支持任意 `MultipleOf(1)` block。
- gfx950 上有 FP8 MLA prefill 专用 kernel（`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`），由 `_fp8_mla_prefill_supported()`（`rocm_aiter_mla.py:35`）自动探测，缺失则回退 `flash_attn_varlen_func`。
- sparse 版给 DeepSeek V4 C4A/C128A 层用，配套 `mla/indexer.py` 选页。

## 怎么做

### AiterMLABackend 能力

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16 | `rocm_aiter_mla.py:55` |
| kv dtype | auto/fp16/bf16/fp8/e4m3/e5m2 | `rocm_aiter_mla.py:56` |
| block sizes | `MultipleOf(1)`（内部展平） | `rocm_aiter_mla.py:70` |
| cudagraph | `UNIFORM_BATCH` | `rocm_aiter_mla.py:141` |
| head sizes | `[]`（任意，受 MLA dims 约束） | `rocm_aiter_mla.py:66` |

`AiterMLAMetadata`（`rocm_aiter_mla.py:110`）含 `AiterMLADecodeMetadata`（paged_kv_indptr/indices/page_sizes 等 AITER 专属布局）。`AiterMLAHelper`（`rocm_aiter_mla.py:637`）做转换辅助。

`AiterMLAImpl.forward_mha`（`rocm_aiter_mla.py:845`）/ `forward_mqa`（`rocm_aiter_mla.py:911`）实现双路径。

### AiterTritonMLA 差异

`AiterTritonMLAImpl`（`aiter_triton_mla.py:16`）只重写 `_flash_attn_varlen_diff_headdims`（`aiter_triton_mla.py:49`）：用 aiter Triton MHA 替换 FA 做不同 qk/v head dim 的 prefill，并转置 LSE。decode 路径完全复用 `AiterMLAImpl`。

### Sparse AITER MLA

`ROCMAiterMLASparseBackend`（`rocm_aiter_mla_sparse.py:265`）继承 `SparseMLAAttentionImpl`，block `MultipleOf(1)`，复用 `AiterMLAHelper`。含多个内嵌 Triton kernel（`_convert_req_index_to_global_index_kernel`、`generate_sparse_seqlen_kernel`、`fetch_id_to_ragged_kernel`）做 sparse index 构造。

### Aiter prefill backend

`AiterFlashAttnPrefillBackend`（`prefill/aiter_flash_attn.py:23`）：

- `supports_compute_capability` 仅 ROCm MI3xx（`on_mi3xx()`）。
- `is_available` 检查 `rocm_aiter_ops`。
- 直接调 `aiter.flash_attn_varlen_func`，原生支持 qk headdim 192 + v headdim 128，无需 padding V。

`prefill/selector.py` 在 ROCm 下优先级：`ROCM_AITER_FA → FLASH_ATTN`。

## 与其它模块/系统配合

- **ops**：`ops/rocm_aiter_mla_sparse.py`（ops 侧）、`ops/triton_fp8_mqa_logits.py`（gfx942 fallback），见 [ops](../../ops.md)。
- **indexer**：sparse 版用 `mla/indexer.py` 选 top-k 页，见 [MLA 总览](../README.md)。
- **selector**：ROCm `_get_backend_priorities`（`platforms/rocm.py:407`）按 `rocm_aiter_ops.is_mla_enabled()` gating，见 [selector](../../selector.md)。
- **MLA 公共层**：继承 `MLACommonBackend`/`MLACommonImpl`/`SparseMLAAttentionImpl`，见 [MLA 总览](../README.md)。
- **gfx950**：MI325/MI350 的 FP8 MLA prefill 自动探测。

## 历史版本演进

- **v0.8.x**：`AiterMLABackend` 引入（ROCm MLA decode），gfx942 为主。
- **v0.9.x**：gfx950 FP8 MLA prefill（`mla_prefill_ps_asm_fwd`）；`ROCMAiterMLASparseBackend`（DeepSeek V4）；AITER 版本 pin。
- **v0.10.x（当前）**：`AiterTritonMLABackend`（prefill 走 aiter Triton MHA，替换 FA，原生 diff head dim）；prefill 解耦到 `prefill/aiter_flash_attn.py`；`AiterMLAHelper` 复用。

---

[← 返回 MLA 首页](../README.md)

## 参见

- [MLA Triton](triton.md)
- [MLA 总览](../README.md)
- [ROCm backend](../rocm.md)
- [底层 ops](../../ops.md)
