# ROCm attention backend

[← Wiki 首页](../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > ROCm

> 源码：`vllm/v1/attention/backends/rocm_attn.py`、`vllm/v1/attention/backends/rocm_aiter_fa.py`、`vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

## 是什么

ROCm 平台有三套非 MLA attention backend，按优先级：

- `RocmAttentionBackend`（`rocm_attn.py:165`）—— 经典 PagedAttention（decode）+ Triton `chunked_prefill_paged_decode`（prefill），`use_cascade_attention` 支持。
- `AiterFlashAttentionBackend`（`rocm_aiter_fa.py`）—— AMD AITER FlashAttention kernel，cascade + FP8。
- `RocmAiterUnifiedAttentionBackend`（`rocm_aiter_unified_attn.py:29`）—— 继承 `RocmAttentionBackend`，用 AITER 统一注意力 kernel，BF16 only、block 64、无 cascade。

## 为什么

ROCm（MI300X/MI325 等）没有 NVIDIA flash-attn，需要 AMD 专属 kernel：

- **RocmAttentionBackend** 是稳定兜底，layout 为 `(2, num_blocks, ...)`（blocks 在第二维），与 KV connector 要求的 blocks-first 布局不兼容，故 `use_kv_connector` 时被排除（`platforms/rocm.py:432`）。
- **AITER** 提供更快的 MHA / MLA kernel，但要求 AITER 包安装且 `rocm_aiter_ops.is_mha_enabled()` / `is_mla_enabled()` 通过。
- **Unified** 是较新的 AITER 统一路径，覆盖 mm_prefix/non_causal，但不支持 cascade。

## 怎么做

### RocmAttentionBackend 要点

| 项 | 值 | 位置 |
|----|----|------|
| `forward_includes_kv_cache_update` | False（KV 写单独） | — |
| block sizes | `MultipleOf(16)` | `rocm_attn.py:181` |
| `supports_mm_prefix` / `supports_non_causal` | `True` | `rocm_attn.py:197/208` |
| `_cudagraph_support` | `ALWAYS` | `rocm_attn.py:77` |
| KV cache 形状 | `(2, num_blocks, block_size, num_kv_heads, head_size/x, x)`（blocks 非首维） | `rocm_attn.py:243` |
| cascade | `use_cascade_attention` 按 common_prefix_len 判定 | `rocm_attn.py:255` |

- prefill 走 `ops/chunked_prefill_paged_decode.py:chunked_prefill_paged_decode`（含 `has_native_kv_cache_layout` 检测）。
- decode 走 `ops/paged_attn.py:PagedAttention` + `vllm._custom_ops`。
- cascade：`use_cascade` 时 prefix 走 paged、suffix 走 ragged，`merge_attn_states` 合并。

### AiterFlashAttentionBackend 要点

| 项 | 值 | 位置 |
|----|----|------|
| `_cudagraph_support` | `UNIFORM_BATCH` | `rocm_aiter_fa.py:401` |
| `supports_sink` | True | `rocm_aiter_fa.py:714` |
| cascade | 支持 | `rocm_aiter_fa.py:706` |
| FP8 KV | 支持 | — |

内含 Triton `cp_mha_gather_cache_kernel`（`rocm_aiter_fa.py:47`）做 cascade 前缀 cache gather。`_PARTITION_SIZE_ROCM=256`、`_CP_TOKENS_PER_ITER_ROCM=32*1024`。

### RocmAiterUnifiedAttentionBackend 要点

| 项 | 值 | 位置 |
|----|----|------|
| dtype | 仅 `bfloat16` | `rocm_aiter_unified_attn.py:30` |
| block sizes | `MultipleOf(16)`，preferred 64 | `rocm_aiter_unified_attn.py:39/43` |
| head_size | ≥32 | `rocm_aiter_unified_attn.py:53` |
| `supports_mm_prefix` / `supports_non_causal` | True | `rocm_aiter_unified_attn.py:57/65` |
| cascade | 不支持（forward assert `use_cascade is False`） | `rocm_aiter_unified_attn.py:190` |
| KV dtype | auto/bf16/fp8/fp8_e4m3 | `rocm_aiter_unified_attn.py:31` |

### ROCm MLA

MLA 走独立 backend：`mla/rocm_aiter_mla.py:AiterMLABackend`、`mla/aiter_triton_mla.py:AiterTritonMLABackend`（prefill 用 aiter Triton MHA，`aiter_triton_mla.py:45`）、`mla/rocm_aiter_mla_sparse.py:ROCMAiterMLASparseBackend`。见 [MLA AITER](mla/aiter.md)。

`rocm_aiter_mla.py:_fp8_mla_prefill_supported`（line 35）自动探测 gfx950 + AITER `mla_prefill_ps_asm_fwd`/`mla_reduce_v1`，缺失则回退 `flash_attn_varlen_func`。

## 与其它模块/系统配合

- **ops**：`chunked_prefill_paged_decode`、`paged_attn`、`merge_attn_states`、`triton_reshape_and_cache_flash`，见 [ops](../ops.md)。
- **platforms/rocm.py**：`_get_backend_priorities`（`rocm.py:407`）按 use_sparse→use_mla→use_kv_connector 排序；AITER 可用性由 `rocm_aiter_ops` 控制。
- **selector**：见 [selector](../selector.md)。
- **KV connector**：`RocmAttentionBackend` 因 blocks 非首维被 KV connector 排除。
- **gfx942 fallback**：`ops/triton_fp8_mqa_logits.py` 为 MI300X 临时复刻 AITER fp8_mqa_logits。

## 历史版本演进

- **v0.6.x**：`RocmAttentionBackend`（V0 PagedAttention 迁移）。
- **v0.7.x**：AITER FlashAttention 引入；cascade；FP8 KV。
- **v0.8.x**：AITER MLA（`AiterMLABackend`）；gfx950 FP8 MLA prefill 自动探测。
- **v0.9.x**：`RocmAiterUnifiedAttentionBackend`；`chunked_prefill_paged_decode`；KV connector 布局排除规则；sparse MLA AITER。
- **v0.10.x（当前）**：`AiterTritonMLABackend`（prefill 走 aiter Triton MHA）；`has_native_kv_cache_layout` 优化；MLA prefill 独立子系统接入。

---

[← 返回注意力首页](../../README.md)

## 参见

- [Triton backend](triton.md)
- [MLA AITER](mla/aiter.md)
- [底层 ops](../ops.md)
- [Utils](utils.md)
