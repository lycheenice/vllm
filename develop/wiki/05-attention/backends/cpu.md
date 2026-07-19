# CPU attention backend

[← Wiki 首页](../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > CPU

> 源码：`vllm/v1/attention/backends/cpu_attn.py`

## 是什么

`CPUAttentionBackend`（`cpu_attn.py:39`）是 CPU 平台唯一注意力 backend，基于 `vllm._custom_ops`（C++ extension）+ PyTorch SDPA。三件套：`CPUAttentionBackend`、`CPUAttentionMetadataBuilder`、`CPUAttentionBackendImpl`。

## 为什么

CPU 无 FlashAttention/FlashInfer，需要原生实现：

- 用 `_custom_ops` 的 paged 读 + `torch.nn.functional.scaled_dot_product_attention`（SDPA）做 prefill，或专用 decode kernel。
- 支持 FP8 KV cache（e4m3/e5m2），靠 CPU AMX/AVX 指令。
- 不支持 MLA / sparse（`platforms/cpu.py:83` 直接 `NotImplementedError`）。
- 强制 `HND` KV cache layout（`cpu_attn.py:100`）。

## 怎么做

### 能力要点

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16/fp32 | `cpu_attn.py:42` |
| kv dtype | auto/fp8/e4m3/e5m2 | `cpu_attn.py:47` |
| block sizes | `MultipleOf(16)` | `cpu_attn.py:55` |
| head sizes | 32,64,80,96,112,128,160,192,224,256,512 | `cpu_attn.py:59` |
| `supports_non_causal` | True | `cpu_attn.py:67` |
| `supports_attn_type` | decoder/encoder/encoder_only/encoder_decoder | `cpu_attn.py:71` |
| `forward_includes_kv_cache_update` | False | `cpu_attn.py:40` |
| KV cache 形状 | `(num_blocks, num_kv_heads, block_size, 2*head_size)`（K/V packed 最后维） | `cpu_attn.py:90` |
| `get_required_kv_cache_layout` | `"HND"`（platform 强制） | `cpu_attn.py:100` |
| cascade | 不支持 | `cpu_attn.py:104` |

### metadata

`CPUAttentionMetadata`（`cpu_attn.py:108`）含 `scheduler_metadata`、`causal`、`dynamic_causal`（per-request 因果）、`use_sdpa_prefill`、`num_decode_tokens`、`sdpa_attn_masks`（SDPA 需要显式 mask）。builder 按 `CpuArchEnum`（来自 `current_platform`）选实现路径。

### forward

`CPUAttentionBackendImpl.forward`（`cpu_attn.py:312`）：prefill 走 SDPA 或专用 kernel，decode 走 paged decode。KV 写入走 `_custom_ops.reshape_and_cache`。

## 与其它模块/系统配合

- **platforms/cpu.py**：`get_attn_backend_cls`（`cpu.py:75`）只返回 `CPU_ATTN`，MLA/sparse 直接报错。
- **selector**：见 [selector](../selector.md)。
- **utils**：`KVCacheLayoutType`，强制 HND。
- **架构适配**：`CpuArchEnum` 区分 x86/ARM 等，影响 kernel 选择（待核实具体分支）。

## 历史版本演进

- **v0.6.x**：CPU backend 初版，SDPA prefill + paged decode。
- **v0.7.x**：FP8 KV cache；`supports_non_causal`；encoder-decoder 支持。
- **v0.8.x**：`get_required_kv_cache_layout="HND"`；head sizes 列表完善。
- **v0.9.x**：`dynamic_causal` per-request；SDPA mask 路径。
- **v0.10.x（当前）**：`supports_attn_type` 含全四种；`BailingLinearAttention` 等不在 CPU 走。

---

[← 返回注意力首页](../../README.md)

## 参见

- [FlashAttention backend](flash-attn.md)
- [selector](../selector.md)
- [Utils](utils.md)
