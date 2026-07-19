# CUTLASS MLA backend

[← Wiki 首页](../../../README.md) > [注意力](../../README.md) > [Backend 列表](../README.md) > [← MLA 首页](../README.md) > CUTLASS MLA

> 源码：`vllm/v1/attention/backends/mla/cutlass_mla.py`

## 是什么

`CutlassMLABackend` / `CutlassMLAImpl`（`cutlass_mla.py:38/112`）是 NVIDIA Blackwell（SM100）专用的 MLA dense backend，基于 CUTLASS 的 `sm100_cutlass_mla_decode` kernel（通过 `vllm._custom_ops`）。继承 `MLACommonBackend`/`MLACommonImpl`，配套 `CutlassMLAMetadataBuilder`（`cutlass_mla.py:31`）与 `SM100Workspace`（`cutlass_mla.py:77`）。

## 为什么

- **Blackwell decode 最优之一**：CUTLASS 针对 SM100 张量核心优化的 MLA decode kernel。
- **block 128 强制**：kernel 要求固定 block_size=128，与 FlashMLA(64)/FlashInfer(32/64) 不同。
- **`UNIFORM_SINGLE_TOKEN_DECODE` cudagraph**：decode-only 捕获。
- **DCP 支持**：`can_return_lse_for_decode` 继承自 `MLACommonImpl`，可吐 LSE 做跨 rank 归约。

## 怎么做

### 能力要点

| 项 | 值 | 位置 |
|----|----|------|
| dtypes | fp16/bf16 | `cutlass_mla.py:39` |
| kv dtype | auto/fp16/bf16/fp8/fp8_e4m3 | `cutlass_mla.py:40` |
| block sizes | `[128]`（固定） | `cutlass_mla.py:49` |
| `supports_compute_capability` | `major == 10`（仅 Blackwell） | `cutlass_mla.py:73` |
| cudagraph | `UNIFORM_SINGLE_TOKEN_DECODE`（decode-only） | `cutlass_mla.py:33` |
| stride_order | blocks 优先 `(1,0,2,3)`（含 layers 维） | `cutlass_mla.py:53` |

### SM100Workspace

`SM100Workspace`（`cutlass_mla.py:77`）：

- `_block_size` 固定 128。
- `ensure_size(attn_metadata, num_kv_splits)` 按 `sm100_cutlass_mla_get_workspace_size` 计算并复用 buffer。
- `_sm_count` 预计算（用 device 0 代理）。

### forward_mqa

`CutlassMLAImpl.forward_mqa`（`cutlass_mla.py:256`）：

- 不支持 `q_scale`/`k_scale` 非 1（`cutlass_mla.py:266` raise）。
- 拆 `q` 为 `q_nope`（kv_lora_rank）+ `q_pe`（qk_rope_head_dim）。
- `ensure_size` → 调 `ops.sm100_cutlass_mla_decode(q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, block_table, workspace, scale, num_kv_splits)`。
- 头数 < `MAX_HEADS` 时切片输出。

prefill 走 `MLACommonImpl.forward_mha` → `prefill/selector.py`（Blackwell 默认 `FlashAttnPrefillBackend`）。

### 选择

`platforms/cuda.py:117` 把 `CUTLASS_MLA` 放 Blackwell MLA 优先级第 3（在 FlashInfer MLA、TokenSpeed 之后）。

## 与其它模块/系统配合

- **ops**：`vllm._custom_ops.sm100_cutlass_mla_decode` / `sm100_cutlass_mla_get_workspace_size`（C++ extension）。
- **workspace manager**：独立 `SM100Workspace` 管理（非全局 workspace manager）。
- **selector**：见 [selector](../../selector.md)。
- **MLA 公共层**：见 [MLA 总览](../README.md)。
- **Blackwell**：SM100 专属，Hopper 不可用。

## 历史版本演进

- **v0.8.x**：`CutlassMLABackend` 引入（Blackwell SM100 MLA decode）。最早 SM100 MLA kernel 之一。
- **v0.9.x**：FP8 KV cache 支持；DCP LSE 返回；stride_order blocks 优先支持跨层布局。
- **v0.10.x（当前）**：`SM100Workspace` 按需 ensure_size；`UNIFORM_SINGLE_TOKEN_DECODE` decode-only cudagraph 稳定；`MAX_HEADS` 切片；prefill 外移到 `prefill/`。

---

[← 返回 MLA 首页](../README.md)

## 参见

- [MLA FlashInfer](flashinfer.md)（Blackwell 另一选择）
- [MLA FlashMLA](flashmla.md)
- [MLA 总览](../README.md)
