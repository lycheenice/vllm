[← Wiki 首页](../README.md) > [LoRA](README.md) > Ops

# LoRA Ops（`vllm/lora/ops/`）

> LoRA 增量计算的底层算子：shrink（X·A）与 expand（·B）的 GEMM，分 Triton / Torch / XPU 三套后端，由 Punica wrapper 按平台选用。

## 是什么

`vllm/lora/ops/` 三个子包：

| 子包 | 文件 | 算子 | 后端 |
|---|---|---|---|
| `torch_ops/` | `lora_ops.py` | `sgmv_shrink/expand`/`sgmv_expand_slice`/`bgmv_shrink/expand`/`bgmv_expand_slice` | 纯 PyTorch（einsum） |
| `triton_ops/` | `lora_shrink_op.py`/`lora_expand_op.py`/`lora_kernel_metadata.py`/`kernel_utils.py`/`utils.py` + MoE 系列 + fp8 系列 | `lora_shrink`/`lora_expand`/`fused_moe_lora*`/`*_fp8` | Triton kernel |
| `xpu_ops/` | `lora_ops.py` | `bgmv_shrink`/`bgmv_expand`/`bgmv_expand_slice` | Intel XPU C 扩展 (`torch.ops._xpu_C`) |

`__init__.py` 分别重导出。

### SGMV vs BGMV

- **SGMV**（Segmented GEMV）：prefill 场景，batch 内按 adapter id 聚簇成段，每段一次 GEMM（`compute_meta` 用 `torch.unique_consecutive` 合并连续同 id）。Torch 版 `sgmv_*` 内部 `torch.repeat_interleave` 后退化为 bgmv。
- **BGMV**（Batched GEMV）：decode 场景，每 token 一个 adapter，`lora_indices_tensor` 索引选行后 `einsum("bi,boi->bo")`。CUDA 上 v0.9 起统一用 Triton shrink/expand，Torch BGMV 主要服务 CPU/XPU。

## 为什么

- **多后端**：CUDA 走 Triton（自调 BLOCK_M/N/K/SPLIT_K/GROUP_SIZE_M，按 rank/dtype/M autotune）；CPU/XPU 走 PyTorch/原生 C 扩展，避免无 Triton 环境。
- **fp8 旁路**：`lora_shrink_fp8_op.py`/`lora_expand_fp8_op.py`/`fused_moe_lora_fp8_op.py` 让 LoRA 权重以 fp8 计算，省带宽。
- **MoE 融合**：`fused_moe_lora_op.py`（1804 行）把 w13/w2 的 shrink/expand 与专家路由 block 对齐，`_get_lora_id`/`_get_expert_id`/`_get_token_offs` 在 kernel 内解 `sorted_token_ids`，避免 Python 循环。
- **cudagraph 友好**：`LoRAKernelMeta`（`lora_kernel_metadata.py`）按 `captured_lora_counts` 预分配 metadata 张量，`meta_args(num_tokens, specialize)` 输出 grid 参数，使不同 active 数量有独立 grid。
- **PDL/TMA**：`utils.supports_pdl`/`supports_tma` 探测 Hopper 特性，kernel 里 `USE_GDC`/`launch_pdl`/`TMA` constexpr 分支。

## 怎么做

### Torch 版（`torch_ops/lora_ops.py`）

```python
bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices, add_inputs):
    selected = lora_b_weights[lora_indices].squeeze(1)
    outputs = torch.einsum("bi,boi->bo", inputs, selected)
    output_tensor[:, :common] += outputs   # or =
```

`sgmv_*` 先 `repeat_interleave(lora_indices, seq_len)` 展开成 token 级再调 bgmv。语义清晰但性能弱，用于 CPU。

### Triton shrink（`triton_ops/lora_shrink_op.py`）

`_lora_shrink_kernel`（`:23`）：按 `token_indices_sorted_by_lora_ids`/`num_tokens_per_lora`/`lora_token_start_loc`/`lora_ids` 段化，`SPLIT_K`/`GROUP_SIZE_M`/`BLOCK_M/N/K` autotune，`scaling` 乘进输出。`do_shrink_kernel`（`kernel_utils.py`）按平台/metadata 选 bgmv 或 sgmv 分支。`lora_shrink` 顶层注册为 custom op（`direct_register_custom_op`）。

### Triton expand（`lora_expand_op.py`）

`_lora_expand_kernel`：同段化策略，`add_inputs` 控制累加 vs 覆盖；`offset_start` 支持 S-LoRA 行并行的列切片写入。`lora_expand` 注册 custom op。

### fused_moe_lora（`fused_moe_lora_op.py`）

`_get_lora_id`/`_get_expert_id`/`_get_token_offs` 在 kernel 内解 `sorted_token_ids`（block assignment），`naive_block_assignment` 分支走 EP 本地排序。`fused_moe_lora_shrink`/`fused_moe_lora_expand`/`fused_moe_lora` 分别处理 w13 shrink、w2 expand、融合入口；`fused_moe_lora_fp8_op.py` 为 fp8 变体。`moe_lora_align_block_size`（在 Punica wrapper）对齐 token×expert block。

### LoRAKernelMeta（`lora_kernel_metadata.py`）

`LoRAKernelMeta.make(max_loras, max_tokens, device, captured_lora_counts)` 预建按 active 计数分桶的 metadata buffer；`prepare_tensors(token_lora_indices)` 填当前步；`meta_args(num_tokens, specialize)` 返回 kernel 所需位置参数。

### XPU 版（`xpu_ops/lora_ops.py`）

`_bgmv_shrink_impl`/`_bgmv_expand_impl` 委托 `torch.ops._xpu_C.bgmv_*`，`bgmv_expand` 处理 weight_out_dim < output_dim 的 slice 写入。注册为 custom op。

## 与其它模块/系统配合

- **PunicaWrapperGPU**：直接调 `lora_shrink`/`lora_expand`/`fused_moe_lora*`，经 `LoRAKernelMeta.meta_args` 传参；见 [punica.md](punica.md)。
- **PunicaWrapperCPU**：调 `torch_ops` 的 `sgmv_*`/`bgmv_*`；见 [punica.md](punica.md)。
- **PunicaWrapperXPU**：调 `xpu_ops` 的 `bgmv_*`（+ 部分复用 Triton fused_moe）；见 [punica.md](punica.md)。
- **utils.get_captured_lora_counts**：决定 `captured_lora_counts`，与 cudagraph 对齐；见 [utils.md](utils.md)、[v1-integration.md](v1-integration.md)。
- [08-硬件平台](../08-platforms/README.md)：`current_platform.get_punica_wrapper()` 决定用哪套 op。
- [模型执行-FusedMoE](../03-model-execution/layers/fused-moe.md)：`MoELoRAContext` 消费 fused_moe_lora。

## 历史版本演进

- **v0.5（首版）**：CUDA Triton shrink/expand（BGMV/SGMV），torch_ops 作 fallback。
- **v0.7/v0.8（MoE）**：`fused_moe_lora_op.py` 引入，专家路由 block 对齐的 w13/w2 kernel。
- **v0.9（多后端拆分）**：`ops/` 重构为 `torch_ops`/`triton_ops`/`xpu_ops` 三子包；XPU 用 `torch.ops._xpu_C`。
- **v0.10/v0.11（fp8 + autotune）**：`*_fp8_op.py` 系列；`LoRAKernelMeta` + `captured_lora_counts` 支持 specialize cudagraph；`fused_moe_lora_fp8_op.py`。
- **main**：PDL/TMA 探测与 constexpr 分支；`fused_moe_lora_op.py` 扩到 1804 行支持 EP 本地排序与 non-gated MoE。

## 参见

- [← 返回 LoRA 首页](README.md)
- [punica.md](punica.md)
- [layers.md](layers.md)
- [v1-integration.md](v1-integration.md)
- [08-硬件平台](../08-platforms/README.md)
