# 自定义算子注册入口（_custom_ops / _aiter_ops / _xpu_ops）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 自定义算子注册

本页覆盖 vLLM 的三份平台分流算子注册文件：`vllm/_custom_ops.py`（3736 行，CUDA 为主）、`vllm/_aiter_ops.py`（3032 行，ROCm AITER）、`vllm/_xpu_ops.py`（1084 行，Intel XPU）。

## 是什么

三者都是"把 C++/Triton 自定义算子挂到 `torch.ops.vllm` 命名空间"的注册层，模式统一：

- 顶部 `current_platform.import_kernels()`（`vllm/_custom_ops.py:20`）让平台按需 dlopen 自己的 `.so`。
- 每个算子定义 Python `impl`（真实实现，调底层扩展）与 `fake`（用于 torch.compile 的 abstract/shape 推断）。
- 通过 `vllm.utils.torch_utils.direct_register_custom_op(name, impl, fake, mutates_args=...)` 注册到 `torch.library`，使 `torch.ops.vllm.<name>` 可用且 torch.compile 友好。

### `vllm/_custom_ops.py`（CUDA 为主）

注册约 34+ 个算子（按 `direct_register_custom_op` 计数），覆盖：

- **注意力**：`paged_attention_rocm`（ROCm 分支）、`mla_decode_kvcache_cpu`（CPU MLA）、`merge_attn_states`。
- **归一化**：`rms_norm`、`fused_add_rms_norm`、`fused_qk_norm_rope`、`rms_norm_dynamic_per_token_quant`、`rms_norm_per_block_quant`、`silu_and_mul_per_block_quant`。
- **RoPE**：`rotary_embedding`。
- **惩罚**：`apply_repetition_penalties_torch`/`_cuda`/（统一入口 `apply_repetition_penalties`）。
- **量化 GEMM**：`awq_dequantize`/`awq_gemm`/`gptq_gemm`/`gptq_shuffle`/`gptq_gemm_rdna3`/`moe_gptq_gemm_rdna3`。
- **CUTLASS**：`cutlass_scaled_mm`/`_azp`/`_fp4`/`_moe_mm`/`_fp4_moe_mm`/`_mxfp4_moe_mm` + 支持探测（`*_supports_fp4/fp8/block_fp8/group_gemm`）+ 数据搬运（`get_cutlass_*`/`shuffle_rows`）。
- **Marlin 重排**：`gptq_marlin_repack`/`awq_marlin_repack`/`gptq_marlin_moe_repack`/`awq_marlin_moe_repack`。
- **FP4**：`create_fp4_scale_tensor`/`create_fp4_output_tensors`/`scaled_fp4_quant` 等，配合 [scalar_type.md](scalar-type.md) 的 NVFP4 类型。
- 大量算子接收 `ScalarType.id` 描述权重量化类型。

### `vllm/_aiter_ops.py`（ROCm AITER）

- `is_aiter_found()`/全局 `IS_AITER_FOUND`（`vllm/_aiter_ops.py:39`/`:49`）检查 `aiter` 包是否可导入；`find_spec` 不 torch.compile 兼容故缓存为全局。
- `_ensure_hipb_mm_extension_initialized()`（`:30`）：按 device 懒初始化 hipBLASLt 扩展，记录已初始化 device 集合。
- `FP8_DTYPE = current_platform.fp8_dtype()`（`:26`）缓存（ROCm 上 `is_fp8_fnuz` 是 host op，避免每步重算）。
- 注册 AITER 提供的 ROCm 专用 attn/moe/norm/gemm 算子，以及 `rocm_aiter_sparse_attn_indexer`（与 `vllm/v1/attention/ops/rocm_aiter_mla_sparse` 配套）。受 `VLLM_ROCM_USE_AITER*` 一族 env 控制是否调用（见 [envs.md](envs.md)）。

### `vllm/_xpu_ops.py`（Intel XPU）

- 以 `xpu_ops` 类（`vllm/_xpu_ops.py:722`）聚合，实现一系列 `_xpu_*_impl` + `_xpu_*_fake`：GDN attention core、DeepSeek scaling rope、FP8 MQA logits（paged/非 paged）、`_topk_topp_sample`、`_xpu_mxfp8_quantize`/`_xpu_mxfp4_quantize`、`_selective_scan_fwd_kernel`（Mamba/SSM）等。
- 同样走 `direct_register_custom_op` 注册。受 `VLLM_XPU_ENABLE_XPU_GRAPH`/`VLLM_XPU_USE_SAMPLER_KERNEL` 控制。

## 为什么

- **平台分流**：CUDA/ROCm/XPU 算子实现完全不同，但语义同名；按平台分别注册文件，由 `current_platform.import_kernels()` 选择加载，避免在业务层写满 `if is_cuda/is_rocm/is_xpu`。
- **torch.compile 必需 fake**：每个算子必须有 abstract 实现，否则 Dynamo 无法追踪；`direct_register_custom_op` 把 fake 与 impl 绑定注册，统一模式。
- **`mutates_args` 显式声明**：torch.compile 需要知道哪些输入被原地修改，否则会误拷贝/破图。
- **懒初始化**：aiter 的 hipBLASLt 扩展、XPU 的 device 上下文按 device 懒初始化，避免无 GPU 环境导入即崩。

## 怎么做

```python
import torch
# 算子在 torch.ops.vllm 命名空间
out = torch.ops.vllm.rms_norm(x, w, eps)
# 注册新算子（开发流程）
from vllm.utils.torch_utils import direct_register_custom_op
direct_register_custom_op("my_op", my_impl, my_fake, mutates_args=("out",))
```

- 新增 CUDA 算子：在 `csrc/` 写 C++、`torch.library` 暴露，在 `_custom_ops.py` 加 Python wrapper + fake + `direct_register_custom_op`。
- 平台专用：放进对应 `_*_ops.py`；用 `IS_AITER_FOUND`/`current_platform` 守卫。
- 算子内部读步级信息：经 `get_forward_context()`（[forward-context.md](forward-context.md)）。

## 与其它模块/系统配合

- [模型执行 · 内核](../03-model-execution/README.md)：量化层、attention 后端、MoE runner 直接调 `torch.ops.vllm.*`。
- [平台](../08-platforms/README.md)：`current_platform.import_kernels()` 是注册触发器；`fp8_dtype()` 等平台 API 被算子文件消费。
- [编译与 IR](../09-compilation-ir/README.md)：fake 实现是 torch.compile 追踪前提；`mutates_args` 影响 cudagraph 捕获。
- [scalar-type.md](scalar-type.md)：量化算子接收 `ScalarType.id`。
- [utils.md](utils.md)：`torch_utils.direct_register_custom_op`、`flashinfer.flashinfer_quant_nvfp4_8x4_sf_layout` 等被本文件复用。
- [envs.md](envs.md)：`VLLM_ROCM_USE_AITER*`、`VLLM_XPU_*`、`VLLM_DISABLED_KERNELS`。
- [注意力](../05-attention/README.md)：`paged_attention_*`、`merge_attn_states`、`rocm_aiter_sparse_attn_indexer` 服务各后端。

## 历史版本演进

- **v0.5–v0.6**：`_custom_ops.py` 已有数百行，主要为 Marlin/GPTQ/AWQ 量化 GEMM 与 paged attention。
- **v0.7–v0.8**：随 torch.compile 集成，全面加 `register_fake`/`mutates_args`；`direct_register_custom_op` 成为标准。
- **v0.9–v0.10**：`_aiter_ops.py` 从 `_custom_ops` 分流（ROCm AITER 算子族扩张）；CUTLASS scaled_mm/moe_mm 族与 FP4 量化算子就位。
- **v0.11–main**：`_xpu_ops.py` 持续扩展（DeepSeek/Mamba/FP4/FP8）；`scaled_fp4_quant` 等 NVFP4 路径完善；`VLLM_DISABLED_KERNELS` 支持运行期禁用。三文件具体算子引入版本（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [scalar-type.md](scalar-type.md)、[forward-context.md](forward-context.md)、[utils.md](utils.md)
- [平台子系统](../08-platforms/README.md)
- [模型执行 · 内核](../03-model-execution/README.md)
- [构建/CI · csrc](../18-build-ci-testing/README.md)
