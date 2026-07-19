# Backend 列表导航

[← Wiki 首页](../../README.md) > [注意力](../README.md) > Backend 列表

> 源码目录：`vllm/v1/attention/backends/`

## 是什么

`backends/` 下放所有 `AttentionBackend` 的具体实现。每个文件通常含 `XxxBackend`（能力声明 + 工厂）、`XxxMetadataBuilder`、`XxxImpl`（kernel 调用）三件套。除 `registry.py` / `utils.py` / `fa_utils.py` 是基础设施外，其余文件都是一个 backend。

## 为什么

按"硬件平台 + 模型族 + 量化模式"纵向切分文件，让每个 backend 独立演进、独立 gating。新增 backend 只加一个文件 + 一条 enum，不动其它 backend。

## 怎么做

### 实际文件清单

| 文件 | 主要类 | 语义 | 支持平台 |
|------|--------|------|----------|
| `flash_attn.py` | `FlashAttentionBackend` / `FlashAttentionImpl` | FlashAttention (FA2/FA3/FA4) varlen + cascade | CUDA / XPU |
| `flash_attn_diffkv.py` | `FlashAttentionDiffKVBackend` | FA3/FA4 diff-kv（K≠V head dim，R1 类） | CUDA |
| `flashinfer.py` | `FlashInferBackend` | FlashInfer（trtllm-gen + cascade + FP8/NVFP4） | CUDA |
| `triton_attn.py` | `TritonAttentionBackend` → `ops.unified_attention` | 纯 Triton 统一 kernel | CUDA / ROCm / XPU |
| `triton_attn_diffkv.py` | `TritonAttentionDiffKVBackend` | Triton diff-kv | CUDA |
| `rocm_attn.py` | `RocmAttentionBackend` | ROCm PagedAttention + Triton prefix prefill | ROCm |
| `rocm_aiter_fa.py` | `AiterFlashAttentionBackend` | ROCm AITER FlashAttention | ROCm |
| `rocm_aiter_unified_attn.py` | `RocmAiterUnifiedAttentionBackend`（继承 RocmAttn）| ROCm AITER 统一注意力 | ROCm |
| `cpu_attn.py` | `CPUAttentionBackend` | CPU SDPA + paged | CPU |
| `flex_attention.py` | `FlexAttentionBackend` | PyTorch `flex_attention` + BlockMask | CUDA（ViT/_encoder） |
| `mamba_attn.py` | `BaseMambaAttentionMetadataBuilder` | SSM 公共基类 | 通用 |
| `mamba1_attn.py` | `Mamba1AttentionBackend` | Mamba1 SSM | CUDA |
| `mamba2_attn.py` | `Mamba2AttentionBackend` | Mamba2 SSD | CUDA |
| `short_conv_attn.py` | `ShortConvAttentionBackend` | ShortConv（继承 Mamba base） | 通用 |
| `linear_attn.py` | `LinearAttentionBackend` | 线性注意力 SSM | 通用 |
| `gdn_attn.py` | `GDNAttentionBackend` | GatedDeltaNet | 通用 |
| `hpc_attn.py` | `HpcAttentionBackend` | 腾讯 hpc-ops（H20/H200，Hy3 模型） | CUDA Hopper |
| `turboquant_attn.py` | `TurboQuantAttentionBackend` | TurboQuant KV cache（k3v4_nc 等） | CUDA / XPU |
| `utils.py` | 工具函数 | KV cache layout、split_decodes/prefills、cascade 工具、DCPhelper | 通用 |
| `fa_utils.py` | 工具函数 | FlashAttention 版本探测 / 导入 / CuTeDSL compile spec | CUDA/ROCm/XPU |
| `mla/` | 子目录 | DeepSeek V2/V3/V4 MLA 全家桶 | 多平台 |

### SSM 族补充说明

`MambaAttentionBackendEnum`（见 注册表）登记 Mamba1/2/ShortConv/Linear/GDN，`is_ssm()` 返回 True，由 `selector.get_mamba_attn_backend()` 选择，走单独的 `MambaSpec`。

### 杂项 backend

- `HPC_ATTN`：仅 Hopper（H20/H200），block_size 必须 64，当前只服务 Hy3 模型（`registry.py:106` 注释）。
- `TURBOQUANT`：KV cache 用 `turboquant_*` dtype 时由 XPU platform 直接路由，CUDA 优先级表末位。
- `TORCH_SDPA`：value 为空串，仅 ViT 用。
- `NO_ATTENTION`：路径 `backends/no_attention.py`，**仓库中未找到该文件**（待核实）。

## 与其它模块/系统配合

每个 backend 都：
- 由 [selector](../selector.md) / registry 选出；
- 实现 [backend 抽象层](../backend-abstraction.md) 的接口；
- 调用 [底层 ops](../ops.md) 的 kernel；
- 被 [执行层-Worker](../../02-execution/worker/README.md) 实例化；
- 其 KV cache 形状被 [引擎核心-KV 管理](../../01-engine-core/kv-cache-management/README.md) 用于分配。

## 历史版本演进

- **v0.6.x**：FlashAttn / Triton / CPU / FlashInfer 初版。
- **v0.7.x**：FlexAttention；`utils.py` 统一 cascade/layout 工具。
- **v0.8.x**：`mla/` 子目录建立；FA3 引入（`fa_utils.py` 版本探测）。
- **v0.9.x**：Mamba2 / GDN / ShortConv / Linear SSM 族扩充；`flash_attn_diffkv` / `triton_attn_diffkv`；TurboQuant。
- **v0.10.x（当前）**：HPC backend；`rocm_aiter_unified_attn`；MLA `prefill/` 解耦子目录；FA4 CuTeDSL warmup。

---

## 子页面

- [FlashAttention](flash-attn.md) ｜ [FlashInfer](flashinfer.md) ｜ [Triton](triton.md)
- [ROCm](rocm.md) ｜ [Mamba/SSM](mamba.md) ｜ [Linear](linear.md)
- [CPU](cpu.md) ｜ [FlexAttention](flex-attention.md) ｜ [Utils](utils.md)
- [MLA 总览](mla/README.md)
  - [AITER](mla/aiter.md) ｜ [Triton](mla/triton.md) ｜ [CUTLASS](mla/cutlass.md)
  - [FlashAttn](mla/flashattn.md) ｜ [FlashInfer](mla/flashinfer.md) ｜ [FlashMLA](mla/flashmla.md)

## 参见

- [backend 抽象层](../backend-abstraction.md)
- [backend 注册表](../backend-registry.md)
- [底层 ops](../ops.md)

[← 返回注意力首页](../README.md)
