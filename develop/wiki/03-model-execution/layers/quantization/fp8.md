[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > FP8 / MXFP4 / FP-Quant

# fp8 — FP8 与相关低比特浮点方案

> 源码（均为扁平 `.py`，**不存在 `fp8/` 子目录**）：
> - `vllm/model_executor/layers/quantization/fp8.py`（Fp8Config + Linear/MoE/KV-cache methods）
> - `vllm/model_executor/layers/quantization/mxfp4.py`（Mxfp4Config / GptOssMxfp4Config + MoE method）
> - `vllm/model_executor/layers/quantization/fp_quant.py`（FPQuantConfig，**已废弃**）
> - `vllm/model_executor/layers/quantization/input_quant_fp8.py`（`QuantFP8` CustomOp，通用 FP8 激活量化）
> - `vllm/model_executor/layers/quantization/qutlass_utils.py`（CUTLASS FP4 blocking 工具）

---

## 是什么

本页涵盖 vLLM 中以 **FP8 e4m3** 为核心及衍生低比特浮点（MXFP4 / NVFP4 / FP-Quant）的方案族。注意任务原描述的"`fp8/`：Fp8 mixed scheme + nvfp4/mxfp4/mxfp8 子目录"**在主线不存在子目录**——相关实现是扁平文件，MXFP8 主要在 `modelopt.py` 与 `online/mxfp8.py`，NVFP4 在 `modelopt.py` 与 `utils/nvfp4_utils.py`。

### `Fp8Config`（`fp8.py:95`）

- 方法名 `"fp8"`（`fp8.py:137`），min capability 75（Turing+，`:144`）。
- 关键字段：`is_checkpoint_fp8_serialized`（checkpoint 是否已存 FP8 权重，否则走在线）、`activation_scheme`（`"static"`/`"dynamic"`，`:90/110`）、`weight_block_size`（2 维 block，仅支持 serialized+dynamic，`:115`）、`store_dtype`（`"mxfp4"` 时 MoE 走 `Mxfp4MoEMethod`，`:204`）、`use_deep_gemm`。
- `get_quant_method`（`:175`）分发：
  - `LinearBase` → 在线（未 serialized）走 `online/fp8.py:Fp8PerTensorOnlineLinearMethod`（`:186`），离线走 `Fp8LinearMethod`（`:194`）；跳层返回 `UnquantizedLinearMethod`。
  - `RoutedExperts` → `store_dtype=="mxfp4"` 走 `Mxfp4MoEMethod`（`:205`）；离线 FP8 走 `Fp8MoEMethod`（`:211`）；在线走 `Fp8PerTensorOnlineMoEMethod`。
  - `Attention` → `Fp8KVCacheMethod`（`:219`，见 [kv_cache.md](kv_cache.md)）。

### `Fp8LinearMethod`（`fp8.py:267`）

支持 per-tensor 静态/动态与 128×128 block（DeepGEMM）。`create_weights`（`:322`）建 `weight`(FP8) + `weight_scale`/`weight_scale_inv`(block) + 可选 `input_scale`(static)；并调 `init_fp8_linear_kernel`（`:387`）按 `(activation_quant_key, weight_quant_key)` 选 CUTLASS/Marlin/DeepGEMM kernel。`process_weights_after_loading`（`:398`）：Marlin 路径做转置+`prepare`；否则 per-tensor 路径 `process_fp8_weight_tensor_strategy` 把多 shard scale 合并为单 scale（CUTLASS 要求）。`apply`（`:446`）含 `VLLM_BATCH_INVARIANT` 特殊路径（BF16 dequant+GEMM 后备）。

### `Fp8MoEMethod`（`fp8.py:492`）

per-tensor 或 block。`select_fp8_moe_backend`（`:527`）在 AITER/CUTLASS/Triton/DeepGEMM 间选；`_setup_kernel`（`:674`）调 `convert_to_fp8_moe_kernel_format`+`make_fp8_moe_kernel` 构造 modular kernel。支持 biased MoE（GPT-OSS，`:602`）、FNUZ 转换（`:730`）。`supports_eplb=True`（`:806`）。

### `Mxfp4Config` / `GptOssMxfp4Config`（`mxfp4.py`）

OCP MXFP4（FP4 + 1×32 E8M0 scale）。`Mxfp4Config`（`:40`）是基类，子类 override `get_name`/`override_quantization_method` 以认领特定 checkpoint（`mxfp4` 与 `gpt_oss_mxfp4`）。Linear 层目前回退 `UnquantizedLinearMethod`（`:83`，注释标"temporary fallback"）；MoE 走 `GptOssMxfp4MoEMethod`（`:89`），backend 选择 `select_mxfp4_moe_backend`/`select_deepseek_v4_mxfp4_moe_backend`。`mxfp4_round_up_hidden_size_and_intermediate_size` 处理对齐。

### `FPQuantConfig`（`fp_quant.py:30`，**已废弃**）

FP-Quant（[arXiv:2509.23202](https://arxiv.org/abs/2509.23202)）：Hadamard 旋转 + mxfp4/nvfp4 前向，需 sm100（`:63`）。在 `DEPRECATED_QUANTIZATION_METHODS`（`__init__.py:49`）。

### `QuantFP8`（`input_quant_fp8.py:29`）

`@CustomOp.register("quant_fp8")`，通用 FP8 激活量化算子，支持 per-tensor/token/channel/group、static/dynamic、TMA 对齐 scale、DeepGEMM E8M0。被各在线/离线 FP8 method 复用。

---

## 为什么

- **FP8 是 vLLM 当前最高吞吐/最有性价比的量化**。e4m3 硬件原生支持（Hopper+）+ CUTLASS `torch._scaled_mm`/DeepGEMM block kernel，几乎无精度损失。
- **一份 Config 同时服务离线与在线**。`is_checkpoint_fp8_serialized` 分支让用户既可加载预量化 checkpoint，也可用 `--quantization fp8`（在线）即时量化 bf16 模型，复用同一 `Fp8Config`。
- **block 量化服务 DeepSeek-V3 系**。128×128 block + DeepGEMM 是 DeepSeek MoE 的标配，`weight_scale_inv` 命名兼容 DeepSeek-V3 checkpoint（`fp8.py:378`）。
- **多后端调度MoE**。AITER(SGX)/Triton/CUTLASS/DeepGEMM 按 capability & TP 自动选择，免用户手调。
- **MXFP4/NVFP4 前沿低比特**。Blackwell 世代 FP4 硬件支持，MXFP4 服务 GPT-OSS / DeepSeek-V4 MoE，`mxfp4.py` + `utils/mxfp4_utils.py` + `utils/nvfp4_utils.py` 提供运行时。

---

## 怎么做

### Linear kernel 选择（`fp8.py:387`，`init_fp8_linear_kernel`）

| weight key | activation key | 选内核 |
|---|---|---|
| `kFp8StaticTensorSym` | static `kFp8StaticTensorSym` | CUTLASS per-tensor |
| `kFp8StaticTensorSym` | dynamic `kFp8DynamicTokenSym`（cutlass_fp8_supported） | CUTLASS per-token |
| `kFp8StaticTensorSym` | dynamic `kFp8DynamicTensorSym`（否则） | per-tensor dynamic |
| `kFp8Static128BlockSym` | `kFp8Dynamic128Sym` | DeepGEMM block / CUTLASS block |
| （<sm89） | — | `MarlinFP8ScaledMMLinearKernel`（weight-only） |

`block_quant` 时 activation key 用 `GroupShape(1, weight_block_size[0])`（`:305`）。

### MoE 后端选择（`fp8.py:527`，`select_fp8_moe_backend`）

输入：`moe config`、`weight_key`、`activation_key`、`allow_vllm_cutlass`。候选 backend（`Fp8MoeBackend` 枚举）：AITER / vLLM-CUTLASS / Triton / DeepGEMM。`convert_to_fp8_moe_kernel_format`（`:685`）按 backend 做 shuffle（AITER 需 `is_shuffled=True`，`:704`）。

### process_weights_after_loading（`fp8.py:398` / MoE `:720`）

- Linear per-tensor：`process_fp8_weight_tensor_strategy` 把 N 个 shard scale 取 max 重量化为单 scale（CUTLASS 要求）。
- Linear block：直接交 `fp8_linear.process_weights_after_loading`（DeepGEMM block scale 处理）。
- MoE：FNUZ 转换（`:730`）→ static input scale `process_fp8_input_tensor_strategy_moe`（`:746`）→ per-tensor `process_fp8_weight_tensor_strategy_moe`（`:756`）→ `_setup_kernel`。

### KV-cache scale 映射（`fp8.py:222`，`get_cache_scale_mapper`）

把 `.q_proj.output_scale`→`.attn.q_scale`、`.k_proj.output_scale`→`.attn.k_scale` 等，再 pipe 基类 mapper。

### MXFP4 MoE 流程（`mxfp4.py`）

`GptOssMxfp4MoEMethod` → `select_[deepseek_v4_]mxfp4_moe_backend` → `convert_[gpt_oss_]weight_to_mxfp4_moe_kernel_format` → `make_mxfp4_moe_kernel`。`Mxfp4Config.is_mxfp4_quant`（基类 `:262`）允许在 moe_config 创建前对齐 hidden_size。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`current_platform.fp8_dtype()`（e4m3fn/fnuz）、`is_fp8_fnuz()`（ROCm FNUZ ×2 scale）、`has_device_capability(89)`（Marlin FP8 回退）、AITER 仅 ROCm/特定卡。
- **[分布式](../../../07-distributed/README.md)**：block 量化要求 `intermediate_size_per_partition % block_n/k == 0`（`fp8.py:562/568`）；`get_tensor_model_parallel_world_size` 影响 MoE 后端。
- **[编译-IR](../../../09-compilation-ir/README.md)**：`QuantFP8` 为 `CustomOp`，支持 `torch.compile`；`Fp8MoEMethod.maybe_make_prepare_finalize` 显式 raise 指示走新 modular 初始化（`:765`）。
- **DeepGEMM**（`vllm/utils/deep_gemm.py`）：`is_deep_gemm_supported`/`per_block_cast_to_fp8`/`is_deep_gemm_e8m0_used`。
- **MoE modular kernel**（`layers/fused_moe/`）：`oracle/fp8.py`、`oracle/mxfp4.py`。
- **在线量化**（`online/fp8.py`）：FP8 在线 method 复用 `init_fp8_linear_kernel` 与 `select_fp8_moe_backend`，见 [online.md](online.md)。
- **KV-cache**：`Fp8KVCacheMethod` 见 [kv_cache.md](kv_cache.md)。

---

## 历史版本演进

- **v0.5**（待核实）：`Fp8Config`/`Fp8LinearMethod` 首次落地，per-tensor static/dynamic。
- **v0.6–v0.7**（待核实）：`Fp8MoEMethod`、`Fp8KVCacheMethod` 加入；CUTLASS per-token 动态。
- **v0.8**（待核实）：128×128 block 量化 + DeepGEMM（DeepSeek-V3）；FNUZ 归一化路径。
- **v0.9–v0.10**（待核实）：`use_deep_gemm` 开关、`store_dtype="mxfp4"` 分支接 `Mxfp4MoEMethod`；`GptOssMxfp4Config`/`gpt_oss_mxfp4`、`deepseek_v4_fp8` 方法名加入。
- **v0.11**（待核实）：`fp_quant` 标记 deprecated；`VLLM_BATCH_INVARIANT` BF16 后备路径。
- **v0.12 / main**：MoE modular kernel 化（`_setup_kernel`/`make_fp8_moe_kernel`）、`supports_eplb`、biased MoE（GPT-OSS）、per-token-head KV cache scales 适配。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [schemes.md](schemes.md) · [utils.md](utils.md) · [online.md](online.md) · [modelopt.md](modelopt.md) · [kv_cache.md](kv_cache.md) · [fbgemm.md](fbgemm.md)
