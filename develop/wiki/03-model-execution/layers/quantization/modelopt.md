[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > ModelOpt

# modelopt — NVIDIA ModelOpt 量化

> 源码（扁平 `.py`）：`vllm/model_executor/layers/quantization/modelopt.py`（2562 行，量化层库中最大单文件）
>
> 方法名：`modelopt`（→`ModelOptFp8Config`）、`modelopt_fp4`（→`ModelOptNvFp4Config`）、`modelopt_mxfp8`/`mxfp8` checkpoint（→`ModelOptMxFp8Config`）、`modelopt_mixed`（→`ModelOptMixedPrecisionConfig`）（`__init__.py:129`）。

---

## 是什么

`modelopt.py` 承载 vLLM 对 **NVIDIA TensorRT-Model-Opt**（NeMo/ModelOpt）导出量化 checkpoint 的运行时支持。4 个 `QuantizationConfig` 子类，对应 4 种 ModelOpt 量化产物：

| Config 类 | 行号 | 方法名 | 含义 |
|---|---|---|---|
| `ModelOptFp8Config` | (待核实声明行) | `modelopt` | FP8 per-tensor/per-channel + online/block |
| `ModelOptNvFp4Config` | (待核实) | `modelopt_fp4` | NVFP4（Blackwell FP4 + 二级 scale） |
| `ModelOptMxFp8Config` | (待核实) | `modelopt_mxfp8` / `mxfp8`(ckpt) | MXFP8（OCP，1×32 E8M0 scale） |
| `ModelOptMixedPrecisionConfig` | (待核实) | `modelopt_mixed` | 混合精度（per-layer 异构：部分 W4A16 + 部分 W8A8 等） |

每个 Config 都实现 `override_quantization_method`（`:415`/`:1064`/`:1722`/`:2321`），通过检查 checkpoint 的 `quant_method`/`weights` 子字段自我认领。

Linear method 变体：
- FP8：`ModelOptFp8LinearMethod`（`:443`）、`ModelOptFp8PcPtLinearMethod`(`:539`, per-channel+per-tensor)、`ModelOptFp8PbWoLinearMethod`(`:620`, per-block weight-only)、`ModelOptFp8MoEMethod`(`:749`)。
- NVFP4：`ModelOptNvFp4LinearMethod`(`:1111`)、`ModelOptNvFp4W4A16LinearMethod`(`:1245`, weight-only A16)、`ModelOptNvFp4FusedMoE`(`:1391`)。
- MXFP8：`ModelOptMxFp8LinearMethod`(`:1778`)、`ModelOptMxFp8FusedMoE`(`:1889`)。
- KV-cache：`ModelOptKVCacheMethod`(`:123`，继承 `BaseKVCacheMethod`)。

内核选择来自 `vllm/model_executor/kernels/linear`：`init_fp8_linear_kernel`、`init_mxfp8_linear_kernel`、`init_nvfp4_linear_kernel`、`MarlinNvFp4LinearKernel`/`NvFp4LinearLayerConfig`（`modelopt.py:14`）。

---

## 为什么

- **NVIDIA 官方量化链路闭环**。ModelOpt 与 TensorRT-LLM 同源，checkpoint 在 vLLM 与 TRT-LLM 间互通；NVFP4/MXFP8 需求来自 Blackwell 世代硬件原生支持。
- **NVFP4 二级 scale**。NVFP4 = FP4 权重 + per-16-group FP8 scale + per-tensor FP32 scale（`utils/quant_utils.py:139` 的 `kNvfp4Dynamic` 有 `scale2`），是 Blackwell 提升精度的关键，本文件是 vLLM 主要承载。
- **混合精度**。`modelopt_mixed` 让同一模型不同模块用不同量化，跳过敏感层，最大化精度/吞吐。
- **复用通用 backends**。FP8 MoE 走 `oracle/fp8`（`select_fp8_moe_backend`），NVFP4 MoE 走 `oracle/nvfp4`（`select_nvfp4_moe_backend`/`make_nvfp4_moe_kernel`），MXFP8 MoE 走 `oracle/mxfp8`。Linear 走通用 `init_*_linear_kernel`。

---

## 怎么做

### override 认领

4 个 `override_quantization_method` 检查 `quant_method` 字串（`"modelopt"` 系）+ checkpoint `weights`/`input_activations` 的 `dtype`/`strategy`。命中后 `ModelConfig` 把 `self.quantization` 置为对应 `modelopt_*` 名。

### Linear 内核选择

- FP8：`init_fp8_linear_kernel(activation_quant_key, weight_quant_key, ...)`（`modelopt.py:17`）。per-channel 走 `ModelOptFp8PcPtLinearMethod`（`kFp8StaticChannelSym`+`kFp8StaticTensorSym`），per-block weight-only 走 `ModelOptFp8PbWoLinearMethod`。
- NVFP4：`init_nvfp4_linear_kernel` → `NvFp4LinearLayerConfig`；weight-only A16 走 `MarlinNvFp4LinearKernel`（`modelopt.py:15`）+ `ModelOptNvFp4W4A16LinearMethod`。
- MXFP8：`init_mxfp8_linear_kernel`，scale dtype `MXFP8_SCALE_DTYPE`、value dtype `MXFP8_VALUE_DTYPE`、block size `MXFP8_BLOCK_SIZE`（`utils/mxfp8_utils.py`，`:71` import）。

### MoE

- FP8 MoE（`:749`）：`select_fp8_moe_backend` + `convert_to_fp8_moe_kernel_format` + `make_fp8_moe_kernel`/`make_fp8_moe_quant_config`。
- NVFP4 MoE（`:1391`）：`select_nvfp4_moe_backend` + `convert_to_nvfp4_moe_kernel_format` + `make_nvfp4_moe_kernel`/`make_nvfp4_moe_quant_config`；`is_global_sf_supported_for_nvfp4_backend` 判定 global scale-factor 支持。
- MXFP8 MoE（`:1889`）：`select_mxfp8_moe_backend`。

### process_weights_after_loading 要点

- FP8：`requantize_with_max_scale`（`utils/w8a8_utils.py`，`:87` import）做 per-shard→per-tensor；FlashInfer `swap_w13_to_w31`（`:61` import）；`process_fp8_weight_tensor_strategy_moe`/`process_fp8_input_tensor_strategy_moe`（`:64/65` import）。
- NVFP4：`convert_to_nvfp4_moe_kernel_format` 把权重/scale/global scale 改排为内核期望。
- 输入量化融合：`expose_input_quant_key`（`layers/fusion/quant_activation.py`，`:47` import）——把 input quant 暴露给上游 fusion 层。

### KV-cache & MLA

`ModelOptKVCacheMethod`（`:123`）继承 `BaseKVCacheMethod`（见 [kv_cache.md](kv_cache.md)）。文件还 import `MLAAttention`（`:21`），支持 DeepSeek MLA 的 q/prob/kv scale。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：NVFP4 需 Blackwell（sm100），`current_platform.has_device_capability`；MXFP8 也偏 Blackwell/Hopper；FNUZ 路径。
- **[分布式](../../../07-distributed/README.md)**：NVFP4 的 global scale-factor TP 适配（`is_global_sf_supported_for_nvfp4_backend`）；MoE backend 选择受 TP 影响。
- **[编译-IR](../../../09-compilation-ir/README.md)**：`init_*_linear_kernel` 返回的 kernel 对象需 torch.compile 兼容；`expose_input_quant_key` 与 fusion 编译配合。
- **DeepGEMM/FlashInfer/AITER**：MoE backend 底层依赖。
- **KV-cache**：`ModelOptKVCacheMethod` + `get_cache_scale_mapper`（ModelOpt 的 `.self_attn.{k,v}_proj.{k,v}_scale`→`.attn.*` 见 `base_config.py:207`）。

---

## 历史版本演进

- **v0.8–v0.9**（待核实）：`ModelOptFp8Config` 首批支持（per-tensor FP8）。
- **v0.10**（待核实）：NVFP4（`modelopt_fp4`）随 Blackwell 引入；MXFP8（`modelopt_mxfp8`）。
- **v0.11**（待核实）：`modelopt_mixed` 混合精度；per-channel/per-block FP8 Linear 变体（`PcPt`/`PbWo`）；NVFP4 weight-only A16（`MarlinNvFp4LinearKernel`）；MLA attention scale 支持。
- **v0.12 / main**：`mxfp8` checkpoint 方法名同步用于 MiniMax 系（`__init__.py:166` 注释）；NVFP4 MoE global sf 支持；与 fusion 层 `expose_input_quant_key` 协同。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [fp8.md](fp8.md) · [kv_cache.md](kv_cache.md) · [humming.md](humming.md) · [utils.md](utils.md) · [schemes.md](schemes.md)
