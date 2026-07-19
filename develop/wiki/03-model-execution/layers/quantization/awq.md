[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > AWQ

# awq — AWQ 量化（AutoAWQConfig + Marlin/Triton）

> 源码（扁平 `.py`，**不存在 `awq/` 子目录**）：
> - `vllm/model_executor/layers/quantization/auto_awq.py`（`AutoAWQConfig` + Linear/MoE methods）
> - `vllm/model_executor/layers/quantization/awq_triton.py`（Triton dequant kernel，4-bit group 32/64/128）
>
> 注册方法名：`awq` / `auto_awq` / `awq_marlin` 三者均映射到 `AutoAWQConfig`（`__init__.py:141`）。

---

## 是什么

AWQ（[Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)）：4-bit INT group 量化（group 32/64/128/None），weight-only，激活保持 bf16/fp16。vLLM 的 `AutoAWQConfig`（`auto_awq.py:163` 之后）消费 AutoAWQ/GPTQModel 导出的 AWQ-format checkpoint，并在加载期把 AWQ 的非标准位序重打包为标准 GPTQ/Marlin 格式，交给 MP（mixed-precision）Linear kernel 框架 + Marlin/Machete/Triton 内核。

关键点：

- **AWQ 位序转换**（`auto_awq.py:73`）：AWQ 在 int32 内按 `[0,4,1,5,2,6,3,7]` 排列 4-bit 值，vLLM 用 `_REVERSE_AWQ_PACK_ORDER` 反转并沿输入维重打包（`_convert_awq_to_standard_format`，`auto_awq.py:93`）以匹配 `MPLinearKernel` 框架期望的格式。
- **MoE WNA16**：`auto_awq.py` 通过 `oracle/int_wna16` 的 `select_wna16_moe_backend`/`make_wna16_moe_kernel` 服务 AWQ-Marlin MoE（也由 `moe_wna16.py` 复用，见 [moe-wna16.md](moe-wna16.md)）。
- **Triton dequant**（`awq_triton.py`）：`awq_dequantize_kernel`（`:11`）支持 group_size ∈ {-1,32,64,128}（`AWQ_TRITON_SUPPORTED_GROUP_SIZES`，`:8`），作为无 Marlin 硬件时的路径。

---

## 为什么

- **W4A16 是显存受限场景的主力**。AWQ 4-bit 近乎无损、激活不量化、与 Marlin/Machete kernel 配合在 Ampere/Hopper 上吞吐优秀。
- **统一 MP kernel 框架**。AWQ/GPTQ/Marlin 共用 `choose_mp_linear_kernel`（`auto_awq.py:15`）+ `MPLinearLayerConfig`，按 capability 在 Marlin/Machete/ExllamaV2/Triton 间自动挑选。
- **支持 MoE**。`MoeWNA16Config` 的 AWQ 路径与 `auto_awq` 共享 `oracle/int_wna16`，服务 W4A16 MoE 模型。

---

## 怎么做

### override 探测（`auto_awq.py:260`，`override_quantization_method`）

`AutoAWQConfig.override_quantization_method` 检查 checkpoint `quant_method` 是否为 `"awq"`/`"awq_marlin"`，命中即认领。在 `ModelConfig._verify_quantization` 的优先级表里，AWQ 系位于 GPTQ 系之后（`vllm/config/model.py:1033`）。

### 权重加载与重打包

`create_weights` 注册 packed weight（int32，`PackedvLLMParameter`）+ group scale/zero（`GroupQuantScaleParameter`），`packed_dim`/`pack_factor` 属性供 TP 切分。`process_weights_after_loading` 调用 `_convert_awq_to_standard_format`（`auto_awq.py:93`）：

1. 解包 int32 → 单个 4-bit（按 AWQ 位序）。
2. `_REVERSE_AWQ_PACK_ORDER`（`:77`）反转位序为标准。
3. 沿输入维重打包为 `(K//pack, N)` int32（`packed_dim=0`），匹配 MP kernel 框架。

### Linear kernel 选择

`choose_mp_linear_kernel(MPLinearLayerConfig)`（`vllm/model_executor/kernels/linear/`）候选：Marlin（Ampere+，speed）、Machete（Hopper）、ExllamaV2、Triton（`awq_triton.py`）。`auto_awq.py` 顶部 `import vllm.model_executor.layers.fused_moe  # noqa` 触发 MoE 注册。

### MoE 路径

WNA16 MoE backend（`oracle/int_wna16.py`）：`select_wna16_moe_backend` → `convert_to_wna16_moe_kernel_format` → `make_wna16_moe_kernel`/`make_wna16_moe_quant_config`。`check_moe_marlin_supports_layer`（`utils/marlin_utils.py`）做能力校验。

### Triton 后备（`awq_triton.py`）

`awq_dequantize_kernel`（`:11`）：2D grid，解包 int32 用 `tl.interleave`×3 + reverse_awq_order 重构 8 个 4-bit 值，乘 scale 减 zero 反量化到 fp16/bf16。仅支持 4 种 group_size。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`AutoAWQConfig.get_min_capability` 与 `current_platform.get_device_capability` 决定 Marlin 可用性；`check_marlin_supported`/`verify_marlin_supported` 集中在 `utils/marlin_utils.py`。
- **[分布式](../../../07-distributed/README.md)**：packed weight 的 `packed_dim=0`（输入维）+ pack_factor，TP 切分由 `weight_loader` 按 `logical_widths` 拆；`marlin_repeat_scales_on_all_ranks`（`utils/marlin_utils.py`）处理 scale 副本。
- **[编译-IR](../../../09-compilation-ir/README.md)**：MP kernel 走 custom op；Triton 路径 torch.compile 友好。
- **MoE**（`layers/fused_moe/`）：`oracle/int_wna16.py` 提供 backend 选择，`moe_wna16.py` 复用（见 [moe-wna16.md](moe-wna16.md)）。
- **GPTQ 共享框架**：AWQ 与 GPTQ 共用 `MPLinearLayerConfig`/`choose_mp_linear_kernel` 与 `scalar_types`（`uint4b8`，`auto_gptq.py:102` TYPE_MAP）。

---

## 历史版本演进

- **v0.5 及之前**：AWQ 首次接入，Marlin AWQ 内核成熟。
- **v0.6–v0.7**（待核实）：`awq_marlin` 方法名合并到 `AutoAWQConfig`（`__init__.py:142`）；MP kernel 框架统一 GPTQ/AWQ。
- **v0.8–v0.9**（待核实）：AWQ MoE（WNA16）支持，`oracle/int_wna16` backend 调度；safetensors 参数元数据驱动打包（`get_safetensors_params_metadata`）。
- **v0.10–v0.12 / main**（待核实）：`_convert_awq_to_standard_format` 位序处理稳化；Machete（Hopper）路径补强；与 moe_wna16 共享后端。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [gptq.md](gptq.md) · [moe-wna16.md](moe-wna16.md) · [utils.md](utils.md) · [schemes.md](schemes.md)
