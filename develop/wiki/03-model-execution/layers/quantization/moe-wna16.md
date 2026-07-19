[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > MoE WNA16 / ExpertsInt8

# moe_wna16 — MoE W8A16/W4A16 与 ExpertsInt8

> 源码（扁平 `.py`）：
> - `vllm/model_executor/layers/quantization/moe_wna16.py`（`MoeWNA16Config`，491 行）
> - `vllm/model_executor/layers/quantization/experts_int8.py`（`ExpertsInt8Config`，60 行，遗留）
>
> 方法名：`moe_wna16`（`__init__.py:158`）、`experts_int8`（`:156`）。

---

## 是什么

### `MoeWNA16Config`（`moe_wna16.py:34`）

为 **MoE（`RoutedExperts`）层**提供 W8A16 / W4A16 量化（INT 权重 + bf16/fp16 激活），Linear 层可选量化。它本身不自建内核，而是把 Linear 端委托给 GPTQ 或 AWQ（`linear_quant_method ∈ {"gptq","awq","awq_marlin"}`，`:58/60`），MoE 端用 `fused_moe/config.py` 的 `int4_w4a16_moe_quant_config`/`int8_w8a16_moe_quant_config`（`:18/19` import）+ modular kernel。

关键字段（`:37`）：`weight_bits`(4/8)、`group_size`、`has_zp`、`lm_head_quantized`、`modules_to_not_convert`、`linear_quant_method`、`full_config`、`bit8_pack_factor`。构造期对 `awq` 系做 capability 校验（`:60-72`，复用 `AutoAWQConfig.get_min_capability()`）。

### `ExpertsInt8Config`（`experts_int8.py:22`）

在线 INT8 量化 MoE 专家权重，Linear 不量化（`get_quant_method` 对 `LinearBase` 返回 `UnquantizedLinearMethod`，`:56`；对 `RoutedExperts` 返回 `Int8OnlineMoEMethod`，`:59`）。**遗留**——代码注释明确建议用 `--quantization int8_per_channel_weight_only` 替代（`experts_int8.py:27`）。

---

## 为什么

- **MoE 量化专门入口**。很多模型仅 MoE 部分需低比特（Linear 保留高精度），`MoeWNA16Config` 把"Linear 走 gptq/awq + MoE 走 int4/int8 wna16"组合成一个 Config。
- **复用成熟 Linear 栈**。委托 GPTQ/AWQ 避免重写 W4A16 Linear 内核；MoE 端用 FusedMoE modular kernel 的 `int4/int8 wna16` quant config。
- **override 认领**。`MoeWNA16Config.override_quantization_method`（`:125`）认领 `moe_wna16` checkpoint，在优先级表里于 awq 之后、modelopt 之前（`vllm/config/model.py:1037`）。
- **ExpertsInt8 向后兼容**。早期 `--quantization experts_int8` 用户可继续用，新用户引导到 online `int8_per_channel_weight_only`（[online.md](online.md)）。

---

## 怎么做

### MoeWNA16 override（`:125`）

`override_quantization_method` 检查 checkpoint `quant_method=="moe_wna16"` + 字段齐全，认领。

### `get_quant_method`

- `LinearBase` → 若 `lm_head_quantized` 或非 skip → 按 `linear_quant_method` 委托（gptq 走 GPTQ Linear，awq 系走 AWQ Linear，复用 `auto_gptq.py`/`auto_awq.py` 的 method 类）；skip → `UnquantizedLinearMethod`。具体委托实现细节 (待核实)。
- `RoutedExperts` → `MoeWNA16MoEMethod`（待核实类名），用 `int4_w4a16_moe_quant_config`/`int8_w8a16_moe_quant_config` 构造 quant config，modular kernel 执行。

### MoE 内核路径

- `int4_w4a16_moe_quant_config` / `int8_w8a16_moe_quant_config`（`layers/fused_moe/config.py`，`:18/19` import）描述权重/scale 形状。
- modular kernel（`layers/fused_moe/`）按 quant config 选 backend（CUTLASS/Marlin-MoE/Triton…）。
- `FusedMoeWeightScaleSupported`（`:12` import）：group/tensor scale 模式标记，供重量化与加载器。

### ExpertsInt8 路径

`Int8OnlineMoEMethod`（`online/int8.py`，`experts_int8.py:17` import）——在线把 bf16 专家权重量化为 INT8 per-channel，前向动态量化激活。与 `online/` 共享实现（见 [online.md](online.md)）。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`current_platform.get_device_capability` 用于 AWQ 系 capability 校验；MoE backend 受平台影响。
- **[分布式](../../../07-distributed/README.md)**：MoE 专家权重按 expert 切分；`get_tensor_model_parallel_rank`/`get_tp_group`（`:8` import）。
- **GPTQ/AWQ 复用**：`auto_gptq.py`/`auto_awq.py` 的 method 类与 `oracle/int_wna16` backend。
- **FusedMoE**（`layers/fused_moe/`）：modular kernel + quant config；WNA16 MoE 也可能走 `oracle/int_wna16` 的 `WNA16MoEBackend`（与 auto_awq/auto_gptq 共享，见 [awq.md](awq.md)/[gptq.md](gptq.md)）。
- **online**：`experts_int8` 复用 `online/int8.py`。

---

## 历史版本演进

- **v0.7–v0.8**（待核实）：`MoeWNA16Config` 首次加入，支持 MoE W4A16/W8A16。
- **v0.9–v0.10**（待核实）：Linear 端委托 gptq/awq 完善；`oracle/int_wna16` backend 整合。
- **v0.11 / main**（待核实）：`experts_int8` 标注为遗留并指向 `int8_per_channel_weight_only`；`moe_wna16` override 优先级固定在 awq 之后。
- **v0.12 / main**：modular kernel 化；与 `moe_wna16` + awq capability 校验保持同步。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [awq.md](awq.md) · [gptq.md](gptq.md) · [online.md](online.md) · [utils.md](utils.md) · [schemes.md](schemes.md)
