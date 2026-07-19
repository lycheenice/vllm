[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > GPTQ

# gptq — GPTQ 量化与 Marlin 系

> 源码（扁平 `.py`，**不存在 `gptq/`、`gptq_marlin/`、`marlin/` 子目录**）：
> - `vllm/model_executor/layers/quantization/auto_gptq.py`（`AutoGPTQConfig` + Linear/MoE methods）
> - Marlin 系工具集中在 `utils/`：`marlin_utils.py`、`marlin_utils_fp8.py`、`marlin_utils_fp4.py`、`machete_utils.py`、`gptq_utils.py`
>
> 注册方法名：`gptq` / `auto_gptq` / `gptq_marlin` 三者均映射到 `AutoGPTQConfig`（`__init__.py:151`）。优先级最高（`vllm/config/model.py:1030`）。

---

## 是什么

GPTQ（[论文](https://arxiv.org/abs/2210.17323)）：基于二阶信息的后训练权重量化，支持 INT4/INT8、group_size（含 -1=per-channel）、`desc_act`（激活顺序重排）、对称/非对称。vLLM 的 `AutoGPTQConfig`（`auto_gptq.py:97`）消费 AutoGPTQ/GPTQModel 导出的 checkpoint，并在运行期交给 MP（mixed-precision）kernel 框架，在 Marlin / Machete / ExllamaV2 / Triton 间自动选择。"`gptq_marlin`"不是独立 Config，而是 GPTQ checkpoint 走 Marlin 内核的代称。

关键特性：

- **`TYPE_MAP`**（`auto_gptq.py:101`）：`(num_bits, is_sym)` → `scalar_types`，目前 `(4,True)→uint4b8`、`(8,True)→uint8b128`（带 zero-point 偏置的对称布局，Marlin 期望）。
- **`desc_act` 激活重排**：`desc_act=True` && `group_size=-1` 时强制视作 `desc_act=False`（`:118`）。
- **per-module 动态规则**（`utils/gptq_utils.py`）：GPTQModel 的 `dynamic` config 允许 `+:`/`-:` 正则匹配模块覆盖基线配置或显式跳过（`auto_gptq.py:82` 的 `get_dynamic_override`）。
- **MoE WNA16**：通过 `oracle/int_wna16` 服务 GPTQ-Marlin MoE（与 AWQ 共享，见 [moe-wna16.md](moe-wna16.md)）。

---

## 为什么

- **GPTQ 是最成熟的 W4A16 量化之一**，社区 checkpoint 量大；Marlin 内核在 Ampere+ 提供近 fp16 吞吐。
- **统一 MP kernel 框架**：GPTQ/AWQ 共用 `choose_mp_linear_kernel`+`MPLinearLayerConfig`（`auto_gptq.py:13`），按 capability+形状选最优内核，用户无感。
- **灵活的 per-module 配置**：`dynamic` 支持同一模型不同层用不同 num_bits/group/skip，`get_moe_quant_method`（`auto_gptq.py:71`）为每个 MoE 层克隆 config 并 `override_config`。
- **优先级最高**：override 表中 GPTQ 系排第一，确保 GPTQ 格式不被其它方案误认领。

---

## 怎么做

### override 探测（`auto_gptq.py:219`，`override_quantization_method`）

检查 `quant_method` 是否为 `"gptq"`/`"gptq_marlin"`/`"auto_gptq"`，并校验 `bits`/`group_size`/`desc_act`/`sym` 等字段存在，命中即认领。

### Linear 流程

`AutoGPTQConfig.get_quant_method` → `LinearGPTQMethod`（经由 `get_linear_quant_method`，`utils/gptq_utils.py`）：
1. `create_weights`：注册 `PackedvLLMParameter`(int32 packed weight, `packed_dim=0`)、`GroupQuantScaleParameter`/`ChannelQuantScaleParameter`（scale/zero）、`PackedColumnParameter`（g_idx，desc_act 时）。
2. `process_weights_after_loading`：按 `desc_act`/group 重排 weight + g_idx；`marlin_repeat_scales_on_all_ranks`（TP>1 时副本）；调 MP kernel 的 `process_weights_after_loading`（Marlin workspace 等）。
3. `apply`：`MPLinearLayerConfig` → `choose_mp_linear_kernel` → 内核 `apply_weights(layer, x, bias)`。

### MoE 流程（`auto_gptq.py:71`，`get_moe_quant_method`）

每层 `deepcopy(config)` → `get_dynamic_override` 判定 skip（False）→ `override_config(prefix)` 应用该层覆盖 → 构造 `moe_method_cls(cloned_config, layer.moe_config)`。MoE method 内部用 `oracle/int_wna16` 选 backend（`WNA16MoEBackend`）、`convert_to_wna16_moe_kernel_format`、`make_wna16_moe_kernel`/`make_wna16_moe_quant_config`。

### Marlin 工具（`utils/marlin_utils*.py`）

- `marlin_utils.py`：`check_marlin_supported`/`verify_marlin_supported`/`check_marlin_supports_layer`/`check_moe_marlin_supports_layer`/`marlin_make_workspace_new`/`marlin_repeat_scales_on_all_ranks`/`get_marlin_input_dtype`。
- `marlin_utils_fp8.py`：Marlin FP8 weight-only 路径（`prepare_fp8_layer_for_marlin`，被 `fbgemm_fp8.py`/`fp8.py` 复用）。
- `marlin_utils_fp4.py`：Marlin FP4。
- `machete_utils.py`：Hopper Machete 混合精度 kernel 胶水。

### 动态规则（`utils/gptq_utils.py`）

- `get_dynamic_override(config, layer_name)`：按 `+:`/`-:` 正则匹配，返回 dict 或 `False`（skip）。
- `override_config(config, prefix)`：把命中的 override 字段并入 cloned config。
- `get_linear_quant_method`：综合 dynamic 规则挑 Linear method。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`get_min_capability` + `check_marlin_supported` 决定内核；ROCm/CPU 可能无 Marlin。
- **[分布式](../../../07-distributed/README.md)**：packed weight `packed_dim=0` + `pack_factor`；`marlin_repeat_scales_on_all_ranks` 保证 TP 各 rank scale 一致；g_idx 在 TP 下对齐。
- **[编译-IR](../../../09-compilation-ir/README.md)**：MP kernel 为 custom op；`g_idx` 重排需在加载期完成以兼容编译。
- **MoE**（`layers/fused_moe/`）：`oracle/int_wna16.py` 提供 GPTQ/AWQ 共享 backend。
- **AWQ 共享**：共用 `MPLinearLayerConfig`/`choose_mp_linear_kernel`/`scalar_types`。

---

## 历史版本演进

- **v0.5 及之前**：GPTQ + GPTQ-Marlin 首批支持，ExllamaV1/V2 内核。
- **v0.6–v0.7**（待核实）：`gptq`/`auto_gptq`/`gptq_marlin` 统一到 `AutoGPTQConfig`；Marlin 成为默认。
- **v0.8–v0.9**（待核实）：GPTQModel `dynamic` per-module 规则支持；MoE WNA16 与 `oracle/int_wna16` 整合；Machete（Hopper）路径。
- **v0.10–v0.12 / main**（待核实）：`gpt_oss_mxfp4` 等新方法分流；GPTQ-Marlin MoE backend 复用 `moe_wna16`；safetensors 元数据驱动打包。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [awq.md](awq.md) · [moe-wna16.md](moe-wna16.md) · [utils.md](utils.md) · [schemes.md](schemes.md)
