[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > FBGEMM FP8

# fbgemm — FBGEMM FP8 量化（已废弃）

> 源码（扁平 `.py`，**不存在 `fbgemm/` 子目录**）：`vllm/model_executor/layers/quantization/fbgemm_fp8.py`
>
> 方法名 `"fbgemm_fp8"`（`__init__.py:145`），**已废弃**（`DEPRECATED_QUANTIZATION_METHODS`，`__init__.py:49`），需 `--allow-deprecated-quantization`。

---

## 是什么

`FBGEMMFp8Config`（`fbgemm_fp8.py:45`）消费 Meta 的 FBGEMM FP8 checkpoint：per-channel FP8 weight + 动态 per-token FP8 activation（`kFp8DynamicTokenSym`/`kFp8StaticTokenSym`，`fbgemm_fp8.py:30`），带 `input_scale_ub`（activation scale upper bound，用于动态上限裁剪）。仅 Linear，无 MoE/KV-cache 分支（`get_quant_method` 只处理 `LinearBase`，`:82`）。

- min capability 80（Ampere，`:67`）。
- `ignore_list`（`modules_to_not_convert`）跳层。
- `use_marlin`（`:55`）：当 `not current_platform.has_device_capability(89)`（即 <Ada Lovelace）时用 Marlin FP8 weight-only 内核（`prepare_fp8_layer_for_marlin`，`utils/marlin_utils_fp8.py`）。

---

## 为什么

- **历史承接 Meta 系模型**。Llama 等用 FBGEMM 导出的 FP8 per-channel checkpoint 需要本路径。
- **为 `fp8` 通用方案让路**。`Fp8Config` 已覆盖 per-tensor/block，偏好更新式样；FBGEMM per-channel + input_scale_ub 的 niche 场景被合并到 `compressed-tensors`/`fp8` 的 per-channel 路径，故主线将其标 deprecated。
- **Marlin 回退**。对无 FP8 硬件（sm80-sm88）提供 Marlin FP8 weight-only 兜底，是早期 Ada 前硬件的可行路径。

---

## 怎么做

### `from_config`（`fbgemm_fp8.py:74`）

读 `modules_to_not_convert`→`ignore_list`、`activation_scale_ub`→`input_scale_ub`。

### `FBGEMMFp8LinearMethod`（`:93`）

- `create_weights`（`:99`）：`weight`（`ModelWeightParameter`，FP8）、`weight_scale`（`ChannelQuantScaleParameter`，per-output-channel）。记录 `logical_widths`/`input_size_per_partition`/`orig_dtype`。
- `process_weights_after_loading`：FNUZ 归一化（`normalize_e4m3fn_to_e4m3fnuz`，`:34` import）；若 `use_marlin` 则 `prepare_fp8_layer_for_marlin`；否则建 `FP8LinearKernel`（`init_fp8_linear_kernel`，`:13`，per-token dynamic）。
- `apply`：kernel `apply_weights`，含 `input_scale_ub` 裁剪逻辑（待核实具体裁剪调用点）。

### KV-cache / MoE

均不实现（`get_quant_method` 对非 Linear 返回 `None`，`:90`）。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`has_device_capability(89)` 决定 Marlin 回退；`is_fp8_fnuz()` 决定 FNUZ 缩放。
- **[编译-IR](../../../09-compilation-ir/README.md)**：走 `FP8LinearKernel` custom op。
- **utils**：`marlin_utils_fp8.py`（Marlin 准备）、`w8a8_utils.py`（FNUZ）、`quant_utils.py`（`kFp8DynamicTokenSym`/`kFp8StaticTokenSym`）。

---

## 历史版本演进

- **v0.6–v0.7**（待核实）：FBGEMM FP8 路径引入，服务 Meta Llama FP8 checkpoint。
- **v0.8–v0.9**（待核实）：per-channel + Marlin 回退路径加入。
- **v0.11 / main**：标记 `DEPRECATED`（`__init__.py:49`），由通用 `fp8`/compressed-tensors per-channel 取代；`ModelConfig._verify_quantization` 对 deprecated 方法强制 `--allow-deprecated-quantization`（`vllm/config/model.py:1105`）。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [fp8.md](fp8.md) · [utils.md](utils.md) · [schemes.md](schemes.md)
