[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > Intel Neural Compressor

# inc — Intel Neural Compressor 量化

> 源码目录：`vllm/model_executor/layers/quantization/inc/`
>
> 方法名 `"inc"`（`__init__.py:160`）。INC = [Intel Neural Compressor](https://github.com/intel/neural-compressor)，面向 Intel CPU/XPU/GPU。

> 说明：任务原始清单中的 `aik.md`（"AIK 量化"）在 vLLM 主线**不存在** `aik/` 目录或 `aik` 方法名（`__init__.py` 的 `QuantizationMethods` 中无 `aik`）。Intel 系量化在本仓库由 `inc/` 承担，故以本页替代 `aik.md`。

---

## 是什么

`INCConfig`（`inc/inc.py:32`）消费 Intel Neural Compressor 导出的 WNA16（int2/int3/int4/int8 weight-only + bf16/fp16 激活）checkpoint，支持 `auto_round:auto_gptq` 与 `auto_round:auto_awq` 两种 packing format，并通过 `backend` 字段在 `gptq`/`gptq:marlin`/`awq`/`awq:marlin`/`marlin`/`auto` 间挑选推理后端。

关键字段（`:49`）：`weight_bits`(∈{2,3,4,8}, `:37`)、`group_size`、`sym`、`packing_format`(∈{"auto_round:auto_gptq","auto_round:auto_awq"}, `:39`)、`data_type`(="int", `:38`)、`backend`(∈{"auto","gptq","gptq:marlin","awq","awq:marlin","marlin"}, `:40`)、`block_name_to_quantize`、`extra_config`、`pack_factor=Fraction(32,weight_bits)`（`:94`）。min capability 60（`:112`，远低于 GPU 方案，因面向 CPU/XPU）。

目录：

| 路径 | 角色 |
|---|---|
| `inc.py` | `INCConfig`、override 认领、`get_quant_method`、`apply_vllm_mapper` |
| `config_parser.py` | `INCConfigParser` + `INCLayerConfig`（解析 per-layer 量化配置） |
| `inc_linear.py` | `INCLinearMethod`（Linear 实现） |
| `schemes/inc_scheme.py` | 抽象 `INCScheme`（见 [schemes.md](schemes.md)） |
| `schemes/inc_wna16_scheme.py` | `INCWna16Scheme`（WNA16 具体方案） |
| `schemes/inc_wna16_linear.py` | WNA16 Linear 内核胶水 |
| `schemes/inc_ark_ops.py` | ARK ops（Intel 专用算子）路径 |
| `schemes/factory.py` | `resolve_scheme(layer_config)`（`:11`）分发到 `INCWna16Scheme` |

---

## 为什么

- **Intel 硬件栈闭环**。INC 是 Intel 官方量化器，CPU/XPU/Intel GPU checkpoint 经此路径运行；`get_min_capability=60` 与 `ark_ops` 反映非 NVIDIA 硬件定位。
- **复用 GPTQ/AWQ/Marlin 后端**。`backend` 字段让同一 INC checkpoint 在有 Marlin 的 GPU 上跑 Marlin、在 CPU/XPU 上跑原生路径，最大化复用。
- **override 认领**。`INCConfig.override_quantization_method`（`:185`）在优先级表里位于 awq 之后、moe_wna16 之前（`vllm/config/model.py:1036`）。
- **per-layer 解析**。`INCConfigParser` 把 INC 的 per-layer 配置解析为 `INCLayerConfig`，由 `resolve_scheme` 分发到具体 scheme。

---

## 怎么做

### `from_config`（`:119`）

读 `quantization_config.json`（`get_config_filenames` 返回 `["quantization_config.json"]`，`:116`），解析 `weight_bits`/`group_size`/`sym`/`packing_format`/`backend`/`block_name_to_quantize`/`extra_config`/`data_type`，构造 `INCConfig` + `INCConfigParser`。

### `get_quant_method`

按 layer 类型与 `block_name_to_quantize`：`LinearBase` → `INCLinearMethod`（`inc_linear.py`），内部 `INCConfigParser` 解析该层 → `resolve_scheme(layer_config)`（`schemes/factory.py:11`）→ `INCWna16Scheme`（目前唯一 scheme，`:12-16`）。`RoutedExperts`/`Attention` (待核实是否支持)。

### scheme 与 backend

`INCWna16Scheme`（`schemes/inc_wna16_scheme.py`）按 `backend`：
- `gptq`/`gptq:marlin` → 走 GPTQ/Marlin 内核栈（复用 `utils/gptq_utils.py`/`marlin_utils.py`）。
- `awq`/`awq:marlin` → 走 AWQ/Marlin 栈（复用 `auto_awq.py` 工具）。
- `marlin` → 直接 Marlin。
- `auto` → 按 capability 自动选。
- CPU/XPU/Intel GPU → `inc_ark_ops.py` 的 ARK ops 路径（待核实触发条件）。

### `apply_vllm_mapper` / `maybe_update_config`

`apply_vllm_mapper`（待核实行号）把 `block_name_to_quantize`/`modules_to_not_convert` 按 HF→vLLM 名映射更新，使其在 vLLM 模型结构下正确匹配。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`get_min_capability=60` 宽容；`current_platform` 决定 CPU/XPU/Intel GPU 分支与 ARK ops 可用性；`backend="auto"` 依赖平台 capability。
- **[分布式](../../../07-distributed/README.md)**：packed weight `packed_dim`+`pack_factor` 供 TP（虽然在 Intel 栈多为单进程）。
- **[编译-IR](../../../09-compilation-ir/README.md)**：ARK ops / Marlin 走 custom op；CPU torch.compile 路径 (待核实)。
- **GPTQ/AWQ 复用**：`utils/gptq_utils.py`、`utils/marlin_utils.py`、`auto_awq.py` 的 MP kernel 框架。
- **Linear 层**：`INCLinearMethod` 实现 `QuantizeMethodBase`。

---

## 历史版本演进

- **v0.10–v0.11**（待核实）：INC 首批接入，WNA16 + auto_round packing、`backend` 选择。
- **v0.11–v0.12 / main**（待核实）：`inc/schemes/` 子目录抽象（`INCScheme`/`factory`）、ARK ops 路径、`block_name_to_quantize`/`extra_config` 完善；override 优先级纳入 `vllm/config/model.py`。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [gptq.md](gptq.md) · [awq.md](awq.md) · [schemes.md](schemes.md) · [utils.md](utils.md) · [平台](../../../08-platforms/README.md)
