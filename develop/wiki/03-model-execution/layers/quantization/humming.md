[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > Humming

# humming — Humming schema 驱动通用量化

> 源码（扁平 `.py`）：`vllm/model_executor/layers/quantization/humming.py`（829 行）
> 工具：`vllm/model_executor/layers/quantization/utils/humming_utils.py`
> 底层：`vllm/utils/humming.py`（`BaseInputSchema`/`BaseWeightSchema`/`HummingInputSchema`/`HummingWeightSchema`）
>
> 方法名 `"humming"`（`__init__.py:164`）。

---

## 是什么

`HummingConfig`（`humming.py` 声明于连载中，`override_quantization_method` 在 `:197`）是 vLLM 内一种**由 schema（量纲描述）驱动**的通用量化方案。它不固定某一种 (dtype, group, static/dynamic) 组合，而是读取 checkpoint 的"权重 schema / 输入 schema"描述，动态映射到 `QuantKey`（见 [schemes.md](schemes.md)）并选择对应内核。可以说 Humming 是"把任意量化布局描述翻译成 vLLM 内核选择 key"的通用前端，介于 compressed-tensors（声明式 config_groups）与具体 kernel 之间。

关键组件：
- `prepare_padded_shape`/`prepare_param`（`:64/69`）：按 padding 对齐与 schema 字段（`scale_type`/`packed_dim`/`output_dim`/`input_dim`）选择参数类（`PackedvLLMParameter`/`BlockQuantScaleParameter`/`ChannelQuantScaleParameter`/`GroupQuantScaleParameter`/`PerTensorScaleParameter`/`ModelWeightParameter`/`RowvLLMParameter`）。
- schema↔QuantKey 映射（`utils/humming_utils.py`）：`weight_schema_to_quant_key`/`input_schema_to_quant_key`（`humming.py:36/39` import）。
- MoE：`select_humming_moe_experts`/`convert_to_humming_moe_kernel_format`/`get_humming_moe_quant_config`/`make_humming_moe_kernel`（`utils/humming_utils.py`，`:34-40` import）。

---

## 为什么

- **支持新量化格式免写新 Config**。只要 checkpoint 描述了 schema（每层的权重 dtype/scale 形状/是否 packed/激活量化描述），Humming 就能映射到现有内核，降低新格式接入成本。
- **schema 驱动统一参数创建**。`prepare_param` 把 schema 字段映射到 vLLM `parameter` 体系（`vllm/model_executor/parameter.py`），TP 切分/加载器自动正确。
- **MoE 通用 backend 调度**。Humming MoE 复用与 FP8/MXFP4 相同的 modular kernel 框架（`make_humming_moe_kernel`）。
- **用户显式优先**。`ModelConfig._verify_quantization` 在 `self.quantization=="humming"` 时把 `humming` 提到 override 表首位（`vllm/config/model.py:1051`），确保用户强制使用 Humming 时不被其它方案抢认领。

---

## 怎么做

### override 认领（`:197`）

`HummingConfig.override_quantization_method` 检查 checkpoint 是否为 Humming 格式（具体判定字段待核实：`quant_method`/`humming` schema 字段）。用户 `--quantization humming` 时强制优先。

### 参数创建（`prepare_param`，`:69`）

按 extra_attrs 选参数类：
- `packed_dim` 存在 → `PackedvLLMParameter`
- `scale_type` ∈ {block/tensor/group/channel/input_scale} → 对应 scale 参数类
- 仅 `input_dim` → `RowvLLMParameter`；仅 `output_dim` → `ChannelQuantScaleParameter`
- `input_dim`+`output_dim` → `ModelWeightParameter`

### schema → QuantKey

`weight_schema_to_quant_key(schema)`/`input_schema_to_quant_key(schema)`（`utils/humming_utils.py`）：把 `HummingWeightSchema`/`HummingInputSchema`（dtype/scale 形状/static/dynamic）翻译为 `QuantKey`，供 Linear kernel 选择器（`init_*_linear_kernel`）与 MoE backend 选择（`select_humming_moe_experts`）使用。

### MoE 流程

`HummingConfig.get_quant_method` 对 `RoutedExperts` 返回 Humming MoE method，内部 `select_humming_moe_experts` 选 backend → `convert_to_humming_moe_kernel_format` 改排 → `make_humming_moe_kernel`/`get_humming_moe_quant_config` 构造 modular kernel。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`QuantKey` 决定的内核再由平台 capability 过滤。
- **[分布式](../../../07-distributed/README.md)**：参数类的 `input_dim`/`output_dim`/`packed_dim` 供 TP 切分；MoE backend 受 TP 影响。
- **[编译-IR](../../../09-compilation-ir/README.md)**：modular MoE kernel 与编译协同。
- **`vllm/utils/humming.py`**：schema 定义（非本目录）；`vllm/model_executor/parameter.py`：参数类体系。
- **定制 Linear / FusedMoE**：复用 `init_*_linear_kernel` 与 modular kernel 框架。

---

## 历史版本演进

- **v0.10–v0.11**（待核实）：Humming 首次引入，schema 驱动机制建立。
- **v0.11**（待核实）：`self.quantization=="humming"` 强制优先规则加入（`vllm/config/model.py:1051`）。
- **v0.12 / main**（待核实）：MoE backend 调度与 modular kernel 整合；schema↔QuantKey 映射随新 dtype（NVFP4/MXFP8）扩展。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [schemes.md](schemes.md) · [utils.md](utils.md) · [modelopt.md](modelopt.md) · [compressed-tensors.md](compressed-tensors.md)
