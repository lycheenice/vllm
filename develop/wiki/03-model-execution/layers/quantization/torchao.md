[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > TorchAO

# torchao — torchao 量化解包代理

> 源码（扁平 `.py`）：`vllm/model_executor/layers/quantization/torchao.py`（399 行）
>
> 方法名 `"torchao"`（`__init__.py:159`）。依赖外部库 `torchao`（PyTorch 官方量化库）。

---

## 是什么

`TorchAOConfig`（`torchao.py` 声明在 `:90` 附近）是 vLLM 对 **PyTorch `torchao` 量化 checkpoint** 的运行时支持。与其它"自建内核"方案不同，torchao 方案以**解包（unwrap）**为主：vLLM 在加载期把 torchao 的 `torch.nn.Module` 量化子类还原为带原 weight 的 `LinearBase`，并尽量把常用布局（int4 weight-only 等）桥接到 vLLM 自有内核；不可桥接时回退到 torchao 自带 forward。

关键辅助（`torchao.py`）：
- `_get_weight_attrs`/`_restore_weight_attrs`（`:39/58`）：记录并恢复 weight 参数上的附加属性（torchao 量化态）。
- `torchao_version_at_least`（`:64`）：版本探测，按版本启用不同特性。
- `should_skip`（`:76`）：稳健的层跳过匹配（精确/equal/子路径），支持 skip_modules 列表。
- `_bond_method_to_cls`（`:31`）：把函数绑定为参数的方法（恢复 bound method）。

---

## 为什么

- **承接 torchao 生态**。torchao 是 PyTorch 官方量化库（int4/int8/fp8/各种布局），社区 checkpoint 多；vLLM 通过解包避免重写所有 torchao 布局内核。
- **尽力复用 vLLM 高性能内核**。能映射到 vLLM `LinearMethodBase` 的布局就映射，不能则回退 torchao，兼顾兼容与性能。
- **torchao 版本兼容**。`torchao_version_at_least` 让 vLLM 在多版本 torchao 间适配行为差异。

---

## 怎么做

### 配置接入

`TorchAOConfig.from_config` 读 checkpoint 的 `quant_method=="torchao"` 与 `torchao` 子配置（具体字段待核实）。`override_quantization_method` 不显式认领（走 `quant_method` 直配，故不在 `vllm/config/model.py` 的 `overrides` 优先级表内）。

### `get_quant_method`

对 `LinearBase`：
1. 若 `should_skip(prefix, skip_modules)` → `UnquantizedLinearMethod`。
2. 否则按 torchao 量化布局选择：可桥接的 → 对应 vLLM `LinearMethodBase`；否则 → torchao 解包/回退 method（待核实类名）。

### 解包与属性恢复

加载期：`_get_weight_attrs(param)` 记录 weight 上的量化态属性 → 解包为原始 weight tensor → `_restore_weight_attrs` 在需要时恢复（`_bond_method_to_cls` 处理 bound method）。这保证 torchao 的 `weight_quantizer`/`input_quantizer` 等状态不丢失，必要时回退 forward 可用。

### 前向

- 桥接到 vLLM 内核路径：调用 vLLM `LinearMethodBase.apply`（如 int4 weight-only 走 Marlin/Machete）。
- 回退路径：用 torchao 原生 `F.linear` + torchao 量化算子。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：min capability 由具体桥接方案决定；torchao 自身 kernel 的平台支持。
- **[编译-IR](../../../09-compilation-ir/README.md)**：torchao 内核对 `torch.compile` 的兼容性随版本改善；vLLM 桥接路径走 custom op。
- **外部库**：`torchao`（`find_spec("torchao")` 探测，`:65`）；`packaging.version` 版本比较。
- **Linear 层**：复用 `LinearMethodBase`/`UnquantizedLinearMethod`。

---

## 历史版本演进

- **v0.8–v0.9**（待核实）：torchao 方案首次接入，int4 weight-only 解包。
- **v0.10–v0.11**（待核实）：`torchao_version_at_least` 版本适配；`should_skip` 稳健匹配；属性记录/恢复机制完善。
- **v0.12 / main**（待核实）：更多布局桥接到 vLLM 内核；与 `torch.compile` 兼容性改进。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [schemes.md](schemes.md) · [utils.md](utils.md) · [gptq.md](gptq.md) · [bnb.md](bnb.md)
