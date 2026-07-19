# QuantizationConfigArgs + QuantSpec（quantization.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > QuantizationConfigArgs

源码：`vllm/config/quantization.py`（约 183 行）。本模块提供用户侧的"在线量化"配置规范：`QuantSpec` 描述单一层类（linear/MoE）的 weight/activation 量化键；`QuantizationConfigArgs` 聚合 linear/moe spec 与 ignore 列表；`_ONLINE_SHORTHANDS` 把 CLI `--quantization` 简写（如 `fp8_per_tensor`/`fp8_per_block`/`mxfp8`/`int8_per_channel_weight_only`）展开为完整 `QuantizationConfigArgs`。它挂在 `ModelConfig.quantization_config`，被 `VllmConfig._get_quantization_config` 经 `resolve_quantization_config` 解析后派生出真正的 `QuantizationConfig`（`vllm/model_executor/layers/quantization/base_config.py`，**非** `vllm/config/`）。

## 是什么

### `QuantSpec`（`quantization.py:63`，`@config`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `weight` | `None` | weight 量化键（`QuantKey` 或 `QUANT_KEY_NAMES` 名） |
| `activation` | `None` | activation 量化键 |

`None` 表示该方法类回退自身默认（典型：继承 checkpoint，或在线时未量化）。

`QuantKeyField`（`quantization.py:53`）：`Annotated[QuantKey | None, GetPydanticSchema(...)]`，阻止 Pydantic 内省 `QuantKey`（含 `ClassVar[GroupShape]` NamedTuple，Pydantic 拒绝），改用 `_coerce_quant_key` 手动校验（字符串→`QUANT_KEY_NAMES` 查 `QuantKey`）。

### `QuantizationConfigArgs`（`quantization.py:78`，`@config`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `linear` | `None` | 应用到 `LinearBase` 层的 spec |
| `moe` | `None` | 应用到 `FusedMoE` 层的 spec |
| `ignore` | `[]` | 跳过量化的层名列表 |

校验器 `_coerce_spec`（`linear`/`moe` before）：字符串→若在 `_ONLINE_SHORTHANDS` 则取该简写的对应字段 spec；否则当 weight 键 `QuantSpec(weight=_coerce_quant_key(v))`。

### `QUANT_KEY_NAMES`（`quantization.py:24`）

用户名→`QuantKey`（来自 `vllm/model_executor/layers/quantization/utils/quant_utils.py`）映射：`fp8_per_tensor_static`/`fp8_per_tensor_dynamic`/`fp8_per_token`/`fp8_per_channel_static`/`fp8_per_block_static`/`fp8_per_block_dynamic`/`mxfp8`/`mxfp4`/`int8_per_channel_static`。

### `_ONLINE_SHORTHANDS` 与解析

`_ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs]`（`quantization.py:114`）把 CLI `--quantization` 简写展开为完整 args：

| 简写 | linear spec | moe spec |
|---|---|---|
| `fp8_per_tensor` | `kFp8StaticTensorSym` | `kFp8StaticTensorSym` |
| `fp8_per_block` | `kFp8Static128BlockSym` | `kFp8Static128BlockSym` |
| `fp8_per_channel` | `kFp8StaticChannelSym` | `kFp8StaticChannelSym` |
| `mxfp8` | `kMxfp8Dynamic` | `kMxfp8Dynamic` |
| `int8_per_channel_weight_only` | （无 linear） | `kInt8StaticChannelSym` |

`ONLINE_QUANT_SHORTHAND_NAMES = (*_ONLINE_SHORTHANDS.keys(), "online")`。

`resolve_quantization_config(quantization, quantization_config)`（`quantization.py:147`）：
1. `quantization`（CLI 简写）不在 `ONLINE_QUANT_SHORTHAND_NAMES` 且 `quantization_config` 非 None → raise（在线配置仅支持在线简写）。
2. `quantization` 非 online 简写且无 `quantization_config` → 返回 `None`（走离线 checkpoint 量化）。
3. `base = _ONLINE_SHORTHANDS.get(quantization)`；`quantization_config` 为 dict 则构造 `QuantizationConfigArgs`。
4. `base`/`quantization_config` 都有则合并（`quantization_config` 字段优先，`None` 字段回退 `base`）。

> 无 `compute_hash`——`QuantizationConfigArgs` 是"用户配置面"，真正影响图形状的是派生出的 `QuantizationConfig`（在 `model_executor/layers/quantization/`），后者哈希已被 `ModelConfig.quantization` 字符串覆盖进 `ModelConfig.compute_hash`。

## 为什么

- **在线 vs 离线量化分离**：离线量化（checkpoint 已量化，如 AWQ/GPTQ checkpoint）：`--quantization awq`，直接从 checkpoint 读量化配置。在线量化（运行时量化未量化 checkpoint）：`--quantization fp8_per_tensor` + 可选 `--quantization-config`，用 `QuantizationConfigArgs` 声明量化规范。`resolve_quantization_config` 区分二者。
- **简写展开**：`fp8_per_tensor` 等简写避免用户了解 `QuantKey` 细节；`_ONLINE_SHORTHANDS` 提供开箱即用规范，`--quantization-config` 允许覆盖（如仅 MoE 量化、linear 不量化：`int8_per_channel_weight_only` 简写 `linear` 为 None）。
- **`QuantKey` Pydantic 适配**：`QuantKey` 含 `ClassVar` NamedTuple，Pydantic 内省会拒绝。`QuantKeyField` 用 `GetPydanticSchema` 阻止内省，改 `_coerce_quant_key` 手动字符串→`QuantKey`，让用户用字符串名配置。
- **`ignore` 跳层**：某些层（如 embedding/最终 logits）不量化，`ignore` 列表按层名排除。
- **派生 `QuantizationConfig`**：`VllmConfig._get_quantization_config` 用 `resolve_quantization_config` 得 `QuantizationConfigArgs`（或 None），再调用对应 `QuantizationConfig` 子类构造（如 `Fp8Config`/`Mxfp8Config`），挂在 `VllmConfig.quant_config`。该校验含 dtype/capability 兼容性检查。

## 怎么做

- **在线 FP8**：`--quantization fp8_per_tensor`（linear+moe 都静态 per-tensor FP8）。
- **仅 MoE INT8**：`--quantization int8_per_channel_weight_only`（linear 不量化，moe INT8 per-channel）。
- **自定义 spec**：`--quantization online --quantization-config '{"linear":{"weight":"fp8_per_token"},"moe":{"weight":"fp8_per_block_dynamic"}}'`。
- **跳层**：`--quantization-config.ignore '["lm_head","re:.*embedding.*"]'`。
- **离线 checkpoint 量化**：`--quantization awq`（不配 `quantization_config`）。

## 与其它模块/系统配合

- **`ModelConfig`（[model-config.md](model-config.md)）**：`quantization`（简写/方法名）+ `quantization_config`（`QuantizationConfigArgs`）字段；`ONLINE_QUANT_SHORTHAND_NAMES` 决定是否走在线路径。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`_get_quantization_config(model_config, load_config)` 调 `get_quant_config` + `resolve_quantization_config` 派生 `quant_config`；校验 dtype/capability 兼容；`quant_config.use_deep_gemm` 自动禁用（Blackwell 上某些 model_type）。
- **量化子系统（[`03-model-execution/layers/quantization/`](../03-model-execution/layers/quantization/README.md)）**：`QuantizationConfig` 子类（`Fp8Config`/`Mxfp8Config`/`AwqConfig`/`GptqConfig`/...）是真正图形状影响者，被 layer 注册表消费；`QuantKey`/`kFp8*`/`kMxfp*` 常量在 `quant_utils.py`。
- **KernelConfig（[kernel-config.md](kernel-config.md)）**：`moe_backend`/`linear_backend` 与量化方法协同（如 DeepGEMM 仅 FP8 block-quant；Marlin 仅 weight-only）。
- **LoadConfig（[load-config.md](load-config.md)）**：`get_quant_config(model_config, load_config)` 读 checkpoint 量化参数（离线路径）。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`has_blocked_weights()`（`quant_config.weight_block_size`）触发 `+quant_fp8` custom op；blocked weights 影响图形状。

## 历史版本演进

- **v0.5–v0.8**：量化仅 `--quantization awq/gptq/fp8`（离线 checkpoint）；`QuantizationConfig` 子类体系在 `model_executor/layers/quantization/`。
- **v0.9（待核实）**：`quantization_config: QuantizationConfigArgs` 字段引入 `ModelConfig`；在线量化（`fp8_per_tensor` 等简写 + `online`）；`QuantSpec`/`QuantKeyField` Pydantic 适配。
- **v0.10**：`fp8_per_block`/`mxfp8`/`mxfp4`/`int8_per_channel_weight_only` 简写扩充；`resolve_quantization_config` 合并逻辑；`ignore` 列表。
- **v0.11 / v0.12 / main**：`fp8_per_block_dynamic`；`QuantKey` 体系扩充；`QuantizationConfig.use_deep_gemm` 自动禁用（Blackwell 精度问题）；与 NVFP4/MXFP8 backend 协同。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — `quantization`/`quantization_config` 字段。
- [vllm-config.md](vllm-config.md) — `_get_quantization_config` 派生 `quant_config`。
- [kernel-config.md](kernel-config.md) — `moe_backend`/`linear_backend` 与量化方法协同。
- [load-config.md](load-config.md) — `get_quant_config` 读 checkpoint 量化参数。
- [../03-model-execution/layers/quantization/README.md](../03-model-execution/layers/quantization/README.md) — `QuantizationConfig` 子类（真正图形状影响者）。
