[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > Quark

# quark — AMD Quark 量化格式

> 源码目录：`vllm/model_executor/layers/quantization/quark/`
>
> 方法名 `"quark"`（`__init__.py:157`）。Quark = AMD 的量化导出工具（[quark](https://github.com/amd/quark)），主要面向 ROCm/NVIDIA/格式移植。

---

## 是什么

`QuarkConfig`（`quark/quark.py:54`）消费 Quark 导出的 checkpoint（`export` 字段、`layer_quant_config`、`kv_cache_group`、`pack_method`）。与 compressed-tensors 类似，Quark 也是声明式多 layer 配置，但 schema 不同（`layer_quant_config` 按 layer 名正则匹配，每层含 weight/input/output `QuantizationArgs`）。vLLM 在 `get_scheme()` 把这些声明映射到内部 `QuarkScheme` 子类。

目录：

| 路径 | 角色 |
|---|---|
| `quark.py` | `QuarkConfig`、`QuarkLinearMethod`、`QuarkKVCacheMethod`、`get_scheme`分发、`maybe_update_config`（DeepSeek-V3 系动态 mxfp4） |
| `quark_moe.py` | `QuarkMoEMethod` + `get_moe_method` 路由 |
| `schemes/quark_scheme.py` | 抽象 `QuarkScheme`（见 [schemes.md](schemes.md)） |
| `schemes/quark_w8a8_fp8.py` | W8A8 FP8 |
| `schemes/quark_w8a8_int8.py` | W8A8 INT8 |
| `schemes/quark_w4a8_mxfp4_fp8.py` | W4A8 MXFP4/Fp8 混合 |
| `schemes/quark_nvfp4.py` | NVFP4 |
| `schemes/quark_ocp_mx.py` | OCP MX（mxfp4/mxfp8） |
| `utils.py` | `deep_compare`、`should_ignore_layer` |

---

## 为什么

- **承接 AMD 量化生态**。Quark 是 ROCm 上主流量化器，MI300X/MI325X checkpoint 经此路径进入 vLLM。
- **DeepSeek-V3 系 fp4 特化**。`maybe_update_config`（`:75`）在 DeepSeek-V3/V3.2 + fp4 权重时开启 `dynamic_mxfp4_quant`（实际默认禁用，因动态量化开销抵消增益，`:69` 注释），保留未来开关。
- **KV-cache group 校验**。`from_config`（`:177`）严格校验 `kv_cache_group` 在 `layer_quant_config` 中能匹配到一致的 KV 量化配置，避免不一致。
- **复用通用工具**。`should_ignore_layer`/`find_matched_target` 与 compressed-tensors 共享逻辑风格；MoE 走 `QuarkMoEMethod.get_moe_method` 按 scheme 路由。

---

## 怎么做

### `from_config`（`:177`）

读 `export`（含 `kv_cache_group`/`pack_method`）、`layer_quant_config`。对 `kv_cache_group` 用 `fnmatch` 模式匹配 `layer_quant_config` 名，收集对应层并校验所有匹配层 output_tensors 配置一致（`deep_compare`，`:222`）。

### `get_quant_method`（`:143`）

- `should_ignore_layer(prefix, exclude_layers, fused_mapping)` → `UnquantizedLinearMethod`（除非 DeepSeek-V3 系 + `dynamic_mxfp4_quant` + LinearBase，走 `get_scheme(..., dynamic_mxfp4_quant=True)`）。
- `LinearBase` → `get_scheme` 存 `layer.scheme`，返回 `QuarkLinearMethod`。
- `Attention` → `QuarkKVCacheMethod`（继承 `BaseKVCacheMethod`）。
- `RoutedExperts` → `QuarkMoEMethod.get_moe_method(self, module=layer, layer_name=prefix)`。

### `get_scheme`

按 weight/input `QuantizationArgs` 的 dtype/strategy/dynamic 组合挑选 `QuarkScheme` 子类（W8A8Fp8/Int8、W4A8_MXFP4_FP8、NVFP4、OCP_MX）。`layer.scheme` 由 `QuarkLinearMethod` 委托调 `create_weights`/`apply_weights`/`process_weights_after_loading`。

### MoE 路由（`quark_moe.py`）

`QuarkMoEMethod.get_moe_method` 按 scheme 选 backend（FP8/MXFP4/NVFP4/INT8），与 `compressed_tensors_moe`、`oracle/` 协作。

### `apply_vllm_mapper`

`QuarkConfig.apply_vllm_mapper`（待核实行号）把 `layer_quant_config` 的层名按 HF→vLLM 模块名映射更新，使 `exclude_layers`/target 在 vLLM 结构下正确匹配。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：min capability 70（`:109`）；主要面向 ROCm（MI300X/MI325X/MI325_OAM），但 NVIDIA/H20/A100/B200 checkpoint 亦支持（见 `utils/configs/` 覆盖矩阵）。
- **[分布式](../../../07-distributed/README.md)**：`packed_modules_mapping` + fnmatch 匹配考虑 fused 模块；MoE backend 受 TP 影响。
- **[编译-IR](../../../09-compilation-ir/README.md)**：scheme `apply_weights` 走 custom op/内核。
- **compressed-tensors 对照**：两者都是声明式多方案框架，但 Quark 用 `layer_quant_config`+`export` 而 CT 用 `config_groups`+`CompressionFormat`；scheme 抽象同形（见 [schemes.md](schemes.md)）。
- **KV-cache**：`QuarkKVCacheMethod` 复用 `BaseKVCacheMethod`（见 [kv_cache.md](kv_cache.md)）。

---

## 历史版本演进

- **v0.7–v0.8**（待核实）：Quark 首批支持（W8A8 FP8/INT8）。
- **v0.9–v0.10**（待核实）：W4A8 MXFP4/Fp8、NVFP4、OCP_MX scheme 加入；DeepSeek-V3 系动态 mxfp4 钩子（默认关闭）。
- **v0.11–v0.12 / main**（待核实）：`maybe_update_config` 按 model_type 触发；KV-cache group 一致性校验强化；MoE modular kernel 整合。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [compressed-tensors.md](compressed-tensors.md) · [schemes.md](schemes.md) · [kv_cache.md](kv_cache.md) · [fp8.md](fp8.md) · [utils.md](utils.md)
