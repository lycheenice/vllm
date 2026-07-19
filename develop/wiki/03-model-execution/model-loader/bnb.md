# BitsAndBytesModelLoader：在线 8/4-bit 量化

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **bitsandbytes**

> 源码：`vllm/model_executor/model_loader/bitsandbytes_loader.py`

---

## 是什么

`BitsAndBytesModelLoader`（`bitsandbytes_loader.py:56`）对应 `load_format="bitsandbytes"`。它读取 HF 原始高精度（fp16/bf16）权重 checkpoint，在加载过程中**在线**完成 bitsandbytes 8-bit/4-bit（NF4/F4）量化，产出带 `quant_state` 的量化参数，直接交给 `Fp8LinearMethod` 之外的 bnb quant method 使用。即"边加载边量化"，不需要预先量化好的 checkpoint。

它不继承 `DefaultModelLoader`，自己实现 `_get_weight_files`/`_prepare_weights`/`load_weights`，因为量化状态结构与普通 tensor 不同。

---

## 为什么

预先量化 checkpoint 需要离线转换工具链，且 bnb 量化状态（`QuantState`、双量化 `state2`）与 vLLM 参数布局耦合较深。在线量化让用户直接喂原始 HF 权重即可用 bnb 推理，降低门槛。代价是首次加载多一次 dequantize→quant 成本。

对 MoE 模型，还要把每个 expert 的 quant_state 融合成 `w13`/`w2` 的合并 quant_state（`_fuse_moe_quant_states`），以匹配 `RoutedExperts` 的 fused 布局。

---

## 怎么做

### 初始化与分类

`__init__`（`bitsandbytes_loader.py:61`）维护若干分类表：`unsharded_weights_modules`（不切）、`column_sharded_weights_modules`（列切）、`maybe_fused_weights_modules`（可能融合）、`target_modules`（bnb 支持的 transformers 模块名）、`tp_disabled_modules`、`expert_params_mapping`、`weight_mapper`、`pre_quant`/`load_8bit`/`is_pool_model` 标志。

`_init_state`（`bitsandbytes_loader.py:560`）从模型实例采集：`is_pooling_model`、`ParamMapping(get_packed_modules_mapping(model))`、MoE 的 `expert_params_mapping = get_moe_expert_mapping(model)`、`hf_to_vllm_mapper`、`_get_bnb_target_modules`、`_classify_module_sharding`。

### 权重文件获取

`_get_weight_files`（`bitsandbytes_loader.py:83`）：本地 glob 或远程 `list_repo_files` + `download_weights_from_hf`，按 `allowed_patterns` 找到匹配文件。`_prepare_weights`（`bitsandbytes_loader.py:119`）在此基础上处理 `consolidated.safetensors.index.json` 去重。

### 量化与加载

`load_weights`（文件后半段）大致流程：

1. 取 safetensors/pt 迭代器产出 `(name, tensor)`。
2. 用 `weight_mapper`（来自 `hf_to_vllm_mapper`）把 HF 名映射到 vLLM 名。
3. 用 `modules_mapping`（`ParamMapping`）处理融合模块（如 `qkv_proj` ← `q/k/v`）。
4. 用 `expert_params_mapping` 处理 MoE 专家权重映射。
5. 调 bnb `quantize_blockwise` 等 API 在线量化，为每个权重构造 `QuantState`；双量化（Double Quantization）在加载期提前 dequantize（`_dequantize_dq`，`bitsandbytes_loader.py:579`），避免推理期开销。
6. MoE：`_fuse_moe_quant_states`（`bitsandbytes_loader.py:613`）把各 expert 的 w1/w2/w3 quant_state 融合成 `w13`/`w2` 的合并 `QuantState`。
7. 按 TP（`column_sharded`/`unsharded`）分片后塞进模型参数。

### `_dequantize_dq`

对 `quant_state.nested=True` 的双量化状态，用 `dequantize_blockwise(quant_state.absmax, quant_state.state2)` 还原 absmax，清掉 `state2`/`offset`/`nested`。用推理期内存换加载期一次性开销。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `bitsandbytes`（外部包） | `QuantState`、`dequantize_blockwise` 等 |
| `layers/linear.py`（`LinearBase`/`MergedColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear`/`ReplicatedLinear`） | 目标模块类型判定 |
| `layers/fused_moe.py::RoutedExperts` | MoE 专家融合 |
| `model_executor/utils.py` | `get_packed_modules_mapping`、`get_moe_expert_mapping`、`set_weight_attrs` |
| `lora/utils.py::is_moe_model`、`models/__init__.py::is_pooling_model` | 模型类型判定 |
| `weight_utils.py` | 下载/迭代器复用 |
| `vllm/distributed` | `get_tensor_model_parallel_rank/world_size` 决定列切分 |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 中期（v0.6–v0.8，待核实） | 引入 `BitsAndBytesModelLoader` 支持 8bit/4bit NF4 在线量化 |
| 中期 | MoE 支持：`_fuse_moe_quant_states` 合并 expert quant_state |
| main | 双量化（Double Quantization）加载期 dequantize 优化 |
| main（#45308） | revision 按 name 传给 index 下载 |
| main（#44589/#47058） | 移除冗余 `load_weights` 方法 |

---

## 参见

- [`weight-utils.md`](weight-utils.md) —— 复用的下载/迭代器
- [`default.md`](default.md) —— 与默认加载的对比
- [`../README.md`](../README.md) —— 返回模型执行首页
