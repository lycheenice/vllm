# TensorizerLoader：tensorizer 序列化加载

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **Tensorizer**

> 源码：`vllm/model_executor/model_loader/tensorizer_loader.py`、`vllm/model_executor/model_loader/tensorizer.py`

---

## 是什么

`TensorizerLoader`（`tensorizer_loader.py:43`）对应 `load_format="tensorizer"`，使用 CoreWeave 的 [tensorizer](https://github.com/coreweave/tensorizer) 库加载**已序列化**的模型权重。tensorizer 把整个 `state_dict` 序列化成紧凑二进制流，加载时直接反序列化到目标设备，比逐 shard safetensors 快很多。支持两种产物：

- **vLLM-tensorized**：用 `examples/features/tensorize_vllm_model.py` 预先把 vLLM 模型序列化，加载时 `init_tensorizer_model` + `deserialize_tensorizer_model`。
- **非 vLLM-tensorized**（仅 HF 序列化）：退化为 `_load_model_serialized_cpu`，仍比 HF 默认快，但不如 vLLM-tensorized。

`tensorizer.py`（30KB）实现 `TensorizerConfig`、`serialize_vllm_model`、`deserialize_tensorizer_model`、`init_tensorizer_model`、`is_vllm_tensorized`、`tensorizer_weights_iterator`。

---

## 为什么

大模型从对象存储/网络盘冷启动时，safetensors 的小张量随机读会放大延迟。tensorizer 把权重流式化为大块连续读，对 S3/GCS 友好，且能直接反序列化到 GPU。序列化是一次性成本，加载是高频成本，划算。

与 `ShardedStateLoader` 的区别：sharded_state 仍是 safetensors 格式只是预切；tensorizer 是自定义二进制格式，需预序列化。

---

## 怎么做

### `__init__`（`tensorizer_loader.py:46`）

`model_loader_extra_config` 既可以是 `TensorizerConfig` 实例，也可以是 dict（取 `["tensorizer_config"]`）。`validate_config` 拒绝 `BLACKLISTED_TENSORIZER_ARGS = {"device","dtype","mode"}`——这三项由 vLLM 决定，用户不能覆盖。

### `load_model`（`tensorizer_loader.py:115`）

- `_verify_config`：对 `model_config` 与 `parallel_config` 校验。
- TP>1 时 `tensorizer_uri` 用 `%` 占位，按 `tp_rank` 替换为各 rank 独立的序列化文件（`tensorizer_loader.py:121`）。
- `is_vllm_tensorized(config)` 为真：`_patch_tensorizer_config` 注入 `model_class`/`hf_config`/`dtype`，`with torch.device(...)` 下 `init_tensorizer_model`，再 `load_weights` → `deserialize_tensorizer_model`。
- 否则走 `_load_model_serialized_cpu`：CPU 上 `initialize_model` + `model.load_weights(tensorizer_weights_iterator(...))`。

### `load_weights`（`tensorizer_loader.py:103`）

vLLM-tensorized 走 `deserialize_tensorizer_model(model, tensorizer_config)`；否则 `model.load_weights(self._get_weights_iterator())`，用 `tensorizer_weights_iterator` 产出 `(name, tensor)`。

### `save_model`（静态，`tensorizer_loader.py:141`）

`serialize_vllm_model(model, tensorizer_config, model_config)`，用于离线序列化。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `loader/utils.py::initialize_model` / `get_model_architecture` | `_load_model_serialized_cpu` 用它构造模型 |
| `config/load.py::LoadConfig` | `model_loader_extra_config` 透传 `TensorizerConfig` |
| `vllm/distributed` | TP rank 替换 `tensorizer_uri` 占位符 |
| `config/ModelConfig`/`ParallelConfig` | `verify_with_model_config`/`verify_with_parallel_config` |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 早期 | 引入 `TensorizerLoader`，仅支持 HF 序列化路径 |
| 中期 | 支持 vLLM-tensorized 路径（`init_tensorizer_model` + `deserialize_tensorizer_model`） |
| main | `BLACKLISTED_TENSORIZER_ARGS` 收紧用户可配项；TP 占位符机制 |

---

## 参见

- [`default.md`](default.md) —— 与默认加载的对比
- [`../README.md`](../README.md) —— 返回模型执行首页
