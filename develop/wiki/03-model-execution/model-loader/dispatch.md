# Loader Dispatch 与 Registry 对接

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **Dispatch**

> 源码：`vllm/model_executor/model_loader/__init__.py`、`vllm/model_executor/models/registry.py`

---

## 是什么

`model_loader/__init__.py` 是加载子系统的总入口，完成两件事：

1. **LoadFormat → Loader 分派**：用 `_LOAD_FORMAT_TO_MODEL_LOADER` 字典把字符串 `load_format` 映射到一个 `BaseModelLoader` 子类。
2. **顶层 `get_model` 入口**：把"选 loader → 实例化 loader → `load_model`"串成一个调用，供 Worker 直接使用。

模型类的解析则下沉到 `model_loader/utils.py::get_model_architecture`，再委托给 `models/registry.py::ModelRegistry.resolve_model_cls`，使 loader 与"模型架构注册表"解耦。

---

## 为什么

把分派做成一张显式字典（而非 if-else 链），是为了：

- 让 `register_model_loader` 装饰器能在运行时往这张字典里插值（out-of-tree loader）。
- 让 `LoadFormats` Literal 与字典 key 一一对应，新增 format 时 IDE/mypy 能立刻提示。
- `LoadConfig` 的 docstring 明确写着"Reminder: Please update docstring in `LoadConfig` if a new load format is added here"（`__init__.py:31`），保证配置侧与代码侧同步。

把"选模型类"从 loader 里挪到 registry，是为了让**任何 loader 都不必关心模型是 Llama 还是 Qwen**——loader 只产出 `(name, tensor)`，模型类由 `model_config.architectures` 决定。

---

## 怎么做

### LoadFormat → Loader 分派表

`__init__.py:33` 定义的 `LoadFormats` Literal（即 `LoadConfig.load_format` 的合法取值）与 `__init__.py:50` 的字典：

| load_format | Loader 类 | 备注 |
|---|---|---|
| `auto` | `DefaultModelLoader` | 在 `_prepare_weights` 内探测到 `consolidated*.safetensors` 时改判 `mistral`，否则按 `hf` |
| `hf` | `DefaultModelLoader` | `*.safetensors` + `*.bin` + `*.pt` 回退 |
| `safetensors` | `DefaultModelLoader` | 仅 `*.safetensors`，不回退 pt |
| `fastsafetensors` | `DefaultModelLoader` | 走 `fastsafetensors_weights_iterator` |
| `instanttensor` | `DefaultModelLoader` | InstantTensor 分布式加载（#36139） |
| `mistral` | `DefaultModelLoader` | `consolidated*.safetensors` + 自定义 index 文件 |
| `pt` | `DefaultModelLoader` | 仅 `*.pt` |
| `npcache` | `DefaultModelLoader` | `.bin` 转 numpy 缓存 |
| `dummy` | `DummyModelLoader` | 随机权重 |
| `sharded_state` | `ShardedStateLoader` | 预分片 TP checkpoint |
| `runai_streamer_sharded` | `ShardedStateLoader` | ShardedState + Run:ai 迭代器 |
| `tensorizer` | `TensorizerLoader` | 序列化加载 |
| `bitsandbytes` | `BitsAndBytesModelLoader` | 在线量化 |
| `runai_streamer` | `RunaiModelStreamerLoader` | 对象存储流式 |
| `modelexpress` | `ModelExpressModelLoader` | 桥接外部包 |

### 选择流程

```mermaid
flowchart TD
    A["LoadConfig.load_format"] --> B{"in _LOAD_FORMAT_TO_MODEL_LOADER?"}
    B -- no --> X["raise ValueError"]
    B -- yes --> C"_LOAD_FORMAT_TO_MODEL_LOADER[fmt"]
    C --> D["loader.load_model(vllm_config, model_config, prefix)"]
```

`get_model_loader` 见 `__init__.py:122`，`get_model` 见 `__init__.py:130`。

### 与 registry 对接

`get_model` 不直接解析模型类；`BaseModelLoader.load_model` 在 `base_loader.py:55` 调 `initialize_model(...)`（`loader/utils.py:42`），后者：

1. `get_model_architecture(model_config)`（`loader/utils.py:228`）—— 带 hash 缓存，命中则直接返回。
2. `_get_model_architecture`（`loader/utils.py:193`）调 `model_config.registry.resolve_model_cls(architectures, model_config)`。
3. `resolve_model_cls`（`models/registry.py:1250`）按 `model_impl`（`vllm`/`auto`/`transformers`/`terratorch`）分支，先试 vLLM 注册表，未命中且 `auto` 时回退到 Transformers 实现。
4. 拿到 `model_cls` 后按 `convert_type`（`none`/`embed`/`classify`）包一层 adapter（`as_embedding_model`/`as_seq_cls_model`）。
5. `initialize_model` 检查 `model_class.__init__` 签名是否含 `vllm_config` 和 `prefix`，是则为 new-style 构造；否则打 `DeprecationWarning` 并尽力兼容旧式签名。

详细注册表机制见 [`../../04-model-zoo/registry.md`](../../04-model-zoo/registry.md)（待补充）。

### 插件注册

`register_model_loader(load_format)` 装饰器（`__init__.py:69`）：

- 若 key 已存在，`logger.warning` 后覆盖。
- 校验被注册类必须是 `BaseModelLoader` 子类。
- 写入 `_LOAD_FORMAT_TO_MODEL_LOADER`。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `vllm/config/load.py::LoadConfig` | `load_format` 字段即分派 key；`model_loader_extra_config` 透传给具体 loader |
| `models/registry.py` | `resolve_model_cls` 提供 `(model_cls, arch)`，loader 不感知具体架构 |
| `vllm/transformers_utils/repo_utils.py` | `list_filtered_repo_files` 用于 `auto` 探测 mistral 格式 |
| `vllm/tracing` | `get_model` 路径上的 `@instrument` span 嵌入 |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 早期 | `LoadFormats` 仅 `auto`/`hf`/`pt`/`safetensors`/`npcache`/`dummy`/`tensorizer`/`bitsandbytes`/`sharded_state`/`runai_streamer` |
| 中期 | 加入 `mistral`、`fastsafetensors`、`runai_streamer_sharded` |
| main | 加入 `instanttensor`（#36139）、`modelexpress`（#43105） |
| v0.7+（待核实） | `register_model_loader` 插件机制引入，允许覆盖已注册 format |

---

## 参见

- [`default.md`](default.md) —— Default/Base 的装配流程
- [`../../04-model-zoo/registry.md`](../../04-model-zoo/registry.md)（待补充） —— 模型注册表详解
- [`../README.md`](../README.md) —— 返回模型执行首页
