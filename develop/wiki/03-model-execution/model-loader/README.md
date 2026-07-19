# 模型加载器总览

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > **模型加载器**

> 源码目录：`vllm/model_executor/model_loader/`

---

## 是什么

模型加载器（model loader）是"把权重从外部存储搬进 `nn.Module` 参数"的统一抽象。本目录以 `BaseModelLoader` 为抽象基类，派生出若干具体 loader，每个对应一种 `LoadConfig.load_format`。所有 loader 都实现同一套契约：`download_model` / `load_weights` / `load_model`，由 `__init__.py` 的 `_LOAD_FORMAT_TO_MODEL_LOADER` 字典做一次性分派。

具体 loader 清单：

| Loader 类 | 源文件 | 对应 load_format | 一句话定位 |
|---|---|---|---|
| `DefaultModelLoader` | `default_loader.py` | `auto`/`hf`/`safetensors`/`pt`/`npcache`/`mistral`/`fastsafetensors`/`instanttensor` | 主力 loader，吃 HF safetensors/bin/pt |
| `DummyModelLoader` | `dummy_loader.py` | `dummy` | 随机权重，用于 profiling / 显存测算 |
| `ShardedStateLoader` | `sharded_state_loader.py` | `sharded_state`/`runai_streamer_sharded` | 预分片 TP checkpoint，每 rank 只读自己的 shard |
| `TensorizerLoader` | `tensorizer_loader.py` | `tensorizer` | CoreWeave tensorizer 序列化加载 |
| `BitsAndBytesModelLoader` | `bitsandbytes_loader.py` | `bitsandbytes` | 在线 8/4-bit (NF4) 量化加载 |
| `RunaiModelStreamerLoader` | `runai_streamer_loader.py` | `runai_streamer` | Run:ai 流式加载本地/S3/GCS/Azure Blob |
| `ModelExpressModelLoader` | `modelexpress_loader.py` | `modelexpress` | 桥接外部 ModelExpress 包 |
| — | `ep_weight_filter.py` | （非独立 loader，被 Default 复用） | EP 下跳过非本 rank 专家权重 |
| `reload/` | `reload/` 子包 | （被 base/default/dummy 复用） | 层次化热重载、在线量化延迟处理 |

辅助文件：

- `weight_utils.py` —— HF 下载、safetensors/pt 迭代器、`default_weight_loader`、dummy 初始化（详见 [`weight-utils.md`](weight-utils.md)）。
- `utils.py` —— `initialize_model`、`process_weights_after_loading`、`get_model_architecture`、`device_loading_context`、`ParamMapping`。
- `tensorizer.py` —— tensorizer 序列化/反序列化与 config 实现（被 `tensorizer_loader.py` 调用）。

---

## 为什么

不同部署场景对"权重从哪来、怎么进显存"的需求差异极大，但**模型架构代码不应为此分叉**。因此 vLLM 把"产出 `(name, tensor)` 迭代器"这一职责压缩到 loader 内，而把"张量如何塞进参数、如何 TP 切分/重排"留给各层的 `weight_loader` 回调（挂在 `BasevLLMParameter` 上）。这样：

- 换权重格式（safetensors → 预分片 → tensorizer）只换 loader，模型不动。
- 在线量化（bnb/torchao）只需 loader 在产出张量时做转换，模型层无感。
- RL 训练循环里"热重载权重"复用同一套 `weight_loader` 回调，靠 `reload/` 子包做 meta device 延迟物化。
- 第三方加载生态（Run:ai、ModelExpress、InstantTensor）以插件形式接入，`register_model_loader` 装饰器允许 out-of-tree 注册。

---

## 怎么做

### 基类契约

`BaseModelLoader`（`base_loader.py:25`）定义三个方法，其中 `load_model` 是具体方法（模板方法模式），`download_model`/`load_weights` 是抽象方法：

```python
class BaseModelLoader(ABC):
    def __init__(self, load_config: LoadConfig): ...
    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None: ...
    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    @instrument(span_name="Load model")
    def load_model(self, vllm_config, model_config, prefix="") -> nn.Module: ...
```

`load_model` 的固定步骤（`base_loader.py:43`）：

1. 解析 `load_device`（`load_config.device` 优先，否则 `device_config.device`）。
2. `with set_default_torch_dtype(...)` + `with target_device:` 下 `initialize_model(...)` 构造模型（参数直接落在目标设备上）。
3. `log_model_inspection`（受 `VLLM_LOG_MODEL_INSPECTION` 控制）。
4. `self.load_weights(model, model_config)` —— 子类实现，喂权重。
5. 记录 peak GPU memory（CUDA/XPU）。
6. 若 `_has_online_quant(model)`（任一 `quant_method.uses_meta_device=True`），调 `finalize_layerwise_processing`。
7. `process_weights_after_loading(model, model_config, target_device)` —— 走所有 quant method / Attention / HPC 模块的 `process_weights_after_loading`。
8. `return model.eval()`。

### Loader 选择

`get_model_loader(load_config)`（`__init__.py:122`）以 `load_format` 查 `_LOAD_FORMAT_TO_MODEL_LOADER` 字典实例化。完整分派表与 registry 对接见 [`dispatch.md`](dispatch.md)。

### 默认权重装配流水

`DefaultModelLoader` 的 `_prepare_weights` → `_get_weights_iterator` → `get_all_weights`（含 `secondary_weights`）→ `model.load_weights`，详见 [`default.md`](default.md)。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `vllm/config/load.py::LoadConfig` | 驱动 loader 选择与行为（`load_format`、`safetensors_load_strategy`、`model_loader_extra_config` 等） |
| `models/registry.py` | `initialize_model` 经 `model_config.registry.resolve_model_cls` 拿模型类 |
| `layers/quantization/` | quant method 的 `create_weights`/`process_weights_after_loading` 决定权重张量形态与后处理 |
| `layers/`（各 `weight_loader` 回调） | 接收 loader 产出的 `(name, tensor)`，做 TP 切分/融合/重排 |
| `v1/worker/gpu_worker.py` | 调 `get_model(...)` 拿到模型；sleep/wake_up 复用 `reload/` |
| `vllm/distributed` | `ShardedStateLoader`/EP filter 读 TP/DP/PCP rank |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 早期 | `loader.py` 单文件含 `get_model`/`get_model_loader` 与 default loader |
| 中期 | 拆分为 `model_loader/` 包，按 loader 分文件；`LoadConfig` 从 `ModelConfig` 独立 |
| v0.7+ | `register_model_loader` 插件注册机制引入（`__init__.py:69`） |
| main | `reload/` 从零散函数整理为子包（`layerwise.py`/`meta.py`/`types.py`/`utils.py`/`sanitize.py`/`torchao_decorator.py`）；`ep_weight_filter.py` 独立（#37136）；`modelexpress_loader.py`（#43105） |

---

## 参见

- [`dispatch.md`](dispatch.md) —— LoadFormat → Loader 分派表与 registry 对接
- [`default.md`](default.md) —— 默认权重装配
- [`../README.md`](../README.md) —— 返回模型执行首页
