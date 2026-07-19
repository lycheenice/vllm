# HF 模块→vLLM LoRA 层映射（module_mapping）

[← Wiki 首页](../README.md) > [模型库](../README.md) > **模块映射**

> 源码：`vllm/model_executor/models/module_mapping.py`（37 行，纯数据类）。

---

## 是什么

`module_mapping.py` 只定义了一个 dataclass `MultiModelKeys`（含一个 `from_string_field` 工厂），用来把一个多模态模型的全部参数名按"功能分区"切成四组：

| 字段 | 含义 | 典型前缀（Llava 例） |
|---|---|---|
| `language_model` | 文本主干（LM backbone） | `"language_model"` |
| `connector` | 多模态投影/连接器（把 vision/audio embedding 投到 LM 隐空间） | `"multi_modal_projector"` |
| `tower_model` | 视觉塔/音频塔等编码器 | `"vision_tower"` |
| `generator` | 生成器（部分模型有独立的 image/audio 生成头） | （多数为空） |

模型类实现 `get_mm_mapping(self) -> MultiModelKeys` 方法返回自身的分区；LoRA manager 实例化时通过 `self.model.get_mm_mapping()` 取得（`lora/model_manager.py:181`）。

---

## 为什么

LoRA 适配器是按"模块前缀"挂载的。纯文本模型只要一个 `packed_modules_mapping`（见 [`interfaces.md` SupportsLoRA）即可，但多模态模型的子模块天然分成"语言侧"和"塔侧"两类，两者的 LoRA 语义不同：

- **语言侧 LoRA**：和文本模型一样，挂在 `q_proj/k_proj/v_proj/o_proj/gate_up_proj/down_proj` 上，受 `max_loras`、`lora_dtype` 等常规调度约束。
- **塔侧 LoRA**（Tower Connector LoRA）：可独立挂到 vision tower 的 attention 或 connector 上，但需要 `--enable-tower-connector-lora` 显式开启；且依赖 `get_num_mm_encoder_tokens` 给出"图像 token 数 → encoder token 数"的预算，因为塔侧 LoRA 的 Punica wrapper 必须知道每个图像占多少 encoder token 才能切片。

`MultiModelKeys` 就是把这套分区信息一行结构化数据传给 LoRA 系统，避免 LoRA manager 反向猜模型结构。它源自 `ms-swift`（见文件顶部注释 `Adapted from modelscope/ms-swift`），后来被 vLLM 内化。

---

## 怎么做

### 在模型里实现 get_mm_mapping

```python
# vllm/model_executor/models/llava.py:730
def get_mm_mapping(self) -> MultiModelKeys:
    return MultiModelKeys.from_string_field(
        language_model="language_model",
        connector="multi_modal_projector",
        tower_model="vision_tower",
    )
```

字段可以是单个字符串（自动包成 list）或字符串列表（多个前缀共用一个角色，比如同时有 `vision_tower` 和 `audio_tower` 的 omni 模型）。

### LoRA manager 消费

`LoRAModelManager.__init__`（`lora/model_manager.py:181`）拿到 `mm_mapping` 后：

1. 断言 `len(mm_mapping.language_model) == 1`——vLLM 只支持一个 LM backbone。
2. 为该前缀创建一个 `PunicaWrapper`（语言侧 LoRA 计算）。
3. 检查模型是否实现了 `get_num_mm_encoder_tokens`，决定 `supports_tower_connector_lora`。
4. 若 `lora_config.enable_tower_connector_lora=True` 且模型支持，再为 connector / tower 各建独立 Punica wrapper。

### 与 packed_modules_mapping 的分工

- `SupportsLoRA.packed_modules_mapping`（`interfaces.py:554`）：描述**单个层**的 packed 拆分，如 `{"qkv_proj": ["q_proj", "k_proj", "v_proj"]}`，给 LoRA kernel 用——LoRA 权重按 `q_proj` 命名，需要 scatter 到合并后的 `qkv_proj` 上。
- `MultiModelKeys`：描述**模块前缀**的语义分区，给 LoRA manager 用——决定哪些前缀走语言侧 wrapper、哪些走塔侧 wrapper。

两者正交，多模态模型通常两套都设。

---

## 与其它模块/系统配合

- **[LoRA](../12-lora/README.md)**：直接消费者；Punica wrapper 按 LM / connector / tower 分配；`enable_tower_connector_lora` 开关决定是否启用塔侧 LoRA。
- **[`interfaces.md`](interfaces.md) `SupportsMultiModal`**：`get_mm_mapping` 与 `get_num_mm_encoder_tokens` / `get_num_mm_connector_tokens` 配套实现，共同构成 LoRA-on-VLM 的契约。
- **[多模态](../11-multimodal/README.md)**：`language_model_only` 模式（仅加载 LM，跳过塔）会让塔侧 LoRA 自动禁用，并在日志里 warning（见 `model_manager.py` 末段）。
- **[模型执行-层库](../03-model-execution/layers/README.md)**：`packed_modules_mapping` 与 `MultiModelKeys` 共同喂养 LoRA 层（`lora_layers.py`）的权重 scatter。

---

## 历史版本演进

| 版本 | 变更 | 动机 |
|---|---|---|
| 早期 | 多模态 LoRA 不支持，塔侧 LoRA 无统一抽象。 | Llava 等模型出现后产生需求。 |
| v0.5–v0.6 | 引入 `MultiModelKeys`（import 自 ms-swift），`get_mm_mapping` 在主要 VLM 里铺开。 | 统一"哪些层是 LM/哪些是塔"。 |
| v0.7–v0.8 | `enable_tower_connector_lora` 配置项上线；`get_num_mm_encoder_tokens` 加入接口，让塔侧 LoRA 能算 Punica 预算。 | 视觉 LoRA 微调需求上升。 |
| main | `from_string_field` 工厂简化常见用法；字段 `generator` 出现以适应带生成头的 omni 模型（Qwen2.5-Omni 等）。 | 多模态模型形态从"理解"扩到"生成"。 |

---

## 参见

- [← 返回模型库首页](../README.md)
- [`interfaces.md`](interfaces.md) — `SupportsLoRA` 与 `SupportsMultiModal` 契约
- [LoRA 子系统](../12-lora/README.md) — `LoRAModelManager` 的实际消费方
