# 模型能力接口契约（interfaces）

[← Wiki 首页](../README.md) > [模型库](../README.md) > **能力接口**

> 源码：`vllm/model_executor/models/interfaces.py`（1703 行）+ `vllm/model_executor/models/interfaces_base.py`（283 行）。

---

## 是什么

`interfaces.py` / `interfaces_base.py` 用 `typing.Protocol`（`@runtime_checkable`）+ `ClassVar` 标签描述"vLLM 期望模型具备的能力"。一个模型类通过 *继承* 这些 Protocol、或仅通过 *鸭子类型* 设置对应 `ClassVar`，就能向注册表和执行框架声明自己的能力。注册表的 `_ModelInfo.from_model_cls`（`registry.py:774`）会读取这些标签生成能力快照，供调度器、注意力后端、KV 缓存管理器早期决策。

两个文件的分工：

| 文件 | 职责 |
|---|---|
| `interfaces_base.py` | 最底座接口：`VllmModel` / `VllmModelForTextGeneration` / `VllmModelForPooling`，以及 `is_vllm_model` / `is_text_generation_model` / `is_pooling_model` 判定函数。给 OOT 模型用——不强制继承，靠 duck-type。 |
| `interfaces.py` | 业务能力接口：多模态、LoRA、PP、Mamba、EAGLE、转写、MRoPE、Encoder CUDA Graph、MoE 等。内置模型普遍直接继承这些 Protocol。 |

### 核心接口清单

| 接口 | 关键 `ClassVar` / 方法 | 用途 |
|---|---|---|
| `VllmModel` | `__init__(vllm_config, prefix)` / `embed_input_ids` / `forward(input_ids, positions)` | 所有模型的最低契约（`interfaces_base.py:47`） |
| `VllmModelForTextGeneration` | `compute_logits(hidden_states) -> T|None` | 生成模型（`interfaces_base.py:114`），TP rank>0 返回 None |
| `VllmModelForPooling` | `is_pooling_model=True`、`default_seq_pooling_type`、`attn_type`、`score_type`、`pooler` | embedding / classify / reward 共用底座（`interfaces_base.py:148`） |
| `SupportsMultiModal` | `supports_multimodal=True`、`_mark_language_model` / `_mark_tower_model` / `get_language_model` / `embed_input_ids` 重载 | 多模态入口（`interfaces.py:94`） |
| `SupportsMultiModalPruning` | `supports_multimodal_pruning=True`、`recompute_mrope_positions` | 动态裁剪多模态 token 后重算 M-RoPE 位置 |
| `SupportsLoRA` | `supports_lora=True`、`embedding_modules`、`packed_modules_mapping`、`lora_skip_prefixes` | LoRA 适配器挂载点（`interfaces.py:538`） |
| `SupportsPP` | `supports_pp=True`、`make_empty_intermediate_tensors`、`forward(..., intermediate_tensors=...)` | 流水并行（`interfaces.py:617`） |
| `HasInnerState` | `has_inner_state=True` | Mamba / Jamba：需要调度器分配 SSM 状态（`interfaces.py:737`） |
| `IsAttentionFree` | `is_attention_free=True` | 纯 SSM：选 attention-free 后端，block manager 无需 attention（`interfaces.py:763`） |
| `IsHybrid` | `is_hybrid=True`、`get_mamba_state_shape_from_config`、`get_mamba_state_copy_func` | Mamba+Attention 混合（Jamba/Nemotron-H）（`interfaces.py:790`） |
| `MixtureOfExperts` | `expert_weights` / `num_moe_layers` / `set_eplb_state` / `update_physical_experts_metadata` | MoE 模型，供 EPLB/专家迁移（`interfaces.py:847`） |
| `HasNoOps` | `has_noops=True` | 含 NoOp 层（MTP/nextn），权重加载需跳过 |
| `SupportsMambaPrefixCaching` | `supports_mamba_prefix_caching=True` | Mamba 状态可做前缀缓存（实验性） |
| `SupportsCrossEncoding` | `score_type="cross-encoder"` | 重排器（cross-encoder rerank） |
| `SupportsLateInteraction` | `score_type="late-interaction"` | ColBERT 风格逐 token 检索 |
| `SupportsQuant` | `hf_to_vllm_mapper`、`packed_modules_mapping`、`quant_config` | 量化基类（`interfaces.py:999`），`__new__` 自动注入 quant_config |
| `SupportsScoreTemplate` | `supports_score_template=True`、`get_score_template` | 模型自带 score 提示模板 |
| `SupportsTranscription` | `supports_transcription=True`、`supported_languages`、`get_generation_prompt`、`get_speech_to_text_config` | ASR 转写模型（`interfaces.py:1078`） |
| `SupportsRealtime` | `supports_realtime=True`、`realtime_max_tokens`、`buffer_realtime_audio` | 实时流式 ASR |
| `SupportsEagleBase` / `SupportsEagle` / `SupportsEagle3` | `supports_eagle` / `supports_eagle3`、`set_aux_hidden_state_layers`、`get_eagle3_default_aux_hidden_state_layers` | EAGLE-1/2/3 投机解码（`interfaces.py:1259`+） |
| `EagleModelMixin` | `aux_hidden_state_layers`、`_maybe_add_hidden_state` | 给 backbone 注入"采集 aux hidden states"逻辑（`interfaces.py:1323`） |
| `LocalArgmaxMixin` | `get_top_tokens` | draft 模型 D2T 感知的 vocab-parallel argmax（`interfaces.py:1288`） |
| `SupportsMRoPE` | `supports_mrope=True`、`get_mrope_input_positions` | M-RoPE（Qwen2-VL 类） |
| `SupportsXDRoPE` | `supports_xdrope=True`、`get_xdrope_input_positions` | XD-RoPE（4D/3D 位置） |
| `SupportsEncoderCudaGraph` | 多个 `get_encoder_cudagraph_*` 方法 | 视觉编码器 CUDA graph 捕获/重放（`interfaces.py:1547`） |

---

## 为什么

- **接口而非继承**：用 `Protocol`（结构性子类型）而不是基类继承，让 OOT 插件模型不必 import vLLM 内部基类就能声明能力。`runtime_checkable` 让 `isinstance` 检查生效，注册表据此探测。
- **ClassVar 标签优先于方法签名**：`supports_multimodal = True` 这种布尔标签远比"是否有某个方法"更便宜、更稳定。`_ModelInfo.from_model_cls` 大量用 `getattr(model, "is_attention_free", False)` 取默认值，不给标签就视为 False，向后兼容。
- **避免基类多重继承爆炸**：`SupportsQuant.__new__`（`interfaces.py:1006`）在实例化时就地 hook `quant_config` 注入与 `hf_to_vllm_mapper` 应用；`EagleModelMixin` 给 backbone 注入 aux hidden state 采集——这些都是 Mixin，不强制整套继承树。
- **任务多态**：同一份 `LlamaForCausalLM` 类，可以按 `runner_type=pooling` 走 `as_embedding_model` 在线子类化为 `LlamaForEmbedding`（见 `adapters.md`），其能力标签随之变更。标签是"类属性"层面的，能被动态子类覆盖。

---

## 怎么做

### 模型类声明能力（典型骨架）

```python
class Qwen3ForCausalLM(
    LocalArgmaxMixin, nn.Module,
    SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    # is_attention_free / is_hybrid / has_inner_state 等默认 False，
    # 不必显式写；Mamba/Jamba 等模型才显式继承对应 Protocol。
```

注意 `Qwen3ForCausalLM` 没有显式继承 `SupportsMultiModal`——`supports_multimodal` 默认 False，由 `getattr` 兜底（`interfaces.py:462`）。

### 多模态模型的子模块标记

`SupportsMultiModal` 提供两个上下文管理器（`interfaces.py:214` + `:249`）：

- `_mark_language_model(vllm_config, targets=...)`：在 `__init__` 里用 `with` 包裹"创建 LM backbone"的代码段，登记到的子模块名进 `_language_model_names`，使 `get_language_model()` 可定位；在 `--mm-encoder-only` 模式下还会用 `StageMissingLayer` 跳过初始化（省显存）。
- `_mark_tower_model(vllm_config, modalities, targets=...)`：同理标记 vision/audio tower；当某模态的 `limit_mm_per_prompt=0` 时跳过塔初始化。
- `_mark_composite_model` 一次性组合 LM + 多 tower（`interfaces.py:294`）。

`get_language_model()`（`interfaces.py:176`）带模块级缓存 `_language_model_by_module`，避免反复遍历子模块。

### EAGLE-3 的 aux hidden state 注入

`EagleModelMixin._maybe_add_hidden_state`（`interfaces.py:1329`）由 backbone 的 decoder layer 在每层 forward 后调用：若当前层索引在 `aux_hidden_state_layers` 里（默认 `2, num_layers//2, num_layers-3`，见 `get_eagle3_default_aux_hidden_state_layers`），就把 `hidden + residual` append 进 aux 列表，供 EAGLE-3 draft 模型作输入。`SupportsEagle3.set_aux_hidden_state_layers` 通过 `get_language_model`/`language_model` 定位到真正的 backbone 后转发。

### SupportsQuant 的自动注入

```python
class SupportsQuant:
    hf_to_vllm_mapper: ClassVar[WeightsMapper | None] = None
    packed_modules_mapping: ClassVar[dict | None] = None
    quant_config: QuantizationConfig | None = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.quant_config = cls._find_quant_config(*args, **kwargs)  # 从 VllmConfig 抠出
        cls._maybe_apply_model_mapping(instance)                        # 把 mapper 应用到 quant_config
        return instance
```

`_maybe_apply_model_mapping`（`interfaces.py:1031`）会调 `quant_config.apply_vllm_mapper` 与 `packed_modules_mapping`，让量化层名与模型实际层名对齐。这是"模型类与量化方法解耦"的关键——同一份模型代码可挂 FP8/GPTQ/AWQ/Marlin。

---

## 与其它模块/系统配合

- **[registry.md](registry.md)**：`_ModelInfo.from_model_cls` 按接口函数（`supports_multimodal(model)` 等）读取标签；JSON 缓存保证了启动性能。
- **[adapters.md](adapters.md)**：`as_embedding_model` / `as_seq_cls_model` 动态子类化时，新类继承 `VllmModelForPooling` 并设 `is_pooling_model=True`，把"生成模型"在线转成"池化模型"。
- **[模块执行-加载器](../03-model-execution/model-loader/README.md)**：`initialize_model` 实例化后，`ModelConfig.is_pooling_model` / `is_multimodal_model` 等已被 `inspect_model_cls` 探测好，决定走 generate runner 还是 pooling runner。
- **[多模态](../11-multimodal/README.md)**：`SupportsMultiModal` 与 `MultiModalRegistry` 互调；`get_num_mm_encoder_tokens` 给 LoRA tower 提供预算估算；`SupportsEncoderCudaGraph` 让 `EncoderCudaGraphManager`（worker 侧）能捕获视觉编码器。
- **[注意力](../05-attention/README.md)**：`IsAttentionFree` 让 attention backend 选择器跳过 attention；`IsHybrid.get_mamba_state_shape_from_config` 决定 SSM 缓存 shape。
- **[采样-投机](../06-sampling-decoding/speculative-decoding/README.md)**：`SupportsEagle3` + `EagleModelMixin` 是 EAGLE-3 的契约；`LocalArgmaxMixin` 处理 draft 词表小于 target 词表的 argmax 映射。
- **[LoRA](../12-lora/README.md)**：`SupportsLoRA.embedding_modules` / `packed_modules_mapping` / `lora_skip_prefixes` 直接被 LoRA manager 消费；`module-mapping.md` 的 `MultiModelKeys` 进一步切分多模态塔。
- **[分布式](../07-distributed/README.md)**：`SupportsPP` 控制 PP 是否可用；`MixtureOfExperts` 给 EPLB 与专家迁移提供 expert 权重视图。

---

## 历史版本演进

| 版本 | 变更 | 动机 |
|---|---|---|
| 早期 | 用 `isinstancelike` 工具函数 + 类属性散乱判断。模型文件各自定义 `supports_*`。 | 简陋，缺统一约束。 |
| v0.5 | 抽出 `interfaces.py`，定义 `SupportsMultiModal` / `SupportsPP` / `HasInnerState` Protocol。 | Mamba/Jamba 入场需要语义化标签。 |
| v0.6–v0.7 | `IsAttentionFree` / `IsHybrid` / `SupportsMambaPrefixCaching` 入场；`Interfaces_base.py` 拆分，给 OOT 模型留 duck-type 通路（避免因继承破坏兼容）。 | OOT 模型不能强依赖 vLLM 内部基类。 |
| v0.8 | `SupportsQuant.__new__` 注入 quant_config；`SupportsLoRA.packed_modules_mapping` 收口。 | 量化与模型解耦，LoRA 路径统一。 |
| v0.9–v0.10 | `SupportsEagle` / `SupportsEagle3` + `EagleModelMixin` + `LocalArgmaxMixin`；`SupportsTranscription` / `SupportsRealtime`。 | EAGLE-3 投机解码与 ASR 转写两块新能力上线。 |
| v0.10.5 | `SupportsEncoderCudaGraph` 大型接口（10+ 方法）落地。 | 视觉编码器 CUDA graph 捕获成为性能关键。 |
| v0.11 | `SupportsQuant._maybe_apply_model_mapping` 把 `hf_to_vllm_mapper` 应用到 `quant_config`；`SupportsMultiModal` 内 `embed_input_ids` 引入 `is_multimodal` 强制参数（PR #16229）。 | OOV 多模态 token 处理与量化层名对齐。 |
| main | `HasNoOps`、`SupportsXDRoPE`、`SupportsScoreTemplate`、`SupportsMultiModalPruning` 等追加。`get_language_model` 加模块级缓存。 | NoOp skip、XD-RoPE、动态裁剪等局部能力持续扩张。 |

---

## 参见

- [← 返回模型库首页](../README.md)
- [`registry.md`](registry.md) — `_ModelInfo` 由这些标签组装
- [`adapters.md`](adapters.md) — 动态子类化如何变更能力标签
- [`module-mapping.md`](module-mapping.md) — `SupportsLoRA.packed_modules_mapping` 的多模态扩展
- [`vision.md`](vision.md) — 视觉塔相关的 `SupportsMRoPE`/`SupportsEncoderCudaGraph` 实践
