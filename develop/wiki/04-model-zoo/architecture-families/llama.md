# Llama 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Llama**

> 代表文件：`llama.py`、`llama4.py`、`llama_eagle.py`、`llama_eagle3.py`、`llama4_eagle.py`、`mllama4.py`、`fairseq2_llama.py`。
> vLLM 的 `LlamaForCausalLM` 是整个模型库的"基元"——很多家族（InternLM3、TeleChat3、IQuest、Mistral、Phi3、Qwen…）直接继承或复用其 `LlamaModel`/`LlamaDecoderLayer`。

---

## 是什么

Llama 家族泛指"以 Llama 架构为模板的 decoder-only transformer"。在 vLLM 内它的地位特殊：`llama.py:344` 的 `LlamaModel` 继承了 `EagleModelMixin`，意味着任何继承它的模型都自动具备采 EAGLE aux hidden state 的能力；`llama.py:446` 的 `LlamaForCausalLM` 同时实现 `SupportsLoRA`、`SupportsPP`、`SupportsQuant`、`SupportsEagle`、`SupportsEagle3`。许多非 Llama 模型干脆在注册表里别名指向 `LlamaForCausalLM`（如 `InternLM3ForCausalLM`、`IQuestCoderForCausalLM`、`CwmForCausalLM`、`TeleChat3ForCausalLM`，见 `registry.py:87, 134-135, 205`）。

家族变体：

- `llama.py`：基类实现。`LlamaMLP`（gate/up/down SwiGLU）、`LlamaAttention`、`LlamaDecoderLayer`、`LlamaModel`、`LlamaForCausalLM` + `LlamaBidirectionalForSequenceClassification` + `LlamaBidirectionalModel`（后者经 `as_*` 适配器派生）。
- `llama4.py`：Llama4 加入 **MoE**（`Llama4MoE`，含 shared expert）+ **vision tower**（ViT）；`Llama4ForCausalLM(LlamaForCausalLM, MixtureOfExperts)`，`Llama4Model` 继承 `LlamaModel`。
- `mllama4.py`：多模态版 `Llama4ForConditionalGeneration`，注册表把 `Llama4ForConditionalGeneration` 指向这里。
- `fairseq2_llama.py`：Meta fairseq2 训练的 Llama 权重适配（层名/编码差异），回归到标准 Llama 类。
- spec draft：`llama_eagle.py:131 EagleLlamaForCausalLM(LlamaForCausalLM)`（EAGLE-1/2）、`llama_eagle3.py:272 Eagle3LlamaForCausalLM(LlamaForCausalLM)`（EAGLE-3，被多个非 Llama target 共用，见注册表 `Eagle3MiniMaxM2`、`Eagle3Qwen3vl` 等）、`llama4_eagle.py EagleLlama4ForCausalLM`。

---

## 为什么

- **生态基元**：Llama 是开源 LLM 的事实标准，权重大多兼容 Llama 命名。把它的实现做成稳定基类，其他家族只需覆盖差异点（attention Norm、MoE 路由、RoPE 类型）即可，减少重复。
- **EAGLE 接口落地最早的家族**：`EagleModelMixin` 由 `LlamaModel` 直接继承，使 EAGLE 的 aux hidden state 采集逻辑天然可用——这是 EAGLE-3 draft 能跨家族复用 `Eagle3LlamaForCausalLM` 的前提。
- **Llama4 的 MoE + VLM 双扩展**：通过 `Llama4Model(LlamaModel)` 复用文本主干，再叠 `Llama4MoE` 与独立 vision tower，体现"基元 + 增量"扩展模式。`Llama4ForCausalLM` 同时挂 `MixtureOfExperts` 接口供 EPLB 使用。

---

## 怎么做

`LlamaDecoderLayer` 结构：`input -> LlamaAttention -> residual -> RMSNorm -> LlamaMLP -> residual`，与 HF `LlamaForCausalLM` 对齐。`LlamaModel` 用 `make_layers`（`utils.py:685`）按 PP 切层，最后一层后做 RMSNorm。

EAGLE-3 的 aux hidden state 由 `LlamaModel._maybe_add_hidden_state`（继承自 `EagleModelMixin`）在每层 forward 后判断 `layer_idx in aux_hidden_state_layers` 收集，默认层位 `(2, num_layers//2, num_layers-3)`。

Llama4 的 MoE 走 `FusedMoE` 层库，`Llama4Attention` 注入 attention bypass 优化；vision tower 输出经 connector 投影后 scatter 进 input_ids（走 `SupportsMultiModal.embed_input_ids` 路径，见 [`interfaces.md`](../interfaces.md)）。

---

## 与其它模块/系统配合

- **[模型执行-层库](../../03-model-execution/layers/README.md)**：消费 `RMSNorm`、`RotaryEmbedding`、`MergedColumnParallelLinear`、`RowParallelLinear`、`FusedMoE`。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：EAGLE-1/2/3 三套 draft；EAGLE-3 由 target 模型声明 `SupportsEagle3`、draft 用 `Eagle3LlamaForCausalLM`。
- **[LoRA](../../12-lora/README.md)**：`LlamaForCausalLM` 的 `packed_modules_mapping` 是 LoRA 散射 `qkv_proj`/`gate_up_proj`/`down_proj` 的依据。
- **[注意力](../../05-attention/README.md)**：`LlamaAttention` 内部 new `Attention(...)` 走 attention backend。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | Llama 首批接入；`LlamaForCausalLM` 作为基元确立。 |
| v0.5 | `LlamaBidirectionalModel` / `LlamaBidirectionalForSequenceClassification` 经 `as_*` 适配器派生，支持 embedding/classify。 |
| v0.6 | `LlamaModel` 继承 `EagleModelMixin`；EAGLE-1 draft `EagleLlamaForCausalLM` 加入。 |
| v0.8 | EAGLE-3 draft `Eagle3LlamaForCausalLM` 落地，逐渐成为跨家族通用 draft。 |
| v0.10 | Llama4 加入（MoE + VLM）；`Llama4ForCausalLM` 挂 `MixtureOfExperts`。 |
| v0.11 | `mllama4.py` 多模态分支；`llama4_eagle.py` 投机 draft。 |
| main | `Fairseq2LlamaForCausalLM` 加入以兼容 Meta fairseq2 权重。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [模型执行-层库](../../03-model-execution/layers/README.md) · [LoRA](../../12-lora/README.md)
