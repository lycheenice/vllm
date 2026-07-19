# Mistral 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Mistral**

> 代表文件：`mistral.py`、`mixtral.py`、`mistral3.py`、`pixtral.py`、`mistral_eagle.py`、`mistral_large_3.py`、`mistral_large_3_eagle.py`。
> 厂商：Mistral AI。

---

## 是什么

- **`mistral.py`**：`MistralForCausalLM(LlamaForCausalLM)`——`MistralAttention(LlamaAttention)`、`MistralDecoderLayer(LlamaDecoderLayer)`、`MistralModel(LlamaModel)`。几乎所有组件复用 Llama，仅微调 RMSNorm/命名。`Ministral3ForCausalLM` 也指向这里的 `MistralForCausalLM`。
- **`mixtral.py`**：MoE 版（`MixtralMoE` 含 8 个 expert + top-2 路由）；`MixtralForCausalLM(nn.Module, SupportsLoRA, SupportsPP, MixtureOfExperts)`。Mistral 系最早的 MoE 实现，是 vLLM `FusedMoE` 层库的早期样本。
- **`mistral3.py`**：`Mistral3ForConditionalGeneration`，多模态版（含 `PixtralHFVisionModel` 视觉塔）。
- **`pixtral.py`**：Pixtral 视觉模型。内含两套实现：自有 `VisionTransformer`（`pixtral.py:817`）+ HF 兼容 `PixtralHFVisionModel`（`pixtral.py:1348`）。`PixtralHFEncoderInfo` 是 `vision.py:VisionEncoderInfo` 的子类之一。语言主干为 `MistralForCausalLM`，且 `PixtralForConditionalGeneration` 支持 `SupportsEagle3`。
- **`mistral_large_3.py`**：Mistral Large 3，dense 但超大；`MistralLarge3ForCausalLM`。
- **spec draft**：`mistral_eagle.py EagleMistralForCausalLM`（EAGLE-1/2）、`mistral_large_3_eagle.py EagleMistralLarge3ForCausalLM`（Mistral Large 3 专用 draft）。

---

## 为什么

- **继承 Llama 减重复**：Mistral 的"普通版"几乎等价 Llama，通过子类化复用 `LlamaForCausalLM` 的全部基础设施（LoRA/PP/EAGLE/quant）。
- **MoE 与 VLM 两条延伸**：Mixtral 引入 MoE，Pixtral 引入 VLM，分别验证 `FusedMoE` 层库与 `vision.py` 工具栈。
- **EAGLE 覆盖**：普通 Mistral 与 Large 3 都有独立 EAGLE draft，证明 Mistral AI 模型在 vLLM 投机解码上是 first-class。

---

## 怎么做

`MixtralMoE` 用 `FusedMoE` + `unquantized_fused_moe_method`（早期形式，路由 softmax + top-k）。现代 MoE（DeepSeek/Qwen）用 aux-loss-free 路由，Mixtral 保留经典形式以兼容旧 checkpoint。

Pixtral 视觉塔内部有两个并发实现：
- 自有 `VisionTransformer`（Mistral 自家 ViT，定义在 `pixtral.py` 内）
- HF 兼容 `PixtralHFVisionModel`（适配 HF 标准命名）

后者导出 `PixtralHFEncoderInfo` 给 `vision.py:get_vision_encoder_info` 识别，是 Pixtral 与共享层的接合点。

---

## 与其它模块/系统配合

- **[模型执行-层库](../../03-model-execution/layers/README.md)**：`FusedMoE` 层库在 Mixtral 上首发验证。
- **[vision.md](../vision.md)**：Pixtral 是 `VisionEncoderInfo` 的三大原生支持之一（CLIP/Siglip/Pixtral）。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：两套 EAGLE draft。
- **[分布式](../../07-distributed/README.md)**：`MixtralForCausalLM` 实现 `MixtureOfExperts` 供 EPLB。
- **[embedding-col](./embedding-col.md)**：`pixtral.py` 不直接做检索，但 ColPali 类基于 PaLI-Gemma 而非 Pixtral。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | `mistral.py`（继承 Llama）+ `mixtral.py` 作为首批 MoE。 |
| v0.5 | sliding window 精细化；Mixtral `MixtureOfExperts` 接口落地。 |
| v0.6 | Pixtral 加入，vision.py 三大 encoder 之一。 |
| v0.7–v0.8 | Mistral3 多模态；EAGLE draft。 |
| v0.10 | Mistral Large 3 + 独立 EAGLE draft。 |
| main | 与 Llama 家族持续同步（继承链保持）。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [llama 家族](./llama.md) · [vision](../vision.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md)
