# InternLM 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **InternLM**

> 代表文件：`internlm2.py`、`internlm.py`（旧版已被 `InternLMForCausalLM` 清退）、`intern_vit.py`、`internvl.py`、`interns1.py`、`interns1_pro.py`、`interns1_vit.py`、`interns2_preview.py`。
> 厂商：上海 AI Lab。

---

## 是什么

- **`internlm2.py`**：dense 文本主干。`InternLM2ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, SupportsQuant)`，含独立 `InternLM2Attention`/`InternLM2MLP`/`InternLMDecoderLayer`/`InternLM2Model`。值得关注的是 `internlm2.py:392 InternLM2ForRewardModel(InternLM2ForCausalLM)`——InternLM2 是少数原生提供 reward model 实现的家族。
- **`intern_vit.py`**：InternViT 视觉塔实现（`InternVisionModel`），被多个 VLM（InternVL/InternS1/InternS2）共享。这是 `vision.py:VisionEncoderInfo` 之外的"自家视觉塔"模式——InternViT 不在 CLIP/Siglip/Pixtral 三家之列，所以编码器元信息走模型内部自定义。
- **`internvl.py`**：`InternVLChatModel`，经典 InternVL 多模态对话模型，使用 InternViT + LM backbone（可配 Llama/Qwen 等）。
- **`interns1.py` / `interns1_pro.py` / `interns1_vit.py`**：InternS1 与 InternS1-Pro，新一代多模态；注册表把 `InternVLForConditionalGeneration`（别名）也指向 interns1 的 `InternS1ForConditionalGeneration`（`registry.py:424`）。
- **`interns2_preview.py`**：InternS2 预览版。

`InternLM3ForCausalLM` 与 `IQuestCoderForCausalLM` 在注册表里直接别名指向 `LlamaForCausalLM`（`registry.py:134-135`）——InternLM3 已与 Llama 架构无差异，故不另写文件。

---

## 为什么

- **reward model 原生实现**：InternLM2 是少数在文件内原生实现 `InternLM2ForRewardModel` 的家族，注册到独立 `_REWARD_MODELS` 字典（与 Qwen2 reward 文件并列），让 reward API 不必走 `as_*` 适配器。
- **视觉塔"自营"**：InternViT 是 Intern 系自家 ViT，不走 CLIP/Siglip/Pixtral 通路，故 `intern_vit.py` 实现完整 forward + weight loading，与 `vision.py` 的 encoder info 体系并行。
- **清退与别名**：旧 InternLM（`InternLMForCausalLM`）已清退（`_PREVIOUSLY_SUPPORTED_MODELS` 中 v0.23.0 退役），现代 InternLM3 用 Llama 别名，体现"无差异就别造文件"。

---

## 怎么做

`InternVLChatModel` 把 InternViT 输出经投影后送进 LM backbone，LM 的选择由 `config.llm_config` 决定——模型文件内动态 import 对应 backbone 类。`interns1.py` 基本沿用这一模式，但 InternS1 的视觉编码走 `interns1_vit.py` 的独立实现。

reward model 实质是 `InternLM2ForCausalLM` 加一个 `score` 头，但文件内直接写好（未走 `as_seq_cls_model`），便于控制 reward 训练特有的细节（如 PRM 的 step-level pooler）。

---

## 与其它模块/系统配合

- **[embedding-col](./embedding-col.md)**：InternLM2 的 reward 与 Qwen2 reward 共同构成 vLLM 原 reward 池。
- **[多模态](../../11-multimodal/README.md)**：Intern 系 VLM 通过 `SupportsMultiModal` 注册 processor。
- **[模型执行-层库](../../03-model-execution/layers/README.md)**：InternViT 与 LM 共用 `layers/` 里 norm/linear/attention 等基础块。
- **[配置](../../10-config/README.md)**：动态选 LM backbone 需要 `config.get_text_config()`/`llm_config` 协议。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | InternLM（旧版）接入。 |
| v0.6 | InternLM2 + reward model 原生实现；InternVL + InternViT。 |
| v0.9 | `InternLMForCausalLM` 清退（v0.23.0 别名）。 |
| v0.10 | InternS1 / InternS1-Pro / InternS2-preview 新一代上线。 |
| main | `InternLM3ForCausalLM` 用 Llama 别名减少冗余。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [llama 家族](./llama.md) · [embedding-col](./embedding-col.md) · [多模态](../../11-multimodal/README.md)
