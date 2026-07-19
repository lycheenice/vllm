# Gemma 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Gemma**

> 代表文件：`gemma.py`、`gemma2.py`、`gemma3.py`、`gemma3_mm.py`、`gemma3n.py`、`gemma3n_mm.py`、`gemma3n_audio_utils.py`、`gemma4.py`、`gemma4_mm.py`、`gemma4_mtp.py`、`gemma4_unified.py`、`paligemma.py`、`diffusion_gemma.py`。
> 厂商：Google。

---

## 是什么

- **`gemma.py`**：Gemma 1，独立实现（不继承 Llama），用 GeGLU + `RMSNorm` + `RotaryEmbedding`；`GemmaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsQuant)`。
- **`gemma2.py`**：Gemma 2 引入**局部 attention + sliding window** 交替（`Gemma2Attention` 用 `sliding_window`）；等价 dense 模型也可做 embedding（`Gemma2Model`/`Gemma2ForCausalLM` 注册到 `_EMBEDDING_MODELS`）。
- **`gemma3.py` / `gemma3_mm.py`**：Gemma 3 含文本版（`Gemma3ForCausalLM`，注册含 `Gemma3TextModel` 作 embedding 入口）+ 多模态版（`Gemma3ForConditionalGeneration`）。文本 backbone 含 sliding window 交替 + QK-norm。`Gemma3TextModelConfig`（`config.py:50`）是配套的 `VerifyAndUpdateConfig`。
- **`gemma3n.py` / `gemma3n_mm.py` / `gemma3n_audio_utils.py`**：Gemma 3n（多模态 + 音频），Perceiver Resampler 风格的多模态对齐。
- **`gemma4.py` / `gemma4_mm.py` / `gemma4_mtp.py` / `gemma4_unified.py`**：Gemma 4，含文本、多模态、MTP（spec draft）、`Gemma4UnifiedForConditionalGeneration`（统一接口）。`Gemma4ForCausalLM` 继承 `EagleModelMixin`。
- **`paligemma.py`**：PaLI-Gemma 视觉模型，含 Siglip 视觉塔 + `PaliGemmaForConditionalGeneration`。`ColPaliForRetrieval`（`colpali.py`）基于此。
- **`diffusion_gemma.py`**：DiffusionGemma——block diffusion 类生成（`DiffusionGemmaForBlockDiffusion`），走多模态注册项，文本生成路径特化。

---

## 为什么

- **数值稳定性特殊**：Gemma2/3/3n/glm4 等都被列入 `_FLOAT16_NOT_SUPPORTED_MODELS`（`config/model.py`），强制 bfloat16/float32。原因：logits 值域大 + RMSNorm 累积误差，FP16 易 inf。
- **sliding window + local attention 先行**：Gemma2 是早期引入"5 局部 + 1 全局"交替模式的架构，vLLM 的 KV cache block manager 需要识别这种 pattern（详见 [`05-attention`](../../05-attention/README.md)）。
- **统一接口实验场**：`gemma4_unified.py` 把 generate / multimodal / diffusion 合并到 `Gemma4UnifiedForConditionalGeneration`，是模型库少见的"单类多任务"路径。

---

## 怎么做

`Gemma3ForCausalLM` 的 sliding window 通过 cache_config 与 attention backend 协商：prefill 时把 sliding window 标志传入 attention metadata，decode 时复用 KV。多模态版 `Gemma3ForConditionalGeneration` 把 vision tower（`SiglipVisionModel`）的输出经 connector 投影，再 scatter 进 input_ids embedding。

Gemma4 MTP 走 vLLM 标准 MTP draft 模板，`Gemma4MTP`（`registry.py:620` 注册为 `Gemma4MTPModel`）。

---

## 与其它模块/系统配合

- **[配置](../../10-config/README.md)**：`Gemma3TextModelConfig.verify_and_update_model_config` 在 ModelConfig 构造期校准 dtype 限制。
- **[注意力](../../05-attention/README.md)**：sliding window / local attention 走特殊 KV cache 路径。
- **[多模态](../../11-multimodal/README.md)** / **[speech-audio](./speech-audio.md)**：Gemma3n 音频路径。
- **[embedding-col](./embedding-col.md)**：Gemma2/3 可作 embedding backbone；`colpali.py` 基于 PaLI-Gemma。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：Gemma4 MTP draft。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.4 | Gemma 1 接入。 |
| v0.5 | Gemma 2（含 sliding window）+ FP16 限制。 |
| v0.7 | PaLI-Gemma + ColPali 检索。 |
| v0.8–v0.9 | Gemma 3 文本/多模态版本。 |
| v0.10 | Gemma3n（含音频）。 |
| v0.11–v0.12 | Gemma4 系列 + MTP + Unified + Diffusion。 |
| main | Diffusion Gemma block diffusion、Gemma4 unified 收口。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [embedding-col](./embedding-col.md) · [speech-audio](./speech-audio.md) · [注意力](../../05-attention/README.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md)
