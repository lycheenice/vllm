# GLM 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **GLM**

> 代表文件：`glm.py`、`glm4.py`、`glm4_moe.py`、`glm4_moe_lite.py`、`glm4v.py`、`glm4_1v.py`、`glm_ocr.py`、`glm_ocr_mtp.py`、`glm4_moe_mtp.py`、`glm4_moe_lite_mtp.py`、`glmasr.py`、`glmasr_utils.py`、`chatglm.py`。
> 厂商：智谱（Zhipu AI）。

---

## 是什么

- **`glm.py:11`**：`GlmForCausalLM(LlamaForCausalLM)` 即旧版 GLM（chatglm 系列），vLLM 通过直接继承 LlamaForCausalLM 复用全部基础设施。
- **`chatglm.py`**：更早的 ChatGLM 架构（含 `ChatGLMModel`/`ChatGLMForConditionalGeneration` 注册别名），与 Llama 差异较大（旋转位置、LayerNorm 顺序）。
- **`glm4.py`**：现代 GLM4 文本主干。`Glm4Model(LlamaModel)` + `Glm4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP)`，自带 MTP/nextn 层（加载时跳过，由独立 draft 文件加载）。注意 `glm4` 在 `_FLOAT16_NOT_SUPPORTED_MODELS` 列表（必须 bfloat16/float32）。
- **`glm4_moe.py`**：GLM4 MoE 版本，含 `Glm4MoE`、`Glm4MixtureOfExperts` 接口实现，可供 EPLB。
- **`glm4_moe_lite.py`**：GLM-4-MoE-Lite，简化专家拓扑；自带独立 MTP（`glm4_moe_lite_mtp.py`）。
- **VLM**：`glm4v.py`（`GLM4VForCausalLM`）、`glm4_1v.py`（`Glm4vForConditionalGeneration`/`Glm4vMoeForConditionalGeneration`，涵盖 MoE 视觉变体）。
- **OCR**：`glm_ocr.py` + `glm_ocr_mtp.py`（带 MTP 的 GLM-OCR）。
- **ASR**：`glmasr.py`/`glmasr_utils.py`：`GlmAsrForConditionalGeneration` 实现转写。
- **GLM-5.2 DSA**：注册表 `"GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM")`（`registry.py:116`）——智谱 GLM-5.2 的 DeepSeek Sparse Attention 复用 `deepseek_v2.py` 内的实现（`deepseek_v2.py:1914`），不单独成文件。

---

## 为什么

- **复用 Llama 基元**：旧 GLM/现代 GLM4 都尽量把 text backbone 写成 `LlamaForCausalLM`/`LlamaModel` 子类，差异只在 attention/Norm/RoPE，省维护成本。
- **MTP 多套并存**：GLM4-MoE、GLM4-MoE-Lite、GLM-OCR 各有独立 MTP 文件，说明 draft 实现因 backbone 差异不能共用——每种拓扑都得单独写 forward。
- **跨厂商架构复用**：GLM-5.2 的 DSA 干脆不写新文件，直接 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`。这是 vLLM 鼓励的"实现复用优先于命名复用"模式。

---

## 怎么做

`Glm4ForCausalLM.__init__` 在构建 decoder layers 时按 `config.num_nextn_predictable_layers` 决定要不要加 MTP/nextn 层；加载阶段 `load_weights` 跳过 `mtp.` 前缀的参数，由 spec draft loader 单独接管（与 DeepSeek/Ernie/Bailing 的 MTP 模式一致）。

VLM 路径：`glm4_1v.py` 同时支持 dense 与 MoE 视觉变体，依赖 `SupportsMultiModal` + `MultiModalRegistry` 注册 processor。

---

## 与其它模块/系统配合

- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：三套 MTP draft；target 模型通过 `has_noops` 让 loader 跳过 nextn 层。
- **[多模态](../../11-multimodal/README.md)**：GLM4V / Glm4v / GLM-OCR / GLM-ASR 共享多模态处理栈。
- **[speech-audio](./speech-audio.md)**：`glmasr.py` 实现 `SupportsTranscription`。
- **[deepseek](./deepseek.md)**：GLM-5.2 DSA 复用 `deepseek_v2.py` 实现。
- **[配置](../../10-config/README.md)**：`glm4` 列入 `FP16_NOT_SUPPORTED`（数值稳定性）。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | `chatglm.py` 接入早期 ChatGLM 系列。 |
| v0.5 | `glm.py` 用 `LlamaForCausalLM` 子类化重构。 |
| v0.8–v0.9 | GLM4 / GLM4V / GLM4Moe 入场。 |
| v0.10 | GLM-OCR / GLM-ASR / GLM4-MoE-Lite + 各自 MTP。 |
| v0.11 | GLM-5.2 DSA 复用 DeepSeek V3.2 实现（`GlmMoeDsaForCausalLM`）。 |
| main | GLM4v / Glm4vMoe 视觉变体补齐。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [deepseek 家族](./deepseek.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [多模态](../../11-multimodal/README.md)
