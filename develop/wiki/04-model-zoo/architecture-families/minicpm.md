# MiniCPM 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **MiniCPM**

> 代表文件：`minicpm.py`、`minicpm3.py`、`minicpmo.py`、`minicpmv.py`、`minicpmv4_6.py`、`minicpm_eagle.py`。
> 厂商：面壁智能（OpenBMB）。

---

## 是什么

- **`minicpm.py`**：MiniCPM 1/2 实现，独立 `MiniCPMForCausalLM` + `MiniCPMModel` + `MiniCPMDecoderLayer`/`MiniCPMAttention`/`MiniCPMMLP`。架构上含 scale emb（放大 embedding 输出）与 Llama 风格残差，但 attention 与 norm 位置略有调整。
- **`minicpm3.py`**：MiniCPM3，走 `MiniCPM3ForCausalLM(MiniCPMForCausalLM)` 子类化复用，`MiniCPM3Model(MiniCPMModel)` 重写少量层。
- **`minicpmv.py`**：MiniCPM-V 视觉对话模型（早期版本），含 ViT 视觉塔 + projector + LM backbone。
- **`minicpmv4_6.py`**：MiniCPM-V 4.6 新版多模态实现（注册项 `MiniCPMV4_6ForConditionalGeneration`）。
- **`minicpmo.py`**：MiniCPM-O 全模态（vision + audio），`MiniCPMO` 注册项。
- **spec draft**：`minicpm_eagle.py: EagleMiniCPMForCausalLM`——EAGLE-1/2 draft（`registry.py:594`）。

---

## 为什么

- **小模型家族的代表**：MiniCPM 主打端侧小模型，vLLM 内的实现要兼顾"小 vocab、scale emb、deep+slim"等设计差异。它没有强行塞进 Llama 子类，而是自营 MiniCPMModel 与 MiniCPMDecoderLayer，保留 attention/MLP 的细节差异。
- **全模态先行者之一**：MiniCPM-O 是早期覆盖 vision + audio 的全模态开源模型，与 Phi4-MM、Qwen2.5-Omni 同列。
- **EAGLE draft 覆盖面**：MiniCPM 提供独立 EAGLE draft，说明小模型也参与 vLLM 投机解码生态。

---

## 怎么做

`MiniCPMDecoderLayer` 与 Llama 的差异在于 attention 后多了一道 `residual`+`RMSNorm` 顺序调整与 `scale_emb`（embedding 输出乘个大常数以匹配深层 norm）。MiniCPM-V/O 的视觉塔用自家 ViT + projector，走标准 `SupportsMultiModal` + `get_mm_mapping`。

EAGLE draft `EagleMiniCPMForCausalLM` 复用 vLLM 通用 EAGLE-1/2 路径，由 target 模型声明 `SupportsEagle` 触发。

---

## 与其它模块/系统配合

- **[多模态](../../11-multimodal/README.md)**：MiniCPM-V / V4.6 / O 共享多模态处理栈。
- **[speech-audio](./speech-audio.md)**：MiniCPM-O 的音频支路。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：`EagleMiniCPMForCausalLM` draft。
- **[LoRA](../../12-lora/README.md)**：含 `MiniCPMV.get_mm_mapping`。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.5 | MiniCPM 1/2 接入。 |
| v0.6 | MiniCPM-V 早期版本。 |
| v0.7 | MiniCPM3 子类化重构（继承 MiniCPM）。 |
| v0.8 | MiniCPM-O 全模态；EAGLE draft。 |
| v0.10 | MiniCPM-V 4.6。 |
| main | 与多模态子系统持续协同。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [多模态](../../11-multimodal/README.md) · [speech-audio](./speech-audio.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md)
