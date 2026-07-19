# Granite 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Granite**

> 代表文件：`granite.py`、`granitemoe.py`、`granitemoehybrid.py`、`granitemoeshared.py`、`granite_speech.py`、`granite_speech_plus.py`、`granite4_vision.py`。
> 厂商：IBM。

---

## 是什么

- **`granite.py`**：Granite dense 文本主干，独立 `GraniteForCausalLM` + `GraniteModel` + `GraniteAttention`/`GraniteMLP`/`GraniteDecoderLayer`。架构与 Llama 接近，但带 Granite 特有的注意力缩放与 tied-embedding 处理。
- **`granitemoe.py`**：`GraniteMoeForCausalLM` + `GraniteMoeMoE`，MoE 版本。`SupportsLoRA`/`SupportsPP`。
- **`granitemoehybrid.py`**：Granite-MoE-Hybrid，混合 Mamba+Attention+MoE，`HasInnerState`+`IsHybrid`（见 [mamba-ssm](./mamba-ssm.md)）。
- **`granitemoeshared.py`**：`GraniteMoeSharedForCausalLM` + `GraniteMoeShared`——含 shared expert 的 MoE 变体；与 DeepSeek/Qwen 的 shared expert 模式同级。
- **`granite_speech.py` / `granite_speech_plus.py`**：Granite Speech 与 Granite Speech Plus，语音对话/转写 VLM。
- **`granite4_vision.py`**：`Granite4VisionForConditionalGeneration`，Granite 4 视觉模型。

---

## 为什么

- **MoE 变体齐全**：Granite 是少有的"Moe / Moe-Hybrid / Moe-Shared 三件套"全部上线的家族，给 vLLM `FusedMoE` 层库提供了多种 expert 拓扑的实战测试。
- **speech 系的两个分支**：`granite_speech` 与 `granite_speech_plus` 并存反映语音 Granite 在能力分档（基础版 + Plus 版）上的设计差异——后者可能含更长上下文或额外模态（待核实）。
- **跨家族标签**：`granitemoehybrid` 与 LFM2/Jamba/Nemotron-H 共享 hybrid SSM 接口语义。

---

## 怎么做

`GraniteMoeShared` 在 `GraniteMoeMoE` 基础上加 shared expert 路径：每个 token 除 top-k routed expert 外，还经过 shared expert（无路由），结果相加。这与 DeepSeek V2/V3 的 shared expert 实现思路一致，但走 IBM 自家 MoE wrapper。

`granite4_vision.py` 把 Granite 4 backbone + 视觉塔（具体塔类型待核实）+ connector 拼成 VLM，注册到 `_MULTIMODAL_MODELS`。

---

## 与其它模块/系统配合

- **[mamba-ssm](./mamba-ssm.md)**：`granitemoehybrid` 共享 hybrid 接口栈。
- **[speech-audio](./speech-audio.md)**：granite_speech 两版。
- **[多模态](../../11-multimodal/README.md)**：granite4_vision。
- **[分布式](../../07-distributed/README.md)**：MoE 模型走 `FusedMoE` 路径与专家迁移。
- **[模型执行-层库](../../03-model-execution/layers/README.md)**：`FusedMoE` shared expert 模式。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.6 | Granite dense 接入。 |
| v0.7 | Granite-MoE。 |
| v0.8 | Granite-MoE-Hybrid + Granite-MoE-Shared。 |
| v0.9–v0.10 | Granite Speech / Speech Plus。 |
| v0.11 | Granite4-Vision。 |
| main | 持续与 hybrid 接口协同。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [mamba-ssm](./mamba-ssm.md) · [speech-audio](./speech-audio.md) · [多模态](../../11-multimodal/README.md)
