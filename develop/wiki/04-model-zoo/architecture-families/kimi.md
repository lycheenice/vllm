# Kimi 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Kimi**

> 代表文件：`kimi_linear.py`、`kimi_vl.py`、`kimi_k25.py`、`kimi_k25_vit.py`、`kimi_audio.py`。
> 厂商：Moonshot AI。

---

## 是什么

- **`kimi_linear.py`**：Kimi-Linear，MoE + MLA + **hybrid SSM** 三合一。`KimiLinearForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid)`——同时挂 MoE 接口与 hybrid 标签。含 `KimiMLAAttention`（MLA attention）+ `KimiMoE`（aux-loss-free 路由）+ `KimiDecoderLayer`。这是 vLLM 内最"综合架构"的家族之一。
- **`kimi_vl.py`**：Kimi-VL VLM，`KimiVLForConditionalGeneration` + 2D-RoPE 视觉塔（与 Qwen2.5-VL 的 3D-rope 区分，见 `vision.py` 的 `rope_2d` 分支）+ `supports_encoder_tp_data=True`。
- **`kimi_k25.py` + `kimi_k25_vit.py`**：Kimi K25 系列视觉模型，与 Kimi-VL 并列新一代；含 `KimiK25ProcessingInfo`/`KimiK25DummyInputsBuilder`/`KimiK25MediaPixelInputs`。
- **`kimi_audio.py`**：`KimiAudioForConditionalGeneration`，`supports_transcription: ClassVar[Literal[True]] = True`（`kimi_audio.py:367`）——Kimi Audio 支持 ASR 转写。

---

## 为什么

- **三种 frontier 架构汇合**：Kimi-Linear 同时实现 MLA + aux-loss-free MoE + hybrid SSM，是 DeepSeek V3 + Jamba 的"合并体"。它把 vLLM `HasInnerState`/`IsHybrid`/`MixtureOfExperts` 三个接口一次性启用，是检验接口组合性的样本。
- **2D-RoPE 视觉塔**：Kimi-VL 用 `rope_2d`（merge_kernel_size 风格），与 Qwen2.5-VL 的 `rope_3d` 并行存在；`vision.py:run_dp_sharded_mrope_vision_model` 为此提供单函数双分支。
- **encoder TP data 友好**：`supports_encoder_tp_data=True` 让视觉塔走 DP-on-encoder 模式，与 Llama4 / Qwen2-VL 等并列。

---

## 怎么做

Kimi-Linear 的 decoder layer 按 `hf_config.layers_block_type` 在 MLA attention 层、Mamba 层、MoE MLP 层间组合，与 Jamba/Nemotron-H 类似但 attention 走 MLA。`KimiMLAAttention` 复用 `MLAAttention` 基类（与 DeepSeek 同源，但 latent 维度/bias 不同）。

Kimi-VL 视觉塔走 2D-RoPE，`run_dp_sharded_mrope_vision_model(..., rope_type="rope_2d")` 按 `merge_kernel_size` 算 reduction_factor。

`KimiAudioForConditionalGeneration` 实现 `SupportsTranscription.supported_languages` 等类方法，走 ASR 路径（与 Whisper/Voxtral 并列）。

---

## 与其它模块/系统配合

- **[deepseek](./deepseek.md)**：Kimi-Linear 复用 MLA 与 DeepSeek 同源（具体类继承待核实）；MoE 路由与 DeepSeek/Qwen MoE 同走 `FusedMoE`。
- **[mamba-ssm](./mamba-ssm.md)**：hybrid SSM 标签共享。
- **[vision.md](../vision.md)**：Kimi-VL 是 `rope_2d` 分支的代表客户。
- **[speech-audio](./speech-audio.md)**：Kimi-Audio 转写。
- **[多模态](../../11-multimodal/README.md)** / **[07 分布式](../../07-distributed/README.md)**：encoder TP 支持 + DP 分片。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.9 | Kimi-VL 与 Kimi-Audio 上线。 |
| v0.10 | Kimi-Linear（MLA + MoE + hybrid）落地。 |
| v0.11 | Kimi K25 视觉系列。 |
| main | 与 DeepSeek V4/Qwen3-Next 共享 MLA 优化；DP 视觉塔持续协同。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [deepseek](./deepseek.md) · [mamba-ssm](./mamba-ssm.md) · [vision](../vision.md) · [speech-audio](./speech-audio.md)
