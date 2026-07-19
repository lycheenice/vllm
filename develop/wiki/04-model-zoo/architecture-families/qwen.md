# Qwen 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Qwen**

> 代表文件：`qwen2.py`、`qwen2_moe.py`、`qwen2_vl.py`、`qwen2_5_vl.py`、`qwen2_audio.py`、`qwen2_5_omni_thinker.py`、`qwen2_rm.py`、`qwen3.py`、`qwen3_moe.py`、`qwen3_next.py`、`qwen3_vl.py`、`qwen3_vl_moe.py`、`qwen3_5.py`、`qwen3_omni_moe_thinker.py`、`qwen3_asr*.py`、`qwen3_eagle3.py`、`qwen3_dflash.py`、`qwen3_dspark.py`、`colqwen3.py`、`colqwen3_5.py`、`qwen3_5_mtp.py`、`qwen3_next_mtp.py`。
> 这是 vLLM 中最大的家族（~23 个文件）。

---

## 是什么

阿里 Qwen 系列几乎覆盖了 vLLM 全部任务面：dense / MoE / VLM / Audio / Omni / ASR / Reward / 检索（Col）。

- **`qwen2.py`**：dense 基类。`Qwen2Model` 继承 `EagleModelMixin`；`Qwen2ForCausalLM` 实现 `SupportsLoRA`/`SupportsPP`/`SupportsQuant`/`SupportsEagle`/`SupportsEagle3`。
- **`qwen3.py`**：vLLM 把 Qwen3 实现为 `Qwen3Model(Qwen2Model)` + `Qwen3ForCausalLM(LocalArgmaxMixin, ...)`。Qwen3 的特色是 `Qwen3Attention`（带 QK-norm），decoder layer 重写。
- **`qwen2_moe.py` / `qwen3_moe.py`**：MoE 版本，含 aux-loss-free 路由 + shared expert。
- **`qwen3_next.py`**：Qwen3-Next，引入 MLA-style attention 与新路由；spec draft `qwen3_next_mtp.py`。
- **VLM**：`qwen2_vl.py`、`qwen2_5_vl.py`（M-RoPE + DP 友好的 vision encoder）、`qwen3_vl.py`、`qwen3_vl_moe.py`（MoE 版）。注册表里 `Qwen3_5ForConditionalGeneration`/`Qwen3_5MoeForConditionalGeneration` 走 `qwen3_5.py`。
- **Audio/Omni**：`qwen2_audio.py`（音频 VLM）、`qwen2_5_omni_thinker.py`（thinker 子模型）、`qwen3_omni_moe_thinker.py`、`qwen3_asr.py`/`qwen3_asr_forced_aligner.py`/`qwen3_asr_realtime.py`（ASR 三个变种，后者支持实时流式）。
- **Reward/检索**：`qwen2_rm.py`（`Qwen2ForRewardModel`/`Qwen2ForProcessRewardModel`）；`colqwen3.py`/`colqwen3_5.py` 实现 `SupportsLateInteraction`（ColBERT 风格逐 token 检索）。
- **spec draft**：`qwen3_eagle3.py`（EAGLE-3 draft）、`qwen3_dflash.py`（DFlash）、`qwen3_dspark.py`（DSpark）；MTP 系列 `qwen3_5_mtp.py`、`qwen3_next_mtp.py`。

---

## 为什么

- **任务面最广，验证接口完备性**：Qwen 把 EAGLE-3、MTP、DFlash、DSpark 四种投机范式覆盖全；同时纳入 ASR 实时流式（`SupportsRealtime`）、Omni 多模态、Reward、Col 检索、token-classification（Forced Aligner），是 vLLM 接口套件的最佳实战样本。
- **M-RoPE 工具的源头**：Qwen2-VL 的 3D M-RoPE 促成了 `vision.py:get_llm_pos_ids_for_vision` 与 `run_dp_sharded_mrope_vision_model`（rope_3d 分支）。
- **Col 检索族**：`colqwen3.py` 表示 Qwen3 backbone 被改造成 late-interaction 检索器，`ColQwen3Model` 继承 `SupportsLateInteraction`；与 `colbert.py`/`colpali.py` 共同构成 retrieval 池。

---

## 怎么做

Qwen2/3 dense 与 Llama 差异主要在：MLP 顺序（`Qwen2MLP` 与 `LlamaMLP` 略微不同）、`Qwen2Attention` 的 QK-norm 与 RMSNorm 位置。MoE 路由用 `FusedMoE` + aux-loss-free bias（与 DeepSeek 共用 `layers/fused_moe` 路由层）。

Qwen3 ASR 的三个文件对应三种用法：`qwen3_asr.py`（离线转写）、`qwen3_asr_realtime.py`（`SupportsRealtime`，`buffer_realtime_audio` 流式）、`qwen3_asr_forced_aligner.py`（token classification，做时间戳对齐）。

---

## 与其它模块/系统配合

- **[多模态](../../11-multimodal/README.md)**：Qwen2-VL 系是 `SupportsMRoPE` 的代表；M-RoPE 位置由 `vision.py` 工具生成。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：EAGLE-3 + MTP + DFlash + DSpark 全套 draft。
- **[embedding-col](./embedding-col.md)**：`colqwen3.py` 复用 Qwen3 backbone。
- **[speech-audio](./speech-audio.md)**：Qwen2-Audio / Qwen3-ASR / Qwen3-Omni 共享音频处理栈。
- **[注意力](../../05-attention/README.md)**：Qwen3-Next 用 MLA，可能走 [`05-attention`](../../05-attention/README.md) MLA 通道。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.3–v0.4 | Qwen（旧 `QWenLMHeadModel`）首批接入，后清退（见 `_PREVIOUSLY_SUPPORTED_MODELS`）。 |
| v0.4 | Qwen2 / Qwen2-MoE / Qwen-VL 落地。 |
| v0.5–v0.6 | Qwen2-VL / Qwen2-Audio 入场，M-RoPE 工具随之补齐。 |
| v0.7 | Qwen2.5-VL / Qwen2.5-Omni（thinker 拆分架构）。 |
| v0.8 | Qwen3 / Qwen3-MoE 入场；EAGLE-3 draft。 |
| v0.10 | Qwen3-Next（MLA-style + MTP）；Qwen3-ASR 与 realtime/forced-aligner 变种；DFlash/DSpark。 |
| v0.11 | Qwen3-VL / Qwen3-VL-MoE / Qwen3-Omni-Moe；ColQwen3 / ColQwen3.5 检索版。 |
| main | Qwen3.5 + Qwen3.5-MTP；任务面持续扩张。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [多模态](../../11-multimodal/README.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [speech-audio](./speech-audio.md) · [embedding-col](./embedding-col.md)
