# Speech / Audio 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Speech/Audio**

> 代表文件：`whisper.py`、`whisper_causal.py`、`whisper_utils.py`、`ultravox.py`、`voxtral.py`、`voxtral_realtime.py`、`parakeet.py`、`funasr.py`、`funaudiochat.py`、`fireredasr2.py`、`fireredlid.py`、`kimi_audio.py`、`glmasr.py`/`glmasr_utils.py`、`moss_audio.py`、`moss_transcribe_diarize.py`、`audioflamingo3.py`、`midashenglm.py`、`conformer_encoder.py`、`aimv2.py`、`hyperclovax.py`（含 ASR）、`granite_speech.py`/`granite_speech_plus.py`（见 [granite](./granite.md)）、`cohere_asr.py`（见 [cohere](./cohere.md)）、`qwen2_audio.py`、`qwen3_asr.py`/`qwen3_asr_forced_aligner.py`/`qwen3_asr_realtime.py`（见 [qwen](./qwen.md)）、`phi4mm_audio.py`（见 [phi](./phi.md)）、`mimo_audio.py`（见 [asian-vendor](./asian-vendor.md)）。

---

## 是什么

Audio 系模型分两类：**纯 ASR 转写** 与 **音频对话/多模态**。

### ASR 转写（`SupportsTranscription`）

- **`whisper.py`**：`WhisperForConditionalGeneration`——encoder-decoder 标准 Whisper；含 `WhisperEncoderAttention`、`WhisperCrossAttention`、`supports_transcription_only = True`（`whisper.py:786`）。`whisper_causal.py`/`whisper_utils.py` 提供因果变体与工具。
- **`voxtral.py` / `voxtral_realtime.py`**：Mistral Voxtral——`VoxtralForConditionalGeneration` 与 `VoxtralRealtimeGeneration`（`SupportsRealtime`，流式）。
- **`parakeet.py`**：NVIDIA Parakeet（含 ASR/CTC）。
- **`funasr.py`**：阿里 FunASR——`FunASRForConditionalGeneration`，`supports_transcription_only = True`。
- **`funaudiochat.py`**：FunAudioChat（对话版音频）。
- **`fireredasr2.py` / `fireredlid.py`**：FireRed ASR2 / FireRed LID（语言识别），`supports_transcription_only = True`。
- **`glmasr.py`**：智谱 GLM-ASR。
- **`kimi_audio.py`**：`KimiAudioForConditionalGeneration`，`supports_transcription=True`。
- **`moss_transcribe_diarize.py`**：MOSS 转写 + 说话人分隔，`supports_transcription_only = True`。
- **`cohere_asr.py`**：Cohere ASR（encoder-decoder，transcription-only）。
- **`qwen3_asr*.py`**：Qwen3 ASR 三变种——离线 / forced-aligner（token cls）/ realtime。

### 多模态音频对话

- **`ultravox.py`**：Ultravox（音频 + 文本，把 audio encoder 接到 LM）。`UltravoxModel`。
- **`audioflamingo3.py`**：Audio Flamingo 3。
- **`granite_speech.py` / `granite_speech_plus.py`**：IBM Granite Speech/Speech Plus。
- **`moss_audio.py`**：MOSS Audio。
- **`midashenglm.py`**：小米 MiDashengLM。
- **`aimv2.py` / `conformer_encoder.py`**：AIMv2 视觉/语音编码器、Conformer encoder（CTC 风格），供其他模型作音频塔。
- **`hyperclovax.py`**：HCX（含 ASR/字幕能力，与 `hyperclovax_vision*.py` 系）。

`qwen2_audio.py`、`mimo_audio.py`、`phi4mm_audio.py`、`qwen2_5_omni_thinker.py`、`qwen3_omni_moe_thinker.py`、`mimo_v2_omni.py`、`gemma3n_audio_utils.py` 等属 Omni/多模态家族，已在各自家族页。

---

## 为什么

- **`SupportsTranscription` 接口的核心落地点**：模型实现 `supported_languages`、`get_generation_prompt`、`get_speech_to_text_config`、`get_num_audio_tokens` 等类方法，让 `/v1/audio/transcriptions` API（见 [13-entrypoints](../../13-entrypoints/README.md)）按统一契约工作。
- **`supports_transcription_only` opt-out**：Whisper/FunASR/FireRedASR2/FireRedLID/CohereASR/MOSS-Diarize 设为 True，意味着不参与文本生成 API，调度器只跑转写路径。
- **`SupportsRealtime` 流式**：Voxtral Realtime 与 Qwen3 ASR Realtime 实现 `buffer_realtime_audio` 异步生成器，对接流式 audio API。
- **encoder-decoder 复活**：Whisper / Cohere ASR / NemotronParse 等是 vLLM 里少数 encoder-decoder 模型（V0 退役期保留 Whisper 作"唯一 encoder-decoder"，其他几款是后期重新引入）。

---

## 怎么做

`WhisperForConditionalGeneration` 含 `WhisperEncoder`（音频 log-mel谱→隐状态）+ `WhisperDecoder`（自回归，带 cross-attention 到 encoder 输出）；`WhisperMultiModalProcessor` 把音频 → mel → input ids。forced aligner 走 `Qwen3ASRForcedAlignerForTokenClassification`（注册到 `_TOKEN_CLASSIFICATION_MODELS`）——同 token cls 路径。

`SupportsRealtime.buffer_realtime_audio` 是 `async` 异步生成器：从 `audio_stream` 读流式 ndarray，按 `realtime_max_tokens` 切段产 `PromptType`，喂给推理。

`ultravox.py` 用 audio encoder（Whisper/Audio Tower）的输出经 connector 投影到 LM 隐空间，scatter 进 LM input embedding——VLM 套路在 audio 上的复刻。

---

## 与其它模块/系统配合

- **[interfaces.md](../interfaces.md)**：`SupportsTranscription` / `SupportsRealtime` 接口定义。
- **[13-entrypoints](../../13-entrypoints/README.md)**：`/v1/audio/transcriptions`、`/v1/audio/translations`、流式 audio API 由这些模型供能。
- **[多模态](../../11-multimodal/README.md)**：音频模态处理；`conformer_encoder.py` 提供共享 encoder。
- **[embedding-col](./embedding-col.md)**：`Qwen3ASRForcedAligner` 走 token classification 通路。
- **[transformers-backend](../transformers-backend.md)**：encoder-decoder 复杂度较高，部分厂家选择 HF 后端 fallback（待核实）。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.5 | Whisper 接入（仅 encoder-decoder 保留）。 |
| v0.7 | Ultravox + Qwen2-Audio。 |
| v0.8 | Voxtral + Parakeet + FireRedASR/LID。 |
| v0.9 | Kimi-Audio + Granite-Speech + GLM-ASR。 |
| v0.10 | FunASR + FunAudioChat + MOSS-Audio/Diarize + AudioFlamingo3 + Qwen3-ASR/Realtime/ForcedAligner。 |
| v0.11 | MiDashengLM + Granite-Speech-Plus + MiMo-Audio。 |
| main | `supports_explicit_language_detection` 子能力落地（Whisper 风格的语言预测）。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [interfaces](../interfaces.md) · [13-entrypoints](../../13-entrypoints/README.md) · [多模态](../../11-multimodal/README.md) · [granite](./granite.md) · [cohere](./cohere.md) · [qwen](./qwen.md)
