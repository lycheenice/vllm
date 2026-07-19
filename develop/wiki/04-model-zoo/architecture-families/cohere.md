# Cohere 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Cohere**

> 代表文件：`commandr.py`、`cohere2_moe.py`、`cohere2_vision.py`、`cohere_asr.py`、`cohere_eagle.py`。
> 厂商：Cohere。

---

## 是什么

- **`commandr.py`**：Command-R / Cohere 基类。`CohereForCausalLM` + 独立 `CohereModel`/`CohereDecoderLayer`/`CohereAttention`/`CohereMLP`。关键差异：用 **QK-norm** + **logit_scale** 缩放 + **sliding window attention** + rotary padding。`Cohere2ForCausalLM` 在注册表里也指向 `commandr.CohereForCausalLM`（`registry.py:85`）。
- **`cohere2_moe.py`**：`Cohere2MoeForCausalLM`，Cohere2 的 MoE 版本，含 `Cohere2Moe` + `Cohere2MoeAttention`；独立 `RMSNorm`、`Cohere2MoeMLP`，签名 `SupportsPP`/`SupportsQuant`。
- **`cohere2_vision.py`**：`Cohere2VisionForConditionalGeneration`，Cohere 多模态版。
- **`cohere_asr.py`**：`CohereAsrForConditionalGeneration`，ASR 转写；goes `_MULTIMODAL_MODELS` 里的 encoder-decoder 区段（`registry.py:575`）；`supports_transcription_only = True`（`cohere_asr.py:2010`）。
- **`cohere_eagle.py`**：`EagleCohereForCausalLM`，EAGLE draft（`registry.py:591`）。

---

## 为什么

- **sliding window + QK-norm 早期实践者**：Cohere 系是 vLLM 早期支持 sliding window attention 的家族之一，对 KV cache 的 sliding 复用机制做了实战验证。
- **任务面相对齐全**：dense / MoE / VLM / ASR / EAGLE 都覆盖，体量适中的"全任务家族"样板。
- **ASR-only 模型**：`cohere_asr.py` 是 `supports_transcription_only=True` 的代表，意味着它不参与文本生成 API，只服务转写——这种 opt-out 在调度器层面省去 generate 路径开销。

---

## 怎么做

`CohereAttention` 在 attention 前先乘 `logit_scale`，attention 后接 QK-norm；sliding window 与 cache_config 协商，前若干 token 不进 KV cache。Cohere2-MoE 走 `FusedMoE` 层库 + QK-norm。

ASR 版本 `CohereAsrForConditionalGeneration` 是 encoder-decoder 结构（注册项在 `_MULTIMODAL_MODELS` 的 `[Encoder-decoder]` 段），与 Whisper 同段。

---

## 与其它模块/系统配合

- **[注意力](../../05-attention/README.md)**：sliding window / QK-norm 的早期实战客户。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：`EagleCohereForCausalLM` draft。
- **[speech-audio](./speech-audio.md)**：`cohere_asr.py` 实现 `SupportsTranscription`。
- **[多模态](../../11-multimodal/README.md)**：Cohere2-Vision。
- **[embedding-col](./embedding-col.md)**：Cohere dense backbone 可被改造成 embedding（注册表条目通过 `_EMBEDDING_MODELS` 复用 Llama 类，不含原生 Cohere pooling）。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.5 | Command-R 接入，sliding window + QK-norm。 |
| v0.6 | `Cohere2` 别名指向同一实现。 |
| v0.8 | Cohere2-MoE + Cohere2-Vision。 |
| v0.9 | Cohere-ASR（transcription-only）。 |
| v0.10 | EAGLE draft。 |
| main | 持续与 attention 后端 sliding window 优化协同。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [注意力](../../05-attention/README.md) · [speech-audio](./speech-audio.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md)
