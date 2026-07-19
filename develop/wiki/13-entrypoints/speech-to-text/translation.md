[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Speech-to-text](README.md) > translation

# speech_to_text/translation/（/v1/audio/translations）

> `translation/` 实现音频翻译：接受音频，输出指定目标语言的文本（与 transcription 协议类似，但模型做语音→目标语言文本转换）。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `OpenAIServingTranslation` | `vllm/entrypoints/speech_to_text/translation/serving.py:29` | 翻译 handler，`SpeechToTextBaseServing` 子类 |
| `translation/protocol.py: TranslationRequest`/`TranslationResponse`/`TranslationResponseVerbose` | — | schema（`run_batch.py:61` import） |
| `translation/api_router.py` | — | `POST /v1/audio/translations` |

流程同 transcription，差异在 prompt/sampler 引导模型产出目标语言（如统一译英文）。

## 为什么

- **与 transcription 共享基建**：仅协议与引导语言不同，复用 `SpeechToTextBaseServing`。
- **OpenAI 兼容**：`/v1/audio/translations` 字段一致。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F file=@audio.mp3 -F model="..."
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIServingTranslation | `vllm/entrypoints/speech_to_text/translation/serving.py:29` |
| TranslationRequest | `vllm/entrypoints/speech_to_text/translation/protocol.py` |
| 离线 batch 翻译 | `vllm/entrypoints/openai/run_batch.py`（`BatchTranslationRequest`） |

## 与其它模块/系统配合

- [base.md](base.md)：父类。
- [多模态](../../11-multimodal/README.md)：音频 MM 输入。

## 历史版本演进

- **v0.10（引入）**：`/v1/audio/translations` + verbose/srt/vtt。
- **main**：与 transcription 协同改进。

## 参见

- [← 返回 Speech-to-text 首页](README.md)
- [base.md](base.md)
- [transcription.md](transcription.md)
