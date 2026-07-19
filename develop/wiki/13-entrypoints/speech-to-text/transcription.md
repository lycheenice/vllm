[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Speech-to-text](README.md) > transcription

# speech_to_text/transcription/（/v1/audio/transcriptions）

> `transcription/` 实现 OpenAI 兼容音频转写：接受音频文件 + 模型，流式或一次性返回转写文本（含 timestamps 的 verbose 形式）。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `OpenAIServingTranscription` | `vllm/entrypoints/speech_to_text/transcription/serving.py:29` | 转录 handler，`SpeechToTextBaseServing` 子类 |
| `transcription/protocol.py: TranscriptionRequest`/`TranscriptionResponse`/`TranscriptionResponseVerbose` | — | schema（`run_batch.py:56` import） |
| `transcription/api_router.py` | — | `POST /v1/audio/transcriptions` |

请求经 `_check_model` → 音频作为 MM 输入预处理 → `engine_client.generate` → `SpeechToTextBaseServing` 组装 → 按 `response_format` 序列化（text/json/verbose_json/srt/vtt）。

## 为什么

- **OpenAI Whisper API 兼容**：schema 与字段一致，客户端可平滑替换。
- **verbose 含时间戳**：`TranscriptionResponseVerbose` 含 segments/words 时间戳，便于字幕与检索。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model="..." -F response_format=verbose_json
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIServingTranscription | `vllm/entrypoints/speech_to_text/transcription/serving.py:29` |
| TranscriptionRequest | `vllm/entrypoints/speech_to_text/transcription/protocol.py` |
| 离线 batch 转录 | `vllm/entrypoints/openai/run_batch.py:76`（`BatchTranscriptionRequest`） |

## 与其它模块/系统配合

- [base.md](base.md)：父类。
- [README.md](README.md)：装配。
- [多模态](../../11-multimodal/README.md)：音频 MM 输入。

## 历史版本演进

- **v0.10（引入）**：`/v1/audio/transcriptions` + verbose/srt/vtt。
- **main**：word-level timestamps（待核实）。

## 参见

- [← 返回 Speech-to-text 首页](README.md)
- [base.md](base.md)
