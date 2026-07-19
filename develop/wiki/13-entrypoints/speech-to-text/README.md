[← Wiki 首页](../README.md) > [API 入口](../README.md) > Speech-to-text

# 语音转文本（speech_to_text/）

> `vllm/entrypoints/speech_to_text/` 实现 OpenAI 兼容的音频转写（`/v1/audio/transcriptions`）、翻译（`/v1/audio/translations`）与实时 ASR（WebSocket `/v1/audio/realtime`）。复用 `GenerateBaseServing` 与 `AsyncLLM`，但输入是音频而非文本。

## 是什么

| 文件/类 | 位置 | 职责 |
|---|---|---|
| `factories.py: register_speech_to_text_api_routers` | `vllm/entrypoints/speech_to_text/factories.py:21` | 按 supported_tasks 挂转录/翻译/realtime 路由 |
| `factories.py: add_websocket_metrics_middleware` | `:40` | 给 realtime 加 WebSocket metrics |
| `factories.py: init_speech_to_text_state` | `:46` | 构造 serving 对象挂 state |
| `base/serving.py: SpeechToTextBaseServing` | `vllm/entrypoints/speech_to_text/base/serving.py:90` | ASR 基类，继承 `GenerateBaseServing` |
| `base/serving.py: asr_inter_chunk_separator` | `:79` | 流式 chunk 间分隔处理 |
| `transcription/serving.py: OpenAIServingTranscription` | `vllm/entrypoints/speech_to_text/transcription/serving.py:29` | `/v1/audio/transcriptions` |
| `translation/serving.py: OpenAIServingTranslation` | `vllm/entrypoints/speech_to_text/translation/serving.py:29` | `/v1/audio/translations` |
| `realtime/serving.py: OpenAIServingRealtime` | `vllm/entrypoints/speech_to_text/realtime/serving.py:23` | WebSocket 实时 ASR，继承 `GenerateBaseServing` |
| `realtime/connection.py: RealtimeConnection` | `vllm/entrypoints/speech_to_text/realtime/connection.py:34` | WebSocket 连接抽象 |
| `realtime/metrics.py`/`protocol.py` | — | realtime 指标与协议 |

`base/protocol.py` 定义转写/翻译请求/响应（`TranscriptionRequest`/`TranscriptionResponse`/`TranslationRequest`/`TranslationResponse`，含 verbose 形式）。

`SpeechToTextBaseServing`（`:90`）封装音频文件→多模态输入→`engine_client.generate`→流式/非流式文本输出的共性；`asr_inter_chunk_separator`（`:79`）处理流式 ASR chunk 之间的分隔（避免词跨 chunk 拼接错误）。

realtime（v0.11）走 WebSocket：`RealtimeConnection`（`connection.py:34`）管理双向音频流，`OpenAIServingRealtime`（`serving.py:23`）把音频分块送引擎、流式返回转写/事件。

## 为什么

- **音频一等输入**：把音频作为多模态输入走同一 `AsyncLLM`，复用 KV/调度/MM 基建，无需独立 ASR 服务。
- **OpenAI Audio API 兼容**：`/v1/audio/transcriptions`/`/v1/audio/translations` 字段与 OpenAI Whisper API 一致，客户端可平滑迁移。
- **实时 WebSocket**：v0.11 加 realtime，支持双向流式音频（会议、直播字幕），`RealtimeConnection` 抽象让 ASR 引擎可换。
- **chunk 分隔**：`asr_inter_chunk_separator` 解决流式 chunk 边界词拼接问题，提升可读性。
- **metrics middleware**：realtime 走 WebSocket 不经普通 HTTP instrumentator，单独 `add_websocket_metrics_middleware`（`:40`）采集。

## 怎么做

`build_app`（`api_server.py:221`）在 transcription/realtime task 时挂路由 + WebSocket metrics（`:265`）；`init_app_state`（`:408`）调 `init_speech_to_text_state`。

### 用法

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model="..."
```

实时：WebSocket 连 `/v1/audio/realtime`，按协议发音频 chunk、收转写事件。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| register_speech_to_text_api_routers | `vllm/entrypoints/speech_to_text/factories.py:21` |
| add_websocket_metrics_middleware | `vllm/entrypoints/speech_to_text/factories.py:40` |
| init_speech_to_text_state | `vllm/entrypoints/speech_to_text/factories.py:46` |
| SpeechToTextBaseServing | `vllm/entrypoints/speech_to_text/base/serving.py:90` |
| asr_inter_chunk_separator | `vllm/entrypoints/speech_to_text/base/serving.py:79` |
| OpenAIServingTranscription | `vllm/entrypoints/speech_to_text/transcription/serving.py:29` |
| OpenAIServingTranslation | `vllm/entrypoints/speech_to_text/translation/serving.py:29` |
| OpenAIServingRealtime | `vllm/entrypoints/speech_to_text/realtime/serving.py:23` |
| RealtimeConnection | `vllm/entrypoints/speech_to_text/realtime/connection.py:34` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：`GenerateBaseServing`/`SpeechToTextBaseServing`。
- [openai/api-server.md](../openai/api-server.md)：`build_app`/`init_app_state`。
- [openai/run-batch.md](../openai/run-batch.md)：`BatchTranscriptionRequest`/`BatchTranslationRequest`。
- [多模态](../../11-multimodal/README.md)：音频作为 MM 输入。
- [可观测-metrics](../../16-observability/README.md)：WebSocket metrics。

## 历史版本演进

- **v0.10（transcription/translation）**：`/v1/audio/transcriptions`/`/v1/audio/translations` + `SpeechToTextBaseServing`；run-batch 批转录。
- **v0.11（realtime）**：WebSocket `/v1/audio/realtime` + `RealtimeConnection` + `add_websocket_metrics_middleware`；`asr_inter_chunk_separator`。
- **main**：实时协议事件族（待核实与 OpenAI Realtime API 对齐程度）；压缩音频格式支持。

## 模块导航

| 页 | 主题 |
|---|---|
| [factories.md](factories.md) | 装配 |
| [base.md](base.md) | `SpeechToTextBaseServing` |
| [transcription.md](transcription.md) | `/v1/audio/transcriptions` |
| [translation.md](translation.md) | `/v1/audio/translations` |
| [realtime.md](realtime.md) | WebSocket 实时 ASR |

## 参见

- [← 返回 API 入口首页](../README.md)
- [generate/base-serves.md](../generate/base-serves.md)
- [多模态](../../11-multimodal/README.md)
- [openai/run-batch.md](../openai/run-batch.md)
