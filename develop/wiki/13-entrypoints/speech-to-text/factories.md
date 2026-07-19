[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Speech-to-text](README.md) > factories

# speech_to_text/factories.py（ASR 装配）

> `factories.py` 按支持任务挂 transcription/translation/realtime 路由、加 WebSocket metrics、初始化 state。

## 是什么

| 函数 | 位置 | 职责 |
|---|---|---|
| `register_speech_to_text_api_routers` | `vllm/entrypoints/speech_to_text/factories.py:21` | 按 supported_tasks 挂路由 |
| `add_websocket_metrics_middleware` | `:40` | realtime WebSocket metrics middleware |
| `init_speech_to_text_state` | `:46` | 构造 `OpenAIServingTranscription`/`Translation`/`Realtime` 挂 state |

`build_app` 在 transcription/realtime task 时调用（`api_server.py:221/265`）。

## 为什么

- **按 task 精准挂**：模型可能仅支持 transcription 不支持 realtime，按 `supported_tasks` 过滤。
- **WebSocket metrics 独立**：realtime 不经 HTTP middleware 链，需单独挂。

## 怎么做

见 [README](README.md)。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| register_speech_to_text_api_routers | `vllm/entrypoints/speech_to_text/factories.py:21` |
| add_websocket_metrics_middleware | `vllm/entrypoints/speech_to_text/factories.py:40` |
| init_speech_to_text_state | `vllm/entrypoints/speech_to_text/factories.py:46` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：调用方。
- [realtime.md](realtime.md)：WebSocket metrics。

## 历史版本演进

- **v0.10（transcription/translation 装配）**。
- **v0.11（realtime + WebSocket metrics）**。

## 参见

- [← 返回 Speech-to-text 首页](README.md)
