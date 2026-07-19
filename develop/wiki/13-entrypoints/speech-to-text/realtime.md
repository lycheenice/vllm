[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Speech-to-text](README.md) > realtime

# speech_to_text/realtime/（WebSocket 实时 ASR）

> `realtime/` 实现 v0.11 引入的实时音频转写：WebSocket `/v1/audio/realtime` 双向流式，客户端推音频 chunk，服务端流式返回转写/事件。`RealtimeConnection` 抽象连接生命周期。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `OpenAIServingRealtime` | `vllm/entrypoints/speech_to_text/realtime/serving.py:23` | WebSocket handler，继承 `GenerateBaseServing` |
| `RealtimeConnection` | `vllm/entrypoints/speech_to_text/realtime/connection.py:34` | 连接抽象（收音频/发事件/生命周期） |
| `realtime/protocol.py` | — | 实时事件 schema（待核实字段） |
| `realtime/metrics.py` | — | 实时指标（与 `add_websocket_metrics_middleware` 配合） |

`OpenAIServingRealtime`（`:23`）在 WebSocket 握手后，用 `RealtimeConnection`（`:34`）接收音频分块 → 提交 `engine_client.generate`（流式）→ 把转写 token 经 `OnlineRenderer`/derenderer 还原成事件 → 流回客户端。支持中断、回合管理（待核实与 OpenAI Realtime API 对齐程度）。

## 为什么

- **双向流式**：会议、直播字幕需要边说边出字，HTTP 请求-响应模型不够；WebSocket 全双工适合。
- **connection 抽象**：`RealtimeConnection` 隔离 WebSocket 细节，handler 专注 ASR 逻辑，便于换传输（如 gRPC bidirectional stream）。
- **独立 metrics**：WebSocket 不经 HTTP instrumentator，单独 middleware 采集连接数/延迟。
- **复用 `GenerateBaseServing`**：ASR 引擎链与非实时一致，仅传输层不同。

## 怎么做

WebSocket 连 `/v1/audio/realtime`，按协议（待核实具体 event schema，对齐 OpenAI Realtime API）发 `audio_append`/`commit`/`response.create` 等事件，收 `transcript.delta`/`transcript.done` 等。

`build_app` 在 realtime task 时额外挂 `add_websocket_metrics_middleware`（`api_server.py:265`）。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIServingRealtime | `vllm/entrypoints/speech_to_text/realtime/serving.py:23` |
| RealtimeConnection | `vllm/entrypoints/speech_to_text/realtime/connection.py:34` |
| metrics middleware | `vllm/entrypoints/speech_to_text/factories.py:40` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：父类。
- [factories.md](factories.md)：metrics middleware。
- [多模态](../../11-multimodal/README.md)：音频流式 MM 输入。
- [可观测-metrics](../../16-observability/README.md)：WebSocket metrics。

## 历史版本演进

- **v0.11（引入）**：WebSocket `/v1/audio/realtime` + `RealtimeConnection` + `add_websocket_metrics_middleware`。
- **main**：事件族与 OpenAI Realtime API 对齐（待核实完整对齐度）；压缩音频流支持。

## 参见

- [← 返回 Speech-to-text 首页](README.md)
- [generate/base-serves.md](../generate/base-serves.md)
- [factories.md](factories.md)
