[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > run_batch

# run_batch.py（JSONL 批处理）

> `vllm/entrypoints/openai/run_batch.py` 实现离线批处理：读一份 JSONL 输入文件，按每行 `endpoint` 字段把请求并发送给本地/远程 vLLM server（`/v1/chat/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/score`、`/v1/audio/transcriptions`、`/v1/audio/translations`），把结果写回 JSONL，并跑 Prometheus 指标端点。它是 OpenAI Batch API 的本地等价物。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `BatchTranscriptionRequest`/`BatchTranslationRequest` | `vllm/entrypoints/openai/run_batch.py:76` | 把 transcription/translation 的 `file` 字段换 `file_url` |
| 请求 schema（pydantic） | 文件中部 | 从输入 JSONL 行解析出对应 endpoint 请求体 |
| progress/render/write 协程 | — | `asyncio` 消费 input 队列、调 aiohttp、写 output |
| `main`/`run_batch` | 文件尾 | 参数解析 + `asyncio.run` + Prometheus `start_http_server` |

`run_batch` 走的不是引擎直连，而是 HTTP 客户端：它假设一个已运行的 vLLM server（`--base-url`），用 `aiohttp` 并发打请求，自己只是编排器+进度条+IO。复用 `init_app_state` 仅在使用本地 in-process server 模式时（待核实默认是否走 in-process）。

支持 endpoint（`run_batch.py` imports 可见）：

- `chat/completions`（`ChatCompletionRequest`/`ChatCompletionResponse`）
- `/v1/chat/completions/batch`（`OpenAIServingChatBatch`，待核实是否经 run_batch）
- `embeddings`（`EmbeddingRequest`/`EmbeddingResponse`）
- `rerank`/`score`（`RerankRequest`/`ScoreRequest`）
- `audio/transcriptions`（`BatchTranscriptionRequest`）
- `audio/translations`（`BatchTranslationRequest`）

## 为什么

- **OpenAI Batch 兼容**：输入 JSONL 行格式 `{"method","url","body",...}` 对齐 OpenAI Batch API 文件 schema，便于迁移；输出行附 `response`/`error`。
- **统一多任务**：一份批处理脚本即可跑 chat/embed/rerank/transcribe，按行 `endpoint` 分派，省去为每种任务写脚本。
- **进程内 Prometheus**：`start_http_server(prometheus_port)` 暴露批处理自身指标（已处理/失败/时长），便于在 K8s 里被 scrape。
- **并发可控**：用 `asyncio` + `aiohttp` 连接池，`--max-concurrency` 限流；tqdm 进度条反映速率。
- **复用协议模型**：直接 import 各 endpoint 的 pydantic 协议（`ChatCompletionRequest` 等）做请求校验，保证与在线 API 一致。

## 怎么做

### CLI（`vllm run-batch`）

```
vllm run-batch --base-url http://localhost:8000 \
  --input in.jsonl --output out.jsonl
```

`RunBatchSubcommand`（`cli/run_batch.py:21`）注册子命令；`run_batch`（`openai/run_batch.py`）解析 `--base-url`/`--input`/`--output`/`--response-role`/`--prometheus-port` 等，`asyncio.run` 起 pipeline。

### 输入 JSONL 示例

```jsonc
{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"...","messages":[{"role":"user","content":"hi"}]}}
{"custom_id":"r2","method":"POST","url":"/v1/embeddings","body":{"model":"...","input":"text"}}
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| Batch transcription | `vllm/entrypoints/openai/run_batch.py:76` |
| imports（endpoint 映射） | `vllm/entrypoints/openai/run_batch.py:34` 起 |
| main | `vllm/entrypoints/openai/run_batch.py`（文件尾部，待核实行号） |
| `init_app_state` 复用 | `vllm/entrypoints/openai/run_batch.py:34`（import） |

> 文件尾部 `main`/`run_batch` 与进度协程的精确行号（待补充，需读取尾部 200 行确认）。

## 与其它模块/系统配合

- [chat-completion.md](chat-completion.md)：复用 `ChatCompletionRequest`/`ChatCompletionResponse`。
- [models.md](models.md)：请求 `model` 字段校验由 server 侧 `OpenAIServingModels` 完成。
- [pooling/README.md](../pooling/README.md)：embed/rerank/score 任务。
- [speech-to-text/README.md](../speech-to-text/README.md)：transcription/translation。
- [cli/run-batch-cmd.md](../cli/run-batch-cmd.md)：CLI 注册。
- [可观测-metrics](../../16-observability/README.md)：Prometheus 端点。

## 历史版本演进

- **v0.7–v0.8（批处理初版）**：仅支持 chat/completions 与 embeddings；进程内 `aiohttp` 客户端；tqdm 进度。
- **v0.9（rerank/score）**：加 scoring endpoint；pydantic 协议统一。
- **v0.10（transcription/translation）**：`BatchTranscriptionRequest`/`BatchTranslationRequest` 用 `file_url` 替代 `file`，支持音频批处理；`/v1/chat/completions/batch` endpoint（待核实是否经 run_batch 调用）。
- **v0.11/main**：Prometheus 端点 port 参数化；`--response-role`/`--system-prefix` 等增强；harmony 渲染兼容（待核实）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [chat-completion.md](chat-completion.md)
- [cli/run-batch-cmd.md](../cli/run-batch-cmd.md)
- [pooling/README.md](../pooling/README.md)
- [speech-to-text/README.md](../speech-to-text/README.md)
