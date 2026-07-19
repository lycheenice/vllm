# HTTP 连接（connections）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > HTTP 连接

本页覆盖 `vllm/connections.py`（394 行），提供 vLLM 内部统一的 HTTP 客户端 `HTTPConnection` 与全局实例 `global_http_connection`。

## 是什么

### `HTTPConnection`（`vllm/connections.py:200`）

封装同步 `requests.Session` 与异步 `aiohttp.ClientSession` 双客户端，按需懒创建（`reuse_client=True` 时复用）。

对外方法（同步/异步成对）：

- `get_response`/`get_async_response`：返回原始 Response（可 stream）。
- `get_bytes`/`async_get_bytes`：取 bytes（带重试）。
- `get_text`/`async_get_text`、`get_json`/`async_get_json`：便捷封装。
- `download_file`/`async_download_file`：分块下载到 `Path`，失败时清理半成品文件（带重试）。

URL 校验：`_validate_http_url`（`:225`）只允许 `http`/`https` scheme，挡住 file/ftp 等。Header 固定带 `User-Agent: vLLM/<version>`（`_headers`，`:233`），便于远端识别。

### 重试机制

- `_RETRY_BACKOFF_FACTOR = 4`（`:27`）：每次重试 per-attempt timeout 乘 4，sleep 也按 4^n 秒退避，吸收"繁忙主机瞬时慢"。
- `_is_retryable(exc)`（`:30`）：可重试 = 超时（aiohttp/requests/stdlib）、连接级失败（refused/reset/DNS）、服务端 5xx（含 S3 503 SlowDown）、`ServerDisconnectedError`；不可重试 = 4xx 客户端错误、编程错误。
- `_sync_retry`/`_async_retry`（`:108`/`:154`）：装饰器，从 `kwargs["timeout"]` 取 base timeout，按 attempt 递增；`max_retries` 取自 `envs.VLLM_MEDIA_FETCH_MAX_RETRIES`（默认 3，见 [envs.md](envs.md)）。
- `_log_retry`（`:76`）：每次重试告警含 URL / 次数 / timeout / 退避秒数。

### 全局实例

`global_http_connection = HTTPConnection()`（`:390`）：vLLM 各模块共享的默认连接。

## 为什么

- **单一出口**：所有远程 HTTP（多模态媒体、权重 manifest、API server 反向探测等）走同一客户端，便于统一加 UA、超时、重试、代理（`aiohttp.ClientSession(trust_env=True)` 读 `HTTP_PROXY`）。
- **重试策略集中**：媒体拉取常遇瞬时 5xx/S3 SlowDown，集中退避避免每处各自实现。
- **同步/异步双栈**：API server 是 async，但 CLI/LLM 同步路径也需要拉取，提供对偶 API。
- **安全**：强制 scheme 白名单，避免误把 `file://`/本地路径当 URL 拉取。

## 怎么做

```python
from vllm.connections import global_http_connection
data = global_http_connection.get_bytes(url, timeout=10)
await global_http_connection.async_download_file(url, Path("a.bin"), timeout=30)
```

- 想自定义超时/重试：直接调 `get_bytes(url, timeout=...)`，重试次数由全局 env 控制，无法按调用覆盖（如需覆盖可另建 `HTTPConnection` 实例）。
- 代理：设 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量，`trust_env=True` 自动生效。

## 与其它模块/系统配合

- [多模态](../11-multimodal/README.md)：`fetch_image`/`fetch_video`/`fetch_audio` 经 `global_http_connection` 拉远程媒体，超时取 `VLLM_IMAGE_FETCH_TIMEOUT`/`_VIDEO_`/`_AUDIO_`。
- [API 入口](../13-entrypoints/README.md)：部分入口探测外部 URL；OpenAI 兼容 server 反向解析模型 manifest。
- [权重加载](../03-model-execution/README.md)：`HF hub` 直连或 `VLLM_USE_MODELSCOPE` 走 ModelScope 时，部分元数据拉取经此通道（待核实，主下载仍走 huggingface_hub）。
- [envs.md](envs.md)：`VLLM_MEDIA_FETCH_MAX_RETRIES`、各 `*_FETCH_TIMEOUT`、`VLLM_MEDIA_URL_ALLOW_REDIRECTS`。

## 历史版本演进

- **v0.5–v0.6**：媒体拉取逻辑分散在 `vllm/multimodal/utils.py`，用裸 `requests`/`aiohttp`，无统一重试。
- **v0.7–v0.8**：`vllm/connections.py` 引入 `HTTPConnection`，集中 sync/async 客户端，但重试较简单。
- **v0.9–v0.10**：指数退避 + per-attempt timeout 递增机制上线（`_RETRY_BACKOFF_FACTOR=4`），覆盖 S3 5xx/SlowDown；`download_file` 加半成品清理。
- **v0.11–main**：`User-Agent` 带 vLLM 版本；`_validate_http_url` scheme 白名单收紧；重试由固定次数改为 env 驱动（待核实具体 PR）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [envs.md](envs.md)（`VLLM_MEDIA_*` 族）
- [多模态子系统](../11-multimodal/README.md)
