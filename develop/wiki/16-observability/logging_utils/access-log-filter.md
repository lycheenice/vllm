[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/access-log-filter

# Access Log Filter（uvicorn access log 静默）

> 源码：`vllm/logging_utils/access_log_filter.py`（144 行）

## 是什么

`access_log_filter.py` 为 uvicorn access log 提供"`/health`、`/metrics` 等高频路径不打 access log"的能力，避免 production 日志被健康检查/指标抓取刷爆。

| 类/函数 | 行号 | 角色 |
|---|---|---|
| `UvicornAccessLogFilter(logging.Filter)` | `access_log_filter.py:15` | `Filter` 子类；按 `excluded_paths` 集合过滤 access log |
| `create_uvicorn_log_config(excluded_paths, log_level)` | `access_log_filter.py:71` | 生成 uvicorn `log_config` dict，含 default/access formatter + default/access handler + uvicorn logger 树 + access_log_filter |

`UvicornAccessLogFilter` 实现：

- `__init__(excluded_paths: list[str] | None)`：保存 `excluded_paths` 为 `set`。
- `filter(record)`：
  1. `excluded_paths` 空时直接 `return True`。
  2. `record.name != "uvicorn.access"` 时 `return True`（仅过滤 uvicorn.access logger）。
  3. `record.args` 是 tuple 且 `len >= 3` 时取第 3 个元素（path with query），按 `urlparse(...).path` 取 path 部分。
  4. path in `excluded_paths` 时 `return False`，否则 `True`。

> uvicorn access log 格式 `'%s - "%s %s HTTP/%s" %d'`（client_addr, method, path, http_version, status_code），`record.args[2]` 是 path。

`create_uvicorn_log_config` 产出的 dict 结构：

```python
{
  "version": 1,
  "disable_existing_loggers": False,
  "filters": {"access_log_filter": {"()": UvicornAccessLogFilter, "excluded_paths": [...]}},
  "formatters": {
    "default": {"()": "uvicorn.logging.DefaultFormatter", "fmt": "%(levelprefix)s %(message)s", "use_colors": None},
    "access":  {"()": "uvicorn.logging.AccessFormatter", "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'},
  },
  "handlers": {
    "default": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
    "access":  {"formatter": "access",  "class": "logging.StreamHandler", "stream": "ext://sys.stdout",
                "filters": ["access_log_filter"]},
  },
  "loggers": {
    "uvicorn":        {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
    "uvicorn.error":  {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
    "uvicorn.access": {"handlers": ["access"],  "level": LOG_LEVEL, "propagate": False},
  },
}
```

## 为什么

- **日志洪水**：k8s liveness/readiness probe 每 1-2s 一次 `/health`，prometheus 抓取 `/metrics` 每 15-60s 一次——若 access log 默认开，access log 会被这些路径淹没，掩盖真正业务请求。
- **路径精确匹配**：用 `set` + `path in excluded_paths` 简单精确匹配；不带 query string（先 `urlparse(...).path`）避免误伤。
- **仅 uvicorn.access**：`record.name != "uvicorn.access"` 时直接放行——不让 filter 影响其他 logger（防止误绑到 root logger 上）。
- **dictConfig 友好**：返回标准 `logging.config.dictConfig` 兼容的 dict，直接交 `uvicorn.run(log_config=...)`。
- **access handler 走 stdout，default/error 走 stderr**：让 access log 与 error log 在容器化环境分流（k8s `kubectl logs` 区分 stdout/stderr）。
- **`propagate=False`**：避免 uvicorn.* logger 把日志冒泡到 root logger 与 vllm logger 重复输出。
- **`use_colors=None`**：让 uvicorn 自己检测终端是否支持色（uvicorn DefaultFormatter 行为）。

## 怎么做

**API server 启用 Filter**（vLLM serve 默认行为，具体调用点 `vllm/entrypoints/openai/api_server.py`，待核实）：

```python
import uvicorn
from vllm.logging_utils import create_uvicorn_log_config

uvicorn.run(
    app,
    host=host, port=port,
    log_config=create_uvicorn_log_config(
        excluded_paths=["/health", "/metrics"],
        log_level="info",
    ),
)
```

**自定义排除路径**：用户在自家 fork 或 plugin 里调 `create_uvicorn_log_config(["/health", "/metrics", "/ready", "/live"])`。CLI 层级暴露（待核实，目前似乎为硬编码 `["/health", "/metrics"]`）。

**直接拿 Filter 用**（用户已有 uvicorn 日志配置想叠加）：

```python
from vllm.logging_utils import UvicornAccessLogFilter

# 加到现有 access handler 的 filters 列表
filter = UvicornAccessLogFilter(excluded_paths=["/health"])
access_handler.addFilter(filter)
```

## 与其它模块/系统配合

- **[../logger.md](../logger.md)**：vllm logger 与 uvicorn logger 是独立的（vLLM 的 `_configure_vllm_root_logger` 仅配 `vllm` logger，不碰 uvicorn.*）；本 Filter 是 uvicorn 端配置。
- **[13-entrypoints/serve](../../13-entrypoints/README.md)** / [13-entrypoints/openai](../../13-entrypoints/README.md)：API server 启动入口调 `create_uvicorn_log_config`。
- **[13-entrypoints/serve/instrumentator.md](../../13-entrypoints/serve/instrumentator.md)**（待补充）—— `/metrics` 端点本身由 `prometheus_client.make_asgi_app()` 提供，与本 filter 的"静默访问日志"是两个独立功能但常配合。
- **uvicorn**：依赖 uvicorn access logger 的 args 元组格式，硬编码 `args[2]`。uvicorn 升级若改 args 顺序会破坏 Filter（版本兼容性待核实）。
- **`urllib.parse.urlparse`**：path 解析依赖标准库。

## 历史版本演进

- **v0.5/v0.6（v0）**：无此模块；vLLM 用 uvicorn 默认 log filter，access log 包含 `/health`；用户反馈日志洪水。
- **v0.7（v1 落地）**：引入 `vllm/logging_utils/access_log_filter.py`；`UvicornAccessLogFilter` + `create_uvicorn_log_config`；API server 默认排除 `/health`、`/metrics`。
- **v0.8–main**：稳定 API；用户反馈路径变体（如 `/health?latency_check=1`）由 `urlparse(...).path` 统一剥离 query。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [logging-readme.md](logging-readme.md) — 子目录总览。
- [../logger.md](../logger.md) — vllm logger 配置（独立 logger 树）。
