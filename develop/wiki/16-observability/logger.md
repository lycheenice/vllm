[← Wiki 首页](../README.md) > [可观测](README.md) > logger

# Logger（全局 logger 配置入口）

> 源码：`vllm/logger.py`（314 行）

## 是什么

`vllm/logger.py` 是 vLLM 全局 logging 配置入口。模块 import 时即调用 `_configure_vllm_root_logger()` 配置 `vllm` 根 logger；所有子模块通过 `init_logger(__name__)` 拿到的 logger 自动挂到根。它还提供 `debug_once`/`info_once`/`warning_once` 三方法（通过 `_VllmLogger` 类型提示 + 方法 patching 注入 logger 实例）与 `enable_trace_function_call` 系统级 trace 工具。

主要 API：

| 名 | 行号 | 角色 |
|---|---|---|
| `_FORMAT` / `_DATE_FORMAT` | `logger.py:22` / `:26` | 默认 logger fmt：`{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s` / `%m-%d %H:%M:%S` |
| `_use_color()` | `logger.py:29` | 决定是否着色：`NO_COLOR`/`VLLM_LOGGING_COLOR=0` 关；`VLLM_LOGGING_COLOR=1` 开；否则根据 stream isatty |
| `DEFAULT_LOGGING_CONFIG` | `logger.py:41` | dictConfig 结构：`vllm` 与 `vllm_color` 两个 formatter（class 指 [NewLineFormatter/ColoredFormatter](logging_utils/formatter.md)），`vllm` handler 选其一，`vllm` logger 关 propagate |
| `_print_debug_once` / `_print_info_once` / `_print_warning_once` | `logger.py:75/81/87` | `@lru_cache` 包裹的 once-only 打印 |
| `LogScope` Literal | `logger.py:93` | `"process" \| "global" \| "local"` |
| `_should_log_with_scope(scope)` | `logger.py:96` | `global` → `is_global_first_rank()`；`local` → `is_local_first_rank()`；`process` → True |
| `_VllmLogger(Logger)` | `logger.py:109` | 类型提示类（不实例化）；`debug_once`/`info_once`/`warning_once` 方法由 `init_logger` 通过 `MethodType` patch 到 logger 实例 |
| `_METHODS_TO_PATCH` | `logger.py:149` | 三个方法名→函数映射，`init_logger` 用它 patch |
| `_configure_vllm_root_logger()` | `logger.py:156` | 应用 `DEFAULT_LOGGING_CONFIG` 或 `VLLM_LOGGING_CONFIG_PATH` 指定的自定义 JSON；处理 `VLLM_CONFIGURE_LOGGING=false` 与 `VLLM_LOGGING_CONFIG_PATH` 冲突校验；历史兼容 `vllm.logging.NewLineFormatter` → `vllm.logging_utils.NewLineFormatter` |
| `init_logger(name)` | `logger.py:204` | 主 API：`logging.getLogger(name)` + patch 三个 `_once` 方法 |
| `suppress_logging(level)` | `logger.py:217` | contextmanager：临时 `logging.disable(level)` |
| `current_formatter_type(logger)` | `logger.py:225` | 遍历 logger 父链找 `vllm` handler 的 formatter，返回 `"color" \| "newline" \| None` |
| `_trace_calls(log_path, root_dir, frame, event, arg)` | `logger.py:251` | `sys.settrace` 回调：record 每个 `call`/`return` 到 log_path |
| `enable_trace_function_call(log_file_path, root_dir)` | `logger.py:294` | 启用 sys.settrace；warning "will slow code"；默认 `root_dir` = vllm 包根 |

模块下两行副作用代码：

```python
_configure_vllm_root_logger()           # import 时执行
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)  # transformers/hub 噪音
logger = init_logger(__name__)
```

## 为什么

- **import-time 配置**：vLLM 任意子模块 `from vllm.logger import init_logger` 即触发配置——所有 `init_logger(name)` 都拿到已配置 logger，无需 main 函数显式 setup。
- **`vllm` 根 logger 关 propagate**：避免 vLLM 日志冒泡到 root logger 与第三方库（uvicorn/fastapi）混流。
- **`VLLM_CONFIGURE_LOGGING` 开关**：让用户完全接管 logging 配置（如 Gunicorn/uvicorn worker 已统一配置），vLLM 不再覆盖。
- **`VLLM_LOGGING_CONFIG_PATH` JSON 自定义**：用户用 dictConfig JSON 完全替换 vLLM 默认；本模块仍保留"旧 class 路径 `vllm.logging.NewLineFormatter` → 新路径"兼容替换（PR #10134 后）。
- **`*_once` 方法 patch 而非子类化**：vLLM 早期试图子类化 `logging.Logger`，但 `intel_extension_for_pytorch.utils._logger` 等第三方库会改 root logger class，子类化冲突；改 patch 方法到 logger 实例避免全局类替换。
- **`*_once` 用 `lru_cache`**：消息 + args 作 key 在进程内只 print 一次；`stacklevel=3` 让日志记录点指向原始 caller 而非 `_print_*_once` 内部。
- **`LogScope`**：分布式训练场景下避免每个 rank 重复打印同一消息——`global` 仅 rank 0 全局打、`local` 仅 rank 0 节点内打、`process` 总打印。配合 `is_global_first_rank()`/`is_local_first_rank()`（[07-distributed](../07-distributed/README.md)）。
- **`enable_trace_function_call` 排障工具**：极慢但全——记录每个 call/return（含 caller 与 callee 文件:行号:函数名），定位 hang/crash 位置；显式 warning 拖慢代码，仅排障用。`root_dir` 默认 vllm 包根避免记录 stdlib 噪音。
- **httpx 降级**：transformers 用 httpx 访问 HuggingFace Hub，httpx INFO 级日志冗长；vLLM INFO 模式下自动把 httpx 设 WARNING。
- **`current_formatter_type` 反向探测**：让外部代码（如检测是否终端）无须 import 直接看当前用哪个 formatter。

## 怎么做

**基础用法**（任意 vllm 子模块）：

```python
from vllm.logger import init_logger
logger = init_logger(__name__)
logger.info("Engine started")
logger.debug_once("Cache compiled for shape %s", shape)  # 仅首次打
logger.info_once("model %s loaded", model_name, scope="global")  # 全局仅 rank0
```

**用户调日志级别**：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model>      # 全局 DEBUG
VLLM_LOGGING_LEVEL=INFO                          # 默认
VLLM_LOGGING_PREFIX="[vllm] "                    # 自定义前缀
VLLM_LOGGING_STREAM=ext://sys.stdout             # 输出流
VLLM_LOGGING_COLOR=0                             # 强制无色
NO_COLOR=1                                       # 兼容 no-color.org
VLLM_LOGGING_CONFIG_PATH=/path/to/log_config.json # 完全自定义 dictConfig
VLLM_CONFIGURE_LOGGING=0                         # vLLM 不配置，由用户接管
```

**自定义 dictConfig JSON** (`VLLM_LOGGING_CONFIG_PATH`)：

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "vllm": {
      "class": "vllm.logging_utils.NewLineFormatter",
      "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    }
  },
  "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm"}},
  "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}}
}
```

**`enable_trace_function_call`**：

```bash
VLLM_TRACE_FUNCTION=1 VLLM_TRACE_FUNCTION_FILE=/tmp/vllm_trace.log vllm serve <model>
```

（待核实 VLLM_TRACE_FUNCTION 环境变量名是否准确；`enable_trace_function_call` 在 vLLM 启动流程检测环境变量后调用，具体调用点 `vllm/engine/llm_engine.py` 或类似入口，待核实）。输出形如：

```
2026-07-19 12:34:56.123456 Call to forward in /opt/vllm/vllm/v1/worker/worker.py:200 from execute_model in /opt/vllm/vllm/v1/engine/core.py:150
2026-07-19 12:34:56.123789 Return from forward to execute_model ...
```

**临时屏蔽日志**（测试场景）：

```python
from vllm.logger import suppress_logging
import logging

with suppress_logging(logging.INFO):
    # 内部不输出 INFO 日志
    noisy_function()
```

## 与其它模块/系统配合

- **[logging_utils/formatter.md](logging_utils/formatter.md)**：`DEFAULT_LOGGING_CONFIG` 引用 `NewLineFormatter`/`ColoredFormatter`。
- **[logging_utils/access-log-filter.md](logging_utils/access-log-filter.md)**：uvicorn 端独立配置，与 vllm logger 互不干扰。
- **[v1/metrics/loggers.md](v1/metrics/loggers.md)**：`LoggingStatLogger.log()` 通过 vllm logger 输出。
- **[07-distributed](../07-distributed/README.md)**：`is_global_first_rank()`/`is_local_first_rank()` 是 LogScope 的实现。
- **[tracing/tracing.md](tracing/tracing.md)**：所有 tracing 模块用 `init_logger(__name__)` 自己的 logger；`is_tracing_available()` 用 logger warning。
- **`vllm/envs.py`**（[17-utils-cross-cutting](../17-utils-cross-cutting/README.md)）：`VLLM_LOGGING_*`、`VLLM_CONFIGURE_LOGGING`、`VLLM_TRACE_FUNCTION*`、`NO_COLOR` 等 env 都在 envs 定义。
- **第三方库（httpx/transformers）**：通过显式 `logging.getLogger("httpx").setLevel(WARNING)` 降级。
- **`logging.config.dictConfig`**：应用 `DEFAULT_LOGGING_CONFIG` 的标准库函数。

## 历史版本演进

- **v0.5（v0）**：`vllm/logger.py` 已存在；`init_logger`/`_configure_vllm_root_logger`；`DEFAULT_LOGGING_CONFIG`；`enable_trace_function_call`。
- **v0.6**：迁 `NewLineFormatter` 到 `vllm/logging_utils/`，保留旧 class 路径兼容替换；`ColoredFormatter` 入默认配置；`httpx` 降级（与 transformers 升级同步）。
- **v0.7（v1 落地）**：`_VllmLogger` 类型提示 + `_METHODS_TO_PATCH` 方法注入避免子类化冲突（早期子类化版本被 PR 改）；`LogScope` 引入 `global`/`local` 区分；`current_formatter_type` 工具。
- **v0.8**：`suppress_logging` contextmanager。
- **v0.9–main**：稳定 API；`_should_log_with_scope` 与 `is_global_first_rank`/`is_local_first_rank` 协同演进。具体版本归属（待核实）。

[← 返回可观测首页](README.md)

## 参见

- [logging_utils/logging-readme.md](logging_utils/logging-readme.md) — Formatter / Filter / 工具子目录。
- [v1/metrics/loggers.md](v1/metrics/loggers.md) — `LoggingStatLogger` 用 vllm logger。
- [tracing/tracing.md](tracing/tracing.md) — `is_tracing_available` 用 logger warning。
- [../10-config/observability-config.md](../10-config/observability-config.md) — 可观测性配置邻居。
