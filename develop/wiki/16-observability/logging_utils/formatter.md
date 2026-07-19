[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/formatter

# Formatter（NewLineFormatter / ColoredFormatter）

> 源码：`vllm/logging_utils/formatter.py`（125 行）

## 是什么

`formatter.py` 定义 vLLM 的两个 logging `Formatter` 子类，由 [`vllm/logger.py`](../logger.md) 在 `DEFAULT_LOGGING_CONFIG` 中作为 `"vllm"` / `"vllm_color"` formatter 类使用。

| 类 | 行号 | 角色 |
|---|---|---|
| `NewLineFormatter` | `formatter.py:10` | `logging.Formatter` 子类：多行 message 每行加 prefix 对齐；DEBUG 模式下用相对路径 `shrink_path` 替换 `record.fileinfo` |
| `ColoredFormatter` | `formatter.py:78` | `NewLineFormatter` 子类：注入 ANSI 色码到 `%(asctime)s`/`[%(fileinfo)s:%(lineno)d]`（灰）与 `levelname`（按级别变色） |

`_FORMAT`（在 [`vllm/logger.py:22`](../logger.md)）：

```python
_FORMAT = f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
```

`NewLineFormatter.format(record)` 流程：

1. `self.use_relpath = envs.VLLM_LOGGING_LEVEL == "DEBUG"`——仅 DEBUG 用相对路径（开销略大）。
2. 若 use_relpath：`abs_path = record.pathname` → `relpath = abs_path.relative_to(self.root_dir)`（root_dir 是 `vllm/logging_utils/` 的上两级即 vllm 根）→ `record.fileinfo = shrink_path(relpath)`。
3. 否则：`record.fileinfo = record.filename`（仅文件名）。
4. `super().format(record)` 得到完整 msg。
5. 若 `record.message != ""`：把 `\n` 替换为 `\r\n` + parts[0]（parts[0] 是 msg 之前的前缀）——让多行 message 每行都有日志前缀。
6. 返回 msg。

`shrink_path(relpath)`（`formatter.py:21`）：

- 去掉 leading `vllm/`。
- 若开头 `v1/`：保留前两级 + `...` + 后两级；例 `vllm/v1/metrics/loggers.py` → `v1/metrics/.../loggers.py`（实际因只有 3 级，结果 `v1/metrics/loggers.py`）。
- 否则保留首级 + 后两级；例 `vllm/model_executor/layers/quantization/utils/fp8_utils.py` → `model_executor/.../quantization/utils/fp8_utils.py`。
- 若不足 4 级直接返回原路径。

`ColoredFormatter` 流程：

1. `__init__`：把 fmt 串中 `%(asctime)s` 替换为 `{GREY}%(asctime)s{RESET}`、`[%(fileinfo)s:%(lineno)d]` 同理。GREY=`\033[90m`，RESET=`\033[0m`。
2. `format(record)`：保存 `orig_levelname`，按 `record.levelname` 查 `COLORS` 取色码覆写 `record.levelname` 为 `\033[Xm{LEVELNAME}\033[0m`，调 `super().format(record)`，最后恢复 `orig_levelname` 防 record 复用污染。

颜色映射：

| Level | ANSI | 颜色 |
|---|---|---|
| DEBUG | `\033[37m` | White |
| INFO | `\033[32m` | Green |
| WARNING | `\033[33m` | Yellow |
| ERROR | `\033[31m` | Red |
| CRITICAL | `\033[35m` | Magenta |
| (timestamp/fileinfo) | `\033[90m` | Grey |

## 为什么

- **多行 message 对齐**：stats logger 一次打多行（如 `Engine 000: Avg prompt throughput: ..., Avg generation throughput: ..., Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%, Prefix cache hit rate: Y%`），若出现 `\n` 默认每行无前缀，stdout 难读。`NewLineFormatter` 让每行对齐前缀——`vllm/.../loggers.py:224` 的 `log_parts` 拼接模式靠此。
- **相对路径只 DEBUG 用**：`Path.resolve().relative_to(self.root_dir)` 调用有 I/O 开销，INFO 级别用裸 `record.filename` 即可；DEBUG 模式下源码定位价值更高。
- **`shrink_path` 折叠中间段**：长路径在窄终端难看，按"首段 + 后两段"折叠让用户看一级 package + 文件名足够定位。
- **`ColoredFormatter` 动态 levelname**：asctime/fileinfo 可在 fmt 串里静态注入 grey（不变）；levelname 按 record 动态变色（必须 override `format()` 而非 fmt）——故采用"父类做 static inject + 子类 override 动态 part"两层设计。
- **尊重 NO_COLOR**：`_use_color`（见 [logger.md](../logger.md)）检查 `NO_COLOR`/`VLLM_LOGGING_COLOR`/`isatty` —— CI/CD/重定向输出自动无色，发展上兼容 [no-color.org](https://no-color.org)。
- **`record` 恢复**：logging 内部可能复用 `LogRecord`（pool），不恢复 `orig_levelname` 会让后续 logger 输出残留色码。
- **`\r\n` 而非 `\n`**：多行回车用 `\r\n` 是为某些日志聚合工具按 `\r\n` 切分重组行（具体设计动因待核实，可能与早期 log ingester 兼容）。

## 怎么做

**默认配置**：vLLM 启动时 `_configure_vllm_root_logger()` 已把 `ColoredFormatter`（when `_use_color()`）或 `NewLineFormatter` 接到 `vllm` logger handler——用户无需手动配置。

**自定义 fmt**：

```python
# 通过环境变量调整 logging prefix
VLLM_LOGGING_PREFIX="[vllm] "  # 默认空串
# 或通过 VLLM_LOGGING_CONFIG_PATH 自定义 logging config JSON
```

`VLLM_LOGGING_CONFIG_PATH` 指定的 JSON 里 `formatters.vllm.class` 可设 `vllm.logging_utils.NewLineFormatter` 或 `ColoredFormatter`，`format` 字段仍需包含 `%(fileinfo)s`（被 Formatter 写入 `record.fileinfo`）。

**程序化取 formatter 类型**（[`logger.py:225`](../logger.md) `current_formatter_type(logger)`）：

```python
from vllm.logger import current_formatter_type
kind = current_formatter_type(logger)  # "color" | "newline" | None
```

遍历 logger 父链找名为 `vllm` 的 handler 看 formatter 类型——常被 `_VllmLogger` 内部判断逻辑用（如检测是否终端）。

## 与其它模块/系统配合

- **[../logger.md](../logger.md)**：`_FORMAT` 与 `DEFAULT_LOGGING_CONFIG` 在 logger.py 定义，本文件提供 Formatter 实现；`current_formatter_type()` 反向探测。
- **[../v1/metrics/loggers.md](../v1/metrics/loggers.md)**：`LoggingStatLogger.log()` 的多行格式依赖 `NewLineFormatter` 对齐显示。
- **`vllm/envs.py`**：`VLLM_LOGGING_LEVEL`/`VLLM_LOGGING_PREFIX`/`VLLM_LOGGING_COLOR`/`VLLM_LOGGING_STREAM`/`NO_COLOR`/`VLLM_LOGGING_CONFIG_PATH` 决定 Formatter 行为。
- **`logging.config.dictConfig`**：`DEFAULT_LOGGING_CONFIG` 由 `dictConfig()` 应用，class 路径字符串解析为 Python 类。

## 历史版本演进

- **v0.5（v0）**：`vllm/logging.py::NewLineFormatter` 仅多行对齐，无 color；`shrink_path` 无；class 路径 `vllm.logging.NewLineFormatter`。
- **v0.6**：迁 `vllm/logging_utils/`；加 `ColoredFormatter`；历史兼容 `vllm.logging.NewLineFormatter` → `vllm.logging_utils.NewLineFormatter`（`logger.py:197`）。
- **v0.7（v1 落地）**：`shrink_path` + DEBUG 用相对路径；`record.fileinfo` 字段引入。
- **v0.8–v0.9**：颜色细节调整（grey 注入 fmt 而非 record；只 override levelname）。
- **v0.10–main**：稳定 API。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [../logger.md](../logger.md) — `_FORMAT` 与配置应用。
- [logging-readme.md](logging-readme.md) — 子目录总览。
