[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/lazy

# Lazy（日志参数惰性求值包装）

> 源码：`vllm/logging_utils/lazy.py`（20 行）

## 是什么

`lazy.py` 提供 `lazy` 类，把零参 callable 包装成"在 `str()`/`repr()` 时才求值"的对象，让日志参数延迟到 logger 实际输出时才计算。

实现全文：

```python
from collections.abc import Callable
from typing import Any

class lazy:
    """Wrap a zero-argument callable evaluated only during log formatting."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    def __str__(self) -> str:
        return str(self._factory())

    def __repr__(self) -> str:
        return str(self)
```

## 为什么

- **避免 INFO 级别下昂贵构造**：`logger.debug("state=%s", heavy_repr())` 默认 Python 先调 `heavy_repr()` 再传给 logger；INFO 级别下 logger 不输出但 `heavy_repr()` 已执行——若 `heavy_repr` 涉及遍历大字典/序列化 tensor 等开销显著。`logger.debug("state=%s", lazy(lambda: heavy_repr()))` 仅在 DEBUG 实际输出时调 `heavy_repr()`。
- **`% (lazy(...))` 求值时机**：Python `logging` 的 `LogRecord.getMessage()` 在 `record.message = self.msg % self.args` 时把 args 填入 fmt；这是在 `Formatter.format()` 调用阶段，而 `Formatter.format()` 仅在 handler 实际处理 record 时调（即级别过滤后）——故 lazy `__str__` 此时才被触发。
- **`__slots__`**：避免 `__dict__` 创建，减少 GC 压力；lazy 对象可能在高频日志点（如 EngineCore 每步）构造。
- **`__repr__ = __str__`**：让 `%r` 也走 factory——某些 Formatter 用 `%r` 而非 `%s`（如调试打印）。
- **零参 callable**：约束 factory 不接收参数，强制用户在 lambda 里闭包捕获变量。若允许多参则用户可能误传昂贵默认参数（`partial(heavy_fn, expensive_default())`）——Zero-arg 让"延迟"语义清晰。
- **vs `logging.LazyLogger`/第三方 lazy log**：vLLM 选择自实现避免外部依赖；标准 `logging` 无原生延迟 args（PEP 555 提案未接，待核实）。
- **vs 把 `logger.debug` 改 `if logger.isEnabledFor(DEBUG):`**：后者每次都要写两行且易忘；`lazy()` 是函数式糖。

## 怎么做

**典型用法**：

```python
from vllm.logger import init_logger
from vllm.logging_utils import lazy

logger = init_logger(__name__)

# 不论 logger 级别，heavy_repr 仅在 DEBUG 输出时调
logger.debug("engine state: %s", lazy(lambda: expensive_repr(engine_state)))

# 闭包捕获变量
def step(scheduler_stats):
    logger.debug("step stats: %s",
                 lazy(lambda: format_stats(scheduler_stats)))
```

**注意事项**：
- factory 抛异常时 logger 会吞下还是冒泡？（待核实；通常 `Formatter.format` 用 try/except 把异常 record 到 `record.exc_info`，故建议 factory 内 try/except 兜底）
- 多个 `%s` 多个 lazy 时分别求值，无共享状态——若 factory 间有依赖（如先算 token 数再用）需在 factory 内统一算。
- `lazy(lambda: x)` 与 `lazy(lambda: str(x))` 略不同——后者强制转 str；通常 `%s` 自动调 `__str__`，无需显式 str。
- 工厂返回 None：`str(None) = "None"`，符合预期。

## 与其它模块/系统配合

- **[../logger.md](../logger.md)**：vLLM 配置的 logger handler 在 format 阶段触发 lazy 求值。
- **[log-time.md](log-time.md)**：`@logtime(logger, lazy(lambda: f"...{ctx}..."))` 让 msg 也 lazy（待核实 logtime 是否支持）。
- **[torch-tensor.md](torch-tensor.md)**：`lazy(lambda: tensors_str_no_data(logits))` 是排障组合拳——DEBUG 时才出 tensor shape。
- **`logging.LogRecord.getMessage`**：触发 `% args` 求值；标准库机制。
- **Python GIL 与线程安全**：factory 在 logger handler 线程（或同线程视 handler 配置）调，需注意 factory 修改共享状态（罕见）。

## 历史版本演进

- **v0.5/v0.6（v0）**：无 `lazy.py`（待核实，可能 PR 较晚）。
- **v0.7（v1 落地）**：引入 `vllm/logging_utils/lazy.py`；20 行实现稳定。
- **v0.8–main**：未变。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [logging-readme.md](logging-readme.md) — 子目录总览。
- [log-time.md](log-time.md) — 配合用 `@logtime`。
- [torch-tensor.md](torch-tensor.md) — 配合用 tensor 截断。
