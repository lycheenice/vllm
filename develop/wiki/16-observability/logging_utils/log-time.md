[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/log-time

# Log Time（函数耗时装饰器）

> 源码：`vllm/logging_utils/log_time.py`（34 行）

## 是什么

`log_time.py` 提供一个极简装饰器 `logtime(logger, msg=None)`，把目标函数用 `time.perf_counter()` 包裹，结束后 `logger.debug("%s: Elapsed time %.7f secs", prefix, elapsed)`。

实现全文：

```python
import functools
import time

def logtime(logger, msg=None):
    """Logs the execution time of the decorated function.
    Always place it beneath other decorators.
    """
    def _inner(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            prefix = (
                f"Function '{func.__module__}.{func.__qualname__}'"
                if msg is None
                else msg
            )
            logger.debug("%s: Elapsed time %.7f secs", prefix, elapsed)
            return result
        return _wrapper
    return _inner
```

## 为什么

- **简化计时**：vLLM 内部某些非热路径（初始化、编译、权重加载）需要可观测耗时；手写 `start=time.perf_counter() ... logger.debug(... time.time()-start)` 模板冗余，`@logtime` 一行解决。
- **DEBUG 级别才输出**：`logger.debug(...)` 让普通 INFO 级别运行零噪音；用户排障时把 `VLLM_LOGGING_LEVEL=DEBUG` 即可看到各阶段耗时。
- **`functools.wraps` 保元数据**：被装饰函数的 `__name__`/`__doc__`/`__module__` 不变，对自我反射的工具（如 FastAPI 路由推导）友好。
- **`time.perf_counter()` 而非 `time.time()`**：`perf_counter` 单调、最高精度，适合测短时区间；不受系统时钟调整影响。
- **`msg` 默认拼 `func.__module__.__qualname__`**：免用户手写函数名；默认消息形如 `Function 'vllm.v1.engine.core.EngineCore.start': Elapsed time 0.1234567 secs`。
- **"Always place it beneath other decorators"**：`@functools.wraps` 仅一层 wrap——若 stacked decorator 中 `logtime` 在外，会测所有内层装饰器开销，包括 `@staticmethod`/`@property` 等；放最内让用户看到 "raw 函数耗时"。这条警示是 docstring 提示开发者顺序。

## 怎么做

**用法**：

```python
from vllm.logger import init_logger
from vllm.logging_utils import logtime

logger = init_logger(__name__)

@logtime(logger, "compile_model")
def compile_model(...):
    ...

@logtime(logger)  # 用默认 prefix
def load_weights(...):
    ...
```

**输出**（`VLLM_LOGGING_LEVEL=DEBUG` 时）：

```
DEBUG compile_model: Elapsed time 12.3456789 secs
DEBUG Function 'vllm.model_executor.model_loader.loader.load_weights': Elapsed time 5.6789012 secs
```

**配合 `lazy`**（[lazy.md](lazy.md)）做更复杂日志：

```python
@logtime(logger, lazy(lambda: f"compile_model(shape={ctx.shape})"))
```

（待核实 `logtime` 是否支持 lazy msg——目前 `_inner` 直接用 `msg`，若想 lazy 需调用方自己包装，但 functools.wraps 不改 msg 形参类型；此用法为非官方扩展，待核实。）

## 与其它模块/系统配合

- **[../logger.md](../logger.md)**：消费者 `init_logger(name)` 返回的 logger 实例；与 vllm logger 体系一脉。
- **[lazy.md](lazy.md)**：常配合使用（msg 用 lazy）。
- **`functools.wraps` / `time.perf_counter`**：Python 标准库。
- 非热路径模块（编译、权重加载、scheduler 初始化）：典型用 `@logtime`。具体 vLLM 内调用点（待核实，按需 grep `@logtime`）。

## 历史版本演进

- **v0.5/v0.6（v0）**：`vllm/logging_utils/log_time.py` 已存在；早期版本可能仅用 `time.time()`。
- **v0.7（v1 落地）**：迁 `time.perf_counter`；docstring 警示"放在其他 decorator 之下"。
- **v0.8–main**：稳定 API，34 行未变。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [logging-readme.md](logging-readme.md) — 子目录总览。
- [lazy.md](lazy.md) — 配合 `@logtime` 用 lazy msg。
- [../logger.md](../logger.md) — logger 入口。
