[← Wiki 首页](../../README.md) > [可观测](../README.md) > logging_utils/dump-input

# Dump Input（引擎异常时匿名列印调度输入）

> 源码：`vllm/logging_utils/dump_input.py`（83 行）

## 是什么

`dump_input.py` 提供 EngineCore 抛异常时把"刚刚送进执行层的输入"以**匿名形式**打日志的能力——`SchedulerOutput` / `SchedulerStats` 含 token id 等敏感信息，直接 dump 会泄漏用户数据；本模块用 `prepare_object_to_dump` 把 tensor 仅出 shape/device/dtype、字符串加单引号转义、enum 转 repr，再 `logger.error` 出。

| 函数 | 行号 | 角色 |
|---|---|---|
| `prepare_object_to_dump(obj)` | `dump_input.py:19` | 递归把任意对象转为匿名字符串：str→`'...'`，dict/list/set/tuple→容器 repr，enum→`repr(obj)`，tensor→`Tensor(shape=..., device=..., dtype=...)`，对象有 `anon_repr()` 方法优先用，否则用 `__dict__` map，最后回退到 `json.dumps` 或 `repr` |
| `dump_engine_exception(config, scheduler_output, scheduler_stats)` | `dump_input.py:56` | 入口：`with contextlib.suppress(Exception)` 包裹 `_dump_engine_exception`（保证 log 失败也不掩盖原异常）|
| `_dump_engine_exception(...)` | `dump_input.py:67` | 调用 `logger.error("Dumping input data for V1 LLM engine (v%s) with config: %s", VLLM_VERSION, config)` + `logger.error("Dumping scheduler output for model execution: %s", prepare_object_to_dump(scheduler_output))` + 可选 `scheduler_stats` 同样 dump |

`prepare_object_to_dump` 分派顺序：

```mermaid
flowchart TD
    IN["obj"]
    IN --> S{"isinstance str?"}
    S -->|yes| STR["f\"'{obj}'\""]
    S -->|no| D{"isinstance dict?"}
    D -->|yes| DCT["{k: prepare(v), ...}"]
    D -->|no| L{"isinstance list/set/tuple?"}
    L -->|yes| LST["[prepare(v), ...]"]
    L -->|no| E{"isinstance Enum?"}
    E -->|yes| ENM["repr(obj)"]
    E -->|no| T{"isinstance torch.Tensor?"}
    T -->|yes| TNS["f\"Tensor(shape={obj.shape}, device={obj.device}, dtype={obj.dtype})\""]
    T -->|no| AR{"hasattr anon_repr?"}
    AR -->|yes| ANON["obj.anon_repr()"]
    AR -->|no| DD{"hasattr __dict__?"}
    DD -->|yes| DIC["TypeName(a=v, b=v, ...)"]
    DD -->|no| JSON["try json.dumps else repr"]

    style TNS fill:#fde,stroke:#c30
    style ANON fill:#dfd,stroke:#393
```

## 为什么

- **崩溃可复现**：EngineCore 异常往往依赖具体 SchedulerOutput（哪个请求、什么 token 数、哪批 LoRA）。无此 dump 用户只能贴 stack trace，开发者难复现；本模块把异常时的输入结构（去敏感）打日志，便于后续从日志构造 junit 测试。
- **隐私第一**：tensor 不出数据（只出 shape/device/dtype）；字符串加单引号但保留内容（小字符串如 model_name 有用，长 prompt 可能略冗但通常 SchedulerOutput 不含完整 prompt 待核实）。`anon_repr()` 钩子让数据类自定义匿名 repr（如 Request 类只出 `req_id` + token 数）。
- **`contextlib.suppress(Exception)` 包裹**：logging 本身的异常（如 __dict__ 引用循环）不应掩盖 EngineCore 原异常——这是排障工具的"自我防破坏"。
- **`VLLM_VERSION` 入日志**：异常 dump 必须带版本号，让 issue 报告可对齐 commit。版本号取 `vllm/version.py::__version__`。
- **`config` 也 dump**：`config` 对象（`VllmConfig`）的 `__dict__` 含所有子配置，调 `prepare_object_to_dump(config)` 给出全配置——比让用户贴 CLI args 完整。
- **`SchedulerStats` 可选**：`dump_engine_exception(config, scheduler_output, scheduler_stats=None)`——若执行前未算 stats 则 None 跳过。
- **`tuple`/`set` 同 list**：Python logging 不区分容器类型，统一用 `[...]` repr 让 JSON-like 显示更统一。

## 怎么做

EngineCore 在调度/执行 catch 块调用（具体调用点在 `vllm/v1/engine/core.py` 或 `vllm/v1/worker/worker.py`，待核实）：

```python
from vllm.logging_utils import dump_engine_exception

try:
    output = self.execute_model(scheduler_output, ...)
except Exception:
    dump_engine_exception(self.vllm_config, scheduler_output, scheduler_stats)
    raise
```

输出形如：

```
ERROR Dumping input data for V1 LLM engine (v0.x.y) with config: VllmConfig(model_config=...)
ERROR Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[ScheduledRequest(req_id='abc', num_tokens=128, ...)], scheduled_cached_reqs=[...], num_scheduled_tokens={...})
ERROR Dumping scheduler stats: SchedulerStats(num_running_reqs=8, num_waiting_reqs=3, ...)
```

**自定义匿名 repr**（开发者想让自家数据类支持 dump）：

```python
class MyDataClass:
    def anon_repr(self) -> str:
        # 只暴露对排障有用、无敏感的部分
        return f"MyDataClass(items={len(self.items)}, total_tokens={self.total_tokens})"
```

`prepare_object_to_dump` 优先调 `anon_repr()`，跳过默认 `__dict__` 序列化。

**禁用**：用户若不希望任何额外日志（如性能压测），可在 fork 里移除调用——本模块无开关 CLI 字段（待核实）。

## 与其它模块/系统配合

- **[../v1/metrics/stats.md](../v1/metrics/stats.md)**：消费 `SchedulerStats` dataclass；其 `__repr__`（`stats.py:340`）已对，让 prepare_object_to_dump 走 `__dict__` 分支输出完整字段。
- **[01-engine-core/engine-core-process.md](../../01-engine-core/engine-core-process.md) / [02-execution/README.md](../../02-execution/README.md)**：调用点，EngineCore/Worker 异常路径。
- **`vllm/version.py`**：`__version__` 注入日志。
- **[../logger.md](../logger.md)**：通过 vllm logger 输出 `logger.error`；本身经 `ColoredFormatter`/`NewLineFormatter` 显示。
- **`torch.Tensor`**：依赖 `obj.shape/device/dtype` 属性——非 tensor 对象在不匹配前面分支时不走此路径。
- **`vllm/v1/core/sched/output.py::SchedulerOutput`**：被 dump 的主要对象；其字段含 token id 等可能敏感——`anon_repr()` 钩子让 SchedulerOutput 子结构自定义（待核实是否已实现）。

## 历史版本演进

- **v0.5/v0.6（v0）**：无此模块；v0 EngineCore 异常仅打 stack trace。
- **v0.7（v1 落地）**：引入 `vllm/logging_utils/dump_input.py`；`prepare_object_to_dump` + `dump_engine_exception` 配合 v1 `SchedulerOutput`/`SchedulerStats`。
- **v0.8**：`anon_repr()` 钩子让数据类自定义匿名 repr；`VLLM_VERSION` 入日志。
- **v0.9**：`dump_engine_exception` 加 `scheduler_stats` 参数（v0.7 时可能仅 scheduler_output，待核实）。
- **v0.10–main**：稳定 API。具体版本归属（待核实）。

[← 返回可观测首页](../README.md)

## 参见

- [logging-readme.md](logging-readme.md) — 子目录总览。
- [../v1/metrics/stats.md](../v1/metrics/stats.md) — 被 dump 的 stats dataclass。
- [../../01-engine-core/engine-core-process.md](../../01-engine-core/engine-core-process.md) — EngineCore 异常路径调用点。
