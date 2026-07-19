# 编译监控与闸门（monitor.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Monitor

源码：`vllm/compilation/monitor.py`（约 104 行）

## 是什么

`monitor.py` 提供编译流程的计时上下文与 CUDA Graph 捕获合法性闸门：

- `torch_compile_start_time`（`monitor.py:14`）：模块级全局，`backends.py` 读它上报 Dynamo 字节码耗时。
- `monitor_torch_compile(vllm_config, message, is_encoder)`（`monitor.py:17`）：上下文管理器，记录 `torch.compile` 起始时间、管理 depyf 调试 dump、正常退出时累加 `compilation_time`/`encoder_compilation_time` 并 log。
- `monitor_profiling_run()`（`monitor.py:62`）：上下文管理器，计时首次 profiling run，并断言期间无后端编译（`num_backend_compilations` 不增）。
- `cudagraph_capturing_enabled`（`monitor.py:87`）：全局布尔闸门。
- `validate_cudagraph_capturing_enabled()`（`monitor.py:90`）：捕获前调用，闸门关闭则抛 RuntimeError。
- `set_cudagraph_capturing_enabled(enabled)`（`monitor.py:102`）：开关闸门。

## 为什么

- **精确归因编译耗时**：`torch.compile` 与首次 profiling/warmup 常一起跑，`monitor_torch_compile` + `monitor_profiling_run` 把两段分开计时，分别上报，便于区分"Dynamo+Inductor 编译"与"首次推理"开销。`compilation_time` 最终进入 `VllmConfig` 汇报（[`配置-compilation`](../10-config/README.md)）。
- **depyf 调试 dump 生命周期**：`VLLM_COMPILE` 模式且 `compile_debug_dump_path()` 存在时，`depyf.prepare_debug(path)` 在编译期启用，把 Dynamo 转换后的字节码/中间图落盘供 `depyf` 可视化。异常时清理 depyf 不 log；正常时 log 编译耗时后清理。
- **profiling 期间禁止后端编译**：`monitor_profiling_run` 断言 `num_backend_compilations` 不增（`monitor.py:75`），保证"所有编译在 profiling 前完成"。若 profiling 触发编译，说明 compile_sizes/ranges 未覆盖实际运行 shape，是潜在 bug。
- **cudagraph 捕获闸门**：某些阶段（如 profiling、warmup、权重加载）不应触发 cudagraph 捕获，`set_cudagraph_capturing_enabled(False)` 关闸，`CUDAGraphWrapper`/`BreakableCUDAGraphWrapper` 在 `_capture` 起点调 `validate_cudagraph_capturing_enabled()`，违例即报错，防止误捕获污染 graph pool。
- **encoder 编译单独计时**：`is_encoder=True` 走 `encoder_compilation_time`，与 backbone 分开，因多模态 encoder 编译时机/耗时差异大。

## 怎么做

### monitor_torch_compile 流程

```python
# monitor.py:17 简化
global torch_compile_start_time
torch_compile_start_time = time.perf_counter()
if mode == VLLM_COMPILE and path:
    depyf_cm = depyf.prepare_debug(path.as_posix()); depyf_cm.__enter__()
try:
    yield
except Exception: raise
else:
    total = time.perf_counter() - torch_compile_start_time
    if mode == VLLM_COMPILE:
        if is_encoder: compilation_config.encoder_compilation_time += total
        else: compilation_config.compilation_time += total
        logger.info_once(message, total)
finally:
    if depyf_cm: depyf_cm.__exit__(None,None,None)  # 异常时也清理
```

### monitor_profiling_run 断言

```python
# monitor.py:62 简化
before = compilation_counter.num_backend_compilations
start = time.perf_counter()
yield
assert compilation_counter.num_backend_compilations == before, \
    "backend compilation occurred during the initial profiling run"
logger.info_once("Initial profiling/warmup run took %.2f s", elapsed)
```

### cudagraph 闸门用法

`CUDAGraphWrapper._capture` 起点调 `validate_cudagraph_capturing_enabled()`（`cuda_graph.py:277`）；Worker 在不该捕获的阶段（如 profile run、权重加载）调 `set_cudagraph_capturing_enabled(False)`，捕获窗口前调回 `True`。

## 与其它模块/系统配合

- [`decorators.py`](decorators.md)：首次编译与 AOT 路径用 `monitor_torch_compile` + `monitor_profiling_run` 包裹。
- [`backends.py`](backends.md)：`VllmBackend.__call__` 读 `torch_compile_start_time` 上报 "Dynamo bytecode transform time"（`backends.py:1147`）并 `instrument_manual` 上报 tracing。
- [`cuda_graph.py`](cuda-graph.md) / [`breakable-cudagraph.py`](breakable-cudagraph.md)：`_capture` 起点调闸门。
- [`caching.py`](caching.md)：`_try_load_aot_compiled_fn` 用 `monitor_torch_compile` 包 AOT 装载。
- [`配置-compilation`](../10-config/README.md)：`compilation_time`/`encoder_compilation_time` 字段；`compile_debug_dump_path()`。
- [`执行-cudagraph`](../02-execution/worker/cudagraph-capture.md)：Worker 控制 `set_cudagraph_capturing_enabled` 闸门窗口。
- [`可观测性`](../16-observability/README.md)：`instrument_manual("Dynamo bytecode transform", ...)` 上报 tracing。

## 历史版本演进

- **v0.7**：`monitor_torch_compile` + `torch_compile_start_time` 引入；`compilation_time` 字段接入 config。
- **v0.8**：`is_encoder` 分支 + `encoder_compilation_time`；depyf dump 生命周期管理。
- **v0.9**：`cudagraph_capturing_enabled` 闸门引入，配合 cudagraph 捕获时机规范化；`monitor_profiling_run` 断言强化。
- **v0.10 / main**：`instrument_manual` 上报 Dynamo 耗时；闸门在 warmup/重编译流程中精细化。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [counter.md](counter.md) — `num_backend_compilations` 断言依据。
- [cuda-graph.md](cuda-graph.md) / [breakable-cudagraph.md](breakable-cudagraph.md) — 捕获闸门调用点。
- [backends.md](backends.md) — Dynamo 字节码耗时上报。
- [../02-execution/worker/cudagraph-capture.md](../02-execution/worker/cudagraph-capture.md) — Worker 侧闸门窗口。
