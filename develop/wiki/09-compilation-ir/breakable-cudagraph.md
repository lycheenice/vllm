# BreakableCUDAGraphWrapper

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Breakable CUDAGraph

源码：`vllm/compilation/breakable_cudagraph.py`（约 424 行）

## 是什么

`BreakableCUDAGraphWrapper`（`breakable_cudagraph.py:246`）是 `CUDAGraphWrapper` 的替代实现，由 `VLLM_USE_BREAKABLE_CUDAGRAPH` 开关启用。它不依赖 `VllmBackend` 的 FX 预切分，而是在**运行期**用 stream-capture 驱动整段 forward，遇到打了 `@eager_break_during_capture` 标记的自定义 op（注意力/KV-cache）就"结束当前图段 → eager 跑该 op → 开新图段"，最终产物是一串零参 callable（图段 `replay` + eager fn），按序回放。

关键成员：

- `eager_break_during_capture(fn)`（`breakable_cudagraph.py:59`）：装饰器，把自定义 op 变成"捕获期断点"。要求 op 写入 caller 提供的 in-place 输出 buffer，且**必须是最外层装饰器**（`@maybe_transfer_kv_layer` 等 host 副作用装饰器要包在它里面）。
- `BreakableCUDAGraphCapture`（`breakable_cudagraph.py:125`）：thread-local 的 stream-capture 上下文。`_begin_segment`/`_end_segment`/`add_eager` 管理图段序列，`replay()` 按序执行 `segments`。
- `BreakableCUDAGraphWrapper`（`breakable_cudagraph.py:246`）：与 `CUDAGraphWrapper` 同 dispatch 契约，按 `BatchDescriptor` 捕获/回放。
- `_BreakableEntry`：per-descriptor 的 capture + output + input_addresses。

## 为什么

- **摆脱 FX 预切分**：`VllmBackend` 的 piecewise 需要把图在编译期按 `splitting_ops` 切开，工程复杂且与 Inductor 自分区/动态形状耦合。breakable 改为运行期 stream capture，无需编译期切图，简化编译路径（受 sglang#19102 启发）。
- **prefill/decode 同图**：breakable 的捕获产物对 prefill 与 decode 一致（都按 op 边界断），`BatchDescriptor` 已编码 prefill/decode 区分，故 wrapper 不按 `runtime_mode` 分派（`breakable_cudagraph.py:318`）。
- **PD 分解的 host 副作用隔离**：`@eager_break_during_capture` 必须在 `@maybe_transfer_kv_layer` 外层，确保 `wait_for_layer_load`/`save_kv_layer` 这类 host 副作用落在 eager 段而非被录进图（否则回放挂死，`breakable_cudagraph.py:69`）。
- **静态 buffer 复用**：eager 段必须写入与捕获期相同的 static output buffer，保证下游图段读到相同 `data_ptr`。装饰器对 args 做 `weak_ref_tensor`，让 replay lambda 的强引用跨 descriptor 维持 cudagraph pool 槽存活。
- **gc 节流**：`_capture` 仅在进入 descriptor 时 `gc.collect()+empty_cache()` 一次，而非每段 `begin/end` 都做（否则多层断点会像旧 piecewise 那样捕获变慢）。
- **`_all_instances` + `clear_all_graphs`**：与 `CUDAGraphWrapper` 对齐，支持重编译清空。

## 怎么做

### eager_break_during_capture 行为

```python
# breakable_cudagraph.py:93 简化
def wrapper(*args, **kwargs):
    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        return fn(*args, **kwargs)               # 非捕获期：正常跑
    if forward_context available and mode == FULL:
        return fn(*args, **kwargs)               # FULL 模式不打断
    weak_args = tuple(weak_ref_tensor(a) if isinstance(a,Tensor) else a for a in args)
    weak_kwargs = {k: weak_ref_tensor(v) if Tensor else v for k,v in kwargs}
    return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))  # 断点
```

### BreakableCUDAGraphCapture.add_eager

```python
# breakable_cudagraph.py:195 简化
def add_eager(self, fn):
    self._end_segment()        # capture_end + append prev graph.replay
    result = fn()              # eager 跑 op（写 static buffer）
    self.segments.append(fn)   # 记录供 replay
    self._num_eager_breaks += 1
    self._begin_segment()      # 开新图段
    return result
```

`__enter__` 开首段，`__exit__` 收尾段；全程 thread-local 单例，不支持嵌套。

### Wrapper 捕获/回放

`_capture`（`breakable_cudagraph.py:353`）：一次性 `gc.collect()+empty_cache()` → `set_graph_pool_id` → `get_offloader().sync_prev_onload()` → `with BreakableCUDAGraphCapture(pool): output = self.runnable(*args); get_offloader().join_after_forward(); output = weak_ref_tensors(output)`。返回 captured output（已弱引用）。

`_replay`：DEBUG 校验 input_addresses → `sync_prev_onload` → `entry.capture.replay()`（逐段执行）→ 返回 `entry.output`。

### replay 顺序

`BreakableCUDAGraphCapture.replay()`（`breakable_cudagraph.py:212`）按 `segments` 顺序：每段是 `CUDAGraph.replay`（bound method）或 eager `fn`，依次 `r()`。图段在 pool 内、eager 段在 capture stream 上，共同复现捕获期数据流。

## 与其它模块/系统配合

- [`cuda_graph.py`](cuda-graph.md)：同 dispatch 契约、同 `_all_instances`/`clear_all_graphs` 模式，平台二选一。
- [`backends.py`](backends.md)：`wrap_with_cudagraph_if_needed` 通过 `get_static_graph_wrapper_cls()` 可能解析到本类（待核实：当前 `backends.py` 默认仍指 `CUDAGraphWrapper`，breakable 经平台类解析或 env 切换）。
- [`monitor.py`](monitor.md)：`validate_cudagraph_capturing_enabled` 同样在 `_capture` 起点检查。
- [`模型执行-custom_op`](../03-model-execution/layers/custom-op.md)：注意力/KV-cache 的自定义 op 用 `@eager_break_during_capture` 标记；`@maybe_transfer_kv_layer`（PD 分解）须在其内层。
- [`执行-cudagraph`](../02-execution/worker/cudagraph-capture.md)：Worker 仍提供 `BatchDescriptor` 与 `cudagraph_runtime_mode`，但 breakable 不严格按 mode 分派。
- [`offloader`](../03-model-execution/README.md)：`sync_prev_onload`/`join_after_forward` 与 copy stream 协同。
- [`分布式`](../07-distributed/README.md)：PD 分解 KV 迁移与 breakable 的 eager 段语义匹配。

## 历史版本演进

- **v0.9（引入）**：`BreakableCUDAGraphCapture` + `BreakableCUDAGraphWrapper` + `eager_break_during_capture` 落地，受 sglang#19102 启发，`VLLM_USE_BREAKABLE_CUDAGRAPH` 默认关。定位为"无需 FX 预切分"的运行期捕获方案。
- **v0.10**：`_collect_tensor_addresses` 同时覆盖 positional 与 kwargs（vLLM 模型常用 kwargs 调用）；gc 节流策略定稿（每 descriptor 一次，非每段）。
- **v0.11 / v0.12 / main**：与 PD 分解 `@maybe_transfer_kv_layer` 装饰器顺序约定固化；逐步作为 piecewise 的替代路径推广（待核实是否默认）。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [cuda-graph.md](cuda-graph.md) — 默认 wrapper，对比理解 breakable。
- [backends.md](backends.md) — piecewise 切分与 breakable 的取舍。
- [base-static-graph.md](base-static-graph.md) — wrapper 的 Protocol 接口。
- [monitor.md](monitor.md) — 捕获合法性闸门。
- [../03-model-execution/layers/custom-op.md](../03-model-execution/layers/custom-op.md) — `@eager_break_during_capture` 标注的 op。
