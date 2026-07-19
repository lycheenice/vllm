# CUDAGraphWrapper

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > CUDAGraph

源码：`vllm/compilation/cuda_graph.py`（约 361 行）

## 是什么

`CUDAGraphWrapper`（`cuda_graph.py:145`）是 vLLM CUDA Graph 捕获/回放的默认实现，由 [`平台`](../08-platforms/README.md) 的 `get_static_graph_wrapper_cls()` 解析（CUDA 平台返回本类）。它把一段 `runnable`（通常是 `PiecewiseBackend` 或整图 callable）包成"按 `BatchDescriptor` 缓存 `torch.cuda.CUDAGraph`、首次捕获后续回放"的包装器，支持 `FULL` 与 `PIECEWISE` 两种 runtime mode。

关键成员：

- `CUDAGraphOptions`（`cuda_graph.py:138`）：`debug_log_enable` / `gc_disable` / `weak_ref_output`，piecewise 首段开 debug、非首段禁 gc、末段 weak_ref 输出省显存。
- `CUDAGraphEntry`（`cuda_graph.py:127`）：`batch_descriptor` → `cudagraph` / `output` / `input_addresses`。
- `CUDAGraphStat` / `CUDAGraphLogging`（`cuda_graph.py:32-124`）：聚合 cudagraph 命中统计并打印 Markdown 表。
- `_all_instances: WeakSet`（`cuda_graph.py:170`）：所有 wrapper 实例，`clear_all_graphs()` 批量清空（用于重编译/重捕获）。
- `__getattr__`：转发到 `runnable` 的属性，使 wrapper 透明。

## 为什么

- **消除 kernel launch 开销**： Decode 阶段每步上万个 kernel，逐个 launch 在 small batch 下成为瓶颈。CUDA Graph 把整段 forward 录制成单图 replay，launch 开销摊薄到一次。
- **PIECEWISE 模式适配动态注意力**：含动态形状的注意力 op 不宜整图捕获，按注意力边界切分后逐段捕获，每段内部静态化。`runtime_mode=PIECEWISE` 的 wrapper 只在 forward_context 的 `cudagraph_runtime_mode` 同为 PIECEWISE 时捕获/回放，FULL 段交给 FULL wrapper。
- **per-descriptor 多图共存**：不同 batch 形状（prefill vs decode、不同 token 数）各捕获一张图，以 `BatchDescriptor` 为 key；运行期按当前 descriptor 选图回放，未命中则首次捕获。
- **弱引用输出省显存**：piecewise 末段输出无人再用于其它图捕获，`weak_ref_tensors(output)` 立即释放强引用，让 cudagraph pool 复用该内存给下一个 descriptor 的捕获。
- **输入地址一致性校验**：`VLLM_LOGGING_LEVEL=DEBUG` 时记录捕获期 input `data_ptr`，回放时校验，避免静态输入缓冲未拷贝导致的静默错位。
- **平台可替换**：本类是 CUDA 实现；ROCm/XPU/TPU 可通过 `get_static_graph_wrapper_cls()` 返回自己的类（如 [`breakable_cudagraph.py`](breakable-cudagraph.md)）。

## 怎么做

### __call__ 分派契约

```python
# cuda_graph.py:233 简化
def __call__(self, *args, **kwargs):
    if not is_forward_context_available():
        return self.runnable(*args, **kwargs)            # 非推理路径（如 vision encoder）直跑
    fc = get_forward_context()
    if fc.cudagraph_runtime_mode == NONE or fc.cudagraph_runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)            # mode 不匹配直跑
    entry = self.concrete_cudagraph_entries.setdefault(fc.batch_descriptor, ...)
    if entry.cudagraph is None:
        return self._capture(entry, args, kwargs)        # 首次捕获
    return self._replay(entry, args, kwargs)
```

> 嵌套时（FULL wrapper 包着 PIECEWISE 段），靠 `runtime_mode` 匹配把 dispatch 路由到正确 wrapper。

### 捕获流程

```python
# cuda_graph.py:265-344 简化
validate_cudagraph_capturing_enabled()                  # 闸门：monitor.py
input_addresses = [x.data_ptr() for x in args if Tensor]
cudagraph = torch.cuda.CUDAGraph()
with ExitStack() as stack:
    if self.cudagraph_options.gc_disable:
        stack.enter_context(patch("gc.collect", noop))  # 非首段禁 gc 提速
        stack.enter_context(patch("torch.accelerator.empty_cache", noop))
    set_graph_pool_id(self.graph_pool or current_platform.graph_pool_handle())
    get_offloader().sync_prev_onload()                  # 等预取完成
    with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=current_stream()):
        output = self.runnable(*args, **kwargs)
        get_offloader().join_after_forward()            # 合并 copy stream
        if self.cudagraph_options.weak_ref_output:
            output = weak_ref_tensors(output)
entry.output = weak_ref_tensors(output)
entry.cudagraph = cudagraph
compilation_counter.num_cudagraph_captured += 1
return output                                           # 返回强引用供 pytorch 内存管理
```

### 回放流程

- DEBUG：校验 `new_input_addresses == entry.input_addresses`。
- `get_offloader().sync_prev_onload()` 等外部依赖。
- `entry.cudagraph.replay()`。
- 返回 `entry.output`（弱引用，调用方持有张量即保持 pool 槽存活）。

### 全局 graph pool

`current_platform.get_global_graph_pool()` 返回跨 wrapper 共享的 pool（`cuda_graph.py:200`），使多段 piecewise 图共用内存池、避免碎片化。多 stream 时该共享不安全（待核实 TODO `cuda_graph.py:197`）。

### 统计与日志

`CUDAGraphLogging.observe(CUDAGraphStat(num_unpadded_tokens, num_padded_tokens, num_paddings, runtime_mode))` 累计命中频次，`log()` 输出按频次降序的 Markdown 表，便于查看实际命中分布与 padding 浪费。

## 与其它模块/系统配合

- [`backends.py`](backends.md)：`wrap_with_cudagraph_if_needed`（`backends.py:628`）按 `has_piecewise_cudagraphs()` 与 `use_inductor_graph_partition` 决定是否包裹；`CUDAGraphOptions` 按 piecewise 首/末段差异化。
- [`breakable_cudagraph.py`](breakable-cudagraph.md)：替代实现，`VLLM_USE_BREAKABLE_CUDAGRAPH` 切换。
- [`base_static_graph.py`](base-static-graph.md)：本类实现的 Protocol 接口。
- [`monitor.py`](monitor.md)：`validate_cudagraph_capturing_enabled` 是捕获前的合法性闸门；`set_cudagraph_capturing_enabled(False)` 在不该捕获时拦住。
- [`平台`](../08-platforms/README.md)：`get_static_graph_wrapper_cls()` / `get_global_graph_pool()` / `graph_pool_handle()`。
- [`执行-cudagraph`](../02-execution/worker/cudagraph-capture.md)：Worker 在 forward_context 里设 `batch_descriptor` 与 `cudagraph_runtime_mode`，决定捕不捕获、捕哪张图。
- [`forward_context`](../17-utils-cross-cutting/README.md)：`BatchDescriptor` 描述 batch 形状/uniformity，是 cache key。
- [`offloader`](../03-model-execution/README.md)：`sync_prev_onload` / `join_after_forward` 与权重预取 copy stream 协同。

## 历史版本演进

- **v0.5–v0.6**：CUDA Graph 在 Worker 层手写管理（v0 路径，待核实）。
- **v0.7**：`CUDAGraphWrapper` 引入并接入 `VllmBackend`，支持 PIECEWISE；`CUDAGraphOptions` 的首/末段差异。
- **v0.8**：`FULL` / `PIECEWISE` 双 runtime mode + 嵌套 dispatch；`weak_ref_tensors` 末段省显存；`_all_instances` + `clear_all_graphs` 支持重捕获。
- **v0.9**：`CUDAGraphStat`/`CUDAGraphLogging` 统计表；`set_graph_pool_id` 跨 wrapper 共享 pool；offloader sync/join 接入。
- **v0.10**：`CUDAGraphMode.FULL_DECODE_ONLY` / `FULL_AND_PIECEWISE` 混合模式（decode 用 FULL、prefill 用 PIECEWISE）。
- **v0.11 / v0.12 / main**：与 breakable cudagraph 共存（平台选择 wrapper 类）；MRv2 切换中 `clear_all_graphs` 配合重编译。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [breakable-cudagraph.md](breakable-cudagraph.md) — 运行期断点捕获替代方案。
- [base-static-graph.md](base-static-graph.md) — 平台 wrapper 的 Protocol。
- [backends.md](backends.md) — `wrap_with_cudagraph_if_needed`。
- [monitor.md](monitor.md) — 捕获合法性闸门。
- [../02-execution/worker/cudagraph-capture.md](../02-execution/worker/cudagraph-capture.md) — Worker 侧捕获协调。
