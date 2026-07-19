# PiecewiseBackend

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > PiecewiseBackend

源码：`vllm/compilation/piecewise_backend.py`（约 380 行）

## 是什么

`PiecewiseBackend`（`piecewise_backend.py:86`）是单个切分子图的"编译 + 运行期分派"单元。`VllmBackend` 切出 N 段子图，每段非切分子图都构造一个 `PiecewiseBackend`，由它对每个 `compile_range` 预编译，运行期据 batch size 选对应 `runnable` 调用。

关键成员：

- `range_entries: dict[Range, RangeEntry]`：每个待编译 range 一条记录，`RangeEntry`（`piecewise_backend.py:79`）含 `compiled`/`runnable`。
- `compile_ranges`：来自 `compilation_config.get_compile_ranges()`，encoder 模式下末段上界改 `MAX_INT32`。
- `compile_sizes`：显式特化的单点 size。
- `sym_shape_indices`：含符号形状的输入下标，用于运行期从 args 取 `runtime_shape`。
- 两种互斥模式：`graph is not None`（冷编译）/ `compiled_runnables is not None`（热装载）。
- `to_bytes()`（`piecewise_backend.py:209`）：mega-AOT 序列化，自定义 `StandaloneCompiledArtifactsPickler` 处理 `CachingAutotuner`。

辅助：`get_fake_args_from_graph()`（从 placeholder meta 取 fake args）、`create_concrete_args()`（把 SymInt 替换为具体 size，用于单点编译）。

## 为什么

- **形状特化但不变 Dynamo**：不同 batch size 对应不同最优 triton kernel，但要复用同一条 FX Graph。`PiecewiseBackend` 一次性为全部 `compile_range` 编译好 `runnable`，运行期 O(1) 分派，避免在 inference 热路径触发 Dynamo/Inductor。
- **支持"单点 + 区间"混合特化**：`compile_sizes` 是精确 size（用于 cudagraph capture），`compile_ranges` 是区间（覆盖区间内任意 size）。`_find_range_for_shape`（`piecewise_backend.py:343`）先查单点精确命中，再回落到区间。
- **冷热路径同构**：`compile_all_ranges()`（冷）与 `load_all_ranges()`（热）走相同 `range_entries` 结构，mega-AOT 热启动时 `PiecewiseBackend` 接收预编译 `compiled_runnables`，无需再持有 FX graph。两类构造分别被 `PiecewiseCompileInterpreter`（冷）与 `reconstruct_serializable_fn_from_mega_artifact`（热）使用，`backends.py:693` 注释要求二者改动同步。
- **tuple 返回适配**：`returns_tuple` 记录子图是否返回 tuple，`get_compiled_graph_wrapper`（`piecewise_backend.py:194`）按需解包，对齐 Inductor `f(list)->tuple` 与 Dynamo `f(*args)->Any` 两种调用约定。
- **encoder 形状不可预测**：encoder 输入 token 数取决于多模态输入，无法按 `max_num_batched_tokens` 设区间上界，故末段上界放开到 `2**31-1`（`piecewise_backend.py:138`）。

## 怎么做

### 冷编译路径

```python
# piecewise_backend.py:245 简化
def compile_all_ranges(self):
    for entry in self.range_entries.values():
        if entry.compiled: continue
        if entry.compile_range.is_single_size():
            args = create_concrete_args(self.graph, entry.compile_range.start)
        else:
            args = get_fake_args_from_graph(self.graph)
        entry.runnable = self.vllm_backend.compiler_manager.compile(
            self.graph, args, self.vllm_backend.inductor_config,
            self.compilation_config, compile_range=entry.compile_range,
            graph_index=self.piecewise_compile_index,
            num_graphs=self.total_piecewise_compiles,
            is_encoder=self.vllm_backend.is_encoder)
        entry.compiled = True
```

`create_concrete_args`（`piecewise_backend.py:37`）用新 `FakeTensorMode(shape_env=ShapeEnv())`，把每个 SymInt 表达式里的符号 subs 成给定 size，重建 as_strided 张量。

### 运行期分派

```python
# piecewise_backend.py:358 简化
def __call__(self, *args):
    if self.sym_shape_indices:
        runtime_shape = args[self.sym_shape_indices[0]]
        entry = self._find_range_for_shape(runtime_shape)  # 先单点后区间
        assert entry is not None
    else:
        # 全静态：用唯一编译好的 entry
        entry = [e for e in self.range_entries.values() if e.compiled][0]
    return entry.runnable(*args)
```

### mega-AOT 序列化

`to_bytes()` 遍历已编译 `range_entries`，对带 `serialize` 方法的 runnable 调 `fn.serialize()`，再用 `StandaloneCompiledArtifactsPickler`（`reducer_override` 处理 `CachingAutotuner.prepare_for_pickle`）dump 成 bytes，key 为 `str(Range)`。这些 bytes 被 `collect_standalone_compile_artifacts()` 收进 `StandaloneCompiledArtifacts`（见 [caching.md](caching.md)）。

### 热装载路径

`load_all_ranges()`（`piecewise_backend.py:319`）从 `compiled_runnables[key=str(range)]` 取预编译 callable，包 `get_compiled_graph_wrapper`，标记 `compiled=True`。`reconstruct_serializable_fn_from_mega_artifact`（`caching.py:411`）先 `standalone_compile_artifacts.load_all()`（多线程 `pickle.loads` + `AOTCompiledArtifact.deserialize`），再按 submod 名逐个构造 `PiecewiseBackend(graph=None, compiled_runnables=...)`。

## 与其它模块/系统配合

- [`VllmBackend`](backends.md)：`PiecewiseCompileInterpreter.call_module` 是构造入口；`compiler_manager.compile` 是实际编译通道。
- [`compiler_interface.py`](compiler-interface.md)：`compile_range.is_single_size()` 决定 `dynamic_shapes` 与 `max_autotune`。
- [`caching.py`](caching.md)：`to_bytes` 产出供 `StandaloneCompiledArtifacts` 去重存储；`load_all` 反向装载。
- [`codegen.py`](codegen.md)：`PiecewiseBackend` 作为 `__vllm_submods__[i]` 被 generated execution_fn 调用。
- [`cuda_graph.py`](cuda-graph.md)：`wrap_with_cudagraph_if_needed` 包裹 `PiecewiseBackend`，运行期先 cudagraph dispatch 再落 to `__call__`。
- [`配置-compilation`](../10-config/README.md)：`get_compile_ranges()` / `compile_sizes` / `compile_mm_encoder`。

## 历史版本演进

- **v0.7**：`PiecewiseBackend` 引入，仅支持冷编译 + 区间特化；`is_first_graph`/`is_last_graph`/`is_full_graph` 标志用于 cudagraph 包装差异化。
- **v0.8**：`returns_tuple` 加入以适配 Inductor 调用约定变化；encoder compile_range 上界放开。
- **v0.9**：`to_bytes` + `StandaloneCompiledArtifactsPickler` 支持 mega-AOT；`load_all_ranges` 热装载路径成型，`backends.py:693` 注释说明冷热路径需同步维护。
- **v0.10**：`_find_range_for_shape` 单点优先策略稳定；`create_concrete_args` 重建 as_strided 以保留 stride/storage_offset。
- **v0.11 / v0.12 / main**：`_log_compile_start` 向 TORCH_TRACE/tlparse 上报编译事件；submod_name 透传用于调试。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [backends.md](backends.md) — `PiecewiseCompileInterpreter` 与 `wrap_with_cudagraph_if_needed`。
- [caching.md](caching.md) — 冷热路径的序列化对称性。
- [compiler-interface.md](compiler-interface.md) — `compile_range` 如何影响 Inductor 配置。
- [cuda-graph.md](cuda-graph.md) — `CUDAGraphOptions` 的首/末段差异。
