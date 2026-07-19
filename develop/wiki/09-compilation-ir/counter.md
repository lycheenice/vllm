# CompilationCounter（counter.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Counter

源码：`vllm/compilation/counter.py`（约 58 行）

## 是什么

`CompilationCounter`（`counter.py:12`）是一个 `@dataclasses.dataclass`，全进程单例 `compilation_counter`（`counter.py:58`），记录编译各阶段发生次数与产物数量。提供 `clone()` 与 `expect(**kwargs)` 上下文管理器供测试断言。

计数字段（`counter.py:13-41`）：

- 模型/图级：`num_models_seen` / `num_graphs_seen` / `num_piecewise_graphs_seen` / `num_piecewise_capturable_graphs_seen`（不含切分段）。
- 编译触发：`num_backend_compilations` / `num_inductor_compiles` / `num_eager_compiles` / `num_aot_compiles` / `stock_torch_compile_count`。
- cudagraph：`num_gpu_runner_capture_triggers` / `num_cudagraph_captured`。
- 缓存：`num_cache_entries_updated` / `num_compiled_artifacts_saved` / `num_compiled_artifacts_loaded` / `num_aot_artifacts_saved` / `num_aot_artifacts_loaded`。

## 为什么

- **测试可断言**：`expect(**kwargs)` 上下文在 `yield` 前后快照，断言某字段增量等于期望值（`counter.py:47`），让编译测试精确验证"走了预期路径"而非间接观察日志。
- **诊断与可观测**：`reset_compile_wrapper`（[`wrapper.py`](wrapper.md)）会清零全部字段以便弹性 EP 重编译后重新计数；运行期日志用这些计数汇报编译开销分布。
- **区分冷热路径**：`num_compiled_artifacts_saved` vs `num_compiled_artifacts_loaded`、`num_aot_compiles` vs `num_aot_artifacts_loaded` 直接反映"冷编译 vs 热装载"占比，是评估缓存有效性的第一手指标。
- **区分 piecewise 段类型**：`num_piecewise_graphs_seen` 含切分段，`num_piecewise_capturable_graphs_seen` 不含，二者差即为切分段数，用于验证 split_graph 行为。

## 怎么做

```python
# counter.py:47 简化
@contextmanager
def expect(self, **kwargs):
    old = self.clone()
    yield
    for k, v in kwargs.items():
        assert getattr(self, k) - getattr(old, k) == v, (
            f"{k} not as expected, before {getattr(old,k)} after {getattr(self,k)} expected diff {v}")
```

测试用法：`with compilation_counter.expect(num_inductor_compiles=1, num_cudagraph_captured=2): ...`。

各字段在对应位置自增，例如 `InductorAdaptor.compile` 起点增 `num_inductor_compiles`（`compiler_interface.py:491`），`CUDAGraphWrapper` 捕获后增 `num_cudagraph_captured`（`cuda_graph.py:339`），`StandaloneCompiledArtifacts.insert` 增 `num_compiled_artifacts_saved`（`caching.py:66`）。

## 与其它模块/系统配合

- 几乎被编译子系统所有模块自增（[`backends.py`](backends.md) / [`compiler_interface.py`](compiler-interface.md) / [`piecewise_backend.py`](piecewise-backend.md) / [`cuda_graph.py`](cuda-graph.md) / [`caching.py`](caching.md) / [`decorators.py`](decorators.md)）。
- [`wrapper.py`](wrapper.md)：`reset_compile_wrapper` 清零全部字段。
- [`monitor.py`](monitor.md)：`monitor_profiling_run` 断言 `num_backend_compilations` 在 profiling 期间不增（`monitor.py:75`）。
- [`可观测性`](../16-observability/README.md)：计数通过 logger 汇报，部分进入 TORCH_TRACE。
- 测试套件（`tests/compilation/`）：`expect` 是编译测试断言主手段。

## 历史版本演进

- **v0.7**：`CompilationCounter` 随 piecewise 引入，初版字段覆盖 graph/piecewise/backend/cudagraph。
- **v0.8**：`num_piecewise_capturable_graphs_seen` 区分切分段；`stock_torch_compile_count` 计 stock 模式。
- **v0.9**：cudagraph 捕获字段细化；`expect` 上下文管理器稳定。
- **v0.10**：`num_compiled_artifacts_saved/loaded`、`num_aot_compiles/artifacts_saved/loaded` 随 mega-AOT/AOT 引入。
- **v0.11 / v0.12 / main**：字段随 AOT/standalone 路径成熟稳定；`reset_compile_wrapper` 清零清单同步扩展。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [monitor.md](monitor.md) — profiling 期间断言后端不编译。
- [backends.md](backends.md) / [compiler-interface.md](compiler-interface.md) — 各自的自增点。
- [wrapper.md](wrapper.md) — `reset_compile_wrapper` 清零。
