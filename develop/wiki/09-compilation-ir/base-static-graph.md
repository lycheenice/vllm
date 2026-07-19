# AbstractStaticGraphWrapper（平台扩展点）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > StaticGraphWrapper

源码：`vllm/compilation/base_static_graph.py`（约 57 行）

## 是什么

`AbstractStaticGraphWrapper`（`base_static_graph.py:10`）是一个 `typing.Protocol`，定义"静态图包装器"的接口契约。它不提供实现，只约束 `__init__` 签名（`runnable` / `vllm_config` / `runtime_mode` / `**kwargs`）与 `__call__` 语义：当 forward_context 的 runtime_mode 与实例 mode 匹配时回放/捕获 CUDAGraph，否则直调原 callable。

合法 `runtime_mode` 仅 `CUDAGraphMode.NONE` / `PIECEWISE` / `FULL`（`valid_runtime_modes()`，见 [`配置-compilation`](../10-config/README.md)）。

## 为什么

- **平台可插拔**：CUDA 用 `CUDAGraphWrapper`，ROCm/XPU/TPU/CPU 可提供自己的实现（或选用 [`breakable_cudagraph.py`](breakable-cudagraph.md)），`backends.py` 通过 `current_platform.get_static_graph_wrapper_cls()` 解析 qualname，本 Protocol 是该解析的契约保证。
- **`__getattr__` 转发约定**：所有实现都须把未知属性转发到 `runnable`（`CUDAGraphWrapper.__getattr__`、`BreakableCUDAGraphWrapper.__getattr__`），使 wrapper 对调用方透明，`unwrap()` 暴露原 runnable，`cudagraph_wrapper` 属性返回自身（用于探测）。
- **统一 dispatch 契约**：`__call__` 必须：(1) 无 forward_context → eager；(2) runtime_mode NONE 或不匹配 → eager；(3) 否则按 descriptor 捕获/回放。这保证嵌套（FULL 包 PIECEWISE）与多平台行为一致。

## 怎么做

Protocol 仅两方法签名 + docstring。实现要点（由 `CUDAGraphWrapper`/`BreakableCUDAGraphWrapper` 演示）：

- `__init__(runnable, vllm_config, runtime_mode, **kwargs)`：`kwargs` 接 `CUDAGraphOptions` 等平台特定配置；`runtime_mode != NONE` 是前置断言。
- `__call__(*args, **kwargs)`：依据 `forward_context.cudagraph_runtime_mode` 与 `self.runtime_mode` 匹配决定捕获/回放/直跑。
- 类属 `clear_all_graphs()` + `_all_instances: WeakSet`：支持重编译批量清空。

## 与其它模块/系统配合

- [`平台`](../08-platforms/README.md)：`get_static_graph_wrapper_cls()` 返回实现类的 qualname；`backends.py` 用 `resolve_obj_by_qualname` 实例化。
- [`cuda_graph.py`](cuda-graph.md) / [`breakable_cudagraph.py`](breakable-cudagraph.md)：两个内置实现。
- [`backends.py`](backends.md)：`wrap_with_cudagraph_if_needed` 与 `customized_cudagraph_wrapper`（`decorators.py:754` Inductor 自分区路径）都遵循本契约。
- [`执行-cudagraph`](../02-execution/worker/cudagraph-capture.md)：Worker 依赖该契约设置 forward_context。

## 历史版本演进

- **v0.8**：`AbstractStaticGraphWrapper` Protocol 抽出，把 `CUDAGraphWrapper` 的接口与实现解耦，便于平台厂商实现自有 wrapper。
- **v0.9**：`breakable_cudagraph.py` 作为第二实现接入，本 Protocol 成为二者共同契约。
- **v0.10 / main**：契约稳定，新增 `cudagraph_wrapper` 属性约定与 `unwrap()` 约定（待核实是否在 Protocol 中显式声明）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [cuda-graph.md](cuda-graph.md) — 内置 CUDA 实现。
- [breakable-cudagraph.md](breakable-cudagraph.md) — 断点捕获实现。
- [backends.md](backends.md) — `wrap_with_cudagraph_if_needed` 实例化路径。
- [../08-platforms/README.md](../08-platforms/README.md) — `get_static_graph_wrapper_cls`。
