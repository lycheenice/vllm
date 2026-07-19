# 拼接图代码生成（codegen.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Codegen

源码：`vllm/compilation/codegen.py`（约 235 行）

## 是什么

`codegen.py` 把 `split_graph()` 产出的拼接图 `split_gm` 转译成一段纯 Python 源码 `execution_fn`，再用 `exec` 编译并绑定子图 callable，生成 `partial(execution_fn, __vllm_submods__=submods_list)`。运行期直接调这个函数，绕开 FX `GraphModule.__call__` 的解释器分派与 `__getattr__` 字典查找开销。

关键成员：

- `generate_execution_code(split_gm) -> (code, submod_names, consts)`（`codegen.py:131`）：主入口，遍历 `split_gm.graph.nodes` 生成源码。
- `generate_execution_code_with_name(...)`（`codegen.py:21`）：内部递归实现，支持内联纯 FX 子图。
- `compile_execution_fn(code, submod_callables, submod_names, consts)`（`codegen.py:164`）：`exec` 编译源码 + 绑定 `__vllm_submods__` 与 `__vllm_consts__`。
- `_node_ref(arg, consts, const_index)`（`codegen.py:207`）：把 FX 节点参数翻成源码引用。
- `del_after` 生命周期分析（`codegen.py:45`）：按"最后使用者"位置插入 `del`，提前释放显存。

## 为什么

- **消除 FX 解释开销**：`GraphModule` 的 `__call__` 仍按 `node.op` 分派（`call_module`/`call_function`/...），每步有 dict lookup 与属性解析。拼接图每步都调一个 `PiecewiseBackend`（或 inlined 子函数），用 generated `execution_fn` 后变成纯函数调用 + 列表索引，热路径成本最低。
- **内联纯 FX 子图**：若某个 `call_module` 的子模块本身是 `torch.fx.GraphModule`（非 piecewise），`generate_execution_code_with_name` 递归内联它（`with_submod=False`），无需序列化进 artifact。
- **非原始常量收集**：`torch.device`、DTensor placement 等 `repr()` 不可 eval 的对象，按 `id()` 去重存入 `consts` 列表，源码里用 `__vllm_consts__[i]` 引用（`codegen.py:228`），保证生成源码可被 `exec` 且可被 pickle 落盘。
- **显存早释放**：`del_after` 在"最后使用者"之后插 `del`，使中间张量在 piecewise 段切换点前释放，降低峰值显存。
- **序列化友好**：`execution_code` + `submod_names` + `consts` 三个字符串/列表被存入 `VllmSerializableFunction`（[`caching.py`](caching.md)），热启动时 `compile_execution_fn` 重建 callable，不必反序列化整个 FX 图。
- **`getitem` 特化**：`operator.getitem` 在源码里翻成 `source[index]`，比 `call_function` 更直观高效。

## 怎么做

### generate_execution_code_with_name 主循环

对每个 `node`：

- `placeholder` → 累入 `param_names`，作为函数参数。
- `call_module` → 若子图是纯 `GraphModule` 则递归内联生成 `__vllm_inlined_submods__{idx}`，否则 `__vllm_submods__idx`；登记 `submod_names`。
- `call_function` → `operator.getitem` 译为 `src[index]`；其余译为 `_get_qualified_name(target)(args, kwargs)`。
- `output` → `return ref(node.args[0])`。
- 在 `i in del_after` 时追加 `del name1, name2`。
- 拼出 `def execution_fn(<params>, *, __vllm_submods__):` 体。

### _node_ref 翻译规则

- `fx.Node` → `arg.name`。
- `list/tuple/dict` → 递归。
- `int/float/bool/str/bytes/None` → `repr(arg)`。
- 其它对象 → 按 `id(arg)` 去重入 `consts`，返回 `__vllm_consts__[idx]`（`codegen.py:230`，**按身份去重**而非相等，因 FX 参数在整次 codegen 存活）。

### compile_execution_fn 绑定

```python
# codegen.py:193 简化
namespace = {}
if consts is not None: namespace["__vllm_consts__"] = consts
exec(code, namespace)
fn = namespace["execution_fn"]
submods_list = [submod_callables.get(name) for name in submod_names]
return partial(fn, __vllm_submods__=submods_list)
```

`.get()` 故意返回 `None` 占位以保索引稳定（内联子图无需绑定），便于调试（`codegen.py:200`）。

## 与其它模块/系统配合

- [`backends.py`](backends.md)：`VllmBackend.__call__` 末尾调 `generate_execution_code` + `compile_execution_fn` 生成 `runtime_callable`，包进 `VllmSerializableFunction`。
- [`caching.py`](caching.md)：`VllmSerializableFunction.__init__` 持 `execution_code`/`submod_names`/`consts`；`reconstruct_serializable_fn_from_mega_artifact`（`caching.py:528`）热装载时优先用 `compile_execution_fn`，缺失则回落 `GraphPickler.loads`。
- [`piecewise_backend.py`](piecewise-backend.md)：`PiecewiseBackend` 实例通过 `submod_callables` 注入 `__vllm_submods__` 列表。
- [`tracing`](../16-observability/README.md)：`trace_structured("artifact", name="vllm_execution_code", payload=code)` 把生成源码上报 TORCH_TRACE/tlparse。
- [`配置-compilation`](../10-config/README.md)：无直接字段，但 `cudagraph_copy_inputs` 决定是否再包 `make_copy_and_call`。

## 历史版本演进

- **v0.9–v0.10**：`codegen.py` 引入，替代 FX 解释执行拼接图；`del_after` 生命周期分析加入。
- **v0.10**：`consts`/`__vllm_consts__` 机制加入，处理 `torch.device` 等不可 eval 对象；纯 FX 子图递归内联。
- **v0.11 / v0.12 / main**：`tuple_return` 兼容 torch 2.12+ `split_module`；`dynamo_timed` 标注 `vllm.generate_execution_code`/`vllm.compile_execution_fn`。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [backends.md](backends.md) — 生成阶段的调用上下文。
- [caching.md](caching.md) — 生成产物的序列化与热装载。
- [piecewise-backend.md](piecewise-backend.md) — `__vllm_submods__` 元素来源。
