# IrOp 与注册体系（op.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > IrOp

源码：`vllm/ir/op.py`（约 664 行）

## 是什么

`op.py` 是 vLLM IR 算子库的核心：定义 `IrOp` 类体系、`@register_op`/`@IrOp.register_impl` 装饰器、provider 优先级 dispatch、maybe_inplace overload、torch 库注册，以及测试/容差支持。

类与成员：

- `vllm_ir_torch_lib = Library("vllm_ir", "FRAGMENT")`（`op.py:21`）：torch 库命名空间，`FRAGMENT` 不占用整个 namespace。
- `register_op(f=None, *, name, activations, allow_inplace)`（`op.py:106`）：装饰器，创建 `IrOp`（或 `IrOpInplace`）并入 `IrOp.registry`。
- `IrOp`（`op.py:155`）：
  - `registry: ClassVar[dict[str, IrOp]]`（`op.py:156`）。
  - `impls: dict[str, IrOpImpl]`（`op.py:159`），含保留的 `"native"`。
  - `activations` / `activation_indices`（`op.py:191`）：Convention：参数名以 `x` 开头视为 activation；`allow_inplace` 时这些参数可被 donate 复用。
  - `torch_op: OpOverload`（`op.py:226`）：`torch.ops.vllm_ir.<name>.default`。
  - `register_fake(fn)`（`op.py:228`）、`register_impl(provider, supported, supports_args, inplace)`（`op.py:244`）、`register_input_generator(fn)`（`op.py:452`）、`override_tolerance(dtype, atol, rtol)`（`op.py:464`）、`get_tolerance(dtype)`（`op.py:469`）。
  - `dispatch(*args) -> IrOpImpl`（`op.py:327`）、`set_default(priority)` / `set_priority(priority)`（`op.py:415`/`428`）、`supported_providers()`（`op.py:445`）、`get_priority()`（`op.py:385`）。
  - `_inner_call(*args)`（`op.py:304`）→ `dispatch` → `impl.func_impl_fn`。
  - `__call__(*args)`（`op.py:368`）：`_ENABLE_TORCH_WRAP` 控制走 `torch_op` 还是 `_inner_call`。
- `IrOpInplace(IrOp)`（`op.py:481`）：`allow_inplace=True`，构造时创建 `IrOpInplaceOverload`。
- `IrOpInplaceOverload`（`op.py:500`）：`<name>.maybe_inplace` overload，`mutates_args=activations`，`__call__`/`_inner_call` 直接用 `impl.impl_fn`（不需 clone，因 maybe_inplace 语义允许 inplace）。
- `IrOpImpl`（`op.py:542`）：`provider`/`impl_fn`/`supported`/`supports_args`/`inplace`/`_registration_stack`。`supports_args` 无则 `supports_all_args=True`；`func_impl_fn`（`op.py:650`）对 inplace impl clone activation 保 functional；`uuid()`（`op.py:637`）`hash_source(impl_fn 源文件)`。

工具与常量：

- `RESERVED_PROVIDERS = ["native", "unfused"]`（`op.py:36`）：不可用于自定义实现。
- `_validate_name`（`op.py:40`）：`^[a-z_][a-z_0-9]*$`。
- `set_default_torch_wrap(enable)` / `enable_torch_wrap(enable)`（`op.py:54`/`62`）：全局 flag 与上下文管理器。
- `_torch_ops_subtree(lib)`（`op.py:26`）：从 `lib.ns` 解析 `torch.ops.<ns>`，兼容 doc mock。

## 为什么

- **native 实现即 reference**：每个 op 注册时 `_f` 成为 `native` impl 与默认 fake impl，是 provider 实现的数值基准与回退。`IrOpImpl` 强制 impl schema 与 native 完全一致（`infer_schema` 比对，`op.py:563`），保 dispatch 切换不改变语义。
- **provider 优先级 + `supports_args` 两级筛选**：`set_default(["cuda","native"])` 静态限定候选，`dispatch` 运行期按 `supports_args(*args)` 动态选首个支持当前 dtype/shape/device 的。这使"fused kernel 仅支持 per-token 量化、其它回落 native"等策略声明式表达。
- **`_filter_priority_impls` 早截断**：若某 impl `supports_all_args`（无 `supports_args`）则它之后不再追加，省 dispatch 比对；末尾强制追加 `native` 兜底（`op.py:412`），保证总有可用实现。
- **maybe_inplace = 显式 donate 语义**：`allow_inplace=True` 生成独立 `maybe_inplace` overload（`mutates_args=activations`），调用方选 default（functional）或 maybe_inplace（允许 inplace）。`VllmIRInplaceFunctionalizationPass` 在 pre-grad 把 maybe_inplace 改回 default 并记 `donated_input_ids`，使 `UnsafeCloneEliminationPass` 知道哪些 graph input 可被 inplace 复用而安全消除 clone。
- **`func_impl_fn` 的 clone 保护**：default overload 必须功能化（functional）。若 provider impl 是 inplace（`inplace=True`），`func_impl_fn` 在调 `impl_fn` 前 clone 所有 activation，保证 default 语义不破坏原输入。lowering 后这些 clone 多数冗余，由 `UnsafeCloneEliminationPass` 清。
- **`uuid` 进缓存键**：`IrOpImpl.uuid` 用 `hash_source(impl_fn 所在文件)`（`weak_cache` 缓存），实现源码变→`VllmIRLoweringPass.uuid` 变→Inductor 重编译。`IrOp.get_priority()` 也进 uuid。
- **torch custom op 注册**：`lib.define(name+schema)` + `lib.impl(CompositeExplicitAutograd)` + `lib._register_fake`，使 IR op 进 torch dispatch、Dynamo 可追踪（走 fake）、Inductor 可 lower、eager 可直调。`CompositeExplicitAutograd` 不被 AOTAutograd 的 ATen IR 归一化分解（`op.py:222`）。
- **`enable_torch_wrap=False`**：eager 热路径或非 Inductor 平台关闭 torch op 层，`__call__` 走 `_inner_call` 直 dispatch，省 torch dispatch 开销（`op.py:368`）。
- **命名校验与保留 provider**：`_validate_name` 防 op/provider 名含非法字符（影响 torch 注册与 pickle）；`RESERVED_PROVIDERS` 防 `native`/`unfused` 被覆盖。
- **registration_stack**：每个 `IrOp`/`IrOpImpl` 存注册调用栈（`op.py:141`/`289`），定位重复注册或顺序问题。

## 怎么做

### 注册 op 与 provider

```python
@register_op(allow_inplace=True)
def fused_add_rms_norm(x, x_residual, weight, epsilon, variance_size=None):
    """Fused add and weighted root-mean-square layer normalization"""
    ...                                              # native reference
fused_add_rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)

@fused_add_rms_norm.register_impl("cuda", inplace=True,
                                  supports_args=lambda x,xr,w,e,vs=None: x.is_cuda)
def _cuda_impl(x, x_residual, weight, epsilon, variance_size=None):
    torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)  # inplace
    return x, x_residual

fused_add_rms_norm.set_default(["cuda", "native"])
```

### dispatch 选择

```python
# op.py:327 简化
def dispatch(self, *args, **kwargs):
    if not self._priority_impls:
        logger.warning_once("Priority not set, using native")
        return self.impls["native"]
    for impl in self._priority_impls:
        if not impl.supported: raise ValueError(...)
        if impl.supports_args(*args, **kwargs): return impl
        logger.debug("Skipping provider %s ...", impl.provider)
    raise RuntimeError("last impl must support all args (native)")
```

### maybe_inplace 调用

```python
# fusion pattern 里
result_rms, residual = vllm.ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, w, eps)
# 调 IrOpInplaceOverload.__call__ → torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(...)
# pre-grad pass 把它改回 default overload + 标记 donated_input_ids
```

### uuid

`IrOpImpl.uuid`（`op.py:637`）：`hash_source(Path(inspect.getfile(impl_fn)))`，`weak_cache` 缓存。`VllmIRLoweringPass.uuid` 汇总所有 op 的 priority + 各 impl uuid。

## 与其它模块/系统配合

- [`ir-README.md`](ir-README.md)：总览与 `vllm_ir` 命名空间。
- [`passes/ir.md`](passes/ir.md)：`VllmIRLoweringPass` 调 `dispatch`/`func_impl_fn`；`VllmIRInplaceFunctionalizationPass` 改 maybe_inplace→default；`UnsafeCloneEliminationPass` 清 `func_impl_fn` 的 clone。
- [`passes/fusion.md`](passes/fusion.md)：fusion pattern 调 `vllm.ir.ops.<name>` / `.maybe_inplace`。
- [`ir-ops.md`](ir-ops.md)：内置 op 的 `@register_op` 实现。
- [`tolerances.md`](tolerances.md)：`get_tolerance` 读 `DEFAULT_TOLERANCES` + `override_tolerance`。
- [`模型执行-custom_op`](../03-model-execution/layers/custom-op.md)：provider `impl_fn` 调 `torch.ops._C.*`/`torch.ops.vllm.*`。
- [`平台`](../08-platforms/README.md)：provider `supported` 常用 `current_platform.is_cuda()` 等；厂商平台可 `set_default` 调整优先级。

## 历史版本演进

- **v0.8（IR 框架成型）**：`IrOp`/`IrOpImpl`/`IrOpInplace`/`IrOpInplaceOverload` 类体系；`register_op`/`register_impl`；`Library("vllm_ir","FRAGMENT")`；`CompositeExplicitAutograd` 注册；`set_default`/`set_priority`/`dispatch`/`_filter_priority_impls`；`func_impl_fn` 的 activation clone。
- **v0.9**：`enable_torch_wrap`/`set_default_torch_wrap`；`IrOpImpl.uuid` + `weak_cache`；`register_input_generator`/`override_tolerance`/`get_tolerance`；`registration_stack` 记录；`_validate_name` + `RESERVED_PROVIDERS`。
- **v0.10 / main**：`_torch_ops_subtree` 兼容 doc mock（`lib.ns` 解析）；`supports_args` 签名校验强化（参数数/名/默认须与 native 一致）；`variance_size` 参数。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [ir-README.md](ir-README.md) — IR 库总览。
- [ir-ops.md](ir-ops.md) — `@register_op` 实例。
- [passes/ir.md](passes/ir.md) — lowering/functionalization/clone 消费 IR op。
- [tolerances.md](tolerances.md) — `get_tolerance` 的容差来源。
- [passes/fusion.md](passes/fusion.md) — pattern 调用 IR op。
