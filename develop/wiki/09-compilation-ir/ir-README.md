# vLLM IR 算子库总览（vllm/ir/）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > vLLM IR

源码：`vllm/ir/`（`__init__.py` / `op.py` / `util.py` / `tolerances.py` / `ops/`）

## 是什么

`vllm/ir/` 是 vLLM 的"中间表示算子库"。它在 torch `Library("vllm_ir", "FRAGMENT")` 命名空间下定义一组**与实现解耦**的算子（`IrOp`），每个算子有一个 native 实现与零到多个 provider 实现（平台/厂商 fused kernel），由"provider 优先级"在运行/编译期 dispatch。它让 fusion Pass 在"抽象 IR 层"匹配 pattern，把"具体实现选择"延后到 `VllmIRLoweringPass`，实现一次 fusion、多平台/多量化方案下沉。

组成：

- `vllm_ir_torch_lib = Library("vllm_ir", "FRAGMENT")`（`op.py:21`）：IR op 的 torch 库命名空间，`torch.ops.vllm_ir.*`。
- `IrOp`（`op.py:155`）/ `IrOpInplace`（`op.py:481`）/ `IrOpInplaceOverload`（`op.py:500`）/ `IrOpImpl`（`op.py:542`）：IR 算子与实现的类体系。
- `register_op`（`op.py:106`）/ `IrOp.register_impl`（`op.py:244`）：注册装饰器。
- `set_default_torch_wrap` / `enable_torch_wrap`（`op.py:54`/`62`）：控制 IR op 是否走 torch custom op dispatch 层（eager 加速 / 非 Inductor 平台）。
- `IrOp.registry: ClassVar[dict[str, IrOp]]`（`op.py:156`）：全局 op 注册表。
- `ops/`：内置 IR 算子实现（目前 `layernorm.py`：`rms_norm` / `fused_add_rms_norm`，见 [`ir-ops.md`](ir-ops.md)）。
- `tolerances.py`：各 dtype 的默认数值比对容差（见 [`tolerances.md`](tolerances.md)）。
- `util.py`：`hash_source` / `weak_lru_cache` / `weak_cache`。

## 为什么

- **fusion 与实现解耦**：fusion Pass 的 pattern 用 `vllm.ir.ops.rms_norm(...)` 写，匹配发生在 IR 层；`VllmIRLoweringPass` 之后才落成 `native` 或 `torch.ops._C.rms_norm_static_fp8_quant` 等 provider 实现。新增 provider 不改 fusion pattern。
- **provider 优先级 dispatch**：`IrOp.set_default(priority)` / `set_priority(priority)` 设优先级列表，`dispatch(*args)` 按 `supports_args` 选首个支持当前参数的实现，回退 `native`。支持"按 dtype/shape 选实现"（如某些 fused kernel 仅支持 per-token 量化）。
- **maybe_inplace 携带 inplace 语义**：`allow_inplace=True` 的 op 自动生成 `maybe_inplace` overload，使 fusion pattern 能声明"激活可被 donate 复用"的意图；`VllmIRInplaceFunctionalizationPass`（[`passes/ir.md`](passes/ir.md)）在 pre-grad 把它 functionalize 回 default。
- **uuid 进缓存键**：`IrOpImpl.uuid`（源码 hash）参与 `VllmIRLoweringPass.uuid`，provider 实现变更→重编译。`IrOp.get_priority` 也进 uuid，优先级变→重编译。
- **torch custom op 集成**：每个 IrOp 在 torch 库 `define` schema + `impl(CompositeExplicitAutograd)` + `_register_fake`，通过 `torch.ops.vllm_ir.<name>` 暴露，使 Dynamo 能追踪、Inductor 能 lower、eager 能直接调。
- **`enable_torch_wrap=False` 绕过 dispatch**：eager 模式或非 Inductor 平台可关闭 torch op 层，`IrOp.__call__` 直路由到 `_inner_call`→`dispatch`，省 torch dispatch 开销。
- **容差体系支持实现验证**：`DEFAULT_TOLERANCES` + `op.override_tolerance` 为"native vs provider 实现数值比对"提供按 dtype 的容差，使 provider 实现可被自动验证（含 FP8/FP4/INT8 低精度）。
- **input generator 供测试**：`op.register_input_generator` + `op.generate_inputs` 让 IR op 能产出测试输入，配合容差自动跑 native vs provider 一致性测试。

## 怎么做

### 定义一个 IR op

```python
# vllm/ir/ops/layernorm.py 简化
@register_op
def rms_norm(x: Tensor, weight: Tensor|None, epsilon: float,
             variance_size: int|None = None) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    ...  # native 实现

@rms_norm.register_input_generator
def _gen(num_tokens, hidden_size, dtype, epsilon=1e-5): ...

rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)
```

### 注册 provider 实现

```python
@rms_norm.register_impl("cuda", supported=torch.cuda.is_available(),
                        supports_args=lambda x, weight, eps, vs=None: x.is_cuda)
def _cuda_rms_norm(x, weight, epsilon, variance_size=None):
    return torch.ops._C.rms_norm(x, weight, epsilon)  # 平台 fused kernel

rms_norm.set_default(["cuda", "native"])  # 优先 cuda，回退 native
```

provider 的 `impl_fn` schema 必须与 native 完全一致（`infer_schema` 校验，`op.py:563`）；`supports_args` 签名须与 native 一致以保 dispatch 快路径。

### 运行期 dispatch

`IrOp.dispatch(*args)`（`op.py:327`）：按 `_priority_impls` 顺序，首个 `supported && supports_args` 的 impl 胜出；全不支持则回退 `native`。在 hot path，须快（`op.py:332` 注释）。

### lowering（编译期）

`VllmIRLoweringPass.lower_matched_op`（`passes/ir/lowering_pass.py:43`）：`ir_op.dispatch(*fake_args)` 选 impl，`match.replace_by_example(ir_op_impl.func_impl_fn, ...)` 把 IR op 节点替换为实现子图。`func_impl_fn` 对 inplace impl 会 clone activation 保 functional 语义。

## 与其它模块/系统配合

- [`passes/ir.md`](passes/ir.md)：`VllmIRLoweringPass` / `VllmIRInplaceFunctionalizationPass` / `UnsafeCloneEliminationPass` 消费 IR op。
- [`passes/fusion.md`](passes/fusion.md)：fusion pattern 用 `vllm.ir.ops.*` 写。
- [`ir-op.md`](ir-op.md)：`IrOp`/`register_op`/`register_impl`/provider 优先级/maybe_inplace 细节。
- [`tolerances.md`](tolerances.md)：`DEFAULT_TOLERANCES` 字段表。
- [`ir-ops.md`](ir-ops.md)：内置 `rms_norm`/`fused_add_rms_norm`。
- [`模型执行-custom_op`](../03-model-execution/layers/custom-op.md)：provider 实现常是 `torch.ops._C.*` / `torch.ops.vllm.*`。
- [`平台`](../08-platforms/README.md)：provider 的 `supported` 检查 `current_platform`；厂商可在自己平台注册 IR op 的 provider。
- [`配置-compilation`](../10-config/README.md)：`PassConfig` 的 fusion 开关决定哪些 IR op 出现在图中。

## 历史版本演进

- **v0.8（IR 框架成型）**：`vllm/ir/` 引入；`IrOp`/`IrOpImpl`/`IrOpInplace`/`maybe_inplace` 类体系；`register_op`/`register_impl`/`set_default`/`set_priority`；首个内置 op `rms_norm`/`fused_add_rms_norm`；`VllmIRLoweringPass`/`VllmIRInplaceFunctionalizationPass` 接入。
- **v0.9**：`enable_torch_wrap`/`set_default_torch_wrap` 引入（eager/非 Inductor 平台绕过 torch dispatch）；`IrOpImpl.uuid` + `weak_cache`；`register_input_generator` + `override_tolerance` + `DEFAULT_TOLERANCES` 完善测试体系。
- **v0.10 / main**：`_validate_name` 强制 `[a-z_][a-z_0-9]*` 命名；`RESERVED_PROVIDERS=["native","unfused"]`；`shape_id`（torch 2.11+）经 fusion 的 `dynamic_arg_dims` 间接支持；`variance_size` 参数支持 RMSNorm 变体。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [ir-op.md](ir-op.md) — `IrOp` 类与 provider 优先级。
- [ir-ops.md](ir-ops.md) — 内置 IR 算子。
- [tolerances.md](tolerances.md) — 数值容差。
- [passes/ir.md](passes/ir.md) — IR lowering/functionalization/clone 消除。
- [passes/fusion.md](passes/fusion.md) — pattern 用 IR op。
