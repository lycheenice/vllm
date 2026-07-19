# IR Pass（passes/ir/）

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > IR Pass

源码：`vllm/compilation/passes/ir/`（`lowering_pass.py` / `clone_elimination.py` / `inplace_functionalization.py` / `utils.py`）

## 是什么

`passes/ir/` 实现与 vLLM IR 算子（`vllm/ir/`，[`ir-README.md`](../ir-README.md)）配合的三段 pass：pre-grad 的 inplace functionalization、post-grad 的 IR lowering、post-lowering 的冗余 clone 消除。

- `VllmIRInplaceFunctionalizationPass`（`inplace_functionalization.py:21`）：pre-grad pass，把 `maybe_inplace` overload 替换成 `default` overload，使图 functional，并标记 `donated_input_ids`。
- `VllmIRLoweringPass`（`lowering_pass.py:25`）：post-grad pass，用 `register_graph_pattern` 匹配每个 `vllm_ir::*` 节点，调 `IrOp.dispatch` 选 provider 实现，`match.replace_by_example` 把 IR op 替换为实现子图。
- `UnsafeCloneEliminationPass`（`clone_elimination.py:72`）：post-lowering pass，消除 IR lowering 为保护 inplace 而插入的冗余 `aten::clone`。
- `utils.py`：`overload_or_default(op)`（packet→default）、`get_ir_op(node)`（识别 `vllm_ir` 命名空间节点并返回 `IrOp`）。

## 为什么

- **`maybe_inplace` → `default` 让图 functional**：vLLM IR op 的 `maybe_inplace` overload（见 [`ir-op.md`](../ir-op.md)）允许 inplace 实现，但 AOTAutograd 需 functional 图。本 pass 在 pre-grad 把 `maybe_inplace` 调用改回 `default` overload，同时校验 activation 输入"用完即弃"（否则 inplace 会破坏后续读取），并把被 donate 的 graph input 记入 `PassContext.donated_input_ids`。
- **IR lowering 是 vLLM IR 的核心下沉点**：fusion pass 用 `vllm.ir.ops.*`（IR op）写 pattern，使 fusion 在"抽象 IR 层"匹配，与具体实现解耦。`VllmIRLoweringPass` 在 fusion 之后把每个 IR op 按 provider 优先级下沉成具体实现子图（native / 平台 fused kernel），实现"一次 fusion、多平台实现"。
- **provider 优先级进 uuid**：`VllmIRLoweringPass.uuid`（`lowering_pass.py:115`）含每个 IR op 的 provider 优先级与各 impl 源码 hash，保证优先级或实现变更→重编译。
- **clone 消除回收显存**：IR lowering 为支持 inplace 实现的 functional 语义，`IrOpImpl.func_impl_fn` 会 clone activation（`ir/op.py:650`）；lowering 后这些 clone 多数冗余（输入已 donate 或无人后续读）。`UnsafeCloneEliminationPass` 用 `donated_input_ids` 与"layout 保持 + 写后无后续读"判定安全消除，否则保留。
- **`get_ir_op` 命名空间识别**：仅 `node.target` 的 `namespace=="vllm_ir"` 才视为 IR op，避免把同名 `torch.ops.vllm.*` 误判；未注册 op 名告 warning（可能是 torch 注册错乱）。
- **"unsafe"的含义**：`UnsafeCloneEliminationPass` 暂不考虑 aliasing（view→q,k,v 等），仅支持已知 vLLM 场景，不保证通用图正确性（`clone_elimination.py:78`）。

## 怎么做

### VllmIRInplaceFunctionalizationPass

```python
# inplace_functionalization.py:40 简化
def __call__(self, graph):
    get_pass_context().donated_input_ids = set()
    for node in graph.nodes:
        if (ir_op := get_ir_op(node)) is None: continue
        overload = overload_or_default(node.target)._overloadname
        if overload != "maybe_inplace": continue
        assert ir_op.allow_inplace
        # 校验 activation 输入无后续使用
        for arg_idx in ir_op.activation_indices:
            for user in node.args[arg_idx].users:
                if idx[user] > idx[node]:
                    raise ValueError("activation used after maybe_inplace, donated semantics broken")
            if node.args[arg_idx].op == "placeholder":
                pass_context.donated_input_ids.add(idx[node.args[arg_idx]])
        node.target = ir_op.torch_op          # maybe_inplace → default
```

### VllmIRLoweringPass

```python
# lowering_pass.py:43 简化
def lower_matched_op(self, match, *args, **kwargs):
    node = match.nodes[0]
    ir_op = get_ir_op(node)
    fake_args = fx.map_arg(node.args, lambda a: a.meta["val"])
    ir_op_impl = ir_op.dispatch(*fake_args)               # 按 provider 优先级选
    self.selected_impls[ir_op.name][node.name] = ir_op_impl.provider
    bound_args = ir_op._py_signature.bind(*node.args); bound_args.apply_defaults()
    match.replace_by_example(ir_op_impl.func_impl_fn, bound_args.args, run_functional_passes=False)
```

`__call__` 末尾扫描残留 `vllm_ir` 节点（lowering 失败）并 warning。`func_impl_fn`（`ir/op.py:650`）对 inplace 实现会 clone activation 保 functional 语义，故 lowering 后图含clone。

### UnsafeCloneEliminationPass

```python
# clone_elimination.py:88 简化
for node in graph.nodes:
    if not is_func(node, torch.ops.aten.clone.default): continue
    original = node.args[0]
    if not clone_preserves_layout(node, original): continue
    write_idxs = [idx[u] for u in node.users if user_writes_to_node(u, node)]
    if write_idxs:  # 有写：检查 original 是否在写后被用、是否 donated graph input
        if any(idx[orig_user] > write_idx for orig_user in original.users): continue
        if original.op=="placeholder" and idx[original] not in donated_input_ids: continue
    node.replace_all_uses_with(original)
    graph.erase_node(node)
```

`clone_preserves_layout` 比较 stride/storage_offset；`user_writes_to_node` 用 op schema 的 `is_write` 标记判断，并特判 `auto_functionalized`/`TritonKernelWrapperFunctional` 等高阶算子。

### utils.get_ir_op

```python
# ir/utils.py:19 简化
if node.op != "call_function" or not isinstance(node.target, (OpOverload,OpOverloadPacket)):
    return None
op_overload = overload_or_default(node.target)
if op_overload.namespace != "vllm_ir": return None
return IrOp.registry.get(op_overload._opname)  # 未注册则 warning
```

## 与其它模块/系统配合

- [`pass-manager.md`](pass-manager.md)：`VllmIRLoweringPass`/`UnsafeCloneEliminationPass` 是固定尾段；`PostCleanupPass` 在其前后各跑一次。
- [`backends.md`](../backends.md)：`configure_post_pass` 把 `VllmIRInplaceFunctionalizationPass` 注入 `pre_grad_custom_pass`（`backends.py:934`）。
- [`ir-op.md`](../ir-op.md)：`IrOp.dispatch`/`IrOpImpl.func_impl_fn`/`IrOpImpl.uuid` 是 lowering 调用链；`maybe_inplace` overload 是 functionalization 的目标。
- [`ir-ops.md`](../ir-ops.md)：`rms_norm`/`fused_add_rms_norm` 是当前内置 IR op，被 fusion pattern 用、被 lowering 下沉。
- [`fusion.md`](fusion.md)：fusion pattern 用 `vllm.ir.ops.*` 写，依赖 lowering 落实；`fused_add_rms_norm` 设 `allow_inplace=True`，被 pre-grad functionalize。
- [`inductor-pass.md`](inductor-pass.md)：`PassContext.donated_input_ids` 由 functionalization 写、clone elimination 读。
- [`配置-compilation`](../../10-config/README.md)：`pre_grad_custom_pass` / `post_grad_custom_post_pass` hook。

## 历史版本演进

- **v0.8（IR 框架成型）**：`VllmIRInplaceFunctionalizationPass`（pre-grad）+ `VllmIRLoweringPass`（post-grad）+ `UnsafeCloneEliminationPass`（post-lowering）三件套落地；`get_ir_op` 命名空间识别。
- **v0.9**：`UnsafeCloneEliminationPass` 的 `donated_input_ids` 协同成熟；`clone_preserves_layout` 引入 stride 校验；`func_impl_fn` 的 activation clone 机制完善。
- **v0.10 / main**：`VllmIRLoweringPass.uuid` 含 provider 优先级 + impl uuid；`replace_by_example(run_functional_passes=False)` 防 DCE 误删 inplace 子图；残留节点 warning。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [pass-manager.md](pass-manager.md) — 三 pass 在流水线中的位置。
- [../ir-op.md](../ir-op.md) — `IrOp.dispatch`/`maybe_inplace`/`func_impl_fn`。
- [../ir-ops.md](../ir-ops.md) — 内置 IR op。
- [fusion.md](fusion.md) — pattern 用 IR op。
- [utility.md](utility.md) — `PostCleanupPass` 在 lowering 前后跑。
