# FX 工具（fx_utils.py）

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > FX Utils

源码：`vllm/compilation/passes/fx_utils.py`（约 77 行）

## 是什么

`fx_utils.py` 是一组无状态 FX Graph 节点查询 helper，封装"判断节点是否为某 op"、"在 `auto_functionalized` 包装下找某 op"、"找 getitem 用户"等高频操作，供 fusion pass 与 utility pass 复用。

函数清单：

- `is_func(node, target)`（`fx_utils.py:13`）：`node.op=="call_function" and node.target==target`。
- `is_auto_func(node, op)`（`fx_utils.py:17`）：节点是 `auto_functionalized` 且 `args[0]==op`。
- `find_auto_fn_maybe(nodes, op)` / `find_auto_fn(nodes, op)`（`fx_utils.py:22`/`30`）：在节点序列里找首个 `auto_functionalized(op)` 节点。
- `find_getitem_maybe(node, idx)` / `find_getitem(node, idx)`（`fx_utils.py:38`/`46`）：找从 `node` 取第 `idx` 元的 `operator.getitem` 用户。
- `find_op_nodes(op, graph)`（`fx_utils.py:54`）：遍历图找所有目标 op 节点，**同时覆盖裸 `call_function(op)` 与 `auto_functionalized(op)` 两种形态**；支持 `OpOverloadPacket`（展开全部 overload）。
- `get_only_user(node)`（`fx_utils.py:75`）：断言节点仅一个 user 并返回。

## 为什么

- **`auto_functionalized` 双形态统一**：vLLM 自定义 op 在 functionalized 图里以 `auto_functionalized(op, **kwargs)` 高阶算子形态出现（输出 tuple，再 getitem 取各结果），但在非 functional 路径或 defunctionalize 后以裸 `op(...)` 出现。`find_op_nodes`/`is_auto_func` 让 pass 同时匹配两种形态，避免漏匹配。
- **OpOverloadPacket 展开**：`torch.ops._C.silu_and_mul` 是 packet，`silu_and_mul.default` 是 overload。`find_op_nodes` 遇 packet 时展开 `op.overloads()` 逐 overload 搜索，调用方只需传 packet 或 overload 任一即可。
- **getitem 定位简化**：functionalized op 的多个输出靠 `operator.getitem(at, i)` 取，`find_getitem` 直接定位第 i 路输出，是 fusion/fix_functionalization 的固定套路。
- **`get_only_user` 防御性断言**：fusion 假设某节点单用户（如 reshape 链折叠），`get_only_user` 显式断言，避免多用户时误替换。

## 怎么做

```python
# 典型用法：在 fix_functionalization 里找 rotary_embedding 的 getitem 用户
at_target = node.args[0]
if at_target == torch.ops._C.rotary_embedding.default:
    getitem_nodes = {}
    for user in node.users:
        if is_func(user, operator.getitem):
            getitem_nodes[user.args[1]] = user
```

```python
# find_op_nodes 覆盖两种形态
for n in find_op_nodes(torch.ops._C.rotary_embedding, graph):
    ...  # 命中 call_function(rotary_embedding) 与 auto_functionalized(rotary_embedding)
```

## 与其它模块/系统配合

- [`vllm-inductor-pass.md`](vllm-inductor-pass.md)：`fold_consecutive_reshapes` / `_remove_noop_permute` 用 `is_func`。
- [`utility.md`](utility.md)：`FixFunctionalizationPass` / `ScatterSplitReplacementPass` 大量用 `is_func`/`find_getitem`/`is_auto_func`。
- [`fusion.md`](fusion.md)：fusion pass 的 pattern 内部检查（如 `_rms_input_weight_dtype_match`）与 replacement 后处理用这些 helper。
- [`ir.md`](ir.md)：`get_ir_op`（`ir/utils.py`）用 `overload_or_default` 解析 packet/overload，思路同 `find_op_nodes`。

## 历史版本演进

- **v0.7**：`is_func`/`find_auto_fn`/`find_getitem` 引入，服务 RMSNorm+quant fusion。
- **v0.8**：`find_op_nodes` 引入，统一 packet/overload + 裸/auto_functionalized 双形态；`is_auto_func` 抽出；`get_only_user` 加入。
- **v0.9 / main**：helper 稳定，随 fusion/utility pass 扩增被更广泛复用。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [vllm-inductor-pass.md](vllm-inductor-pass.md) — `_trace_fn` 用 `is_func` 折叠 reshape。
- [utility.md](utility.md) — `FixFunctionalizationPass` 的 getitem 定位。
- [fusion.md](fusion.md) — pattern 内部检查。
