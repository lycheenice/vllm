# 分区规则（partition_rules.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Partition Rules

源码：`vllm/compilation/partition_rules.py`（约 75 行）

## 是什么

`partition_rules.py` 提供两件事：(1) `should_split(node, splitting_ops)` 判断 FX 节点是否为切分点；(2) `inductor_partition_rule_context(splitting_ops)` 临时注册 Inductor 分区规则的上下文管理器。

- `should_split`（`partition_rules.py:14`）：操作 Dynamo 图，`node.target` 须是 `OpOverload`/`OpOverloadPacket`，按 `_qualified_op_name` 或 `name()` 匹配 `splitting_ops`。
- `inductor_partition_rule_context`（`partition_rules.py:41`）：保存 `torch._inductor.config.custom_should_partition_ops`，临时设为 `splitting_ops`，退出恢复。

## 为什么

- **FX 切分与 Inductor 切分共用切分点定义**：`splitting_ops`（默认注意力类 op）既被 `split_graph()`（Dynamo FX 层）判断，也被 Inductor 自分区路径（lowering 后调度层）使用，单一来源保证两路径边界一致。
- **`use_inductor_graph_partition` 双模式**：`VllmBackend` 在 `use_inductor_graph_partition=False` 时由 `split_graph` 预切 FX 图；`True` 时不预切（`fx_split_ops=[]`），改由 Inductor 调度器按 `custom_should_partition_ops` 自行分区。后者靠 `inductor_partition_rule_context` 在 `CompilerManager.compile_context` 内注册。
- **OpOverload 包络兼容**：`should_split` 同时处理 `OpOverloadPacket`（如 `aten::add`）与具体 `OpOverload`（如 `aten::add.default`），按 packet 名与 overload 全名都匹配，避免遗漏。

## 怎么做

### should_split 判定

```python
# partition_rules.py:14 简化
if node.op != "call_function": return False
target = node.target
if isinstance(target, OpOverloadPacket):
    return target._qualified_op_name in splitting_ops
if isinstance(target, OpOverload):
    packet_name = target.name()                      # "aten::add"
    overload_name = f"{packet_name}.{target._overloadname}"  # "aten::add.default"
    return overload_name in splitting_ops or packet_name in splitting_ops
return False
```

### compile_context 注册

`CompilerManager.compile_context`（`backends.py:149`）：

```python
with pass_context(compile_range):
    if self.compilation_config.use_inductor_graph_partition:
        with inductor_partition_rule_context(self.compilation_config.splitting_ops):
            yield
    else:
        yield
```

`inductor_partition_rule_context` 把 `splitting_ops` 写进 `torch._inductor.config.custom_should_partition_ops`，让 Inductor 调度器在分区时强制按这些 op 切。`splitting_ops` 为空则跳过注册（`partition_rules.py:54`）。

## 与其它模块/系统配合

- [`backends.py`](backends.md)：`split_graph` 用 `should_split`；`compile_context` 用 `inductor_partition_rule_context`。
- [`compiler_interface.py`](compiler-interface.md)：`InductorStandaloneAdaptor.compile` 在 `compile_context` 内调用，读 `custom_should_partition_ops`。
- [`配置-compilation`](../10-config/README.md)：`CompilationConfig.splitting_ops` / `use_inductor_graph_partition`。
- [`注意力`](../05-attention/README.md)：`splitting_ops` 主要是注意力类 op，以注意力为切分边界。

## 历史版本演进

- **v0.7**：`should_split` 随 `split_graph` 引入，FX 预切路径。
- **v0.8**：`inductor_partition_rule_context` + `use_inductor_graph_partition` 引入，支持 Inductor 自分区。
- **v0.9 / main**：`custom_should_partition_ops` 的保存/恢复语义稳定；`_qualified_op_name` 匹配兼容新版 torch op 命名。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [backends.md](backends.md) — `split_graph` 与 `compile_context`。
- [compiler-interface.md](compiler-interface.md) — 适配器如何读分区配置。
- [../10-config/README.md](../10-config/README.md) — `splitting_ops` / `use_inductor_graph_partition`。
