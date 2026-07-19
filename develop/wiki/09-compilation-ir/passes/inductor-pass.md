# InductorPass 基类

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > InductorPass

源码：`vllm/compilation/passes/inductor_pass.py`（约 146 行）

## 是什么

`InductorPass`（`inductor_pass.py:66`，继承 torch `CustomGraphPass`）是所有 vLLM 自定义 Inductor pass 的基类。它用"自身源码 hash"作为默认 `uuid()`，并提供 `PassContext`（编译期上下文）、`enable_fake_mode`（装饰器）等基础设施。

关键成员：

- `PassContext`（`inductor_pass.py:29`）：`compile_range: Range` + `donated_input_ids: set[int]`。
- `get_pass_context()` / `pass_context(compile_range)`（`inductor_pass.py:37`/`43`）：模块级全局上下文管理器，由 [`backends.py`](../backends.md) `CompilerManager.compile_context` 进入。
- `InductorPass.uuid()`（`inductor_pass.py:72`）：默认 `hash_source(self)`。
- `InductorPass.hash_source(*srcs)`（`inductor_pass.py:81`）：`functools.cache` 缓存的源码 SHA256，按"实例的类"解析以避免重复 `inspect.getsource`。
- `InductorPass.hash_dict(dict_)`（`inductor_pass.py:99`）：字典 JSON 的 SHA256，供 pass 把多字段状态压成 uuid。
- `is_applicable_for_range(compile_range)`（`inductor_pass.py:110`）：默认恒真，子类按 range 覆写。
- `CallableInductorPass`（`inductor_pass.py:114`）：把普通 callable 包成 pass，uuid 取 callable 源码 hash。
- `enable_fake_mode(fn)`（`inductor_pass.py:133`）：装饰器，在 `torch._guards.tracing(None)` + `unset_fake_temporarily()` + `FakeTensorMode()` 下运行，用于 pattern 注册等不需要真张量的场景。

## 为什么

- **uuid = cache key**：Inductor 把 `post_grad_custom_post_pass` 的 `uuid` 纳入 code-cache hash。`InductorPass.uuid` 默认 hash 自身源码，保证"pass 代码变 → uuid 变 → 重编译"，无需手动维护版本号。
- **`hash_source` 缓存**：`inspect.getsource` 较慢，`functools.cache` 按 `(str|type|FunctionType)` 解析后的 key 缓存，多个 pass 实例/多次调用只算一次。实例按 `__class__` 解析（同一类多实例共享 uuid）。
- **`PassContext` 跨 pass 共享**：`compile_range` 让 pass 知道当前编译的形状区间（影响 fusion 适用性）；`donated_input_ids` 由 `VllmIRInplaceFunctionalizationPass`（pre-grad）写入，`UnsafeCloneEliminationPass`（post-grad）读取，使两段 pass 协同决定哪些 graph input 的 clone 可安全消除。
- **`hash_dict` 结构化 uuid**：pass 把"实现相关字段"组成 dict（如 IR lowering 的 provider 优先级），`hash_dict` 转 JSON 后 SHA256，比拼源码字符串更稳健。
- **`is_applicable_for_range` 区间闸门**：`SequenceParallelismPass` 等仅对大 token 数有益，覆写此方法让 `PostGradPassManager` 跳过不适用的 range，既省编译时间又让 uuid 因 range 而分桶。
- **`CallableInductorPass` 适配用户 pass**：`CompilationConfig.inductor_passes` 允许用户传 callable，本类把它包成符合 `InductorPass` 接口的对象。
- **`enable_fake_mode` for pattern tracing**：`register_replacement` 需要示例张量 trace pattern，但 pattern 注册阶段不该占真显存/真 backend，fake mode 提供元数据级张量。

## 怎么做

### uuid 典型覆写

- 默认：`InductorPass.uuid() → hash_source(self)` → 按 `self.__class__` 取源码。
- `VllmIRLoweringPass.uuid`（`ir/lowering_pass.py:115`）：`super().uuid() + priorities_str + impl_uuids_str`，含每个 IR op 的 provider 优先级与各 impl 的源码 hash。
- `PostGradPassManager.uuid`（`pass_manager.py:206`）：`hash_dict({pass_config, compile_range, passes=[各 pass uuid]})`。
- `VllmFusionPatternMatcherPass.uuid`：`hash_source(type(self), *pattern_replacement 类型)`。

### PassContext 生命周期

`CompilerManager.compile_context`（`backends.py:149`）`with pass_context(compile_range):` 进入 → 各 pass 经 `get_pass_context()` 读 `compile_range`/`donated_input_ids` → 退出恢复前值（支持嵌套编译）。

### enable_fake_mode 用法

```python
@enable_fake_mode
def register(self, pr: VllmPatternReplacement):
    pm.register_replacement(pr.pattern, pr.replacement, pr.get_inputs(), self._trace_fn, self.pm_pass)
```

`FakeTensorMode` 提供假张量，`pm.register_replacement` 据此 trace pattern/replacement 子图。

## 与其它模块/系统配合

- [`backends.md`](../backends.md)：`compile_context` 进出 `pass_context`。
- [`pass-manager.md`](pass-manager.md)：`PostGradPassManager` 用 `is_applicable_for_range` 跳过、用 `uuid` 汇总。
- [`vllm-inductor-pass.md`](vllm-inductor-pass.md)：`VllmInductorPass`/`VllmPatternMatcherPass` 继承本类。
- [`ir.md`](ir.md)：`VllmIRLoweringPass`/`UnsafeCloneEliminationPass`/`VllmIRInplaceFunctionalizationPass` 继承本类，复用 `PassContext`。
- [`utility.md`](utility.md)：各 utility pass 继承本类。
- [`配置-compilation`](../../10-config/README.md)：`CompilationConfig.inductor_passes` 经 `CallableInductorPass` 接入。
- [`ir-op.md`](../ir-op.md)：`IrOpImpl.uuid`（`vllm/ir/op.py:637`）用同款 `hash_source` 思路，供 IR lowering 的 uuid 引用。

## 历史版本演进

- **v0.7**：`InductorPass` + `CustomGraphPass` 引入；`uuid` 默认源码 hash；`hash_source` 未缓存。
- **v0.8**：`PassContext` + `pass_context` 引入（配合 IR inplace functionalization 的 `donated_input_ids`）；`is_applicable_for_range` 引入。
- **v0.9**：`hash_source` 加 `functools.cache`；`CallableInductorPass` 适配 `inductor_passes` 用户 pass；`enable_fake_mode` 装饰器规范化。
- **v0.10 / main**：`hash_dict` 用于 IR lowering 的 provider 优先级 uuid；`hash_source` 按实例类解析。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [pass-manager.md](pass-manager.md) — 编排与 uuid 汇总。
- [vllm-inductor-pass.md](vllm-inductor-pass.md) — vLLM 专属 pass 基类。
- [ir.md](ir.md) — PassContext 的 `donated_input_ids` 读写。
- [../ir-op.md](../ir-op.md) — `IrOpImpl.uuid` 与本类 `hash_source` 同源。
