# VllmInductorPass（vLLM 专属 Pass 基类）

[← Wiki 首页](../../README.md) > [编译与 IR](../README.md) > [Pass 系统](README.md) > VllmInductorPass

源码：`vllm/compilation/passes/vllm_inductor_pass.py`（约 344 行）

## 是什么

`vllm_inductor_pass.py` 提供三层 vLLM 专属 pass 基类与 pattern 注册抽象，在 [`InductorPass`](inductor-pass.md) 之上叠加"vLLM 配置访问、计时日志、graph dump、pattern matcher 计数"等通用能力。

关键成员：

- `InductorCompilationConfig`（`vllm_inductor_pass.py:31`）：pass 持有的精简配置（`splitting_ops` / `use_inductor_graph_partition`），避免持完整 `CompilationConfig`（含模型指针，不安全）。
- `VllmInductorPass`（`vllm_inductor_pass.py:37`）：继承 `InductorPass`，持 `InductorCompilationConfig` + `pass_config` + `model_dtype` + `device`。提供 `time_and_log` 装饰器（`begin`/`dump_graph("before")`/`call_fn`/`dump_graph("after")`/`end_and_log`）、`dump_prefix` 类属性（pass 序号，供 dump 排序）。
- `VllmPatternMatcherPass`（`vllm_inductor_pass.py:95`）：使用 `torch._inductor.pattern_matcher.PatternMatcherPass` 的 pass 基类。`match_table: ClassVar[defaultdict[str,int]]` 跨 pass 累计匹配计数，`log_match_summary()` 总结；`dump_patterns` 把 pattern 反打印成伪 Python 供调试。
- `VllmPatternReplacement`（`vllm_inductor_pass.py:197`）：ABC + `Generic[P,R]`，定义 `pattern`/`replacement`/`get_inputs` 三抽象成员 + 常用 dtype 的 `empty_*` helper，是"pattern/replacement 对"的规范载体。
- `VllmFusionPatternMatcherPass`（`vllm_inductor_pass.py:296`）：用 `VllmPatternReplacement` 注册 pattern 的 pass 基类。子类在 `__init__` 调 `self.register(pr)`；`_trace_fn` 统一做 `view_to_reshape` + `_remove_noop_permute`；`__call__` 由 `@time_and_log` 包 `pm_pass.apply(graph)`。
- 工具函数：`fold_consecutive_reshapes`（`vllm_inductor_pass.py:258`）、`_remove_noop_permute`、`_fx_view_to_reshape`、`PrinterInductorPass`。
- `get_match_table()`（`vllm_inductor_pass.py:90`）：导出 `match_table` 快照，供测试/诊断。

## 为什么

- **统一计时与 dump**：每个 pass 想"跑前/跑后 dump graph + 计时 ms"，`time_and_log` 装饰器一处实现。`dump_prefix` 在 `PostGradPassManager` 中按 pass 序号递增，使 `lazy_format_graph_code` 产物在 depyf/tlparse 里有序可读。
- **精简配置隔离**：`InductorCompilationConfig` 只装 pass 真正需要的两个字段，避免把 `VllmConfig`（含 `model_config` 指针）塞进 pass——后者会被 Inductor 序列化进 cache 元数据，导致模型指针泄漏或 pickle 失败。
- **pattern 计数可观测**：`match_table` 按 pass 名累计匹配次数，`log_match_summary()` 在 `PostGradPassManager` 末尾打印，让"哪些 fusion 实际命中、命中多少"一目了然，是调参/验证 fusion 生效的主依据。
- **`VllmPatternReplacement` 规范 pattern 对**：把 `pattern`（要找的子图闭包）、`replacement`（替换闭包）、`get_inputs`（trace 用示例张量）三件事强制成 abstract 成员，新 fusion 只需实现三者，注册细节由基类统一处理。`empty_*` helper 提供常见 dtype 的未初始化张量。
- **trace_fn 统一清洗**：`_trace_fn` 在 pattern trace 后做 `view_to_reshape`（把 view 节点转 reshape 以匹配 Inductor 规范化后的图）+ `_remove_noop_permute` + `fold_consecutive_reshapes`，使 pattern 能匹配编译后图里多出的 reshape/permute，避免漏匹配。
- **uuid 含全部 pattern 类型**：`VllmFusionPatternMatcherPass.uuid` 用 `hash_source(type(self), *[type(pr) for pr in pattern_replacements])`，任一 pattern 实现变更→重编译。
- **`PrinterInductorPass` 调试**：仅 dump graph 不改图，用于开发期定位 pass 间图状态。

## 怎么做

### VllmInductorPass 时间装饰器

```python
@VllmInductorPass.time_and_log
def __call__(self, graph):
    self.matched_count = self.pm_pass.apply(graph)
    VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
```

`time_and_log`（`vllm_inductor_pass.py:60`）：`begin()` 记 ns → `dump_graph(graph,"before")` → 调被装饰函数 → `dump_graph(graph,"after")` → `end_and_log()` 输出 ms。

### 新 fusion pass 范式

```python
class MyFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config, pass_name):
        super().__init__(config, pass_name)
        self.register(MyPatternReplacement(...))

class MyPatternReplacement(VllmPatternReplacement):
    @property
    def pattern(self): ...       # 闭包定义待找子图
    @property
    def replacement(self): ...   # 闭包定义替换子图
    def get_inputs(self): ...    # 示例张量
```

`register`（`vllm_inductor_pass.py:308`，`@enable_fake_mode`）：`pm.register_replacement(pr.pattern, pr.replacement, pr.get_inputs(), self._trace_fn, self.pm_pass)`，并登记到 `_pattern_replacements` 供 uuid。

### dump_patterns

若 `compile_debug_dump_path()` 存在，`dump_patterns`（`vllm_inductor_pass.py:125`）把每个 `PatternMatcherPass.patterns` 用 `PatternPrettyPrinter` 打印成伪 Python（含 `auto_functionalized`/`vllm`/`vllm_ir` 命名空间），`_replace_op_overloads` 美化 `OpOverload` repr，便于人读与导航。

### match_table

`VllmPatternMatcherPass.match_table: ClassVar[defaultdict[str,int]]` 跨所有子类实例累计。`PostGradPassManager.__call__` 末尾调 `VllmPatternMatcherPass.log_match_summary()` 输出，`get_match_table()` 返回快照供测试断言。

## 与其它模块/系统配合

- [`inductor-pass.md`](inductor-pass.md)：本类继承 `InductorPass`，复用 `uuid`/`PassContext`。
- [`pass-manager.md`](pass-manager.md)：`PostGradPassManager` 递增 `dump_prefix`，末尾调 `log_match_summary`。
- [`fusion.md`](fusion.md)：几乎所有 fusion pass 继承 `VllmFusionPatternMatcherPass` 或 `VllmPatternMatcherPass`，用 `VllmPatternReplacement` 定义 pattern。
- [`ir.md`](ir.md)：`VllmIRLoweringPass` 继承 `VllmInductorPass` 但用 `register_graph_pattern` 而非 `VllmFusionPatternMatcherPass`（lowering 不是简单 pattern/replacement）。
- [`fx-utils.md`](fx-utils.md)：`is_func` 等 helper 被 pattern 实现与 `fold_consecutive_reshapes` 使用。
- [`配置-compilation`](../../10-config/README.md)：`compile_debug_dump_path` / `VLLM_PATTERN_MATCH_DEBUG`。
- [`可观测性`](../../16-observability/README.md)：`lazy_format_graph_code` 产物进 depyf/tlparse。

## 历史版本演进

- **v0.7**：`VllmInductorPass` + `VllmPatternMatcherPass` 引入，仅 `time_and_log` 与 `match_table`。
- **v0.8**：`InductorCompilationConfig` 抽出隔离配置；`dump_patterns` 反打印 pattern；`dump_prefix` 序号排序。
- **v0.9**：`VllmPatternReplacement` ABC + `VllmFusionPatternMatcherPass` 引入，新 fusion 范式定型；`_trace_fn` 统一 `view_to_reshape` + noop permute + `fold_consecutive_reshapes`。
- **v0.10 / main**：`get_match_table` 导出供测试；`DEVICE_TYPE` 从 `current_platform.device_type` 取，使 pattern helper 支持多平台 dtype。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [inductor-pass.md](inductor-pass.md) — 基类与 `PassContext`。
- [pass-manager.md](pass-manager.md) — `dump_prefix` 与 `log_match_summary` 调用点。
- [fusion.md](fusion.md) — `VllmPatternReplacement` 的具体实现者。
- [fx-utils.md](fx-utils.md) — pattern 实现用的 FX 工具。
