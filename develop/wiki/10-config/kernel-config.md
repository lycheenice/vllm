# KernelConfig + IrOpPriorityConfig（kernel.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > KernelConfig

源码：`vllm/config/kernel.py`（约 295 行）。`KernelConfig` 描述内核选择与 warmup 行为：vLLM IR op 优先级、MoE/linear GEMM 后端、FlashInfer autotune、CuTeDSL warmup。`IrOpPriorityConfig` 是其内嵌子配置，控制 IR op 的 dispatch 优先级表。它是 `VllmConfig.kernel_config`，被 `vllm/ir/`、`vllm/model_executor/layers/` 与 warmup 模块消费。

## 是什么

### `IrOpPriorityConfig`（`kernel.py:19`）

控制 vLLM IR op 在 forward 中的 dispatch/lowering 优先级。每个字段是一个 `list[str]`（也接受逗号串），在 worker init 时经 `vllm.ir.ops.<op_name>.set_default()` 安装。

| 字段 | 默认 | 含义 |
|---|---|---|
| `rms_norm` | `[]` | `vllm.ir.ops.rms_norm` 优先级 |
| `fused_add_rms_norm` | `[]` | `vllm.ir.ops.fused_add_rms_norm` 优先级 |

> 当前仅这两个 op；后续会随 vLLM IR 扩充增加字段（`with_default(default, **kwargs)` 辅助构造）。

方法：
- `set_default()`：永久设 IR op 优先级。
- `set_priority()`（contextmanager）：临时设优先级（含 `current_platform.import_ir_kernels()` 确保实现可用）。
- `_iter_op_priorities()`：导入平台 IR 内核 + 校验每项后 yield `(IrOp, priority_list)`。
- `compute_hash()`：除字段哈希外，显式把每个 op 的实现 `uuid()` 纳入 `factors["_impls"]`（因实现被 Dynamo 隐藏，不会出现在 trace 文件列表，须手动入指纹）。

### `KernelConfig`（`kernel.py:165`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `ir_op_priority` | `IrOpPriorityConfig()` | IR op 优先级（平台默认在 `set_platform_defaults` 追加） |
| `enable_flashinfer_autotune` | `None`(三态) | warmup 时跑 FlashInfer autotune |
| `enable_cutedsl_warmup` | `True` | warmup 时跑 CuTeDSL 编译 warmup |
| `moe_backend` | `"auto"` | MoE 专家计算内核：`auto`/`triton`/`deep_gemm(_mega_moe)`/`cutlass`/`flashinfer_*`/`marlin`/`humming`/`triton_unfused`/`aiter`/`flydsl`/`hpc`/`emulation` 等 |
| `linear_backend` | `"auto"` | 量化线性 GEMM 内核：`auto`/`cutlass`/`flashinfer_*`/`marlin`/`triton`/`deep_gemm`/`torch`/`aiter`/`machete`/`fbgemm`/`conch`/`exllama`/`emulation`/`xpu(_woq)` 等 |

校验器：`_normalize_moe_backend`/`_normalize_linear_backend`（lowercase + `-`→`_`）；`enable_flashinfer_autotune`/`enable_cutedsl_warmup` 用 `_skip_none_validation` wrap 支持延迟初始化。

方法：`set_platform_defaults(vllm_config)`——调 `current_platform.get_default_ir_op_priority(vllm_config)` 取平台默认 IrOpPriorityConfig，对每个 op：当前为 `None` 则赋值，否则**追加**（去重，幂等因 `set_platform_defaults` 可能被多次调）。

`compute_hash()`：排除 `enable_cutedsl_warmup`/`enable_flashinfer_autotune`（warmup 行为不影响图）与 `ir_op_priority`（单独算），其余（`moe_backend`/`linear_backend`）+ `ir_op_priority.compute_hash()` 纳入。

## 为什么

- **IR op 优先级 = dispatch 真相源**：vLLM IR 把 `rms_norm`/`fused_add_rms_norm` 等 op 抽象为可多实现的 dispatch 点。`IrOpPriorityConfig` 让平台/用户指定优先实现顺序（如 `["aiter","native"]`），`set_priority()` contextmanager 让 warmup/编译在不同优先级下 trace 产生不同图。
- **实现 uuid 入哈希**：IR op 实现是 `@register` 的对象，被 Dynamo 隐藏，不出现在 trace 文件列表。`compute_hash` 显式 `IrOp.registry[name].impls[provider].uuid()` 纳入，确保换实现版本时编译缓存失效。
- **平台默认追加而非覆盖**：用户显式 `ir_op_priority.rms_norm=["native"]` 时，平台默认（如 `["aiter"]`）**追加**为 `["native","aiter"]`，保留用户首选同时让平台实现可 fallback。
- **MoE/linear backend 字符串化**：`moe_backend`/`linear_backend` 是高层选择，底层各 layer 按 quant 方法 + 硬件再 dispatch。`_normalize_*` 容忍大小写/连字符。
- **warmup 与图解耦**：`enable_flashinfer_autotune`/`enable_cutedsl_warmup` 是 warmup 期行为，不影响编译图形状，故排除出哈希。

## 怎么做

- **IR op 优先级**：`--kernel-config.ir-op-priority.rms-norm='["aiter","native"]'`（逗号串也行）。
- **MoE 后端**：`--moe-backend flashinfer_cutlass` 或 `--moe-backend deep_gemm_mega_moe`。
- **线性后端**：`--linear-backend cutlass`。
- **autotune**：`-O1` 起自动开 `enable_flashinfer_autotune`（`OPTIMIZATION_LEVEL_01` dict）；`-O0` 关；手动 `--kernel-config.enable-flashinfer-autotune=true`。
- **平台默认**：`VllmConfig.__post_init` 在 `kernel_config.set_platform_defaults(self)` 调用，自动追加平台 IR op 优先级。

## 与其它模块/系统配合

- **vLLM IR（[`09-compilation-ir/`](../09-compilation-ir/README.md) 与 `vllm/ir/`）**：`IrOpPriorityConfig.set_default`/`set_priority` 安装到 `IrOp.registry`；Dynamo trace 时按当前优先级选实现，编译缓存键含实现 uuid。
- **编译（[compilation-config.md](compilation-config.md)）**：`OPTIMIZATION_LEVEL_*` dict 的 `enable_norm_fusion`/`enable_act_fusion` 等 callable 会查 `ir_op_priority.rms_norm[0] != "native"` 决定融合；`enable_norm_pad_fusion` 查 `fused_add_rms_norm[0] == "aiter"`。
- **Model 执行（[`03-model-execution/`](../03-model-execution/README.md)）**：`moe_backend`/`linear_backend` 驱动 `FusedMoE`/`LinearBase` 层的内核选择；`enable_flashinfer_autotune`/`enable_cutedsl_warmup` 驱动 warmup 模块（[`03-model-execution/warmup.md`](../03-model-execution/warmup.md)）。
- **平台（[`08-platforms/`](../08-platforms/README.md)）**：`current_platform.get_default_ir_op_priority(vllm_config)` 与 `import_ir_kernels()` 提供平台 IR 实现；`set_platform_defaults` 不覆盖用户优先级。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init` 在优化级别展开前调 `set_platform_defaults`，之后 `_apply_optimization_level_defaults` 填 `enable_flashinfer_autotune`（若仍 `None` 则 raise，确保优化级别已决策）。

## 历史版本演进

- **v0.5–v0.7**：无独立 `KernelConfig`；内核选择散在 `CompilationConfig.custom_ops` 与各 layer 的硬编码。
- **v0.8.x（vLLM IR 成型期，待核实）**：`IrOpPriorityConfig` 与 `KernelConfig` 引入；`rms_norm`/`fused_add_rms_norm` IR op 上线；平台默认追加机制。
- **v0.9**：`moe_backend`/`linear_backend` 取值大幅扩充（`deep_gemm_mega_moe`/`flashinfer_cutedsl`/`flashinfer_b12x`/`humming`/`flydsl`/`hpc`）；`enable_flashinfer_autotune` 接入优化级别 dict。
- **v0.10**：`enable_cutedsl_warmup`；`compute_hash` 显式 IR 实现 uuid；`set_platform_defaults` 幂等性处理。
- **v0.11 / v0.12 / main**：IR op 字段随 vLLM IR 扩充（更多 op 待加入）；NVFP4/MXFP8 backend；`xpu`/`xpu_woq` linear backend；SM12x `flashinfer_b12x`。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `set_platform_defaults` 与优化级别展开时序。
- [compilation-config.md](compilation-config.md) — `pass_config.fuse_*` callable 查 `ir_op_priority`。
- [../09-compilation-ir/](../09-compilation-ir/README.md) — IR op 实现与 dispatch 消费方。
- [../03-model-execution/warmup.md](../03-model-execution/warmup.md) — autotune/cutedsl warmup 消费方。
