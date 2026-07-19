# 归一化层（norm）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > 归一化层

本页整合 vLLM 层库中所有与归一化相关的实现，主体在 `vllm/model_executor/layers/layernorm.py`，并与 `fused_allreduce_gemma_rms_norm.py`、`minimax_rms_norm/` 子目录、`fla/ops/layernorm_guard.py`、`mamba/mamba_mixer2.py` 中的局部 RMSNorm 联动。

> 备注：任务文档中 "norm/RMSNorm/Serialized…" 里的 "Serialized" 在当前代码库中未发现对应类名 `(待核实)`，疑为早期命名或外部文档残留。本页以现行源码为准。

## 是什么

| 类/函数 | 文件 | 角色 |
|---|---|---|
| `RMSNorm` | `layernorm.py:37` | 标准 RMSNorm，注册为 `CustomOp` 命名 `rms_norm`；支持 `residual` 走 fused add+norm |
| `GemmaRMSNorm` | `layernorm.py:129` | Gemma 风格：`x * (1 + w)` 且 `(x*w).to(orig_dtype)` |
| `RMSNormGated` | `layernorm.py:169` | 带 gate 与可选 group norm 的 RMSNorm，FLA 模型用 |
| `LayerNorm` | `layernorm.py:305` | 标准层归一化（fp32 内部计算），少数模型使用 |
| `poly_norm` | `layernorm.py:19` | 多项式归一化算子（`ops.poly_norm`），特殊模型路径 `(待核实)` |
| `fused_allreduce_gemma_rms_norm` | `fused_allreduce_gemma_rms_norm.py:103` | TP all-reduce + residual-add + GemmaRMSNorm 三合一融合内核（FlashInfer） |
| `RMSNormTP` | `minimax_rms_norm/rms_norm_tp.py` | MiniMax-M3 的跨 rank 方差 all-reduce RMSNorm（带 Lamport 内核） |
| `rmsnorm_fn` | `fla/ops/layernorm_guard.py` | FLA 路径的 RMSNorm+gate 调用入口（被 `RMSNormGated.forward_cuda` 调用） |

## 为什么

归一化层是 transformer block 中除线性层外最频繁调用的算子，把它工程化的动机包括：

1. **融合 add+norm**：`RMSNorm.forward(x, residual)` 在有 `residual` 时走 `ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, ...)`（`layernorm.py:88-94`），把 residual 加法和归一化合并为一次 kernel launch，省一次访存；这是解码路径性能关键。
2. **TP all-reduce 融合**：`fused_allreduce_gemma_rms_norm.py` 在 `RowParallelLinear` 之后立刻 all-reduce + add residual + RMSNorm，借助 FlashInfer 的 `kARResidualRMSNorm` 单 kernel 完成（`fused_allreduce_gemma_rms_norm.py:103-139`）。MiniMax-M3 的 `RMSNormTP` 把方差的 all-reduce 与归一化合并（Lamport fused kernel，`minimax_rms_norm/rms_norm_tp.py`）。
3. **IR 内联**：`RMSNorm.forward_native` 已切换到 `vllm.ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm`（`layernorm.py:81-94`），让 IR priority 机制在 `torch.compile` 时把 RMSNorm 内联到周围图里，与 QuantFP8、all-reduce 等做 fusion pattern。
4. **权重可选**：`has_weight=False` 路径（`layernorm.py:62-72`）支持部分模型无 weight 的 RMSNorm；`pass_weight`/`pass_weight_add` 控制是否把 weight 传给底层内核，对不支持 `weight=None` 的实现走 IR op fallback。
5. **batch-invariant 路径**：`VLLM_BATCH_INVARIANT=1` 时 `forward_cuda` 走 `rms_norm_batch_invariant`（`batch_invariant.py`），避免大批量时反复重算。
6. **Group / Gated 变体**：`RMSNormGated` 支持 `group_size` 做 GroupNorm、`norm_before_gate` 决定先 norm 还是先 gate，服务于 FLA 系列与 Mamba2 GDN。

## 怎么做

### `RMSNorm`

`layernorm.py:46-124`：

- `__init__` 持有 `hidden_size`、`variance_epsilon`、可选 `var_hidden_size`（variance 维度覆盖，部分模型用）、`has_weight`、`weight_dtype`。
- 前向：
  - `forward_native(x, residual=None)` 调 `ir.ops.rms_norm(x, weight, eps, var_hidden_size)` 或 `ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, eps, var_hidden_size)`。
  - `forward_cuda(x, residual=None)`：若 `VLLM_BATCH_INVARIANT` 则 `rms_norm_batch_invariant(x, weight, eps, residual=residual)`，否则 `self.forward_native(x, residual)`。
  - `forward_xpu` 转发到 `forward_cuda`。
- `variance_size_override` 只在 `var_hidden_size != hidden_size` 时启用，断言 batch-invariant 路径不支持 override（`layernorm.py:101-104`）。

### `GemmaRMSNorm`

`layernorm.py:129-164`：与 `RMSNorm` 的两处差异：

- weight 初始化为 `torch.zeros` 而非 `torch.ones`，前向计算 `weight = self.weight.float() + 1.0`。
- 保留 `(x * w).to(orig_dtype)` 的精度策略。`forward_cuda` 直接转发到 `forward_native`，因此 `GemmaRMSNorm` 与 IR 体系一致。

### `RMSNormGated`

`layernorm.py:169-302`：实现 `out = rms_norm(x) * act(z)` 或 `rms_norm(x * act(z))` 两种模式，可选 group norm。

- 核心逻辑在 `forward_static`（`layernorm.py:218-266`）——静态方法，被 `forward_native` 与编译期 pattern matcher（`MatcherRMSNormGated`）共享同一实现。
- `forward_cuda` 走 `vllm.model_executor.layers.fla.ops.layernorm_guard.rmsnorm_fn`（`layernorm.py:283-297`），把 `weight/bias/z/eps/group_size/norm_before_gate/activation` 一并传入 FLA 内核。
- 支持 `silu`/`sigmoid`/`swish` 三种激活；`group_size=None` 等价于 1 个大组。

### `LayerNorm`

`layernorm.py:305-320`：标准 PyTorch `F.layer_norm`，但内部以 `float32` 计算后再 `type_as(x)`。用于早期模型（如 BART、T5 等仍用 LayerNorm 而非 RMSNorm 的实现）。

### `fused_allreduce_gemma_rms_norm`

`fused_allreduce_gemma_rms_norm.py:103-143`：被模型代码（如 Gemma2/3 的 attention 后）显式调用而非通过 `nn.Module` 调用。流程：

1. `tp_size == 1`：直接 `return norm(hidden_states, residual)`，等价于不融合。
2. `_can_use_flashinfer(hidden_states, tp_size)`：检查 FlashInfer 可用、`bf16/fp16`、`is_cuda`、`is_contiguous`、`dim==2`、token 数 ≤ workspace 上限、NVSwitch 可用。
3. 满足条件则调 `flashinfer_trtllm_fused_allreduce_norm(..., pattern_code=kARResidualRMSNorm, norm_out=norm_out, ...)`，把 all-reduce 后的 hidden 写回 `hidden_states` buffer 当作 new residual，归一化结果写到 `norm_out`。
4. 不满足则 fallback 到 `tensor_model_parallel_all_reduce(hidden_states)` + `norm(reduced, residual)`，数值等价于不融合的模型原路径。

### `RMSNormTP`（MiniMax-M3）

`minimax_rms_norm/rms_norm_tp.py`：MiniMax-M3 QK-norm 需要跨 rank 同步方差。`_all_reduce_variance(var)` 在 fp32 里 all-reduce per-token 方差（`rms_norm_tp.py:29-40`），并说明为何要 flatten 成 1D 走 pynccl/custom-AR 而非 FlashInfer fp16-bias workspace。Lamport fused kernel `minimax_allreduce_rms_qk` 在 token 数 ≤ `MINIMAX_QK_NORM_MAX_TOKEN_NUM=2048` 时启用，否则回到 eager 路径。

## 与其它模块/系统配合

- [linear.md](linear.md)：`RowParallelLinear` 的 `skip_bias_add` 让 bias 在 RMSNorm 之前合并；`fused_allreduce_gemma_rms_norm` 直接消费 `RowParallelLinear` 的未 all-reduce 输出。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`vllm.ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` 被 RMSNorm 调用；`compilation/passes/fusion/allreduce_rms_fusion.py` 与 `rms_quant_fusion.py` 用 pattern match 把 RMSNorm 与上游 all-reduce 或下游量化合并；`RMSNormGated` 的 `forward_static` 由 `MatcherRMSNormGated` 复用。
- [custom-op.md](custom-op.md)：`RMSNorm`/`GemmaRMSNorm`/`RMSNormGated` 都注册为 `CustomOp`，可由 `compilation_config.custom_ops` 全局开关；`enabled()=False` 时走 `forward_native`+torch.compile。
- [mamba-ssm.md](mamba-ssm.md)：Mamba2 GDN（`mamba/gdn/`）和 `mamba_mixer2` 内部使用 `RMSNorm`；`RMSNormGated` 的 `forward_cuda` 走 FLA 的 `layernorm_guard.rmsnorm_fn`。
- [attention #05](../../05-attention/README.md)：Gemma 的 attention 输出经 `fused_allreduce_gemma_rms_norm` 直接进 next block；MiniMax-M3 QK-norm 与 sparse-attn indexer 协同。
- [distributed #07](../../07-distributed/README.md)：`tensor_model_parallel_all_reduce`、`get_tp_group()`；`fused_allreduce_gemma_rms_norm` 与 `compilation/passes/fusion/allreduce_rms_fusion.py` 共享 FlashInfer workspace（懒构造 + 全局缓存）。
- [model-zoo #04](../../04-model-zoo/README.md)：模型 `__init__` 里大量出现 `RMSNorm(hidden_size, eps=config.rms_norm_eps)`；Gemma 系列使用 `GemmaRMSNorm`；FLA/Mamba 系列使用 `RMSNormGated`。

## 历史版本演进

- **早期**：`RMSNorm` 仅 `forward_cuda`（调 `_custom_ops.rms_norm`），无 residual 融合、无 IR。
- **v0.6–v0.7**：`fused_add_rms_norm` kernel 引入支持 residual 路径；`GemmaRMSNorm` 与 Gemma 模型一起加入。
- **v0.7–v0.8**：`RMSNormGated` 引入支持 Mamba/FLA；后续补 group_size、`norm_before_gate` 与 FLA `rmsnorm_fn` 内核。
- **v0.8**：`CustomOp.register("rms_norm")` 与 `compilation_config.custom_ops` 联动，把 RMSNorm 纳入全局开关。
- **v0.9–v0.10**：`fused_allreduce_gemma_rms_norm.py` 引入服务 Gemma2/3，把 all-reduce + residual + GemmaRMSNorm 三合一；FlashInfer workspace 全局缓存机制成熟。
- **v0.10–v0.11**：`RMSNorm.forward_native` 切换到 `vllm.ir.ops.rms_norm` / `fused_add_rms_norm`，与 IR 优先级机制协作（`layernorm.py:81-94`）；`has_weight=False` 路径加入，部分 IR op（如 `oink`）不支持 `weight=None` 时通过 priority fallback。
- **v0.11–v0.12**：`minimax_rms_norm/` 子目录引入支持 MiniMax-M3 的 TP-QK-norm（Lamport fused AR+RMS kernel）；`RMSNormGated.forward_static` 抽出供 pattern matcher 共享。`VLLM_BATCH_INVARIANT` 与 `variance_size_override` 的冲突显式断言（`layernorm.py:101-104`）。
- **v0.12 / main**：`poly_norm` 算子引入 `(待核实)`，疑为特定模型路径专用；`pass_weight` / `pass_weight_add` 控制 weight=None 透传逻辑稳定；FLA `rmsnorm_fn` 路径成熟。

[← 返回层库首页](../README.md)

## 参见

- [custom-op.md](custom-op.md)：`CustomOp.register("rms_norm")` 与 dispatch 机制。
- [compilation-ir #09](../../09-compilation-ir/README.md)：IR ops 与 fusion pass。
- [distributed #07](../../07-distributed/README.md)：`tensor_model_parallel_all_reduce` 与 FlashInfer workspace 缓存。
- [mamba-ssm.md](mamba-ssm.md)：Mamba2 GDN 与 `RMSNormGated` 的联动。
