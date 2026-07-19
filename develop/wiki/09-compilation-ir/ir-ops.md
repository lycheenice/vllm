# 内置 IR 算子（ops/）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > IR Ops

源码：`vllm/ir/ops/__init__.py`、`vllm/ir/ops/layernorm.py`（约 72 行）

## 是什么

`vllm/ir/ops/` 是内置 IR 算子的实现目录，目前仅 `layernorm.py`，导出两个 op：

- `rms_norm`（`layernorm.py:10`，`@register_op`）：加权 RMS 归一化。
  - 签名：`rms_norm(x: Tensor, weight: Tensor|None, epsilon: float, variance_size: int|None=None) -> Tensor`。
  - native 实现：升 FP32 → 计算 `variance`（按 `variance_size` 截取）→ `rsqrt(variance+eps)` → 加权 → 回原 dtype。
  - `register_input_generator`：`_rms_norm_input_generator(num_tokens, hidden_size, dtype, epsilon=1e-5)`。
  - `override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)`：大 shape 归约舍入放宽。
- `fused_add_rms_norm`（`layernorm.py:39`，`@register_op(allow_inplace=True)`）：融合残差加 + RMS 归一化。
  - 签名：`fused_add_rms_norm(x: Tensor, x_residual: Tensor, weight: Tensor|None, epsilon: float, variance_size: int|None=None) -> tuple[Tensor, Tensor]`。
  - native 实现：`x = x + x_residual`（FP32）→ 归一化 → 返回 `(norm_out, residual_out)`。
  - `allow_inplace=True`：自带 `maybe_inplace` overload，activation `x`/`x_residual` 可被 donate 复用。
  - `override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)`。
  - `register_input_generator`：`_fused_add_rms_norm_input_generator`。

`__init__.py` 仅 `from .layernorm import fused_add_rms_norm, rms_norm`，`__all__ = ["rms_norm", "fused_add_rms_norm"]`。

## 为什么

- **norm 是 fusion 的中心枢纽**：`rms_norm`/`fused_add_rms_norm` 是 transformer 每层出口，下游紧跟量化（FP8/NVFP4）或 all_reduce。把它们做成 IR op，让 `RMSNormQuantFusionPass`/`AllReduceFusionPass`/`SequenceParallelismPass` 都在 IR 层匹配 pattern，lowering 时再落 native 或 `torch.ops._C.rms_norm_static_fp8_quant` 等 fused kernel。
- **`fused_add_rms_norm` 的 inplace 价值**：残差加天然适合 inplace（`x += residual`），`allow_inplace=True` 使 fusion pattern 能用 `maybe_inplace` 声明 donate 意图，`VllmIRInplaceFunctionalizationPass` 标 `donated_input_ids` 后，`UnsafeCloneEliminationPass` 可消 clone、provider inplace 实现可直接复用 activation 显存。
- **`variance_size` 参数**：支持按部分维度算方差（如 MLA 的 `variance_size`），使同一 IR op 服务多种 norm 变体，避免为每变体新建 op。
- **native 即 reference**：native 实现是纯 PyTorch（升 FP32 算），作为 provider 实现的数值基准，配合 [`tolerances.md`](tolerances.md) 自动验证 provider。
- **input_generator 服务 CI**：注册 input generator 使一致性测试可参数化 `(num_tokens, hidden_size, dtype, epsilon)`，跨 dtype 跑 native vs provider。
- **float16 容差放宽**：大 hidden_size（如 32768×16384）归约累计舍入使个别元素超默认 1e-3，per-op override 到 1e-2/2e-3 避免误报。

## 怎么做

### 在 fusion pattern 中使用

```python
# rms_quant_fusion.py 的 pattern
def pattern(input, weight, scale):
    result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
    return self.quant_matcher(result_rms, scale)[0]

# allreduce_rms_fusion.py 的 pattern（用 maybe_inplace）
result, residual = vllm.ir.ops.fused_add_rms_norm.maybe_inplace(
    input, residual, weight, self.epsilon)
```

### provider 注册（示意）

```python
@rms_norm.register_impl("cuda", supported=torch.cuda.is_available(),
                        supports_args=lambda x, w, eps, vs=None: x.is_cuda)
def _cuda(x, weight, epsilon, variance_size=None):
    return torch.ops._C.rms_norm(x, weight, epsilon)

rms_norm.set_default(["cuda", "native"])
```

### lowering

`VllmIRLoweringPass` 匹配 `torch.ops.vllm_ir.rms_norm.default` 节点 → `rms_norm.dispatch(*fake_args)` 选 provider → `replace_by_example(impl.func_impl_fn, ...)`。lowering 后图中 `vllm_ir.rms_norm` 节点消失，替换为 provider 实现子图（或 native 的 aten 算子序列）。

### native 实现要点

- 升 FP32 算方差与 rsqrt，避免低精度数值问题。
- `variance_size is None` 用全尾维，否则 `x[..., :variance_size]`。
- `weight is not None` 才加权，否则只归一化。
- 回原 dtype（`x.to(orig_dtype)`）。

## 与其它模块/系统配合

- [`ir-README.md`](ir-README.md) / [`ir-op.md`](ir-op.md)：`@register_op`/`@register_impl`/`IrOp` 体系。
- [`tolerances.md`](tolerances.md)：float16 override。
- [`passes/fusion.md`](passes/fusion.md)：`RMSNormQuantFusionPass`/`AllReduceFusionPass`/`SequenceParallelismPass`/ROCm aiter norm fusion 的 pattern 都用这两个 op。
- [`passes/ir.md`](passes/ir.md)：`VllmIRLoweringPass` 下沉、`VllmIRInplaceFunctionalizationPass` functionalize `fused_add_rms_norm.maybe_inplace`。
- [`模型执行-custom_op`](../03-model-execution/layers/custom-op.md)：provider 实现调 `torch.ops._C.rms_norm`/`fused_add_rms_norm`/`rms_norm_static_fp8_quant` 等。
- [`注意力`](../05-attention/README.md)：MLA 的 `variance_size` 用法（待核实具体模型）。

## 历史版本演进

- **v0.8**：`rms_norm`/`fused_add_rms_norm` 作为首批 IR op 引入；`allow_inplace=True` on `fused_add_rms_norm`；float16 override。
- **v0.9**：`variance_size` 参数加入支持变体；`register_input_generator` 接入。
- **v0.10 / main**：被多个 fusion（rms_quant/allreduce_rms/sp/rocm_aiter）pattern 共用；provider 实现随 `torch.ops._C.*` kernel 扩充。具体版本归属（部分待核实，后续 IR op如 `silu_and_mul`/`rotary_embedding` 是否计划迁入 `vllm/ir/ops/` 待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [ir-README.md](ir-README.md) — IR 库总览。
- [ir-op.md](ir-op.md) — `register_op`/`allow_inplace`/`maybe_inplace` 机制。
- [tolerances.md](tolerances.md) — float16 override 来源。
- [passes/fusion.md](passes/fusion.md) — 消费这两个 op 的 fusion pass。
- [passes/ir.md](passes/ir.md) — lowering 与 functionalization。
