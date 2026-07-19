# 数值容差（tolerances.py）

[← Wiki 首页](../README.md) > [编译与 IR](../README.md) > Tolerances

源码：`vllm/ir/tolerances.py`（约 36 行）

## 是什么

`tolerances.py` 定义 vLLM IR op 的"实现数值比对默认容差"`DEFAULT_TOLERANCES`，配合 `IrOp.override_tolerance` / `IrOp.get_tolerance`（[`ir-op.md`](ir-op.md)），为"native reference 实现 vs provider 实现"的一致性测试提供按 dtype 的 `atol`/`rtol`。

```python
ToleranceSpec = dict[torch.dtype, dict[str, float]]

DEFAULT_TOLERANCES: ToleranceSpec = {
    torch.float64:         {"atol": 1e-8,  "rtol": 1e-8},
    torch.float32:         {"atol": 1e-5,  "rtol": 1.3e-6},
    torch.float16:         {"atol": 1e-3,  "rtol": 1e-3},
    torch.bfloat16:        {"atol": 1e-3,  "rtol": 1.6e-2},
    torch.float8_e4m3fn:   {"atol": 1e-1,  "rtol": 1e-1},
    torch.float8_e5m2:     {"atol": 2e-1,  "rtol": 2e-1},
    torch.float4_e2m1fn_x2:{"atol": 3e-1,  "rtol": 3e-1},
    torch.int8:            {"atol": 1,     "rtol": 0},
}
```

每个条目注释了 mantissa 位数与来源（PyTorch 参考默认、vLLM kernel测试、fp4 测试等）。

## 为什么

- **低精度比对需要按 dtype 区分**：FP8/FP4 的 machine epsilon 远大于 FP32，普通 1e-5 容差会误报。`DEFAULT_TOLERANCES` 按 mantissa 位数粗略递增收敛，避免 provider 实现因合理舍入被误判失败。
- **per-op 覆写**：`rms_norm`/`fused_add_rms_norm` 在大 shape 下归约累计舍入，`override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)` 放宽（`ops/layernorm.py:36`）；其它 op 仍用默认。`get_tolerance(dtype)` 优先返回 override，其次 DEFAULT_TOLERANCES，未定义则报错（`op.py:469`）。
- **int8 特例**：`rtol=0` 因相对误差对小整数无意义；`atol=1` 容忍 off-by-one 舍入。
- **`float4_e2m1fn_x2`**：packed pair（x2）格式，容差源自 vLLM nvfp4 测试（`silu_mul_nvfp4_quant`）。
- **provider 实现可自动验证**：`register_input_generator` 产输入 → 跑 native 与 provider → `torch.testing.assert_close(atol=..., rtol=...)` 用 `get_tolerance`，实现 CI 自动 gate provider 接入。

## 怎么做

```python
# ir/op.py:464
def override_tolerance(self, dtype, *, atol, rtol):
    self._tolerance_overrides[dtype] = {"atol": atol, "rtol": rtol}

def get_tolerance(self, dtype):
    if dtype in self._tolerance_overrides: return self._tolerance_overrides[dtype]
    if dtype in DEFAULT_TOLERANCES: return DEFAULT_TOLERANCES[dtype]
    raise ValueError(f"No tolerance defined for dtype {dtype} in op '{self.name}'")
```

测试用法（示意）：`tol = rms_norm.get_tolerance(torch.bfloat16)` → `assert_close(native_out, provider_out, atol=tol["atol"], rtol=tol["rtol"])`。

## 与其它模块/系统配合

- [`ir-op.md`](ir-op.md)：`IrOp._tolerance_overrides` / `override_tolerance` / `get_tolerance`。
- [`ir-ops.md`](ir-ops.md)：`rms_norm`/`fused_add_rms_norm` 对 float16 覆写。
- [`ir-README.md`](ir-README.md)：`register_input_generator` + 容差构成测试体系。
- [`平台`](../08-platforms/README.md)：provider 实现的一致性测试用本容差。

## 历史版本演进

- **v0.8**：`DEFAULT_TOLERANCES` 引入，覆盖 float32/16/bf16/fp8_e4m3fn；`override_tolerance`/`get_tolerance`。
- **v0.9**：扩充 `float8_e5m2`、`float4_e2m1fn_x2`、`int8`；per-op override（rms_norm/fused_add_rms_norm float16 放宽）。
- **v0.10 / main**：容差值随 nvfp4/group quant kernel 接入微调；注释补充来源。具体版本归属（部分待核实）。

[← 返回编译与 IR 首页](../README.md)

## 参见

- [ir-op.md](ir-op.md) — `override_tolerance`/`get_tolerance` 方法。
- [ir-ops.md](ir-ops.md) — `rms_norm`/`fused_add_rms_norm` 的 float16 覆写。
- [ir-README.md](ir-README.md) — 测试体系。
