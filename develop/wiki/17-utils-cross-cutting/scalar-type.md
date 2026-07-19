# 标量类型（ScalarType）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 标量类型

本页覆盖 `vllm/scalar_type.py`（358 行），定义 vLLM 的子字节/带偏置标量类型系统 `ScalarType`，与 C++ 端 `csrc/core/scalar_type.hpp` 镜像。

## 是什么

### `ScalarType`（`vllm/scalar_type.py:23`，`@dataclass(frozen=True)`）

可表示多种浮点/整数类型，特别是 torch.dtype 不支持的**子字节类型**与**带偏置类型**。字段：

- `exponent: int`：浮点指数位（整数类型为 0）。
- `mantissa: int`：浮点尾数位，或整数类型中"不含符号位的位数"。
- `signed: bool`：是否有符号位。
- `bias: int`：编码偏置（`stored = value + bias`，如 GPTQ 4bit 用 bias=8）。
- `_finite_values_only: bool`：是否不含 inf（私有，用 `has_infs()`）。
- `nan_repr: NanRepr`：NaN 表示方式（`NONE`/`IEEE_754`/`EXTD_RANGE_MAX_MIN`，`vllm/scalar_type.py:13`）。

关键属性/方法：

- `size_bits`（`:167`）= `exponent + mantissa + signed`。
- `id`（`:136`，`cached_property`）：把字段位打包成 int64，作为传给 PyTorch custom op 的句柄；布局必须与 C++ `ScalarType::from_id` 同步，注册到 `_SCALAR_TYPES_ID_MAP`。
- `min()`/`max()`（`:170`/`:177`）：考虑 bias 后的表示范围；浮点走 `_floating_point_max_int()` 比特拼 double，整数走位移。
- `is_floating_point()`/`is_integer()`/`is_signed()`/`has_bias()`/`has_infs()`/`has_nans()`/`is_ieee_754()`。
- `__str__`（`:218`）：遵循 ml_dtypes 命名——浮点 `float<size>_e<e>m<m>[flags]`（`f`=finite-only、`n`=nan），整数 `[u]int<size>[b<bias>]`。

### 便利构造器（`:266` 起）

- `int_(size_bits, bias)`：有符号整数（`size_bits` 含符号位）。
- `uint(size_bits, bias)`：无符号整数。
- `float_IEEE754(exponent, mantissa)`：标准浮点。
- `float_(exponent, mantissa, finite_values_only, nan_repr)`：非标准浮点。
- `from_id(id)`：从打包 int 反查（需先构造过以注册 map）。

### 预定义实例（`scalar_types` 类，`:327`）

`int4`/`uint4`/`int8`/`uint8`、FP8 族（`float8_e4m3fn`/`float8_e5m2`/`float8_e8m0fnu`）、FP6（`float6_e3m2f`/`float6_e2m3f`）、FP4（`float4_e2m1f`）、GPTQ 偏置族（`uint2b2`/`uint3b4`/`uint4b8`/`uint5b16`/`uint6b32`/`uint7b64`/`uint8b128`）、俗名（`bfloat16`=`float16_e8m7`，`float16`=`float16_e5m10`）。

## 为什么

- **torch.dtype 不够**：FP4/FP6/带 bias 的 GPTQ 类型 PyTorch 原生不支持，量化内核需要精确的类型描述。
- **与 C++ 镜像**：自定义算子用 `int64 id` 传类型，避免跨语言传对象；两侧 `from_id` 必须一致，文件注释明确要求保持同步（`vllm/scalar_type.py:19`）。
- **命名一致**：跟随 `ml_dtypes`/`jax-ml` 与 OCP MX 规范，便于生态互通。
- **范围/NaN 语义精确**：`EXTD_RANGE_MAX_MIN` 用全 1 尾数扩展范围（无 NaN），`IEEE_754` 保留 NaN——因 NVIDIA Blackwall FP8 = e4m3fn 用扩展范围而非 IEEE。

## 怎么做

```python
from vllm.scalar_type import scalar_types, ScalarType
t = scalar_types.uint4b8        # GPTQ 4bit unsigned + bias 8
op(t.id, x)                     # 把 id 传给 custom op
s = ScalarType.float_IEEE754(5, 2)   # float8_e5m2
name = str(s)                   # "float8_e5m2"
```

- 新增类型：用便利构造器创建实例并挂到 `scalar_types`；`id` 会自动缓存注册。
- 跨语言：C++ 端用相同字段布局解析 `id`。

## 与其它模块/系统配合

- [模型执行 · 量化](../03-model-execution/README.md)：CompressedTensors/Marlin/GPTQ/AWQ/Fp8 等量化方案用 `ScalarType` 描述权重/激活类型。
- [custom-ops.md](custom-ops.md)：`_custom_ops.py` 的众多量化/反量化/GEMM 算子接收 `ScalarType.id`（如 `gptq_gemm`、`scaled_fp4_quant`）。
- [csrc 镜像](../18-build-ci-testing/README.md)：`csrc/core/scalar_type.hpp` 是 C++ 对偶，注释要求同步。
- [编译与 IR](../09-compilation-ir/README.md)：torch.compile 需把 `ScalarType` 当作稳定常量处理。

## 历史版本演进

- **v0.5–v0.6**：`ScalarType` 已存在，主要服务 Marlin/GPTQ 整数量化。
- **v0.7–v0.8**：随 FP8 量化落地，补 `float8_e4m3fn`/`float8_e5m2`；`id` 位布局稳定。
- **v0.9–v0.10**：MXFP4/NVFP4 兴起，引入 `float4_e2m1f`、`float8_e8m0fnu`（OCP MX e8m0 scale）、FP6 族；`has_nans`/`is_ieee_754` 语义细化。
- **v0.11–main**：随 DeepGEMM/CUTLASS FP4 GEMM 路径扩展使用面；`from_id`/`_SCALAR_TYPES_ID_MAP` 行为稳定（具体字段引入版本待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [custom-ops.md](custom-ops.md)
- [模型执行 · 量化](../03-model-execution/README.md)
- [构建/CI · csrc](../18-build-ci-testing/README.md)
