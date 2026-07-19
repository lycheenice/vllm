# CUTE / CUTLASS DSL 工具（cute_utils）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > CUTE 工具

本页覆盖 `vllm/cute_utils/`，描述基于 CUTLASS Cute DSL（Python 端 `cutlass.cute`）构建的底层 GPU 算子辅助代码，服务于 SM90+（Hopper/Blackwell）TMA / tcgen05 等硬件特性。

## 是什么

目录文件：

- `__init__.py`（148 行）：通用 CUTE 辅助算子。
- `_tcgen05.py`（223 行）：tcgen05（Blackwell Tensor Core 第 5 代）相关原语。
- `cvt.py`（145 行）：类型/布局转换辅助。

### `__init__.py` 内容

- 顶部从 `cutlass`/`cutlass._mlir`/`cutlass.cute.nvgpu.cpasync`/`cutlass.cutlass_dsl` 导入，依赖 CUTLASS v4.3+。定义 TMA cache policy 常量（`vllm/cute_utils/__init__.py:10`）：`EVICT_NORMAL`、`EVICT_FIRST`、`EVICT_LAST`（与 `cute/arch/copy_sm90_desc.hpp` 对应的位模式）。
- `simple_tma_copy(atom, src, dst, mbar, cache_policy)`（`:20`）：G2S/S2G 两种 TMA 拷贝的简化 wrapper，自动按 atom.op 类型判定方向，组合 `tma_partition` + `cute.copy`，附带 mbarrier/cache_policy。
- `mma_bf16`（`:66`，`@dsl_user_op`）：bf16 输入的 m16n8k16 MMA，内部把 bf16 recast 成 `Uint32` 走 inline PTX `mma.sync.aligned...f32.bf16...`。
- 一族 `_bf16x2_*`（`:98` 起）：`abs`/`neg`/`max`/`mul`/`sub` 等基于 `bf16x2` PTX 指令的双宽运算，`@dsl_user_op` 装饰，输出 `Uint32`。
- `recast_val`、`fence_before_tma_store`（`:51`）：bitcast 与 TMA store 前的 `fence.proxy.async::generic.release...` 代理栅栏内联汇编。

### `_tcgen05.py` / `cvt.py`

- `_tcgen05.py`：封装 tcgen05 MMA scale factor 与权重布局相关原语（Blackwell NVFP4/MXFP4 路径用）。
- `cvt.py`：CUTE 张量在 layout/dtype 间的转换辅助。

## 为什么

- **CUTLASS DSL 表达力补充**：TMA、tcgen05、cluster-scale barrier 等硬件特性用纯 Triton 难以精确控制，CUTE DSL 直接映射到 PTX/MLIR，性能与可控性更高。
- **NVFP4/MXFP4 路径刚需**：Blackwell FP4 量化的 scale 布局（128×4 swizzle、tcgen05 scale）依赖 CUTE 表达。
- **集中封装**：把这些"底层内联汇编 + MLIR"集中在一处，便于随 CUTLASS 版本升级统一调整，避免散落各模型。

## 怎么做

- 这些函数主要被 [模型执行 · 内核](../03-model-execution/README.md) 与 [custom-ops.md](custom-ops.md) 中的 CUTLASS 路径调用，而非业务层直接使用。
- 新增 CUTE 算子：用 `@dsl_user_op` 装饰，通过 `T`/`cute.TensorSSA` 表达类型，必要时 `llvm.inline_asm` 注入 PTX。
- 依赖：需要安装 `cutlass`（Python DSL），否则相关 import 失败——目前不通过占位符兜底（待核实是否有 lazy import 守卫）。

## 与其它模块/系统配合

- [模型执行 · 量化/内核](../03-model-execution/README.md)：NVFP4/MXFP4 GEMM、tcgen05 MMA 消费。
- [custom-ops.md](custom-ops.md)：CUTLASS scaled_mm/fp4 路径与 CUTE 辅助配合。
- [scalar-type.md](scalar-type.md)：FP4 类型描述与 CUTE scale 布局互补。
- [平台](../08-platforms/README.md)：仅 SM90+ 平台启用。

## 历史版本演进

- **v0.9–v0.10**：随 Hopper TMA 集成，`cute_utils/__init__.py` 的 `simple_tma_copy`/cache policy 常量就位。
- **v0.11**：Blackwell tcgen05/NVFP4 路径引入 `_tcgen05.py`；`mma_bf16`/`_bf16x2_*` 族加入。
- **main**：`cvt.py` 等布局转换辅助持续扩充。整体为较新子系统，具体版本（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [custom-ops.md](custom-ops.md)、[scalar-type.md](scalar-type.md)
- [模型执行 · 内核](../03-model-execution/README.md)
- [平台子系统](../08-platforms/README.md)
