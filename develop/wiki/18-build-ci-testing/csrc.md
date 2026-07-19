# csrc/ 原生 C/C++/CUDA 内核

[← Wiki 首页](../README.md) > [构建/CI/测试](../README.md) > csrc

## 是什么

`csrc/` 是 vLLM 的原生 C/C++/CUDA 源码树，承担最热路径上 PyTorch 无法高效表达或为了与底层库（CUTLASS、FlashAttention、cuDNN、NCCL、cuMemcpy 等）直连而手写的算子。它通过 `torch_bindings.cpp`（PyTorch C++ extension 入口 + `pybind`）注册为 `vllm._C` 之类的扩展模块；构建由 `cmake.md` 中的 `CMakeLists.txt` 与 `setup.py` 联动驱动。

主要子树：
- `attention/` — PagedAttention / FlashAttention 相关内核源（含 `vllm_flash_attn/` 经由 `vllm/vllm_flash_attn/` Python 包装）。
- `moe/` — MoE 路由 + 专家 GEMM（含 DeepGEMM 风格内核）。
- `quantization/` — FP8/AWQ/GPTQ/Marlin/Machete 等 C++ 量化内核（Python 侧在 [`03-model-execution/layers/quantization/`](../03-model-execution/layers/quantization/README.md)）。
- `core/` — 调度器辅助 + block 池跟踪相关 C++ helper。
- `cpu/` — CPU 平台特化内核（与 [`08-platforms/cpu.md`](../08-platforms/cpu.md) 对接）。
- `rocm/` — ROCm/AMD 平台特化（与 [`08-platforms/rocm.md`](../08-platforms/rocm.md) 对接）。
- `custom_all_reduce.cuh`、`custom_quick_reduce.cu`、`quickreduce/` — 自定义 all-reduce 内核（被 [`07-distributed/device-communicators/custom-all-reduce.md`](../07-distributed/device-communicators/custom-all-reduce.md) 调用）。
- `cumem_allocator.cpp` — CUDA driver "cumem" pluggable allocator，支撑 sleep mode（被 [`08-platforms/device-allocator.md`](../08-platforms/device-allocator.md) 调用）。
- `cutlass_extensions/` — CUTLASS 修补层（用于 Marlin/Machete）。
- `fs_io.cpp` — KV offload 文件 I/O C 扩展（被 [`15-kv-cache-offload/tiering-fs.md`](../15-kv-cache-offload/tiering-fs.md) 调用，符号 `vllm.fs_io_C`）。
- `spinloop.cpp` — all-reduce 用的自旋等待。
- `qutlass_registration.cpp` — qutlass（FP8 GEMM）算子注册。

## 为什么

- **性能**：自定义 reduce/quant/sparse MLA 在 PyTorch eager 不可达；C++/CUDA 直写避免 dispatch 开销与中间 tensor。
- **硬件最新特性**：Blackwell NVLink（MNNVL）、TMA、Hopper FP8、cuDNN front-end 等需要原生绑定。
- **跨平台**：同一抽象（如 all-reduce）在 CUDA/ROCm/CPU 走不同 C++ 实现，统一在 `csrc/` 下分目录而非 Python if/else。
- **减少 Python 启动开销**：把 `cumem_allocator`、`fs_io` 等做成 C 扩展，避免每个 step 进入 ctypes。

## 怎么做

- `torch_bindings.cpp` 是入口：用 `PYBIND11_MODULE` 注册函数，用 `TORCH_LIBRARY`（`vllm_C` 等 namespace）注册自定义 op。
- `setup.py` 调用 CMake（见 `cmake.md`）；`VLLM_USE_PRECOMPILED=1` 时跳过本地编译直接下载 wheel（详见 [`17-utils-cross-cutting/envs.md`](../17-utils-cross-cutting/envs.md)）。
- Python 侧调用点：`vllm/_custom_ops.py`（统一包装入口，见 [`17-utils-cross-cutting/custom-ops.md`](../17-utils-cross-cutting/custom-ops.md)）。

## 与其它模块/系统配合

- 内核由 [`03-model-execution/layers/`](../03-model-execution/layers/README.md)（linear/quantization/fused_moe）、[`05-attention/`](../05-attention/README.md) 调用。
- [`07-distributed/`](../07-distributed/README.md) 使用 `custom_all_reduce` / `qutlass`。
- [`08-platforms/device-allocator.md`](../08-platforms/device-allocator.md) 使用 `cumem_allocator`。
- [`15-kv-cache-offload/tiering-fs.md`](../15-kv-cache-offload/tiering-fs.md) 使用 `fs_io.cpp`。

## 历史版本演进

- **v0.3–v0.5（早期）**：PagedAttention CUDA 内核 + Marlin 量化 + 简单 all-reduce。
- **v0.6–v0.7**：fp8、Machete、custom_all_reduce 扩族；cumem_allocator 引入支撑 sleep mode。
- **v0.8**：MoE/DeepEP/DeepGEMM 风格内核；srt-cutlass 系列接入。
- **v0.9–v0.10**：qutlass（FP8 GEMM 调度）、`cutlass_extensions/` 升级至 Hopper；sparse MLA C++ 路径成型。
- **v0.11–v0.12 / main**：Blackwell/MNNVL 支持；`fs_io.cpp` 落地服务于 tiered KV offload；CPU/ROCm C++ 子树扩充；qutlass_registration 抽出——具体小版本归属（待核实）。

[← 返回构建/CI/测试首页](../README.md)

## 参见

- `cmake.md`
- `rust.md`
- [`03-model-execution/layers/quantization/README.md`](../03-model-execution/layers/quantization/README.md)
- [`07-distributed/device-communicators/custom-all-reduce.md`](../07-distributed/device-communicators/custom-all-reduce.md)
- [`08-platforms/device-allocator.md`](../08-platforms/device-allocator.md)
