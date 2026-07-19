# Rust 组件 / CMake / CI / 测试 / Benchmark / Docker / 工具 / 依赖 / 文档

[← Wiki 首页](../README.md) > [构建/CI/测试](../README.md)

为节省篇幅，本节把几页工程化外壳合并呈现；每节都遵循相同五段式。各模块可作为后续拆分页面的入口。

---

## Rust（`rust/` + `build_rust.sh` + `rust-toolchain.toml`）

### 是什么
`rust/` 子树提供 vLLM 的 Rust 组件，主要是 [`RustFrontendProcessManager`](../01-engine-core/engine-core-process.md) 等提到的 Rust 前端 / 工具。`rust-toolchain.toml` 固定工具链版本；`build_rust.sh` 是构建驱动脚本；`setup.py` 与 CMake 在编译时调用它。

### 为什么
- 性能/内存安全的 IPC/前端解析需要 Rust；vLLM 走"Python + 原生 C++ + Rust"三元语言栈。
- 隔离：把高风险的字符串/二进制解析逻辑放进 Rust。

### 怎么做
`build_rust.sh` 调 `cargo build --release`，把产物打包进 wheel；运行时按需加载。

### 配合
与 [`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md) 的 IPC pin-One 协议协作；由 [`17-utils-cross-cutting/envs.md`](../17-utils-cross-cutting/envs.md) 中的 `VLLM_*` 开关启用/禁用。

### 演进
- v0.10 引入（与 scale-out / disaggregated serving 一同落地），main 持续扩充——具体功能边界（待核实）。

---

## CMake（`cmake/` + `CMakeLists.txt` + `setup.py` + `pyproject.toml`）

### 是什么
vLLM 的 C/C++/CUDA/Rust 构建管线：`CMakeLists.txt` 顶层入口；`cmake/` 子目录放辅助 `.cmake`；`setup.py` 在 `VLLM_USE_PRECOMPILED=1` 时下载预编译 wheel，否则驱动 CMake；`pyproject.toml` 是 PEP 517 元数据。

### 为什么
- 跨平台（CUDA / ROCm / XPU / CPU）需要条件编译与变体管理。
- 预编译 wheel 加速安装（不经本地 nvcc）。

### 怎么做
`uv pip install -e . --torch-backend=auto` 让本地编译；`VLLM_USE_PRECOMPILED=1` 走预编译路径。AGENTS.md 规定禁止用 system `python3`/`pip`，必须 `uv` + `.venv/bin/python`。

### 配合
为 [`csrc.md`](csrc.md)、`rust.md` 提供 build；产物被 [`17-utils-cross-cutting/custom-ops.md`](../17-utils-cross-cutting/custom-ops.md) 导入。

### 演进
- v0.5：CMake 模板初步成型。
- v0.6–v0.7：引入 `VLLM_USE_PRECOMPILED` 预编译 wheel。
- v0.8+：多 backend（CUDA/ROCm/XPU/CPU）共用 CMake；GPU arch 矩阵扩展；Hopper/Blackwell 自动检测。

---

## Buildkite（`.buildkite/`）

### 是什么
Buildkite 是 vLLM 主 CI 平台。`.buildkite/` 含 `ci_config.yaml`、`ci_config_intel.yaml`、`ci_config_rocm.yaml`（平台分支）、`release-pipeline.yaml`、`image_build/`（镜像构建）、`performance-benchmarks/`、`hardware_tests/`、`intel_jobs/`、`lm-eval-harness/`，以及 `check-wheel-size.py`。

### 为什么
- 大量 GPU/多平台并发任务超出 GitHub Actions 免费额度，需自托管 Buildkite agent。
- 性能回归需要专用硬件。

### 怎么做
PR 推送触发 Buildkite；按修改路径选 job（如改 `csrc/` 跑 build/test，改 `vllm/v1/` 跑 v1 引擎测试）。CI 失败排查可加载 `ci-fails-buildkite` skill。

### 配合
作为 [`04-model-zoo/`](../04-model-zoo/README.md) 新模型准入前的回归闸门；与 `tests.md`、`benchmarks.md` 直接对应。

### 演进
- v0.5：CI 雏形。
- v0.7：v1 引擎测试矩阵拆分。
- v0.8+：Intel/ROCm 分支独立；`release-pipeline.yaml` 自动化。

---

## GitHub Actions（`.github/workflows/`）

### 是什么
轻量级 PR 预检（labeler、pre-commit、doc build、依赖图）。

### 为什么
Buildkite 排队长时，先用 GH Actions 做 cheap check。

### 怎么做
YAML workflow；pre-commit 跑 ruff/mypy/yapf/clang-format/markdownlint。AGENTS.md 要求 `pre-commit install`。

### 配合
gate 住 `docs.md` 与 lint。

### 演进
- 主要常态化维护；新增 lint 规则按版本。

---

## 测试（`tests/`）

### 是什么
`tests/` 按 vLLM 顶层目录镜像组织：`basic_correctness/`、`compile/`、`config/`、`cuda/`、`detokenizer/`、`distributed/`、`engine/`、`entrypoints/`、`evals/`、`fusion/`、`ir/`、`kernels/`、`lora/`、`model_executor/`、`models/`、`multimodal/`、`parser/`、`plugins/`、`plugins_tests/`、`prompts/`、`quantization/`、`reasoning/`、`renderers/`、`rocm/`、`samplers/`、`spec_decode/` 等；`conftest.py` 是全局 fixture；`ci_envs.py` 暴露 CI 环境判定。

### 为什么
- vLLM 覆盖广，必须按子系统分目录并行执行。
- 大量 GPU-only 测试需要远程执行。

### 怎么做
AGENTS.md 中给的标准命令：
```
uv pip install -r requirements/test/cuda.in
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```
x86_64 可直接用 `requirements/test/cuda.txt`（pinned）。

### 配合
对应每个运行时子系统的 wiki（如 [`01-engine-core/`](../01-engine-core/README.md)、[`05-attention/`](../05-attention/README.md)、[`12-lora/`](../12-lora/README.md)）。

### 演进
- v0.3：单一 tests/ 目录。
- v0.7+：随 v1 拆分；新增 `compile/`、`ir/`、`renderers/`、`reasoning/`。
- v0.9+：scale-out / kv_offload 测试入列。

---

## Benchmark（`benchmarks/` + `vllm/benchmarks/` + `vllm bench`）

### 是什么
两组 benchmark：仓库根 `benchmarks/` 含 `benchmark_latency.py`、`benchmark_throughput.py`、`benchmark_serving.py`、`benchmark_prefix_caching.py`、`attention_benchmarks/`、`auto_tune/` 等；`vllm/benchmarks/` 与 [`13-entrypoints/cli/bench.md`](../13-entrypoints/cli/bench.md) 对接（`vllm bench latency/throughput/serve/startup/sweep/mm-processor`）。

### 为什么
- 性能回归量化。
- 自动调优（`auto_tune/`）。

### 怎么做
独立脚本直接 `python benchmarks/benchmark_latency.py`；CLI 走 `vllm bench <type>`。

### 配合
Buildkite 的 `performance-benchmarks/` 调用；[`16-observability/`](../16-observability/README.md) 的 metrics 字段对位。

### 演进
- v0.5–v0.8：latency/throughput/serving 三大脚本成型。
- v0.10+：`vllm bench` CLI 统一入口；`sweep`/`startup`/`mm-processor` 后续接入。

---

## Docker（`docker/`）

### 是什么
镜像构建文件集（Dockerfile.*、entrypoint、辅助脚本），覆盖 CUDA / ROCm / CPU / XPU 各变体。`.dockerignore` 在根目录。

### 为什么
- 一键起服务环境。
- 解决 CUDA/cuDNN/Flash-Attn 版本匹配痛。

### 怎么做
`docker build -f docker/Dockerfile.<variant> .`，或直接 `docker pull vllm/vllm-...`。

### 配合
被 [`13-entrypoints/cli/serve-cmd.md`](../13-entrypoints/cli/serve-cmd.md) 默认使用；CI `image_build/` 也用它。

### 演进
- 每个版本镜像更新；v0.9+ 引入 official image registry。

---

## Tools（`tools/`）与依赖（`requirements/`）与文档（`docs/` + `mkdocs.yaml`）

### 是什么
- `tools/`：仓库外部维护脚本（如 `vllm/` 镜像提示词生成、CI 修复工具）。
- `requirements/`：分组依赖（`build.txt`、`lint.txt`、`test/{cuda,cuda.in}.txt`、`rocm.txt`、`xpu.txt`、`cpu.txt` 等）。
- `docs/`：mkdocs 站点源；`mkdocs.yaml` 是导航配置；`.readthedocs.yaml` 是 RTD 部署。

### 为什么
- 依赖按场景分组，CI 与本地按需安装。
- 文档与代码同仓，避免漂移。

### 怎么做
按 AGENTS.md：`uv pip install -r requirements/lint.txt`、`uv pip install -r requirements/test/cuda.in`。docs 通过 mkdocs 本地预览：`mkdocs serve`。

### 配合
- `requirements/lint.txt` 驱动 pre-commit（见 [`17-utils-cross-cutting/envs.md`](../17-utils-cross-cutting/envs.md)）。
- `docs/` 与本 wiki（`develop/wiki/`）独立，本 wiki 不进 mkdocs 站点，仅作为内部代码剖析。

### 演进
- 持续维护；v0.8+ 引入按平台 `.in` 源文件解 pin；docs 在 v0.7 v1 切换前后重构。

[← 返回构建/CI/测试首页](README.md)

## 参见

- [`csrc.md`](csrc.md)
- [`17-utils-cross-cutting/envs.md`](../17-utils-cross-cutting/envs.md)
- [`16-observability/`](../16-observability/README.md)
