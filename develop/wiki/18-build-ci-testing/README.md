# 18 · 构建 / CI / 测试

[← Wiki 首页](../README.md) > 构建 / CI / 测试

本子系统覆盖 vLLM 工程化外壳：原生 C/C++/CUDA 内核、Rust 组件、CMake 构建、Buildkite/GitHub CI、测试目录、Benchmark、Docker、第三方打包工具。它不参与运行时推理，但决定"能不能装起来、能不能验证、能不能上线"。

| 主题 | 简介 |
|---|---|
| [csrc.md](csrc.md) | `csrc/` 原生 C/C++/CUDA/Triton 内核源与构建调用 |
| `rust.md` | `rust/` Rust 组件（RustFrontend 等）+ `build_rust.sh` |
| `cmake.md` | `cmake/` + 顶层 `CMakeLists.txt` + `setup.py` + `pyproject.toml` |
| `buildkite.md` | `.buildkite/` 流水线定义 |
| `github-ci.md` | `.github/workflows/` 轻量 GitHub Actions |
| `tests.md` | `tests/` 目录组织与运行约定 |
| `benchmarks.md` | `benchmarks/` + `vllm/benchmarks/` + `vllm bench` CLI |
| `docker.md` | `docker/` 镜像构建 |
| `tools.md` | `tools/` 仓库外部工具脚本 |
| `requirements.md` | `requirements/` 分组依赖 |
| `docs.md` | `docs/` + `mkdocs.yaml` + Read the Docs |

[← 返回 Wiki 首页](../README.md)
