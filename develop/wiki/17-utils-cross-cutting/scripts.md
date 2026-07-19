# 脚本入口（scripts）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 脚本入口

本页覆盖 `vllm/scripts.py`（17 行），仅是一个向后兼容垫片。

## 是什么

`vllm/scripts.py` 定义 `main()`（`:12`），它：

1. 用 `init_logger(__name__)` 取 logger，发一条 `WARNING`：`vllm.scripts.main() is deprecated. Please re-install vllm or use vllm.entrypoints.cli.main.main() instead.`
2. 调 `from vllm.entrypoints.cli.main import main as vllm_main` 后执行 `vllm_main()`。

模块顶部说明（`:10`）：注释解释这是为了兼容"`vllm.scripts` 迁移到 `vllm.entrypoints.cli.main`"的旧调用路径。

## 为什么

- **历史 `console_scripts` 入口**：早期 vLLM 的 `pyproject.toml` 把 `vllm` 命令的 entry point 指向 `vllm.scripts:main`；重构后真正入口搬到 `vllm.entrypoints.cli.main:main`（见 [API 入口 · CLI](../13-entrypoints/README.md)）。
- **避免破坏旧安装**：用户若未重装 vLLM，`vllm` 命令仍指向旧路径；本垫片让旧 wrapper 继续可用，同时提示重装。
- **极简**：不承载任何业务，纯转发 + 告警，便于未来移除。

## 怎么做

- **正常路径**：直接用 `vllm.entrypoints.cli.main.main`（或 shell 的 `vllm` 命令，由 `pyproject.toml` entry point 指向新地址）。
- 发现调用栈出现 `vllm.scripts.main`：重装 vLLM 以刷新 entry point，消除 deprecation warning。

## 与其它模块/系统配合

- [API 入口 · CLI](../13-entrypoints/README.md)：真正的 CLI 主入口 `vllm.entrypoints.cli.main`。
- [构建/CI](../18-build-ci-testing/README.md)：`pyproject.toml` 的 `[project.scripts]` 决定 `vllm` 命令指向何处。

## 历史版本演进

- **早期 v0**：CLI 入口在 `vllm/scripts.py`，`vllm` 命令经此启动 serve/generate 等。
- **v0.7–v0.8**：CLI 重构，主入口迁至 `vllm/entrypoints/cli/main.py`；`scripts.py` 降级为转发垫片并加 deprecation warning。
- **v0.9–main**：保持垫片存在以防外部脚本/旧文档引用；最终移除时间未定（待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [API 入口 · CLI](../13-entrypoints/README.md)
- [构建/CI](../18-build-ci-testing/README.md)
