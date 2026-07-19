[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > bench

# bench 子命令（vllm bench）

> `cli/benchmark/` 注册 `vllm bench`，下面挂 `serve`/`latency`/`throughput`/`sweep`/`startup`/`mm_processor` 子-子命令，分别对应不同维度的 vLLM 性能基准。它们调 `benchmarks/` 仓库脚本或内联实现，起 server/客户端打流量并汇总指标。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `BenchmarkSubcommand` | `vllm/entrypoints/cli/benchmark/main.py:29` | `vllm bench` 顶层子命令 |
| `_import_bench_subcommand_modules` | `:18` | 懒 import 各 bench 子命令模块 |
| `cmd_init` | `:78` | 返回 `[BenchmarkSubcommand()]` |
| `serve.py` | `cli/benchmark/serve.py` | `vllm bench serve`：起 server 并打流量 |
| `latency.py` | `cli/benchmark/latency.py` | `vllm bench latency`：延迟基准 |
| `throughput.py` | `cli/benchmark/throughput.py` | `vllm bench throughput`：吞吐基准 |
| `sweep.py` | `cli/benchmark/sweep.py` | `vllm bench sweep`：参数扫描 |
| `startup.py` | `cli/benchmark/startup.py` | `vllm bench startup`：冷启动时延 |
| `mm_processor.py` | `cli/benchmark/mm_processor.py` | `vllm bench mm_processor`：多模态处理器基准 |
| `base.py` | `cli/benchmark/base.py` | bench 子命令基类 |

`BenchmarkSubcommand` 用嵌套 sub-subparsers 装配各子命令（与 `launch` 模式类似）。`main.py:58` 在 `vllm bench` 时把 `UnspecifiedPlatform` 切到 `CpuPlatform`（见 [main.md](main.md)），让 bench 在无 GPU 机器也能跑逻辑校验。

## 为什么

- **多维基准一站式**：延迟、吞吐、冷启动、参数扫描、多模态处理器分别独立子命令，按需运行，避免单一脚本臃肿。
- **与 server 子命令同构**：`vllm bench serve` 内部起一个 server 再打流量，复用 `serve` 子命令的参数与启动逻辑，保证基准环境与生产一致。
- **CPU platform 兜底**：bench 子命令常在 CI/无 GPU 机器跑纯逻辑验证，强制 CPU platform 避免设备推断报错。
- **参数扫描自动化**：`sweep` 跨多个参数组合跑，输出汇总，便于调参。

## 怎么做

```bash
vllm bench serve --model <m> --base-url http://localhost:8000
vllm bench latency --model <m> --num-prompts 100
vllm bench throughput --model <m> --num-prompts 1000
vllm bench sweep --config sweep.yaml
vllm bench startup --model <m>
vllm bench mm_processor --model <m> --image-dir ./imgs
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| BenchmarkSubcommand | `vllm/entrypoints/cli/benchmark/main.py:29` |
| 模块 import | `vllm/entrypoints/cli/benchmark/main.py:18` |
| cmd_init | `vllm/entrypoints/cli/benchmark/main.py:78` |
| serve 子命令 | `vllm/entrypoints/cli/benchmark/serve.py` |
| latency 子命令 | `vllm/entrypoints/cli/benchmark/latency.py` |
| throughput 子命令 | `vllm/entrypoints/cli/benchmark/throughput.py` |
| sweep 子命令 | `vllm/entrypoints/cli/benchmark/sweep.py` |
| startup 子命令 | `vllm/entrypoints/cli/benchmark/startup.py` |
| mm_processor 子命令 | `vllm/entrypoints/cli/benchmark/mm_processor.py` |
| base 子命令基类 | `vllm/entrypoints/cli/benchmark/base.py` |

## 与其它模块/系统配合

- [serve-cmd.md](serve-cmd.md)：`bench serve` 复用 server 启动。
- [openai/chat-completion.md](../openai/chat-completion.md)/[completion.md](../openai/completion.md)：bench 客户端打 `/v1/*`。
- [08-platforms](../../08-platforms/README.md)：bench 的 CPU platform 切换。
- [18-build-ci-testing](../../18-build-ci-testing/README.md)：CI 中跑 bench。

## 历史版本演进

- **v0.7（bench 子命令）**：`vllm bench` 框架 + serve/latency/throughput。
- **v0.9（sweep/startup）**：参数扫描与冷启动基准。
- **v0.10（mm_processor）**：多模态处理器基准子命令。
- **main**：CPU platform 兜底；base 子命令基类抽象（`base.py`，待核实完整接口）。

## 参见

- [← 返回 CLI 首页](README.md)
- [serve-cmd.md](serve-cmd.md)
- [18-build-ci-testing](../../18-build-ci-testing/README.md)
