[← Wiki 首页](../README.md) > [API 入口](../README.md) > CLI

# CLI 子系统（cli/）

> `vllm/entrypoints/cli/` 是 `vllm` 命令行的实现：用 `CLISubcommand` 框架把 `serve`/`launch`/`openai`/`bench`/`run-batch`/`collect-env` 拼成统一入口，每个子命令各自定义 argparse 与 dispatch 函数，懒加载避免 eager import 破坏。

## 是什么

| 文件 | 核心 | 职责 |
|---|---|---|
| `main.py` | `main`（`vllm/entrypoints/cli/main.py:17`） | 顶层入口：聚合 CMD_MODULES，建 subparser，dispatch |
| `types.py` | `CLISubcommand`（`vllm/entrypoints/cli/types.py:13`） | 子命令抽象基类 |
| `serve.py` | `ServeSubcommand` | `vllm serve` |
| `launch.py` | `LaunchSubcommand`/`RenderSubcommand` | `vllm launch render` |
| `openai.py` | `ChatCommand`/`CompleteCommand` | `vllm chat`/`vllm complete` 交互式 CLI |
| `run_batch.py` | `RunBatchSubcommand` | `vllm run-batch` |
| `collect_env.py` | `CollectEnvSubcommand` | `vllm collect-env` |
| `benchmark/main.py` | `BenchmarkSubcommand` | `vllm bench {serve,latency,throughput,sweep,startup,mm_processor}` |

`main` 流程：

1. 懒 import 6 个 `CMD_MODULES`。
2. `cli_env_setup()`。
3. 若 `--omni` 在 `sys.argv` 委托 `vllm_omni`（若装）。
4. 若首参为 `bench`，把 `UnspecifiedPlatform` 切到 `CpuPlatform`（避免设备推断报错）。
5. 建 `FlexibleArgumentParser`，子命令 subparser；对每个 module 的 `cmd_init()` 返回的 `CLISubcommand`，调 `subparser_init` 注册参数并 `set_defaults(dispatch_function=cmd.cmd)`。
6. `parse_args` → 调对应 `cmd.validate(args)` → `args.dispatch_function(args)`，无 subparser 则 print_help。

`CLISubcommand`（`types.py:13`）契约：`name`、静态 `cmd(args)`、`validate(args)`（默认 no-op）、`subparser_init(subparsers)` 返回子 parser。

## 为什么

- **懒加载防破坏**：`main` 在函数体内 import 各 module，避免顶层 import 触发重型依赖（torch/CUDA）；注释明确"further modules must be lazily loaded within main to avoid eager import breakage"。
- **可插拔子命令**：新子命令只需写 `cmd_init() -> [CLISubcommand]` 并加入 `CMD_MODULES`，main 自动注册，无需改 main 逻辑。
- **platform 切换**：`vllm bench` 在无 GPU 机器上跑前先把 platform 设为 CPU，让 benchmark 子命令能纯 CPU 验证逻辑。
- **`--omni` 委托**：vLLM Omni 扩展通过 `--omni` 接管入口，主 CLI 不污染。
- **统一 epilog**：`VLLM_SUBCMD_PARSER_EPILOG` 给所有子命令一致帮助尾部，引导 `--help=<group>` 检索。

## 怎么做

### 子命令骨架

```python
class FooSubcommand(CLISubcommand):
    name = "foo"
    @staticmethod
    def cmd(args): ...
    def subparser_init(self, subparsers):
        p = subparsers.add_parser(self.name, ...)
        # add_argument...
        return p

def cmd_init() -> list[CLISubcommand]:
    return [FooSubcommand()]
```

### 命令一览

| CLI | 子命令类 | dispatch |
|---|---|---|
| `vllm serve <model>` | `ServeSubcommand` | `serve.py:50` |
| `vllm launch render` | `RenderSubcommand` | `launch.py:56` → `run_launch_fastapi` |
| `vllm chat` | `ChatCommand` | `openai.py:155` |
| `vllm complete` | `CompleteCommand` | `openai.py:237` |
| `vllm run-batch` | `RunBatchSubcommand` | `run_batch.py:21` |
| `vllm collect-env` | `CollectEnvSubcommand` | `collect_env.py:16` |
| `vllm bench {sub}` | `BenchmarkSubcommand` | `benchmark/main.py:29` |

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| main | `vllm/entrypoints/cli/main.py:17` |
| CMD_MODULES | `vllm/entrypoints/cli/main.py:30` |
| omni 委托 | `vllm/entrypoints/cli/main.py:42` |
| bench platform 切换 | `vllm/entrypoints/cli/main.py:58` |
| dispatch 装配 | `vllm/entrypoints/cli/main.py:84` |
| CLISubcommand | `vllm/entrypoints/cli/types.py:13` |

## 与其它模块/系统配合

- [serve-cmd.md](serve-cmd.md)：`vllm serve` 调 `run_server`/`run_dp_supervisor`/`run_multi_api_server`。
- [openai/api-server.md](../openai/api-server.md)：serve/launch 复用 `setup_server`/`build_and_serve`。
- [openai/dp-supervisor.md](../openai/dp-supervisor.md)：serve 多端口分支。
- [openai/run-batch.md](../openai/run-batch.md)：run-batch 子命令。
- [launcher.md](../launcher.md)：serve/launch 终汇入 `serve_http`。
- [08-platforms](../../08-platforms/README.md)：bench 的 CPU platform 切换。

## 历史版本演进

- **v0.5–v0.6（脚本式）**：`python -m vllm.entrypoints.openai.api_server` 直跑，无统一 CLI。
- **v0.7–v0.8（CLI 框架）**：引入 `CLISubcommand` + `vllm` 入口；`serve`/`openai`(chat/complete)/`bench` 子命令落地。
- **v0.9（run-batch + collect-env）**：补 `run-batch`、`collect-env`；懒加载策略固化。
- **v0.10（launch render）**：`vllm launch render` 支持 CPU-only 渲染 server；`--omni` 委托。
- **v0.11/main**：bench platform 切换；CMD_MODULES 列表化；`VLLM_SUBCMD_PARSER_EPILOG` 统一；子命令 `validate` 串入。

## 模块导航

| 页 | 主题 |
|---|---|
| [main.md](main.md) | `main.py` + `types.py` |
| [serve-cmd.md](serve-cmd.md) | `vllm serve` |
| [openai-cmd.md](openai-cmd.md) | `vllm chat`/`complete` |
| [launch-cmd.md](launch-cmd.md) | `vllm launch render` |
| [run-batch-cmd.md](run-batch-cmd.md) | `vllm run-batch` |
| [collect-env-cmd.md](collect-env-cmd.md) | `vllm collect-env` |
| [bench.md](bench.md) | `vllm bench` 子命令 |

## 参见

- [← 返回 API 入口首页](../README.md)
- [openai/api-server.md](../openai/api-server.md)
- [launcher.md](../launcher.md)
