[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > collect-env

# collect-env 子命令（vllm collect-env）

> `cli/collect_env.py` 的 `CollectEnvSubcommand` 注册 `vllm collect-env`：收集 vLLM 运行环境信息（OS、Python、torch、CUDA、依赖版本、关键 env 变量）打印，便于 bug 报告。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `CollectEnvSubcommand` | `vllm/entrypoints/cli/collect_env.py:16` | `vllm collect-env` 子命令 |
| `cmd_init` | `:37` | 返回 `[CollectEnvSubcommand()]` |

`cmd` 调用环境收集逻辑（具体实现待核实是否复用 `vllm/envs` 或类似 `torch.utils.collect_env`），打印到 stdout。

## 为什么

- **快速诊断**：用户报 bug 时一条命令拿到全量环境信息，避免反复追问版本。
- **无副作用**：不连网、不起引擎，纯本地读取，安全可随时运行。

## 怎么做

```bash
vllm collect-env
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| CollectEnvSubcommand | `vllm/entrypoints/cli/collect_env.py:16` |
| cmd_init | `vllm/entrypoints/cli/collect_env.py:37` |

## 与其它模块/系统配合

- [README.md](README.md)：`main` 注册。
- [17-utils-cross-cutting](../../17-utils-cross-cutting/README.md)：`vllm/envs`、平台信息。

## 历史版本演进

- **v0.9（引入）**：`vllm collect-env` 子命令落地。
- **main**：字段随版本演进增补（待核实具体字段集）。

## 参见

- [← 返回 CLI 首页](README.md)
