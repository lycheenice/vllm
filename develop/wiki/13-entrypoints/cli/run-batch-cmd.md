[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [CLI](README.md) > run-batch

# run-batch 子命令（vllm run-batch）

> `cli/run_batch.py` 的 `RunBatchSubcommand` 注册 `vllm run-batch`：把行参数转交给 `vllm/entrypoints/openai/run_batch.py` 执行 JSONL 批处理。它是 [run-batch.md](../openai/run-batch.md) 描述的批处理器 CLI 壳。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `RunBatchSubcommand` | `vllm/entrypoints/cli/run_batch.py:21` | `vllm run-batch` 子命令 |
| `cmd_init` | `:67` | 返回 `[RunBatchSubcommand()]` |

`RunBatchSubcommand` 在 `subparser_init` 注册参数（`--base-url`/`--input`/`--output`/`--prometheus-port` 等），`cmd` 静态方法调 `openai/run_batch.py` 的主函数（具体函数名 `run_batch`/`main` 待核实，见 [run-batch.md](../openai/run-batch.md)）。

## 为什么

- **薄 CLI 壳**：参数解析与执行分离——CLI 只负责 argparse，重逻辑在 `openai/run_batch.py`，便于复用与测试。
- **统一 `CLISubcommand`**：与其他子命令一致注册路径，由 `main` 自动 dispatch。

## 怎么做

### 用法

```bash
vllm run-batch --base-url http://localhost:8000 \
  --input in.jsonl --output out.jsonl
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| RunBatchSubcommand | `vllm/entrypoints/cli/run_batch.py:21` |
| cmd_init | `vllm/entrypoints/cli/run_batch.py:67` |

## 与其它模块/系统配合

- [openai/run-batch.md](../openai/run-batch.md)：实际实现。
- [README.md](README.md)：`main` 注册。

## 历史版本演进

- **v0.7–v0.8（引入）**：`vllm run-batch` 子命令，仅 chat/embeddings。
- **v0.9+（多 endpoint）**：随 `openai/run_batch.py` 扩展支持 rerank/transcription/translation。

## 参见

- [← 返回 CLI 首页](README.md)
- [openai/run-batch.md](../openai/run-batch.md)
