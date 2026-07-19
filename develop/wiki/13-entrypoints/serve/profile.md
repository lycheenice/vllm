[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > profile

# profile/（/start_profile & /stop_profile）

> `serve/profile/` 暴露 PyTorch profiler 控制端点，让外部按需启动/停止 GPU kernel 级性能采集，产出 trace 供 PyTorch Profeter/Perfetto 分析。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `engine_client` 依赖 | `vllm/entrypoints/serve/profile/api_router.py:17` | 从 request 取 engine_client |
| `start_profile` | `:22` | `POST /start_profile` → `engine_client.collective_rpc("start_profile")` |
| `stop_profile` | `:30` | `POST /stop_profile` → `engine_client.collective_rpc("stop_profile")` |
| `attach_router` | `:37` | 注册两个端点 |

采集实际由 worker 端 `Profiler` 控制（见 [可观测-profiler](../../16-observability/README.md)）。

## 为什么

- **按需采集**：长跑服务里只对感兴趣时段开 profiler，避免全量采集开销与磁盘占用。
- **集体 RPC 广播**：`collective_rpc("start_profile")` 广播到所有 worker，TP/PP 各 rank 同步采集，trace 可对齐。
- **简单无体**：端点无 request body（或仅可选 `duration` 待核实），调用即开/关，便于脚本化。

## 怎么做

```bash
curl -X POST http://localhost:8000/start_profile
# 跑一段时间流量
curl -X POST http://localhost:8000/stop_profile
# trace 落盘到 worker 节点（路径由 ProfilerConfig 决定）
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| start_profile | `vllm/entrypoints/serve/profile/api_router.py:22` |
| stop_profile | `vllm/entrypoints/serve/profile/api_router.py:30` |
| attach_router | `vllm/entrypoints/serve/profile/api_router.py:37` |

## 与其它模块/系统配合

- [可观测-profiler](../../16-observability/README.md)：worker `Profiler` 实现。
- [engine-serve.md](engine-serve.md)：`collective_rpc` 通道。
- [配置-scheduler](../../10-config/scheduler-config.md)：`ProfilerConfig.output_dir`。

## 历史版本演进

- **v0.7（引入）**：`/start_profile`/`/stop_profile` 端点。
- **v0.9（serve/profile 子包）**：抽到独立子包。
- **main**：远程输出路径协商（待核实）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [可观测-profiler](../../16-observability/README.md)
