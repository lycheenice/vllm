# ProfilerConfig（profiler.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ProfilerConfig

源码：`vllm/config/profiler.py`（约 147 行）。`ProfilerConfig` 描述引擎级性能分析器：torch profiler / cuda profiler、trace 保存目录、stack/flops/gzip/memory/shape 选项、前端忽略、delay/max/warmup/active/wait 迭代调度。它是 `VllmConfig.profiler_config`，被 `AsyncLLM`（前端）与 `Worker`（CPU+GPU trace）经 `/start_profile`/`/stop_profile` API 触发消费。

## 是什么

`@config` 装饰（`profiler.py:33`）。`ProfilerKind = Literal["torch","cuda"]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `profiler` | `None` | `torch`/`cuda`/`None` |
| `torch_profiler_dir` | `""` | torch profiler trace 保存目录（须绝对路径或 URI 如 `gs://`/`s3://`）；仅 `profiler="torch"` 有效 |
| `torch_profiler_with_stack` | `True` | 启用 stack tracing（调试有用，可关） |
| `torch_profiler_with_flops` | `False` | 启用 FLOPS 计数 |
| `torch_profiler_use_gzip` | `True` | gzip 压缩 trace |
| `torch_profiler_dump_cuda_time_total` | `True` | dump 总 CUDA 时间 |
| `torch_profiler_record_shapes` | `False` | 记录张量形状 |
| `torch_profiler_with_memory` | `False` | 内存 profiling |
| `ignore_frontend` | `False` | 忽略 AsyncLLM 前端 profiling（delay/limit 时推荐开，减开销） |
| `delay_iterations` | `0` | 跳过前 N 步再开始 profiling |
| `max_iterations` | `0`(无限) | profiling 最大步数 |
| `warmup_iterations` | `0`(禁调度式) | warmup 步数（数据丢弃，减 JIT 噪声） |
| `active_iterations` | `5` | active 步数（实际采集） |
| `wait_iterations` | `0` | wait 步数（profiler 全关，零开销） |

辅助 `_is_uri_path(path)`：检测 `gs://`/`s3://`/`hdfs://` 等 URI（多字符 scheme），URI 不转绝对路径。

校验（`_validate_profiler_config`）：`profiler="torch"` + delay/limit + `ignore_frontend=False` 时 warning（前端开销高）；`torch_profiler_dir` 仅 `profiler="torch"` 有效，`profiler="torch"` 须设 dir；非 URI 路径转绝对路径。

`compute_hash`：空 factors——profiling 不影响编译图形状。

## 为什么

- **profiler 调度**：PyTorch profiler 有 wait/warmup/active 三段循环。`wait_iterations`（零开销跳过）→`warmup_iterations`（运行但丢数据减 JIT 噪声）→`active_iterations`（采集）。`delay_iterations` 在 schedule 之外额外跳过初始步。
- **前端与 worker 分离**：`AsyncLLM`（前端，CPU trace）与 `Worker`（CPU+GPU trace）各自 profiler。`ignore_frontend` 在用 delay/limit 时关前端，因前端 profiler 不计迭代会捕获整个范围导致开销膨胀。
- **URI 支持**：`torch_profiler_dir` 支持 `gs://`/`s3://`/`hdfs://` 直接上传云存储，避免本地盘占用；`_is_uri_path` 防止把 URI 误转绝对路径。
- **stack 默认开**：`torch_profiler_with_stack=True` 因调试价值高；可显式关减开销。
- **`compute_hash` 空**：profiling 热路径仅在 `/start_profile` 后激活，不改变图形状，故缓存键不区分。

## 怎么做

- **torch profiler**：通过 API `POST /start_profile {"profiler":"torch","torch_profiler_dir":"/tmp/traces","torch_profiler_record_shapes":true}` 触发，`/stop_profile` 停止。
- **调度式**：`warmup_iterations=2, active_iterations=10, wait_iterations=5` 循环采集。
- **delay/limit**：`delay_iterations=10, max_iterations=20, ignore_frontend=true`。
- **云上传**：`torch_profiler_dir="gs://my-bucket/traces/"`。
- **cuda profiler**：`profiler="cuda"`（无 dir，用 `cuda` 工具）。

## 与其它模块/系统配合

- **AsyncLLM（[`01-engine-core/async-llm-frontend.md`](../01-engine-core/async-llm-frontend.md)）**：`/start_profile`/`/stop_profile` 经 EngineCore 转发；前端按 `ignore_frontend` 决定是否捕获 CPU trace。
- **Worker / ModelRunner（[`02-execution/worker/gpu-worker.md`](../02-execution/worker/gpu-worker.md)）**：worker 收到 profile 指令后启 `torch.profiler.profile`，捕获 CPU+GPU trace，保存到 `torch_profiler_dir`。
- **ObservabilityConfig（[observability-config.md](observability-config.md)）**：`enable_layerwise_nvtx_tracing` 是轻量标记，profiler 是全量捕获，二者层次不同；`enable_mfu_metrics` 与 `torch_profiler_with_flops` 都涉及 FLOPS 但前者是 Prometheus 指标，后者是 trace 字段。
- **API server（[`13-entrypoints/`](../13-entrypoints/README.md)）**：`/start_profile`/`/stop_profile` 端点把请求体映射为本配置覆盖。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`profiler_config.compute_hash()`（空）进顶层哈希保持聚合完整。

## 历史版本演进

- **v0.5/v0.6（v0）**：`--profile` 简单开关，v0 用 `torch.profiler` 基础捕获。
- **v0.7（v1 落地）**：`ProfilerConfig` 独立；`torch_profiler_dir`/`with_stack`/`with_flops`/`use_gzip`/`record_shapes`/`with_memory` 字段；`/start_profile`/`/stop_profile` API。
- **v0.8**：`delay_iterations`/`max_iterations`；`ignore_frontend`；`cuda` profiler 选项。
- **v0.9**：`warmup_iterations`/`active_iterations`/`wait_iterations` 调度式 profiler；URI 路径支持（`gs://`/`s3://`/`hdfs://`）；`torch_profiler_dump_cuda_time_total`。
- **v0.10–main**：`_is_uri_path` 多字符 scheme 判定（排 Windows 盘符）；与 MRv2 的 profile 集成。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [observability-config.md](observability-config.md) — NVTX/MFU 与 profiler 的层次关系。
- [vllm-config.md](vllm-config.md) — `profiler_config.compute_hash()` 进顶层哈希。
- [../16-observability/README.md](../16-observability/README.md) — profiler 子系统消费方。
- [../01-engine-core/async-llm-frontend.md](../01-engine-core/async-llm-frontend.md) — `/start_profile` 前端处理。
