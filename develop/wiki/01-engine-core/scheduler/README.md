# Scheduler 子模块

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [Scheduler](README.md)

`vllm/v1/core/sched/` 子目录是 v1 调度器的全部实现。调度器是 EngineCore 进程内最关键的决策组件：每个调度步决定"哪些请求跑、各跑多少 token、抢谁的资源"。

## 子模块边界

调度器以 `SchedulerInterface`（`interface.py`）为抽象基类，`Scheduler`（`scheduler.py`）为默认实现，`AsyncScheduler`（`async_scheduler.py`）为异步调度变体。它消费 `Request` 状态机（`vllm/v1/request.py`），产出 `SchedulerOutput`（`output.py`），并在 `update_from_output` 中消费 `ModelRunnerOutput` 更新状态、构造 `EngineCoreOutputs`。

调度器内部依赖：
- [`KVCacheManager`](../kv-cache-management/kv-cache-manager.md)：物理块分配、prefix cache 命中、sliding window 跳块。
- `EncoderCacheManager`：多模态编码器输出的预约与释放。
- `StructuredOutput_manager`：grammar 编译状态与 bitmask。
- 可选 `KVConnector` / `ECConnector`：P/D 转移与外部编码器缓存。

## 与 EngineCore 的关系

`EngineCore.step` / `step_with_batch_queue` 严格按 `scheduler.schedule() → execute_model → scheduler.update_from_output()` 串联。调度器不知道 GPU/Executor 存在，仅通过 `SchedulerOutput` 与 `ModelRunnerOutput` 这对数据结构与之对话。

## 子目录导航表

| 文档 | 简介 | 主要源码 |
|---|---|---|
| [scheduler.md](scheduler.md) | `Scheduler` 主类：schedule 主循环、update_from_output、抢占、stop 判定、KV connector 协同 | `scheduler.py` |
| [queues.md](queues.md) | `RequestQueue`/`FCFSRequestQueue`/`PriorityRequestQueue`、`skipped_waiting`、`PauseState` | `request_queue.py`、`interface.py` |
| [chunked-prefill.md](chunked-prefill.md) | `max_num_scheduled_tokens` 预算切分、`long_prefill_token_threshold`、Mamba block-aligned split | `scheduler.py` |
| [preemption.md](preemption.md) | 资源不足时的抢占、`_preempt_request`、deferred free、priority 反向抢占 | `scheduler.py` |
| [parallel-sampling.md](parallel-sampling.md) | `n>1` 并行采样：`ParentRequest` fan-out 与子请求聚合 | `vllm/v1/engine/parallel_sampling.py` |

## 关键设计原则

1. **统一调度模型**：无显式 prefill/decode phase，仅维护每请求的 `num_computed_tokens` 与 `num_tokens_with_spec`；chunked prefill、prefix caching、speculative decode 都是对"新 token 数"的不同切分。
2. **FCFS 默认 + Priority 可选**：通过 `scheduler_config.policy` 选择，`RequestQueue` 抽象屏蔽底层 deque/heap 差异。
3. **skipped_waiting**：把 KV 加载未完成、grammar 未编译、streaming_input 等待中的请求暂存，下次调度再尝试 promote，避免它们阻塞 waiting 队列。
4. **block-aligned admission**：Mamba/SSM 模型要求 chunk 按 block_size 对齐以缓存状态；`_mamba_block_aligned_split` 在调度层面强制。
5. **DP prefill 节拍**：MoE DP 下 `_should_throttle_prefills` + `prefill_schedule_interval` 让各 rank 的 prefill 步对齐，避免 all-to-all 死锁。

[← 返回引擎核心首页](../README.md)
