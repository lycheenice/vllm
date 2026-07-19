# KVCacheMetricsCollector

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [KV 缓存管理](README.md) > KVCacheMetrics

源码：`vllm/v1/core/kv_cache_metrics.py`（约 96 行）。本模块提供块级生命周期采样与驱逐事件收集，是 v1 可观测性在 KV cache 层的探针。

## 是什么

### `BlockMetricsState`（`kv_cache_metrics.py:16`）

单个物理块的生命周期状态：
- `birth_time_ns`：被采样分配时的 `time.monotonic_ns()`。
- `last_access_ns`：最近一次 `record_access` 时间。
- `access_history: deque[int]`（`maxlen=4`）：访问时间戳序列，上界 4 防止频繁访问的块无界增长。
- `record_access()`：更新 `last_access_ns` 并 append 历史。
- `get_lifetime_seconds()` / `get_idle_time_seconds()` / `get_reuse_gaps_seconds()`：驱逐时计算指标。

### `KVCacheMetricsCollector`（`kv_cache_metrics.py:46`）

按 `sample_rate`（默认 0.01，1% 块）采样追踪：
- `block_metrics: dict[int, BlockMetricsState]`：被采样的 block_id → state。
- `_eviction_events: list[KVCacheEvictionEvent]`：驱逐事件累积。
- `should_sample_block() -> bool`：`random.random() < sample_rate`。
- 钩子：
  - `on_block_allocated(block)`：采样命中时新建 `BlockMetricsState`。
  - `on_block_accessed(block)`：被采样块记录访问。
  - `on_block_evicted(block)`：从 `block_metrics` pop，计算 lifetime/idle/reuse_gaps，构造 `KVCacheEvictionEvent` 追加。
- `reset()`：cache reset 时清空全部状态（`Scheduler.reset_prefix_cache` 触发）。
- `drain_events() -> list[KVCacheEvictionEvent]`：scheduler `make_stats` 时 drain。

`KVCacheEvictionEvent`（来自 `vllm.v1.metrics.stats`）字段：`lifetime_seconds`/`idle_seconds`/`reuse_gaps_seconds: tuple`。

## 为什么

- **采样而非全量**：百万级 block 全量追踪开销大；1% 采样足以为驱逐策略与缓存命中率分析提供统计信号。
- **access_history 上界 4**：热点块可能被同请求多次 touch、被多请求命中；无界 deque 会爆内存。4 个时间戳足够计算"是否被复用"与"复用间隔"。
- **驱逐时计算**：`on_block_evicted` 是唯一计算 lifetime/idle 的点，避免每个 access 都计算；事件延迟到 `drain_events` 由 scheduler 取走，与 prometheus 推送节拍对齐。
- **与 prefix cache reset 协同**：`reset_prefix_cache` 调 `metrics_collector.reset()`，避免被驱逐块在 reset 后仍出现在 `block_metrics` 造成悬挂引用。
- **可观测性接口**：`drain_events` 把 `KVCacheEvictionEvent` 列表交给 `SchedulerStats.kv_cache_eviction_events`，最终经 `StatLoggerManager` 落 prometheus/log（见 [`16-observability`](../../16-observability/README.md)）。
- **零开销禁用路径**：`observability_config.kv_cache_metrics=False` 时 scheduler 不创建 collector，所有钩子路径跳过。

## 怎么做

### 与 BlockPool 集成

`BlockPool.__init__` 接受 `metrics_collector`，在三个关键时刻调用：
- `get_new_blocks`：每个新分配块 `on_block_allocated(block)`（caching 开关两条路径都调）。
- `touch`：命中复用时 `on_block_accessed(block)`。
- `_maybe_evict_cached_block`：驱逐前 `on_block_evicted(block)`（在清理 hash 之前，确保 state 仍可读）。

`reset_prefix_cache` 末尾 `metrics_collector.reset()`。

### 数据流

```mermaid
flowchart LR
    BP[BlockPool] -->|on_block_allocated / accessed / evicted| MC[KVCacheMetricsCollector]
    MC -->|采样命中| BMS[BlockMetricsState per block_id]
    MC -->|驱逐时| EV[_eviction_events: list[KVCacheEvictionEvent]]
    EV -->|drain_events| SCH[Scheduler.make_stats]
    SCH -->|kv_cache_eviction_events| SS[SchedulerStats]
    SS --> SLM[StatLoggerManager.record]
    SLM --> Prom[Prometheus / 日志]
```

### scheduler 调用

`Scheduler.__init__`：
```python
self.kv_metrics_collector = None
if self.observability_config.kv_cache_metrics:
    self.kv_metrics_collector = KVCacheMetricsCollector(
        self.observability_config.kv_cache_metrics_sample,
    )
```

`Scheduler.make_stats`（`scheduler.py:2276`）：
```python
eviction_events = (
    self.kv_metrics_collector.drain_events()
    if self.kv_metrics_collector is not None
    else []
)
return SchedulerStats(
    ...
    kv_cache_eviction_events=eviction_events,
)
```

`SchedulerStats` 把事件列表序列化（msgspec）随 `EngineCoreOutputs.scheduler_stats` 回前端，`StatLoggerManager` 消费后上报。

### 配置开关

`observability_config.kv_cache_metrics`（bool）：是否启用。`kv_cache_metrics_sample`（float，0~1）：采样率，默认值在 `ObservabilityConfig`（待核实具体默认）。启用时 scheduler 与 `BlockPool` 都持 collector 引用。

## 与其它模块/系统配合

- **[block-pool.md](./block-pool.md)**：三个钩子的唯一调用方；`reset_prefix_cache` 触发 `collector.reset()`。
- **[kv-cache-manager.md](./kv-cache-manager.md)**：`metrics_collector` 在 `KVCacheManager.__init__` 透传给 coordinator → block_pool。
- **[Scheduler](../scheduler/scheduler.md)**：`make_stats` drain 事件；`reset_prefix_cache` 触发 reset。
- **[16-observability](../../16-observability/README.md)**：`StatLoggerManager` 与 prometheus 上报；`KVCacheEvictionEvent` 数据结构定义在 `vllm/v1/metrics/stats.py`。
- **[KV events（BlockStored/BlockRemoved）](./block-pool.md)**：本 collector 与 KV 事件框架互补——前者是采样统计，后者是结构性事件（每块必发）；二者都通过 `SchedulerStats` 上报。

## 历史版本演进

- **v0.7/v0.8（v1 早期）**：无独立 metrics collector；KV cache 指标仅有 `kv_cache_usage` 与 `prefix_cache_stats`。
- **v0.9（待核实）**：`KVCacheMetricsCollector` 与 `BlockMetricsState` 引入；`observability_config.kv_cache_metrics` 与 `kv_cache_metrics_sample` 配置加入。最早可能版本（待核实）。
- **v0.10 / main**：`access_history` maxlen=4 限制；与 `reset_prefix_cache` 协同 reset；`KVCacheEvictionEvent` 字段稳定。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [block-pool.md](./block-pool.md) — 钩子的调用方。
- [kv-cache-manager.md](./kv-cache-manager.md) — collector 的注入路径。
- [../scheduler/scheduler.md](../scheduler/scheduler.md) — `make_stats` 的 drain 点。
