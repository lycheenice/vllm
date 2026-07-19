# KVEventsConfig（kv_events.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > KVEventsConfig

源码：`vllm/config/kv_events.py`（约 52 行）。`KVEventsConfig` 描述 KV cache 事件发布：是否启用、publisher 后端、zmq endpoint、重放端点、缓冲与队列上限、topic。它是 `VllmConfig.kv_events_config`（`None` 表示未启用），被 `BlockPool` 的事件发布与外部消费者（如 KV cache 跟踪器、disagg 协调器）消费。

## 是什么

`@config` 装饰（`kv_events.py:10`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `enable_kv_cache_events` | `False` | 启用 block 存储与移除事件跟踪 |
| `publisher` | `None`(派生) | `"null"`/`"zmq"`；`None` 时按 `enable_kv_cache_events` 自动（True→`zmq`，False→`null`） |
| `endpoint` | `"tcp://*:5557"` | zmq 发布端点 |
| `replay_endpoint` | `None` | zmq 重放端点 |
| `buffer_steps` | `10_000` | 重放端点缓存步数（仅存最近 N 步事件） |
| `hwm` | `100_000` | zmq 高水位（队列满后丢事件） |
| `max_queue_size` | `100_000` | 发布前最大排队事件数 |
| `topic` | `""` | 发布 topic（订阅者按 topic 过滤） |

`__post_init__`：`publisher` 为 `None` 时按 `enable_kv_cache_events` 派生。

> 无 `compute_hash`——事件发布不影响编译图形状（`VllmConfig.compute_hash` 中 `kv_events_config` **未**纳入因子列表，与 `kv_transfer_config` 一致）。

## 为什么

- **观测面正交于迁移**：KV 迁移（`KVTransferConfig`）做 KV 搬运；KV 事件（本配置）做 block 生命周期观测（存储/移除/复用），二者正交。事件让外部系统跟踪 prefix cache 命中、block 驻留时长、复用 gap 等，支撑跨实例协调与可观测性。
- **zmq 发布/订阅**：`publisher="zmq"` 用 zmq PUB socket；`hwm` 控制背压（消费者跟不上则丢）；`topic` 支持多订阅者按主题过滤；`replay_endpoint` 让历史事件可重放（仅存 `buffer_steps` 步）。
- **`publisher` 自动派生**：用户只设 `enable_kv_cache_events=True` 即自动选 zmq publisher，减少必填项。
- **与 prefix caching 联动**：`VllmConfig.__post_init` 校验——事件开但 prefix caching 关时 warning（事件依赖 prefix cache block 哈希）；publisher 非 null 但 `enable_kv_cache_events=False` 时 warning（配置矛盾）。

## 怎么做

- **启用**：`--kv-events-config '{"enable_kv_cache_events":true,"endpoint":"tcp://*:5557"}'`，publisher 自动 `zmq`。
- **重放**：`--kv-events-config.replay-endpoint tcp://*:5558 --kv-events-config.buffer-steps 20000`。
- **背压**：`--kv-events-config.hwm 200000 --kv-events-config.max-queue-size 200000`。
- **topic**：`--kv-events-config.topic mycluster`。

## 与其它模块/系统配合

- **BlockPool（[`01-engine-core/kv-cache-management/block-pool.md`](../01-engine-core/kv-cache-management/block-pool.md)）**：block 分配/释放/驱逐时发布事件（经 `kv_event_publisher`）。
- **Scheduler（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`take_events`/`publish` 钩子；`SchedulerStats` 含 KV 事件统计。
- **ObservabilityConfig（[observability-config.md](observability-config.md)）**：`kv_cache_metrics=True` 时采样 block 驻留指标，与事件互补（事件是 push 流，指标是 pull 采样）。
- **CacheConfig（[cache-config.md](cache-config.md)）**：`enable_prefix_caching` 是事件的前置条件（`VllmConfig` 校验）。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init` 校验 `kv_events_config` 与 `enable_prefix_caching`、`publisher` 与 `enable_kv_cache_events` 的一致性。

## 历史版本演进

- **v0.8/v0.9**：`KVEventsConfig` 引入，zmq publisher；`enable_kv_cache_events`/`endpoint`/`replay_endpoint`/`buffer_steps`/`hwm`/`max_queue_size`/`topic`。
- **v0.10**：与 `kv_cache_metrics`（`ObservabilityConfig`）互补的观测面成形；`VllmConfig` 校验 prefix caching 一致性。
- **v0.11 / v0.12 / main**：事件发布与 multi-connector/FlexKV 协同；replay 用于跨实例 prefix cache 预热。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [kv-transfer-config.md](kv-transfer-config.md) — KV 搬运（与本观测面正交）。
- [cache-config.md](cache-config.md) — `enable_prefix_caching` 是事件前置。
- [observability-config.md](observability-config.md) — `kv_cache_metrics` 互补采样。
- [../01-engine-core/kv-cache-management/block-pool.md](../01-engine-core/kv-cache-management/block-pool.md) — 事件发布消费方。
