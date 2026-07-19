# KVCacheManager（门面）

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [KV 缓存管理](README.md) > KVCacheManager

源码：`vllm/v1/core/kv_cache_manager.py`（约 646 行）。`KVCacheManager` 是调度器与底层块池/协调器之间的唯一门面，把多类型 KV cache 的复杂性封装在 `KVCacheBlocks` 接口背后。

## 是什么

### `KVCacheBlocks`（`kv_cache_manager.py:29`）

不可变快照 dataclass：
- `blocks: tuple[Sequence[KVCacheBlock], ...]`：外层索引 = KV cache group id，内层是该 group 内的 block 序列。
- `__add__`：按 group 拼接两个 instance（prefix cache 命中 + 新分配）。
- `get_block_ids(allow_none=False) -> tuple[list[int], ...] | None`：转成纯 int；`allow_none=True` 且全空时返回 None（V1 路径优化）。
- `get_unhashed_block_ids_all_groups()`：返回每组中 `block_hash is None` 的 block_id（用于缓存键清理）。
- `new_empty()`：构造同 group 数量的空 instance。

### `KVCacheManager`（`kv_cache_manager.py:114`）

构造参数：`kv_cache_config`、`max_model_len`、`scheduler_block_size`、`hash_block_size`、`max_num_batched_tokens`、`enable_caching`、`use_eagle`、`log_stats`、`enable_kv_cache_events`、`dcp_world_size`/`pcp_world_size`、`metrics_collector`、`watermark`。

核心字段：
- `coordinator: KVCacheCoordinator`（由 `get_kv_cache_coordinator` 工厂按 group 数与 caching 开关选择 NoPrefixCache/Unitary/Hybrid）。
- `block_pool = coordinator.block_pool`：物理块池。
- `num_kv_cache_groups`、`kv_cache_config`：保存 spec 元数据用于事件注解。
- `watermark_blocks = int(watermark * num_blocks)`：准入门控的水线。
- `kv_cache_event_metadata`：`tuple[(spec_kind, sliding_window), ...]`，在 `take_events` 时把 `BlockStored` 注解完整。
- `empty_kv_cache_blocks`：预构造空 instance，避免 GC。

主要方法：
- `get_computed_blocks(request) -> (KVCacheBlocks, int)`（`kv_cache_manager.py:206`）：查询 prefix cache 命中（`skip_reading_prefix_cache` 时直接返回空）。
- `allocate_slots(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0, has_scheduled_reqs=True) -> KVCacheBlocks | None`（`kv_cache_manager.py:248`）：核心准入 + 分配。
- `free(request)` / `pop_blocks_for_free(request)` / `evict_blocks(block_ids)`
- `cache_blocks(request, num_computed_tokens)`：显式缓存（async 调度路径）。
- `reset_prefix_cache()` / `take_events()` / `usage` / `make_prefix_cache_stats()`
- `get_blocks(request_id)` / `get_block_ids(request_id)` / `get_block_ids_for_computed_tokens`
- `take_new_block_ids()`：drain 各 manager 的 `new_block_ids`（供 Mamba 等需要 zeroing 的场景）。
- `new_step_starts()`：透传给 coordinator。

## 为什么

- **隐藏内部结构**：调度器只看到 `KVCacheBlocks` 与 `allocate_slots/free`，不需要知道有多少 group、哪个 group 是 SWA、是否 hybrid。这让 scheduler 主循环代码简洁。
- **watermark 准入**：`watermark_blocks` 仅对 WAITING/PREEMPTED 且 `has_scheduled_reqs=True` 时施加，避免空引擎时也留水线导致首请求启动慢。
- **三段式 allocate_slots**（注释 `kv_cache_manager.py:294-315` 的 ASCII 图）：
  1. Free 不必要的 block（sliding window gap 等）+ 检查空闲块数。
  2. 处理 prefix tokens（local + connector 外部）：先 `allocate_new_computed_blocks` 把命中块"接上"，必要时为 `ext_comp` tokens 分配新块。
  3. 为 `new + lookahead` tokens 分配新块。
- **two-phase 外部块分配**（`coordinator.allocate_new_computed_blocks`，issue #33775）：先 `add_local_computed_blocks`（每个 group 触摸命中块、增 ref_cnt），再 `allocate_external_computed_blocks`；避免早 group 的外部 `get_new_blocks` 驱逐晚 group 未触摸的命中块。
- **`full_sequence_must_fit`**：sliding window/chunked-local 等回收型 spec 需要准入时检查整个序列能否放下，避免 chunked prefill 中途 OOM。
- **`reserved_blocks`**：KV connector 异步加载时，要预留块给已 in-flight 的 prefill 序列完成；`_inflight_prefill_reserved_blocks` 在 scheduler 端计算。
- **spec token 限制缓存**：注释 `kv_cache_manager.py:328-330` 说明：`new` token 含已拒绝的 draft token，只能 cache verified token，故 `num_tokens_to_cache = min(total_computed + num_new, request.num_tokens)`。
- **event 注解分层**：`BlockPool` 只发结构化 `BlockStored`，`KVCacheManager.take_events` 用 `kv_cache_event_metadata` 注入语义（spec_kind、sliding_window），让 BlockPool 不持有 spec 元数据。
- **auto-fit 回流**：`EngineCoreReadyResponse` 通过 `cache_config.kv_cache_size_tokens`/`kv_cache_max_concurrency` 把 manager 实际容量回传前端。

## 怎么做

### allocate_slots 流程

```mermaid
flowchart TD
    A[allocate_slots 调用] --> B{num_new_tokens==0 且 ext==0?}
    B -- 是 --> ERR[raise ValueError]
    B -- 否 --> C[num_local_computed = req.num_computed + new_computed<br/>total = min(local+ext, max_model_len)]
    C --> D{has_scheduled_reqs 且 WAITING/PREEMPTED?}
    D -- 是 --> WM[watermark_blocks = self.watermark_blocks]
    D -- 否 --> WM0[watermark_blocks = 0]
    WM --> E{full_sequence_must_fit?}
    WM0 --> E
    E -- 是 --> FS[get_num_blocks_to_allocate(全序列, apply_admission_cap=True)<br/>若 +watermark > free_blocks: return None]
    E -- 否 --> SW
    FS --> SW[remove_skipped_blocks(SWA gap eviction)]
    SW --> GC[get_num_blocks_to_allocate(num_tokens_need_slot)]
    GC --> CK{required + watermark > free - reserved?}
    CK -- 是 --> RTN[return None 触发抢占]
    CK -- 否 --> AC[allocate_new_computed_blocks<br/>local 各组 + external 各组 two-phase]
    AC --> AN[allocate_new_blocks 各组]
    AN --> DC{delay_cache_blocks 或 not enable_caching?}
    DC -- 是 --> RT[return create_kv_cache_blocks new_blocks]
    DC -- 否 --> CB[cache_blocks(min(total+new, num_tokens))]
    CB --> RT
```

### get_computed_blocks 命中查找

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0

# 全命中时退一格以重算最后 token 取 logits
max_cache_hit_length = request.num_tokens - 1
computed_blocks, num_new_computed_tokens = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
# hybrid 模型在 coordinator 内可能产生 num_uncached_common_prefix_tokens
# （用于 Marconi APC，scheduler 端读取）
return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

### get_num_common_prefix_blocks（cascade attention）

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
    return self.coordinator.get_num_common_prefix_blocks(running_request_id)
```
每个 group 返回一个数：所有持有 KV block 的请求（≥ scheduled 请求数）的公共前缀 block 数。调度器把它放进 `SchedulerOutput.num_common_prefix_blocks`，worker 据此决定是否启用 cascade attention。注释 `kv_cache_manager.py:540-554` 说明：未调度但持有 block 的 request 可能拉低这个值（边缘 case 返回 0）。

### take_new_block_ids（Mamba zeroing）

```python
def take_new_block_ids(self) -> list[int]:
    ids: list[int] = []
    for mgr in self.coordinator.single_type_managers:
        ids.extend(mgr.take_new_block_ids())
    return ids
```
每个 `SingleTypeKVCacheManager` 维护 `new_block_ids: list[int]`，在 `allocate_new_blocks` 时追加；scheduler 在 `schedule` 末尾 `take_new_block_ids()`（`scheduler.py:1079`），仅当 `needs_kv_cache_zeroing`（Mamba 等需清零的 spec）时填入 `SchedulerOutput.new_block_ids_to_zero`，worker 在 forward 前 zeroing 这些 block 防止 stale NaN。

### free / pop_blocks_for_free / evict_blocks

- `free(request)`：常规路径，coordinator.free → block_pool 归还，REV 顺序。
- `pop_blocks_for_free(request)`：deferred free 路径（PP/async + KV consumer），抽取 block 不归还，由 scheduler 的 `_drain_deferred_frees` 调 `block_pool.free_blocks(reversed(blocks))`。
- `evict_blocks(block_ids)`：KV connector 报告 invalid block 时按 id 驱逐（仅清哈希表项，ref_cnt>0 的 block 物理上不释放）。

## 与其它模块/系统配合

- **[Scheduler](../scheduler/scheduler.md)**：唯一消费者；每步 `new_step_starts` + 多次 `allocate_slots` + `cache_blocks`。
- **[coordinator.md](./coordinator.md)**：`get_kv_cache_coordinator` 工厂决定子类；`find_longest_cache_hit` 是命中算法核心。
- **[block-pool.md](./block-pool.md)**：`block_pool` 字段直接暴露供 coordinator 与 single-type manager 共用。
- **[spec.md](./spec.md)**：`kv_cache_config.kv_cache_groups` 决定 manager 组数与类型；`kv_cache_event_metadata` 由 spec kind 注入事件。
- **[EngineCore](../engine-core-process.md)**：`_initialize_kv_caches` 构造 `KVCacheConfig` 并设置 `cache_config.kv_cache_size_tokens`。
- **[KV connector](../../15-kv-cache-offload/README.md)**：`num_external_computed_tokens`/`delay_cache_blocks` 路径；`evict_blocks` 由 connector invalid_block_ids 触发。
- **[Encoder cache](./encoder-cache.md)**：`num_encoder_tokens` 走 `CrossAttentionManager.get_num_blocks_to_allocate` 单独路径。
- **[02-execution](../../02-execution/README.md)**：worker 端 `ModelRunner` 按 `SchedulerOutput.num_common_prefix_blocks` 决定 cascade；按 `new_block_ids_to_zero` zeroing。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 `BlockSpaceAllocator` 分 `gpu_allocator`/`cpu_allocator`，单一 attention 类型；`allocate_slots` 复杂且耦合 SequenceGroup。
- **v0.7（v1 落地）**：`KVCacheManager` + `KVCacheBlocks` 抽出；`KVCacheCoordinator` 引入，但仅支持单类型全注意力。
- **v0.7.x**：SlidingWindow/Mamba spec 引入；`SingleTypeKVCacheManager` 抽象基类 + `get_manager_for_kv_cache_spec` 工厂。
- **v0.8（v1 默认）**：`HybridKVCacheCoordinator` 上线，混合 attention 模型（Gemma2 等）原生支持；`watermark_blocks` 准入；`scheduler_reserve_full_isl` 等门控完善。
- **v0.9**：`num_external_computed_tokens`/`delay_cache_blocks` KV connector 路径；`pop_blocks_for_free` + deferred free 围栏；two-phase 外部块分配（issue #33775）。
- **v0.10**：Marconi APC（`num_uncached_common_prefix_tokens`）；`take_new_block_ids` zeroing 机制；`routed_experts` 与 KV 无关但同 manager 注入。
- **v0.11 / v0.12 / main**：DCP/PCP（decode/prefill context parallel）下 `block_size *= dcp_world_size * pcp_world_size`；NVFP4/INT4 KV quant spec；厂商自定义 spec 通过 `register_kv_cache_spec` 接入。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [coordinator.md](./coordinator.md) — `allocate_slots` 的实际执行者。
- [block-pool.md](./block-pool.md) — 物理 block 生命周期。
- [spec.md](./spec.md) — `KVCacheConfig` 如何形成。
- [../scheduler/scheduler.md](../scheduler/scheduler.md) — `allocate_slots` 的调用方。
