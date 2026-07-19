# BlockPool（物理块池）

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [KV 缓存管理](README.md) > BlockPool

源码：`vllm/v1/core/block_pool.py`（约 723 行）。`BlockPool` 是 KV cache 物理块的底层管家：分配、释放、LRU 驱逐、prefix cache 哈希表、KV 事件发布。

## 是什么

### 协助类 `BlockHashToBlockMap`（`block_pool.py:34`）

prefix cache 哈希表，`{BlockHashWithGroupId: KVCacheBlock | dict[int, KVCacheBlock]}`：
- 单 block 命中：值是 `KVCacheBlock`。
- 多 block 冲突（同 hash 多个物理块）：值是 `{block_id: KVCacheBlock}` dict。
- 方法：`get_one_block(key)` / `contain(key, block_id)` / `insert(key, block)` / `pop(key, block_id)`。
- 设计取舍：注释 `# NOTE #1` 说明不去重——同一 hash 的多个块都保留，保证 block_id 不变（block table append-only）；`# NOTE #2` 用 union 减少 GC 开销。

### `BlockPool`（`block_pool.py:144`）

构造参数：`num_gpu_blocks`、`enable_caching`、`hash_block_size`、`enable_kv_cache_events`、`metrics_collector`。

字段：
- `blocks: list[KVCacheBlock]`：所有物理块（按 idx 索引，idx 即 block_id）。
- `free_block_queue: FreeKVCacheBlockQueue`：双向链表，按驱逐顺序维护空闲块；caching 启用时含"ref_cnt=0 但有 hash"的候选驱逐块。
- `cached_block_hash_to_block: BlockHashToBlockMap`：hash → block 主索引。
- `cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]]`：block_id → 额外 hash 集合（一个 block 可被多个 hash key 引用，partial cache 等）。
- `null_block`：`block_id=0` 的占位块，`is_null=True`，不参与 ref_cnt 维护。
- `kv_event_queue: list[KVCacheEvent]`：待发布事件（`BlockStored`/`BlockRemoved`/`AllBlocksCleared`）。
- `metrics_collector`：可选 `KVCacheMetricsCollector`。

核心方法：
- `get_cached_block(block_hash, kv_cache_group_ids) -> list[KVCacheBlock] | None`：每组都要命中才返回，否则 None。
- `cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask=None)`：把新满 block 加入哈希表，发 `BlockStored` 事件。
- `cache_partial_block(request, block, num_tokens, kv_cache_group_id, block_size) -> BlockHashWithGroupId | None`：为已存在 block 注册 sub-block-size 的部分命中键。
- `get_new_blocks(num_blocks) -> list[KVCacheBlock]`：从 free_block_queue 取 N 个块；caching 时 evict 占用的 hash 块、inc ref_cnt、调 `metrics_collector.on_block_allocated`。
- `touch(blocks)`：ref_cnt +1；ref_cnt 从 0→1 时从 free queue 移除（命中复用）。
- `free_blocks(ordered_blocks)`：ref_cnt -1；归 0 的非 null 块回到 free queue——无 hash 块 prepend（优先驱逐），有 hash 块 append（保留为缓存候选）。
- `evict_blocks(block_ids)`：按 id 从哈希表驱逐（ref_cnt>0 不释放物理块）。
- `reset_prefix_cache() -> bool`：仅当除 null_block 外全部空闲时成功；清空哈希表与所有 block 的 hash，发 `AllBlocksCleared` 事件。
- `get_num_free_blocks` / `get_usage` / `take_events()`。

## 为什么

- **共享池**：所有 KV cache group 共用一个 BlockPool，避免按类型分池碎片化；`KVCacheBlock` 与物理 GPU 张量一一对应。
- **LRU+ref_cnt 混合**：`FreeKVCacheBlockQueue` 是双向链表，按"释放顺序"维护；ref_cnt=0 且有 hash 的块仍留在 queue 尾部作为驱逐候选——caching hit 时 `touch` 把它捞出复活，免费复用已写入的 KV。
- **free 时的分流**：`free_blocks` 把无 hash 块 prepend（队首，先被分配驱逐）、有 hash 块 append（队尾，保留为缓存）；这样新请求优先拿"干净块"，缓存命中块能尽量保留。
- **block_id 稳定**：注释 `NOTE #1` 强调 block_id 一经分配就稳定，使 worker 的 block_table 是 append-only，避免重写。
- **Partial cache**：`cache_partial_block` 让 sub-block-size 的前缀也能被命中（大 block_size 场景降低粒度），通过 `cached_block_hashes_by_block` 反向索引便于 eviction 时一次性清理所有指向该 block 的 hash key。
- **驱逐事件**：`BlockRemoved` 事件让 KV connector 知道哪些块被驱逐（必要时重新传输）；metrics_collector 记录块生命周期。
- **reset 安全**：`reset_prefix_cache` 仅在所有非 null 块都空闲时执行，否则警告失败——避免运行中请求的 block 被错误清空。

## 怎么做

### get_new_blocks 分配

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)
    ret = self.free_block_queue.popleft_n(num_blocks)
    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)  # 清 hash、发 BlockRemoved
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
    else:
        for block in ret:
            block.ref_cnt += 1
            ...
    return ret
```

caching 关闭时跳过 evict，简化路径。

### _maybe_evict_cached_block

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
    if self.metrics_collector:
        self.metrics_collector.on_block_evicted(block)
    evicted_hashes = self._remove_cached_block_hashes(block)  # 反向索引清理
    if not evicted_hashes:
        return False  # 无 hash，无需驱逐
    self._emit_block_removed_events(evicted_hashes)
    return True
```

`_remove_cached_block_hashes` 同时清理 `block.block_hash`（主 key）与 `cached_block_hashes_by_block[block_id]`（partial key 集合），然后 `block.reset_hash()`。

### cache_full_blocks 流程

```python
new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
# block_size == hash_block_size: 直接用 request.block_hashes
# 否则: BlockHashListWithBlockSize 把 hash 序列重映射
for i, blk in enumerate(new_full_blocks):
    if blk.is_null or (block_mask and not block_mask[i]):
        continue  # SWA 等跳过的 block 不入缓存
    block_hash = new_block_hashes[i]
    if blk.block_hash is not None:
        # partial→full 升级：先删旧 hash
        removed = self._remove_cached_block_hashes(blk)
        self._emit_block_removed_events(removed)
    self._insert_block_hash(hash_with_gid, blk, num_tokens=...)
    if enable_kv_cache_events:
        new_hashes.append(maybe_convert_block_hash(block_hash))

if enable_kv_cache_events:
    # 构造 BlockStored 事件，含 parent_block_hash、token_ids、extra_keys、lora_id、group_idx
    self.kv_event_queue.append(BlockStored(...))
```

`block_mask` 参数：SWA 等稀疏命中 group 只把"真正能服务命中"的 block 入哈希表，让不可命中的 block 不占索引空间。

### free_blocks 顺序

```python
def free_blocks(self, ordered_blocks):
    blocks_with_hash, blocks_without_hash = [], []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            (blocks_with_hash if block.block_hash else blocks_without_hash).append(block)
    # 无 hash 优先驱逐：prepend 到队尾（popleft 优先取它）
    self.free_block_queue.prepend_n(blocks_without_hash)
    # 有 hash 保留为缓存候选：append 到队尾（最后才驱逐）
    self.free_block_queue.append_n(blocks_with_hash)
```

调度器调 `free_blocks(reversed(blocks))` 让"tail block 先被驱逐"（caching 模式下保序符合 LRU 语义）。

### reset_prefix_cache

```python
def reset_prefix_cache(self) -> bool:
    num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
    if num_used_blocks != 1:  # 只有 null_block
        logger.warning("Failed to reset prefix cache...")
        return False
    self.cached_block_hash_to_block = BlockHashToBlockMap()
    self.cached_block_hashes_by_block.clear()
    for block in self.blocks:
        block.reset_hash()
    if self.metrics_collector:
        self.metrics_collector.reset()
    if self.enable_kv_cache_events:
        self.kv_event_queue.append(AllBlocksCleared())
    return True
```

scheduler 调用前会先 `reset_prefix_cache(reset_running_requests=True)` 把所有 running 抢占，使所有非 null block ref_cnt 归 0。

### evict_blocks（KV connector）

```python
def evict_blocks(self, block_ids: set[int]) -> None:
    for block_id in block_ids:
        assert block_id < len(self.blocks)
        block = self.blocks[block_id]
        self._maybe_evict_cached_block(block)
```
仅清哈希表项，ref_cnt>0 的 block 物理上仍在使用（不归还 free queue）。这是 KV connector invalid_block_ids 路径：让其它请求后续无法命中该块，但当前持有者继续用。

## 与其它模块/系统配合

- **[coordinator.md](./coordinator.md)**：`block_pool` 字段共享给所有 single-type manager；`null_block` 作为占位 `block_id=0`。
- **[kv-cache-manager.md](./kv-cache-manager.md)**：`evict_blocks`/`reset_prefix_cache`/`take_events` 透传接口；`get_num_common_prefix_blocks` 通过 manager 间接用 block_pool 的哈希表。
- **[SingleTypeKVCacheManager](./spec.md)**：`cache_full_blocks`/`cache_partial_block` 由各 manager 调用；`get_new_blocks`/`touch` 是分配原语。
- **[metrics.md](./metrics.md)**：`on_block_allocated`/`on_block_accessed`/`on_block_evicted` 钩子。
- **[Scheduler](../scheduler/scheduler.md)**：`reset_prefix_cache(reset_running_requests=True)` 先抢占再调 block_pool.reset；`deferred_frees` 最终调 `block_pool.free_blocks(reversed(blocks))`。
- **[KV connector](../../15-kv-cache-offload/README.md)**：`evict_blocks` 处理 invalid block；`BlockRemoved` 事件让 connector 知道驱逐。
- **[KV events](../../16-observability/README.md)**：`take_events` 由 scheduler 在 `update_from_output` 末尾消费，发布到 `kv_event_publisher`。
- **[02-execution](../../02-execution/README.md)**：worker 用 block_id 索引物理 KV 张量；`new_block_ids_to_zero` 让 worker 在 forward 前 zeroing。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 `CpuGpuBlockAllocator` 分 gpu/cpu 池，`BlockAllocator` 内部维护 free + cached；support 全注意力与 sliding window 两种。
- **v0.7（v1 落地）**：`BlockPool` 重写，`FreeKVCacheBlockQueue` 双向链表 + `BlockHashToBlockMap` 哈希表；`null_block` 占位 `block_id=0`；KV 事件框架引入。
- **v0.7.x**：`cache_partial_block` 加入，支持 sub-block-size 命中；`cached_block_hashes_by_block` 反向索引便于清理 partial key。
- **v0.8（v1 默认）**：`block_mask` 参数让 SWA group 跳过不可命中的 block；`metrics_collector` 钩子接入；`evict_blocks` KV connector 路径。
- **v0.9**：`\BlockRemoved` 事件完善；`reset_prefix_cache` 路径与 scheduler 抢占协同；partial cache 推广到 block_size ≠ hash_block_size 场景。
- **v0.10**：`BlockStored` 事件加上 `group_idx`/`extra_keys`；`free_blocks` 的"无 hash prepend / 有 hash append"分流优化 LRU 顺序。
- **v0.11 / v0.12 / main**：`BlockHashListWithBlockSize` 在不同 group block_size 间的桥梁；多前端 scale-out 下 block_pool 仍单进程内（scheduler 进程）。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [coordinator.md](./coordinator.md) — block_pool 的创建者与协调者。
- [kv-cache-manager.md](./kv-cache-manager.md) — scheduler 端的调用接口。
- [metrics.md](./metrics.md) — `metrics_collector` 钩子。
- [../scheduler/preemption.md](../scheduler/preemption.md) — `free_blocks(reversed(blocks))` 的 LRU 语义。
