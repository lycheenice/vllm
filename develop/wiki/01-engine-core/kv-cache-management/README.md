# KV 缓存管理子模块

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [KV 缓存管理](README.md)

`vllm/v1/core/` 与 `vllm/v1/kv_cache_interface.py` 共同实现 v1 引擎的 KV 缓存抽象。这是 v1 相对 v0 最重要的重构之一：从"单一全注意力块池"演化为"多类型 spec + 多组 manager + 协调器"的混合架构，原生支持 FullAttention/MLA/SlidingWindow/Mamba/ChunkedLocal/CrossAttention 等同存。

## 子模块边界

KV 缓存管理位于调度器之下、Block 池之上，提供：
- **物理块管理**：`BlockPool` 负责块的分配/释放/LRU 驱逐/哈希表。
- **类型管理**：每种 attention 类型一个 `SingleTypeKVCacheManager` 子类，维护该组请求级 block 记账与命中查找。
- **协调**：`KVCacheCoordinator` 系列把多个 single-type manager 统一暴露给 `KVCacheManager` 门面。
- **门面**：`KVCacheManager` 是调度器唯一入口，封装 `KVCacheBlocks` 接口。
- **Spec 定义**：`kv_cache_interface.py` 定义全部 `KVCacheSpec` 子类与 `KVCacheConfig`；`kv_cache_spec_registry.py` 提供注册机制供厂商扩展。
- ** Encoder 缓存**：`encoder_cache_manager.py` 管理多模态编码器输出的 LRU 缓存。
- **指标**：`kv_cache_metrics.py` 采样块生命周期、记录驱逐事件。

## 关键设计

1. **混合块池**：所有 KV cache group 共享同一个 `BlockPool`（`num_blocks` 个 `KVCacheBlock`），避免按 group 划分池造成的碎片。
2. **prefix cache 全局哈希表**：`BlockHashToBlockMap` 按 `(block_hash, group_id)` 索引；同一物理块可被多 group 命中（`cached_block_hashes_by_block` 维护反向映射以便驱逐时清理）。
3. **统一调度粒度**：`scheduler_block_size` = 所有 group `block_size` 的 LCM；hash 用更细的 `hash_block_size`，使混合模型也能 fine-grained 命中。
4. **Hybrid 不动点算法**：`HybridKVCacheCoordinator.find_longest_cache_hit` 通过多类型迭代收缩候选长度，收敛到所有类型都能命中前缀。
5. **Block-at-a-time 接口**：`KVCacheBlocks` 是不可变快照，`get_block_ids()` 返回 `tuple[list[int], ...]`（外层按 group），调度器不直接接触 `KVCacheBlock`。

## 子目录导航表

| 文档 | 简介 | 主要源码 |
|---|---|---|
| [kv-cache-manager.md](kv-cache-manager.md) | `KVCacheManager` 门面与 `KVCacheBlocks` 接口 | `vllm/v1/core/kv_cache_manager.py` |
| [coordinator.md](coordinator.md) | `HybridKVCacheCoordinator`/`UnitaryKVCacheCoordinator`：混合协调与不动点命中 | `vllm/v1/core/kv_cache_coordinator.py` |
| [block-pool.md](block-pool.md) | `BlockPool`：物理块、LRU、哈希表、KV 事件 | `vllm/v1/core/block_pool.py` |
| [spec.md](spec.md) | `KVCacheSpec` 体系 + `KVCacheSpecRegistry`：多类型 spec 注册与合并 | `vllm/v1/kv_cache_interface.py`、`kv_cache_spec_registry.py` |
| [encoder-cache.md](encoder-cache.md) | `EncoderCacheManager`/`EncoderDecoderCacheManager`：多模态编码器输出 LRU | `vllm/v1/core/encoder_cache_manager.py` |
| [metrics.md](metrics.md) | `KVCacheMetricsCollector`：块生命周期采样与驱逐事件 | `vllm/v1/core/kv_cache_metrics.py` |

## 与调度器/EngineCore 的协作

- EngineCore 在 `_initialize_kv_caches` 调 `register_all_kvcache_specs` + `model_executor.get_kv_cache_specs()` + `get_kv_cache_configs` + `generate_scheduler_kv_cache_config`，最终 `Scheduler.__init__` 用得到的 `KVCacheConfig` 构造 `KVCacheManager`。
- `Scheduler.schedule` 每步 `kv_cache_manager.new_step_starts()` → `get_computed_blocks`（waiting）或直接 `allocate_slots`（running）→ `_update_after_schedule` 推进 `num_computed_tokens`。
- `update_from_output` 后，对 RUN 且未完成请求调 `cache_blocks`（async 调度在 `AsyncScheduler._update_request_with_output` 内）。

[← 返回引擎核心首页](../README.md)
