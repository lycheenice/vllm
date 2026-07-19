# cache.py · 处理器缓存（Sender / Receiver / SHM）

[← Wiki 首页](../README.md) > [多模态](../README.md) > cache

## 是什么

`vllm/multimodal/cache.py` 实现多模态**处理器输出**的跨进程缓存体系。它把"HF Processor 已经处理过的张量 + 占位符更新"按 `mm_hash` 缓存，避免同一张图被多个请求重复处理、重复 IPC。整体分为 sender（API/前端进程 P0）与 receiver（engine core / worker 进程 P1）两侧，按 `mm_processor_cache_type` 配置切换三种实现：

| 配置 | P0 sender | P1 receiver | 适用 |
|---|---|---|---|
| `processor_only` | `MultiModalProcessorOnlyCache`（存全量 item） | 无 | 多 API 进程 / DP>1 无 IPC |
| `lru` | `MultiModalProcessorSenderCache`（仅存元数据） | `MultiModalReceiverCache`（P1 存张量） | 单 API 进程，msgspec IPC |
| `shm` | `ShmObjectStoreSenderCache`（P0 存 SHM 地址） | `ShmObjectStoreReceiverCache`（worker 直接读 SHM） | 单 API 进程，零拷贝 |

所有缓存的 value 类型由 `MultiModalCacheValue` 别名（`:87`）汇总，size 度量统一走 `MultiModalCache.get_leaf_size` / `get_item_size`。

## 为什么

多模态 HF 处理（特别是大图 / 长视频）耗时可数秒，而同一张图往往被多个请求复用（图床、 batching）。但 vLLM 是多进程架构（P0 API server ↔ P1 engine core ↔ worker），简单 LRU 摆在哪一侧都不够：

- 摆 P0：engine 仍要每请求收到全量 tensor 走 IPC，带宽爆炸。
- 摆 P1：API 不能预先知道是否命中，仍要做 HF 处理。

`BaseMultiModalCache` 的设计是**两侧镜像**：P0 用 `is_cached()` 多次查询不更新驱逐序，P0 与 P1 严格按相同顺序调 `get_and_update()`，使两侧 key 集合始终一致；P0 因此能"看着自己的缓存"判断 P1 是否命中，命中即把 `data` 置 `None` 走轻量 IPC。`shm` 模式进一步把 tensor 落到 `SingleWriterShmObjectStorage`，P0 只发 `(address, monotonic_id)` 句柄，worker 直接零拷贝读。

## 怎么做

### MultiModalCache 工具类

`MultiModalCache`（`:98`）是 classmethod 工具集：

- `get_leaf_size(leaf)`（`:100`）：递归展开多层包装——`MultiModalProcessorCacheItem` 取 `.item`、`...Metadata` 取 `.item_size`、`MultiModalKwargsItems/Item/FieldElem` 取 `.data`、Tensor 取 `.nbytes`、其它走 `sys.getsizeof`。
- `get_item_size(value, *, debug)`（`:120`）：用 `json_reduce_leaves(operator.add, json_map_leaves(get_leaf_size, value))` 求总字节，debug 时打印 `format_gib`。
- `get_lru_cache(capacity_gb, value_type, *, debug)`（`:158`）：构造 `LRUCache`，`getsizeof=lambda x: get_item_size(x)`，capacity 按 `GiB_bytes * capacity_gb`。LRU 驱逐以字节而非条数为单位。

### BaseMultiModalCache

`BaseMultiModalCache`（`:175`，`Generic[_I, _O]`）定义两侧共同协议：

- `get_and_update_item(mm_item, mm_hash) -> _O`：abstract，单 item 的"查 + 写 + 更新驱逐序"。
- `get_and_update(mm_items, mm_hashes) -> list[_O]`（`:221`）：批量调单 item 版本，要求 `len(mm_items) == len(mm_hashes)`。
- `clear_cache()`：abstract。

类文档（`:178`）描述 P0/P1 的镜像模型：`is_cached()` 在 P0 可任意调，`get_and_update()` 必须两侧同序。

### ProcessorCache 类型

- `MultiModalProcessorCacheItem`（`:42`）：P0-only 模式存全量，含 `item: MultiModalKwargsItem` + `prompt_updates`。
- `MultiModalProcessorCacheItemMetadata`（`:62`）：P0-IPC 模式只存 `item_size` + `prompt_updates`，避免在 P0 留张量。

`MultiModalProcessorCacheInItem`（`:252`）= `(item, prompt_updates) | None`（None 表示 P0 期望命中）；`MultiModalProcessorCacheOutItem`（`:257`）= `(item | None, prompt_updates)`（item None 表示已缓存）。

### BaseMultiModalProcessorCache（P0 侧）

`BaseMultiModalProcessorCache`（`:262`）在 `BaseMultiModalCache` 上加：

- `is_cached_item(mm_hash) -> bool`：abstract，**不更新驱逐序**。
- `is_cached(mm_hashes) -> list[bool]`：批量。
- `close()`：默认 no-op。
- `touch_sender_cache_item(mm_hash)`：abstract，更新驱逐序不发 IPC。
- `make_stats(*, delta) -> CacheInfo`：abstract，给可观测性用。

#### MultiModalProcessorOnlyCache（`:326`）

P0-only。命中→返回 cache 里的 `(item, prompt_updates)`；未命中→必须 `mm_item is not None`，写入 `MultiModalProcessorCacheItem`，原样返回。`touch_sender_cache_item` 调 `LRUCache.touch`。

#### MultiModalProcessorSenderCache（`:379`）

P0-IPC，只存元数据。命中→返回 `(None, cached.prompt_updates)`（让上层把 `data` 置空，省 IPC）；未命中→写入 `MultiModalProcessorCacheItemMetadata`（只记 size + prompt_updates），原样返回 mm_item。

#### ShmObjectStoreSenderCache（`:437`）

构造时建 `SingleWriterShmRingBuffer`（`create=True`，name 来自 `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`）+ `SingleWriterShmObjectStorage`（`n_readers=world_size`，`max_object_size=mm_shm_cache_max_object_size_mb * MiB`，serde=`MsgpackSerde`）。还维护本地 `_p0_cache: dict[str, prompt_updates]`（prompt_updates 必须留 P0，因部分模型依赖处理后张量）。

`get_and_update_item`（`:488`）：

1. 命中 SHM→`_hits++`，取 `(address, monotonic_id)`，`address_as_item` 包装成 `MultiModalKwargsItem({"address":..., "monotonic_id":...})`（field 用 `MultiModalBatchedField`），返回。
2. 未命中→`put(mm_hash, item)` 到 SHM；若 `_p0_cache` 比实际 key_index 大 2 倍以上，`remove_dangling_items` 清悬挂键；写 `_p0_cache[mm_hash]=prompt_updates`，返回地址 item。
3. `put` 抛 `ValueError`（oversize 或 duplicate key，"already exists" 警告抑制）→原样返回；`MemoryError`（SHM 满）→debug 日志，原样返回。

`touch_sender_cache_item` 调 `_shm_cache.touch(mm_hash)`；`close` 调 `_shm_cache.close()`。

### BaseMultiModalReceiverCache 与实现（P1 侧）

`BaseMultiModalReceiverCache`（`:584`，`Generic[MultiModalKwargsItem | None, MultiModalKwargsItem]`）：

- `get_and_update_features(mm_features)`（`:589`）：先 `touch_receiver_cache_item` 所有 feature（防更新过程中被驱逐），再逐个 `get_and_update_item`。cache_key 用 `feature.mm_hash or feature.identifier`（mm_hash 让 LoRA 间共享）。
- `touch_receiver_cache_item(mm_hash, mm_item=None)`：abstract。

#### MultiModalReceiverCache（`:630`）

P1-`lru`。命中→返回缓存 item；未命中→断言 mm_item not None，写入，返回。`touch` 调 `LRUCache.touch`。

#### ShmObjectStoreReceiverCache（`:678`）

P1-`shm`，构造 SHM（`create=False`，reader 端）+ `reader_lock=shared_worker_lock`。`get_and_update_item`：若 item 含 `"address"` key，从 SHM `get(address, monotonic_id)` 取回 `MultiModalKwargsItem`；否则原样返回（P0 未缓存走 msgspec IPC 的 fallback）。`touch_receiver_cache_item` 在 item 有 address 时调 SHM `touch(mm_hash, address, monotonic_id)` 增 reader_count。

## 与其它模块/系统配合

- **registry.py**：`_get_cache_type` + 四个 `*_from_config` 方法按 config 选实现，详见 [registry.md](registry.md)。
- **hasher.py**：cache key 由 `MultiModalHasher.hash_kwargs` 生成，详见 [hasher.md](hasher.md)。
- **processing/processor.py**：`_cached_apply_hf_processor` 通过 `cache.is_cached` / `get_and_update` 决定是否真跑 HF，详见 [processing.md](processing.md)。
- **inputs.py**：`MultiModalKwargsItem` 是缓存 value 单元；`ShmObjectStoreSenderCache.address_as_item` 用 `MultiModalBatchedField` 伪装地址为 item。
- **v1/engine**：`EngineCore` 在收到请求时调 `mm_receiver_cache.get_and_update_features(request.mm_features)`（`v1/engine/core.py:864`），把命中 item 从 `None` 还原成 `MultiModalKwargsItem`，详见 [v1-integration.md](v1-integration.md)。
- **v1/worker**：`shm` 模式下 worker 用 `ShmObjectStoreReceiverCache` 读 SHM，`shared_worker_lock` 来自 worker 进程组。
- **distributed/shm_object_storage**：`SingleWriterShmObjectStorage` / `SingleWriterShmRingBuffer` / `MsgpackSerde` 由 `vllm/distributed/device_communicators/shm_object_storage.py` 提供（属于 `07-distributed`，本子系统仅消费）。
- **utils.cache**：`LRUCache` / `CacheInfo` 来自 `vllm/utils/cache.py`。
- **配置**：`mm_processor_cache_gb` / `mm_processor_cache_type` / `mm_shm_cache_max_object_size_mb` 来自 [`MultiModalConfig`](../10-config/multimodal-config.md)。

## 历史版本演进

- **v0.5（LLaVA 初版）**：无处理器缓存，每个请求都跑全量 HF 处理。
- **v0.7（v1 化）**：引入 `MultiModalProcessorCache`（P0 单侧 LRU），key 用 image URL 哈希；P1 无缓存，全量 IPC tensor。
- **v0.9（hash+cache）**：P0/P1 双侧缓存合入，`MultiModalProcessorOnlyCache` / `MultiModalProcessorSenderCache` / `MultiModalReceiverCache` 三件套出现，`is_cached` 与 `get_and_update` 分离，`touch_sender_cache_item` ABI 稳定。
- **v0.10**：`ShmObjectStoreSenderCache` / `ShmObjectStoreReceiverCache` 加入，复用 `SingleWriterShmObjectStorage`；`address_as_item` 用 `MultiModalBatchedField` 让地址伪装成普通 item；`remove_dangling_items` 防止 P0 元数据膨胀。
- **v0.11（EVS）**：`get_and_update_features` 改用 `mm_hash or identifier` 作为 cache_key，让 receiver cache 跨 LoRA 共享编码器输出（配合 `_get_mm_identifier` 的 LoRA 前缀化 identifier）。
- **main**：`MultiModalCache.get_item_complexity` 加入（调试用，统计叶节点数）；`put` 的 `MemoryError` 路径细化（protected items 阻止驱逐时降级为不缓存）；`make_stats` 支持 `delta` 增量上报。

[← 返回多模态首页](../README.md)

## 参见

- [hasher.md](hasher.md)：cache key 的来源。
- [registry.md](registry.md)：根据 config 实例化哪种 cache。
- [processing.md](processing.md)：`is_cached` / `get_and_update` 的调用点。
- [inputs.md](inputs.md)：缓存的 value 类型。
- [v1-integration.md](v1-integration.md)：P0/P1 在 v1 引擎中的装配位置。
