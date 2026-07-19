# cpu/ — CPU 主层（primary tier）

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > cpu

源码目录：`vllm/v1/kv_offload/cpu/`（9 文件）。这是新管线里**CPU 内存**作为 offload 介质的具体实现，也是 [tiering.md](tiering.md) 多层编排里的 primary tier。它直接与 GPU 通信——所有 secondary tier 的数据都要经它中转。

```
cpu/
├── __init__.py
├── spec.py                # CPUOffloadingSpec
├── manager.py             # CPUOffloadingManager（scheduler 侧）
├── gpu_worker.py          # CPUOffloadingWorker + SingleDirectionOffloadingHandler
├── common.py              # CPULoadStoreSpec + CPUOffloadingMetrics
├── shared_offload_region.py # /dev/shm mmap 共享区
├── swap_blocks_triton.py  # Triton 拷贝 kernel
└── policies/
    ├── __init__.py
    ├── base.py            # CachePolicy ABC + BlockStatus
    ├── lru.py             # LRUCachePolicy
    └── arc.py             # ARCCachePolicy
```

---

## 是什么

### 1. `CPUOffloadingSpec`（`spec.py`）

继承 [base.md](base.md) 的 `OffloadingSpec`。负责把 `VllmConfig + KVCacheConfig` 翻译成 CPU 容量参数与 manager/worker 工厂。

容量推导（`spec.py:54`）：

- 读 `extra_config["cpu_bytes_to_use"]`（**必填**，否则 raise）。
- `BLOCK_SIZE_ALIGNMENT = 1`（CPU tier 默认无对齐要求；TieringOffloadingSpec 会覆盖为 `mmap.PAGESIZE` 以匹配 `SharedOffloadRegion`）。
- 若 `kv_cache_config.num_blocks > 0`：根据 GPU KV tensor 总字节 / num_blocks * world_size 算 `kv_bytes_per_block`，乘 `block_size_factor` 得 `kv_bytes_per_offloaded_block`；`cpu_page_size_per_worker = kv_bytes_per_offloaded_block // world_size`；最终 `num_blocks = cpu_bytes_to_use // aligned_kv_bytes_per_offloaded_block`。
- `eviction_policy` 默认 `"lru"`，可选 `"arc"`。
- `BLOCK_SIZE_ALIGNMENT` 在 TieringSpec 子类中被覆盖为 `SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT`（即 `mmap.PAGESIZE`），因为 SharedOffloadRegion 要求块大小页对齐。

构造 manager（`spec.py:108`）：实例化 `CPUOffloadingManager`，传 `num_blocks / cache_policy / enable_events / store_threshold / max_tracker_size`。`store_threshold` 默认 0（无过滤）；`>=2` 时启用 `CPUOffloadingManager.counts` 跟踪表过滤一次性块。

构造 worker（`spec.py:128`）：`CPUOffloadingWorker(kv_caches, block_size_factor, num_cpu_blocks)`；并校验 `current_platform.is_cuda_alike() or is_xpu()`，CPU tier 当前只支持 CUDA-like 与 XPU。

### 2. `CPUOffloadingManager`（`manager.py`）— scheduler 侧

实现 [base.md](base.md) 的 `OffloadingManager` 接口。内部把"块组织（数据结构）"和"驱逐决策"委托给 `CachePolicy`，自己只管 ref-count / 池管理 / event 生成 / store 骨架。

关键状态：

- `_free_list: list[int]`：可复用的物理 CPU 块槽 ID。
- `_num_allocated_blocks`：已分配峰值（用于 fresh 分配）。
- `_policy: CachePolicy`：`lru` 或 `arc`。
- `_num_evictable_cache_blocks`：当前 ref_cnt=0 的块数。
- `counts: OrderedDict[OffloadKey, int] | None`：仅当 `store_threshold >= 2` 时启用，统计块在 `lookup()` 中出现次数。
- `store_threshold / max_tracker_size / stores_skipped_in_current_batch`：过滤一次性块的 LRU 表大小上限与跳过计数。

关键方法逻辑：

- **`lookup`**（`manager.py:116`）：维护 counts 表；查 policy.get → 没找到 `MISS`；`not is_ready`（ref_cnt=-1，store 还没完成）`HIT_PENDING`；否则 `HIT`。
- **`prepare_load`**（`:132`）：从 policy 取出每个块、断言 `is_ready`、ref_cnt 0→1 时减 evictable 计数并 `mark_non_evictable`；返回 `CPULoadStoreSpec(block_ids)`。
- **`prepare_store`**（`:169`）：先 `store_threshold` 过滤；再剔除已在 cache 的 key；计算需要驱逐的块数；若驱逐空间不足 `return None`；调 `policy.evict(n, protected=set(keys))`（protected 排除本次要存的 key）—— evict 返回 None 也 `return None`；否则 `_allocate_blocks` 后 `policy.insert`。
- **`complete_store`**（`:240`）：成功时把每个未就绪块置 `ref_cnt=0`、`mark_evictable`、记 `OffloadingEvent(removed=False)`；失败时 `policy.remove + _free_block`。
- **`reset_cache`**（`:273`）：无条件清空——scheduler 侧的 `_stale_job_threshold` 保证此时不会再有 `complete_*` 调用进来。
- **`get_stats`**：返回 `CPU_CACHE_USAGE_PERC = used/num_blocks`；若 `store_threshold>=2` 也返回 `STORES_SKIPPED` 计数。

### 3. `CPUOffloadingWorker`（`gpu_worker.py`）— worker 侧

实现 `OffloadingWorker`。组合两个 `SingleDirectionOffloadingHandler`：一个管 GPU→CPU（store），一个管 CPU→GPU（load）。

#### CPU tensors 分配

- 若 `mmap_region is None`：`torch.zeros(num_cpu_blocks, cpu_page_size, dtype=int8, pin_memory=PIN_MEMORY)` 直接分配 pinned tensor。
- 否则 `mmap_region.create_next_view(cpu_page_size_bytes)` 从共享 mmap 切视图（见下文 `SharedOffloadRegion`）。

#### `SingleDirectionOffloadingHandler`

每个方向独占一份 handler，关键设计：

1. **拷贝函数选择**（`_select_swap_blocks_fn`，`gpu_worker.py:35`）：
   - GPU→CPU：总是用 `ops.swap_blocks_batch`（C++ DMA 拷贝引擎），带宽优先。
   - CPU→GPU：先用 Triton `swap_blocks_batch`，仅当 page_size < `THRESHOLD_BYTES` (28KiB) 且 8 字节对齐且 `HAS_TRITON` 且非 XPU；否则回落到 C++ `swap_blocks_batch`。XPU 不支持 Triton 直接 deref CPU 指针，因为 XPU 没有 CUDA 的统一虚拟地址空间。
2. **CUDA stream 串行化**：每个 transfer 独立 stream；新 transfer 等待 `last_transfer.end_event`，保证按 job_id 顺序执行。
3. **GPU→CPU 等 compute stream**：`stream.wait_stream(current_platform.current_stream())`——必须等 model 计算完才能拷出。
4. **`CU_MEMCPY_SRC_ACCESS_ORDER_ANY`**：CPU→GPU 读 pinned 内存不被 GPU stream 写，所以可以"任意源访问顺序"让驱动 pipeline 读；GPU→CPU 读 live GPU KV cache 被 compute stream 写，必须保留 STREAM 顺序。这点通过 `is_src_access_order_any=not gpu_to_cpu` 传给 `swap_blocks_batch`。
5. **批量描述符池**：`_buffer_pool / _stream_pool / _event_pool` 三个列表复用描述符/stream/event，避免每次 transfer 都新建。
6. **block_size_factor > 1 处理**：`compute_sub_block_ptrs`（`gpu_worker.py:73`）把 GPU block IDs 展开成字节指针数组，支持首尾块"跳过几个 sub-block"的对齐场景——这是 `OffloadingSpec.block_size_factor` 特性的 worker 侧落地。

#### 完成检测

`get_finished()` 用 `torch.Event.query()` 非阻塞轮询 `self._transfers` 队首；只要队首完成就 `popleft`，回收 stream/event/buffer 到 pool。

### 4. `SharedOffloadRegion`（`shared_offload_region.py`）

多 worker 共享的 `/dev/shm/vllm_offload_{instance_id}.mmap` 区域。每个 TP/DP worker 切一块自己的视图（worker0_block0 | worker1_block0 | ... | worker0_block1 | ...）。

- **首创建者**：`os.open(O_CREAT | O_EXCL)` 抢锁，赢的 worker `ftruncate` 设大小，其它 worker 等文件达到预期大小（`_wait_for_file_size`）。
- **`MADV_POPULATE_WRITE`**（Linux 5.14+）：按 rank 仅 populate 自己的 sub-pages，避免 bzero 全图。每个 block 行 `madvise` 一段，相比整区 populate 减少内存压力。
- **`cudaHostRegister`**：`pin_mmap_region()`（`gpu_worker.py:123`）在 CUDA-like 平台把整块 mmap 注册为 pinned memory（注意不是 `torch.zeros(..., pin_memory=True)`；后者会复制一份）。注册失败仅 warning，DMA 仍可工作但慢。
- **`create_next_view(tensor_page_size)`**：用 `torch.as_strided` 切一个 `(num_blocks, tensor_page_size)` 视图，`stride=(row_stride, 1)`——用 int8 持有保证 stride==bytes，使 `swap_blocks` 指针算术无需 dtype 转换。每个 worker 把自己的 canonical tensor 顺序串联在自己 area 内。
- **`create_kv_memoryview()`**：返回 `(num_blocks, row_stride_bytes)` 的零拷贝 memoryview，专供 secondary tier 用 `view[b]` 寻址——这是 cascade/promotion 到 FS/Obj/P2P tier 时的"远程侧 byte layout"。
- **`cleanup`**：`cudaHostUnregister` → 清空 views → 关闭 mmap_obj → 关 fd → 创建者 unlink 文件。

> scheduler 与 worker 各持一份 `SharedOffloadRegion`：scheduler 侧用 `rank=None`（看整图/memoryview）、worker 侧用 `rank=local_rank`（只 populate + view 自己的 area）。

### 5. `swap_blocks_triton.py`

Triton kernel `_swap_blocks_kernel`（`:24`）做"批 (src_ptr, dst_ptr, size) 三元组"拷贝。常量在 H100 PCIe Gen5 上经验调优：`NUM_SMS=12`、`THRESHOLD_BYTES=28KiB`、`MIN_N=16`——批次太小（<16）走 C++ `swap_blocks_batch`，page 太大（>=28KiB）也走 C++（DMA 在大块 bandwidth 优势）。

`swap_blocks_batch(src, dst, sizes, is_src_access_order_any=False, *, bytes_per_chunk)`：批次 < MIN_N 时调 C++；否则启动 `(min(NUM_SMS, n),)` grid，每 program 循环 over jobs，把每个 job 的 `words = size//8` 个 int64 用 `tl.load/tl.store` 拷过来。

### 6. `policies/`

#### `BlockStatus`（`policies/base.py:10`，ctypes.Structure）

```
_fields_ = [("ref_cnt", c_int32), ("block_id", c_int64)]
# ref_cnt = -1 表示 store 尚未完成（"未就绪"）
# ref_cnt = 0  表示可驱逐
# ref_cnt > 0  表示有 in-flight load 在读，不可驱逐
```

`is_ready = ref_cnt >= 0`。

#### `CachePolicy`（ABC）

抽象方法：`__init__(cache_capacity)`、`get` / `insert` / `remove` / `touch` / `evict(n, protected)` / `clear`。`evict` 是原子的——选不出 n 个就返回 `None` 不改任何状态。`mark_evictable / mark_non_evictable` 默认空实现，LRU/ARC 重写。

#### `LRUCachePolicy`（`policies/lru.py`）

- `evictable_blocks: OrderedDict[key, None]`：只放 ref_cnt=0 的块，按 LRU 排序。
- `blocks: dict[key, BlockStatus]`：所有 resident 块。
- `evict(n, protected)`：按 OrderedDict 顺序找 n 个非 protected 且 ref_cnt=0 的；找不够返回 None。
- `mark_evictable/mark_non_evictable`：在 `evictable_blocks` 里加/删 key。

#### `ARCCachePolicy`（`policies/arc.py`）

经典 ARC（Adaptive Replacement Cache）：

- `T1`（recent, 一次命中）/ `T2`（frequent, 多次命中）/ `B1`（T1 ghost）/ `B2`（T2 ghost）四个 OrderedDict。
- `target_t1_size` 自适应：B1 hit 增大（更看重 recency）、B2 hit 减小（更看重 frequency）；delta 取 `max(1, len(other_ghost)/len(self_ghost))`。
- `touch`：T1→T2（promotion），B1/B2 hit 调 `target_t1_size`。
- `evict`：若 `len(T1) >= target_t1_size` 从 T1 选；否则从 T2 选。被驱逐的块进对应 ghost list，ghost list 长度限制在 `cache_capacity`。
- 优势：在"扫一遍长 prefix 后又来回访问若干热点"这种 mix workload 上抗扫描污染优于 LRU。

`_CACHE_POLICIES = {"lru": LRUCachePolicy, "arc": ARCCachePolicy}`（`manager.py:30`）。

---

## 为什么

- **pinned memory 是 DMA 前提**：CPU tier 必须用 `cudaHostRegister` 或 `pin_memory=True` 才能让 GPU DMA 直接寻址 host 内存，否则走 pageable 缓冲中转慢一个量级。`SharedOffloadRegion` 的"`/dev/shm` mmap + 后注册 pinned"路径让多 worker 共享同一物理页且仍 pinned。
- **batch DMA > 单块 DMA**：`swap_blocks_batch` 把 (src,dst,size) 三元组数组提交给 cuMemcpyBatchAsync，比循环 cuMemcpyAsync 减少启动开销；Triton kernel 在小页场景又能进一步减/host kernel dispatch 开销。
- **store_threshold 过滤**：reasoning、agent 等"扫一次就忘"的 workload 会产生海量只命中一次的块——直接 offload 把 CPU tier 灌爆。引入 `store_threshold>=2` 的 LRU counts 表，让块先在 mini-track 里出现两次才允许进 CPU tier。
- **LRU vs ARC**：LRU 实现简单、常数小；ARC 在扫描+热点混合场景下抗污染更好，但 ghost list 增加内存与维护成本。两者通过 `CachePolicy` 抽象可互换。
- **`block_size_factor`**：在 CPU 侧用更大块（如 64 tokens vs GPU 16 tokens）减少元数据/IO 次数；代价是首尾块需要 `compute_sub_block_ptrs` 处理对齐，trade-off 由 `OffloadingSpec.__init__` 的 `block_size` 配置项驱动。
- **`ref_cnt` 兼做"在途"状态**：`-1` 表示 store 写中、`0` 表示就绪可逐、`>0` 表示有 load 在读——一个数据结构兼三职，简化 policy 实现。

---

## 怎么做

### 配置示例

```json
{
  "spec_name": "CPUOffloadingSpec",
  "cpu_bytes_to_use": 8589934592,
  "eviction_policy": "arc",
  "store_threshold": 2,
  "max_tracker_size": 100000,
  "offload_prompt_only": true,
  "block_size": 64
}
```

### 添加新 eviction policy

1. 在 `cpu/policies/` 下加 `mypolicy.py`，实现 `CachePolicy` 全部抽象方法。
2. 在 `cpu/manager.py:30` 的 `_CACHE_POLICIES` 字典注册 `"mypolicy": MyPolicy`。
3. `extra_config["eviction_policy"] = "mypolicy"`。

### 调优要点

- `cpu_bytes_to_use` 过大会挤占 model weights 内存，过小命中率低。建议从 GPU 显存的 0.5–1× 起步测。
- `store_threshold=2` 在 prefix cache 共享度高的工作负载上能显著减 CPU 污染，但单次 lookup 块永远进不了 offload——按需开关。
- ARC 在长 conversation + 多用户共享 system prompt 场景通常优于 LRU。
- `block_size` 设大于 GPU block 能显著降 IO 次数；首块未对齐的 trade-off 在 manager.py `prepare_load` 的 `block_indices` 里有处理。

---

## 与其它模块/系统配合

- [base.md](base.md)：实现的抽象接口来源。
- [tiering.md](tiering.md)：`CPUPrimaryTierOffloadingManager`（`tiering/manager.py:74`）继承 `CPUOffloadingManager`，把 `prepare_load/store` 别名为 `prepare_read/write` 以区分"secondary tier 访问 primary"的方向语义。
- [02-execution/worker/kv-connector-mixin](../02-execution/worker/kv-connector-mixin.md)：worker 进程在 `register_kv_caches` 阶段调 `OffloadingConnectorWorker.register_kv_caches`，进而构造 `CPUOffloadingWorker` 持有 GPU tensors 引用。
- [10-config/kv-transfer-config](../10-config/kv-transfer-config.md)：`kv_connector_extra_config` 字段。
- 指标出口：`OffloadingConnectorStats` → [07-distributed/kv-transfer/offloading](../07-distributed/kv-transfer/offloading.md) 上报 Prometheus。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.9 | CPU tier 首发：`CPUOffloadingSpec` + `CPUOffloadingManager`（仅 LRU）+ `CPUOffloadingWorker`；pinned tensor 直分配（`torch.zeros(pin_memory=True)`），无 mmap |
| v0.9.x | 加入 `store_threshold` / `max_tracker_size` 的 counts LRU 跟踪表；`OffloadingEvent` 用于事件出口；XPU 平台支持（`swap_blocks_batch` fallback 路径） |
| v0.10 | `block_size_factor` 落地：`compute_sub_block_ptrs` 处理 offloaded 块 > GPU 块的首尾对齐；引入 `CanonicalKVCaches`，worker 不再依赖原始 attention 张量布局 |
| v0.10.x | `CachePolicy` 抽象从 manager 内拆出 `policies/` 子包；引入 `ARCCachePolicy` |
| v0.11 | `SharedOffloadRegion` 落地：`/dev/shm` mmap + `MADV_POPULATE_WRITE` + `cudaHostRegister`，替代 per-worker pinned tensor；scheduler 侧也持一份 mmap 以提供 `memoryview` 给 secondary tier |
| main | `swap_blocks_triton.py` 加入：CPU→GPU 小块（<28KiB、8 字节对齐）场景用 Triton kernel 击败 cuMemcpyBatchAsync；`CU_MEMCPY_SRC_ACCESS_ORDER_ANY` 优化 CPU 源读取；`is_src_access_order_any` 参数贯通 C++/Triton 两路（待核实具体落地 commit） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [base.md](base.md)：`OffloadingManager/Worker/Spec` 抽象。
- [tiering.md](tiering.md)：`CPUPrimaryTierOffloadingManager` 把本页 manager 作为 primary tier 用法。
- [simple-kv-offload.md](simple-kv-offload.md)：旧管线的 CPU 卸载，与本页对照。
