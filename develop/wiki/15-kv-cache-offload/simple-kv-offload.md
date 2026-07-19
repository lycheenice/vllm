# simple_kv_offload/ — 旧管线 SimpleCPUOffloadConnector

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > simple-kv-offload

源码目录：`vllm/v1/simple_kv_offload/`（5 文件 + `__init__.py`）。KV 卸载的**第二条管线**：仅 CPU 单层，独占实现冲 manager/worker/metadata，不经 [base.md](base.md) 抽象。由 `vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py` 的 `SimpleCPUOffloadConnector(KVConnectorBase_V1, SupportsHMA)` 接入引擎，需 `enable_prefix_caching=True` 才启用。

```
simple_kv_offload/
├── __init__.py
├── manager.py        # SimpleCPUOffloadScheduler（scheduler 侧）
├── worker.py         # SimpleCPUOffloadWorker（worker 侧）
├── copy_backend.py   # DmaCopyBackend（cuMemcpyBatchAsync + 后台线程）
├── cuda_mem_ops.py   # pin_tensor + BatchMemcpyParams + copy_blocks + CU_MEMCPY_SRC_ACCESS_ORDER_*
└── metadata.py       # SimpleCPUOffloadMetadata / SimpleCPUOffloadWorkerMetadata
```

---

## 是什么

### 1. `SimpleCPUOffloadConnector`（外层胶水）

`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:45`。在 `__init__` 里按 `KVConnectorRole` 分流：

- scheduler 角色：创建 `SimpleCPUOffloadScheduler`，把 GPU KVCacheConfig 反推成 CPU `KVCacheConfig`、`scheduler_block_size / hash_block_size` 传进来。
- worker 角色：创建 `SimpleCPUOffloadWorker`。

类支持 `SupportsHMA`（hybrid multi-step attention），配置入口是 `kv_transfer_config.kv_connector_extra_config`：

| 键 | 含义 |
|---|---|
| `cpu_bytes_to_use` | server-wide 字节数（除以 world_size 得 per-rank） |
| `cpu_bytes_to_use_per_rank` | 显式 per-rank 字节；省略时按 world_size 均分 |
| `lazy_offload` | `"eager"`（默认，scheduler immediate 推 store）/`"lazy"`（按需在 GPU free 队列游标处 store） |

### 2. `SimpleCPUOffloadMetadata` / `SimpleCPUOffloadWorkerMetadata`（`metadata.py`）

- `SimpleCPUOffloadMetadata`（scheduler→worker）：每步一道 `load_event`（INVALID_JOB_ID 表无 load）+ `load_gpu_blocks`/`load_cpu_blocks`；一道 `store_event` + `store_gpu_blocks`/`store_cpu_blocks`；`need_flush` 标记本步有 preemption。
- `SimpleCPUOffloadWorkerMetadata`（worker→scheduler）：`completed_store_events: dict[event_idx, int]`，每个 worker 自报"我完成的 store event"。`aggregate()` 在多 worker 间累加；scheduler manager 等 count 达到 `world_size` 才认为该 store event 完成。

worker 不知道 req_id——所有 job↔req_id 解析由 scheduler 端 inverse map 完成。这让 metadata 协议紧凑。

### 3. `SimpleCPUOffloadScheduler`（`manager.py`，scheduler 侧）

**关键设计：在 scheduler 内复刻一套 CPU `KVCacheCoordinator` + `BlockPool`**（`manager.py:122`），用 `KVCacheBlock` 抽象管理 CPU 块的引用计数与 LRU。这与新管线 `CPUOffloadingManager` 用 `CachePolicy + BlockStatus` 的方式完全不同。

构造（`manager.py:70`）：

1. `_derive_cpu_config(gpu_config, cpu_capacity_bytes)`（`:182`）：按 CPU/GPU 内存比例算 `num_cpu_blocks`，复制 GPU `kv_cache_groups`/`kv_cache_tensors` 结构。
2. `get_kv_cache_coordinator(kv_cache_config=self.cpu_kv_cache_config, max_model_len=..., enable_caching=True, dcp_world_size=..., pcp_world_size=..., scheduler_block_size=self.block_size, hash_block_size=self.hash_block_size)` 复用 [引擎核心-KV管理](../01-engine-core/kv-cache-management/README.md) 的 coordinator。
3. 找到 full-attention group 的 `fa_gidx`、`fa_block_size`（注意 hybrid 模型 FA 块大小可能 != scheduler 块大小，等于 LCM）。
4. 加载反查表 `_reqs_to_load`/`_load_event_to_reqs` 与 store 反查表 `_store_event_to_blocks`/`_store_event_to_reqs`，event_idx 单调递增。
5. `_expected_worker_count = world_size`：store event 必须等所有 worker 报完成才推进。
6. `_lazy_mode` 与 `_target_free`：lazy 模式按"目标保留空闲块数"驱动 store；eager 模式 scheduler 收到新 prefill 立即推 store。

关键操作：

- **store 触发**：在 `start_kv_offload`/`update_state_after_alloc` 钩子中：
  - eager 模式：扫 GPU 输出 block_hashes 与 GPU free 队列，把"刚完成 prefill"的块 → `StoreRequestState.block_ids` 累积；一批 ≥ 1 时给 `store_event` 编号，把 (`gpu_blocks, cpu_blocks`) 列表写入 `SimpleCPUOffloadMetadata`。
  - lazy 模式：用 `_cursor` 在 GPU free 队列游标扫描"应继续 offload 的块"，按"目标保留空闲块数" `_target_free` 触发。
- **load 触发**：preempted 请求重新调度或新请求 prefix 命中 CPU 时，`find_longest_cache_hit` 在 CPU coordinator 找匹配块、`prepare_blocks` 提前分配 CPU 槽 + 增 ref_cnt；之后 `load_event` 编号并写 metadata。
- **`_process_store_event`**（管理 worker reported 完成的 store event）：当某 `event_idx` 计数达 `world_size` 时把对应 GPU blocks 标"已 offloaded"，更新 CPU coordinator 的 block_hashes 表，触发 `KVCacheEvent` 让外部观测者（[07-distributed/kv-events](../07-distributed/kv-events.md)）知道有新 KV 进 cache。
- **`_process_load_event`**：load 完成后让 GPU `BlockPool` 知道这些块的 hash 已就绪可被复用。

### 4. `SimpleCPUOffloadWorker`（`worker.py`，worker 侧）

构造：

- `load_stream` / `store_stream`：两条独立 CUDA stream，分别做 CPU→GPU 与 GPU→CPU。
- `DmaCopyBackend()`：拷贝后台线程。
- `_load_events` / `_store_events`：list of `(event_idx, torch.Event)`，按 event_idx 排序。每条 store launch 前需 `stream.wait_event(_store_compute_done)` 等 compute stream 完成。
- `_load_hwm` / `_store_hwm`：每 stream 已完成最高水位。

`register_kv_caches(kv_caches: dict[str, Tensor])`：对每层 GPU kv_cache 造一个 `torch.zeros(..., pin_memory=True)` CPU 副本（或用 `cudaHostRegister` 注册的 slicing）；准备 `DmaCopyBackend.init(gpu_caches, cpu_caches, device, load_stream, store_stream)`。

`bind_connector_metadata(metadata) -> get_finished()` 流程：

1. 从 metadata 取 `load_event/load_gpu_blocks/load_cpu_blocks`：若有 load job，`DmaCopyBackend.launch_copy(src=cpu_blocks, dst=gpu_blocks, is_store=False, event_idx, events_list)`。
2. 同上 store：若有 store job，先 record `_store_compute_done`（等 compute stream 完），然后 `launch_copy(src=gpu_blocks, dst=cpu_blocks, is_store=True, event_idx, events_list, wait_event=_store_compute_done)`。
3. `get_finished()`：query 每条 event 看是否完成；store 事件按 `event_idx → 1` 投入 `SimpleCPUOffloadWorkerMetadata.completed_store_events`；load 事件回写 `_load_hwm` 与 `-1` 标记的 list。

### 5. `DmaCopyBackend`（`copy_backend.py`）

单生产者-单消费者 `queue.SimpleQueue` + 单后台线程：

- `init` 在 worker `register_kv_caches` 时调一次，调 `build_params(gpu_caches, cpu_caches, store_stream, src_access_order=STREAM)` / `build_params(cpu_caches, gpu_caches, load_stream, src_access_order=ANY)` 为两方向各预算 `BatchMemcpyParams`。
- `launch_copy` 把 `(src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event)` 入队即返回——非阻塞。
- `_copy_loop`：daemon 线程循环，set device、`get` 入队项、`stream.wait_event(wait_event)`（store 方向等 compute-done）、`copy_blocks(...)` 提交批量 DMA、`torch.Event().record(stream)` 并 append 到 `events_list`。
- `shutdown`：put `None` sentinel 让线程退出，`join(timeout=5.0)`。

### 6. `cuda_mem_ops.py`

`cuMemcpyBatchAsync` 的 ctypes 直绑：

- `pin_tensor(tensor)`：`cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)`，绕开 PyTorch `pin_memory=True` 会把内存 round-up 到下一个 2 的幂（注释见 `cuda_mem_ops.py:24`）。
- `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM=1` / `CU_MEMCPY_SRC_ACCESS_ORDER_ANY=3` 常量。
- `_CUmemLocation` / `_CUmemcpyAttributes` ctypes Structure。
- `BatchMemcpyParams`：预构建每层 cache 的指针表 + descriptor buffers。`build_params(src_caches, dst_caches, stream, src_access_order)` 返回 opaque struct。
- `copy_blocks(src_blocks, dst_blocks, params)`：把 GPU/CPU block IDs 映射到指针、填 descriptor、调 `cuMemcpyBatchAsync`。

`STREAM` vs `ANY` 语义参见 [cpu.md](cpu.md) 中 `SingleDirectionOffloadingHandler` 的同样处理——store 读 live GPU cache 须 STREAM、load 读稳定 pinned host 可 ANY。

---

## 为什么

- **直接复用 engine KV 管理抽象**：旧管线最大价值在于把 CPU 当作"另一个 device"——直接用 engine 的 `KVCacheCoordinator` / `BlockPool` / `KVCacheBlock` 抽象管 CPU 块、让 prefix cache / hash 匹配代码无需改。代价是依赖 [01-engine-core/kv-cache-management](../01-engine-core/kv-cache-management/README.md) 的内部细节。
- **HMA 支持**：Hybrid Multi-step Attention 让 prefill 与 decode 在同一 step 内交替跑——`SimpleCPUOffloadConnector` 把 KV 移走/拉回的时机与 HMA flow 紧耦合；新管线 [`cpu/`](cpu.md) 不内置 HMA。
- **lazy vs eager**：reasoning 等长 prefix workload 在 eager 模式下可能 store 跟不上；lazy 用 GPU free 队列游标按需 store，限速到不阻塞 GPU 计算的程度。
- **world_size 同步**：TP/PP 多 worker 同一 store event 必须所有 worker 都完成才认为 CPU 数据 ready（否则其他 worker 可能读到旧字节）。用 worker metadata 的 `completed_store_events` aggregate 实现。
- **后台线程而非 stream pool**：worker 用单独 daemon 线程 push 任务到 `DmaCopyBackend._thread`，API 侧 `launch_copy` 永远非阻塞；与新管线 `SingleDirectionOffloadingHandler` 的 `stream_pool/event_pool/buffer_pool` 路径不同——后者更省内存但要求 worker 在主线程直接 record 事件。
- **绕过 pin_memory round-up**：`pin_tensor` 直调 `cudaHostRegister`，避免 PyTorch allocator 把 100GB 圆整到 128GB 的浪费。

---

## 怎么做

### 配置示例

最小化启用：

```python
VllmConfig(
    cache_config=CacheConfig(enable_prefix_caching=True),
    kv_transfer_config=KVTransferConfig(
        kv_connector=SimpleCPUOffloadConnector,
        kv_connector_extra_config={
            "cpu_bytes_to_use": 8589934592,  # 8 GB / world_size
            "lazy_offload": False,           # eager
        },
    ),
)
```

per-rank 显式：

```json
{"cpu_bytes_to_use_per_rank": 4294967296, "lazy_offload": true}
```

### 新旧管线如何选

| 场景 | 选 |
|---|---|
| 仅需 CPU offload、要 HMA、要 eager/lazy 灵活调度 | `simple_kv_offload/` |
| 需要 tiering（FS/Obj/P2P） | 新管线 (`TieringOffloadingSpec`)，因为旧管线无 secondary tier 概念 |
| 需要 ARC、`store_threshold`、`block_size_factor` | 新管线 `CPUOffloadingSpec` |
| 需要 `OffloadingEvent`/`take_events` 与外部 KV events 消费 | 两者都支持出口事件但接口不同 |

两条管线**不能同时启用**——一个实例只能配一个 `kv_connector`。

### 调优

- `lazy_offload=true` 在 reasoning 模型长 prefill 场景稳定，但 KV 出现在 CPU cache 的时间稍晚；命中前缀延迟时间换稳定性。
- `_target_free` 由 `_estimate_lazy_target_blocks` 自动估，按 `max_num_batched_tokens / fa_block_size * cp_world_size` 推导（待核实具体计算位置 `manager.py` 内）。
- store event 同步等所有 worker 完成——TP=8 时任一 worker 慢就拖累所有 worker 的 store event 进度。`pre-commit run` 不能测，需真实多 GPU 调。

---

## 与其它模块/系统配合

- 接入引擎：`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`（属 [07-distributed/kv-transfer](../07-distributed/kv-transfer/README.md)）。
- 复用：[01-engine-core/kv-cache-management](../01-engine-core/kv-cache-management/README.md) 的 `BlockPool`/`KVCacheCoordinator`/`KVCacheBlock`。
- 事件出口：[07-distributed/kv-events](../07-distributed/kv-events.md) 的 `KVCacheEvent`（`BlockStored`/`BlockRemoved` 等）。
- 配置：[10-config/kv-transfer-config](../10-config/kv-transfer-config.md) 的 `kv_connector_extra_config`。
- worker mixin：[02-execution/worker/kv-connector-mixin](../02-execution/worker/kv-connector-mixin.md)。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.6（待核实） | `simple_kv_offload/` 前身 `v1/offloading/` 随 v1 vLLM 引入；最初只有 eager 模式与单 worker 假设 |
| v0.7–v0.8 | 加入 `lazy_offload` 与 `_target_free` 估算；HMA 支持；TP/PP 多 worker 的 `world_size` 同步；`SupportsHMA` 标记 |
| v0.9 | 新管线 `kv_offload/` 出现后，`simple_kv_offload/` 进入稳定维护期；不再添加新特性，新特性统一进新管线 |
| v0.10 | `DmaCopyBackend` 加 `wait_event` 参数让 store 等 compute stream；`CU_MEMCPY_SRC_ACCESS_ORDER_*` 引入 |
| main | `pin_tensor` 绕开 PyTorch host allocator round-up；`cpu_bytes_to_use_per_rank` 加入让集群部署更灵活（v0.10 后待核实具体 commit） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [cpu.md](cpu.md)：新管线单层 CPU 实现的对照。
- [base.md](base.md)：新管线的抽象，本管线**不**实现。
- [01-engine-core/kv-cache-management](../01-engine-core/kv-cache-management/README.md)：被本管线复用的 `BlockPool`/`KVCacheCoordinator`。
- [07-distributed/kv-transfer](../07-distributed/kv-transfer/README.md)：connector 注册与角色分流。
