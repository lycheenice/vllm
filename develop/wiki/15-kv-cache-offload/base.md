# base.py — KV 卸载核心抽象

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > base

源码：`vllm/v1/kv_offload/base.py`（约 588 行）。这里是整个新管线的**契约层**：定义 block 的标识 `OffloadKey`、scheduler 侧 `OffloadingManager`、worker 侧 `OffloadingWorker`、spec 基类 `OffloadingSpec`，以及把多布局 KV cache 规范成"按 block 行索引的字节视图"的 `CanonicalKVCaches`。所有具体 tier（cpu / fs / obj / p2p）都实现这里的接口。

> 旧管线 `simple_kv_offload/` 不实现本文件抽象，而是另起一套 manager/worker 接口（见 [simple-kv-offload.md](simple-kv-offload.md)）。

---

## 是什么

### 关键类型清单

| 类型 | 行号 | 角色 |
|---|---|---|
| `OffloadKey = NewType("OffloadKey", bytes)` | `base.py:30` | block hash + group_idx 拼成的 bytes，用作 offloaded 块的全局唯一索引；用 bytes 避免 tuple GC 开销 |
| `make_offload_key(block_hash, group_idx)` | `base.py:35` | 把 `block_hash + group_idx.to_bytes(4, "big")` 拼成 key |
| `get_offload_block_hash / get_offload_group_idx` | `base.py:40` / `:45` | 反向拆解 |
| `ReqContext` | `base.py:50` | per-request 上下文：`req_id` + `kv_transfer_params`（P2P tier 用） |
| `LookupResult` | `base.py:56` | `MISS / HIT / HIT_PENDING / RETRY` 四态枚举 |
| `OffloadPolicy` | `base.py:65` | `BLOCK_LEVEL`（仅 offload 新算块）/`REQUEST_LEVEL`（含 prefix 命中块，用于需要完整 KV 上下文的 tier） |
| `ScheduleEndContext` | `base.py:79` | `on_schedule_end()` 收到的每步调度信息（new_req_ids / preempted_req_ids） |
| `LoadStoreSpec`（ABC） | `base.py:88` | worker 用来定位读写位置的元数据；`medium()` 静态方法返回介质名 |
| `OffloadingEvent` | `base.py:111` | 给 KV events 子系统用的 (keys, medium, removed) 三元组 |
| `OffloadingManager`（ABC） | `base.py:177` | scheduler 侧接口；详见下文 |
| `OffloadingWorker`（ABC） | `base.py:459` | worker 侧接口；`submit_store/submit_load/get_finished/wait` |
| `OffloadingSpec`（ABC） | `base.py:486` | spec 基类，由 `VllmConfig + KVCacheConfig` 构造，能产出 manager 和 worker |
| `CanonicalKVCaches` | `base.py:432` | 跨 attention backend 的统一 KV cache 表示，供 worker 计算拷贝指针 |

### `OffloadingManager` 接口（scheduler 侧）

抽象方法（必须实现）：

- `lookup(key, req_context) -> LookupResult`：单块存在性查询，必返回四态之一。
- `prepare_load(keys, req_context) -> LoadStoreSpec`：把给定块标"读占用"（ref_cnt+1，防驱逐），返回位置 spec 交给 worker。
- `prepare_store(keys, req_context) -> PrepareStoreOutput | None`：分配/驱逐 CPU 槽；`None` 表示放不下（如驱逐仍不够）。返回 `(keys_to_store, store_spec, evicted_keys)`。
- `on_new_request(req_context) -> RequestOffloadingContext`：第一次见到该请求时调一次；tier 在此声明 `OffloadPolicy`（决定 `prepare_store` 是否含 prefix 命中块）。

默认实现（可 override）：

- `touch(keys, req_context)`：标记最近使用，影响 LRU/ARC 排序。
- `complete_load / complete_store`：与 `prepare_*` 配对释放 ref_cnt；`complete_store(success=False)` 时未完成的块被回滚。
- `on_request_finished`：请求结束钩子；不保证已持久化（异步 transfer 可能仍在飞）。
- `take_events / on_schedule_end / has_pending_work / reset_cache / get_stats / shutdown`：事件、每步收尾、是否继续空转 tick、整体清空（sleep/weight update 用）、指标、关闭。

### `OffloadingWorker` 接口（worker 侧）

方向**显式**——`submit_store(job_id, src=GPULoadStoreSpec, dst=LoadStoreSpec)` 把 GPU 块写到 offload 介质；`submit_load(src=offload, dst=GPU)` 反向。`get_finished()` 返回 `list[TransferResult]`（含 `transfer_size/transfer_time` 用于 metrics），`wait(job_ids)` 阻塞等待。

### `OffloadingSpec` 基类构造逻辑

`OffloadingSpec.__init__`（`base.py:496`）做几件所有 tier 共享的事：

1. 从 `kv_transfer_config.kv_connector_extra_config` 取 `extra_config` 字典。
2. 解析 `kv_events_config` → `OffloadingKVEventsConfig{enable_kv_cache_events, self_describing_kv_events}`；后者控制 connector 是否在 `BlockStored` 事件里附带自描述侧表。
3. `offload_prompt_only`（默认 True）：只 offload prefill 块，跳过 decode 块——对 reasoning 模型等"丢掉思考 token"的场景有用。
4. `gpu_block_size`（按 KV cache group 计算，乘以 context parallel factor）、`hash_block_size`（必须整除 `gpu_block_size`，否则 raise）。
5. `block_size_factor`：若 extra_config 显式给了 `block_size`，则 offloaded 块更大、`block_size_factor = offloaded_block_size // gpu_block_size`。这允许在 offload 侧用更稀疏的块（例如 GPU 块 16，offloaded 块 64），减少元数据量、提升 IO 吞吐，代价是 load 时首尾块需对齐处理。

两个抽象方法：`get_manager() -> OffloadingManager`（scheduler 侧单例）与 `get_worker(kv_caches) -> OffloadingWorker`（worker 侧单例）。

### `CanonicalKVCaches`

不同 attention backend 的 KV cache 张量物理布局差异巨大（FlashAttention 是 `(2, num_blocks, num_heads, head_size, ...)`、MLA 是 latent 形 etc.）。`CanonicalKVCaches` 把它们统一成"`tensors: list[CanonicalKVCacheTensor]`，每张 shape `(num_blocks, page_size_bytes)` 且 dtype=int8"的规范形式，并配 `group_data_refs` 描述每个 KV cache group 在哪张 tensor 的哪段 page。`CanonicalKVCacheRef.page_size_bytes` 是**未填充**的真实 page，与 `CanonicalKVCacheTensor.page_size_bytes`（可能 padded）配合供 `compute_sub_block_ptrs` 计算精确字节指针。

---

## 为什么

- **解耦 scheduler 与 worker**：scheduler 只看 `OffloadKey` 与 `LookupResult`，不关心介质；worker 通过 `LoadStoreSpec` 拿到字节级位置。两端通过 spec 序列化传递，无共享内存。
- **支持异构 tier**：`OffloadingSpec` 把"如何从 VllmConfig 派生 manager/worker"封装成可注册的工厂，使 CPU/FS/Obj/P2P/自定义 spec 都按同套流程接入。
- **block_size_factor**：让 offload 层用比 GPU 更大的块以减少 IO 次数——代价是 worker 的 `compute_sub_block_ptrs` 需处理首尾块对齐（见 `cpu/gpu_worker.py:73`）。
- **`CanonicalKVCaches`**：消除 attention backend 差异，使 worker 的拷贝路径只需面对"扁平字节块数组"。
- **`LookupResult` 四态**：`HIT_PENDING` 区分"块虽在表里但还在写"vs"可读"；`RETRY` 让多层 tiering 的异步 promotion 不阻塞 scheduler——下一步再来查即可。

---

## 怎么做

### 添加一个新 spec 的最小骨架

```python
class MySpec(OffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)
        # 从 self.extra_config 读自定义参数
        ...
    def get_manager(self) -> OffloadingManager: ...
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker: ...
```

注册两种方式：

1. 修改 `vllm/v1/kv_offload/factory.py:65` 末尾的 `register_spec("MySpec", "module.path", "MySpec")`。
2. 通过 `extra_config["spec_module_path"]` + `extra_config["spec_name"]` 让 `OffloadingSpecFactory.get_spec_cls` 动态 import。

### 自定义 tier 的 lookup 行为

`lookup` 是性能热点（每 step 每块都查）。注意：

- `CPUOffloadingManager.lookup` 会顺手维护一个 `counts: OrderedDict` 用作 `store_threshold` 过滤——LRU 表满时 O(1) 淘汰最老 entry（`manager.py:122`）。
- 多层 tiering 不要在 `lookup` 里阻塞 I/O；用 `RETRY` 表达"异步未就绪"并通过 `on_schedule_end` + `has_pending_work` 让 engine 继续 tick。

### 指标

`OffloadingSpec.build_metric_definitions` 返回 `dict[str, OffloadingMetricMetadata]`，由 `OffloadingConnectorScheduler` 转成 Prometheus metric。`OffloadingManager.get_stats()` 返回 `OffloadingConnectorStats`，由 scheduler 聚合后上报。三种 metadata：counter / gauge / histogram（带 buckets）。

---

## 与其它模块/系统配合

- 调用方：[07-distributed/kv-transfer/offloading](../07-distributed/kv-transfer/offloading.md) 的 scheduler/worker mixin。
- 注册表：[factory.md](factory.md)。
- CPU tier 实现：[cpu.md](cpu.md)；tiering orchestrator：[tiering.md](tiering.md)。
- `CanonicalKVCaches` 由 `OffloadingConnectorWorker.get_kv_caches(...)` 经 `vllm/v1/kv_cache_interface.py` 的 `get_kv_cache_config` 路径衍生（待核实具体调用栈）。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.9 | 首次引入 `base.py`：`OffloadKey`、`OffloadingManager` 仅含 `lookup/prepare_load/prepare_store/complete_*` + `on_new_request`；`OffloadingSpec` 仅 `get_manager/get_worker` |
| v0.9.x | 增加 `touch / on_request_finished / take_events / on_schedule_end / has_pending_work / reset_cache / get_stats / shutdown` 默认实现，支撑异步 tier 与 sleep mode |
| v0.10 | 引入 `CanonicalKVCaches`/`CanonicalKVCacheTensor`/`CanonicalKVCacheRef`，替代原本直接传 `dict[str, Tensor]` 的 worker 接口，使多 KV group / hybrid 模型可统一寻址 |
| v0.10 | 加入 `OffloadPolicy` + `RequestOffloadingContext`，让 tier 声明 `REQUEST_LEVEL` 以触发"已存在 primary 的块也 cascade"路径 |
| v0.11 | `OffloadingWorker` 改为显式方向 API（`submit_store`/`submit_load`），不再用 `(src_medium, dst_medium)` 路由——简化 P2P tier 双向通信 |
| main | `OffloadingKVEventsConfig` 引入 `self_describing_kv_events`；`OffloadingMetricMetadata` 系列让 tier 自描述 Prometheus 指标（取代之前在 connector 侧硬编码） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [factory.md](factory.md)：spec 注册与解析。
- [cpu.md](cpu.md)：`CPUOffloadingSpec/Manager/Worker` 参考实现。
- [tiering.md](tiering.md)：`SecondaryTierManager` 与多层编排。
