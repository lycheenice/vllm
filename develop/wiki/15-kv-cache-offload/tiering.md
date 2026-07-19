# tiering/ — 多层 KV 卸载编排

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > tiering

源码目录：`vllm/v1/kv_offload/tiering/`（base/spec/manager/factory/async_lookup + example/ 子包）。本目录定义"CPU primary tier + 任意多 secondary tier"的编排框架：spec 解析 extra_config 生成 tiering manager，manager 协调 cascade（store 时 primary→所有 secondary）与 promotion（load 时 secondary→primary→GPU）。

```
tiering/
├── __init__.py
├── base.py         # SecondaryTierManager ABC + JobMetadata/JobResult
├── spec.py         # TieringOffloadingSpec
├── manager.py      # TieringOffloadingManager + CPUPrimaryTierOffloadingManager
├── factory.py      # SecondaryTierFactory 注册表
├── async_lookup.py # AsyncLookupManager（per-tier 异步存在性查询）
├── example/        # ExampleSecondaryTierManager（参考实现 + 测试 tier）
│   └── manager.py
├── fs/             # 见 tiering-fs.md
├── obj/            # 见 tiering-obj.md
└── p2p/            # 见 tiering-p2p.md
```

---

## 是什么

### 1. `SecondaryTierManager`（`base.py`）

ABC，描述一个不直接访问 GPU 的二级层。所有方法**在 scheduler 进程**调用、必须轻量非阻塞：

| 方法 | 必实现 | 语义 |
|---|---|---|
| `lookup(key, req_context) -> LookupResult` | ✓ | `HIT/MISS/RETRY`（没有 `HIT_PENDING`，因 secondary 不暴露"写中"状态） |
| `submit_store(JobMetadata)` | ✓ | 异步：primary block_ids 读出来写到本 tier |
| `submit_load(JobMetadata)` | ✓ | 异步：本 tier 数据写回 primary block_ids |
| `get_finished_jobs() -> Iterable[JobResult]` | ✓ | 轮询完成的 job |
| `on_new_request(req_context) -> RequestOffloadingContext` | ✓ | tier 声明 `OffloadPolicy`（REQUEST_LEVEL 会触发"已 primary 块也 cascade"） |
| `drain_jobs()` | ✓ | 阻塞直到所有在飞 transfer 完成或失败（`reset_cache` 调用，保证下文释放 primary 槽不与 IO 抢内存） |

默认实现：`has_pending_work / touch / on_request_finished / on_schedule_end / shutdown / build_metric_definitions / get_stats`。

构造签名：`__init__(offloading_spec, primary_kv_view: memoryview, tier_type: str)`。`primary_kv_view` 是 [cpu.md](cpu.md) `SharedOffloadRegion.create_kv_memoryview()` 返回的零拷贝 view，shape `(num_blocks, row_stride_bytes)`，让 secondary tier 用 `view[b]` 找到任意 primary block 的字节。

`JobMetadata`（`base.py:33`）：`{job_id, keys, block_ids: np.ndarray, is_promotion: bool, req_context}`。`is_promotion=True` 表示 secondary→primary（load），`False` 表示 primary→secondary（store/cascade）。

`JobResult`：`{job_id, success}`。

### 2. `SecondaryTierFactory`（`factory.py`）

与 [factory.md](factory.md) 的 `OffloadingSpecFactory` 同构。模块末尾预注册四个 tier type（`factory.py:57`）：

```python
register_tier("example", "vllm.v1.kv_offload.tiering.example.manager", "ExampleSecondaryTierManager")
register_tier("fs",      "vllm.v1.kv_offload.tiering.fs.manager",      "FileSystemTierManager")
register_tier("p2p",     "vllm.v1.kv_offload.tiering.p2p.manager",     "P2PSecondaryTierManager")
register_tier("obj",     "vllm.v1.kv_offload.tiering.obj.manager",     "ObjectStoreSecondaryTierManager")
```

`create_secondary_tier(tier_config, primary_kv_view, spec)` 从配置 pop `type` 后调 `tier_cls(offloading_spec, primary_kv_view, tier_type, **config)`——剩余字段都原样转发到 tier 构造函数（每个 tier 自行声明支持的 kwargs）。

### 3. `TieringOffloadingSpec`（`spec.py`）

继承 [cpu.md](cpu.md) 的 `CPUOffloadingSpec`，复用其容量推导、`get_worker()`。仅 override：

- **`BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT`（即 `mmap.PAGESIZE`）**——CPU 容量推导必须按页对齐才允许多 worker mmap 切片。
- **`build_metric_definitions`**：合并父类（CPU）的定义与每个 secondary tier 的 `build_metric_definitions(tier_config)`。
- **禁用 `self_describing_kv_events`**（`spec.py:94`）：tiering 的 promotion 路径会产生"非 GPU store 引起"的 primary store 事件，self-describing 侧表无法正确描述，故禁用。
- **禁用 `store_threshold>=2`**（`spec.py:173`）：tiering 的多层语义与 store_threshold 互斥。
- **`get_manager()`**（`:112`）：
  1. 构造 scheduler-side `SharedOffloadRegion(rank=None)` 仅创建 memoryview 不 populate worker area。
  2. 构造 `CPUPrimaryTierOffloadingManager(mmap_region=scheduler_mmap)`。
  3. `primary_tier.get_kv_memoryview()` 取出 view，`SecondaryTierFactory.create_secondary_tier(...)` 逐个建 secondary tier。
  4. 实例化 `TieringOffloadingManager(primary_tier, secondary_tiers, enable_events)`。
- **`create_worker()`**（`:190`）：每个 worker 进程调一次，用 `torch.accelerator.current_device_index()` 作为 rank 构造自己的 `SharedOffloadRegion`，再 `CPUOffloadingWorker(mmap_region=worker_mmap, ...)`。

### 4. `CPUPrimaryTierOffloadingManager`（`manager.py:74`）

继承 `CPUOffloadingManager`，加两个能力：

1. **别名暴露 secondary-tier 视角的读写接口**（`manager.py:100`）：
   - `prepare_read = prepare_load` / `complete_read = complete_load`：secondary tier 的 `submit_store` 把 primary 当作"源"读出来。
   - `prepare_write = prepare_store` / `complete_write = complete_store`：secondary tier 的 `submit_load` 把 primary 当作"目标"写入。
   - 这样 `TieringOffloadingManager` 在 cascade 路径里调 `prepare_read` 而不是 `prepare_load`，避免"在 store 路径里调 load"的语义混淆。
2. **`get_kv_memoryview()`**：返回 `SharedOffloadRegion.create_kv_memoryview()` 的结果，传给每个 secondary tier。

### 5. `TieringOffloadingManager`（`manager.py:123`）

实现 [base.md](base.md) 的 `OffloadingManager`，是 orchestration 核心。内部状态：

- `primary_tier: CPUPrimaryTierOffloadingManager`
- `secondary_tiers: list[SecondaryTierManager]`
- `_transfer_jobs: dict[JobId, JobMetadata]`：所有在飞 transfer，区分方向用 `JobMetadata.is_promotion`。
- `_pending_load_submissions: dict[tier, dict[req_id, PendingPromotion]]`：promotion 请求在 lookup 期间延后收集，`on_schedule_end` 一次批量 `submit_load`。
- `_req_state: dict[req_id, RequestState]`：per-request 的 `pending_primary_stores` 与 `request_level_tiers`，用于延迟 on_request_finished 转发（直到 pending primary store 全部 complete）。
- `_processed_jobs_this_step`：本 step 是否已经 polling 过 finished jobs（避免重复）。

#### 关键路径

**`lookup`**（`manager.py:238`）：

1. `_maybe_process_finished_jobs()`（每步首次调用完成一次轮询）。
2. primary `lookup`：HIT 即返回；HIT_PENDING 也透传。
3. miss → 遍历 secondary_tiers：
   - `HIT` → `_initiate_promotion(tier, key, req_context)`：调 `primary_tier.prepare_write([key])` 在 CPU 预留槽（ref_cnt=-1，本步内 lookup 看到该 key 为 in-flight 防止重复 promotion），把 (`key`, `block_id`) push 到 `_pending_load_submissions[tier][req_id]`。返回 `RETRY`。
   - `RETRY` → 累加 any_retry。
   - 都 miss → `MISS`。

**`_flush_pending_promotions`**（`manager.py:331`，由 `on_schedule_end` 调）：对每个 (tier, req_ctx) 把累积的 keys + block_ids 一次性 `tier.submit_load(JobMetadata(is_promotion=True))`，记入 `_transfer_jobs`。

**`prepare_load`**（`:355`）：先 `_maybe_process_finished_jobs` 确保已完成 promotion 落 primary → 再 `primary_tier.prepare_load`。

**`prepare_store`**（`:408`）：

1. `_maybe_process_finished_jobs`。
2. `primary_tier.prepare_store` 留出 CPU 槽。
3. 若该请求的 tier 声明了 `REQUEST_LEVEL` policy，把"已在 primary 的块"也 cascade 到 request_level_tiers（`_cascade_existing_blocks_to_request_level_tiers`，`:461`）。

**`complete_store`**（`:498`）：primary `complete_store` 后，对**每个** secondary tier：

1. `primary_tier.prepare_read(keys)`：拿 block_ids 并 +1 ref_cnt 保护。
2. `tier.submit_store(JobMetadata(is_promotion=False))`：发起到 secondary 的异步 cascade。
3. 记入 `_transfer_jobs`；同时 `state.pending_primary_stores -= 1` 并 `_maybe_finalize_request`（pending 为 0 且 is_finished 时才转发 `on_request_finished` 给 secondary tier）。

**`_process_finished_jobs`**（`:202`）：

- 对每个 secondary tier 调 `get_finished_jobs()`。
- `is_promotion=True`（load 完成）→ `primary_tier.complete_write(keys, success)`：让 CPU 槽就绪。
- `is_promotion=False`（store cascade 完成）→ `primary_tier.complete_read(keys)`：释放 ref_cnt。

**`reset_cache`**（`:643`）：sleep / weight update 用。先 `tier.drain_jobs()` 等所有 secondary transfer 收尾 → `_process_finished_jobs` 消费完成事件 → 清 `_pending_load_submissions` → 保留 finished 请求状态让它们 finalize → `primary_tier.reset_cache()`。**secondary tier 故意不 reset**：FS/Obj/P2P 的持久数据要跨 reset 复用；只清掉已 finished 的请求 bookkeeping。

### 6. `AsyncLookupManager`（`async_lookup.py`）

per-tier 的非阻塞存在性查询框架。FS 与 Obj tier 用文件/对象存在性查询慢于内存查 dict，所以用后台线程批量查。

设计：

- `_lookup_state: dict[OffloadKey, LookupState]`（scheduler 拥有，无锁）：缓存 (key → result | None)。`None` 表示未决。
- `_lookup_batch: list[(key, req_context)]`（scheduler 拥有）：本 step 累积的待查 key。
- `_lookup_queue: SimpleQueue`（scheduler→worker）：每 step `flush()` 把整 batch 投递一次。
- `_pending_results: SimpleQueue`（worker→scheduler）：worker 完成后投递 `list[(key, found)]`。
- `_thread`：daemon 后台线程名 `vllm_offloading_lookup_{tier_type}`，循环 `get` batch、按 req_id 分组调 `batch_lookup()`、把结果 put 回去。

子类只需实现 `batch_lookup(keys, req_context) -> Iterable[bool]`：

- FsAsyncLookupManager（[tiering-fs.md](tiering-fs.md)）：对每个 key 调 `file_mapper.get_file_name`，批量 `os.path.exists` 或 C 扩展 `batch_lookup_C`（释放 GIL 的 `faccessat` 批量轮询）。
- ObjAsyncLookupManager（[tiering-obj.md](tiering-obj.md)）：构造 NIXL 探针 descriptor（addr=0, len=1, dev_id=0, obj_key=path），`agent.query_memory` 一次查全部。

tier 把自己的 `lookup()` 委托给 `AsyncLookupManager.lookup()`：返回 `True/False/None`，None 转 `LookupResult.RETRY`。`on_schedule_end()` 转发 `flush()`，`on_request_finished()` 转发 `cleanup(req_id)`（清掉只属于该请求的 key 状态，使用 reverse index `_req_keys`）。

### 7. `ExampleSecondaryTierManager`（`example/manager.py`）

参考实现：纯内存 dict 存 key，`submit_store/load` 立即在 `completed_jobs` push `JobResult`，`get_finished_jobs` 直接吐出。`drain_jobs` 是 no-op（同步 tier 无需等待）。用于：

- 介绍 tier API 最小骨架；
- 在 `tests/` 里测 `TieringOffloadingManager` 编排逻辑而不需真实 FS/NIXL/S3。

`custom_param` 参数演示从 tier_config 透传字段的能力。

---

## 为什么

- **CPU 是必经枢纽**：所有 secondary tier 都不能直接访问 GPU，所以"先 primary 后 secondary"的级联+回填两段式是必然结构。
- **layered orchestration over 抢占式 cache**：早期 vLLM 的 swap 是直接 GPU↔CPU 单层；新增 secondary 简单"挂载"在 primary 后面不需要重写 GPU 路径。
- **staged promotion + RETRY**：lookup 命中 secondary 时不阻塞 scheduler，先在 CPU 预留槽、延后批量 `submit_load`，下一步再来查；engine 自然通过 `has_pending_work` 继续 tick 直到 promotion 完成。
- **REQUEST_LEVEL policy**：某些 tier（如 P2P prefiller 给 decoder 推 KV）需要请求的完整 KV 上下文，而非仅"新算块"——这种 tier 通过 `on_new_request` 声明 REQUEST_LEVEL，让 `prepare_store` 把已 primary 块也走一次 cascade（`_cascade_existing_blocks_to_request_level_tiers`）。
- **延迟 `on_request_finished`**：complete_store 可能在 `on_request_finished` 之后才到（GPU→primary 异步），但其仍会触发 cascade 到 secondary tier——所以 secondary tier 的 `on_request_finished` 必须延迟到 `pending_primary_stores == 0`。
- **`AsyncLookupManager` 共享代码**：FS 的 `os.path.exists` 批量与 Obj 的 `query_memory` 批量本质都是"一组 key 异步查存在性"——抽出公共基类让 tier 实现只关注 IO。

---

## 怎么做

### 配置示例：CPU + FS + P2P 三层

```json
{
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 10737418240,
  "block_size": 16,
  "eviction_policy": "arc",
  "secondary_tiers": [
    {"type": "fs", "root_dir": "/mnt/ssd/kv", "n_read_threads": 16, "n_write_threads": 16},
    {"type": "p2p", "host": "0.0.0.0", "port": 7777, "backends": ["UCX"], "num_threads": 4}
  ]
}
```

### 自定义 secondary tier

```python
class MyTier(SecondaryTierManager):
    def __init__(self, offloading_spec, primary_kv_view, tier_type, my_param=0):
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        ...
    def lookup(self, key, req_context) -> LookupResult: ...
    def submit_store(self, job_metadata) -> None: ...
    def submit_load(self, job_metadata) -> None: ...
    def get_finished_jobs(self) -> Iterable[JobResult]: ...
    def on_new_request(self, req_context) -> RequestOffloadingContext: ...
    def drain_jobs(self) -> None: ...

# 注册：
SecondaryTierFactory.register_tier("mytier", "my_pkg.tier", "MyTier")

# extra_config:
# {"secondary_tiers": [{"type": "mytier", "my_param": 42}]}
```

`drain_jobs()` 必须真正阻塞——`reset_cache` 期间释放 primary 槽时不能有 IO 还在写 view。

### 添加异步 lookup 到自定义 tier

继承 `AsyncLookupManager` 实现 `batch_lookup`，tier 内部组合一个实例，`lookup() / on_schedule_end() / on_request_finished()` 转发给它。

---

## 与其它模块/系统配合

- 上游：[factory.md](factory.md) 把 `TieringOffloadingSpec` 注册到 `OffloadingSpecFactory`；`SecondaryTierFactory` 注册 tier type。
- primary tier：[cpu.md](cpu.md)（`CPUPrimaryTierOffloadingManager` 继承 `CPUOffloadingManager`）。
- 调用方：[07-distributed/kv-transfer/offloading](../07-distributed/kv-transfer/offloading.md) 把 `TieringOffloadingManager` 当 `OffloadingManager` 用，所有 `lookup/prepare_load/prepare_store` 调用走 tiering orchestrator。
- 示例 tier：`example/manager.py`。
- 实际 tier：[tiering-fs.md](tiering-fs.md) / [tiering-obj.md](tiering-obj.md) / [tiering-p2p.md](tiering-p2p.md)。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.9.x | 仅 CPU tier，无编排层 |
| v0.10 | **tiering 框架首发**：`SecondaryTierManager`/`SecondaryTierFactory`/`TieringOffloadingManager`/`CPUPrimaryTierOffloadingManager`；首个 secondary tier 为 `fs/`；`example/` 作为参考实现 |
| v0.10.x | `AsyncLookupManager` 抽出，FS tier 顺手用上——前身为简单同步 `os.path.exists` 调用 |
| v0.11 | P2P tier 落地（`p2p/`）；加入 `REQUEST_LEVEL` policy 与 `_cascade_existing_blocks_to_request_level_tiers`；`_pending_load_submissions` 与 `on_schedule_end` 批量 flush 优化；`drain_jobs` 成为 `SecondaryTierManager` 必实现方法，由 `reset_cache` 强约束 |
| main | `obj/` tier 落地；`_maybe_finalize_request` 引入以正确处理"on_request_finished 之后还有 complete_store cascade"的时序；禁用 `store_threshold>=2` 与 `self_describing_kv_events`（语义不兼容） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [cpu.md](cpu.md)：primary tier 实现。
- [tiering-fs.md](tiering-fs.md)、[tiering-obj.md](tiering-obj.md)、[tiering-p2p.md](tiering-p2p.md)：内置 secondary tier。
