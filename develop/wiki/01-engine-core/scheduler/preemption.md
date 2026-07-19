# 抢占机制（Preemption）

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [Scheduler](README.md) > 抢占机制

源码：`vllm/v1/core/sched/scheduler.py` 的 `_preempt_request`（`scheduler.py:1140`）、`_free_request_blocks`（`scheduler.py:2130`）、`_drain_deferred_frees`（`scheduler.py:2145`）、`reset_prefix_cache(reset_running_requests=True)`（`scheduler.py:2196`）。

## 是什么

抢占是调度器在 KV cache 资源不足时，主动把 running 队列中的请求"踢回" waiting 队列、释放其 KV block，让更高优先级或更早到达的请求能继续调度的机制。v1 的抢占语义与 v0 类似但实现简化：

- **触发点**：`schedule()` 在 running 段为某请求 `kv_cache_manager.allocate_slots(...)` 返回 `None` 时进入抢占循环。
- **抢占对象选择**：
  - FCFS：`self.running.pop()`（最后一个，最晚进入 running 的）。
  - Priority：`max(self.running, key=lambda r: (r.priority, r.arrival_time))`——选最低优先级（priority 数值最大）、最晚到达的请求；若它在 `scheduled_running_reqs` 中（本步已调度），还需撤销其 token 预算与 block 记录。
- **`_preempt_request` 后果**：释放 KV block + encoder cache；`num_computed_tokens=0`（全量重算）；`spec_token_ids=[]`；`num_preemptions+=1`；记 `PREEMPTED` event；`waiting.prepend_request`；加入 `reset_preempted_req_ids`（V2 model runner 需要显式 flush persistent batch）。
- **重调度**：被抢占请求在下一步以 `RequestStatus.PREEMPTED` 进入 waiting；`schedule` waiting 段把它当 `scheduled_resumed_reqs` 处理（需 prefix cache 命中以避免全量重算，但 `num_computed_tokens=0` 通常导致重算）。
- **deferred free**：当 `defer_block_free=True`（KV consumer + 多 inflight batch），`_free_request_blocks` 不立刻归还 block，而是 `pop_blocks_for_free` + `deferred_frees.append((sched_step_seq, blocks))`；`_drain_deferred_frees` 在 `processed_step_seq >= fence_seq` 时把 block 反向（tail first）归还 `block_pool.free_blocks`。
- **`reset_prefix_cache(reset_running_requests=True)`**：强制抢占所有 running 请求以清空 prefix cache；同时设 `async_tokens_to_discard = num_output_placeholders` 让 async 调度丢弃过期输出帧。

## 为什么

- **资源回收的唯一手段**：KV cache 满时只能撤请求；不像 v0 还可 "recompute its KV"（部分保留），v1 简化为全量重算（`num_computed_tokens=0`），减少状态复杂度。配合 prefix cache 可让"重算"实际只重算少量 block。
- **公平性**：FCFS 抢最后一个（最新进入）合理——它"还没怎么算"；Priority 反向选最低优先级，保证高优先级不被饿死。
- **本步已调度的撤销**：若被抢占请求本步已分配 token/scheduled_running_reqs/encoder_inputs，必须把 `token_budget` 还回去、`req_to_new_blocks`/`num_scheduled_tokens`/`scheduled_spec_decode_tokens`/`scheduled_encoder_inputs` 全部清理，避免预算泄漏与 worker 端空跑。
- **encoder cache 同步释放**：抢占后 `_preempt_request` 调 `encoder_cache_manager.free(request)`，否则多模态请求会逐渐耗尽 encoder 缓存。
- **deferred free 必要性**：async scheduling/PP 下，一个 step 可能正在写某 request 的 KV block（forward pass 还没完成）；若此时该 request 被 finish/abort 并立刻 free block 给 consumer connector 重新分配并 load，会与正在进行的 GPU 写入竞争。deferred free 用 `sched_step_seq`/`processed_step_seq` 的围栏保证只在 GPU 写完成后归还。
- **reset_prefix_cache 的 async 兼容**：async 调度下被抢占请求可能还有 in-flight 输出帧在路上（`num_output_placeholders > 0`），若不丢弃会让 `_update_request_with_output` 把无效 token 当真实输出。`async_tokens_to_discard` 在 `AsyncScheduler._update_request_with_output` 中递减并返回空 token 列表。
- **`_inflight_prefills` 清理**：`_preempt_request` 调 `self._inflight_prefills.discard(request)`，避免被抢占的 prefill 仍占用 reserved_blocks 计算（KV connector 异步加载时的死锁防护）。

## 怎么做

### 抢占循环（`scheduler.py:533-583`）

```mermaid
flowchart TD
    A[allocate_slots 返回 None] --> B{policy?}
    B -- Priority --> C[preempted = max running by (priority, arrival_time)]
    B -- FCFS --> D[preempted = running.pop]
    C --> E{preempted in scheduled_running_reqs?}
    E -- 是 --> F[撤销本步调度:<br/>scheduled_running_reqs.remove<br/>token_budget += num_scheduled_tokens.pop<br/>req_to_new_blocks.pop<br/>scheduled_spec_decode_tokens.pop<br/>scheduled_encoder_inputs.pop + 还原 encoder_compute_budget<br/>req_index -= 1]
    E -- 否 --> G[running 已在 pop 时移除]
    D --> G
    F --> H[_preempt_request preempted, ts]
    G --> H
    H --> I[preempted_reqs.append preempted]
    I --> J{preempted == request?}
    J -- 是 --> K[break: 无更多可抢占]
    J -- 否 --> L[回到 allocate_slots 重试]
```

注意 FCFS 路径用 `self.running.pop()` 直接修改了 running，而 Priority 路径先 `self.running.remove(preempted_req)`。`req_index -= 1` 让循环回头重新处理刚被撤销位置之后的请求（其实策略略不同——`continue` 风格）。

### _preempt_request（`scheduler.py:1140`）

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    assert request.status == RequestStatus.RUNNING
    self._free_request_blocks(request)          # 触发 deferred free 或立即 free
    self.encoder_cache_manager.free(request)
    self._inflight_prefills.discard(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    if request.spec_token_ids:
        request.spec_token_ids = []
    request.num_preemptions += 1
    if self.log_stats:
        request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
    self.waiting.prepend_request(request)       # 插回 waiting 队首
    self.reset_preempted_req_ids.add(request.request_id)
```

### deferred free 围栏（`scheduler.py:2130`）

```python
def _free_request_blocks(self, request: Request):
    if not self.defer_block_free or (
        request.last_sched_seq <= self.processed_step_seq
    ):
        # 已无 in-flight GPU 写：立刻 free
        self.kv_cache_manager.free(request)
        return
    # 否则把 block 暂存，等 update_from_output 推进 processed_step_seq
    blocks = self.kv_cache_manager.pop_blocks_for_free(request)
    if blocks:
        self.deferred_frees.append((self.sched_step_seq, blocks))

def _drain_deferred_frees(self):
    while self.deferred_frees:
        fence, _ = self.deferred_frees[0]
        if fence > self.processed_step_seq:
            break
        _, blocks = self.deferred_frees.popleft()
        # 反向归还：tail block 先被驱逐（caching enabled 时影响 LRU 顺序）
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
```

围栏序号语义：
- `sched_step_seq`：每次非空 schedule +1（`scheduler.py:1128`）。
- `processed_step_seq`：每次非空 update_from_output +1（`scheduler.py:1515`）。
- `request.last_sched_seq`：在 `_update_after_schedule` 中记录该请求最后一次被调度的 `sched_step_seq`。

### reset_prefix_cache 强制抢占（`scheduler.py:2206`）

```python
if reset_running_requests:
    timestamp = time.monotonic()
    while self.running:
        request = self.running.pop()
        self._preempt_request(request, timestamp)
        # 丢弃 async 调度下还在路上的输出帧
        request.async_tokens_to_discard = request.num_output_placeholders
        request.num_output_placeholders = 0
    self.prev_step_scheduled_req_ids.clear()  # V1 路径需要
reset_successful = self.kv_cache_manager.reset_prefix_cache()
if reset_connector:
    reset_successful = self.reset_connector_cache() and reset_successful
```

### Priority 抢占的 `req_index` 调整

Priority 模式下被抢占请求可能位于 `scheduled_running_reqs` 中（已本步调度）；撤销时除清理预算外还 `req_index -= 1`，因为 `running[req_index]` 已被移除，后续 `req_index += 1` 在循环末尾仍会推进，所以 -=1 让指针停在原索引位置继续处理下一个。这是相对隐晦的细节（`scheduler.py:570`）。

## 与其它模块/系统配合

- **[scheduler.md](./scheduler.md)**：抢占是 schedule running 段的失败回退路径。
- **[queues.md](./queues.md)**：`waiting.prepend_request` 让被抢占请求优先重调度；`reset_preempted_req_ids` 由 V2 model runner 消费。
- **[kv-cache-manager.md](../kv-cache-management/kv-cache-manager.md)**：`allocate_slots` 返回 None 是触发条件；`pop_blocks_for_free` 抽取 block 而不归还，留给 deferred free。
- **[block-pool.md](../kv-cache-management/block-pool.md)**：`free_blocks(reversed(blocks))` 把 block 按"tail first" 还回 free queue，使 caching 模式下 tail block 优先被驱逐，符合 LRU 语义。
- **[encoder-cache.md](../kv-cache-management/encoder-cache.md)**：抢占同步释放 encoder 引用，否则多模态请求会泄漏。
- **[AsyncScheduler](./scheduler.md)**：`async_tokens_to_discard` 在 `_update_request_with_output` 中递减并丢弃过期帧。
- **[EngineCore](../engine-core-process.md)**：`shutdown_timeout=0` 时 `finish_requests(None, FINISHED_ABORTED)` 走类似路径；`pause_scheduler(abort)` 也走 `finish_requests` 而非 `_preempt_request`（不重排，直接 abort）。
- **[KV connector](../../15-kv-cache-offload/README.md)**：`WAITING_FOR_REMOTE_KVS` 请求若被 abort，`delay_free_blocks=True` 等异步 KV 完成；`_inflight_prefills` 跟踪 prefill 中请求的 reserved_blocks。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 `PREEMPTION_MODE=RECOMPUTE/RECOMPUTE_DEFAULT`/`SWAP` 三种模式；v1 简化为只有"recompute 全量"路径（`num_computed_tokens=0`），swap 模式废弃。
- **v0.7（v1 落地）**：`_preempt_request` 简化成型；Priority 抢占引入 `max(running, key=...)` 反向选择；`reset_preempted_req_ids` 为 V2 model runner 加入。
- **v0.8（v1 默认）**：`defer_block_free` 引入，处理 async scheduling/PP 与 KV consumer 的写后释放竞争；`_inflight_prefills` set 跟踪 reserved_blocks。
- **v0.9**：`async_tokens_to_discard` 字段加入 `reset_prefix_cache` 路径，确保 async 调度下强制抢占不会让过期输出污染状态。
- **v0.10**：`scheduler_reserve_full_isl` 准入门控减少 mid-prefill OOM，间接降低抢占频率。
- **v0.11 / main**：deferred free 围栏机制稳定；Marconi APC 让被抢占请求重调度时 prefix cache 命中更高，减少重算代价。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [scheduler.md](./scheduler.md) — 抢占在 schedule 主循环中的位置。
- [queues.md](./queues.md) — `prepend_request` 的语义。
- [chunked-prefill.md](./chunked-prefill.md) — `full_sequence_must_fit` 准入避免抢占。
- [kv-cache-management/block-pool.md](../kv-cache-management/block-pool.md) — `free_blocks` 的 LRU 顺序。
