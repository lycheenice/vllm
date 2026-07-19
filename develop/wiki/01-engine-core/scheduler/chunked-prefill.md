# Chunked Prefill 策略

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [Scheduler](README.md) > Chunked Prefill

源码：`vllm/v1/core/sched/scheduler.py` 的 `schedule()` 主循环（`scheduler.py:396` 起）与 `_mamba_block_aligned_split`（`scheduler.py:338`）。相关配置：`scheduler_config.enable_chunked_prefill`、`max_num_batched_tokens`、`max_num_scheduled_tokens`、`long_prefill_token_threshold`、`scheduler_reserve_full_isl`、`prefill_schedule_interval`。

## 是什么

chunked prefill 是 v1 调度器的默认行为：当一条请求的 prompt 超过单步能容纳的 token 预算时，把 prefill 拆成多个 chunk 跨步执行，与 decoding 请求同 batch 共享预算。其核心不在某个独立类，而在 `schedule()` 内部对 `num_new_tokens` 的多级约束：

1. **token 预算**：`token_budget = self.max_num_scheduled_tokens`（默认等于 `max_num_batched_tokens`，可独立配置）；`PAUSED_ALL` 时归零。
2. **running 段切分**：每个 running 请求的 `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`，先与 `long_prefill_token_threshold` 取 min，再与剩余 `token_budget` 取 min。
3. **waiting 段切分**：`num_new_tokens = request.num_tokens - num_computed_tokens`（含 prefix cache 命中后的剩余），同样受 `threshold` 与 `token_budget` 约束。
4. **chunked_prefill 开关**：`enable_chunked_prefill=False` 时，若 waiting 请求 `num_new_tokens > token_budget` 直接 `break`，不在本步 admit；为 True 时允许首 chunk 不满 prompt 全长。
5. **`long_prefill_token_threshold`**：>0 时把单请求的 `num_new_tokens` 上限截到该值，剩余留到后续步——这是细化 chunked prefill 的旋钮。
6. **Mamba block-aligned split**：`_mamba_block_aligned_split` 强制 num_new_tokens 是 `block_size` 倍数（除最末 chunk），保证 SSM 状态按 block 缓存。
7. **DP prefill 节拍**：`prefill_schedule_interval` > 1 时，`_should_throttle_prefills` 在非对齐步返回 True，`defer_prefills` 跳过 running 中 `is_prefill_chunk=True` 的请求与 waiting 中未完成的 prefill。
8. **`scheduler_reserve_full_isl`**：`full_sequence_must_fit=True` 时的准入门控，要求整个序列（减去 prefix hit）能放入 KV cache 才 admit，避免 chunked 把请求放进来后中途 OOM。

## 为什么

- **吞吐与 TTFT 折中**：长 prompt 一次性 prefill 会占满 GPU 几个 step，decode 请求被饿死；chunked 让 prefill 与 decode 同步推进。但完全无限制的 chunked 会让长 prompt 的首 token 延迟变高，故有 `long_prefill_token_threshold` 给"长但不要太碎"的折中。
- **预算共享**：running 与 waiting 在同一 `token_budget` 池里抢，保证 decode（每步 1 token/请求）总能拿到剩余预算。
- **prefix cache 兼容**：`get_computed_blocks` 先扣除已命中 token，`num_new_tokens` 只算未命中部分；chunked 仅作用于真正要算的 token。
- **Mamba 状态对齐**：SSM 的 `cached_state` 按 block 存储时，chunk 末尾必须落在 block 边界才能写状态；`mamba_cache_mode='align'` 模式下 `_mamba_block_aligned_split` 强制对齐，`'all'` 模式则不限。
- **Eagle prune 与最末 chunk**：Eagle 模式下 FullAttn 会 prune 最后一个匹配 block，为避免 Mamba cache miss，最末 chunk 必须 ≥ `block_size`（见 `_mamba_block_aligned_split` 注释）。
- **DP MoE 平衡**：MoE DP 的 expert all-to-all 要求各 rank 同步 prefill，否则 all-to-all 形状不匹配；`prefill_schedule_interval` 让 prefill 只在 `step_counter % interval == 0` 步发生，其它步纯 decode，各 rank 自然对齐。`prefill_capacity_bound` 处理"上次对齐步已耗尽 waiting"的饱和场景，避免盲目 throttle 致死锁。
- **admission 准入**：`full_sequence_must_fit` 用于 sliding window / chunked-local 等会回收 block 的 spec，确保 admission 时的峰值占用与 startup pool sizer 一致，避免 mid-prefill OOM。

## 怎么做

### running 段切分（`scheduler.py:473`）

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
# spec decode 还需限制不超 max_model_len
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - request.num_computed_tokens - num_sampled_tokens_per_step,
)
```

随后 `_try_schedule_encoder_inputs` 可能再次缩减 `num_new_tokens`（若 encoder 预算/缓存不足）；`_mamba_block_aligned_split` 在需要时再对齐。

### waiting 段切分（`scheduler.py:806`）

```python
num_new_tokens = request.num_tokens - num_computed_tokens

# spec decode uniform padding: 为保 cudagraph，把 decode 也 pad 成 1+num_spec_tokens
if (num_spec_tokens > 0 and dynamic_sd_lookup is None
        and num_new_tokens == 1
        and scheduled_running_reqs and not prefill_scheduled):
    num_new_tokens = 1 + num_spec_tokens
    pad_spec_decode = True

threshold = long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold

# chunked_prefill=False 时不允许 admit 超预算的请求
if (not enable_chunked_prefill and num_new_tokens > token_budget):
    break

num_new_tokens = min(num_new_tokens, token_budget)
```

关键：`enable_chunked_prefill=False` 时 `break` 直接结束 waiting 段，请求继续排队等下一步（这与 v0 行为类似）；`True` 时请求被 admit，剩余 token 留到下一 step，请求被标 `is_prefill_chunk=True` 加入 running。

### is_prefill_chunk 判定

`_update_after_schedule`（`scheduler.py:1164`）：
```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```
即"仍有未计算 token"→ 还在 prefill chunking 状态。这影响：
- spec decode：`is_prefill_chunk=True` 时忽略 draft token。
- DP throttle：`defer_prefills = ... and any(not r.is_prefill_chunk for r in running)` 判断是否有非 chunk 的纯 decode 在跑。
- struct output：`has_structured_output_requests |= use_structured_output and not is_prefill_chunk`（grammar 仅在非 prefill chunk 生效）。

### Mamba block-aligned split（`scheduler.py:338`）

只在 `need_mamba_block_aligned_split`（=`has_mamba_layers and mamba_cache_mode=='align'`）时调用。逻辑分三段：

1. **中间 chunk**：`num_computed_tokens_after_sched < last_cache_position` → `num_new_tokens //= block_size * block_size`（向下对齐）。
2. **跨 cache_position 边界**：`num_computed < last_cache_position < after_sched` → `num_new_tokens = last_cache_position - num_computed_tokens`（强制到边界）。
3. **最末 chunk**：`else` → 不变。

Eagle 模式额外把 `last_cache_position` 后退一个 block（`max(last_cache_position - block_size, 0)`），防止 Eagle prune 导致 Mamba cache miss。Marconi APC 优化：当未缓存的公共前缀 ≥ block_size 且大于 num_new_tokens 时，把 chunk 缩到公共前缀长度（对齐 block_size）。

### DP prefill 节拍（`scheduler.py:1916`，`DPEngineCoreProc`）

```python
def _should_throttle_prefills(self) -> bool:
    return (
        self.prefill_schedule_interval > 1
        and self.step_counter % self.prefill_schedule_interval != 0
    )
```

`schedule` 中：
```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```
- running 段：`if defer_prefills and request.is_prefill_chunk: continue`（已有 decode 在跑，跳过 prefill chunk）。
- waiting 段：`elif defer_prefills and num_computed_tokens < request.num_tokens - 1: break`（不在对齐步，整体跳过 waiting admit）。

`prefill_capacity_bound` 在每次非 defer 步末更新为 `bool(self.waiting)`（`scheduler.py:1019`）：若上次对齐步已耗尽 waiting，本次即使非对齐步也允许 prefill（否则空等）。

### full_sequence_must_fit 准入（KVCacheManager.allocate_slots）

`Scheduler` 把 `scheduler_reserve_full_isl` 透传给 `allocate_slots(full_sequence_must_fit=...)`。后者（`kv_cache_manager.py:376`）：
```python
num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
    num_tokens=full_num_tokens, ..., apply_admission_cap=True,
)
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > block_pool.get_num_free_blocks():
    return None  # 不 admit
```
与 `SlidingWindowSpec.max_admission_blocks_per_request` / `ChunkedLocalAttentionSpec` 的回收感知上限制保持一致，避免调度器 admit 后下游 OOM。

## 与其它模块/系统配合

- **[scheduler.md](./scheduler.md)**：本页是其核心切分逻辑的展开。
- **[preemption.md](./preemption.md)**：`token_budget` 不足时优先压缩 waiting admit；running 内资源不足才触发抢占。
- **[kv-cache-management/kv-cache-manager.md](../kv-cache-management/kv-cache-manager.md)**：`allocate_slots` 是准入门控；`num_new_computed_tokens` 来自 prefix cache 命中。
- **[Encoder cache](../kv-cache-management/encoder-cache.md)**：`_try_schedule_encoder_inputs` 可能把 `num_new_tokens` 截断到 encoder 预算/缓存上限。
- **[Speculative decode](../../06-sampling-decoding/README.md)**：`pad_spec_decode`、`num_lookahead_tokens`、`dynamic_sd_lookup` 都影响 num_new_tokens。
- **[EngineCore DP](../engine-core-process.md)**：`DPEngineCoreProc._should_throttle_prefills` 注入节拍。
- **[Mamba / SSM](../../05-attention/README.md)**：`mamba_cache_mode` 决定是否需要 block-aligned split。

## 历史版本演进

- **v0.5**：v0 引入 chunked prefill（`--enable-chunked-prefill`），但默认 False；与 decode 分离调度。
- **v0.7（v1 落地）**：v1 调度器把 chunked prefill 内化为"统一 num_new_tokens 切分"；默认开启；`max_num_scheduled_tokens` 独立于 `max_num_batched_tokens`。
- **v0.7.x**：`long_prefill_token_threshold` 引入，让长 prompt 不被切过碎。
- **v0.8（v1 默认）**：`async_scheduling` 下 `num_output_placeholders` 影响 chunking；`is_prefill_chunk` 显式跟踪。
- **v0.9**：`scheduler_reserve_full_isl` 准入门控；DP `prefill_schedule_interval` 加入。
- **v0.10**：`prefill_capacity_bound` 防饱和死锁；Marconi APC（Mamba `align` 模式）— 公共前缀优先缓存。
- **v0.11 / main**：`disable_chunked_mm_input` 在 `compute_mm_encoder_budget` 中校验（encoder_cache_manager.py）。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [scheduler.md](./scheduler.md) — 主循环与本页切分逻辑的上下文。
- [preemption.md](./preemption.md) — 资源不足时的退化路径。
- [kv-cache-management/kv-cache-manager.md](../kv-cache-management/kv-cache-manager.md) — 准入判断的实际执行点。
- [queues.md](./queues.md) — `waiting` 队列在 chunked 关闭时的 break 行为。
