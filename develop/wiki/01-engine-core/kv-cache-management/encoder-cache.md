# EncoderCacheManager（多模态编码器缓存）

[← Wiki 首页](../../README.md) > [引擎核心](../README.md) > [KV 缓存管理](README.md) > EncoderCacheManager

源码：`vllm/v1/core/encoder_cache_manager.py`（约 381 行）。`EncoderCacheManager` 管理多模态模型（如 LLaVA、Qwen-VL）编码器输出的内存感知 LRU 缓存；`EncoderDecoderCacheManager` 是 encoder-decoder 模型（如 Whisper）的简化变体。

## 是什么

### `EncoderCacheManager`（`encoder_cache_manager.py:17`）

构造参数：`cache_size`（以 encoder embedding 数量计的容量上限，来自 `MultiModalBudget.encoder_cache_size`）。

字段：
- `cache_size` / `num_free_slots` / `num_freeable_slots`：容量与当前可用（含可回收）。
- `cached: dict[str, set[str]]`：`mm_hash → 引用该 hash 的 request_id 集合`；空集表示数据在内存但无请求引用，可回收。
- `request_cached_ids: dict[str, set[int]]`：`request_id → 该请求缓存的 input_id 集合`。
- `freeable: OrderedDict[str, int]`：`mm_hash → num_encoder_embeds` 的 LRU 有序表，记录"零引用、可被驱逐"的条目。
- `freed: list[str]`：自上次 `get_freed_mm_hashes()` 以来实际被驱逐的 mm_hash 列表，清空后返回给 scheduler 通知 worker 释放物理内存。

核心方法：
- `check_and_update_cache(request, input_id) -> bool`：检查某 input 是否已缓存；若是且当前无引用，从 `freeable` 摘出并扣 `num_freeable_slots`；加 request_id 引用、记录到 `request_cached_ids`。
- `can_allocate(request, input_id, encoder_compute_budget, num_embeds_to_schedule) -> bool`：内存感知准入。先尝试 `num_free_slots`，不够再从 `freeable` LRU 头驱逐到够为止；若 `freeable` 也不够则 False。驱逐不立即释放物理内存，只是把 mm_hash 入 `freed`。
- `allocate(request, input_id)`：扣 `num_free_slots`/`num_freeable_slots`，记引用。仅预约，物理内存由 model runner 写入。
- `get_cached_input_ids(request) -> set[int]`。
- `free_encoder_input(request, input_id)`：撤销该请求对该 input 的引用；若 hash 引用集变空，移入 `freeable` 并恢复 `num_freeable_slots`。物理内存保持（待驱逐时才释放）。
- `free(request)`：撤销该请求所有缓存引用（请求 finish/abort 时调）。
- `reset()`：清空所有状态；用于权重更新后失效 stale embedding。
- `get_freed_mm_hashes() -> list[str]`：drain `freed`，scheduler 把它放进 `SchedulerOutput.free_encoder_mm_hashes` 通知 worker。

辅助函数 `compute_mm_encoder_budget(scheduler_config, mm_max_toks_per_item) -> (compute_budget, cache_size)`：`compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)`；当 `disable_chunked_mm_input=True` 且 max_tokens_per_mm_item > max_num_batched_tokens 时报错。

### `EncoderDecoderCacheManager`（`encoder_cache_manager.py:323`）

`EncoderCacheManager` 的"调度-only"子类，给 encoder-decoder 模型（Whisper）用：
- 不维护 `cached`/`freeable`/`freed`；用 `allocated: list[str]` + `to_free: list[str]` 两队列实现"上一批 free、当前批 allocated"的延迟释放语义。
- `check_and_update_cache` 恒返回 False（enc-dec 不复用缓存）。
- `can_allocate`/`allocate`/`free`/`get_freed_mm_hashes` 简化实现。
- 注释说明"待 enc-dec 模型与 MM 模型差异缩小后并入 `EncoderCacheManager`"。

## 为什么

- **避免重算昂贵编码器**：视觉/音频编码器一次前向可能数百 ms；同图像被多请求引用（few-shot、RAG）时复用 embedding 显著降本。
- **按 embedding 而非 token 计量**：注释 `encoder_cache_manager.py:42-46` 说明：cache_size 以"encoder embedding 数"（如一张图的 576 个 vision token）为单位，不含 break/text token；这样容量与 modality item 数量直接对应，便于 `MultiModalBudget` 推算。
- **零引用 ≠ 立即释放**：`free_encoder_input` 仅把 hash 移入 `freeable`、恢复计数；物理内存保留到下次 `can_allocate` 驱逐时才通过 `freed` 通知 worker。这避免"释放-重分配"抖动。
- **驱逐 → worker 通知**：scheduler 在 `SchedulerOutput.free_encoder_mm_hashes` 携带驱逐的 mm_hash；worker 端 model runner 据此真正释放 GPU tensor。逻辑状态（manager）与物理状态（worker）解耦。
- **compute_budget 独立**：encoder 前向算力预算（`max_num_encoder_input_tokens`）与 cache 容量是两个独立维度——前者限制单步能算多少 encoder token，后者限制能存多少。`can_allocate` 同时检查两者。
- **chunked multimodal input**：长视频等多模态输入可被分多步处理；编码器预算 + 缓存配合让 chunking 可控。`disable_chunked_mm_input` 关闭此特性并强校验。
- **enc-dec 简化路径**：Whisper 等模型每次推理都重新算编码器（音频不同），无缓存复用需求，但调度仍需"上一批 free、当前批 allocated"的延迟语义以避免 forward 中释放正在用的 tensor。

## 怎么做

### scheduler 调用流程（`Scheduler._try_schedule_encoder_inputs`）

```mermaid
flowchart TD
    A[_try_schedule_encoder_inputs<br/>request, num_computed_tokens, num_new_tokens, budget] --> B[遍历 request.mm_features]
    B --> C{mm_position 与 [num_computed, num_computed+num_new] 重叠?}
    C -- 否 --> B
    C -- 是 --> D{check_and_update_cache 命中?}
    D -- 是 --> E[无需重算, 继续下一 input]
    D -- 否 --> F[can_allocate:<br/>check compute_budget + cache + eviction]
    F -- False --> G[截断 num_new_tokens 到该 input 之前<br/>return [], 截断后, 还原budget, []]
    F -- True --> H[allocate request, input_id<br/>加入 encoder_inputs_to_schedule]
    H --> B
    B --> Done[return encoder_inputs_to_schedule, num_new_tokens, new_budget, external_load]
```

关键：若某 input 因预算/缓存不足无法 schedule，则把 `num_new_tokens` 截断到该 input 的 `mm_position.offset` 之前（让 decoder token 先跑），剩余等下一步。

### 释放流程（`Scheduler._free_encoder_inputs`，`scheduler.py:1915`）

```python
spec_lookahead = 1 if self.use_eagle else 0
for input_id in list(cached_encoder_input_ids):
    mm_feature = request.mm_features[input_id]
    start_pos = mm_feature.mm_position.offset
    num_tokens = mm_feature.mm_position.length
    if self.is_encoder_decoder and request.num_computed_tokens > 0:
        # Whisper: 一旦生成首 token 就释放编码器输入
        self.encoder_cache_manager.free_encoder_input(request, input_id)
    elif (start_pos + num_tokens + spec_lookahead
          <= request.num_computed_tokens - request.num_output_placeholders):
        # decoder 已计算过该 input + lookahead, 可释放
        self.encoder_cache_manager.free_encoder_input(request, input_id)
```

`spec_lookahead` 让 EAGLE draft 模型的 +1 预测也能读到编码器输出，避免过早释放。

### `SchedulerOutput.free_encoder_mm_hashes`

```python
# scheduler.py:1106
scheduler_output = SchedulerOutput(
    ...
    free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
)
```

worker 端 model runner 在 prepare_inputs 阶段消费此列表，从其 encoder cache 字典中删除对应 mm_hash 的 tensor 并释放显存。

### abort 路径

`Scheduler._preempt_request` / `_free_request` 调 `encoder_cache_manager.free(request)`，撤销该请求所有引用；若某 mm_hash 引用集因此变空，移入 `freeable` 等待未来驱逐。物理内存不立即释放（一致性：freeable 条目仍可能在下一步被驱逐进 `freed`）。

### 与 EC connector 协同

`ec_connector.update_state_after_alloc(request, input_id)` 在 scheduler allocate 后调用，让外部 encoder cache connector（如跨节点共享编码器输出）记录状态。`ec_connector.ensure_cache_available(request, num_computed_tokens)` 在 waiting 段最早处检查：若 EC connector 提供该 mm 数据但本地缓存未就绪，跳过该请求等下一步。

## 与其它模块/系统配合

- **[Scheduler](../scheduler/scheduler.md)**：唯一调用方；`_try_schedule_encoder_inputs` 是请求侧入口，`_free_encoder_inputs` 是释放侧；`_preempt_request`/`_free_request` 调 `free(request)`。
- **[Request.mm_features](../data-model.md)**：`MultiModalFeatureSpec.identifier` 是 cache key（`InputProcessor._get_mm_identifier` 在 LoRA 启用时前缀化）；`mm_position.offset`/`length` 决定调度时机。
- **[InputProcessor](../input-processor.md)**：`mm_encoder_cache_size` 通过 `MultiModalBudget.encoder_cache_size` 喂入 manager 构造；`skip_prompt_length_check` 源自 mm processor info。
- **[02-execution](../../02-execution/README.md)**：worker 端 model runner 维护物理 encoder cache dict，按 `free_encoder_mm_hashes` 释放；按 `scheduled_encoder_inputs` 决定本步哪些 input 要跑编码器。
- **[多模态子系统](../../11-multimodal/README.md)**：`MultiModalBudget` 计算 `encoder_compute_budget`/`encoder_cache_size`；`mm_max_toks_per_item` 决定单 item token 数。
- **[Speculative decode](../../06-sampling-decoding/README.md)**：`use_eagle` 让 `_free_encoder_inputs` 加 `spec_lookahead=1` 延迟释放。
- **[EC connector](../../15-kv-cache-offload/README.md)**：`ec_connector.ensure_cache_available`/`update_state_after_alloc` 在 waiting 段最早处与 allocate 后调用。
- **[KV connector](../../15-kv-cache-offload/README.md)**：`WAITING_FOR_REMOTE_KVS` 不涉及编码器缓存，但若 EC connector 也配置则两者并行。

## 历史版本演进

- **v0.5/v0.6（v0）**：v0 `LRUCache` _mapper 在 scheduler 内联；多模态缓存与 KV cache 解耦较弱。
- **v0.7（v1 落地）**：`EncoderCacheManager` 抽出，`num_free_slots`/`num_freeable_slots`/`freeable` 三态机制成型；`freed` 列表与 `SchedulerOutput.free_encoder_mm_hashes` 协议建立。
- **v0.7.x**：`EncoderDecoderCacheManager` 为 Whisper 等 enc-dec 模型引入；`compute_mm_encoder_budget` 工具函数抽出。
- **v0.8（v1 默认）**：EAGLE `spec_lookahead` 让释放延迟 1 token；`disable_chunked_mm_input` 校验路径加入。
- **v0.9**：EC connector（external encoder cache）接入；`ensure_cache_available`/`update_state_after_alloc` 钩子；enc-dec 路径与 MM 路径差异显式标注（"待合并"注释）。
- **v0.10 / v0.11 / main**：保持稳定；`MultiModalBudget.reset_cache` 在 InputProcessor 构造后调用避免重复持有。具体版本归属（待核实）。

[← 返回引擎核心首页](../README.md)

## 参见

- [kv-cache-manager.md](./kv-cache-manager.md) — `num_encoder_tokens` 走 `CrossAttentionManager` 单独路径。
- [spec.md](./spec.md) — `EncoderOnlyAttentionSpec`/`CrossAttentionSpec` 的容量计算。
- [../scheduler/scheduler.md](../scheduler/scheduler.md) — `_try_schedule_encoder_inputs`/`_free_encoder_inputs` 的调用上下文。
- [../input-processor.md](../input-processor.md) — `mm_encoder_cache_size` 与 `MultiModalBudget`。
