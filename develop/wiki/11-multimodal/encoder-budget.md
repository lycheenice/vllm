# encoder_budget.py · 多模态编码器预算

[← Wiki 首页](../README.md) > [多模态](../README.md) > encoder-budget

## 是什么

`vllm/multimodal/encoder_budget.py` 提供 `MultiModalBudget` 与 `get_mm_max_toks_per_item`，在引擎初始化期计算"每个模态单项最大 token 数"、编码器计算预算、编码器缓存上限、以及每 prompt / 每 batch 的最大 item 数。这些数字驱动 `Scheduler` 的 `EncoderCacheManager.can_allocate` 准入决策与 `InputProcessor` 的 prompt 长度校验。

## 为什么

多模态模型推理时，图像 / 视频 / 音频要先过编码器塔（ViT / AudioEncoder）输出 embedding，再注入 LLM 输入序列。这个塔的计算与显存开销被 v1 调度器单独预算：

- **编码器计算预算**：每步允许新算多少 encoder embedding token，防止一个超大图把整步预算吃光、阻塞文本请求。
- **编码器缓存上限**：GPU 上能存多少 encoder embedding token，超过即驱逐旧项。
- **per-prompt / per-batch item 上限**：把上述预算换算成"可同时调度多少张图"，避免 `--limit-mm-per-prompt` 设过大时实际跑不动。

`get_mm_max_toks_per_item` 决定预算转换的"汇率"：每张图最多展开成多少 token。如果模型能自报（`get_mm_max_tokens_per_item` 返回非 None），用快速路径；否则回退到造最大 dummy input 跑一遍 processor 量 `mm_placeholders`，对 Qwen2.5-VL 等大图模型会贡献数秒启动延迟。

## 怎么做

### get_mm_max_toks_per_item

`get_mm_max_toks_per_item(model_config, mm_registry, processor, mm_counts)`（`:15`）：

1. 调 `processor.info.get_mm_max_tokens_per_item(seq_len=model_config.max_model_len, mm_counts=mm_counts)`。模型若实现了快速路径返回非 None，直接 return。
2. 否则 `mm_registry.get_dummy_mm_inputs(model_config, mm_counts=mm_counts, processor=processor)`，跑一遍 processor 拿 `mm_placeholders`。
3. 返回 `{modality: sum(item.get_num_embeds() for item in placeholders)}`。注意用 `get_num_embeds()`（受 `is_embed` 掩码影响）而非 `length`，与 EVS 等剪枝场景一致。

### MultiModalBudget

`MultiModalBudget.__init__(vllm_config, mm_registry)`（`:47`）流程：

1. 取 `max_model_len` / `max_num_seqs`。
2. 在 `set_default_torch_num_threads()` 上下文里（避免启动期 hang）：
   - 用 `mm_registry.processor_only_cache_from_config(vllm_config)` 取一个临时 cache（仅用于内部 processor 调用，事后 `reset_cache`）。
   - `mm_registry.create_processor(model_config, cache=cache)` 得到 processor。
   - 读 `mm_config.enable_mm_embeds`、`processor.info.supported_mm_limits` / `allowed_mm_limits`。
   - 把模态分为 `tower_modalities`（`mm_limits > 0`，过编码器塔）与 `embed_only_modalities`（`enable_mm_embeds and mm_limits == 0`，旁路 embedding 仍占 encoder cache 但不占 tower 算力）。
   - `get_mm_max_toks_per_item(model_config, mm_registry, processor, mm_counts=dict.fromkeys(active_modalities, 1))`。
3. 过滤到 `active_mm_max_toks_per_item`（仅含独立 placeholder token 的模态——Qwen3Omni `use_audio_in_video=True` 会共享 placeholder，部分模态不在 dict 里）。
4. 派生 `tower_mm_max_toks_per_item`（仅 tower 模态），用于 per-prompt/batch 上限。
5. `compute_mm_encoder_budget(scheduler_config, active_mm_max_toks_per_item)`（来自 `vllm/v1/core/encoder_cache_manager.py:269`）算出 `(encoder_compute_budget, encoder_cache_size)`——前者取 `max(max_num_encoder_input_tokens, max_mm_tokens_per_item)`，后者取 `max(encoder_cache_size, max_mm_tokens_per_item)`，并校验 `disable_chunked_mm_input` 下 `max_tokens_per_mm_item <= max_num_batched_tokens`。
6. 对每个 tower 模态调 `_get_max_items`。

`_get_max_items(modality, max_tokens_per_item)`（`:140`）：

- `max_tokens_per_item == 0` → `(0, 0)`。
- `encoder_budget = min(compute_budget, cache_size)`；0 → `(0, 0)`。
- `max_encoder_items_per_batch = encoder_budget // max_tokens_per_item`。
- `max_items_per_prompt = max(1, min(mm_limit, max_model_len // max_tokens_per_item))`。
- 非 chunked prefill 时，`max_num_reqs = min(max_num_reqs, max_num_batched_tokens // max_tokens_per_item)`。
- `max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt`。
- `max_items_per_batch = max(1, min(max_encoder_items_per_batch, max_decoder_items_per_batch))`。

最终对外属性：`encoder_compute_budget` / `encoder_cache_size` / `mm_max_toks_per_item`（tower only）/ `mm_max_items_per_prompt` / `mm_max_items_per_batch`。

`get_modality_with_max_tokens()`（`:182`）返回 token 数最大的模态（用于无 `--mm-max-items-per-batch` 时的兜底）；`reset_cache()`（`:191`）清理临时 cache。

## 与其它模块/系统配合

- **registry.py**：`processor_only_cache_from_config` / `create_processor` / `get_dummy_mm_inputs` 都走 registry。
- **processing/context.py**：`BaseProcessingInfo.get_mm_max_tokens_per_item` 是模型可选实现的快速路径（默认 None 走 dummy 回退）；`supported_mm_limits` / `allowed_mm_limits` 来自 `BaseProcessingInfo`。
- **v1/core/encoder_cache_manager.py**：`compute_mm_encoder_budget` 定义在此（被本文件 import），`EncoderCacheManager` 用本文件输出的 `encoder_cache_size` 作 `cache_size`。
- **v1/engine/input_processor.py**：`mm_budget.encoder_cache_size` 被存为 `self.mm_encoder_cache_size`，用于 `_validate_model_input` 校验"单 item embedding 数 ≤ cache_size"，并在 prompt 总长校验时作 encoder 侧上限。
- **v1/core/sched/scheduler.py**：`mm_budget.encoder_compute_budget` 直接喂 `Scheduler.schedule` 的 `encoder_compute_budget` 变量；实时模型分支还断言 `len(mm_max_toks_per_item) <= 1`（实时调度假设单模态）。
- **v1/core/sched/interface.py** / **llm_engine.py** / **async_llm.py**：构造 `Scheduler`/`InputProcessor` 时注入 `mm_registry=MULTIMODAL_REGISTRY`，由它们内部建 `MultiModalBudget`。
- **配置**：`SchedulerConfig.max_num_encoder_input_tokens` / `encoder_cache_size` / `disable_chunked_mm_input` / `max_num_batched_tokens` / `enable_chunked_prefill` / `max_num_seqs` 来自 [`SchedulerConfig`](../10-config/multimodal-config.md)；`MultiModalConfig.enable_mm_embeds` / `limit_per_prompt` 来自 [`MultiModalConfig`](../10-config/multimodal-config.md)。

## 历史版本演进

- **v0.7（v1 化）**：`MultiModalBudget` 引入，仅支持 image；`get_mm_max_toks_per_item` 总走 dummy 回退，启动慢。
- **v0.8**：`get_mm_max_tokens_per_item` 快速路径加入，Qwen2.5-VL 等模型自报数字，启动从数十秒降到秒级。
- **v0.9（hash+cache）**：`processor_only_cache_from_config` 在 budget 构造期用一个临时 cache，事后 `reset_cache` 释放。
- **v0.10**：`disable_chunked_mm_input` 校验加入（防止用户关 chunked 时单 item 超 `max_num_batched_tokens` 直接崩）。
- **v0.11（EVS）**：`get_mm_max_toks_per_item` 用 `get_num_embeds()` 而非 `length`，让 EVS 剪枝后实际 token 数被正确反映（不影响预算但影响调度映射，详见 [evs.md](evs.md)）。
- **main**：`tower_modalities` / `embed_only_modalities` 显式分离；`enable_mm_embeds=True` 且某模态 `limit=0` 时仍纳入 `active_mm_max_toks_per_item` 算 encoder cache 大小，但不算 tower token 上限（embedding 旁路不挤 encoder 算力）。

[← 返回多模态首页](../README.md)

## 参见

- [v1-integration.md](v1-integration.md)：预算数字如何被 scheduler / input_processor 消费。
- [registry.md](registry.md)：`get_dummy_mm_inputs` / `create_processor` 来源。
- [processing.md](processing.md)：`get_mm_max_tokens_per_item` 快速路径的模型侧实现位置。
- [01-engine-core/kv-cache-management/encoder-cache.md](../01-engine-core/kv-cache-management/encoder-cache.md)：`EncoderCacheManager` 的驱逐细节。
