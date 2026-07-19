# v1-integration.md · 多模态在 v1 引擎中的集成

[← Wiki 首页](../README.md) > [多模态](../README.md) > v1-integration

## 是什么

本页汇总 `vllm/multimodal/` 与 v1 引擎（`vllm/v1/`）的所有集成点，把它们串成一条从 API 请求到 GPU 编码器塔再到 LLM 输入嵌入的完整路径。涉及的 v1 模块：

- `vllm/v1/engine/__init__.py`：`EngineCoreRequest.mm_features: list[MultiModalFeatureSpec] | None` 字段定义。
- `vllm/v1/engine/input_processor.py`：`InputProcessor.process_inputs` 把 processor 输出的三 dict 拍平成 `mm_features`；`_validate_model_input` 用 `mm_encoder_cache_size` 校验单 item 上限。
- `vllm/v1/engine/tensor_ipc.py`：`TensorIpcSender`/`TensorIpcReceiver`，TP>1/PP>1 时多模态张量经 `torch.multiprocessing.Queue` 零拷贝。
- `vllm/v1/engine/core.py`：`EngineCore` 收到请求时调 `mm_receiver_cache.get_and_update_features` 还原缓存命中的 item。
- `vllm/v1/core/encoder_cache_manager.py`：`EncoderCacheManager`/`EncoderDecoderCacheManager` 做 GPU 编码器输出的准入/驱逐；`compute_mm_encoder_budget` 算预算。
- `vllm/v1/core/sched/scheduler.py`：`Scheduler._schedule_encoder_inputs` 决定每 step 排程哪些编码器输入；`SchedulerOutput.scheduled_encoder_inputs` 与 `free_encoder_mm_hashes` 是与 worker 的接口。
- `vllm/v1/core/sched/output.py`：`SchedulerOutput` 携带 `mm_features` 与 `free_encoder_mm_hashes`。
- `vllm/v1/worker/gpu/model_runner.py`：`EncoderRunner`/`EncoderCache` 装配、`free_encoder_cache`、编码器执行调度。
- `vllm/v1/worker/gpu/mm/encoder_cache.py`：worker 侧 GPU `EncoderCache`（`mm_features` per req + `encoder_outputs` per hash）。
- `vllm/v1/worker/gpu/mm/encoder_runner.py`：`prepare_mm_inputs`/`execute_mm_encoder`/`gather_mm_embeddings`/`get_inputs_embeds`。
- `vllm/v1/worker/gpu/attn_utils.py`：`compute_mm_prefix_ranges` 构造 PrefixLM 双向注意力区间。
- `vllm/v1/worker/encoder_cudagraph.py`：`EncoderCudaGraphManager`，捕获/重放编码器塔的 budget-batch CUDA graph。
- `vllm/v1/worker/gpu_model_runner.py`：`(待核实)` 旧路径别名，部分 cuda graph 装配在此。

## 为什么

v1 架构把"输入处理 / 调度 / 执行"分到不同进程与不同的 step：

- **API 前端进程**（P0）跑 HF Processor（CPU 重），需要缓存处理结果避免重复；
- **EngineCore 进程**（P1）做调度，需要在不接触 GPU 的情况下判断"本 step 能否排程这个编码器输入"；
- **Worker 进程**（P1，多 TP rank）跑 GPU 编码器塔与 LLM 前向，需要张量级缓存与 CUDA graph。

三段对 mm 数据的需求不同：前端要 per-item 可缓存、可 Null；调度器要 token 级预算与位置区间；worker 要 batched 张量与 GPU 显存。`MultiModalFeatureSpec` + `PlaceholderRange` + `EncoderCacheManager` 这套类型正好对齐三段需求，让 mm 数据能在三段间无损传递。

## 怎么做

### 1. 请求构造（InputProcessor）

`InputProcessor.__init__`（`input_processor.py:37`）：

- `supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)`。
- 若支持 MM：`mm_budget = MultiModalBudget(vllm_config, mm_registry)`，存 `mm_encoder_cache_size = mm_budget.encoder_cache_cache_size`，`skip_prompt_length_check = mm_budget.processor.info.skip_prompt_length_check`，`mm_budget.reset_cache()`（临时 cache 释放）。

`process_inputs`（`:242`）当 `decoder_inputs["type"] == "multimodal"`：

1. 取 `mm_kwargs` / `mm_placeholders` / `mm_hashes`（来自 `Renderer` 调用 processor `apply` 的输出）。
2. 校验 `mm_hashes` 全为 str（防自定义 processor 实现 bug）。
3. `sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)`（按 `offset` 排序）。
4. 对每个 `(modality, idx)` 构造 `MultiModalFeatureSpec(data=mm_kwargs[modality][idx], modality=modality, identifier=_get_mm_identifier(base_mm_hash, lora_request), mm_position=mm_positions[modality][idx], mm_hash=base_mm_hash)`。
5. 装入 `EngineCoreRequest.mm_features`。

`_get_mm_identifier`（`:165`）：`enable_tower_connector_lora` 时返 `f"{lora_name}:{mm_hash}"`，否则透传 mm_hash——让 receiver cache 用 mm_hash 跨 LoRA 共享，而 GPU encoder cache 用 identifier 区分 LoRA。

`inject_into_mm_cache`（`:183`）：当 mm_kwargs 已被前端外部预处理（如远程 renderer），通过 `renderer.mm_processor_cache.get_and_update_item((item, []), mm_hash)` 注入，让 cache 命中率统计准确。

`_validate_model_input`（`:434`）：对每个 `mm_position` 校验 `get_num_embeds() <= mm_encoder_cache_size`，超限 raise 提示 `--limit-mm-per-prompt`。

### 2. EngineCore 接收（core.py）

`EngineCore` 持 `mm_receiver_cache`（由 `registry.engine_receiver_cache_from_config` 构造，`lru` 模式才有）。

收到 `EngineCoreRequest` 时（`engine/core.py:864`）：

```python
if self.mm_receiver_cache is not None and request.mm_features:
    request.mm_features = self.mm_receiver_cache.get_and_update_features(request.mm_features)
```

`get_and_update_features`（`cache.py:589`）先 `touch` 所有 feature（防更新中被驱逐），再逐个 `get_and_update_item`：P1 命中即从 cache 返回 `MultiModalKwargsItem`，未命中（item not None）写入并返回。这样 P0 命中（item=None）的 feature 在 P1 也命中（cache 有），数据无需 IPC 完整传递。

`reset_encoder_cache`（`:687`）：调 `scheduler.reset_encoder_cache()` + `model_executor.reset_encoder_cache()`，权重更新后使 stale embedding 失效。

### 3. Tensor IPC（tensor_ipc.py）

`mm_tensor_ipc == "torch_shm"` 时（`v1/engine/utils.py:1102`），构造单 `TensorIpcQueue` 指向 rank 0。`TensorIpcSender`（`tensor_ipc.py:45`）把 msgpack 序列化中遇到的张量通过 `share_memory_()` 放共享内存，发 `(sender_id, message_id, tensor_id)` 句柄。`TensorIpcReceiver`（`:114`）用 drain-and-buffer 模式从队列拉张量，按句柄查 buffer，支持乱序到达与多 producer。

仅支持单 engine（`set_target_engine(0)`），DP>1 不支持（注释明确）。

### 4. 调度器排程（scheduler.py + encoder_cache_manager.py）

`Scheduler.__init__`（`scheduler.py:76`）：

- `mm_budget = MultiModalBudget(vllm_config, mm_registry)`（若支持 MM）。
- `encoder_compute_budget = mm_budget.encoder_compute_budget`。
- `encoder_cache_size = mm_budget.encoder_cache_size`。
- `encoder_cache_manager = EncoderDecoderCacheManager(cache_size)` 若 enc-dec，否则 `EncoderCacheManager(cache_size)`。

#### EncoderCacheManager（encoder_cache_manager.py）

`EncoderCacheManager`（`:17`）按"mm_hash → 引用它的 request_id 集合"管理：

- `cached: dict[str, set[str]]`：mm_hash → 引用 req ids。
- `request_cached_ids: dict[str, set[int]]`：req_id → 已缓存的 input_id 集合。
- `freeable: OrderedDict[mm_hash, int]`：引用集为空但仍在 GPU 的项（可驱逐），值是 num_encoder_embeds。
- `freed: list[str]`：实际被驱逐的 mm_hash，等 worker 来取。

关键方法：

- `check_and_update_cache(request, input_id)`（`:94`）：cache 有→加 req 引用，返 True（跳过重复 encode）。
- `can_allocate(request, input_id, encoder_compute_budget, num_embeds_to_schedule)`（`:123`）：算 `num_embeds = request.get_num_encoder_embeds(input_id) + num_embeds_to_schedule`，超 compute budget 返 False；先吃 `num_free_slots`，不够再吃 `num_freeable_slots`（驱逐 oldest），仍不够返 False。驱逐时把 mm_hash 加 `freed`。
- `allocate(request, input_id)`（`:184`）：扣 `num_free_slots`/`num_freeable_slots`，加 req 引用。
- `free_encoder_input(request, input_id)`（`:216`）：移 req 引用，引用集空→加入 `freeable`。
- `free(request)`（`:243`）：对所有 cached input_id 调 `free_encoder_input`（请求结束/abort 时）。
- `get_freed_mm_hashes()`（`:255`）：返回并清 `freed`，由 scheduler 写入 `SchedulerOutput.free_encoder_mm_hashes`，worker 据此 `encoder_cache.free_encoder_cache(mm_hash)`。

`compute_mm_encoder_budget(scheduler_config, mm_max_toks_per_item)`（`:269`）：

- `max_tokens_per_mm_item = max(values)`。
- `disable_chunked_mm_input` 且 `max_tokens_per_mm_item > max_num_batched_tokens` → raise。
- `encoder_compute_budget = max(max_num_encoder_input_tokens, max_tokens_per_mm_item)`。
- `encoder_cache_size = max(encoder_cache_size, max_tokens_per_mm_item)`。

`EncoderDecoderCacheManager`（`:323`）：enc-dec 临时实现，不享 cache，`to_free`/`allocated` 双 buffer 模拟状态机。

#### Scheduler._schedule_encoder_inputs（scheduler.py:1346）

`get_mm_features_in_window(mm_features, start=num_computed_tokens, end=num_computed_tokens+num_new_tokens+shift)` 取窗口内 feature，逐个：

1. enc-dec 且 `num_computed_tokens > 0` → 跳过（encoder 输入已在首 step 处理）。
2. 非 enc-dec：若 `identifier` 已在本 step 排程→跳过；若 `check_and_update_cache` 命中→跳过。
3. `disable_chunked_mm_input` 且本 step 跨不到整段 mm input → 回滚 `num_new_tokens` 到 `start_pos` 前。
4. `can_allocate(...)` 不够 → 截断 `num_new_tokens` 至 `start_pos` 前（或 0）并 break。
5. 算窗口内 embedding 数 `curr_embeds_start/curr_embeds_end = mm_position.get_embeds_indices_in_range(start_idx_rel, end_idx_rel)`，0 则跳过。
6. 若 `ec_connector.has_cache_item(identifier)` → external_load 路径；否则扣 `encoder_compute_budget`，加入 `encoder_inputs_to_schedule`。

返回 `(encoder_inputs_to_schedule, num_new_tokens, encoder_compute_budget, external_load_encoder_input)`。

`SchedulerOutput`（`sched/output.py`）：每个 `ScheduledRequestFull` 携带 `mm_features=request.mm_features`；顶层 `free_encoder_mm_hashes`。

### 5. GPU ModelRunner（v1/worker/gpu/model_runner.py）

`GPUModelRunner.__init__`（`model_runner.py:180`）：

- `mm_registry = MULTIMODAL_REGISTRY`。
- `supports_mm_inputs = mm_registry.supports_multimodal_inputs(...)`。
- `encoder_cache = EncoderCache()` 若支持 MM 且 `is_first_pp_rank`。
- 通过 `ModelStateFactory` 注入 `encoder_cache` 到 `ModelState`。

`update_states`（`:738`/`:754`）：对每个 finished/removed 请求 `encoder_cache.remove_request(req_id)`；对 `scheduler_output.free_encoder_mm_hashes` 每个 mm_hash `encoder_cache.free_encoder_cache`；对 `ScheduledRequestFull.mm_features` 新增的 `encoder_cache.add_request(req_id, mm_features)`。

`reset_encoder_cache`（`:672`）：调 `encoder_cache.reset_encoder_cache()`（权重更新后）。

执行前向（`:1231`）若 `supports_mm_inputs and is_first_pp_rank`：

1. `set_active_mm_loras(...)`（启用 LoRA tower connector）。
2. `inputs_embeds = model_state.get_mm_embeddings(scheduled_encoder_inputs, input_batch, req_states)`——内部走 `EncoderRunner`。

### 6. EncoderRunner（v1/worker/gpu/mm/encoder_runner.py）

`EncoderRunner` 持 `model`/`max_num_tokens`/`hidden_size`/`encoder_cache`/预分配 `inputs_embeds`。

`prepare_mm_inputs(scheduled_encoder_inputs)`（`:35`）：按 req_id + input_id 取 `mm_feature`，`data is None` 跳过（无 encoder 数据），收集 `(mm_hash, (modality, mm_feature.data))`。

`execute_mm_encoder(mm_kwargs)`（`:52`）：`group_and_batch_mm_kwargs` 按 modality 合批，调 `model.embed_multimodal(**batch)` 跑编码器塔，`sanity_check_mm_encoder_outputs` 校验 item 数，把输出 extend 到 list。

`gather_mm_embeddings(req_ids, total_num_scheduled_tokens, num_scheduled_tokens, query_start_loc, prefill_lens, num_computed_tokens, draft_lookahead=0)`（`:64`）：

1. 非 realtime 模型：全 decode step 直接返空（媒体 embedding 只在 prompt 内）。
2. 对每个 req，用 `get_mm_features_in_window(mm_features, query_start, query_end)` 找相交 feature。
3. 对每个 feature：`start_idx/end_idx` 算与 query 窗口的交集，`get_embeds_indices_in_range` 映射到 encoder output 索引；`encoder_cache.encoder_outputs[mm_hash]` 取输出切片，append 到 `mm_embeds`；同时填 `is_mm_embed` 布尔掩码供 `embed_prompt_ids` 知道哪些位置用媒体 embedding、哪些用 token embedding。
4. 投机解码 `draft_lookahead`：若 feature 起点过远（超前看），允许 cache miss 跳过（用 token embedding 兜底）。

`get_inputs_embeds(input_ids, mm_embeds, is_mm_embed)`（`:147`）：调 `model.embed_prompt_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)`，把结果拷到预分配 `inputs_embeds` buffer（cudagraph 兼容）。

### 7. GPU EncoderCache（v1/worker/gpu/mm/encoder_cache.py）

`EncoderCache`（43 行）极简：

- `mm_features: dict[str, list[MultiModalFeatureSpec]]`：per req 的 feature 列表（`add_request`/`remove_request`）。
- `encoder_outputs: dict[str, torch.Tensor]`：mm_hash → encoder 输出（`free_encoder_cache` 弹出）。
- `reset_encoder_cache`：清 `encoder_outputs`（权重更新后）。
- `reset_mm_cache`：profiling 后清 mm_features 占位（TODO 未实现）。

### 8. 双向注意力区间（attn_utils.py）

`compute_mm_prefix_ranges(req_ids, mm_features, sliding_window=None)`（`attn_utils.py:640`）：

- 对每个 req 的每个 `mm_feature`：若 modality 是 `image`/`video`，调 `mm_position.extract_embeds_range()` 取连续 embedding 段 `(start, end)`（绝对坐标）。
- `sliding_window` 不为 None 时跳过超长段（防早期 token 跨整图 attend）。
- 返回 `dict[req_idx, list[(start, end)]]`，作为 `CommonAttentionMetadata.mm_req_doc_ranges` 传给注意力后端，使其在计算视觉 token 间相互 attention 时用 PrefixLM 双向语义。

### 9. Encoder CUDA Graph（encoder_cudagraph.py）

`EncoderCudaGraphManager`（`encoder_cudagraph.py:53`，`SupportsEncoderCudaGraph`）：

- 从 `model.get_encoder_cudagraph_config()` 拿 `EncoderCudaGraphConfig`（含 `padding_logics` 等）。
- 从 `compilation_config.encoder_cudagraph_token_budgets`/`encoder_cudagraph_max_vision_items_per_batch`/`encoder_cudagraph_max_frames_per_batch` 拿用户预算，与 `model.get_encoder_cudagraph_budget_range()` 算出的 min/max 交叉。
- 每个 token budget 捕获一个 `torch.cuda.CUDAGraph`，存 `BudgetGraphMetadata(token_budget, max_batch_size, max_frames_per_batch, graph, input_buffers, output_buffer)`。
- 执行时（`execute`）：`select_encoder_cudagraph_items` 按 TP rank 切分 batch，把 mm_kwargs copy 到 `input_buffers`（按 `padding_logics` 填充），`graph.replay()`，从 `output_buffer` 取结果。
- 被 `GPUModelRunner._create_encoder_cudagraph_manager`（`gpu/model_runner.py:6462` 或 `gpu_model_runner.py:6462`）按 `compilation_config.cudagraph_mm_encoder and supports_mm_inputs` 装配；capture 在 `configure_cudagraphs` 阶段（`:6599`），wake_up/重 capture 在 `_maybe_init_encoder_cudagraph_manager`（`:6491`）。
- TP>1 时 `tensor_model_parallel_all_gather` 汇总各 rank 输出。

## 与其它模块/系统配合

- **InputProcessor**（[`01-engine-core/input-processor.md`](../01-engine-core/input-processor.md)）：mm 数据的 v1 入口。
- **EncoderCacheManager**（[`01-engine-core/kv-cache-management/encoder-cache.md`](../01-engine-core/kv-cache-management/encoder-cache.md)）：调度器侧的状态机。
- **model_runner**（[`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md)）：worker 侧装配与执行。
- **注意力**（[`05-attention/README.md`](../05-attention/README.md)）：`mm_req_doc_ranges` 与 mrope。
- **Scheduler**（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）：`_schedule_encoder_inputs`。
- **MultiModalBudget**（[encoder-budget.md](encoder-budget.md)）：预算数字的来源。
- **cache.py**（[cache.md](cache.md)）：P0/P1 processor cache（不同于 P1 worker 的 GPU encoder cache）。
- **inputs.py**（[inputs.md](inputs.md)）：`MultiModalFeatureSpec`/`PlaceholderRange` 是全程类型。
- **evs.py**（[evs.md](evs.md)）：剪枝影响 `is_embed`，进而影响 `get_embeds_indices_in_range`。
- **配置**：[`MultiModalConfig`](../10-config/multimodal-config.md)（`mm_tensor_ipc`、`mm_processor_cache_*`、`mm_ipc_gpu_memory_gb`、`enable_mm_embeds`）、`SchedulerConfig`（`max_num_encoder_input_tokens`、`encoder_cache_size`、`disable_chunked_mm_input`、`enable_chunked_prefill`）、`CompilationConfig`（`cudagraph_mm_encoder`、`encoder_cudagraph_*`）。

## 历史版本演进

- **v0.7（v1 化）**：v1 引擎引入，`EngineCoreRequest.mm_features` + `MultiModalFeatureSpec` 落地；`EncoderCacheManager` 出现（仅 image）；`Scheduler._schedule_encoder_inputs` 初版。
- **v0.8**：chunked mm input 支持（`disable_chunked_mm_input` 配置）；`EncoderRunner.gather_mm_embeddings` 支持窗口内部分 embedding 取数；`attn_utils.compute_mm_prefix_ranges` 加入支撑 PrefixLM VLM。
- **v0.9（hash+cache）**：P0/P1 双侧 processor cache 接入 `EngineCore`，`mm_receiver_cache.get_and_update_features` 在请求到达时还原 item；`_get_mm_identifier` 区分 `mm_hash` 与 `identifier`（LoRA tower connector）。
- **v0.10**：`EncoderCudaGraphManager` 与 `SupportsEncoderCudaGraph` 接口加入，配合 `compilation_config.cudagraph_mm_encoder`；`ShmObjectStoreReceiverCache` 让多 TP worker 共享 SHM 地址；`tensor_ipc.py`（`torch_shm` 模式）加入支撑 PP>1 张量传递。
- **v0.11（EVS）**：`PlaceholderRange.is_embed` 在 `gather_mm_embeddings` 与 `_schedule_encoder_inputs` 全面铺开，让 EVS 剪枝后的稀疏 mask 被调度器/runner 正确处理；`compute_mm_prefix_ranges` 用 `extract_embeds_range` 取稀疏段；`EncoderCacheManager.get_and_update_features` 改用 `mm_hash or identifier` 跨 LoRA 共享。
- **main**：`MultiModalBudget` 区分 `tower_modalities`/`embed_only_modalities`，`enable_mm_embeds=True` 时 embedding-only 模态进 `active_mm_max_toks_per_item` 算 cache size 但不占 tower budget；`support_realtime`（`supports_realtime(model)`）让 realtime 模型在 decode step 也取媒体 embedding；投机解码 `draft_lookahead` 处理 drafter +1 超前看的 cache miss 容错（`encoder_runner.py:129`）；`EncoderDecoderCacheManager` 的 `to_free`/`allocated` 双 buffer 稳定（注释 NickLucche，待与 `EncoderCacheManager` 合并）。

## 待核实内容

- `vllm/v1/worker/gpu_model_runner.py` 与 `vllm/v1/worker/gpu/model_runner.py` 的关系：前者疑似旧路径别名/再导出（grep 显示 `encoder_cudagraph_manager` 装配代码在 `gpu_model_runner.py`，mm 主代码在 `gpu/model_runner.py`）。`（待核实）` 二者是否为同文件不同名，或 `gpu_model_runner.py` 是 transition shim。
- `gpu_ipc_memory.py` 的 `MultiModalGPUMemoryPool.acquire` 在 `PyNvVideoCodecVideoBackendMixin` 内的具体调用位置 `（待补充）`。
- `compute_mm_prefix_ranges` 的 `sliding_window` 参数实际取值来自 `（待核实）`（应在 `attn_metadata_builder` 调用处）。
- `EncoderDecoderCacheManager` 注释明确"临时实现，待与 `EncoderCacheManager` 合并" `（待核实）` 合并进度。
- 实时模型（`supports_realtime`）的具体判定与调用点 `（待补充）`。

[← 返回多模态首页](../README.md)

## 参见

- [01-engine-core/input-processor.md](../01-engine-core/input-processor.md)
- [01-engine-core/kv-cache-management/encoder-cache.md](../01-engine-core/kv-cache-management/encoder-cache.md)
- [01-engine-core/scheduler/scheduler.md](../01-engine-core/scheduler/scheduler.md)
- [02-execution/worker/gpu-model-runner.md](../02-execution/worker/gpu-model-runner.md)
- [05-attention/README.md](../05-attention/README.md)
- [04-model-zoo/architecture-families/llava.md](../04-model-zoo/architecture-families/llava.md)
- [10-config/multimodal-config.md](../10-config/multimodal-config.md)
- [encoder-budget.md](encoder-budget.md)
- [cache.md](cache.md)
- [inputs.md](inputs.md)
- [evs.md](evs.md)
