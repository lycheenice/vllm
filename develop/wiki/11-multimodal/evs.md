# evs.py · Elastic Vision Streaming

[← Wiki 首页](../README.md) > [多模态](../README.md) > evs

## 是什么

`vllm/multimodal/evs.py` 实现 **Elastic Vision Streaming（EVS）**：在视频 embedding 进入 LLM 之前，按帧间相似度剪枝冗余 token，动态缩减视频 token 数。提供三个纯函数：`compute_retained_tokens_count`（数/token 计数）、`compute_retention_mask`（生成保留掩码）、`compute_mrope_for_media`（算 mrope 位置）、`recompute_mrope_positions`（剪枝后重算 LLM mrope 位置）。EVS 由 NVIDIA 贡献（文件头版权声明），主要服务 Qwen2.5-VL / Qwen3-VL 类使用 mrope 的视频模型。

## 为什么

视频打 token 后总量极大（每帧数百 token × 数十帧），且大量相邻帧视觉上几乎相同——烹饪视频里锅不动镜头切菜的几秒、监控视频里静止背景段，这些帧的 embedding 高度相似，全量喂 LLM 既浪费显存又稀释注意力。EVS 用帧间余弦相似度排序：

- 第一帧全保留（保证有 anchor）。
- 其余帧按"相邻帧 dissimilarity"排序，取 top-k 保留，其余 token 被 `is_embed=False` mask 掉。
- 保留比例由 `q`（剪枝率）控制，`q=0` 不剪枝。

剪枝后 token 序列变短，但 mrope（多段旋转位置编码，Qwen-VL 用 `[t, h, w]` 三段）需要重算——原本位置基于"全 token 都在"假设，剪掉一些后剩余 token 的相对位置变了，且 LLM 输入序列里媒体 token 实际占的格数也变了。`recompute_mrope_positions` 就负责把新 mrope 同步到 `mrope_positions` 张量。

## 怎么做

### compute_retained_tokens_count

`compute_retained_tokens_count(tokens_per_frame, num_frames, q)`（`:16`）：

```
total_tokens = tokens_per_frame * num_frames
evs_num_tokens = int(total_tokens * (1 - q))
min_num_tokens = tokens_per_frame          # 第一帧全留
return max(min_num_tokens, evs_num_tokens)
```

保证至少留一帧，防止 `q` 过大时 retention 为 0。

### compute_retention_mask

`compute_retention_mask(video_embeds, video_size_thw, spatial_merge_size, q)`（`:38`）：

1. 把 `video_embeds`（shape `(T*H*W // spatial_merge_size^2, hidden)`）reshape 成 `(T, H//sm, W//sm, hidden)`。
2. 算相邻帧余弦相似度：`similarity = cosine_similarity(embeds[1:], embeds[:-1])`，`dissimilarity = 1 - similarity`。
3. 第一帧位置填 `255`（极大值）保证 argsort 后必然入选 top-k。
4. flatten，`order = argsort(dissimilarity, descending=True, stable=True)`。
5. `retain_num_tokens = compute_retained_tokens_count(...)`，`topk_indices = order[:retain_num_tokens]`。
6. `retention_mask = zeros_like(dissimilarity_flat, bool); index_fill_(0, topk_indices, True)`，reshape 回 `(T, H//sm, W//sm)` 后 view 成 1D 返回。

返回掩码用于构造 `PlaceholderRange.is_embed`（详见 [inputs.md](inputs.md)），让调度器与 encoder runner 知道哪些位置要取 embedding、哪些用文本 token embedding。

### compute_mrope_for_media

`compute_mrope_for_media(video_size_thw, spatial_merge_size, tokens_per_second=1.0, video_second_per_grid=1.0)`（`:95`）算 `[t_index, h_index, w_index, llm_grid_w]` 四通道位置（最后一段是 max_width 重复，Qwen2.5-VL 的实现约定）。媒体被假设位于序列 0 位置（绝对位置由 `recompute_mrope_positions` 平移）。

### recompute_mrope_positions

`recompute_mrope_positions(input_ids, multimodal_positions, mrope_positions, num_computed_tokens, vision_start_token_id, image_token_id, video_token_id)`（`:154`）是把剪枝后 mrope 同步到 LLM `mrope_positions` 的核心。每个 `mm_pos` 可能是：

- **4 通道 `(4, N)`**：Qwen2.5-VL，纯视频/图像 embedding 段，通道 `[t, h, w, max_width]`。
- **5 通道 `(5, N)`**：Qwen3-VL，video embedding 段中夹杂 timestamp / vision_start / vision_end 文本 token，通道 `[t, h, w, is_vision_start, is_vision]`。

算法主流程（每个 `mm_pos`）：

1. 用 `input_ids == vision_start_token_id` 找所有 vision_start 索引。
2. 区分"当前 prefill chunk 是否处于某媒体段中间"（chunked prefill 可能把一个媒体切成两半）。判定依据：`seen_mm_tokens`（已算的媒体 token）vs `seen_vision_start_indices` 中最后一个 vision_start 之前的媒体 token 数。
3. 计算 `global_mm_start`（媒体段在 LLM 序列的起点）与 `mm_embeddings_seen`（本 chunk 已投递的 embedding 数）。
4. 5 通道时调整 `global_mm_start` 以指向首个 timestamp token（Qwen3-VL 在 vision_start 之前还有 timestamp 文本 token）。
5. `base = positions[-1, global_mm_start - 1] + 1`，`local_start = global_mm_start (+1 + mm_embeddings_seen)`，`local_end = local_start + mm_pos.shape[1]`。
6. `positions[:, local_start:local_end] = mm_pos[0:3] + base` 重写媒体段的 mrope。
7. 算 `offset`（5 通道用 `mm_pos[0:3].max() + base + 1`；4 通道用 `mm_pos[3, 0] + base`），用 `text_mask[local_end:]` 的 cumsum 把后续文本 token 的 mrope 平移。
8. `num_computed_tokens += mm_pos.shape[1]` 进入下一段。

返回 `(positions, delta)`，`delta = positions.max() + 1 - N` 是序列总长的偏移。

### chunked prefill 兼容

`recompute_mrope_positions` 显式处理"同一媒体段被切到多个 prefill chunk"：

- 若本 chunk 的 `num_computed_tokens` 落在某 vision_start 之后但下一个 vision_start 之前，且已算媒体 token 数 > 上个 vision_start 前的媒体 token 数 → "in_the_middle_of_media"，从 `last_vision_start_token` 起算 `mm_embeddings_seen`。
- Qwen3-VL 还处理"已过 vision_start 但还没到首个 video embedding"（中间全是 timestamp 文本 token）的边界。

## 与其它模块/系统配合

- **v1/worker/gpu/mm**：剪枝在编码器塔输出后、注入 LLM 前进行；`compute_retention_mask` 的输出用于构造或更新 `PlaceholderRange.is_embed`，进而影响 `EncoderRunner.gather_mm_embeddings` 取哪些行（详见 [v1-integration.md](v1-integration.md)）。
- **inputs.py**：`PlaceholderRange.is_embed` / `get_embeds_indices_in_range` / `extract_embeds_range` 是 EVS mask 的下游消费者，调度器与双向注意力都据此工作。
- **v1/worker/gpu/rope.py**（`vllm/v1/worker/gpu/mm/rope.py`）：mrope 计算入口，把 `mm_features` 与 `prefill_token_ids` 喂给 `recompute_mrope_positions`。
- **v1/core/sched/scheduler.py**：`_schedule_encoder_inputs` 用 `curr_embeds_start/curr_embeds_end`（由 `is_embed` 推导）判断"窗口内是否有 embedding"决定是否调度编码器计算——剪枝掉的位置不占编码器算力。
- **v1/worker/gpu/attn_utils.py**：`compute_mm_prefix_ranges` 用 `extract_embeds_range()` 取连续 embedding 段构造 PrefixLM 双向注意力区间；EVS 把原本一整段视频变成多个稀疏小区间。
- **模型库**：Qwen2.5-VL / Qwen3-VL 在模型侧调用这些函数（具体调用点 `(待补充)`）。
- **配置**：`q`（剪枝率）来自 `(待核实)` 模型 config 或 `--mm-*` 参数。

## 历史版本演进

- **v0.11（EVS）**：本文件首次合入，NVIDIA 贡献。`compute_retention_mask` / `compute_mrope_for_media` / `recompute_mrope_positions` 三个函数立项。仅支持 Qwen2.5-VL（4 通道）。
- **main**：加入 Qwen3-VL 5 通道分支——`mm_pos` 形状扩展为 `(5, N)`，处理 timestamp token / vision_start / vision_end 文本夹杂场景；`recompute_mrope_positions` 加入 `adjusted_for_timestamps` 路径与"过 vision_start 但未到 video embedding"的中间态判定；早期分支处理无媒体的 delta 计算。

[← 返回多模态首页](../README.md)

## 参见

- [inputs.md](inputs.md)：`is_embed` / `extract_embeds_range` 是 EVS mask 的承载。
- [v1-integration.md](v1-integration.md)：调度器与编码器 runner 如何消费 EVS。
- [05-attention/README.md](../05-attention/README.md)：mrope 与双向注意力后端。
- [04-model-zoo/architecture-families/llava.md](../04-model-zoo/architecture-families/llava.md)：消费者模型家族（Qwen-VL）。
