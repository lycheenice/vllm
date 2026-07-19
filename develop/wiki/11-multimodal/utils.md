# utils.py · 多模态工具函数集

[← Wiki 首页](../README.md) > [多模态](../README.md) > utils

## 是什么

`vllm/multimodal/utils.py` 收纳跨模块复用的工具函数与对外 fetch API：

- 媒体编码 helper：`encode_audio_base64` / `encode_audio_url` / `encode_image_base64` / `encode_image_url` / `encode_video_base64` / `encode_video_url`。
- 调度映射：`get_mm_features_in_window`、`argsort_mm_positions`。
- 合批 helper：`group_and_batch_mm_items` / `group_and_batch_mm_kwargs`（老名 `group_mm_kwargs_by_modality` 已 deprecated，v0.19 移除）。
- 用户级 fetch API：`fetch_audio` / `fetch_image` / `fetch_video`。

这些函数被 `inputs.py` / `v1/worker/gpu/mm/encoder_runner.py` / `v1/core/sched/scheduler.py` / `v1/engine/input_processor.py` 等多处调用，是 subprocess 内的"通用粘合层"。

## 为什么

多模态子系统的数据结构设计偏 per-item / per-modality，但下游消费方有三个反复出现的需求：

1. **调度窗口映射**：调度器在某 step 处理 `[num_computed_tokens, num_computed_tokens + num_new_tokens)` 窗口的 token，需要知道窗口里有哪些 `mm_feature`。`mm_features` 按 `offset` 排序且不重叠，`get_mm_features_in_window` 用 `bisect` 二分查找，避免每次 step 线性扫描。
2. **跨请求合批**：多个请求的 image item 形状一致时，`torch.stack` 一次喂编码器比逐张跑高效；但跨 modality 或跨"shared field 值"的 item 不能合并。`group_and_batch_mm_items` 按 `(key, group_hash)` groupby 连续元素，每组调 `_batch_mm_items.reduce_data`。
3. **用户测试 / 离线 fetch**：`fetch_image/fetch_video/fetch_audio` 让用户代码（非在线 server）能从 URL 拿媒体对象；在线 server 通过 `Renderer` 路径走 `MediaConnector`，不走这些 helper（避免绕过校验）。

`encode_*` 系列则服务于需要把媒体回写成 base64 / data URL 的场景（如返回给客户端、缓存到 DB）。

## 怎么做

### 媒体编码 helper

`encode_audio_base64(audio, sampling_rate, *, format="WAV")`（`:35`）：构造 `AudioMediaIO()`，调 `encode_base64((audio, sr), audio_format=format)`。
`encode_audio_url`（`:46`）：base64 + mimetype → `data:{mimetype};base64,{b64}`。
`encode_image_base64(image, *, image_mode="RGB", format="PNG")`（`:58`）：`ImageMediaIO(image_mode).encode_base64(image, image_format=format)`。
`encode_image_url`（`:73`）：同理拼 data URL。
`encode_video_base64(frames, *, format="JPEG")`（`:89`）：`VideoMediaIO(ImageMediaIO()).encode_base64(frames, video_format=format)`。
`encode_video_url`（`:99`）：JPEG 用 `video/jpeg` mimetype（mimetypes 无 jpeg），其它走 `mimetypes.types_map`。

### get_mm_features_in_window

`get_mm_features_in_window(mm_features, start, end) -> (lo, hi)`（`:114`）：

- 假设 `mm_features` 已按 `offset` 排序且不重叠（`offset + length` 也因此排序）。
- `lo = bisect_left(mm_features, start + 1, key=lambda f: f.mm_position.offset + f.mm_position.length)`——首个"end > start"的 feature。
- `hi = bisect_left(mm_features, end, key=lambda f: f.mm_position.offset)`——首个"offset >= end"的 feature。
- 返回 `[lo, hi)` 半开区间，覆盖所有与 `[start, end)` 相交的 feature。

被 `Scheduler._schedule_encoder_inputs` 与 `EncoderRunner.gather_mm_embeddings` 共用。

### argsort_mm_positions

`argsort_mm_positions(mm_positions) -> list[(modality, idx)]`（`:137`）：

1. 拍平成 `(modality, idx, item)` 序列。
2. 按 `item.offset` 升序 sort。
3. 返回 `[(modality, idx), ...]`。

被 `InputProcessor.process_inputs`（`v1/engine/input_processor.py:352`）用来从 `mm_kwargs/mm_placeholders/mm_hashes` 三 dict 拍平成按位置排序的 `MultiModalFeatureSpec` 列表。

### 合批 helper

`_get_group_hash(elem)`（`:160`）：仅 `MultiModalSharedField` 返回 `MultiModalHasher.hash_kwargs(data=elem.data)`；其它 field 返回 None（可自然合并）。这样 shared field 值不同（如不同 `image_grid_thw`）的 item 不会错合并。

`_batch_mm_items(items, *, device, pin_memory)`（`:167`）：按 key 聚类 `MultiModalFieldElem`，对每组调 `elems[0].field.reduce_data(elems, device=device, pin_memory=pin_memory)`。

`group_and_batch_mm_items(items, *, device, pin_memory)`（`:188`），generator：

1. 算每个 item 的 `group_id = tuple((key, _get_group_hash(elem)) for key, elem in sorted(item.items()))`。
2. `groupby(group_ids)` 得连续相同组的大小。
3. 对每组切片调 `_batch_mm_items`，`yield (num_items, batched_kwargs)`。

`group_and_batch_mm_kwargs(mm_kwargs, *, device, pin_memory)`（`:236`）：在 `group_and_batch_mm_items` 上加 modality 维度——`groupby(mm_kwargs, key=lambda x: x[0])` 按 modality 分，再每组调 `group_and_batch_mm_items`，`yield (modality, num_items, batched_kwargs)`。被 `EncoderRunner.execute_mm_encoder` 用于按 modality 喂 `model.embed_multimodal`。

`group_mm_kwargs_by_modality`（`:279`，`@deprecated`）：旧名，v0.19 移除。

### 用户级 fetch API

`fetch_audio(audio_url, audio_io_kwargs=None)` / `fetch_image(image_url, image_io_kwargs=None)` / `fetch_video(video_url, video_io_kwargs=None)`（`:288`/`:309`/`:330`）：

- 构造 `MediaConnector(media_io_kwargs=..., allowed_local_media_path="/")`（注意：传 `/` 意味着本地任意路径可读，**仅用户代码用**）。
- 调对应 `fetch_*` 同步方法。
- docstring 显式 warning："direct access to local files, only intended for user code. Never call from online server!"。

在线 server 的 fetch 走 `Renderer` 注入的受限 `MediaConnector`（`allowed_local_media_path` 与 `allowed_media_domains` 校验）。

## 与其它模块/系统配合

- **inputs.py**：`MultiModalKwargsItems.get_data` 调 `group_and_batch_mm_items` 完成最后合批。
- **media.py**：`encode_*` helper 与 `fetch_*` 都基于 `MediaIO` / `MediaConnector`，是 [media.md](media.md) 的对外门面。
- **v1/engine/input_processor.py**：`argsort_mm_positions` 是请求进入引擎时 mm 字段拍平的入口（详见 [v1-integration.md](v1-integration.md)）。
- **v1/core/sched/scheduler.py**：`get_mm_features_in_window` 在 `_schedule_encoder_inputs` 决定窗口内要排程哪些编码器输入。
- **v1/worker/gpu/mm/encoder_runner.py**：`get_mm_features_in_window`（gather embeddings 时）+ `group_and_batch_mm_kwargs`（execute encoder 时）。
- **hasher.py**：`_get_group_hash` 用 `MultiModalHasher.hash_kwargs` 算 shared field 的 group id。
- **13-entrypoints**：API server 的 `Renderer` 不直接调 `fetch_*`，而是注入受控 `MediaConnector`，本文件的 `fetch_*` 仅服务于 `LLM` 离线 API 与文档示例（[13-entrypoints](../13-entrypoints/README.md)）。

## 历史版本演进

- **v0.5（LLaVA 初版）**：`encode_image_base64` / `fetch_image` 加入，仅 image。
- **v0.6**：`encode_audio_*` / `encode_video_*` 与对应 `fetch_*` 加入；`group_and_batch_mm_kwargs` 前身（按 modality 合批）。
- **v0.7（v1 化）**：`argsort_mm_positions` 引入支撑 v1 请求拍平；`get_mm_features_in_window` 加入配合 chunked prefill 窗口调度。
- **v0.8**：`MultiModalSharedField` 的 group hash 加入，防止 shared field 不同值错合并。
- **v0.9（hash+cache）**：`group_mm_kwargs_by_modality` 重命名为 `group_and_batch_mm_kwargs`，旧名 `@deprecated` 标 v0.19 移除。
- **v0.10**：`fetch_*` 系列文档加强安全 warning，禁止在线 server 使用。
- **v0.11（EVS）**：`get_mm_features_in_window` 的 bisect 实现优化（之前是线性扫描），适应 EVS 下更频繁的窗口查询。
- **main**：`encode_video_url` 显式处理 JPEG mimetype（`mimetypes` 不认识 `video/jpeg`）；`group_and_batch_mm_items` 的 group_id 用 sorted dict items 保证 key 顺序稳定。

[← 返回多模态首页](../README.md)

## 参见

- [inputs.md](inputs.md)：`get_data` 调 `group_and_batch_mm_items`。
- [media.md](media.md)：`encode_*` 与 `fetch_*` 的底层 `MediaIO` / `MediaConnector`。
- [v1-integration.md](v1-integration.md)：调度器与编码器 runner 的调用点。
