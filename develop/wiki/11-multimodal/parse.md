# parse.py · 多模态数据解析器

[← Wiki 首页](../README.md) > [多模态](../README.md) > parse

## 是什么

`vllm/multimodal/parse.py` 把用户传入的 `MultiModalDataDict`（来自 API `multi_modal_data` 字段）规范化为 `MultiModalDataItems`，后者是 `{modality: ModalityDataItems}` 的 `UserDict`。每个 modality 对应一个 `ModalityDataItems` 子类：原始媒体走 `ProcessorBatchItems`（图片/视频/音频）或 `VisionChunkProcessorItems`（vision_chunk）；预计算 embedding 走 `EmbeddingItems`（image/audio/video embeds）或 `DictEmbeddingItems`（HF processor 输出风格的 dict）。解析器 `MultiModalDataParser` 是一个有状态对象，持有 `AudioResampler`、`target_channels`、`video_needs_metadata`、`expected_hidden_size` 等参数，dispatch 到四个子解析器。

## 为什么

用户的 `multi_modal_data` 形态极不统一：

- **图片**：可能是 `PIL.Image`、`np.ndarray(H,W,C)`、`np.ndarray(C,H,W)`、`torch.Tensor`、`MediaWithBytes[Image]`、单个或 list、或 3D embedding / list of 2D embedding。
- **音频**：可能是 `np.ndarray`、`list[float]`、`torch.Tensor`、`(ndarray, sample_rate)` 元组（需重采样）、3D embedding。
- **视频**：除了多种容器，还可能带 `metadata` dict（`total_num_frames` / `fps` / `duration` / `frames_indices`），部分模型（如 Qwen2-VL）要求 metadata。
- **vision_chunk**：统一图像与视频块（多模态流式），不支持 embedding。

HF Processor 不能直接消化这些异质输入；且 vLLM 需要在解析阶段就完成"是否 embedding"判定（决定走旁路），以及把 EXIF 方向、音频重采样、声道归一化等"与模型无关的清洗"提前一次性做完。`parse.py` 就是这层清洗 + 类型归一化。

## 怎么做

### ModalityDataItems 抽象

`ModalityDataItems`（`parse.py:49`）是 `ABC, Generic[_T, _I]`，要求实现 `get_count` / `get(index)` / `get_processor_data` / `get_passthrough_data`。区别：

- `get_processor_data()`：传给 HF Processor 的字典（如 `{"images": [...]}`）。
- `get_passthrough_data()`：直接传给模型的字典（如 `{"image_embeds": tensor}`），不进 HF。

`get_item_for_hash` / `get_all_items_for_hash`（`:88`）返回用于哈希的对象——对 `ProcessorBatchItems` 返回 raw item（保留 `MediaWithBytes` 的 `original_bytes`），对其它返回 `get(index)`。

### ProcessorBatchItems

`ProcessorBatchItems`（`:105`）是"list-like"基类。`_unwrap` 把 `MediaWithBytes` 解包为底层 media。`get_processor_data` 返回 `{f"{modality}s": self.get_all()}`（注意复数 `s`）。`get_passthrough_data` 返回空 dict。

### EmbeddingItems 与校验

`EmbeddingItems`（`:153`）持 `torch.Tensor | list[torch.Tensor]`。`_validate_ndim`（`:176`）要求 batched tensor 3D、list 元素 2D；`_validate_hidden_size`（`:190`）在 `expected_hidden_size` 给出时校验最后一维，防止"ndim 正确但 hidden 错"导致推理期崩。`get_passthrough_data` 返回 `{f"{modality}_embeds": data}`。`get_feature_size(item_idx)` 返回该 item 的 seq_len。

`DictEmbeddingItems`（`:240`）接收已经是 HF 输出风格的 dict（如 `{"pixel_values":..., "image_grid_thw":...}`），用 `MultiModalKwargsItems.from_hf_inputs` 拆成 per-item，要求声明 `required_fields` 与 `fields_factory`。

### 具体 Items

| 类 | modality | 数据形态 | 备注 |
|---|---|---|---|
| `AudioProcessorItems`（`:300`） | `audio` | `Sequence[HfAudioItem \| None]` | `get_audio_length` 取 `len(audio)` |
| `AudioEmbeddingItems`（`:312`） | `audio` | `Tensor \| list[Tensor]` | 继承 `EmbeddingItems` |
| `ImageProcessorItems`（`:326`） | `image` | `Sequence[HfImageItem \| None]` | `get_image_size` 兼容 PIL / HWC / CHW |
| `ImageEmbeddingItems`（`:350`） | `image` | `Tensor \| list[Tensor]` | |
| `VideoProcessorItems`（`:359`） | `video` | `Sequence[HfVideoItem \| None]` + `metadata` | `get_num_frames` / `get_frame_size` |
| `VideoEmbeddingItems`（`:401`） | `video` | `Tensor \| list[Tensor]` | |
| `VisionChunkProcessorItems`（`:410`） | `vision_chunk` | `Sequence[Any]` | 统一图/视频块 |

`ImageProcessorItems.get_image_size`（`:330`）同时支持 PIL.Image（`image.size`）与 ndarray/Tensor，按 `shape[-1] in (1,3,4)` 判 HWC 还是 CHW。`VideoProcessorItems.get_frame_size` 取 `video[0]` 做同样判定。

### MultiModalDataItems 容器

`MultiModalDataItems`（`:420`）是 `UserDict[str, ModalityDataItems]`。方法：

- `select(modalities)`（`:426`）：子集过滤。
- `get_count(modality, *, strict=True)`（`:435`）：strict=False 时缺失返 0。
- `get_all_counts()`（`:454`）：每个 modality 的 item 数。
- `get_items(modality, typ)`（`:458`）：带类型校验的取值，类型不符 raise `TypeError`。

### MultiModalDataParser

`MultiModalDataParser`（`:490`）构造参数：`target_sr`（音频目标采样率）、`target_channels`（声道归一化）、`audio_resample_method`（`pyav`/`scipy`/`soxr`）、`video_needs_metadata`、`expected_hidden_size`。

`is_embeddings`（class method，`:527`）做 TypeGuard：单 Tensor 3D、或非空 list of 2D Tensor，判为 embedding。

四个 `_parse_*_data`：

1. `_parse_audio_data`（`:567`）：embedding 优先；否则按"1D ndarray/list/tuple → 单 item；2D ndarray → 按行拆"归一化；每个 item 经 `_get_audio_with_sr` 分离 (audio, sr)，需重采样的走 `AudioResampler.resample`，再 `normalize_audio(spec)` 做声道归一化，产出 `AudioProcessorItems`。
2. `_parse_image_data`（`:606`）：embedding 优先；否则按"PIL/MediaWithBytes/3D ndarray → 单 item；高维 ndarray → 按行拆"归一化；每个 PIL item 做 `normalize_image` + `convert_image_mode(item, "RGB")`，产出 `ImageProcessorItems`。
3. `_parse_video_data`（`:634`）：embedding 优先；否则按"list of PIL / 4D ndarray / 2-tuple → 单 item；高维 ndarray → 按行拆"归一化；`_get_video_with_metadata` 分离 (video, metadata)；若 `video_needs_metadata` 且缺失则 raise；产出 `VideoProcessorItems(new_videos, metadata=metadata_lst)`。
4. `_parse_vision_chunk_data`（`:676`）：embedding 不支持（raise）；dict 包成 list，产出 `VisionChunkProcessorItems`。

`_get_subparsers`（`:692`）返回四者字典；`parse_mm_data(mm_data)`（`:700`）遍历 `mm_data`，对每个 modality 调对应子解析器，跳过空 embedding（`None` 返回），未支持的 modality raise。

### UUID 解析

`parse_mm_uuids`（`:722`）把 `MultiModalUUIDDict` 归一化成 `{modality: list[str|None]}`，供 `ProcessorInputs.get_mm_hashes` 在 hash 计算时优先采用用户指定 UUID。

## 与其它模块/系统配合

- **audio.py / image.py**：解析器调用 `AudioResampler`、`normalize_audio`、`normalize_image`、`convert_image_mode`，详见 [audio.md](audio.md) 与 [image.md](image.md)。
- **media.py**：`MediaWithBytes` 包装来自 `ImageMediaIO.load_bytes`，解析器 `_unwrap` 在 `get` 时拆包，但 `get_item_for_hash` 保留包装以让 hasher 用 `original_bytes`（详见 [hasher.md](hasher.md)）。
- **processing/context.py**：`BaseProcessingInfo.get_data_parser`（`context.py:353`）构造 parser；`expected_hidden_size` 由 `_get_expected_hidden_size` 基于 `enable_mm_embeds` 与 `get_inputs_embeds_size` 决定。
- **processing/inputs.py**：`ProcessorInputs.get_mm_hashes` 用 `get_all_items_for_hash` 取哈希源对象。
- **inputs.py**：`DictEmbeddingItems` 直接产出 `MultiModalKwargsItems`，复用 `from_hf_inputs`。
- **配置**：`target_sr` / `target_channels` / `audio_resample_method` / `video_needs_metadata` 来自 `BaseProcessingInfo` 子类的 `get_data_parser` 实现，最终源自模型/HF config（见 [processing.md](processing.md)）。

## 历史版本演进

- **v0.5（LLaVA 初版）**：仅支持 image，`MultiModalDataDict` 直接进 `ImageProcessor`；无统一解析器。
- **v0.6**：引入 `MultiModalDataParser` 与 `ModalityDataItems`，支持 audio/video；`get_processor_data` / `get_passthrough_data` 分离。
- **v0.7（v1 化）**：`EmbeddingItems` 抽出，支持 `image_embeds` / `audio_embeds` / `video_embeds` 旁路；`VisionChunkProcessorItems` 加入支撑流式 vision chunk。
- **v0.9（hash+cache）**：`MediaWithBytes` 引入，`get_item_for_hash` 区分 raw 与 unwrapped，让 hasher 用原始字节提高稳定性。
- **v0.10**：`video_needs_metadata` 与 `VideoProcessorItems.metadata` 出现，支撑 Qwen2-VL 等需要 metadata 的 processor；`_validate_hidden_size` 加入防止远程 embedding shape 攻击。
- **v0.11（EVS）**：vision_chunk 流式被进一步用于 EVS 相关模型（见 [evs.md](evs.md)）。
- **main**：`target_channels` / `AudioSpec` 合入解析器；`expected_hidden_size` 透传到 `EmbeddingItems` 校验；`_validate_ndim` 独立方法，先后校验 ndim 再校验 hidden，避免依赖错误 ndim 的 hidden 校验失效。

[← 返回多模态首页](../README.md)

## 参见

- [audio.md](audio.md) / [image.md](image.md)：被解析器调用的清洗函数。
- [inputs.md](inputs.md)：`DictEmbeddingItems` 直接产出 `MultiModalKwargsItems`。
- [hasher.md](hasher.md)：消费 `get_all_items_for_hash`。
- [processing.md](processing.md)：`BaseProcessingInfo.get_data_parser` 配置解析器参数。
- [media.md](media.md)：`MediaWithBytes` 的来源。
