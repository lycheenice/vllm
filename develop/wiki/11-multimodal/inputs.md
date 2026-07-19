# inputs.py · 多模态输入数据模型

[← Wiki 首页](../README.md) > [多模态](../README.md) > inputs

## 是什么

`vllm/multimodal/inputs.py` 定义多模态数据在"已被 HF Processor 处理之后、进入模型之前"的全部数据结构，以及它们之间的拆分 / 合并 / 缓存语义。它是整个子系统的"数据合同"——`parse.py` 输出的 items、`processing/processor.py` 输出的 kwargs、`cache.py` 缓存的 value、`v1/engine` 跨进程序列化的字段，统统落在这套类型上。核心类型包括：

- 类型别名 `ImageItem` / `VideoItem` / `AudioItem` / `HfImageItem` / `HfVideoItem` / `HfAudioItem` / `VisionChunk(Image|Video)`。
- `PlaceholderRange`：占位符 token 区间 + `is_embed` 掩码。
- `NestedTensors` / `BatchedTensorInputs`：异形张量容器。
- `MultiModalFieldElem` / `BaseMultiModalField`（`Batched` / `Flat` / `Shared`）/ `MultiModalFieldConfig`：描述"如何把一个 key 拆成 per-item、再合并回 batched"。
- `MultiModalKwargsItem` / `MultiModalKwargsItems`：per-item 与 per-modality 的处理结果。
- `MultiModalFeatureSpec`：贯穿引擎核心的请求级单 item 描述。

## 为什么

HF Processor 一次性返回一个扁平的 `BatchFeature`（如 `pixel_values` 形状 `(N, ...)`、`image_grid_thw` 形状 `(N, 3)`），但 vLLM 需要解决三个 HF 不关心的问题：

1. **缓存粒度**：同一张图被多个请求复用时，不能整批进 IPC，需要 per-item 命中/置空。
2. **跨请求合批**：多个请求的图片若形状一致，应 `torch.stack` 成一个 batch 一次喂编码器；不一致要降级为 list。
3. **占位符稀疏**：Qwen3-VL 等模型在 `<vision_start>` / timestamp / `<image>` 之间夹杂文本 token，`PlaceholderRange.is_embed` 必须 mask 出真正要去编码器输出取的行。

因此 `inputs.py` 把"一个 key 的数据该如何切分与合并"显式建模成 `BaseMultiModalField`，让模型作者通过 `MultiModalFieldConfig.batched/flat/flat_from_sizes/shared` 一行声明，框架据此自动 build_elems（拆）与 reduce_data（合）。

## 怎么做

### PlaceholderRange

`PlaceholderRange`（`inputs.py:118`）是 frozen dataclass，字段 `offset` / `length` / `is_embed`。`is_embed` 为 `None` 时整段都是 embedding；非 None 时是 `(length,)` 布尔掩码。关键方法：

- `embeds_cumsum`（`:148`）：用 python list 缓存 `is_embed.cumsum().tolist()`，避免反旋 torch C++ 开销。
- `get_num_embeds()`（`:152`）：返回真正 embedding 数（用 cumsum 末位）。
- `get_embeds_indices_in_range(start_idx, end_idx)`（`:158`）：把"[start_idx, end_idx) 的占位符窗口"映射到"[embeds_cumsum 的对应起止)"，供 `Scheduler` 与 `EncoderRunner` 计算需要取多少行 encoder 输出。
- `extract_embeds_range()`（`:179`）：返回 `[(abs_start, abs_end), ...]` 的连续 embedding 段（绝对坐标），供 `attn_utils.compute_mm_prefix_ranges` 构造 PrefixLM 双向注意力区间。

### NestedTensors

`NestedTensors`（`:218`）= `list[NestedTensors] | list[Tensor] | Tensor | tuple[Tensor, ...]`。当 batch 内各元素形状一致用 Tensor，否则用 list。`nested_tensors_equal`（`:229`）做结构敏感的相等比较，供缓存 key 校验；`_nested_tensors_h2d`（`:269`）用 `json_map_leaves` 递归 `to(device)`，保留非 Tensor 叶子。

### Field 体系

`MultiModalFieldElem`（`:348`）= `(data: NestedTensors, field: BaseMultiModalField)`。`BaseMultiModalField`（`:385`）abstract，定义两个方向：

- `build_elems(modality, key, data)`：HF 输出 → per-item elem 列表（拆）。
- `reduce_data(elems, *, device, pin_memory) -> NestedTensors`：per-item elem → batched 张量（合）。
- `keep_on_cpu`：True 时该字段不进 GPU（如 `image_grid_thw` 等小元数据）。

四种实现：

| 类 | 工厂 | 拆分语义 | 合并语义 |
|---|---|---|---|
| `MultiModalBatchedField`（`:463`） | `MultiModalFieldConfig.batched` | 沿 dim0 索引 | `torch.stack`（形状一致）/ list（不一致） |
| `MultiModalFlatField`（`:511`） | `flat` / `flat_from_sizes` | 沿 dim=`dim` 切片 | `torch.concat`（一致）/ 零填充切片赋值（变长，issue #31658） |
| `MultiModalSharedField`（`:618`） | `shared` | 整体返回，重复 `batch_size` 次 | 取第一个（所有 item 共享） |

`reduce_data` 对 batch_size==1 都做了 zero-copy 优化：`batched` 用 `unsqueeze(0).contiguous()`，`flat` 用 `contiguous()`，避免无意义 `stack/concat`。`MultiModalBatchedField._reduce_data` 还处理 pin_memory 的"先 contiguous 再 pin"避免双拷贝。

### MultiModalKwargsItem(s)

`MultiModalKwargsItem`（`:866`）继承 `UserDict[str, MultiModalFieldElem]`，是"单个 modality item 的所有 kwargs"（如一张图的 `pixel_values` + `image_grid_thw`）。`dummy(nbytes)` 是测试用便利构造。`get_data()` 返回 `{key: elem.data}` 的扁平版本。

`MultiModalKwargsItems`（`:894`，泛型 `_I` 默认 `MultiModalKwargsItem`，另可选 `MultiModalKwargsItem | None`）是 `{modality: Sequence[item]}`。两类核心方法：

- `from_hf_inputs(hf_inputs, config_by_key)`（`:930`）：按 `MultiModalFieldConfig` 把 HF `BatchFeature` 拆成 per-item；同 modality 内 batch_size 必须一致（否则 raise）。
- `get_data(*, device, pin_memory)`（`:983`）：调用 `utils.group_and_batch_mm_items` 把所有 item 重组合并，要求最终每个 modality 恰好一个 batch（否则 `RuntimeError`）。

`MultiModalKwargsOptionalItems`（`:1024`）别名允许 item 为 `None`（缓存命中置空）。

### MultiModalFeatureSpec

`MultiModalFeatureSpec`（`:301`）是"贯穿引擎核心"的单 item 描述：

- `data: MultiModalKwargsItem | None`：`None` 表示已缓存于 P1，跳过 IPC。
- `modality: str`、`identifier: str`（带 LoRA 前缀的缓存 key）、`mm_hash: str | None`（不带 LoRA 前缀的处理器缓存 key）。
- `mm_position: PlaceholderRange`。
- `gather_kwargs(features, keys)`（`:334`）：静态方法，从多个 spec 收集指定 keys 的 data 列表，供 `EncoderRunner.prepare_mm_inputs` 用。

## 与其它模块/系统配合

- **parse.py**：`ImageProcessorItems` / `AudioProcessorItems` 等"原始媒体容器"是 `MultiModalDataItems` 的 value；它们产出 `MultiModalKwargsItems` 的入口是 `BaseMultiModalProcessor` 内部调用 `from_hf_inputs`。
- **processing/processor.py**：`BaseMultiModalProcessor._get_mm_fields_config` 是模型作者实现 `_get_mm_fields_config` 的地方，返回 `Mapping[str, MultiModalFieldConfig]`，本文件据此 build_elems。
- **cache.py**：`MultiModalProcessorCacheItem` / `MultiModalProcessorCacheItemMetadata` 把 `MultiModalKwargsItem` + `ResolvedPromptUpdate` 一起缓存；`ShmObjectStoreSenderCache.address_as_item` 用 `MultiModalBatchedField` 把 SHM 地址伪装成 `MultiModalKwargsItem`，receiver 侧再 `get(address)` 反序列化。
- **utils.py**：`group_and_batch_mm_items` / `group_and_batch_mm_kwargs` 是 `get_data` 与 `EncoderRunner` 合批的底层。
- **v1/engine**：`EngineCoreRequest.mm_features: list[MultiModalFeatureSpec]`（`vllm/v1/engine/__init__.py:96`）是跨进程消息；`InputProcessor.process_inputs`（`v1/engine/input_processor.py:333`）把 `mm_kwargs/mm_placeholders/mm_hashes` 三 dict 按 `argsort_mm_positions` 拍平成 `MultiModalFeatureSpec` 列表。详见 [v1-integration.md](v1-integration.md)。
- **v1/worker/gpu**：`EncoderCache.mm_features` / `encoder_outputs`、`EncoderRunner.prepare_mm_inputs` / `gather_mm_embeddings`、`attn_utils.compute_mm_prefix_ranges` 全部基于本文件的类型。

## 历史版本演进

- **v0.5（LLaVA 初版）**：仅有 `PlaceholderRange(offset, length)` 与 flat dict 的 `MultiModalKwargs`；`NestedTensors` 已存在但用得少。
- **v0.6**：引入 `MultiModalFieldConfig`（仅 `batched` / `flat`），把"per-item 拆分"从各模型抽到框架。
- **v0.7（v1 化）**：`MultiModalKwargsItem` 改为 `UserDict`；`MultiModalFeatureSpec` 出现作为 v1 请求字段，`identifier` 与 `mm_hash` 分离（为 LoRA tower connector 准备）。
- **v0.8**：`MultiModalSharedField` 加入，支持跨 item 共享元数据（如 Qwen2-VL 的 `image_grid_thw` 在 batch 内共享场景）；`flat_from_sizes` 取代手算 slice。
- **v0.9（hash+cache）**：`MultiModalKwargsItems` 泛型化支持 `None` item，配合 sender 缓存命中置空；`embeds_cumsum` 缓存到 `cached_property`。
- **v0.10**：`is_embed` 字段加入 `PlaceholderRange`，`get_embeds_indices_in_range` / `extract_embeds_range` 出现，支撑 Qwen3-VL timestamp token 混入场景。
- **v0.11（EVS）**：`MultiModalFeatureSpec` 的 `mm_hash` 字段加入（之前只有 `identifier`），让 receiver cache 能跨 LoRA 共享编码器输出。
- **main**：`MultiModalFlatField._reduce_data` 加入变长 slice-assign 路径（issue #31658，Ultravox 不同音频时长）；`validate_embedding_ndim` 风格的 hidden_size 校验由 `parse.EmbeddingItems` 承担（见 [parse.md](parse.md)）。

[← 返回多模态首页](../README.md)

## 参见

- [parse.md](parse.md)：产生 items 的前置阶段。
- [processing.md](processing.md)：消费 `MultiModalFieldConfig` 的处理器。
- [cache.md](cache.md)：以 `MultiModalKwargsItem` 为缓存 value。
- [utils.md](utils.md)：`group_and_batch_mm_items` 等 helper。
- [v1-integration.md](v1-integration.md)：`MultiModalFeatureSpec` 在 v1 的传递路径。
