# Llava 经典 VLM 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Llava**

> 代表文件：`llava.py`、`llava_next.py`、`llava_next_video.py`、`llava_onevision.py`、`llava_onevision2.py`、`h2ovl.py`、`fuyu.py`、`paligemma.py`（见 [gemma](./gemma.md)）、`chameleon.py`、`idefics2_vision_model.py`、`idefics3.py`、`molmo.py`、`molmo2.py`、`moondream3.py`、`smolvlm.py`。
> 厂商：多源（Llava 系列、H2O、Fuyu/Adept、Idefics/HuggingFace、Molmo/AllenAI、Moondream、SmolVLM）。

---

## 是什么

Llava 家族不是一个厂商系列，而是"以经典 Llava 范式（vision tower + projector + LM）为代表的 VLM 聚类"：

- **`llava.py`**：`LlavaForConditionalGeneration`——CLIP/Siglip 视觉塔（可选）+ `LlavaMultiModalProjector` + LM backbone（由 config 决定）。同时定义 `BaseLlavaProcessingInfo`/`LlavaProcessingInfo`/`BaseLlavaMultiModalProcessor`/`LlavaMultiModalProcessor`，是 vLLM 多模态 processor 模式的范本之一。也含 `PixtralHFProcessingInfo`/`PixtralHFMultiModalProcessor`（让 PixtralHF 与 Llava 共享处理框架）。
- **`llava_next.py`**：Llava-Next，引入 AnyRes token 分块（每图划分为多个 grid patch，per-patch 视觉编码）。
- **`llava_next_video.py`**：视频版 Llava-Next。
- **`llava_onevision.py` / `llava_onevision2.py`**：Llava-OneVision（图像+视频+音频），`LlavaOnevisionMultiModalProjector` 与多模态输入 schema。
- **`h2ovl.py`**：H2OVL，类 Llava-Next VLM。
- **`fuyu.py`**：Adept Fuyu，特殊（直接 patch embedding 而非 ViT）。
- **`idefics2_vision_model.py` + `idefics3.py`**：HuggingFace Idefics 系列。
- **`molmo.py` / `molmo2.py`**：AllenAI Molmo 系。
- **`moondream3.py`**：Moondream 3。
- **`smolvlm.py`**：HuggingFace SmolVLM。
- **`chameleon.py`**：Meta Chameleon（早期混合模态 token 化方案）。

---

## 为什么

- **vLLM 多模态框架的范本**：`llava.py` 的 `BaseLlavaProcessingInfo` 被 Pixtral 等复用，`LlavaMultiModalProjector` 是 connector 的样板。`get_mm_mapping`（`llava.py:730`）示范了 LM/connector/tower 三段切分。
- **`-1` token 占位 + scatter**：Llava 在 input_ids 里用 image_token_index 占位，输入处理时 INSERT 占位 token 让 KV cache 预留位置，再在 forward 时用 `_merge_multimodal_embeddings` 把视觉 embedding scatter 进去。这是 vLLM `SupportsMultiModal.embed_input_ids` 的典型实现。
- **processor 抽象的发源地**：`BaseMultiModalProcessor` + `BaseProcessingInfo` 在 Llava 系最先成熟，后被所有 VLM 沿用（见 [多模态子系统](../../11-multimodal/README.md)）。

---

## 怎么做

经典流程：

1. **输入处理**（CPU 离线）：`LlavaMultiModalProcessor` 把图像 + 文本 prompt 拼成 input_ids，image_token 插入到指定位置；输出 `MultiModalFeatureSpec`。
2. **视觉塔前向**（GPU）：`vision_tower(pixel_values)` 得到 image embedding，connector 投影。
3. **scatter**：`embed_input_ids` 先算 text embedding，再 `_merge_multimodal_embeddings` 按 `is_multimodal` mask 把 image embedding 散射到对应位置。
4. **LM forward**：与纯文本完全一致。

Llava-Next 的 AnyRes 在第 1 步把单图切多块、在视觉塔输出后再 reshape；Llava-OneVision 在多模态输入上支持 image/video/audio 三模态。

---

## 与其它模块/系统配合

- **[多模态](../../11-multimodal/README.md)**：本家族是 vLLM 多模态处理框架的样板来源。
- **[vision.md](../vision.md)**：图像 token 数计算、特征选择策略、DP 分片由 vision.py 提供。
- **[embedding-col](./embedding-col.md)**：`colpali.py` 基于 PaLI-Gemma（属本家族延伸），`ColPaliModel` 实现 late-interaction。
- **[模型执行-加载器](../../03-model-execution/model-loader/README.md)**：`LlavaForConditionalGeneration.load_weights` 自适应 LM backbone 的层名。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | `llava.py` 落地，是 vLLM 第一批 VLM。 |
| v0.5 | Llava-Next（AnyRes）+ 视频；H2OVL、Idefics、Fuyu。 |
| v0.6–v0.7 | Llava-OneVision；processor 抽象（`BaseProcessingInfo`）成熟；DP 分片视觉塔。 |
| v0.8 | Molmo / Moondream3 / SmolVLM。 |
| v0.9–v0.10 | Llava-Onevision2；与 encoder CUDA graph 接口对齐。 |
| main | 处理框架持续重构，signpost 各家族 processor 复用模式。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [vision](../vision.md) · [多模态](../../11-multimodal/README.md) · [embedding-col](./embedding-col.md) · [vlm-misc](./vlm-misc.md)
