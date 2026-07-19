# Phi 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Phi**

> 代表文件：`phi.py`、`phi3.py`、`phi3v.py`、`phi4mm.py`、`phi4mm_audio.py`、`phi4mm_utils.py`、`phi4siglip.py`、`phimoe.py`、（清退：`Phi3SmallForCausalLM`、`Phi4FlashForCausalLM`、`Phi4MultimodalForCausalLM`）。
> 厂商：Microsoft。

---

## 是什么

- **`phi.py`**：Phi-1/1.5/2 的早期实现，独立 `PhiForCausalLM`（非 Llama 子类），含自己的 `PhiAttention`/`PhiMLP`。
- **`phi3.py:10`**：`Phi3ForCausalLM(LlamaForCausalLM)`——从 Phi-3 起 Microsoft 把架构对齐 Llama，vLLM 直接子类化复用。`Phi3ForCausalLM` 在 registry 中还作 embedding backbone（`_EMBEDDING_MODELS` 里 `Phi3ForCausalLM` 指向 phi3）。
- **`phi3v.py`**：Phi-3-Vision 多模态版，`Phi3VForCausalLM`。
- **`phi4mm.py`** + `phi4mm_audio.py` + `phi4mm_utils.py`：Phi-4-MultiModal，支持图像 + 音频 + 多模态输入，是 Phi 系最全任务模型。
- **`phi4siglip.py`**：Phi-4 with Siglip 视觉塔（`Phi4ForCausalLMV`），用 SiglipVisionModel 作视觉编码器。
- **`phimoe.py`**：PhiMoE，MoE 版本。
- **spec decode**：Phi 系未提供原生 MTP/EAGLE draft；`registry.py` 的 `_SPECULATIVE_DECODING_MODELS` 里没有 Phi target——但 Phi3 可作 embedding backbone，部分 Phi 模型可被通用 draft（如 `ExtractHiddenStatesModel`）使用。

### 已清退

`_PREVIOUSLY_SUPPORTED_MODELS`（`registry.py:707+`）：
- `Phi3SmallForCausalLM`：v0.9.2 退役
- `Phi4FlashForCausalLM`：v0.10.2 退役
- `Phi4MultimodalForCausalLM`：v0.12.0 退役

这些早期实现因维护成本被移除，回退到老版本可用。

---

## 为什么

- **架构对齐 Llama 的样板**：Phi-3 起的设计让 vLLM 直接走 `Phi3ForCausalLM(LlamaForCausalLM)`，省一大堆 attention/MLP/decoder 代码，只覆盖差异点（如 RoPE 变体、特殊初始化）。
- **多模态扩展接口测试场**：Phi-4-MM 覆盖图像 + 音频，是 vLLM `SupportsMultiModal` 多模态组合的实战样本；`phi4mm_utils.py` 抽出共享前处理。
- **清退反映维护成本**：Phi3-Small / Phi4-Flash / Phi4-Multimodal 退役说明，特殊变体（Flash 量化、私有 embedding 结构）长期维护成本高于价值。

---

## 怎么做

`Phi3ForCausalLM(LlamaForCausalLM)` 的子类化很薄——只是把 `LlamaForCausalLM` 的 `model`（`LlamaModel`）替换为需要的 config 解析路径，权重加载用 `AutoWeightsLoader` + 自定义 `hf_to_vllm_mapper` 做层名对齐（HF Phi3 与 Llama 命名略有差异）。

`Phi4MMForCausalLM` 含独立 `Phi4MMImageEncoder` + 音频 encoder + 多模态 processor；走标准 `SupportsMultiModal` 三件套（`_mark_language_model`/`_mark_tower_model`/`get_mm_mapping`）。

---

## 与其它模块/系统配合

- **[llama 家族](./llama.md)**：Phi3+ 全部继承 LlamaForCausalLM。
- **[多模态](../../11-multimodal/README.md)**：Phi3V / Phi4MM / Phi4-Siglip 共享多模态处理栈；Siglip 视觉塔由 `siglip.py` 实现。
- **[embedding-col](./embedding-col.md)**：Phi3 可作 embedding backbone。
- **[speech-audio](./speech-audio.md)**：Phi4MM 的音频路径。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.4 | Phi-1/1.5/2 接入（`phi.py` 独立实现）。 |
| v0.5 | Phi-3 重构为 `LlamaForCausalLM` 子类；Phi-3-Vision。 |
| v0.7 | PhiMoE 上线。 |
| v0.8 | Phi3Small / Phi4-Flash 退役前曾短暂支持。 |
| v0.9 | Phi3Small 退役（v0.9.2）。 |
| v0.10 | Phi4Flash 退役；Phi4-MM（图像+音频）落地。 |
| v0.11 | Phi4-Siglip（Siglip 视觉塔）。 |
| v0.12 | Phi4-Multimodal 退役（旧实现）；新统一 Phi4-MM 稳定。 |
| main | Phi 系与 Llama 持续同步升级。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [llama 家族](./llama.md) · [多模态](../../11-multimodal/README.md) · [speech-audio](./speech-audio.md)
