# Encoder-only 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Encoder-only**

> 代表文件：`bert.py`、`bert_with_rope.py`、`roberta.py`、`modernbert.py`、`jina.py`。
> 配套：`colbert.py`、`colmodernvbert.py`、`colpali.py`、`colqwen3.py`、`colqwen3_5.py`、`gritlm.py`、`voyage.py`、`extract_hidden_states.py` 见 [embedding-col](./embedding-col.md)。

---

## 是什么

- **`bert.py`**：BERT 全套。`BertModel(nn.Module, SupportsQuant)` 是底座；衍生：`BertPoolingModel(BertModel)`、`BertEmbeddingModel(nn.Module, SupportsQuant)`（句向量）、`BertMLMHead`（masked LM）、`SPLADESparsePooler` + `BertSpladeSparseEmbeddingModel(BertEmbeddingModel)`（稀疏检索）、`BertForSequenceClassification(SupportsCrossEncoding, SupportsQuant)`（cross-encoder rerank）、`BertForTokenClassification`、`BertForImageAndTextRetrieval`？。`registry.py:212` 注册 `BertModel`/`BertSpladeSparseEmbeddingModel` 入 embedding。
- **`bert_with_rope.py`**：带 RoPE 的 BERT 变体（如 Nomic BERT、GTE），`NomicBertModel` / `GteNewModel` / `GteModel` / `SnowflakeGteNewModel`。
- **`roberta.py`**：RoBERTa，`RobertaEmbeddingModel(BertEmbeddingModel)`、`RobertaForSequenceClassification`、`RobertaForMaskedLM`。XLM-RoBERTa 走 `RobertaForSequenceClassification`（`registry.py:319, 245`）。
- **`modernbert.py`**：ModernBERT（HF 2024），`ModernBertModel` 含 RoPE + alternated local/global attention + MLP；可作 embedding、seq cls、token cls。
- **`jina.py`**：Jina embeddings 系列（v3/v5），`JinaEmbeddingsV5Model` 与 `JinaForRanking`（late-interaction 风格），`JinaVLForSequenceClassification`（VLM rerank，注册项）。

`colbert.py` 中 `ColBERTModel(ColBERTMixin, BertEmbeddingModel)`、`ColBERTModernBertModel`、`ColBERTJinaRobertaModel`、`ColBERTLfm2Model` 复用上述 encoder 作 late-interaction 检索。

`HfMoondream` 等的 vision backbone（CLIP/Siglip）不在本页，归属 [vlm-misc](./vlm-misc.md) 或各厂商。

---

## 为什么

- **embedding/classify 的底座**：BERT 系是 vLLM `/v1/embeddings` 与 `/v1/score`、`/v1/classify` 的主力后端，与 [`adapters.md`](../adapters.md) 的"生成模型在线转 pool"形成互补。
- **`SupportsCrossEncoding` 的样板**：`BertForSequenceClassification` 是 cross-encoder rerank 的范本（`score_type="cross-encoder"`），是 [`interfaces.md`](../interfaces.md) 该接口的典型实现。
- **ModernBERT 引入 ModernBERT 后的 attention pattern**：alternated local/global + RoPE 让 encoder-only 也能用 sliding window，与 decoder 系如 Gemma2 相呼应。

---

## 怎么做

`BertModel` 含 `BertEmbedding`（token+position+segment）+ `BertEncoder`（多层 `BertLayer` = `BertAttention` + `BertIntermediate` + `BertOutput`）+ pooler。`BertEmbeddingModel` 在此基础上做 mean/cls 池化（由 `VllmModelForPooling.default_seq_pooling_type` 配置）。

`BertForSequenceClassification` 用 `score` 头（Linear）+ DispatchPooler；`SPLADESparsePooler` 输出稀疏向量。`BertForTokenClassification` 用 per-token 头（用于 NER 或 ASR forced-alignment 的对齐——`Qwen3ASRForcedAlignerForTokenClassification` 复用 `_TOKEN_CLASSIFICATION_MODELS` 通路）。

`ModernBertModel` 取代旧 BERT 在新版本中的位置：含 RoPE、局部 attention、co-trained MLP，既可 pooling 也可 seq cls。

---

## 与其它模块/系统配合

- **[embedding-col](./embedding-col.md)**：ColBERT 系复用 encoder 是下游。
- **[adapters.md](../adapters.md)**：encoder 模型原生 pooling，与"生成模型在线转 pool"互补。
- **[interfaces.md](../interfaces.md)**：`SupportsCrossEncoding` / `VllmModelForPooling` 等接口的样板所在。
- **[transformers-backend](../transformers-backend.md)**：ModernBERT 也通过 vLLM 原生实现（不强制走 HF 后端）。
- **[多模态](../../11-multimodal/README.md)**：`Terratorch` 与 `PrithviGeoSpatialMAE` 走 embedding 注册（地理空间 MAE）。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | BERT / RoBERTa 接入，做 embedding / classify。 |
| v0.5 | ColBERT 接入（late-interaction）；SPLADE 稀疏嵌入。 |
| v0.7 | Jina v3 + ModernBERT。 |
| v0.8 | Jina v5 + JinaForRanking；ModernBERT 加进 seq/token cls。 |
| v0.10 | BERT 系与 `as_*` 适配器协同统一 pool 接口；`gritlm.py` 双向模型。 |
| main | ModernBERT 在 attention pattern 上与 decoder 家族共享 sliding window 工具。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [embedding-col](./embedding-col.md) · [adapters](../adapters.md) · [interfaces](../interfaces.md)
