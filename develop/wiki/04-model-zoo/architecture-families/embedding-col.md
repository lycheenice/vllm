# Embedding / 检索家族（含 Col）

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Embedding/检索**

> 代表文件：`bert.py`、`bert_with_rope.py`、`roberta.py`、`modernbert.py`、`jina.py`、`jina_vl.py`、`gritlm.py`、`voyage.py`、`colbert.py`、`colmodernvbert.py`、`colpali.py`、`colqwen3.py`、`colqwen3_5.py`、`extract_hidden_states.py`。
> 底座详见 [encoder-only](./encoder-only.md)。

---

## 是什么

vLLM 的 pooler 任务族（embedding/classify/score/reward/late-interaction），统一走 `VllmModelForPooling` 接口（见 [`interfaces_base.py:148`](../interfaces.md)）：

- **纯句向量**：`BertEmbeddingModel`、`RobertaEmbeddingModel`、`ModernBertModel`、`JinaEmbeddingsV5Model`、`BgeM3EmbeddingModel`（基于 roberta）、`NomicBertModel`、`GteNewModel`/`SnowflakeGteNewModel`、`SiglipEmbeddingModel`、`CLIPEmbeddingModel`、`VoyageQwen3BidirectionalEmbedModel`。
- **生成模型在线转 embedding**：通过 [`adapters.md`](../adapters.md) `as_embedding_model`，把 `LlamaForCausalLM`/`Qwen2ForCausalLM`/`Gemma2Model`/`MistralModel` 等转成 `*ForEmbedding` 类，注册项含 `LlamaModel`/`Qwen2Model`/`Gemma2Model`/`Gemma3TextModel`/`GlmForCausalLM` 等。
- **Cross-encoder rerank**：`BertForSequenceClassification`、`RobertaForSequenceClassification`、`XLMRobertaForSequenceClassification`、`GteNewForSequenceClassification`、`JambaForSequenceClassification`、`LlamaBidirectionalForSequenceClassification`、`ModernBertForSequenceClassification`、`JinaVLForSequenceClassification`、`LlamaNemotronVLForSequenceClassification`。`score_type="cross-encoder"`。
- **Late-interaction（逐 token 检索）**：`ColBERTModel`、`ColBERTModernBertModel`、`ColBERTJinaRobertaModel`、`ColBERTLfm2Model`、`ColPaliModel`、`ColQwen3Model`、`ColQwen3_5Model`、`ColModernVBertForRetrieval`、`JinaForRanking`。`SupportsLateInteraction`：`score_type="late-interaction"`。
- **Reward**：`InternLM2ForRewardModel`、`Qwen2ForRewardModel`、`Qwen2ForProcessRewardModel`。
- **SPLADE 稀疏**：`BertSpladeSparseEmbeddingModel`。
- **特殊**：`ExtractHiddenStatesModel`（`extract_hidden_states.py`）作为 spec draft（注册到 `_SPECULATIVE_DECODING_MODELS`），用于把 hidden state 喂给外部 draft。
- **GritLM**：`gritlm.py GritLM`，生成 + embedding 双向模型（同时注册到 generate 与 embedding）。
- **Voyage**：`voyage.py VoyageQwen3BidirectionalEmbedModel`，Voyage AI 基于 Qwen3 的双向 embedding。

---

## 为什么

- **一种底座，多种 score_type**：BERT 既能做 embedding（bi-encoder）又能做 cross-encoder rerank，区别仅在 `score_type` 标签 + `DispatchPooler` 选择。vLLM 用 `VllmModelForPooling.score_type` 字段统一这条路径。
- **late-interaction 独立分支**：ColBERT 风格的"每 token embedding + MaxSim"算法需要返回 token-level 而非 seq-level 输出，所以独立 `SupportsLateInteraction` 标签，与 bi/cross 区别。
- **生成模型作为 embedding 来源**：很多新 embedding（如 Voyage、LlamaBidirectional）走"生成模型 + 双向 attention 改造 + pooling"路径，由 `as_embedding_model`/`as_seq_cls_model` 完成。

---

## 怎么做

`ColPaliModel`（`colpali.py:89`）基于 PaLI-Gemma：复用其LM backbone，但取 per-token 输出做检索。`ColQwen3Model`/`ColQwen3_5Model` 基于 Qwen3 backbone。`ColBERTModel(ColBERTMixin, BertEmbeddingModel)` 把 BERT 输出 token-level 化。

`VoyageQwen3BidirectionalEmbedModel` 把 Qwen3 改造成双向（attention mask 全填），再 pooler。

`LlamaBidirectionalModel`/`LlamaBidirectionalForSequenceClassification` 由 `as_embedding_model`/`as_seq_cls_model` 派生（见 `llama.py:543-549`），改 attention 为双向。

`Score API` 由 `ModelConfig.score_type` 路由：`bi-encoder` → `/v1/embeddings`、`cross-encoder` → `/v1/score`、`late-interaction` → `/v1/score`（token_embed task）。详见 [配置](../../10-config/README.md) 与 [13-entrypoints](../../13-entrypoints/README.md)。

---

## 与其它模块/系统配合

- **[adapters.md](../adapters.md)**：把生成模型在线转 embedding/classify 的核心。
- **[interfaces.md](../interfaces.md)**：`VllmModelForPooling` / `SupportsCrossEncoding` / `SupportsLateInteraction` 标签。
- **[encoder-only](./encoder-only.md)**：底座实现。
- **[模型执行-层库](../../03-model-execution/layers/README.md)**：`DispatchPooler` / `SequencePooler` / `SPLADESparsePooler` 由 `layers/pooler.py` 提供。
- **[13-entrypoints](../../13-entrypoints/README.md)**：Pooler API（`/v1/embeddings`、`/v1/score`、`/v1/rank`）由这些模型供能。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：`ExtractHiddenStatesModel` 作 draft 外部接口。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | BERT/RoBERTa 做 embedding/classify。 |
| v0.5 | ColBERT（late-interaction）+ ColPali（VLM 检索）。 |
| v0.6 | SPLADE 稀疏 + GritLM 双向。 |
| v0.7 | ModernBERT + Jina v3。 |
| v0.8 | Jina v5 + JinaForRanking + ColQwen3。 |
| v0.10 | `as_seq_cls_model` 的 `from_2_way_softmax` 让 Qwen3-Reranker 在线转。 |
| v0.11 | Voyage + ColQwen3.5 + ColModernVBert。 |
| main | Reward/PRM（Qwen2 RM）接口稳定；token_embed API 上线。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [encoder-only](./encoder-only.md) · [adapters](../adapters.md) · [interfaces](../interfaces.md) · [13-entrypoints](../../13-entrypoints/README.md)
