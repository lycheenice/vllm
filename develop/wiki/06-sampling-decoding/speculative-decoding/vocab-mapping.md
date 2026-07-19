[← Wiki 首页](../../README.md) > [采样与解码](../../README.md) > [投机解码](../README.md) > VocabMapping

# VocabMapping（异构 vocab 映射）

> 源码：`vllm/v1/spec_decode/vocab_mapping.py`

---

## 是什么

`VocabMapping` 用于 draft 模型与 target 模型使用不同 tokenizer / vocab 时的 token id 翻译。它构造两组 LUT：

- `draft_to_target_ids: LongTensor[draft_vocab_size]`：每个 draft vocab 中的 token id 对应的 target token id。
- `target_to_draft_ids: LongTensor[target_vocab_size]`：反向映射。
- `intersection_mask_draft: BoolTensor[draft_vocab_size]`：draft vocab 中"存在于 target vocab"的位 mask，用于 constrain draft logits。

不在交集中的 token 会被映射到 `unk_token_id`（fallback 到 `eos_token_id`，再 fallback 抛 ValueError）。

类签名（`vllm/v1/spec_decode/vocab_mapping.py:68`）：

```python
class VocabMapping:
    def __init__(self, target_tokenizer, draft_tokenizer,
                 target_vocab_size, draft_vocab_size, device): ...
    def map_target_to_draft_ids(self, target_ids): ...
    def map_draft_to_target_ids(self, draft_ids): ...
    def constrain_draft_logits(self, logits): ...
```

由 `DraftModelProposer` 在 `use_heterogeneous_vocab=True` 时实例化（`vllm/v1/spec_decode/draft_model.py:49`）；EAGLE 等需要 hidden state 的 drafter 也支持此机制但代码路径类似（待核实）。

## 为什么

- **跨 tokenizer spec decode**：target 用 LLaMA tokenizer（vocab=32000），draft 用 Qwen tokenizer（vocab=151936）或反之；不映射则 draft token 在 target 看是乱码，拒绝采样必失败。
- **lossless 保证**：rejection sampling 要求 draft 与 target 在共同 token 上有相同语义。仅对"共同 vocab 子集"做映射与 constrain，可保证 draft 概率分布不漂移——lossless rejection sampling 不退化。
- **space-prefix 归一化**：BPE 用 `Ġ`（U+0120）表示词首空格，SentencePiece 用 `▁`（U+2581）；`_detect_space_prefix` 通过 tokenize `" a"` 动态探测，避免硬编码。
- **UNK fallback**：out-of-intersection token 必须有合理 fallback，否则 `target_to_draft_ids[i] = -1` 会让后续操作崩；优先 `unk_token_id` → `eos_token_id` → 抛错。
- **intersection size warning**：当 intersection < 100 时 warning，提示用户"drafter 精度会严重受损"。

## 怎么做

### __init__ 流程（行 68）

1. 取 target/draft 的 unk_token_id（`_get_unk_token_id` 行 42，注释明确 `unk_token_id == 0` 也是合法的有效值，不能用 `or 0`）。
2. 调 `_detect_space_prefix(tokenizer)` 探测两 tokenizer 的 space prefix character（BPE 的 Ġ / SentencePiece 的 ▁）。
3. 取 `tokenizer.get_vocab()` dict，构造 normalized vocab：
   ```python
   target_normalized[norm_token] = tid  # 用 _normalize_token 去掉 space prefix
   ```
4. 计算两 normalized vocab 的交集 `common_tokens`。
5. 填充 LUT：
   - `draft_to_target_ids[d_id] = t_id` 对应每个 common token。
   - `target_to_draft_ids[t_id] = d_id`。
   - `intersection_mask_draft[d_id] = True`。
6. 移到 device，记录 `intersection_size`、log 信息。

### map_target_to_draft_ids（行 138）

```python
def map_target_to_draft_ids(self, target_ids):
    draft_ids = self.target_to_draft_ids[target_ids]  # new tensor
    missing = draft_ids == -1
    if missing.any():
        draft_ids[missing] = self.draft_unk_token_id
    return draft_ids.to(target_ids.dtype)
```

- 用 `index_select` 等价的 `[]` 操作（返回新 tensor，不修改 LUT）。
- `missing` 集合填 `draft_unk_token_id`。
- dtype 保持与输入一致（避免后续 op dtype 不匹配）。

### map_draft_to_target_ids（行 145）

对称地映射回去，missing 填 `target_unk_token_id`。

### constrain_draft_logits（行 152）

```python
def constrain_draft_logits(self, logits):
    return logits.masked_fill(~self.intersection_mask_draft, float("-inf"))
```

在 draft model `compute_logits` 之后、采样之前调用——把不在交集的 draft vocab 位置全部 mask 成 `-inf`，强制 drafter 只采交集内的 token。此举让后续 `map_draft_to_target_ids` 必然成功（不触发 UNK fallback）。

### 在 drafter 中的集成

`DraftModelProposer._sample_draft_tokens`（基类行 468）：

```python
if self.use_heterogeneous_vocab:
    logits = self.vocab_mapping.constrain_draft_logits(logits)
draft_token_ids, draft_probs = self._sample_from_logits(logits, ...)
if self.use_heterogeneous_vocab:
    draft_token_ids = self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
    # probabilistic + heterogeneous not yet supported
    assert draft_probs is None
```

`set_inputs_first_pass` 入口也会先调 `map_target_to_draft_ids(target_token_ids)` 与 `map_target_to_draft_ids(next_token_ids)`，确保 drafter 输入也是 draft vocab 空间。

### 配置开关

`SpeculativeConfig.use_heterogeneous_vocab`（默认 False）开启此机制；开启时 `verify_equal_vocab_size_if_draft_model` 跳过断言（否则 vocab 不等会报错）。

## 与其它模块/系统配合

- [draft-model.md](draft-model.md)：本章主要使用方；独立小模型 drafter 与 target vocab 不同的场景最常见。
- [eagle.md](eagle.md) / [gemma4.md](gemma4.md) / [step3p5.md](step3p5.md)：EAGLE 家族也支持 heterogeneous vocab，但通常 drafter 与 target 同源 vocab，故此机制默认关闭。
- [../rejection-sampler.md](../rejection-sampler.md)：drafter 输出 draft_token_ids 已是 target vocab 空间；rejection sampler 直接用，不需要二次映射。
- [tokenizers](../../14-tokenizers-transformers/README.md)：`get_tokenizer` 工厂 + `cached_tokenizer_from_config`；mapping 在 ModelRunner 初始化时一次性建立。
- [配置体系-SpeculativeConfig](../../10-config/README.md)（待补充）：`use_heterogeneous_vocab` + config validation。

## 历史版本演进

- **v0.10.0**：VocabMapping 引入；与 DraftModelProposer 同期 landfall。
- **v0.10.5**：`_detect_space_prefix` 动态探测加入（之前硬编码 BPE `Ġ`）；处理 BPE/SentencePiece 混合。
- **v0.10.5+**：`_get_unk_token_id` 修正：`unk_token_id == 0` 是合法值（之前用 `or 0` 会误处理）；fallback 到 `eos_token_id`。
- **v0.11.0**：WARNING 当 intersection < 100 加入；与 probabilistic draft sampling 显式 atomic check（draft_probs must be None with heterogeneous vocab）。
- **v0.12 / main**：draft + target tokenizer 缓存机制接入 `cached_tokenizer_from_config`。

[← 返回投机解码](../README.md)

## 参见

- [draft-model.md](draft-model.md)
- [eagle.md](eagle.md)
- [tokenizers](../../14-tokenizers-transformers/README.md)
