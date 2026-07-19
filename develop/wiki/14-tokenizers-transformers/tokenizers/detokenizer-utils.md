[← 分词与转换器](../README.md) > [分词器后端](README.md) > detokenizer_utils

# detokenizer_utils.py — 增量 detokenize

## 是什么

`vllm/tokenizers/detokenizer_utils.py` 提供 EngineCore Detokenizer 在每一步 decode 后增量更新文本的工具函数。所有函数都以 `TokenizerLike` 为参数，与具体后端无关。

公开 API：

| 函数 | 行号 | 作用 |
|---|---|---|
| `INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET` | `:56` | 常量 `5`，决定 prompt 末尾预扫描多少 token |
| `convert_prompt_ids_to_tokens(tokenizer, prompt_ids, skip_special_tokens=False) -> (new_tokens, prefix_offset, read_offset)` | `:59` | 第一帧：只把 prompt 末尾若干 token 转成字符串并算 offset，不转全部（省 CPU） |
| `convert_ids_list_to_tokens(tokenizer, token_ids) -> list[str]` | `:83` | 逐 id `decode([id])`，测试/日志用 |
| `detokenize_incrementally(tokenizer, all_input_ids, prev_tokens, prefix_offset, read_offset, skip_special_tokens, spaces_between_special_tokens) -> (new_tokens, new_text, new_prefix_offset, new_read_offset)` | `:110` | 核心：每步把新 token 拼入 `prev_tokens`，扣除 `prefix_text` 后只返回 delta 文本 |

私有：`_replace_none_with_empty`（`:8`，挡 OOV）、`_convert_tokens_to_string_with_added_encoders`（`:14`，adapted from transformers v4.28.0 的 slow tokenizer 实现，用于非 fast 且有 added_vocab 的 tokenizer）。

## 为什么

- **增量不是简单 `decode([new_id])`**：BPE/SentencePiece 的 cleanup 算法依赖上下文（例如是否在 token 前后加空格、是否合并字节序列），单 token 解码会出现 `'Hello'`/`' World'` 之类的边界歧义。算法借鉴了 HF text-generation-inference v0.9.4（注释 `:107`）：维护 `prefix_offset`/`read_offset` 两个游标，把 prefix_text 作为"已确认不会变"基线，只对 `prefix_offset:` 之后的 token 调 `convert_tokens_to_string` 再扣除 prefix，得到 delta。
- **首帧不扫全 prompt**：prompt 可能几万 token，但 cleanup 只影响末尾；`convert_prompt_ids_to_tokens` 只取 `[-(INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 2):]`，避免冷启动延迟（`:72`）。
- **byte fallback**：当 token id 是不完整 UTF-8 字节序列（TikToken/Tekken 类），`convert_tokens_to_string` 会返回带 `�` 的字符串，算法判定 `new_text.endswith("�")` 时返回空 delta 等下一帧补全（`:194`）。
- **与 `TokenizerLike.is_fast` 协作**：fast tokenizer 直接 `convert_tokens_to_string`；slow + 有 added_vocab 时走慢路径 `_convert_tokens_to_string_with_added_encoders`，按 added_vocab 集合分段 join。

## 怎么做

Detokenizer 维护每条 Request 的 `(prev_tokens, prefix_offset, read_offset)` 三元组状态。每步：

```mermaid
flowchart LR
  New[新 token id] --> Append[prev_tokens + new_tokens]
  Append --> Prefix[convert_tokens_to_string prev_tokens prefix_offset:read_offset]
  Append --> Curr[convert_tokens_to_string prev_tokens prefix_offset:]
  Prefix --> Diff[new_text - prefix_text]
  Curr --> Diff
  Diff --> End{len<=len_prefix 或 endsWith �?}
  End -- yes --> Empty[返回 '' + 不动 offset]
  End -- no --> Advance[返回 delta + 更新 offset=read_offset, len(output_tokens)]
```

`is_first_iter=True`（`prev_tokens is None`）时先 `convert_prompt_ids_to_tokens` 初始化，并把全部新 tokens 作为 output 返回（首帧要给 requester 看见 prompt 末尾）。

`skip_special_tokens` 由 `SamplingParams` 决定；`spaces_between_special_tokens` 同样。

## 与其它模块/系统配合

- **引擎核心 Detokenizer**：见 [`../../01-engine-core/input-processor.md`](../../01-engine-core/input-processor.md)。EngineCore 在每个 step 把新生成的 token id 喂给 `detokenize_incrementally`，把返回的 `new_text` 累积到 `RequestOutput` 上推到 API server。
- **`vllm/renderers/online_derenderer.py`**：scale-out 路径上有"token-in-token-out"模式，detokenizer 的结果由 `OnlineDerenderer.derender_chat` 二次解析（解 tool/reasoning）。
- **`TokenizerLike`**：本文件是 Protocol 接口最直接的用户之一——证实 Protocol 足够支撑增量 detokenize 而无需任何后端特定代码。

## 历史版本演进

- **v0.3/v0.4（早期）**：`detokenize_incrementally` 从 HF text-generation-inference 移植。
- **v0.5–v0.8**：随 Mistral/Kimi 后端加入，验证算法对非 HF 后端也正确（Mistral 的 `convert_tokens_to_string` 用字节回退路径；TikToken 的 `�` 路径）。
- **v0.10**：把 detokenizer 工具从 `vllm/transformers_utils/detokenizer_utils.py` 迁到 `vllm/tokenizers/detokenizer_utils.py`，凸显其属于分词器子系统（待核实具体 PR）。
- **main**：算法本体稳定；仅对边缘情况（OOV、多字节）做小修补。

---

[← 返回分词器后端首页](README.md)

## 参见

- [protocol.md](protocol.md) — 本文件消费的接口契约。
- [hf.md](hf.md) / [mistral.md](mistral.md) — 不同后端在 `convert_tokens_to_string` 上的行为差异。
- `../renderers/online-derenderer.md` — 反渲染与 detokenizer 交界。
