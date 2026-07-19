[← 分词与转换器](../README.md) > [分词器后端](README.md) > KimiAudioTokenizer

# kimi_audio.py — Kimi-Audio 的 TikToken 后端

## 是什么

`vllm/tokenizers/kimi_audio.py` 实现 `KimiAudioTokenizer(TokenizerLike)`（`vllm/tokenizers/kimi_audio.py:52`），用 TikToken（`tiktoken.Encoding`）支持 `moonshotai/Kimi-Audio-7B-Instruct`。注册表键：`kimi_audio`。

加载策略（`from_pretrained`，`:55`）：

1. 路径是文件 → 直接用。
2. 路径是目录 → 优先 `tiktoken.model`，回退 `tokenizer.model`。
3. 否则从 HF Hub 下载 `tiktoken.model`（失败回退 `tokenizer.model`），附带拉 `tokenizer_config.json`。

词表加载（`_load_tiktoken_encoding`，`:24`）：每行 `<b64> <rank>` 两列，base64 解码成 bytes 作为 `mergeable_ranks`；用固定的 Kimi-Audio TikToken 分词正则；`special_tokens` 由 `tokenizer_config.json` 的 `added_tokens_decoder` 提供。

Kimi-Audio 专用特殊 token（`:178`）：
```
<|im_media_begin|>      151661
<|im_media_end|>        151663
<|im_kimia_text_blank|> 151666
<|im_msg_end|>          151645
<|im_kimia_user_msg_start|>      151670
<|im_kimia_assistant_msg_start|> 151671
```
默认 BOS=151643、EOS=151644、PAD=EOS、UNK=PAD（`:165`）。

`apply_chat_template`（`:384`）用 `transformers.utils.chat_template_utils.render_jinja_template` 渲染（不走 tokenizer 内置模板），再把 prompt `encode` 成 ids（默认 `tokenize=False` 返回字符串）。

## 为什么

- TikToken 的 mergeable_ranks 文件格式简单且高性能，但 transformers 没有内建 TikToken AutoTokenizer；必须自实现。
- Kimi-Audio 是音频多模态模型，其特殊 token 体系（`<|im_media_*|>`、`<|im_kimia_*|>`）与文本模型不同，需要在 `encode` 时显式列入 `allowed_special` 避免被 TikToken 当作未知字节切分。
- 与 vLLM 的 `TokenizerLike` Protocol 对齐，让 `BaseRenderer` / `OnlineRenderer` 对 Kimi-Audio 与其它模型用同样代码路径。
- 注意 `is_fast=False`（`:224`）：与 HF fast tokenizer 区分；调用方据此选择 detokenize 分支。

## 怎么做

- `encode(text)`：先 `self._tokenizer.encode(text, allowed_special={...})`（六个 Kimi-Audio special token），再按 `truncation_side` 截断。
- `decode(ids, skip_special_tokens)`：若跳 special，从 `_special_tokens.values()` 构 set 过滤。
- `convert_tokens_to_ids` / `convert_ids_to_tokens`：用自维护 `_token_to_id` / `_id_to_token` dict，未知 token 回退到 `_unk_token_id` / `<|unk|>`。
- `__call__` 手工构造 `BatchEncoding({"input_ids":..., "attention_mask":...})`，与 HF 接口对齐。
- `added_tokens_decoder` 是可写 property：setter 会根据 token 字符串自动更新 BOS/EOS（`:247`），兼容某些远程代码 processor 在加载后再注入 special token 的行为。

`apply_chat_template` 同时接受 `messages` 与 `conversation` 两个键名以兼容 Protocol 与旧调用方（`:393`）。

## 与其它模块/系统配合

- **registry**：`kimi_audio` 模式直接由 ModelConfig 显式指定（无自动检测）。
- **renderers**：在 `_VLLM_RENDERERS` 中 `kimi_audio` 复用 `HfRenderer`（`vllm/renderers/registry.py:27`），因为 Kimi-Audio 不需要特殊 chat 渲染逻辑，统一走 HF-style render_messages。
- **transformers_utils/processors/kimi_audio.py**：配套的多模态 processor（音频解析）。
- **`vllm/transformers_utils/repo_utils.hf_api`**：用于 Hub 下载。

## 历史版本演进

- **早期**：Kimi-Audio 由社区远程代码（`trust_remote_code`）支持，加载路径依赖 transformers 的 dynamic module 机制，脆弱且非进程内缓存友好。
- **v0.9（PR #36127 "Add support for moonshotai/Kimi-Audio-7B-Instruct"）**：把 TikToken 后端内建到 `vllm/tokenizers/`，注册 `kimi_audio` 模式，并配套 `transformers_utils/processors/kimi_audio.py`。
- **main（PR #36882 等）**：mypy 类型修复；`added_tokens_decoder` 的 setter 加入以兼容远程 processor 注入 special token 的用法。

---

[← 返回分词器后端首页](README.md)

## 参见

- [registry.md](registry.md) — `kimi_audio` 模式如何被选中。
- [../transformers_utils/processor.md](../transformers_utils/processor.md) — 配套多模态 processor 加载。
