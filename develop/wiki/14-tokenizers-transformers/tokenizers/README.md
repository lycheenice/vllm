[← 分词与转换器](../README.md) > 分词器后端

# 分词器后端 (vllm/tokenizers/)

## 是什么

`vllm/tokenizers/` 是 vLLM 的分词器加载与使用后端。它对外暴露统一 `TokenizerLike` Protocol（[`protocol.md`](protocol.md)），通过 `TokenizerRegistry`（[`registry.md`](registry.md)）按 `tokenizer_mode` 分发到五种后端实现。所有"如何把文本切成 token id / 把 token id 拼回文本"的细节都集中在这里。

文件清单：

| 文件 | 角色 |
|---|---|
| `__init__.py` | 包导出（`TokenizerRegistry` / `cached_get_tokenizer` / `get_tokenizer` / `cached_tokenizer_from_config` / `TokenizerLike` / `maybe_make_thread_pool`） |
| `protocol.py` | `TokenizerLike` Protocol 接口定义 | [protocol.md](protocol.md) |
| `registry.py` | 注册表、`cached_get_tokenizer`、参数归一化 | [registry.md](registry.md) |
| `hf.py` | 默认 `hf` 后端 `CachedHfTokenizer`、`get_cached_tokenizer`、`maybe_make_thread_pool` | [hf.md](hf.md) |
| `mistral.py` | `mistral_common` 包装（Tekken/SentencePiece + grammar） | [mistral.md](mistral.md) |
| `kimi_audio.py` | TikToken 后端 for Kimi-Audio | [kimi-audio.md](kimi-audio.md) |
| `deepseek_v32.py` + `deepseek_v32_encoding.py` | DeepSeek-V3.2 DSML chat 模板 | [deepseek-v32.md](deepseek-v32.md) |
| `deepseek_v4.py` + `deepseek_v4_encoding.py` | DeepSeek-V4 chat 模板 + reasoning_effort | [deepseek-v4.md](deepseek-v4.md) |
| `detokenizer_utils.py` | 增量 detokenize（引擎 Detokenizer 用） | [detokenizer-utils.md](detokenizer-utils.md) |
| `fastokens.py` | `VLLM_USE_FASTOKENS=1` Rust BPE 后端 shim | [fastokens.md](fastokens.md) |

## 为什么

- vLLM 早期只依赖 HF `AutoTokenizer`，但 Mistral 官方分词器、TikToken 类、DeepSeek 自有 chat 编码都无法被 `AutoTokenizer` 表达；需要统一抽象 + 多后端共存。
- detokenizer 在每一步 decode 中只更新末尾 token，必须用增量算法避开 cleanup 算法的歧义；这套算法与具体 tokenizer 解耦，集中放在 `detokenizer_utils.py`。
- HF fast tokenizer 的所有公开方法非线程安全，多请求并发时需要一个可 pickle 的线程池包装。

## 怎么做

加载路径见 [`registry.md`](registry.md)。运行期使用：

- **Tokenizer** 经 `cached_tokenizer_from_config` 进入 `BaseRenderer`，由其调用 `apply_chat_template` / `__call__` / `encode` / `decode`。
- **Detokenizer** 在 EngineCore 内对每个新 token 调 `detokenize_incrementally`（[`detokenizer-utils.md`](detokenizer-utils.md)），输出 `RequestOutput.delta_text`。
- **Grammar/Structured Output**：Mistral tekken tokenizer 通过 `tokenizer.grammar_factory` / `llg_tokenizer` 暴露给结构化输出后端；tool parser 通过 `structural_tag_model` 类属性配合 xgrammar 内建模板（见 `../tool_parsers/structural-tag-registry.md`）。

## 与其它模块/系统配合

- **transformers_utils**：`registry.get_tokenizer` 在加载前调 `get_config` 并 `_maybe_register_hf_config`；`mistral.py` 用 `vllm.utils.mistral.is_mistral_tokenizer` 做类型识别。
- **renderers**：`renderer_from_config` 用 `tokenizer_args_from_config` 推断 renderer 模式，使得 tokenizer 与 renderer 一一对应（如 `mistral` 模式→`MistralRenderer`）。
- **tool_parsers / reasoning**：parser 拿到的是 `TokenizerLike` 实例，通过 `vocab` / `convert_tokens_to_ids` 与特殊 token id 做切分。
- **引擎核心 Detokenizer**：见 [`../../01-engine-core/input-processor.md`](../../01-engine-core/input-processor.md)。

## 历史版本演进

- v0.5 之前：仅 `AutoTokenizer`，无注册表。
- v0.6（PR #7739）：Mistral 后端加入，注册表诞生。
- v0.9（PR #36127）：`kimi_audio` 加入。
- v0.10（PR #29837）：`deepseek_v32` + DSML 模板加入。
- v0.11（PR #41741/#43168）：`fastokens` 集成；线程安全包装 `maybe_make_thread_pool` 进场（用于多 worker 并发）。
- v0.12/main（PR #45877）：`deepseek_v4` 后端加入；`_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` 随 step3/step3p7/unlimited-ocr 上线扩充。

---

[← 返回子系统首页](../README.md)

## 参见

- [registry.md](registry.md) / [hf.md](hf.md) / [protocol.md](protocol.md)
- [mistral.md](mistral.md) / [kimi-audio.md](kimi-audio.md) / [deepseek-v32.md](deepseek-v32.md) / [deepseek-v4.md](deepseek-v4.md)
- [detokenizer-utils.md](detokenizer-utils.md) / [fastokens.md](fastokens.md)
