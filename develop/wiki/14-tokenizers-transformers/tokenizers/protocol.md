[← 分词与转换器](../README.md) > [分词器后端](README.md) > TokenizerLike

# protocol.py — TokenizerLike Protocol

## 是什么

`vllm/tokenizers/protocol.py` 定义 `TokenizerLike`（`typing.Protocol`），是所有分词器后端必须满足的鸭子类型接口。它把 HF `PreTrainedTokenizer`/`PreTrainedTokenizerFast` 的子集+vLLM 特有属性固化下来，方便 `BaseRenderer`/`ToolParser`/`ReasoningParser` 等下游组件静态类型检查。

接口分组（`vllm/tokenizers/protocol.py:13`）：

- **类方法**：`from_pretrained(path_or_repo_id, *, trust_remote_code, revision, download_dir, **kwargs) -> TokenizerLike`
- **特殊 token 属性**：`all_special_tokens` / `all_special_ids` / `bos_token_id` / `eos_token_id` / `pad_token_id` / `num_special_tokens_to_add()`
- **元数据**：`is_fast` / `vocab_size` / `max_token_id` / `max_chars_per_token` / `truncation_side`
- **编码/解码**：`__call__`（返回 `BatchEncoding`）、`encode`、`decode`、`convert_tokens_to_ids`（重载 str|list）、`convert_ids_to_tokens`、`convert_tokens_to_string`、`get_vocab` / `get_added_vocab`
- **chat 模板**：`apply_chat_template(messages, tools, **kwargs) -> str | list[int]`
- **协议固有**：`__hash__ = hash(id(self))`、`__len__ = vocab_size`

## 为什么

- **Protocol 而非 ABC**：vLLM 既需要兼容未继承任何 vLLM 基类的 transformers 原生 `PreTrainedTokenizerFast`（直接被 `TokenizerRegistry` 当 `hf` 后端用），又需要 Mistral/TikToken/DeepSeek 等纯自实现后端。Protocol 提供"结构子类型化"，让两类对象都能通过类型检查。
- **`__hash__` 默认 `id(self)`**：因为 `lru_cache` 在 `cached_get_tokenizer` 上以 tokenizer 为参数之一（间接），需要可哈希；而 HF fast tokenizer 默认不可哈希，Protocol 显式声明让 mypy 接受 `__hash__`。
- **`max_token_id` / `max_chars_per_token`**：vLLM 自定义属性（HF 没有），用于在采样器/结构化输出里快速确定词表上界、估算长度，避免每次重算。`get_cached_tokenizer` 会把它俩缓存为 property。

## 怎么做

实现一个新后端只需要写一个 class 满足 Protocol（可直接 `class X(TokenizerLike)` 但非强制），并在 `registry.py` 的 `_VLLM_TOKENIZERS` 注册一条 `"mode": ("module", "ClassName")`。

`from_pretrained` 是 classmethod 工厂入口，所有后端的 `__init__` 都不直接被外部调用——`TokenizerRegistry.load_tokenizer` 统一走 `cls.from_pretrained(*args, **kwargs)`（`vllm/tokenizers/registry.py:78`）。

## 与其它模块/系统配合

- `BaseRenderer[_T: TokenizerLike]`（`vllm/renderers/base.py:72`）把具体后端作为泛型参数，例如 `HfRenderer(BaseRenderer[HfTokenizer])`、`MistralRenderer(BaseRenderer[MistralTokenizer])`，从而 renderer 内可直接调用具体类型特有方法（如 `MistralTokenizer.supports_grammar`）。
- `ToolParser.__init__(tokenizer: TokenizerLike, tools)` 与 `ReasoningParser.__init__(tokenizer, *args, **kwargs)` 都以 Protocol 形参，再用 `cached_property vocab` 调 `get_vocab()`。
- `detokenize_incrementally(tokenizer: TokenizerLike, ...)` 只用 Protocol 上的 `convert_ids_to_tokens` / `convert_tokens_to_string` / `get_added_vocab` / `all_special_tokens` / `is_fast` / `__len__`，因此对所有后端通用。

## 历史版本演进

- **早期**：vLLM 直接对 `transformers.PreTrainedTokenizer` 编程，没有显式 Protocol。
- **v0.8/v0.9**：随着 Mistral/Kimi-Audio 等非 HF 后端增多，逐步显式化接口；`TokenizerLike` 作为 Protocol 沉淀到 `protocol.py`。
- **v0.10（PR #30009 "Fix TokenizerLike interface"）**：把 `from_pretrained` 的 kwargs（`trust_remote_code` / `revision` / `download_dir`）显式列在 Protocol 上，统一所有后端的工厂签名。
- **v0.11/main**：renderers 子系统采用 `Protocol` 作为泛型上界，protocol.py 不再频繁变动。

---

[← 返回分词器后端首页](README.md)

## 参见

- [registry.md](registry.md) — 通过 Protocol 加载的入口。
- [hf.md](hf.md) / [mistral.md](mistral.md) — 两个典型 Protocol 实现。
