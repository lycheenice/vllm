[← 分词与转换器](../README.md) > [分词器后端](README.md) > DeepseekV32Tokenizer

# deepseek_v32.py — DeepSeek-V3.2 DSML 模板后端

## 是什么

`vllm/tokenizers/deepseek_v32.py` + `deepseek_v32_encoding.py` 共同实现 `deepseek_v32` 模式。`DeepseekV32Tokenizer(TokenizerLike)`（`vllm/tokenizers/deepseek_v32.py:85`）以 `PreTrainedTokenizerFast.from_pretrained` 为底座，再用 `get_deepseek_v32_tokenizer`（`:15`）做动态子类化（`_DeepseekV32Tokenizer`）覆盖 `apply_chat_template`，把消息走自定义 `encode_messages`（来自 encoding 文件）。

`deepseek_v32_encoding.py`（471 行）是 DeepSeek 官方 encoding 的拷贝（注释 `copy from huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py`），定义：

- 特殊 token：`<｜begin▁of▁sentence｜>` / `<｜end▁of▁sentence｜>` / `thinking_start_token="imid"` / `thinking_end_token="imo"` / `dsml_token="｜DSML｜"`
- 消息模板：`system_msg_template` / `user_msg_template="<｜User｜>{content}<｜Assistant｜>"` / `assistant_msg_template="{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"`
- 工具调用：`TOOLS_SYSTEM_TEMPLATE`（含 DSML 函数调用语法示例）+ `tool_call_template` / `tool_calls_template` / `tool_output_template`
- `response_format_template`：强制 schema 应答
- `encode_messages(messages, *, thinking_mode, drop_thinking)`：核心编码函数

`get_deepseek_v32_tokenizer` 用 `copy.copy` + 动态类（避免子类化整个 tokenizer 已有类），`__reduce__` 让 pickle 重建通过该函数；`__len__` 处理 `▝` 等 added token 与 `vocab_size` 不一致问题（`:69`）；`get_added_vocab` 返回冻结副本。

## 为什么

- DeepSeek-V3.2 自有 DSML（DeepSeek Markup Language）函数调用语法 `<｜DSML｜function_calls>...<｜DSML｜invoke name="...">`，HF jinja 模板已被官方替换为这套 encoding；vLLM 必须忠实复刻。
- thinking 模式：`thinking_mode="thinking"` 启用 `imid...imo` 思维链块；`drop_thinking=True` 在新 user 消息进入时丢弃历史 reasoning content（官方约定）。这些不能靠简单 jinja 表达。
- 不能简单重写 `PreTrainedTokenizerFast`：vLLM 期望保留所有底层方法（`encode`/`decode`/`convert_*` 等）不变，只换 `apply_chat_template`；动态子类化是最小侵入方案。

## 怎么做

`DeepseekV32Tokenizer.from_pretrained(*args, **kwargs)`（`:86`）：

1. `PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)` 拿到 base tokenizer；
2. `get_deepseek_v32_tokenizer(tokenizer)` 包一层，把 `apply_chat_template` 替换为 DSML 版本；
3. `get_cached_tokenizer(...)` 缓存常用属性（与 `hf.py` 共用）。

被覆盖的 `apply_chat_template`（`:26`）：

- 解析 `kwargs.thinking` / `kwargs.enable_thinking` 决定 `thinking_mode`；
- 若有 `tools`：在 messages 头部插一条 `{"role": "system", "tools": tools}`（DSML 系统模板会渲染）；
- `drop_thinking = messages[-1]["role"] == "user"`；
- 调 `encode_messages(messages, thinking_mode=..., drop_thinking=...)` 得 prompt 字符串；
- 按 `kwargs.tokenize` 决定返回 ids 或字符串。

## 与其它模块/系统配合

- **`renderers/deepseek_v32.py`**：`DeepseekV32Renderer(BaseRenderer[DeepseekV32Tokenizer])` 把 `apply_chat_template` 委托给 tokenizer 自身实现（renderer 自身不重写），并支持同时传 `conversation=` 与 `messages=` 两个键名兼容。
- **`tool_parsers/deepseekv32_engine_tool_parser.py`** 与 `reasoning/` 中 DeepSeek-V3.2 相关 parser：消费 `imid`/`imo`/`｜DSML｜` 边界，配合 DSML 输出做 tool call 抽取。
- **`registry._VLLM_TOKENIZERS["deepseek_v32"]`** 是入口。
- **xgrammar builtin structural tag `deepseek_v3_2`**（`vllm/tool_parsers/structural_tag_registry.py:64`）：tool parser 设 `structural_tag_model="deepseek_v3_2"` 时与该 tokenizer 协同做严格工具调用。

## 历史版本演进

- **v0.10（PR #29837 "supports deepseekv32 chat template"）**：首次引入 `deepseek_v32` 模式，把官方 encoding 文件原样拷入 `deepseek_v32_encoding.py`。
- **v0.10（PR #30009 "Fix TokenizerLike interface"）**：随 Protocol 统一，调整 `from_pretrained` 签名。
- **v0.10（PR #30025 "fixed deepseekv32 tool calling error"）**：修复 tool 模式下的渲染细节。
- **main（PR #45877 等）**：在引入 Streaming Parser Engine 后，DeepSeek-V3.2 也可作为引擎化 parser 的子类（`registered_adapters.py` 内），与该 tokenizer 配合。

---

[← 返回分词器后端首页](README.md)

## 参见

- [deepseek-v4.md](deepseek-v4.md) — V4 沿用同一架构，新增 reasoning_effort 与 quick instruction task。
- `../renderers/deepseek-v32.md` — 配套 renderer。
- [../tool_parsers/README.md](../tool_parsers/README.md) — DeepSeek V3/V3.1/V3.2/V4 系列 parser。
