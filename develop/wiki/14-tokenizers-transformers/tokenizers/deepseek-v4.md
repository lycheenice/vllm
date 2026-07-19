[← 分词与转换器](../README.md) > [分词器后端](README.md) > DeepseekV4Tokenizer

# deepseek_v4.py — DeepSeek-V4 模板后端

## 是什么

`vllm/tokenizers/deepseek_v4.py` + `deepseek_v4_encoding.py` 实现 `deepseek_v4` 模式。结构与 [deepseek-v32.md](deepseek-v32.md) 几乎一致：

- `DeepseekV4Tokenizer(TokenizerLike)`（`vllm/tokenizers/deepseek_v4.py:92`）：`from_pretrained` 调 `PreTrainedTokenizerFast.from_pretrained` + `get_deepseek_v4_tokenizer` 包装 + `get_cached_tokenizer` 缓存。
- `get_deepseek_v4_tokenizer`（`:15`）：动态子类 `_DeepseekV4Tokenizer` 覆盖 `apply_chat_template`，`__reduce__` 让 pickle 重建。

`deepseek_v4_encoding.py`（757 行）的扩展点（相对 V3.2）：

- **任务分类 token**：`DS_TASK_SP_TOKENS`（`action`/`query`/`authority`/`domain`/`title`/`read_url`）配合 `VALID_TASKS`，供 quick-instruction 内部任务使用。
- **`latest_reminder_msg_template` + `<｜latest_reminder｜>`**：在多轮中插入"最新提醒"。
- **`reasoning_effort`**：`encode_config` 多了 `reasoning_effort` 参数；映射规则在 `apply_chat_template` 内（`vllm/tokenizers/deepseek_v4.py:43`）：
  - `"none"` → `thinking_mode="chat"`、`reasoning_effort=None`
  - `"max"`/`"xhigh"` → `"max"`
  - 其它 → `"high"`
- **`REASONING_EFFORT_MAX`**：在 prompt 中注入"绝对最大思考"占位文本（`vllm/tokenizers/deepseek_v4_encoding.py:70`），引导模型进入最强推理。
- **`tool_calls_block_name="tool_calls"`**（V3.2 也用 `function_calls`，V4 改为 `tool_calls`）。
- **`tool_output_template`**：从 `<result>...</result>` 改为 `<tool_result>...</tool_result>`。

## 为什么

- DeepSeek-V4 在 V3.2 DSML 基础上引入 `reasoning_effort`（与 OpenAI `reasoning_effort`、Mistral v15 的同名参数对齐），让用户显式控制思考深度。
- 任务分类 token 与 quick-instruction 是 V4 线上业务（搜索/阅读理解等）专用，必须由 chat 模板注入而非用户手写。
- 仍用动态子类化：与 V3.2 一致的实现风格，便于维护——同一作者/同一 PR 链路。

## 怎么做

`apply_chat_template` 流程（`vllm/tokenizers/deepseek_v4.py:26`）：

1. 解析 `thinking` / `enable_thinking` → `thinking_mode`；
2. 解析 `reasoning_effort`（仅当是字符串）按上述规则映射；`"none"` 强制关闭思考；
3. 若有 `tools`：插 system 消息带 `tools` 字段；
4. `encode_messages(messages, thinking_mode=..., drop_thinking=..., reasoning_effort=...)`；
5. 按 `kwargs.tokenize` 返回 ids 或字符串。

`tool_calls_block_name` 与 parser 侧的 DSML token 必须严格一致，否则 `tool_parsers/deepseekv4_engine_tool_parser.py` 找不到边界。

## 与其它模块/系统配合

- **`renderers/deepseek_v4.py`**：`DeepseekV4Renderer(BaseRenderer[DeepseekV4Tokenizer])`，结构与 V3.2 renderer 同构，只是泛型参数换成 V4。
- **`tool_parsers/deepseekv4_engine_tool_parser.py`** + **`reasoning/deepseek_v4_engine_reasoning_parser.py`**：均从 `vllm/parser/engine/registered_adapters.py` 导入引擎化 adapter（参考 v0.12 PR #45877）。
- **xgrammar builtin structural tag `deepseek_v4`**（`structural_tag_registry.py:66`）。
- **`vllm/reasoning/abs_reasoning_parsers.py.count_reasoning_tokens`** 与 `reasoning_effort` 计费/截断协同（见 `../06-sampling-decoding/thinking-budget.md`）。

## 历史版本演进

- **v0.12/main（PR #45877 "Port DeepSeek V4 to streaming parser engine framework"）**：V4 后端与引擎化 parser 同步落地。同时引入 `reasoning_effort` 与 `REASONING_EFFORT_MAX` 文本注入机制。
- 备注：V3.2 与 V4 在 main 共存，encoding 文件互相独立不共享，避免 V4 改动回潮影响 V3.2。（待核实：是否有计划统一两者 encoding。）

---

[← 返回分词器后端首页](README.md)

## 参见

- [deepseek-v32.md](deepseek-v32.md) — 前代版本，结构与特殊 token 大体一致。
- `../renderers/deepseek-v4.md` — 配套 renderer。
- [../tool_parsers/README.md](../tool_parsers/README.md) — V4 引擎化 tool parser。
- ../reasoning.md — V4 引擎化 reasoning parser。
