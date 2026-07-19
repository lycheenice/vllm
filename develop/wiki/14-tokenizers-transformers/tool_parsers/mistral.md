[← 分词与转换器](../README.md) > [工具调用解析](README.md) > MistralToolParser

# mistral_tool_parser.py — Mistral 工具调用解析

## 是什么

`vllm/tool_parsers/mistral_tool_parser.py`（769 行）定义 `MistralToolParser(ToolParser)`，注册名 `mistral`。它处理 `mistral_common` 官方工具调用协议（v3/v5/v7/v11+），并直接调用 `mistral_common` 库做 encode/decode，不重新实现协议。

关键类与常量（`vllm/tool_parsers/mistral_tool_parser.py`）：

- `IS_MISTRAL_TOOL_PARSER = True`（`:107`）：被 `vllm.utils.mistral.is_mistral_tool_parser` 识别。
- `MistralToolCall(ToolCall)`（`:73`）：override id 生成——`generate_random_id()` 返回 9 字符 alphanumeric（Mistral 协议硬约束）。
- `StreamingState` Enum（`:57`）：流式状态机，9 个状态覆盖 `WAITING_FOR_TOOL_START → PARSING_NAME → PARSING_ARGUMENTS → TOOL_COMPLETE → ALL_TOOLS_COMPLETE` 等。
- `model_can_reason: bool = False`：用于 `adjust_request` 决定是否在 grammar 中给 reasoning 留通道。
- `_is_pre_v11_tokeniser(model_tokenizer)`（`:87`）：Mistral v11 引入 `[ARGS]` token，该函数判断是否需要走老协议。
- `_DEFAULT_JSON_SCHEMA = {"anyOf": [{"type":"object"},{"type":"array"}]}`：tool_choice 不是命名/required 时的兜底 grammar。

## 为什么

- Mistral 协议有官方参考实现（`mistral_common`），重新实现风险高；本 parser 直接调 `MistralTokenizer` 暴露的方法。
- v11+ 模型以 `[TOOL_CALLS]` 单 token 触发，参数走 `[ARGS] ... [/ARGS]` 块；v11 前则需要识别其他边界——`_is_pre_v11_tokeniser` 分支兜底。
- 流式状态机复杂（9 个状态）：每个 token 增量只触发相邻状态转移，避免整段重解析（与 Hermes 的"全量重解析+差分"风格不同）。
- 9 字符 id 限制：Mistral 协议规定 tool_call_id 必须正好 9 字符 alphanumeric，所有 id 必须重新生成（不能继承 OpenAI 风格 `call_abc123...`）。
- Mistral tokenizer 与 grammar：Mistral tekken v11+ 支持 grammar，`adjust_request` 会根据 tool_choice 与 `model_can_reason` 构造对应的 grammar schema 给 mistral_common 的 `Request`。

## 怎么做

`adjust_request`（main 中体量大）：

1. 处理 `tool_choice`：`none`/`auto`/`required`/命名 function 分别构造不同 grammar；
2. 若 `model_can_reason`，在 grammar 中预留 `[TOOL_CALLS]` 与 reasoning tokens 的关系；
3. 把 `request.structured_outputs` 设上对应 grammar，或对 Responses API 设置 `request.text.format`。

`extract_tool_calls`（非流式）：

- 用 mistral_common 的 `parse_tool_calls` 把 `model_output` 拆为 `list[ToolCall]`；
- 把 mistral ToolCall 转 vLLM `ToolCall`（含 `MistralToolCall.generate_random_id`）。

`extract_tool_calls_streaming`：状态机驱动，按 `StreamingState` 当前态推进，产出 `DeltaToolCall`。`function_name` 必须一次性发出（OpenAI 约定）；arguments 增量推送。

## 与其它模块/系统配合

- **`vllm/tokenizers/mistral.py`**：`is_mistral_tokenizer(tokenizer)` 把底层 `tokenizer.tokenizer` 暴露给本 parser。
- **`vllm/parser/mistral.py`**：`MistralParser(DelegatingParser)` 当 `tool_parser_cls` 是 Mistral 时被 `ParserManager` 选中；它override 部分 `extract_*` 让 reasoning 与 tool 调用专门协同。
- **`vllm/renderers/online_renderer.py`**：Mistral 请求期会调 `_mt.maybe_serialize_tool_calls`/`_mt.truncate_tool_call_ids`/`_mt.validate_request_params` 预处理（在调 parser 之前）。
- **`vllm/utils/mistral.py`**：`is_mistral_tool_parser`、`mt` 共享助手。
- **`vllm/tool_parsers/streaming.py`**：`extract_named_tool_call_streaming` 用 `MistralToolCall.generate_random_id` 为命名 tool_choice 生成 id。

## 历史版本演进

- **v0.5（PR #5649）**：Mistral parser 首版，对应 mistral_common v3 协议。
- **v0.6–v0.9**：随 mistral_common v7/v11 update，逐步加入 `[ARGS]`/`[TOOL_CALLS]` 流式；`MistralToolCall` 9 字符 id 约束固化。
- **v0.10**：`StreamingState` 状态机细化；与 reasoning parser 协同（`model_can_reason`）。
- **v0.11/main**：`adjust_request` 大改以支持 grammar + Responses API；与 `MistralParser` 协同抽出 reasoning/tool 分流。

---

[← 返回工具调用解析首页](README.md)

## 参见

- [abstract-tool-parser.md](abstract-tool-parser.md) / `hermes.md`
- [../tokenizers/mistral.md](../tokenizers/mistral.md) — `MistralTokenizer` 包装。
- `../renderers/mistral.md` — Mistral renderer。
