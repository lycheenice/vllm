# 模型输出解析器（parser）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 模型输出解析器

本页覆盖 `vllm/parser/`，描述统一 `Parser` 抽象——把推理（reasoning，如 `<think>` 段）与工具调用（tool call，XML/JSON）解析合并为单一接口，以及一批模型专用 parser 实现。

> 注意：本目录**不是** CLI 参数解析（那是 [utils.md](utils.md) 的 `argparse_utils.py` 与各 `cli` 子命令），而是**模型生成文本的结构化解析**。

## 是什么

### 抽象层

- `Parser`（`vllm/parser/abstract_parser.py:86`）：抽象基类，统一 `ReasoningParser`（`vllm.reasoning`）与 `ToolParser`（`vllm.tool_parsers`）两套既有接口。子类可：①直接重写抽象方法；②设置 `reasoning_parser_cls`/`tool_parser_cls` 委托给既有实现。核心抽象方法：`is_reasoning_end`/`extract_content_ids`/`extract_reasoning`/`extract_reasoning_streaming`/`extract_tool_calls`/`extract_tool_calls_streaming`/`parse`/`parse_delta`/`adjust_request`。
- `DelegatingParser`（`abstract_parser.py:371`）：推荐基类，把方法委托给内部 `_reasoning_parser`/`_tool_parser`；实现 `parse_delta` 的核心流式编排：先用 `StreamState`（`:44`，每流一份）做 reasoning phase → tool phase 状态机，处理 reasoning 结束过渡、tool_call 起始、required/named tool choice、engine-based parser flush（`_flush_engine_parsers`，`:922`）、`finalize_generation`（`:755`）兜底未完成生成。
- `ParserManager`（`parser_manager.py:21`）：组合入口，`get_tool_parser`/`get_reasoning_parser` 从各自注册表按名取类，组装出统一 `Parser`。
- `StreamState`（`abstract_parser.py:44`）：流式状态（`reasoning_ended`/`tool_call_text_started`/`previous_text`/`previous_token_ids`/`history_tool_call_cnt`/`function_name_returned`/`engine_based` 等）；`advance`/`commit` 管理增量累加，engine-based 模式下不再累加（每步独立）。

### `parser/engine/`：可组合解析引擎

`__init__.py`、`adapters.py`、`events.py`、`incremental_lexer.py`、`parser_engine.py`、`parser_engine_config.py`、`registered_adapters.py`、`streaming_parser_engine.py`、`token_id_scanner.py`——一套基于 lexer/adapter 的可扩展解析框架，支持按 token id 与增量文本驱动，供模型专用 parser 复用（受 `VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT` 控制）。

### 模型专用 parser（节选）

`harmony.py`（GPT-OSS Harmony）、`deepseek_v3.py`/`deepseek_v4.py`、`glm47_moe.py`、`kimi_k2.py`、`minimax_m2.py`、`qwen3.py`、`gemma4.py`、`mistral.py`、`nemotron_v3.py`、`seed_oss.py`——每个对应一族模型的 reasoning/tool 输出格式，继承 `Parser`/`DelegatingParser`，设置相应 `reasoning_parser_cls`/`tool_parser_cls` 或重写方法。

### 辅助

- `metrics.py`：`record_tool_parser_invocation`，在 `DelegatingParser` 的 `extract_tool_calls`/`_streaming` finally 块中记录调用与是否命中（见 [可观测性](../16-observability/README.md)）。
- `utils.py`：`count_history_tool_calls` 等历史 tool call 计数（kimi_k2 id 类型专用）。

## 为什么

- **双 parser 合一**：reasoning 与 tool call 在流式输出里是**连续两段**（先 think 后 tool），需要共享 stream state 协调过渡（reasoning 结束→切到 tool phase）。旧的两套独立接口难以正确处理过渡边界，`Parser` 统一状态机。
- **engine-based 与否**：部分 parser 由引擎侧（structured engine/guided decoding）确认 reasoning 结束，需 `engine_based_streaming` 模式与不同的 advance/commit 语义；`StreamState.engine_based` 切换两种路径。
- **模型差异收敛**：各厂商模型输出格式千差万别，统一抽象让 API server 只调 `Parser`，不感知具体模型。
- **`adjust_request` 注入结构化输出**：DelegatingParser 在请求阶段把 structural_tag 转成 `StructuredOutputsParams`（见 [sampling-params.md](sampling-params.md)），让引擎在采样端约束输出格式。

## 怎么做

- **选用 parser**：API server 经 `ParserManager` 按 `reasoning_parser`/`tool_parser` 名称组装实例，注入请求处理路径。
- **新增模型 parser**：继承 `DelegatingParser`，`__init__` 中设 `self._reasoning_parser`/`self._tool_parser`（或设类属性 `reasoning_parser_cls`/`tool_parser_cls`），按需重写 `extract_*`；复杂格式可借助 `parser/engine/` 框架。
- **流式口径**：实现 `parse_delta(delta_text, delta_token_ids, request, prompt_token_ids, *, finished)`，返回 `DeltaMessage | None`；`finished=True` 时框架会调 `finalize_generation` + `_flush_engine_parsers`。
- 结构化输出：重写 `adjust_request` 或依赖 `_apply_structural_tag` 注入 `StructuralOutputsParams`。

## 与其它模块/系统配合

- [API 入口](../13-entrypoints/README.md)：OpenAI/Anthropic/Responses 路径用 `Parser` 把模型输出转 `DeltaMessage`、`FunctionCall`。
- [分词与转换器](../14-tokenizers-transformers/README.md)：`vllm.tool_parsers`/`vllm.reasoning` 是委托目标；`tokenizer` 由 `Parser` 持有。
- [采样与解码 · 结构化输出](../06-sampling-decoding/README.md)：`StructuredOutputsParams`/`structural_tag` 经 parser 注入。
- [sampling-params.md](sampling-params.md)：`StructuredOutputsParams` 类型定义于此。
- [可观测性](../16-observability/README.md)：`metrics.record_tool_parser_invocation`。
- [envs.md](envs.md)：`VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS`、`VLLM_ENFORCE_STRICT_TOOL_CALLING`、`VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS`、`VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT`。

## 历史版本演进

- **v0.5–v0.6**：reasoning 与 tool parser 各自独立，无统一抽象；流式过渡边界易出 bug。
- **v0.7–v0.8**：`vllm.tool_parsers`/`vllm.reasoning` 体系成型，但仍分离。
- **v0.9–v0.10**：引入 `vllm/parser/` 与统一 `Parser`/`DelegatingParser`/`ParserManager`，`StreamState` 状态机解决过渡问题；`parser/engine/` 实验框架引入。
- **v0.11–main**：大批模型专用 parser（harmony/deepseek/glm/kimi/minimax/qwen/gemma/mistral/nemotron/seed_oss）就位；engine-based streaming 与 structural_tag 集成完善（具体版本待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [sampling-params.md](sampling-params.md)（`StructuredOutputsParams`）
- [分词与转换器](../14-tokenizers-transformers/README.md)
- [API 入口](../13-entrypoints/README.md)
- [可观测性](../16-observability/README.md)
