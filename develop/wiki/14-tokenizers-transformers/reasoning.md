# 推理解析器（Reasoning Parsers）

[← Wiki 首页](../README.md) > [分词与转换器](../README.md) > 推理解析器

## 是什么

`vllm/reasoning/` 子包提供"推理段解析器（reasoning parser）"集合，专门用于把模型的原始输出 token 流切分为「思考（reasoning）」与「最终答复（answer）」两段文本。它的源头是 `vllm/reasoning/abs_reasoning_parser.py` 定义的抽象基类 `ReasoningParser`，所有具名 parser（DeepSeek-R1、Qwen3、GLM4.7、Gemma4、Kimi K2、MiniMax M2/M3、Mistral、Step3、Cohere Command、Granite、Ernie4.5、Olmo3、Nemotron V3、Hunyuan A13B、HY-V3、Seed-OSS、GPT-OSS 等）都继承该基类并实现 `extract_reasoning_content()` 与 `parse_reasoning_streaming()` 等钩子。

每个 parser 通过 `vllm/config/reasoning.py::ReasoningConfig.parser` 或 `ReasoningParserManager`（位于 `abs_reasoning_parsers.py`）按名字登记并被选择。它本质上是一组**模型特定的字符串/Token 边界识别规则**：识别 `<think>...</think>`、`<reasoning>`、`<|channel|>analysis`、`<|start_thinking|>` 等分隔标签，然后产出 `ReasoningParserOutput(reasoning_content, content)`。

## 为什么

- **统一对外接口**：上游（`OpenAIServingResponses`、`OpenAIServingChat`、`tool_parsers/`）只关心"思考 vs 答复"两段，不关心具体模型用了哪种 delimiter。parser 在中间抹平厂商差异。
- **流式正确性**：思考段常常边输出边拼接，必须**增量识别开/闭标签**才能在 SSE 流中实时给出 `reasoning` 字段而不把标签泄漏给用户。这正是 `parse_reasoning_streaming()` 的职责。
- **可与 tool_call 协同**：模型可能先思考、再调用工具；parser 需要把 tool_call 部分剥离，交给 [`14-tokenizers-transformers/tool_parsers/`](tool_parsers/README.md)。
- **与 thinking_budget 联动**：[`06-sampling-decoding/thinking-budget.md`](../06-sampling-decoding/thinking-budget.md) 控制思考段**输出长度上限**，parser 负责**前端识别/截断/包装**。

## 怎么做

### 注册与选择
- `ReasoningParser` 基类（`abs_reasoning_parsers.py`）定义 `@classmethod find_allowed_names()`、`extract_reasoning_content()`、`parse_reasoning_streaming()` 等。
- 各 parser 在文件末尾通过 `ReasoningParserManager.register_module(<name>)` 装饰自身，名字与 `ReasoningConfig.parser` 字段对应。
- 引擎初始化时（`vllm/v1/engine/...`）按配置构造 parser 实例，注入到下游使用点。

### 基类流程
1. `parse_reasoning_streaming(prev_delta_text, ...)`：维护内部 `current_mode`（reasoning/content）与 pending 缓冲；逐 delta 检测开标签 → 切 reasoning 模式；检测闭标签 → 切 content 模式；用 `previous_token`/`delta_token_id` 处理跨 delta 的部分标签匹配（如 `<` → `<thi` → `>...`）。
2. `extract_reasoning_content(content, ...)`：对**完整最后文本**做一次性分割，结果给非流式路径。
3. `parse_reasoning_streaming` 返回 `ReasoningParserOutput(reasoning_content=..., content=...)`，由调用方决定走 `reasoning` SSE 字段还是 `delta.content`。

### 代表性 parser
| parser | 关键 delimiter |
|---|---|
| `deepseek_r1_reasoning_parser.py` | `<think>...</think>` |
| `qwen3_engine_reasoning_parser.py` | `<think>...` / `<|im_end|>` |
| `glm47_moe_reasoning_parser.py` | `<think>...`（兼容 GLM 风格） |
| `gemma4_engine_reasoning_parser.py` | `<start_thinking>/<end_thinking>`（搭配 `gemma4_utils.py` 工具） |
| `minimax_m2_reasoning_parser.py` / `minimax_m3_reasoning_parser.py` | MiniMax 自定义边界 |
| `mistral_reasoning_parser.py` | `<think>...</think>` + Mistral tekken 兼容 |
| `step3_reasoning_parser.py` / `step3p5_reasoning_parser.py` | Step3 的 channel-based 分析段 |
| `cohere_command_reasoning_parser.py` / `granite_reasoning_parser.py` | 标签差异 |
| `deepseek_v3_reasoning_parser.py` / `deepseek_v4_engine_reasoning_parser.py` | DeepSeek V3/V4 引擎集成 |
| `identity_reasoning_parser.py` | 不切割，原样返回（调试用） |
| `basic_parsers.py` | 通用"先 think 后 answer"模板 |

## 与其它模块/系统配合

- **入口**：[`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md) 在 Responses API 输出时按 `reasoning` 字段拆 SSE；`OpenAIServingChat` 在 chat completions 流式与非流式路径都调用 parser。
- **配置**：由 [`10-config/reasoning-config.md`](../10-config/reasoning-config.md) 的 `ReasoningConfig.parser` 与 `reasoning_effort`/`enable_thinking` 等字段驱动；同时受 [`10-config/structured-outputs-config.md`](../10-config/structured-outputs-config.md) 与 reasoning_parser 字段影响。
- **采样**：parser 只识别**已生成**的思考段，是否允许继续思考由 [`06-sampling-decoding/thinking-budget.md`](../06-sampling-decoding/thinking-budget.md) 强制截断（生成 `force_index`）。
- **工具调用**：与 [`14-tokenizers-transformers/tool_parsers/`](tool_parsers/README.md) 串联——parser 先剥离思考，tool_parser 再从答复段提取工具调用。
- **detokenizer**：parser 输入是 `Detokenizer`（[`01-engine-core/detokenizer.md`](../01-engine-core/detokenizer.md)）产出的 `delta_text`，故必须处理 BPE 边界带来的字符乱码（用 `previous_token_ids` 反转义）。
- **模型库**：每个 parser 通常对应一类模型族，详见 [`04-model-zoo/architecture-families/deepseek.md`](../04-model-zoo/architecture-families/deepseek.md)、`qwen.md`、`glm.md`、`gemma.md`、`mistral.md` 等。

## 历史版本演进

- **v0.7 之前（早期）**：仅有一个简单 `reasoning_parser` 字符串切 `<think>` 的内置实现，未单独成包。
- **v0.7（v1 化）**：`vllm/reasoning/` 独立成包，引入 `abs_reasoning_parsers.py::ReasoningParser` ABC 与 `ReasoningParserManager` 工厂；DeepSeek-R1 / Qwen3 首版接入。
- **v0.8–v0.9**：扩族接入 GLM、Granite、Cohere、Step3、Ernie4.5、Olmo3、Nemotron V3、Hunyuan、HY-V3、Seed-OSS、GPT-OSS 等；流式识别完整化（处理跨 delta 部分标签）。
- **v0.10**：引入 `Responses API`（[`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md)），reasoning parser 直接服务该 API 的 `reasoning` 输出字段；Mistral tekken 兼容路径落地。
- **v0.10.x–v0.11**：与 `thinking_budget` 联动；DeepSeek V4 / MiniMax M3 引擎内置 parser 接入；`gemma4_utils.py` 抽出共享工具。
- **v0.12 / main**：parser 数量扩展到 ~25 个；`poolside_v1_reasoning_parser.py` 等长尾厂商接入；与 MCP 工具调用伦理（[`13-entrypoints/mcp/`](../13-entrypoints/mcp/README.md)）协同细化——具体归属（待核实）。

[← 返回分词与转换器首页](../README.md)

## 参见

- [`tool_parsers/`](tool_parsers/README.md)：工具调用解析，常与 reasoning parser 链式使用。
- [`transformers_utils/config.md`](transformers_utils/config.md)：模型 config 中 `reasoning_parser` 字段来源。
- [`06-sampling-decoding/thinking-budget.md`](../06-sampling-decoding/thinking-budget.md)：思考段输出长度上限。
- [`10-config/reasoning-config.md`](../10-config/reasoning-config.md)：`ReasoningConfig`。
- [`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md)：Responses API 的 reasoning 字段。
