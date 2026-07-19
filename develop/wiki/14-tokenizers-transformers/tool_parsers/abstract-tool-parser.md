[← 分词与转换器](../README.md) > [工具调用解析](README.md) > ToolParser ABC

# abstract_tool_parser.py — ToolParser 与注册表

## 是什么

`vllm/tool_parsers/abstract_tool_parser.py`（378 行）定义工具调用解析的抽象基类与注册表。

### `ToolParser`（`:43`）

抽象基类，所有具体 parser 继承它。重要类属性：

- `supports_required_and_named: bool = True`（`:59`）：默认走标准 JSON-based 解析处理 `tool_choice="required"` 与命名 function；子类置 False 时退化为 `auto` 路径（如 GLM XML 类）。
- `structural_tag_model: str | None = None`（`:62`）：xgrammar builtin 模板 key（如 `"hermes"`/`"llama"`/`"deepseek_v3_2"`）。
- `engine_based_streaming: bool = False`（`:63`）：声明使用 Streaming Parser Engine 框架（v0.12/main），由 `DelegatingParser` 据此切换 stream state 行为。

`__init_subclass__`（`:65`）：声明 `structural_tag_model` 且 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 时，自动把 `supports_required_and_named = False`。

实例状态（`:73`）：

- `prev_tool_call_arr: list[dict]` — 已解析出的 tool call 数组（用于差分流式）
- `current_tool_id` / `current_tool_name_sent` / `streamed_args_for_tool` — 流式状态机
- `model_tokenizer`、`tools`（仅保留 `ChatCompletionToolsParam`/`FunctionTool`）

核心方法：

- `adjust_request(request)`（`:119`）：按 tool schema 注入 `StructuredOutputsParams(json=...)` 或 `ResponseTextConfig(format=json_schema)`；若 locked `structural_tag_model` 则不重置（由 `DelegatingParser._apply_structural_tag` 统一处理）。
- `get_structural_tag(request, *, reasoning=False)`（`:168`）：当 `structural_tag_model` 与 `VLLM_ENFORCE_STRICT_TOOL_CALLING` 同时开启，调 `structural_tag_registry.get_model_structural_tag` 构造 `StructuralTag`。
- `extract_tool_calls(model_output, request) -> ExtractedToolCallInformation`（`:187`）：非流式抽取，返回 `tools_called`/`tool_calls`/`content`。ABC 强制子类实现。
- `extract_tool_calls_streaming(previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids, request) -> DeltaMessage | None`（`:201`）：流式增量抽取，需实例方法（保留状态）。ABC 强制实现。
- `get_remaining_unstreamed_args() -> str`（`:94`）：返回还没流出的 tool args，在 `DelegatingParser.finalize_generation` 中被 flush。
- `cached_property vocab`（`:113`）：`tokenizer.get_vocab()`。

### `ToolParserManager`（`:223`）

注册表，eager + lazy 双模：

- `tool_parsers: dict[str, type[ToolParser]]`、`lazy_parsers: dict[str, tuple[module_path, class_name]]`
- `get_tool_parser(name)`（`:235`）：先 eager 后 lazy；lazy 首次访问 import + issubclass 校验 + 缓存。
- `register_module(name=None, force=True, module=None)`（`:317`）：装饰器/直传二选一；装饰器用法只注册 lazy 映射（不 import），与 `register_lazy_module` 一致。
- `register_lazy_module(name, module_path, class_name)`（`:303`）。
- `_register_module(module, module_name, force)`（`:274`）：eager 注册，校验 `issubclass ToolParser`。
- `list_registered()`（`:363`）：返回所有名（用于错误信息）。
- `import_tool_parser(plugin_path)`（`:368`）：用户自定义 parser 文件路径加载（`import_from_path`），失败仅 logger.exception 不抛错。

## 为什么

- **eager + lazy 共存**：约 45 个 parser 总和代码量很大，启动期全部 import 会显著拖慢；lazy 让"用不到的不 import"。同时保留 eager 注册给第三方插件。
- **`supports_required_and_named` 双路径**：标准 JSON-based 强约束路径（xgrammar guided decoding 喂合法 JSON）只对部分模型可行；其余模型必须用 `auto` 风格 parser 解析自由文本。该开关让 serving 层自动选路。
- **`structural_tag_model`**：vLLM 在 v0.12/main 与 xgrammar 协同引入"内建 structural tag 模板"，避免每个模型都自己写 grammar；parser 只需声明家族名，`structural_tag_registry` 负责 translate。
- **`engine_based_streaming`**：与 Streaming Parser Engine 协同——声明为 True 的 parser 不再使用 `parse_delta` 的累积文本状态机（`StreamState.engine_based=True`，`advance/commit` 退化为直通），改由引擎化 parser 自管缓冲。
- **可 pickle**：parser 实例中 `model_tokenizer` 可能是 `MistralTokenizer`/`CachedHfTokenizer`，注册表存的是类不是实例，跨进程安全。

## 怎么做

写一个新 parser 的最简模板：

```python
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

@ToolParserManager.register_module("my_parser")
class MyToolParser(ToolParser):
    structural_tag_model = "my_format"  # 若有 xgrammar 内建模板

    def extract_tool_calls(self, model_output, request):
        ...
        return ExtractedToolCallInformation(tools_called=..., tool_calls=[...], content=...)

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, ...):
        ...
        return DeltaMessage(tool_calls=[DeltaToolCall(...)]) or None
```

或在 `__init__.py` 的 `_TOOL_PARSERS_TO_REGISTER` 加一条 `"name": ("filename", "ClassName")`，与 `register_lazy_tool_parsers()` 总注册配合。

`extract_tool_calls_streaming` 的"差分"模式几乎都是同一思路：每步用最新 `current_text` 重新解析全部已生成内容，与 `prev_tool_call_arr`/`streamed_args_for_tool` 比对，仅推送新增部分。这是为了避免 cleanup 算法把已推送 delta 反转。

## 与其它模块/系统配合

- **`vllm/parser/abstract_parser.py.DelegatingParser`**：把 `ReasoningParser` 与 `ToolParser` 组合进统一 `Parser`；调用 `tool_parser.adjust_request`、`extract_tool_calls`、`extract_tool_calls_streaming`，根据 `supports_required_and_named` 路由（见 `vllm/parser/abstract_parser.py:417`）。
- **`vllm/parser/parser_manager.py`**：`get_tool_parser(name, enable_auto_tools, model_name)` 包装 `ToolParserManager.get_tool_parser`，并对 Llama-3.2 pythonic 发 warning。
- **`structural_tag_registry.py`**：`get_model_structural_tag` 把声明转换成 `StructuralTag`，由 `DelegatingParser._apply_structural_tag` 写入 `request.structured_outputs`。
- **`OnlineRenderer`**（`vllm/renderers/online_renderer.py`）：启动期通过 `ParserManager.get_parser(tool_parser_name=..., reasoning_parser_name=..., ...)` 拿到组合 Parser 类；运行期 `parser(...).adjust_request(request)` 把 tool schema 落到 `structured_outputs`。
- **`OnlineDerenderer`**（`vllm/renderers/online_derenderer.py`）：反向路径，调 `parser.parse(decoded_text, request, enable_auto_tools=...)` 抽取 tool_calls。
- **`vllm/entrypoints/openai/engine/protocol.py`**：定义 `ExtractedToolCallInformation`、`ToolCall`、`DeltaToolCall`、`FunctionCall`、`DeltaFunctionCall` 等 dataclass。

## 历史版本演进

- **v0.5（PR #5649 "OpenAI-Compatible Tools API + Streaming for Hermes & Mistral models"）**：`ToolParser` ABC 与 `ToolParserManager` 诞生；首批 Hermes/Mistral。
- **v0.6（PR #8343 "Llama 3.1 and 3.2 tool use"）**：Llama 系 parser 加入；`supports_required_and_named` 概念逐步成型。
- **v0.7–v0.9**：持续扩充；eager 注册逐步迁到 lazy（`register_lazy_tool_parsers`）以缩短启动。
- **v0.10/v0.11**：`structural_tag_model` 与 xgrammar 协同引入；`engine_based_streaming` 类属性出现。
- **v0.12/main（PR #45413/#45588/#45755/#45877/#45915/#46314）**：Streaming Parser Engine 框架落地，DeepSeek/Nemotron/Gemma4/Qwen3/seed_oss/GLM4.7 等被 port；`VLLM_ENFORCE_STRICT_TOOL_CALLING` 与 `__init_subclass__` 自动降级 `supports_required_and_named`。
- **main**：`import_tool_parser` 插件路径加载；注册名集合突破 45 项。

---

[← 返回工具调用解析首页](README.md)

## 参见

- `hermes.md` / [mistral.md](mistral.md) / `pythonic.md` / `granite.md` / `granite-20b-fc.md` / `llama3-json.md`
- `structural-tag-registry.md` / `utils.md` / `streaming.md`
- ../reasoning.md — 与 ToolParser 对偶的 ReasoningParser。
