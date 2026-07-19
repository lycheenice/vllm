[← 分词与转换器](../README.md) > 工具调用解析

# tool_parsers (vllm/tool_parsers/)

## 是什么

`vllm/tool_parsers/` 是 vLLM 的工具调用（function calling）解析子系统。它把模型生成的自由文本/特殊 token流拆解为 OpenAI 兼容的 `ToolCall`/`DeltaToolCall`，统一喂回 serving 层（chat completion / responses API）。

抽象层与注册表：

- [`abstract_tool_parser.py`](abstract-tool-parser.md)：`ToolParser` ABC + `ToolParserManager` 注册表（eager+lazy）。
- `__init__.py`：`_TOOL_PARSERS_TO_REGISTER` 字典（约 45 个名字→文件→类），`register_lazy_tool_parsers()` 启动期注册。

按厂商/格式分组的实存 parser（约 45 个注册名 / 约 40 个文件）：

| 家族 | 文件 | 注册名 | 详文 |
|---|---|---|---|
| Hermes | `hermes_tool_parser.py` | `hermes` | `hermes.md` |
| Mistral | `mistral_tool_parser.py` | `mistral` | [mistral.md](mistral.md) |
| Llama 3.x/4 JSON | `llama_tool_parser.py` | `llama3_json`、`llama4_json` | `llama3-json.md` |
| Llama 4 Pythonic | `llama4_pythonic_tool_parser.py` | `llama4_pythonic` | — |
| Pythonic | `pythonic_tool_parser.py` | `pythonic` | `pythonic.md` |
| Granite 3.0/3.1 | `granite_tool_parser.py` | `granite` | `granite.md` |
| Granite-20b-FC（async） | `granite_20b_fc_tool_parser.py` | `granite-20b-fc` | `granite-20b-fc.md` |
| Granite4 | `granite4_tool_parser.py` | `granite4` | — |
| DeepSeek V3/V3.1 | `deepseekv3_tool_parser.py`/`deepseekv31_tool_parser.py` | `deepseek_v3`/`deepseek_v31` | — |
| DeepSeek V3.2/V4 (engine) | `deepseekv32_engine_tool_parser.py`/`deepseekv4_engine_tool_parser.py` | `deepseek_v32`/`deepseek_v4` | — |
| Cohere Command 3/4 | `cohere_command_tool_parser.py` | `cohere_command3`/`cohere_command4` | — |
| Ernie45 | `ernie45_tool_parser.py` | `ernie45` | — |
| GLM 4.5/4.7 | `glm47_moe_tool_parser.py` | `glm45`/`glm47` | — |
| Hunyuan A13B | `hunyuan_a13b_tool_parser.py` | `hunyuan_a13b` | — |
| HY v3 | `hy_v3_tool_parser.py` | `hy_v3` | — |
| InternLM 2 | `internlm2_tool_parser.py` | `internlm` | — |
| Jamba | `jamba_tool_parser.py` | `jamba` | — |
| Lfm2 | `lfm2_tool_parser.py` | `lfm2` | — |
| Kimi K2 | `kimi_k2_tool_parser.py` | `kimi_k2` | — |
| Longcat | `longcat_tool_parser.py` | `longcat` | — |
| MiMo (Qwen3 引擎子类) | `qwen3_engine_tool_parser.py` | `mimo` | — |
| MiniMax M2/M3 | `minimax_m2_tool_parser.py`/`minimax_m3_tool_parser.py` | `minimax_m2`/`minimax_m3` | — |
| MiniCPM5 XML | `minicpm5xml_tool_parser.py` | `minicpm5` | — |
| Olmo3 Pythonic | `olmo3_tool_parser.py` | `olmo3` | — |
| GPT-OSS (Harmony) | `gptoss_tool_parser.py` | `openai` | — |
| Phi4 Mini JSON | `phi4mini_tool_parser.py` | `phi4_mini_json` | — |
| Seed OSS (engine) | `seed_oss_engine_tool_parser.py` | `seed_oss` | — |
| Step3/3p5 | `step3_tool_parser.py`/`step3p5_tool_parser.py` | `step3`/`step3p5` | — |
| XLAM | `xlam_tool_parser.py` | `xlam` | — |
| GigaChat3 | `gigachat3_tool_parser.py` | `gigachat3` | — |
| FunctionGemma | `functiongemma_tool_parser.py` | `functiongemma` | — |
| Gemma4 (engine) | `gemma4_engine_tool_parser.py` | `gemma4` | — |
| Apertus | `apertus_tool_parser.py` | `apertus` | — |
| Qwen3 Coder/XML (engine) | `qwen3_engine_tool_parser.py` | `qwen3_coder`/`qwen3_xml` | — |
| Poolside v1 | `poolside_v1_tool_parser.py` | `poolside_v1` | — |
| Rust | `rust_tool_parser.py` | （未在 init 注册，待核实） | — |
| Apertus | 同上 | — |

辅助模块：

- `utils.py`：`Tool` alias、`partial_json_loads`、`is_complete_json`、`find_common_prefix`、`partial_tag_overlap`、`extract_intermediate_diff`、`handle_single_tool`/`make_valid_python`/`compute_tool_delta`（pythonic 专用）、`get_json_schema_from_tools`、Responses tool 命名空间工具。
- `streaming.py`：`extract_named_tool_call_streaming`、`extract_required_tool_call_streaming`、`filter_delta_text`（required tool choice 流式专用）。
- `structural_tag_registry.py`：xgrammar builtin structural tag 模板注册（`XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS`：`llama`/`kimi`/`deepseek_r1`/`deepseek_v3_1`/`qwen_3_5`/`qwen_3_coder`/`qwen_3`/`harmony`/`deepseek_v3_2`/`glm_4_7`/`deepseek_v4`）+ vLLM builtin（`hermes`、`minimax`）。

## 为什么

- 每家厂商的 function-call 文本格式不同（Hermes 标签对、Mistral `[ARGS]`、Llama `<|python_tag|>`+JSON、Granite JSON 数组、DeepSeek DSML、Harmony channel 等），需要可插拔解析层。
- 流式增量解析不能用简单 `decode([new_id])`，因为部分 JSON 的清理算法会反复回改已发送的 delta；必须维护"已发送状态"做差分。
- `tool_choice="required"`/命名 function 在结构化输出可引导时走 xgrammar 内建模板，让模型物理上不可生成非法格式；该路径由 `structural_tag_model` + `VLLM_ENFORCE_STRICT_TOOL_CALLING` 控制。
- v0.12/main 引入 Streaming Parser Engine 后，传统"基于文本差分"与"基于 token-id 引擎化"两种风格并存，后者声明 `engine_based_streaming=True` 走更精简路径。

## 怎么做

启动期：`register_lazy_tool_parsers()` 把字典里所有名字注册为 lazy 映射。

请求期（chat completion 为例）：

```mermaid
flowchart TD
  Req[ChatCompletionRequest with tools] --> PM[ParserManager.get_parser]
  PM --> TP[ToolParserManager.get_tool_parser name]
  TP -- lazy import --> Cls[具体 ToolParser 子类]
  Cls --> ParserInst[parser=DelegatingParser(tok, tools, ...)]
  ParserInst --> AR[adjust_request: 设置 structured_outputs/json_schema 或 structural_tag]
  AR --> Engine[引导解码]
  Engine --> Tokens[token ids 流]
  Tokens --> Stream[extract_tool_calls_streaming 差分]
  Stream --> Delta[DeltaMessage.tool_calls]
  Delta --> Resp[ChatCompletionResponse]
```

非流式：`parser.extract_tool_calls(content, request)` 一次性抽取（被 `DelegatingParser._extract_tool_calls` 调，按 `tool_choice`/`supports_required_and_named` 分支）。

## 与其它模块/系统配合

- **`vllm/parser/`**：`ParserManager.get_parser` 把 `ToolParser`+`ReasoningParser` 组合成 `DelegatingParser`（或 `HarmonyParser`/`MistralParser`），统一暴露 `parse`/`parse_delta` 给 serving 层。
- **`vllm/reasoning/`**：见 ../reasoning.md，对偶的 reasoning 抽取层。
- **`vllm/renderers/online_renderer.py`**：启动期持有 parser 类，请求期 `adjust_request`。
- **`vllm/renderers/online_derenderer.py`**：scale-out 反向路径调 `parser.parse(decoded_text, request, enable_auto_tools=...)`。
- **`vllm/sampling_params.StructuredOutputsParams`**：`adjust_request` 写入 `json`/`structural_tag` 字段，引导 [`06-sampling-decoding/structured-output/`](../../06-sampling-decoding/structured-output/README.md) 后端。
- **API 入口**：`../13-entrypoints/openai/chat-completion.md`、`../13-entrypoints/openai/responses.md`。

## 历史版本演进

| 版本 | 变更要点 |
|---|---|
| v0.5（PR #5649） | OpenAI 兼容工具 API + Hermes/Mistral 最初 streaming |
| v0.6（PR #8343/#8405） | Llama 3.1/3.2、InternLM2 加入 |
| v0.7–v0.9 | 持续扩充（Cohere、Jamba、Granite、Phi4、Step3、XLAM 等） |
| v0.10 | `register_lazy_tool_parsers` 完全替代 eager 注册；引入 `structural_tag_model` 概念雏形 |
| v0.11 | 大量模型专用 parser 加入（Kimi K2、MiniMax、Hunyuan、Poolside、Longcat、Apertus、GigaChat3 等），规模 ~45 |
| v0.12/main（PR #45413/#45588/#45755/#45877/#45915/#46314） | 引入 Streaming Parser Engine 框架；Qwen3/Nemotron/Gemma4/DeepSeek-V4/GLM4.7/seed_oss port；`engine_based_streaming=True` 路径成型 |
| main | `VLLM_ENFORCE_STRICT_TOOL_CALLING` + `__init_subclass__` 自动降级 `supports_required_and_named` |

---

[← 返回子系统首页](../README.md)

## 参见

- [abstract-tool-parser.md](abstract-tool-parser.md) — ABC 与注册表。
- `hermes.md` / [mistral.md](mistral.md) / `pythonic.md` / `granite.md` / `granite-20b-fc.md` / `llama3-json.md`
- `structural-tag-registry.md` / `utils.md` / `streaming.md`
- ../reasoning.md — 对偶的 reasoning 解析。
