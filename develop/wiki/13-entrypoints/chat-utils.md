[← Wiki 首页](../README.md) > [API 入口](README.md) > chat_utils

# chat_utils（Chat 消息与多模态解析）

> `vllm/entrypoints/chat_utils.py` 是 OpenAI/Anthropic 风格 chat 协议到 vLLM `EngineInput` 的"翻译层"：解析 messages 内容（文本/image/audio/video/url/PIL)、套用 chat template、收集多模态项交给 processor，同时提供同步与异步两条解析路径。

## 是什么

文件 ~1970 行，核心组件：

| 组件 | 位置 | 职责 |
|---|---|---|
| 类型定义 | `vllm/entrypoints/chat_utils.py:122` 起 | `AudioURL`、`ChatCompletionContentPartAudioParam`、`VideoURL`、`PILImage`、`CustomChatCompletionMessageParam`、`ConversationMessage` 等 TypedDict |
| `ChatTemplateConfig` | `vllm/entrypoints/chat_utils.py:1247` | chat template 来源（文件路径/字面串/字典） |
| `validate_chat_template` / `load_chat_template` | `:1253` / `:1335` | 校验 + 加载 template（支持 `file://`、单行字面量、JSON） |
| `BaseMultiModalItemTracker` | `:521` | 多模态项收集器抽象（Generic） |
| `MultiModalItemTracker` / `AsyncMultiModalItemTracker` | `:792` / `:817` | 同步/异步两种 tracker |
| `BaseMultiModalContentParser` | `:846` | 解析 content part 的抽象 |
| `MultiModalContentParser` / `AsyncMultiModalContentParser` | `:917` / `:1065` | 同步/异步解析器 |
| `parse_chat_messages` / `parse_chat_messages_async` | `:1863` / `:1902` | 入口：把 messages → (conversations, engine_prompts) |
| `_parse_chat_message_content_part` | `:1627` | 单个 content part 分派 |
| `make_tool_call_id` / `get_tool_call_id_type` | `:1964` / `:1953` | 工具调用 ID 生成与类型选择 |

`ConversationMessage`（`:363`）是脱壳后的内部消息表示，`{role, content, tool_calls, ...}`，供 chat template 渲染。

## 为什么

- **同步/异步双轨**：在线 HTTP 用 `parse_chat_messages_async`（异步下载 URL、异步 processor），离线 `LLM.chat` 用同步版。二者共享 `BaseMultiModalContentParser`，避免逻辑漂移。
- **TypedDict 兼容 OpenAI**：直接对齐 `openai.types.chat.*` 的 TypedDict，让 `extra="allow"` 的请求体可被静态检查；同时 `PILImage`（`:193`）支持直接传 PIL 对象（本地场景）。
- **embeds 快捷通道**：`ChatCompletionContentPartImageEmbedsParam`（`:136`）等类型允许直接传预计算 embedding，跳过媒体下载/编码，用于高频复用媒体。
- **chat template 三态**：`load_chat_template` 支持（1）HuggingFace repo 名、（2）`file://path`、（3）单行字面量、（4）JSON dict 多 template；`ChatTemplateResolutionError`（`:76`）统一报错。
- **token 占位符**：`_reject_reserved_placeholder_in_text`（`:1609`）防止用户文本里出现模型保留的多模态占位符（如 `<image>`）造成 template 错位。
- **postprocess**：`_postprocess_messages`（`:1820`）按 model_config 收尾（如合并连续 user、补 system），与 HuggingFace `apply_chat_template` 行为对齐。

## 怎么做

### 调用入口

serving 层通常这样用：

```python
conversations, engine_prompts = await parse_chat_messages_async(
    request.messages, model_config, tokenizer,
    content_format, mm_parser_factory,
)
```

返回 `(list[ConversationMessage], list[EngineInput])`，`EngineInput` 含 `prompt_token_ids`、`mm_kwargs`、`mm_placeholders`。

### 加载 chat template

```python
tpl = load_chat_template(args.chat_template)  # str 或 None
```

若 `chat_template` 形如 `file:///a/b.jinja` 则读文件；若是 `{"name_a": "...", "name_b": "..."}` JSON 则存为字典，按 `request.chat_template` 选。

### 多模态项收集

`MultiModalItemTracker` 由 `BaseMultiModalItemTracker` 派生，`_parse_chat_message_content_mm_part`（`:1454`）按 `part["type"]` 分派到 image_url/audio_url/video_url/embeds，调用 tracker 收集；最终由 processor 一次性吃下。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| 同步解析入口 | `vllm/entrypoints/chat_utils.py:1863` |
| 异步解析入口 | `vllm/entrypoints/chat_utils.py:1902` |
| content part 分派 | `vllm/entrypoints/chat_utils.py:1627` |
| parts 聚合 | `vllm/entrypoints/chat_utils.py:1567` |
| 交叉文本 prompt | `vllm/entrypoints/chat_utils.py:1343` |
| 全量 MM 文本 prompt | `vllm/entrypoints/chat_utils.py:1355` |
| template 校验 | `vllm/entrypoints/chat_utils.py:1253` |
| template 加载 | `vllm/entrypoints/chat_utils.py:1335` |
| 工具调用计数 | `vllm/entrypoints/chat_utils.py:1941` |
| tool_call_id 生成 | `vllm/entrypoints/chat_utils.py:1964` |

## 与其它模块/系统配合

- [openai/chat-completion.md](openai/chat-completion.md)：`OpenAIServingChat` 通过 `OnlineRenderer.preprocess_chat` 间接调用本模块。
- [多模态](../11-multimodal/README.md)：tracker 收集的媒体项交由 MM processor/cache；`_detect_field`（`:417`）用 `MultiModalSharedField` 描述跨 modality 共享字段。
- [tokenizers-transformers-renderers](../14-tokenizers-transformers/README.md)：`ChatTemplateContentFormatOption`（`string`/`openai`）控制 content 渲染形式，与 `renderers/` 配合。
- [llm.md](llm.md)：`LLM.__init__` 调 `load_chat_template`。
- [openai/responses.md](openai/responses.md)：Responses API 的 `construct_input_messages`（`responses/utils.py:158`）也产出 `ChatCompletionMessageParam` 走本模块。

## 历史版本演进

- **v0.5–v0.6（OpenAI 对齐）**：初版 `parse_chat_messages`，支持 text/image_url；`ChatCompletionMessageParam` TypedDict 落地。
- **v0.7（多模态扩展）**：加 audio、video、PIL image、embeds 类型；`BaseMultiModalItemTracker` 抽象出同步/异步双轨。
- **v0.8（chat template 三态）**：`load_chat_template` 支持 `file://` 与字面量；`ChatTemplateConfig` dataclass 化。
- **v0.10（harmony/responses）**：`CustomChatCompletionContentToolReferenceParam`（`:287`）支持 Responses 的 tool reference；为 harmony 渲染预留 `parse_chat_input_to_harmony_message`（在 `parser/harmony_utils.py`）。
- **v0.11/main**：`allowed_media_domains` 安全校验；`_reject_reserved_placeholder_in_text` 防占位符冲突；`get_tool_call_id_type` 按模型配置选 `uuid`/`int`/`random`；异步 parser 用 `AsyncMultiModalItemTracker` 与线程池下载器配合。

## 参见

- [← 返回 API 入口首页](README.md)
- [openai/chat-completion.md](openai/chat-completion.md)
- [多模态](../11-multimodal/README.md)
- [tokenizers-transformers](../14-tokenizers-transformers/README.md)
