# StructuredOutputsConfig（structured_outputs.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > StructuredOutputsConfig

源码：`vllm/config/structured_outputs.py`（约 74 行）。`StructuredOutputsConfig` 描述结构化输出（JSON schema / regex 等）的后端选择与行为开关，并承载 reasoning parser 的环境配置（历史字段，与 `ReasoningConfig` 部分重叠）。它是 `VllmConfig.structured_outputs_config`，被 `vllm/v1/structured_outputs/` 的 `StructuredOutputManager` 与 `EngineCore.step` 的 grammar bitmask 生成消费。

## 是什么

`@config` 装饰（`structured_outputs.py:17`）。`StructuredOutputsBackend = Literal["auto","xgrammar","guidance","outlines","lm-format-enforcer"]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `backend` | `"auto"` | 结构化输出后端；`auto` 按请求内容与库支持做选择（每版本可能变） |
| `disable_any_whitespace` | `False` | JSON 输出强制紧凑无空白（仅 `xgrammar`/`guidance` 支持） |
| `disable_additional_properties` | `False` | `guidance` 不用 `additionalProperties`（对齐 `outlines`/`xgrammar` 行为） |
| `reasoning_parser` | `""` | reasoning parser 名（按模型选，解析 reasoning content 到 OpenAI 格式） |
| `reasoning_parser_plugin` | `""` | 动态加载注册的 reasoning parser 插件路径 |
| `enable_in_reasoning` | `False` | reasoning 时是否用结构化输入 |

校验（`_validate_structured_output_config`）：`disable_any_whitespace` 仅 `xgrammar`/`guidance`；`disable_additional_properties` 仅 `guidance`。

`compute_hash`：空 factors——结构化输出后端在采样层做事（grammar bitmask），不改前向图形状。

> 注意：`reasoning_parser`/`reasoning_parser_plugin`/`enable_in_reasoning` 字段在历史上有过渡重叠，`ReasoningConfig`（[reasoning-config.md](reasoning-config.md)）是 reasoning 专用配置，二者关系 `(待核实)`，可能 `StructuredOutputsConfig.reasoning_*` 是旧路径保留兼容。

## 为什么

- **多后端竞争**：xgrammar/guidance/outlines/lm-format-enforcer 各有优势（速度、JSON schema 覆盖、regex 支持）。`auto` 让 vLLM 按请求内容（JSON vs regex）与当前库支持做选择，避免用户了解各库细节。
- **JSON 紧凑化**：`disable_any_whitespace` 强制 JSON 无空白，减 token；仅 xgrammar/guidance 实现该能力。
- **`additionalProperties` 对齐**：`guidance` 默认用 `additionalProperties`，与 outlines/xgrammar 行为不一致，`disable_additional_properties` 让其与后两者对齐，便于跨后端迁移。
- **reasoning parser 过渡**：`reasoning_parser`/`reasoning_parser_plugin` 是把 reasoning content 解析到 OpenAI API 格式的工具，与 `ReasoningConfig`（token 边界）协同：parser 提供起止字符串，`ReasoningConfig` 把它们 tokenize 为 token IDs 供调度器/输出处理器识别。
- **`compute_hash` 空**：grammar bitmask 在 `EngineCore.step` 经 `get_grammar_bitmask` 动态生成，作用于采样 logits，不进前向编译图。

## 怎么做

- **选后端**：`--guided-decoding-backend xgrammar`（或 `--structured-outputs-config.backend`）。
- **紧凑 JSON**：`--structured-outputs-config.disable-any-whitespace`（须 xgrammar/guidance）。
- **reasoning parser**：`--reasoning-parser deepseek_r1`（按模型选）。
- **parser 插件**：`--reasoning-parser-plugin mypkg.parsers:MyParser`。

## 与其它模块/系统配合

- **结构化输出（[`06-sampling-decoding/`](../06-sampling-decoding/README.md) 与 `vllm/v1/structured_outputs/`）**：`backend` 驱动 `StructuredOutputManager` 选 grammar 引擎；`grammar.accept_tokens` 在 scheduler `update_from_output`；`get_grammar_bitmask` 在 `EngineCore.step`。
- **Scheduler（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 状态；`grammar_init` 在 `preprocess_add_request`；`should_advance` + `grammar.accept_tokens` 在 `update_from_output`。
- **ReasoningConfig（[reasoning-config.md](reasoning-config.md)）**：`reasoning_parser` 协同——parser 提供 `reasoning_start_str`/`reasoning_end_str`，`ReasoningConfig.initialize_token_ids` tokenize。
- **API server（[`13-entrypoints/`](../13-entrypoints/README.md)）**：`response_format`/`guided_json`/`guided_regex` 请求字段触发结构化输出；`reasoning_effort`/`thinking` 字段触发 reasoning。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`structured_outputs_config.compute_hash()`（空）进顶层哈希保持聚合完整。

## 历史版本演进

- **v0.6**：structured output 起步（outlines/guidance）；`StructuredOutputsConfig` 初版。
- **v0.7（v1 落地）**：v1 `StructuredOutputManager` + grammar bitmask；xgrammar 后端；`disable_any_whitespace`。
- **v0.8**：`lm-format-enforcer`；`disable_additional_properties`；`backend="auto"` 智能选择。
- **v0.9**：reasoning parser 集成（DeepSeek R1 等）；`reasoning_parser_plugin` 动态加载；`enable_in_reasoning`。
- **v0.10–main**：`reasoning_config`（`ReasoningConfig`）独立后，`structured_outputs_config.reasoning_*` 保留兼容（`(待核实)` 过渡关系）；V2 model runner 对 reasoning `thinking_token_budget` 的支持差异。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [reasoning-config.md](reasoning-config.md) — reasoning 专用配置，`reasoning_parser` 协同。
- [vllm-config.md](vllm-config.md) — `structured_outputs_config.compute_hash()` 进顶层哈希。
- [../06-sampling-decoding/](../06-sampling-decoding/README.md) — 结构化输出子系统消费方。
- [../01-engine-core/scheduler/scheduler.md](../01-engine-core/scheduler/scheduler.md) — grammar 生命周期钩子。
