# ReasoningConfig（reasoning.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ReasoningConfig

源码：`vllm/config/reasoning.py`（约 107 行）。`ReasoningConfig` 描述推理模型（如 DeepSeek R1、Qwen3 thinking）的 reasoning 块 token 边界：起始/结束字符串、parser 名、以及经 tokenizer 自动派生的 token IDs。它是 `VllmConfig.reasoning_config`（`None` 表示未启用），被 `VllmConfig.__post_init`（调 `initialize_token_ids`）、输出处理器与调度器消费。

## 是什么

`@config` 装饰（`reasoning.py:12`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `reasoning_parser` | `""` | `ReasoningParserManager` 中注册的 parser 名（如 `deepseek_r1`） |
| `reasoning_start_str` | `""` | reasoning 块起始字符串（如 `"</think>"`） |

Private 派生字段（`init=False, repr=False`）：`_reasoning_start_token_ids`/`_reasoning_end_token_ids`/`_enabled`。

属性：`enabled`（token IDs 已初始化）、`reasoning_start_token_ids`/`reasoning_end_token_ids`（只读 property）。

方法 `initialize_token_ids(model_config)`（`reasoning.py:62`）：
1. 已初始化则直接返回。
2. 从 `cached_tokenizer_from_config(model_config)` 取分词器。
3. 若 `reasoning_parser` 设但 start/end 字符串未全设，经 `ReasoningParserManager.get_reasoning_parser` 取 parser 实例的 `reasoning_start_str`/`reasoning_end_str` 补全。
4. 字符串仍空 → 返回（`_enabled` 保持 `False`）。
5. `tokenizer.encode(str, add_special_tokens=False)` 得 token IDs，存私有字段，`_enabled=True`。
6. encode 结果空则 raise。

`compute_hash`：未定义（`ReasoningConfig` 无 `compute_hash`，故 `VllmConfig.compute_hash` 中**未**纳入 `reasoning_config`）——reasoning token 边界在采样/输出层做事，不改前向图形状。

> 注意：本配置与 `StructuredOutputsConfig.reasoning_parser`（[structured-outputs-config.md](structured-outputs-config.md)）有历史重叠，过渡关系 `(待核实)`。当前 `VllmConfig.reasoning_config` 是 reasoning 专用配置。

## 为什么

- **token 边界自动派生**：用户只需给 `reasoning_start_str`/`reasoning_end_str`（或仅给 `reasoning_parser` 让 parser 提供字符串），`initialize_token_ids` 经 tokenizer 自动得 token IDs，避免用户手填易错的 token ID。
- **parser 协同**：`reasoning_parser` 名查 `ReasoningParserManager`，parser 类提供默认 start/end 字符串 + 把 reasoning content 解析到 OpenAI API 格式。故仅给 parser 名也能初始化。
- **lazy 初始化**：`initialize_token_ids` 在 `VllmConfig.__post_init` 调（需 `model_config`），若 token IDs 派生失败（字符串无效）则 `enabled=False` 并打 warning，不阻断启动。
- **调度器/输出处理器消费**：`reasoning_start_token_ids`/`reasoning_end_token_ids` 让输出处理器识别 reasoning 块边界，分离 reasoning content 与 final answer，按 OpenAI reasoning 格式返回。调度器可据 thinking budget 控制推理 token 数。
- **不进哈希**：reasoning 边界是输出层行为，不改前向图，故 `VllmConfig.compute_hash` 不纳入（与 `structured_outputs_config`/`kv_events_config` 等输出层配置一致）。

## 怎么做

- **字符串**：`--reasoning-config '{"reasoning_start_str":"\n","reasoning_end_str":""}'`。
- **parser**：`--reasoning-parser deepseek_r1`（parser 提供字符串）。
- **parser 插件**：配合 `--structured-outputs-config.reasoning-parser-plugin mypkg:MyParser` 动态注册。

## 与其它模块/系统配合

- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init` 末段 `reasoning_config.initialize_token_ids(model_config)`；失败 warning。`_validate_v2_model_runner` 对 V2 + reasoning 的 `thinking_token_budget` 请求参数打 warning（V2 暂不支持）。
- **输出处理器（[`01-engine-core/output-processor.md`](../01-engine-core/output-processor.md)）**：`reasoning_start/end_token_ids` 识别 reasoning 块，分离 reasoning content 与 final answer。
- **调度器（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：thinking budget 控制推理 token 数上限。
- **StructuredOutputsConfig（[structured-outputs-config.md](structured-outputs-config.md)）**：`reasoning_parser`/`reasoning_parser_plugin`/`enable_in_reasoning` 字段重叠，`ReasoningConfig` 是较新的 reasoning 专用配置。
- **分词器（[`14-tokenizers-transformers/`](../14-tokenizers-transformers/README.md)）**：`cached_tokenizer_from_config(model_config)` 取缓存分词器 encode 字符串。
- **API server（[`13-entrypoints/`](../13-entrypoints/README.md)）**：`reasoning_effort`/`thinking` 请求字段触发 reasoning 行为；OpenAI reasoning content 格式由 parser 产出。

## 历史版本演进

- **v0.9/v0.10**：reasoning parser 集成（DeepSeek R1 等）；`reasoning_parser`/`reasoning_start_str`/`reasoning_end_str` 字段初在 `StructuredOutputsConfig`。
- **v0.10/v0.11（待核实）**：`ReasoningConfig` 独立为 `vllm/config/reasoning.py`；`initialize_token_ids` 自动派生 token IDs；`enabled`/`reasoning_*_token_ids` property。
- **v0.12 / main**：V2 model runner 对 `thinking_token_budget` 请求参数支持差异（V2 暂不支持，打 warning）；thinking budget 调度器侧支持。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [structured-outputs-config.md](structured-outputs-config.md) — `reasoning_parser` 字段重叠关系。
- [vllm-config.md](vllm-config.md) — `initialize_token_ids` 调用与 V2 warning。
- [../01-engine-core/output-processor.md](../01-engine-core/output-processor.md) — token 边界消费方。
- [../14-tokenizers-transformers/](../14-tokenizers-transformers/README.md) — `cached_tokenizer_from_config`。
