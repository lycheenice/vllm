[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > cli_args

# cli_args.py（serve 参数定义）

> `vllm/entrypoints/openai/cli_args.py` 集中定义 `vllm serve` 的全部命令行参数与校验，供 `ServeSubcommand`/`run_batch`/`launch` 复用，并提供文档专用 parser。它用 `@config` 装饰的 dataclass + `add_cli_args` 类方法避免手写 `add_argument` 重复。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `LoRAParserAction` | `vllm/entrypoints/openai/cli_args.py:32` | argparse Action：把 `--lora-modules` 值解析成 `list[LoRAModulePath]`（兼容 `name=path` 与 JSON） |
| `BaseFrontendArgs` | `:67` | `@config` dataclass：前端通用参数（lora/chat_template/tool/reasoning/parser/middleware/ssl/cors...） |
| `FrontendArgs` | `:224` | `BaseFrontendArgs` 子类，补 host/port/uds/ssl 等 server-only 参数 |
| `make_arg_parser` | `:339` | 组装 `model_tag`/`--headless`/`--api-server-count`/`--config`/`--grpc` + `FrontendArgs.add_cli_args` + `AsyncEngineArgs.add_cli_args` |
| `validate_parsed_serve_args` | `:386` | 启动前校验：chat template、tool/reasoning parser、log outputs、per-request metrics、multi-port LB |
| `create_parser_for_docs` | `:416` | 文档生成用纯净 parser |

`BaseFrontendArgs`（`:67`）字段大类：

- **LoRA/适配器**：`lora_modules`、`enable_auto_tool_choice`、`tool_call_parser`、`exclude_tools_when_tool_choice_none`、`tool_parser_plugin`。
- **chat 渲染**：`chat_template`、`chat_template_content_format`、`trust_request_chat_template`、`default_chat_template_kwargs`。
- **reasoning**：`reasoning_parser`、`reasoning_parser_plugin`（经 `structured_outputs_config`）。
- **日志/指标**：`enable_log_requests`、`enable_log_outputs`、`max_log_len`、`disable_log_stats`、`enable_per_request_metrics`、`enable_server_load_tracking`。
- **token/usage**：`return_tokens_as_token_ids`、`enable_prompt_tokens_details`、`enable_force_include_usage`。
- **HTTP/h11**：`disable_uvicorn_access_log`、`uvicorn_log_level`、`h11_max_incomplete_event_size`、`h11_max_header_count`、`enable_ssl_refresh`。
- **middleware/docs**：`middleware`、`allowed_origins/credentials/methods/headers`、`disable_fastapi_docs`、`enable_offline_docs`、`root_path`、`api_key`。

`FrontendArgs`（`:224`）补 `host`/`port`/`uds`/`ssl_*`/`enable_request_id_headers`，并 `_customize_cli_kwargs`（`:308`）做参数分组/重命名。

## 为什么

- **dataclass + `@config` 单一来源**：`@config`（`vllm.config`）让 dataclass 字段自动生成 `--xxx` argparse 参数与帮助文本（取自 docstring），`add_cli_args`（`:198`）遍历字段调 `parser.add_argument`。改字段即改 CLI，避免漂移。
- **`_customize_cli_kwargs`**：子类可覆盖字段在 argparse 中的呈现（分组、别名、`action`），`BaseFrontendArgs` 与 `FrontendArgs` 分别注册不同分组。
- **LoRA 双格式**：`LoRAParserAction`（`:32`）兼容老式 `name=path` 与新式 JSON/JSON-list（含 `base_model_name`/`is_3d_lora_weight`），方便脚本 与 K8s 配置共存。
- **校验前置**：`validate_parsed_serve_args`（`:386`）在引擎构造前 raise，把"必须 `--tool-call-parser` 才能开 auto tool"等规则早暴露；multi-port LB 则委托 `dp_supervisor.validate_multi_port_external_lb_args`。
- **文档专用 parser**：`create_parser_for_docs` 给 Sphinx/`gen_docs` 用，去掉运行期副作用。

## 怎么做

### 参数分组

`make_arg_parser`（`:339`）注册顶层 `model_tag`（位置可选）、`--headless`、`--api-server-count`/`-asc`、`--config`、`--grpc`，再依次调 `FrontendArgs.add_cli_args(parser)` 与 `AsyncEngineArgs.add_cli_args(parser)`。`--config` 指 YAML 配置文件（见 `docs/usage/configuration/serve_args`），由 `FlexibleArgumentParser` 合并。

### 用 `--help=<ConfigGroup>` 探索

SPDX 注释提示可按分组检索（如 `--help=ModelConfig`、`--help=Frontend`、`--help=all`），实现于 `FlexibleArgumentParser`（`vllm/utils/argparse_utils.py`，待核实细节）。

### 关键校验规则（`validate_parsed_serve_args`）

| 条件 | 错误 |
|---|---|
| `enable_auto_tool_choice and not tool_call_parser` | `--enable-auto-tool-choice requires --tool-call-parser` |
| `enable_log_outputs and not enable_log_requests` | `--enable-log-outputs requires --enable-log-requests` |
| `enable_per_request_metrics and disable_log_stats` | 需要 stats logging |
| `data_parallel_multi_port_external_lb` | 委托 `validate_multi_port_external_lb_args` |
| `chat_template` 不可解析 | `validate_chat_template` raise |

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| LoRA action | `vllm/entrypoints/openai/cli_args.py:32` |
| BaseFrontendArgs | `vllm/entrypoints/openai/cli_args.py:67` |
| add_cli_args 基类 | `vllm/entrypoints/openai/cli_args.py:198` |
| FrontendArgs | `vllm/entrypoints/openai/cli_args.py:224` |
| make_arg_parser | `vllm/entrypoints/openai/cli_args.py:339` |
| validate_parsed_serve_args | `vllm/entrypoints/openai/cli_args.py:386` |
| create_parser_for_docs | `vllm/entrypoints/openai/cli_args.py:416` |

## 与其它模块/系统配合

- [api-server.md](api-server.md)：`make_arg_parser`/`validate_parsed_serve_args` 在 `__main__` 与 `ServeSubcommand` 用。
- [cli/serve-cmd.md](../cli/serve-cmd.md)：`ServeSubcommand.subparser_init` 调 `make_arg_parser`。
- [cli/launch-cmd.md](../cli/launch-cmd.md)：`LaunchSubcommandBase.add_cli_args` 调 `make_arg_parser`。
- [run-batch.md](run-batch.md)：`run_batch` 复用 `BaseFrontendArgs` 部分参数。
- [chat-utils.md](../chat-utils.md)：`validate_chat_template`/`ChatTemplateContentFormatOption`。
- [LoRA](../../12-lora/README.md)：`LoRAModulePath`、`is_3d_lora_weight`。
- [配置-scheduler](../../10-config/scheduler-config.md)：`AsyncEngineArgs` 字段最终进 `VllmConfig`。

## 历史版本演进

- **v0.5–v0.6（手写 argparse）**：所有 `--xxx` 散落在 `api_server.py`，重复维护。
- **v0.7–v0.8（抽 cli_args）**：独立文件 + `make_arg_parser`；加 `--ssl-*`/`--middleware`/`--api-key`。
- **v0.9（@config dataclass）**：引入 `BaseFrontendArgs`/`FrontendArgs` dataclass 与 `add_cli_args`，参数文档化注释统一；`LoRAParserAction` 支持新 JSON 格式。
- **v0.10（multi-port & grpc）**：`--grpc`、`--data-parallel-multi-port-external-lb` 校验委托 dp_supervisor；`--config` YAML 支持。
- **v0.11/main**：`--api-server-count`/`-asc` 别名；`--headless` 走 headless 分支；`h11_max_*` 头部限制；`enable_request_id_headers`；`reasoning_parser` 经 `structured_outputs_config` 路径（待核实完整字段集）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [api-server.md](api-server.md)
- [cli/serve-cmd.md](../cli/serve-cmd.md)
- [dp-supervisor.md](dp-supervisor.md)
- [LoRA](../../12-lora/README.md)
