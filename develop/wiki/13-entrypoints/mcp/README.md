[← Wiki 首页](../README.md) > [API 入口](../README.md) > MCP

# MCP（Model Context Protocol 工具执行）

> `vllm/entrypoints/mcp/` 让 vLLM（尤其 Responses API）能调用外部 MCP（Model Context Protocol）server 提供的工具，把"模型生成 + 工具调用 + 工具结果回灌"形成闭环。`Tool` 抽象 harmony 工具，`ToolServer`/`MCPToolServer` 管理远程 MCP server 连接与调用。

## 是什么

| 文件/类 | 位置 | 职责 |
|---|---|---|
| `tool.py: Tool` | `vllm/entrypoints/mcp/tool.py:49` | 工具 ABC |
| `tool.py: HarmonyBrowserTool` | `:59` | harmony 浏览器工具（recipient 路由） |
| `tool.py: HarmonyPythonTool` | `:101` | harmony Python 工具 |
| `tool.py: validate_gpt_oss_install` | `:25` | 校验 gpt-oss 环境 |
| `tool_server.py: list_server_and_tools` | `vllm/entrypoints/mcp/tool_server.py:18` | 列远程 MCP server 与工具 |
| `tool_server.py: trim_schema`/`post_process_tools_description` | `:31/56` | 工具 schema 精简/后处理 |
| `tool_server.py: ToolServer` | `:74` | 工具 server ABC |
| `tool_server.py: MCPToolServer` | `:102` | MCP 协议实现：连远程 MCP server、调工具、收结果 |
| `tool_server.py: DemoToolServer` | `:197` | 演示用本地工具 server |

`Tool`（`:49`）抽象一个工具的调用接口；`HarmonyBrowserTool`/`HarmonyPythonTool` 是 harmony 协议内工具（gpt-oss 系列），由 recipient 字段路由。

`MCPToolServer`（`:102`）是核心：

1. `list_server_and_tools(server_url)`（`:18`）从远程 MCP server 拉工具列表 + schema。
2. `trim_schema`（`:31`）/`post_process_tools_description`（`:56`）精简 schema（去超大字段、统一描述）。
3. 运行期：模型流式产出 tool call → `MCPToolServer` 调远程 MCP server 执行 → 结果作为 `response.output_item` 流回（与 [openai/responses.md](../openai/responses.md) 配合）。

`DemoToolServer`（`:197`）提供本地假工具，便于开发测试。

## 为什么

- **MCP 标准**：MCP 是 Anthropic 提出的工具协议，生态有大量现成 server（文件系统、浏览器、数据库…）；vLLM 通过 `MCPToolServer` 直接消费，免去逐个适配。
- **与 Responses API 闭环**：Responses API 的 `tools` 含 `MCP` 类型时，`_extract_allowed_tools_from_mcp_requests`（`openai/responses/serving.py:111`）抽 `allowed_tools`，`MCPToolServer` 实时调用，结果进 `response.mcp_call.*` 事件——形成"模型→工具→模型"多轮。
- **harmony 工具原生**：`HarmonyBrowserTool`/`HarmonyPythonTool` 让 harmony 模型（gpt-oss）能调内置浏览器/Python 工具，recipient 路由与 harmony 解析（`openai/responses/harmony.py:_parse_mcp_call`）配合。
- **schema 精简**：MCP server 的 JSON schema 可能很大，`trim_schema` 避免塞爆模型 context。
- **DTO 隔离**：`ToolServer` ABC 让本地（`DemoToolServer`）与远程（`MCPToolServer`）统一接口，便于测试与扩展。

## 怎么做

Responses API 请求带 MCP tool：

```jsonc
{
  "model": "...",
  "input": [...],
  "tools": [
    {"type":"mcp","server_label":"fs","server_url":"http://mcp:8000","allowed_tools":["read_file"]}
  ]
}
```

`MCPToolServer` 在 stream 期间调远程 `http://mcp:8000` 的 `read_file`，结果流回。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| Tool ABC | `vllm/entrypoints/mcp/tool.py:49` |
| HarmonyBrowserTool | `vllm/entrypoints/mcp/tool.py:59` |
| HarmonyPythonTool | `vllm/entrypoints/mcp/tool.py:101` |
| validate_gpt_oss_install | `vllm/entrypoints/mcp/tool.py:25` |
| ToolServer ABC | `vllm/entrypoints/mcp/tool_server.py:74` |
| MCPToolServer | `vllm/entrypoints/mcp/tool_server.py:102` |
| DemoToolServer | `vllm/entrypoints/mcp/tool_server.py:197` |
| list_server_and_tools | `vllm/entrypoints/mcp/tool_server.py:18` |
| trim_schema | `vllm/entrypoints/mcp/tool_server.py:31` |
| post_process_tools_description | `vllm/entrypoints/mcp/tool_server.py:56` |

## 与其它模块/系统配合

- [openai/responses.md](../openai/responses.md)：Responses API 是 MCP 主要消费方；`_extract_allowed_tools_from_mcp_requests`、`emit_mcp_delta_events`/`emit_mcp_completion_events`（`openai/responses/streaming_events.py:287/517`）。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：harmony 工具与 `parser/harmony_utils.py`。
- [可观测-metrics](../../16-observability/README.md)：工具调用指标（待核实）。
- 外部：MCP 协议规范（modelcontextprotocol）。

## 历史版本演进

- **v0.10.x（MCP 集成）**：`ToolServer`/`MCPToolServer`/`DemoToolServer` + `list_server_and_tools`/`trim_schema`；Responses API `MCP` tool 类型；`emit_mcp_*` 事件。
- **v0.11（harmony 工具）**：`HarmonyBrowserTool`/`HarmonyPythonTool`（gpt-oss 浏览器/Python 工具）；`validate_gpt_oss_install`。
- **main**：`allowed_tools` 支持 `McpAllowedToolsMcpToolFilter` 对象（`serving.py:138`）；并发工具调用（待核实）；schema 精简策略增强。

## 模块导航

| 页 | 主题 |
|---|---|
| [tool.md](tool.md) | `Tool`/`HarmonyBrowserTool`/`HarmonyPythonTool` |
| [tool-server.md](tool-server.md) | `ToolServer`/`MCPToolServer`/`DemoToolServer` |

## 参见

- [← 返回 API 入口首页](../README.md)
- [openai/responses.md](../openai/responses.md)
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)
