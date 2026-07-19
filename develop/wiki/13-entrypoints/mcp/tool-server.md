[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [MCP](README.md) > tool_server

# mcp/tool_server.py（ToolServer + MCPToolServer）

> `tool_server.py` 定义工具 server 抽象 `ToolServer` 与 MCP 协议实现 `MCPToolServer`、演示实现 `DemoToolServer`。`MCPToolServer` 是 vLLM 调用外部 MCP server 的桥梁：列工具、调工具、收结果。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `list_server_and_tools` | `vllm/entrypoints/mcp/tool_server.py:18` | 从远程 MCP server 拉工具列表 |
| `trim_schema` | `:31` | 精简 JSON schema（去超大字段） |
| `post_process_tools_description` | `:56` | 后处理工具描述（统一格式） |
| `ToolServer` | `:74` | 工具 server ABC |
| `MCPToolServer` | `:102` | MCP 协议实现 |
| `DemoToolServer` | `:197` | 本地演示 server |

`MCPToolServer`（`:102`）流程：

1. 初始化时 `list_server_and_tools(server_url)`（`:18`）拉远程工具 + schema。
2. `post_process_tools_description`（`:56`）+ `trim_schema`（`:31`）把工具描述喂给模型（作为 Responses `tools` 字段）。
3. 运行期：模型流式产出 `mcp_call` → `MCPToolServer` 调远程 MCP server `tools/call`（MCP 协议）→ 结果作为 `response.output_item`（`mcp_call` 事件）流回。
4. `allowed_tools` 过滤：仅调允许的工具，安全控制。

`DemoToolServer`（`:197`）：本地假工具（如 echo），供开发/测试，无需外部 MCP server。

## 为什么

- **远程 MCP 复用**：MCP 生态有成熟 server（FS、git、browser、DB…），`MCPToolServer` 直接消费，vLLM 无需重造。
- **schema 精简**：MCP 工具 JSON schema 可能很大（含 `$schema`/`examples` 等），`trim_schema` 去冗余避免 context 溢出。
- **allowed_tools 安全**：按 request 限定可调工具子集，防止模型乱调高危工具。
- **ABC 测试**：`DemoToolServer` 让 Responses API 工具调用链可在无外部 MCP 时单测。
- **流式闭环**：`MCPToolServer` 与 Responses streaming 配合，工具调用在同一个 SSE 流里以 `mcp_call.delta`/`mcp_call.done` 事件呈现。

## 怎么做

见 [README](README.md) 请求示例。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| list_server_and_tools | `vllm/entrypoints/mcp/tool_server.py:18` |
| trim_schema | `vllm/entrypoints/mcp/tool_server.py:31` |
| post_process_tools_description | `vllm/entrypoints/mcp/tool_server.py:56` |
| ToolServer | `vllm/entrypoints/mcp/tool_server.py:74` |
| MCPToolServer | `vllm/entrypoints/mcp/tool_server.py:102` |
| DemoToolServer | `vllm/entrypoints/mcp/tool_server.py:197` |

## 与其它模块/系统配合

- [openai/responses.md](../openai/responses.md)：`_extract_allowed_tools_from_mcp_requests`、`emit_mcp_*`。
- [tool.md](tool.md)：`Tool` 本地工具互补。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：工具描述渲染。
- 外部：MCP 协议规范、各类 MCP server。

## 历史版本演进

- **v0.10.x（MCP 集成）**：`ToolServer`/`MCPToolServer`/`DemoToolServer` + `list_server_and_tools`/`trim_schema`/`post_process_tools_description`；Responses `MCP` tool 类型。
- **v0.11（并发与过滤）**：`allowed_tools` 支持 `McpAllowedToolsMcpToolFilter`；并发工具调用（待核实）。
- **main**：schema 精简策略增强；远程 server 健康检查（待核实）。

## 参见

- [← 返回 MCP 首页](README.md)
- [tool.md](tool.md)
- [openai/responses.md](../openai/responses.md)
