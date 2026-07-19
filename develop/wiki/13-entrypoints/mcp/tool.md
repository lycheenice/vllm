[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [MCP](README.md) > tool

# mcp/tool.py（Tool 抽象 + harmony 工具）

> `tool.py` 定义工具抽象 `Tool` 与两类 harmony 工具 `HarmonyBrowserTool`/`HarmonyPythonTool`，分别对应 gpt-oss harmony 协议里的浏览器与 Python 工具，由 recipient 字段路由。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `validate_gpt_oss_install` | `vllm/entrypoints/mcp/tool.py:25` | 校验 gpt-oss 运行环境 |
| `Tool` | `:49` | 工具 ABC |
| `HarmonyBrowserTool` | `:59` | harmony 浏览器工具 |
| `HarmonyPythonTool` | `:101` | harmony Python 工具 |

`Tool`（`:49`）定义工具调用接口（执行、结果格式化）；`HarmonyBrowserTool`（`:59`）实现浏览器类工具（如导航、点击、提取内容），由 harmony `recipient`（如 `browser`）路由调用；`HarmonyPythonTool`（`:101`）实现代码执行类工具（如沙箱 Python），`recipient` 为对应 namespace。

## 为什么

- **harmony recipient 路由**：harmony 模型输出带 `recipient` 字段标识目标工具，`HarmonyBrowserTool`/`PythonTool` 按 namespace 匹配，实现单模型多工具并行。
- **gpt-oss 依赖校验**：`validate_gpt_oss_install`（`:25`）保证运行 harmony 工具所需环境（gpt-oss 包/权重）就绪，避免运行期失败。
- **ABC 扩展**：`Tool` ABC 让自定义工具（如内部知识库）可插入 harmony 协议。

## 怎么做

由 `OpenAIServingResponses` 在 harmony 模式下按 `tools` 配置实例化对应 `Tool`，streaming parser 解析 recipient 后调用。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| validate_gpt_oss_install | `vllm/entrypoints/mcp/tool.py:25` |
| Tool | `vllm/entrypoints/mcp/tool.py:49` |
| HarmonyBrowserTool | `vllm/entrypoints/mcp/tool.py:59` |
| HarmonyPythonTool | `vllm/entrypoints/mcp/tool.py:101` |

## 与其它模块/系统配合

- [tool-server.md](tool-server.md)：`MCPToolServer` 是远程工具实现，与本地 `Tool` 互补。
- [openai/responses.md](../openai/responses.md)：harmony `_parse_mcp_recipient`/`emit_browser_tool_events`。
- [tokenizers-transformers](../../14-tokenizers-transformers/README.md)：`harmony_utils`。

## 历史版本演进

- **v0.11（引入）**：`HarmonyBrowserTool`/`HarmonyPythonTool` + `validate_gpt_oss_install`，配合 gpt-oss 模型。
- **main**：浏览器/Python 工具事件族（`streaming_events.emit_browser_tool_events`/`emit_code_interpreter_*`）。

## 参见

- [← 返回 MCP 首页](README.md)
- [tool-server.md](tool-server.md)
- [openai/responses.md](../openai/responses.md)
