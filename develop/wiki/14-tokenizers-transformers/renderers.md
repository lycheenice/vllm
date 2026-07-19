# 渲染器（Renderers）

[← Wiki 首页](../README.md) > [分词与转换器](../README.md) > 渲染器

## 是什么

`vllm/renderers/` 子包提供"渲染器（renderer）"机制，引擎在收到请求时（input 阶段）可选地把 prompt **预渲染**为某种结构化输出，或反向"去渲染（derender）"为 token。它由 `base.py::Renderer` 抽象基类 + `registry.py::RendererRegistry` 注册表 + 一组具体实现（`online_renderer.py`、`online_derenderer.py`、`mistral.py`、`deepseek_v32.py`、`deepseek_v4.py`、`hf.py`、`terratorch.py`、`embed_utils.py`、`params.py::RendererParams`、`inputs/` 子包）组成。渲染器位于 [`13-entrypoints/serve/`](../13-entrypoints/serve/README.md) 与 [`13-entrypoints/scale-out/`](../13-entrypoints/scale-out/README.md) 后端，是"立即返回结果但不走 LLM 解码"路径（如在线 deprefill、derender）的关键。

## 为什么

- **解耦"输入加工"与"解码"**：某些场景（diffusion-LM 的 mask 渲染、scale-out 的 derender 阶段、Mistral tekken prompt 预处理）不需要或不应该走完整生成，但又要返回合法输出——renderer 提供该 escape hatch。
- **多策略可插拔**：通过 registry 让用户按 `renderer: online|online_derenderer|mistral|deepseek_v4|hf|terratensor` 等选择；新加厂商只需实现 `Renderer` ABC + 注册。
- **scale-out 历史协作**：在 disaggregated prefill/decode 分离部署中，prefill 节点产出"已渲染的中间表示"给 decode 节点消费，renderer 负责"以服务形态输出"。
- **与 reasoning/tool 协同**：某些 renderer 会在 prompt 上预先插入分隔模板，使后续 [`reasoning.md`](./reasoning.md) / [`tool_parsers/`](./tool_parsers/README.md) 能直接识别。

## 怎么做

### 抽象层
- `base.py::Renderer`：声明 `render(request, ...) -> RendererOutput`、`supports_*` 能力位、`get_renderer_info()` 等接口。
- `params.py::RendererParams`：dataclass，承载 renderer 选择 + 渲染参数。
- `registry.py::RendererRegistry`：`@register_renderer("name")` 装饰器 + 名字 → 类映射；`get_renderer(name)` 工厂。
- `inputs/` 子包：renderer 专用的输入类型（含 multimodal variant）。

### 具体实现
| 文件 | 角色 |
|---|---|
| `online_renderer.py` | 在线渲染：直接对当前 prompt 做一次 forward 并把 logits/中间态作为"输出"返回，供 scale-out 的 `render` 端点使用 |
| `online_derenderer.py` | 反向：把上层传来的结构化渲染结果"还原"为 token ids（用于 derender 端点） |
| `mistral.py` | Mistral tekken / Artemis 专用的 prompt 预处理（含像素图嵌入） |
| `deepseek_v32.py` / `deepseek_v4.py` | DeepSeek V3/V4 自家 special token + 压缩流预处理 |
| `hf.py` | 走 HF processor 路径的通用渲染（fallback） |
| `terratorch.py` | TerraTorch 地理模型专用 |
| `embed_utils.py` | embedding 预计算辅助（与 [`03-model-execution/layers/pooler.md`](../03-model-execution/layers/pooler.md) 互通） |

### 注册流程
1. 引擎/服务启动时，`RendererRegistry` 被填充（每个 renderer 文件在被 import 时执行 `register_renderer`）。
2. API server / scale-out endpoint 按 `Request.renderer` 或配置项选择 renderer，调用 `render()`。
3. 渲染结果直接构造 `RequestOutput` 返回，或被注入到下游请求作为"已渲染 prompt"。

## 与其它模块/系统配合

- **API 入口**：
  - [`13-entrypoints/scale-out/render.md`](../13-entrypoints/scale-out/render.md)、[`13-entrypoints/scale-out/derender.md`](../13-entrypoints/scale-out/derender.md)：scale-out 直接调用 renderer。
  - [`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md)：Responses API 在特殊 renderer（如 deepseek_v4）下会走预渲染。
- **引擎核心**：renderer 路径**绕开** `EngineCore` 的标准 decode 循环；与 [`01-engine-core/input-processor.md`](../01-engine-core/input-processor.md) 共享 tokenizer/多模态预算，但产出直接进 [`01-engine-core/output-processor.md`](../01-engine-core/output-processor.md)。
- **多模态**：renderer 调用 [`11-multimodal/processing.md`](../11-multimodal/processing.md) 装配多模态输入；`inputs/` 子包承载 renderer 专用输入类型。
- **分词**：与 [`tokenizers/registry.md`](./tokenizers/registry.md) 协作，把渲染结果转 token（derenderer）或反向。
- **模型库**：DeepSeek V4 / Mistral / Gemma4 等专用 renderer 直接对应 [`04-model-zoo/architecture-families/deepseek.md`](../04-model-zoo/architecture-families/deepseek.md) 等家族。
- **配置**：通过 `RendererParams`（在 [`17-utils-cross-cutting/`](../17-utils-cross-cutting/README.md) 中归类为"横切"）或 API 字段传参，受 `scale_out` / `disaggregated` 相关字段约束（具体配置归属待核实）。

## 历史版本演进

- **v0.7 之前（早期）**：无独立 renderer 概念；Mistral 的 tekken 预处理散落在 `transformers_utils/processor.py`。
- **v0.9（中期）**：scale-out 端点（render/derender）落地，引入 `online_renderer.py`/`online_derenderer.py` 与 `Renderer` ABC + `RendererRegistry`。
- **v0.10**：DeepSeek V3/V4 / Mistral 专用 renderer 接入；`inputs/` 子包扩 multimodal 支持。
- **v0.10.x–v0.11**：TerraTorch、HF 通用 renderer 落地；与 Responses API 协同（在 reasoning+tool 场景预渲染 prompt 段）。
- **v0.12 / main**：`embed_utils.py` 等辅助成型；renderer 用于 diffusion-LM 与"立即返回 embedding"路径；与 [`15-kv-cache-offload/`](../15-kv-cache-offload/README.md) 的 P2P transfer 在预渲染场景挂钩——具体时间归属（待核实）。

[← 返回分词与转换器首页](../README.md)

## 参见

- [`reasoning.md`](./reasoning.md)：思考段解析，常与 renderer 配合预插入分隔模板。
- [`tool_parsers/`](./tool_parsers/README.md)：工具调用解析，渲染后阶段衔接。
- [`13-entrypoints/scale-out/`](../13-entrypoints/scale-out/README.md)：scale-out endpoints。
- [`11-multimodal/processing.md`](../11-multimodal/processing.md)：多模态输入预处理。
- [`03-model-execution/layers/pooler.md`](../03-model-execution/layers/pooler.md)：pooler 层（`embed_utils.py` 互通）。
