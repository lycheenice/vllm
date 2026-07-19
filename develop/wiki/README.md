# vLLM 框架 Wiki

> 本 Wiki 以树形结构系统性拆解 vLLM 框架，按"子系统 → 子模块"两层组织，每个子模块均包含：**是什么 / 为什么 / 怎么做 / 与其它模块配合 / 历史版本演进** 五段式内容。所有层级之间以 Markdown 相对链接互联。

代码库根目录：`/home/lychee/mycode/vllm`（所有源码引用以此为准）。

> 想本地以网站形式浏览？参见 [SERVE.md](SERVE.md)（mkdocs 一键起服务，mermaid 图直接渲染）。

---

## 阅读路径

- 第一次接触 vLLM：先读 [`00-overview/`](00-overview/README.md)。
- 想理解一个请求如何流过整个框架：[`00-overview/request-lifecycle.md`](00-overview/request-lifecycle.md)。
- 想了解 v0 到 v1 的架构跃迁：[`00-overview/v0-vs-v1.md`](00-overview/v0-vs-v1.md)。
- 想直接进入某个子系统：见下方导航树。

---

## 子系统导航树

| # | 子系统 | 简介 | 入口 |
|---|---|---|---|
| 00 | 全局资产 | 总览图、请求生命周期、版本演进、术语表 | [`00-overview/`](00-overview/README.md) |
| 01 | 引擎核心 | AsyncLLM 前端 + EngineCore 进程 + 调度器 + KV 缓存管理 | [`01-engine-core/`](01-engine-core/README.md) |
| 02 | 执行层 | Executor 与 Worker（GPU/CPU/XPU/Model Runner V2） | [`02-execution/`](02-execution/README.md) |
| 03 | 模型执行 | 模型加载器 + 层库 + 量化 + 内核 + 卸载/预热 | [`03-model-execution/`](03-model-execution/README.md) |
| 04 | 模型库 | 模型注册表与 ~280 个模型架构分类 | [`04-model-zoo/`](04-model-zoo/README.md) |
| 05 | 注意力 | 注意力后端抽象与各厂商后端 + MLA | [`05-attention/`](05-attention/README.md) |
| 06 | 采样与解码 | 采样器 + 投机解码 + 结构化输出 | [`06-sampling-decoding/`](06-sampling-decoding/README.md) |
| 07 | 分布式 | 并行状态 / 设备通信 / KV 迁移 / EP/EPLB/ 权重迁移 | [`07-distributed/`](07-distributed/README.md) |
| 08 | 硬件平台 | CUDA/ROCm/TPU/XPU/CPU/Zen + 设备内存分配器 | [`08-platforms/`](08-platforms/README.md) |
| 09 | 编译与 IR | torch.compile 集成 + Inductor Pass + vLLM IR | [`09-compilation-ir/`](09-compilation-ir/README.md) |
| 10 | 配置体系 | VllmConfig 与全部子配置 dataclass | [`10-config/`](10-config/README.md) |
| 11 | 多模态 | 注册 / 解析 / 缓存 / 处理 / 媒体 / 编码器预算 | [`11-multimodal/`](11-multimodal/README.md) |
| 12 | LoRA | LoRA 层 / 内核 / Punica / 管理器 | [`12-lora/`](12-lora/README.md) |
| 13 | API 入口 | OpenAI/Anthropic/CLI/LLM/Serve/Pooling/Scale-out | [`13-entrypoints/`](13-entrypoints/README.md) |
| 14 | 分词与转换器 | tokenizers / transformers_utils / tool_parsers / reasoning / renderers | [`14-tokenizers-transformers/`](14-tokenizers-transformers/README.md) |
| 15 | KV 缓存卸载 | kv_offload / simple_kv_offload / 分层 | [`15-kv-cache-offload/`](15-kv-cache-offload/README.md) |
| 16 | 可观测性 | metrics / profiler / tracing / logging_utils | [`16-observability/`](16-observability/README.md) |
| 17 | 工具与横切 | envs / exceptions / forward_context / sequence / utils / plugins | [`17-utils-cross-cutting/`](17-utils-cross-cutting/README.md) |
| 18 | 构建/CI/测试 | csrc / rust / cmake / buildkite / tests / benchmarks | [`18-build-ci-testing/`](18-build-ci-testing/README.md) |
| 19 | 附录 | 关联规范 / 扩展术语 / 外部参考 | [`19-appendix/`](19-appendix/README.md) |

---

## 写作与链接规范

1. **语言**：中文为主，专有名词、类名、配置项保留英文原词。
2. **路径引用**：源码引用统一使用 `相对/路径/to/file.py:行号` 的形式，便于 IDE 跳转。
3. **模块五段式**：每个模块页须包含 `## 是什么`、`## 为什么`、`## 怎么做`、`## 与其它模块/系统配合`、`## 历史版本演进` 五个二级标题（次序固定）。
4. **导航约定**：
   - 模块页顶部放置面包屑：`[← 子系统首页](README.md) > [子系统名](README.md)`。
   - 模块页底部放置返回链接：`[← 返回子系统首页](README.md)`，并附"参见"区块链接到强相关兄弟模块。
   - 子系统 README 顶部包含 `[← Wiki 首页](README.md)`。
5. **图示**：能用 [Mermaid](https://mermaid.js.org/) 流程图/时序图说明的优先使用，避免大段文字。
6. **演进记录**：以 vLLM 版本号（如 `v0.5.0`、`v0.6.0`、`v0.7.0`、`v0.8.0`、`v0.9.0`、`v0.10.0`、`v0.11.0`、`v0.12.0`、`main`）或日期为锚点，描述"变更要点 + 触发动机 + 影响"。无法确定版本号的写"早期/中期/近期"。
7. **不臆造**：无法验证的内容必须显式标注 `（待补充）` 或 `（待核实）`，不杜撰类名、文件名或行号。

---

## 维护

- 本 Wiki 由子系统级 sub-agent 并行调研产出，最后统一审校链接一致性。
- 任何子模块新增/重组需同步更新所在子系统 README 的导航表与顶层导航树。
