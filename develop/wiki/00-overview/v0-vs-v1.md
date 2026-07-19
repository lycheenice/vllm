# v0 → v1 引擎架构跃迁

[← 全局资产首页](README.md) > [全局资产](README.md)

## 背景

vLLM 早期（v0.x 之前）以 `vllm/engine/llm_engine.py`（`LLMEngine`）与 `vllm/engine/async_llm_engine.py`（`AsyncLLMEngine`）为引擎核心，调度器与 KV 管理散落在 `vllm/core/`、`vllm/worker/`、`vllm/attention/` 等顶层目录。自 v0.7 起引入 **v1 架构**，目标是：多模态/LoRA/投机解码/结构化输出统一支持、单进程到多进程拓扑、消灭历史技术债。

## v0 与 v1 的对应关系

| 关注点 | v0（legacy） | v1（现行主线） | 现状 |
|---|---|---|---|
| 同步引擎入口 | `vllm/engine/llm_engine.py` | `vllm/v1/engine/llm_engine.py` | v0 文件退化为 7 行 shim：`LLMEngine = V1LLMEngine` |
| 异步引擎入口 | `vllm/engine/async_llm_engine.py` | `vllm/v1/engine/async_llm.py` | v0 文件退化为 7 行 shim：`AsyncLLMEngine = AsyncLLM` |
| 引擎对外协议 | — | `vllm/engine/protocol.py::EngineClient` ABC | **仍活跃**，AsyncLLM 实现该 ABC，API server 依赖此协议 |
| 引擎参数构造 | `vllm/engine/arg_utils.py`（`EngineArgs`/`AsyncEngineArgs`） | （沿用 v0 文件） | **仍活跃**，未被迁移，v1 引擎与 serve 入口共用 |
| 调度器 | `vllm/core/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | v1 独占；v0 路径已废弃；含 chunked prefill、prefix cache、ubatch、spec 内嵌 |
| KV 缓存管理 | `vllm/core/block_manager.py` | `vllm/v1/core/kv_cache_manager.py` + `kv_cache_coordinator.py` + `block_pool.py` | v1 独占；引入"多类型/混合 KV spec"机制 |
| Sequence 抽象 | `vllm/sequence.py::SequenceGroup` / `Sequence` | `vllm/v1/request.py::Request` | v1 用 `Request` 替代；`vllm/sequence.py` 仍保留部分对外类型 |
| Worker | `vllm/worker/worker.py` | `vllm/v1/worker/gpu_worker.py` 等 | v1 独占；MRv2（`v1/worker/gpu/`）在 v0.11+ 引入 |
| Executor | `vllm/executor/` | `vllm/v1/executor/` | v1 独占；新增 `RayExecutorV2` + shm broadcast |
| Attention | `vllm/attention/`（v0） | `vllm/v1/attention/` | v1 独占；含 MLA、Mamba/linear、breakable cudagraph 等新特性 |
| 编译/cudagraph | `vllm/compilation/`（与 v1 共享） | 同上 | **共享基础设施**，非 v0/v1 二分；v1 worker 重度消费 |
| LoRA | `vllm/lora/`（共享） | 通过 `vllm/v1/worker/lora_model_runner_mixin.py` 集成 | 共享底层，v1 加 mixin |
| Spec decoding | `vllm/spec_decode/`（v0） | `vllm/v1/spec_decode/` | v1 独占；新增 Eagle3 / MTP / NgramGPU / dflash / dspark / suffix 等 |

## v1 关键架构特性

1. **进程拓扑**：前端 (`AsyncLLM`) 与 `EngineCore` 拆分到独立进程，经 ZMQ 通信（`vllm/v1/engine/core.py` / `core_client.py`）。支持数据并行（`DPCoordinator`，`vllm/v1/engine/coordinator.py`）。
2. **msgspec + ZMQ**：所有跨进程数据结构（`EngineCoreRequest`/`EngineCoreOutput` 事件、`SchedulerOutput`、`ModelRunnerOutput`）采用 `msgspec` 结构化序列化，提升性能、便于演化（`vllm/v1/engine/__init__.py`）。
3. **统一调度**：prefill/decode 在同一 batch 内共存（chunked prefill 默认开启），消除 v0 中显式切换。
4. **KV cache 多类型 / 混合 spec**：`vllm/v1/kv_cache_interface.py` + `kv_cache_spec_registry.py` 让不同层声明不同 spec（标准 attention / MLA / Mamba SSM / 编码器 cross-attn），调度器据此分配混合块池。
5. **cudagraph 内嵌**：`vllm/v1/attention/`、`v1/worker/gpu_model_runner.py` 与 `vllm/compilation/` 形成"piecewise/breakable cudagraph + Inductor pass"流水（[`09-compilation-ir/`](../09-compilation-ir/README.md)）。
6. **投机解码一体化**：从 v0 的"独立 coordinator"模式迁移到 `v1/spec_decode/` 的 Proposer + RejectionSampler 同进程模式。
7. **统一 LoRA / 多模态**：通过 `Worker` 侧 mixin 与 `InputProcessor` 把 LoRA、多模态预算、思考预算等能力无侵入注入。
8. **结构化输出内嵌调度**：`StructuredOutputManager` 直接挂在 `Scheduler`，grammar 状态随 `SchedulerOutput` 下发到 Worker。

## 何时还看 v0

- `vllm/engine/arg_utils.py`、`vllm/engine/protocol.py` 仍是 v1 的事实依赖，需要查阅。
- 部分顶层 `vllm/sequence.py`、`vllm/outputs.py` 类型被 `v1` 复用为对外接口。
- 历史调研 / 升级排查时需对比 v0 路径（旧 issue / 旧 PR）。

## 渐进迁移建议

- 若你的扩展运行在 v1 之上，直接在 `vllm/v1/` 的对应子目录改造即可，不要回退到 v0 顶层目录。
- 若使用第三方插件（platform / kernel），通过 `Platform`、`@register_kv_cache_spec`、`@register_op` 等注册机制接入，避免改 v1 核心。

[← 返回全局资产首页](README.md)
