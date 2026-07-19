# 跨子系统强相关矩阵

[← Wiki 首页](../README.md) > [附录](../README.md) > 跨子系统强相关矩阵

下表列出"理解一方必须同时看另一方"的强相关模块对。仅放紧耦合，不放一般引用。

| A 模块 | B 模块 | 强相关点 |
|---|---|---|
| [`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md) | [`01-engine-core/kv-cache-management/coordinator.md`](../01-engine-core/kv-cache-management/coordinator.md) | 调度步内对 KV block 的申请/复用/驱逐决策一体；拆开看会断流 |
| [`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md) | [`02-execution/executor/abstract.md`](../02-execution/executor/abstract.md) | SchedulerOutput 出口即 Executor 入口；进程拓扑同源 |
| [`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md) | [`05-attention/backend-abstraction.md`](../05-attention/backend-abstraction.md) | model runner 装配 AttentionMetadata，attention 后端由其驱动 |
| [`02-execution/worker/cudagraph-capture.md`](../02-execution/worker/cudagraph-capture.md) | [`09-compilation-ir/cuda-graph.md`](../09-compilation-ir/cuda-graph.md) | Worker 是 cuda graph 的捕获/重放主体，编译层提供 wrapper |
| [`03-model-execution/model-loader/dispatch.md`](../03-model-execution/model-loader/dispatch.md) | [`04-model-zoo/registry.md`](../04-model-zoo/registry.md) | loader 通过 registry 选模型类；二者必须同改 |
| [`03-model-execution/layers/fused-moe.md`](../03-model-execution/layers/fused-moe.md) | [`07-distributed/device-communicators/all2all.md`](../07-distributed/device-communicators/all2all.md) | MoE EP 路由经 all2all 落地，互相限定调用流 |
| [`05-attention/backends/mla/README.md`](../05-attention/backends/mla/README.md) | [`04-model-zoo/architecture-families/deepseek.md`](../04-model-zoo/architecture-families/deepseek.md) | MLA 内核受模型驱动选择，参数（LSE base/absorb）由模型类决定 |
| [`06-sampling-decoding/speculative-decoding/llm-base-proposer.md`](../06-sampling-decoding/speculative-decoding/llm-base-proposer.md) | [`06-sampling-decoding/rejection-sampler.md`](../06-sampling-decoding/rejection-sampler.md) | draft→target→accept 链式一体 |
| [`06-sampling-decoding/structured-output/manager.md`](../06-sampling-decoding/structured-output/manager.md) | [`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md) | grammar 状态随 SchedulerOutput 下发，调度器持有 manager |
| [`07-distributed/kv-transfer/README.md`](../07-distributed/kv-transfer/README.md) | [`15-kv-cache-offload/README.md`](../15-kv-cache-offload/README.md) | 二者统一构成"跨引擎/跨节点 KV 流动"全景；边界在 in-engine vs cross-engine |
| [`08-platforms/interface.md`](../08-platforms/interface.md) | [`09-compilation-ir/compiler-interface.md`](../09-compilation-ir/compiler-interface.md) | Platform 提供编译后端与 pass manager 钩子 |
| [`08-platforms/device-allocator.md`](../08-platforms/device-allocator.md) | [`15-kv-cache-offload/sleep-mode.md`](../15-kv-cache-offload/sleep-mode.md) | sleep mode 由 device_allocator 实现且与 KV reset 联动 |
| [`10-config/vllm-config.md`](../10-config/vllm-config.md) | 全部子系统 | VllmConfig 是各子系参数面，各 sub-config 页都回链到对应消费子系统 |
| [`11-multimodal/v1-integration.md`](../11-multimodal/v1-integration.md) | [`01-engine-core/input-processor.md`](../01-engine-core/input-processor.md) | 多模态预算/占位符在 InputProcessor 内联 |
| [`11-multimodal/v1-integration.md`](../11-multimodal/v1-integration.md) | [`01-engine-core/kv-cache-management/encoder-cache.md`](../01-engine-core/kv-cache-management/encoder-cache.md) | encoder 缓存由 EngineCore 管理，供 worker 消费 |
| [`12-lora/v1-integration.md`](../12-lora/v1-integration.md) | [`02-execution/worker/lora-mixin.md`](../02-execution/worker/lora-mixin.md) | LoRA 在 v1 worker 通过 mixin 接入 |
| [`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md) | [`14-tokenizers-transformers/reasoning.md`](../14-tokenizers-transformers/reasoning.md) | Responses API 的 reasoning 字段依赖 parser |
| [`14-tokenizers-transformers/tool_parsers/README.md`](../14-tokenizers-transformers/tool_parsers/README.md) | [`13-entrypoints/openai/responses.md`](../13-entrypoints/openai/responses.md) | tool_call 输出经 parser 进 Responses/Chat |
| [`16-observability/stats.md`](../16-observability/v1/metrics/stats.md) | [`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md) | EngineCore 每步产出 IterationStats |
| [`17-utils-cross-cutting/envs.md`](../17-utils-cross-cutting/envs.md) | 全部子系统 | 几乎所有子系统的开关都从 envs 取 |
| [`09-compilation-ir/passes/fusion.md`](../09-compilation-ir/passes/fusion.md) | [`07-distributed/parallel-state.md`](../07-distributed/parallel-state.md) | AsyncTP/SP 融合 pass 直接面向 TP 拓扑 |
| [`02-execution/worker/model-runner-v2.md`](../02-execution/worker/model-runner-v2.md) | [`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md) | MRv2 是 MRv1 的渐进替换，迁移期同存 |

[← 返回附录首页](../README.md)

## 参见

- [`conventions.md`](conventions.md)
- [`external-references.md`](external-references.md)
