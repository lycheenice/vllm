# 术语表

[← 全局资产首页](README.md) > [全局资产](README.md)

本表收录跨子系统通用术语，子系统专属术语在各模块页内解释。

| 术语 | 释义 | 相关子系统 |
|---|---|---|
| **v1** | vLLM 现行引擎架构（自 v0.7 起），所有调度/worker/attention 都在 `vllm/v1/` 下 | #01–#17 |
| **EngineCore** | v1 引擎的调度循环进程，与前端经 ZMQ 通信 | #01 |
| **AsyncLLM** | v1 异步前端，实现 `EngineClient` ABC，是 API/LLM 类的入口 | #01/#13 |
| **SchedulerOutput** | EngineCore 每步下发给 Executor 的调度结果（含各请求的 block_id、是否 prefill 等） | #01/#02 |
| **ModelRunnerOutput** | Worker 每步返回的工具：token ids、logprobs、spec acceptance 等 | #02 |
| **EngineCoreRequest / Output** | 跨进程的 msgspec 传输结构 | #01 |
| **Request** | v1 内部请求运行时对象（替代 v0 的 `SequenceGroup`） | #01 |
| **chunked prefill** | 把长 prefill 切成多步执行，与 decode 共同组 batch；v1 默认 | #01 |
| **prefix cache** | 复用相同 prompt 前缀的 KV block，避免重复 prefill | #01 |
| **KVCacheManager / Coordinator / BlockPool** | v1 KV 调度三件套：混合类型块池 + 前缀寻址 + 块分配 | #01 |
| **KVCacheSpec / kv_cache_spec_registry** | 让每层声明自己的 KV 形状/类型，调度器据此建块池 | #01 |
| **Executor** | 决定如何拉起 Worker 集合：uniproc / multiproc / ray / ray-v2 | #02 |
| **Worker / ModelRunner** | 设备侧进程 / 步前向驱动器；cudagraph 重放在此 | #02 |
| **MRv2** | Model Runner V2（`vllm/v1/worker/gpu/`，v0.11+ 实验性下一代） | #02 |
| **ubatch / micro-batching** | 在 batch 内再切片执行，提升吞吐 | #02 |
| **breakable cudagraph** | 在 stream capture 时按需断点的 cudagraph 模式（区别于 FX 预切分的 piecewise） | #09 |
| **piecewise compile** | 把 FX 图按 attention/custom-op 边界切分为可捕获子图后交给 Inductor | #09 |
| **Inductor pass** | torch._inductor 的自定义图 pass；vLLM 在 `post_grad_custom_post_pass` 挂入 | #09 |
| **vLLM IR** | `vllm/ir/` 下的 `vllm_ir` torch 库命名空间，定义可替换实现的算子（rms_norm 等） | #09 |
| **AttentionMetadata** | 每步前向传给 AttentionBackend 的元数据（seq lens、block_table、cu_seqlens 等） | #05 |
| **MLA** | Multi-head Latent Attention（DeepSeek 系列），低秩 KV 压缩 | #05 |
| **Mamba / SSM** | 状态空间模型注意力分支（线性注意力 + 选择性 SSM） | #05 |
| **TP / PP / DP / EP** | 张量/流水/数据/专家并行 | #07 |
| **EPLB** | Expert-Parallel Load Balancer，冗余专家再平衡 | #07 |
| **NIXL** | NVIDIA Inference Transfer Lib，KV/数据跨节点高速搬移 | #07 |
| **KV connector** | 解耦/跨引擎 KV 迁移的可插拔框架（LMCache/FlexKV/NIXL/Mooncake/MoriIO/HF3FS） | #07/#15 |
| **EC transfer** | Expert-Connector 跨引擎迁移 | #07 |
| **weight transfer** | 在线权重同步/补丁，用于 scale-out | #07 |
| **Sampler** | v1 的下一步 token 采样器，nn.Module | #06 |
| **RejectionSampler** | 投机解码目标模型侧的接受/拒绝算法 | #06 |
| **Proposer** | 投机解码的草稿模型或启发式（Eagle/Medusa/MTP/Ngram/Suffix/...） | #06 |
| **structured output** | 受 JSON schema / regex / grammar 约束的解码（xgrammar/outlines/guidance/lm-format-enforcer） | #06 |
| **thinking budget** | reasoning 模型思考段输出长度控制 | #06 |
| **Platform** | 硬件抽象基类（`vllm/platforms/interface.py`），`current_platform` 是单例 | #08 |
| **cumem / sleep mode** | CUDA driver pluggable allocator，让 KV 显存可入睡/唤醒 | #08 |
| **ModelLoader** | 模型装配器：解析 registry → 下载权重 → 注入 nn.Module | #03 |
| **registry** | `vllm/model_executor/models/registry.py`，HF `architectures` → vLLM 模型类映射 | #04 |
| **CustomOp** | `@CustomOpRegister` 装饰的算子，可按平台切自定义实现与 torch 实现 | #03 |
| **VllmConfig** | 顶层复合配置，聚合 ModelConfig/CacheConfig/ParallelConfig/... | #10 |
| **MultiModalRegistry** | 多模态处理器注册表与调度 | #11 |
| **LoRA mapping / Punica** | token → adapter 索引 + 多 LoRA 批量 GEMM 后端 | #12 |
| **renderers** | 针对特定模型的流式渲染（diffusion/reasoning/...） | #14 |
| **forward_context** | 步前向上下文管理器，承载 AttentionMetadata + DP/cudagraph 描述 | #17 |

[← 返回全局资产首页](README.md)
