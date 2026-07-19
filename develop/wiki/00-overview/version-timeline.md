# 顶层版本演进时间线

[← 全局资产首页](README.md) > [全局资产](README.md)

本页给出 vLLM 顶层版本的演进主线，子模块的精细化演进记录在各自的"历史版本演进"小节。

> 版本号取自 `vllm/version.py` 与 GitHub release；具体细节以代码为准，标注不明的写"早期/中期/近期"。

## 主线时间线

| 版本 | 时间（约） | 主要里程碑 |
|---|---|---|
| `v0.3–v0.5` | 2024 上半年 | PagedAttention + v0 引擎主线；prefix cache 初版；tensor parallel + pipeline parallel；OpenAI API server |
| `v0.5` | 2024-08 | 多模态输入（LLaVA 系列）正式化；chunked prefill 默认可用 |
| `v0.6` | 2024-09 | 引入 v1 预研；speculative decoding（n-gram / Eagle v1）；structured output（outlines / guidance）起步 |
| `v0.7` | 2024-11 | **v1 引擎架构落地**：AsyncLLM + EngineCore 双进程 + ZMQ；`vllm/v1/` 全面铺开。前端 / API server 经 `EngineClient` 协议解耦 |
| `v0.7.x` | 2024 Q4 | v1 调度器、KV 多类型 spec、MRv1 worker；FlashInfer/Triton attention 接入 |
| `v0.8` | 2025-02 | v1 设为默认（关键里程碑）；v0 `LLMEngine`/`AsyncLLMEngine` 退化为 shim；分布式 EP/EPLB/elastic-ep 入场；Mooncake/NIXL KV connector |
| `v0.8.x` | 2025 Q1 | MLA 后端矩阵扩充（aiter/triton/cutlass/flashmla）；vLLM IR + Inductor pass 框架成型 |
| `v0.9` | 2025-04 | 投机解码扩族（Eagle3 / MTP / dflash / dspark / suffix_decoding）；breakable cudagraph；kv_offload tiering |
| `v0.9.x` | 2025 Q2 | multi-connector / FlexKV / HF3FS / MoriIO；EC transfer；weight live patch；MRv2（`v1/worker/gpu/`）实验性引入 |
| `v0.10` | 2025-06 | DeepSeek V4 / Qwen3 / GLM4 MoE 系列模型批量接入；reasoning parser、thinking budget 入主线 |
| `v0.10.x` | 2025 Q3 | Anthropic Messages API、Responses API、gRPC server、scale-out（render/derender）端点 |
| `v0.11` | 2025 Q3 | MRv2 渐进切换；vendor-split 模型（`vllm/models/deepseek_v4/{amd,nvidia,xpu}`）；warmup 模块成型 |
| `v0.12` | 2025 Q4 | MNNVL/Blackwell all2all；UVA offload；`v1/worker/gpu/pool/` pooling MRv2 |
| `main` | 持续 | 本仓库 `develop/` 工作树对应最新主线，含 MRv2、kv_offload tiering、scale-out、structured output 多后端 |

## 演进维度索引

- 引擎进程拓扑：参见 [`01-engine-core/engine-core-process.md`](../01-engine-core/engine-core-process.md) 的演进小节。
- KV 缓存管理：参见 [`01-engine-core/kv-cache-management/`](../01-engine-core/kv-cache-management/README.md)。
- 调度器：参见 [`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)。
- 注意力后端：参见 [`05-attention/`](../05-attention/README.md)。
- 投机解码：参见 [`06-sampling-decoding/speculative-decoding/`](../06-sampling-decoding/speculative-decoding/README.md)。
- 分布式：参见 [`07-distributed/`](../07-distributed/README.md)。
- 编译/IR：参见 [`09-compilation-ir/`](../09-compilation-ir/README.md)。
- 模型库扩展：参见 [`04-model-zoo/`](../04-model-zoo/README.md)。

[← 返回全局资产首页](README.md)
