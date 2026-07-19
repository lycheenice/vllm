# 外部参考（论文 / 项目）

[← Wiki 首页](../README.md) > [附录](../README.md) > 外部参考

下表列出 vLLM 实现与外部论文/项目的一一对应关系。论文 arXiv 编号在子 agent 产出中部分标 `(待核实)`，使用前请二次核对。

| 主题 | 外部参考 | 对应 vLLM 模块 |
|---|---|---|
| PagedAttention | vLLM paper (SOSP'23) | [`01-engine-core/kv-cache-management/`](../01-engine-core/kv-cache-management/README.md)、[`18-build-ci-testing/csrc.md`](../18-build-ci-testing/csrc.md) |
| Prefix caching | RadixAttention / SGLang paper | [`01-engine-core/kv-cache-management/coordinator.md`](../01-engine-core/kv-cache-management/coordinator.md) |
| Chunked prefill | Sarathi-Serve paper | [`01-engine-core/scheduler/chunked-prefill.md`](../01-engine-core/scheduler/chunked-prefill.md) |
| FlashAttention | Dao et al. | [`05-attention/backends/flash-attn.md`](../05-attention/backends/flash-attn.md) |
| FlashInfer | FlashInfer project | [`05-attention/backends/flashinfer.md`](../05-attention/backends/flashinfer.md) |
| FlashMLA | DeepSeek FlashMLA | [`05-attention/backends/mla/flashmla.md`](../05-attention/backends/mla/flashmla.md) |
| Multi-head Latent Attention | DeepSeek-V2/V3 paper | [`05-attention/backends/mla/README.md`](../05-attention/backends/mla/README.md) |
| DeepEP / DeepEP-v2 | DeepEP project | [`07-distributed/device-communicators/all2all.md`](../07-distributed/device-communicators/all2all.md) |
| DeepGEMM | DeepGEMM project | [`03-model-execution/layers/fused-moe.md`](../03-model-execution/layers/fused-moe.md)、[`18-build-ci-testing/csrc.md`](../18-build-ci-testing/csrc.md) |
| MNNVL all2all | Blackwell NVLink | [`07-distributed/device-communicators/mnnvl-compat.md`](../07-distributed/device-communicators/mnnvl-compat.md) |
| NIXL | NVIDIA Inference Transfer Lib | [`07-distributed/nixl-utils.md`](../07-distributed/nixl-utils.md)、[`07-distributed/kv-transfer/transports/nixl.md`](../07-distributed/kv-transfer/transports/nixl.md) |
| Mooncake | Mooncake KV store (Moonshot) | [`07-distributed/kv-transfer/transports/mooncake.md`](../07-distributed/kv-transfer/transports/mooncake.md) |
| HF3FS | 3FS (DeepSeek) | [`07-distributed/kv-transfer/transports/hf3fs.md`](../07-distributed/kv-transfer/transports/hf3fs.md) |
| MoriIO | MoriIO | [`07-distributed/kv-transfer/transports/moriio.md`](../07-distributed/kv-transfer/transports/moriio.md) |
| LMCache | LMCache project | [`07-distributed/kv-transfer/lmcache.md`](../07-distributed/kv-transfer/lmcache.md) |
| FlexKV | FlexKV | [`07-distributed/kv-transfer/flexkv.md`](../07-distributed/kv-transfer/flexkv.md) |
| EAGLE / EAGLE-2 / EAGLE-3 | EAGLE paper | [`06-sampling-decoding/speculative-decoding/eagle.md`](../06-sampling-decoding/speculative-decoding/eagle.md) |
| Medusa | Medusa paper | [`06-sampling-decoding/speculative-decoding/medusa.md`](../06-sampling-decoding/speculative-decoding/medusa.md) |
| MTP（多 token 预测） | DeepSeek-V3 paper | [`06-sampling-decoding/speculative-decoding/mtp.md`](../06-sampling-decoding/speculative-decoding/mtp.md) |
| n-gram prompt lookup | prompt-lookup-decoding | [`06-sampling-decoding/speculative-decoding/ngram.md`](../06-sampling-decoding/speculative-decoding/ngram.md) |
| Suffix decoding | Arctic Inference | [`06-sampling-decoding/speculative-decoding/suffix.md`](../06-sampling-decoding/speculative-decoding/suffix.md) |
| DFlash / DSpark | DFlash/DSpark | [`06-sampling-decoding/speculative-decoding/dflash.md`](../06-sampling-decoding/speculative-decoding/dflash.md)、[`dspark.md`](../06-sampling-decoding/speculative-decoding/dspark.md) |
| SpecDecoding rejection sampling | Chen et al. / Leviathan | [`06-sampling-decoding/rejection-sampler.md`](../06-sampling-decoding/rejection-sampler.md) |
| Xgrammar | Xgrammar | [`06-sampling-decoding/structured-output/backend-xgrammar.md`](../06-sampling-decoding/structured-output/backend-xgrammar.md) |
| Outlines / outlines_core | Outlines project | [`06-sampling-decoding/structured-output/backend-outlines.md`](../06-sampling-decoding/structured-output/backend-outlines.md) |
| Guidance / llguidance | Guidance | [`06-sampling-decoding/structured-output/backend-guidance.md`](../06-sampling-decoding/structured-output/backend-guidance.md) |
| lm-format-enforcer | lm-format-enforcer | [`06-sampling-decoding/structured-output/backend-lm-format-enforcer.md`](../06-sampling-decoding/structured-output/backend-lm-format-enforcer.md) |
| Punica | Punica project | [`12-lora/punica.md`](../12-lora/punica.md) |
| SGMV / BGMV | SGMV paper | [`12-lora/ops.md`](../12-lora/ops.md) |
| Megatron-LM parallel state | Megatron-LM | [`07-distributed/parallel-state.md`](../07-distributed/parallel-state.md) |
| torch.compile / Inductor | PyTorch | [`09-compilation-ir/compiler-interface.md`](../09-compilation-ir/compiler-interface.md) |
| CUDA graph | PyTorch CUDA graph | [`09-compilation-ir/cuda-graph.md`](../09-compilation-ir/cuda-graph.md) |
| Custom all-reduce | vLLM TamingLatency blog | [`07-distributed/device-communicators/custom-all-reduce.md`](../07-distributed/device-communicators/custom-all-reduce.md) |
| Sleep mode / cumem | vLLM RFC #34303 | [`08-platforms/device-allocator.md`](../08-platforms/device-allocator.md)、[`15-kv-cache-offload/sleep-mode.md`](../15-kv-cache-offload/sleep-mode.md) |
| DeepSeek-V2 / V3 / V4 | DeepSeek paper | [`04-model-zoo/architecture-families/deepseek.md`](../04-model-zoo/architecture-families/deepseek.md) |
| Qwen2 / Qwen3 / VL / MoE | Qwen paper | [`04-model-zoo/architecture-families/qwen.md`](../04-model-zoo/architecture-families/qwen.md) |
| GLM4 / GLM-4.7 | GLM paper | [`04-model-zoo/architecture-families/glm.md`](../04-model-zoo/architecture-families/glm.md) |
| Mamba / Mamba-2 | Mamba paper | [`04-model-zoo/architecture-families/mamba-ssm.md`](../04-model-zoo/architecture-families/mamba-ssm.md) |
| ColBERT / ColPali / ColQwen | ColBERT paper | [`04-model-zoo/architecture-families/embedding-col.md`](../04-model-zoo/architecture-families/embedding-col.md) |
| Whisper / FunASR | Whisper paper | [`04-model-zoo/architecture-families/speech-audio.md`](../04-model-zoo/architecture-families/speech-audio.md) |

[← 返回附录首页](../README.md)

## 参见

- [`conventions.md`](conventions.md)
- [`cross-reference.md`](cross-reference.md)
