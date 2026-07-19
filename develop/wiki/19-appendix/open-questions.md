# 待核实 / 待补充清单

[← Wiki 首页](../README.md) > [附录](../README.md) > 待核实清单

本页汇总各子 agent 在产出过程中显式标注 `(待核实)` / `(待补充)` 的内容。按子系统分组，每条标注源文件位置与原因，便于后续人工或新一轮 agent 收敛。

> 本清单为人工汇总初版；可用 `grep -rn '待核实\|待补充' develop/wiki/` 重新生成全量。

## 01-engine-core
- `kv-cache-management/metrics.md`：`KVCacheMetricsCollector` 引入版本（v0.9 待核实）；`kv_cache_metrics_sample` 默认值未在源码直接确认。
- `async-llm-frontend.md` / `engine-core-process.md`：weight transfer、`scale_elastic_ep`、`ECConnectorOutput`、`enable_return_routed_experts`、`tensor_ipc` torch_shm 等的精确版本归属。
- `coordinator.md` / `block-pool.md` / `spec.md`：DCP/PCP block_size 放大、NVFP4/INT4 spec、SlidingWindowMLASpec 等字段的精确引入版本。

## 02-execution
- `worker/model-runner-v2.md`：MRv2 迁移路线无公开时间表（待补充）。
- `worker/ubatching.md`：DBO 与 MRv2 集成仍在推进（待补充）。
- `worker/ec-connector-mixin.md`：EC connector backend（NIXL 等）实现仍在演进（待补充）。

## 03-model-execution
- 各 loader/layer/quantization 页面：PR 号与发行版本号多处标 `(待核实)`，CHANGELOG 未逐一核对。
- `model-loader/modelexpress.md`：`MxModelLoader` 是否完整复刻 base 模板（依赖外部包）。
- `layers/norm.md`：`Serialized…` 类名、`poly_norm` 用途。
- `layers/rejection-sampler-layer.md`：`mps/sync` 类/函数不存在，需重新归位。
- `layers/rotary.md`：DeepseekScaling mscale、MLA decoupled RoPE 细节。
- `layers/mamba-ssm.md`：`mamba_cache_mode` 取值；`Mixer2RMSNormGated` 与 `RMSNormGated` 关系。
- `layers/parameter.md`：OOT 厂商特定 Parameter 子类是否存在。
- `layers/sampler-layer.md`：`LogprobsMode` 枚举确切命名。

## 04-model-zoo
- `granite.md`：`granite_speech` vs `granite_speech_plus` 能力分档差异。
- `kimi.md`：`kimi_linear` 是否直接继承 `MLAAttention`。
- `vlm-misc.md`：`deepencoder.py`/`deepencoder2.py` 用途。
- 部分版本号按"中期/近期"标注，未逐一 git log。

## 05-attention
- `backends/`：`no_attention.py` backend 路径不存在，登记方式待核实。
- `backends/mla/`：`DeepseekV32/V4IndexerBackend` 注册路径。
- `backends/cpu.md`：`CpuArchEnum`（x86/ARM/RISC-V RVV）分支逻辑细节。
- DeepSeek V4 模型驱动的 sparse MLA backend 归属（在模型库侧待补）。

## 06-sampling-decoding
- `dspark.md`：drafter 实例化路径、`dspark_bonus_anchor` 语义、是否走通用 drafter。
- `mtp.md`：DeepSeek V2/V3 MTP 的 method 分支识别；V4 是否引入并行方案。
- `ngram.md`：`prompt_lookup_min/max` 默认值；与 thinking_budget 兼容性。
- `ngram-gpu.md`：max_ngram_len > 8 优化 commit。
- `rejection-sampler.md`：并行 drafting 下 max_spec_len 上限。
- `sampler.md`：logit_bias + LoRA 优先级。
- `eagle.md`：EAGLE 论文 arXiv 编号 `2401.15077` 待核实。
- `sampling-ops.md`：topk_topp_triton.py arXiv `2602.01518` 编号异常（疑似 typo）。
- `structured-output/`：与 `StructuredOutputsConfig` 字段交叉核对、`reasoning_parser_plugin` 加载路径。

## 07-distributed
- 多处 v0.8–main 函数集与平台 qualname 精确版本。
- `pynccl.md`/`all2all.md`：v0.11/main NCCL symm mem scheduler、ROCm DeepEP 兼容进展。
- `shm-object-storage.md`：multiproc executor 父↔子确切使用点。
- `offloading.md`/`flexkv.md`/`transports/*`：`kv_connector_extra_config` 部分字段名。
- `ec-transfer.md`：`register_caches` P2P 路径进度。
- `elastic-ep.md`：`execute_reconfigure_distributed` 多节点 stateless 突破确认。
- `weight-transfer.md`：`rebuild_cuda_tensor` 位置、layerwise reload 协同。

## 08-platforms
- v0.5–v0.6 早期实现细节无法从 git 可靠追溯。
- `tpu.md`：`tpu_inference` 外置包内 hook 覆写细节（6 处）。
- v0.11/main 部分 PR 时间线未一一对应版本号。

## 09-compilation-ir
- v0.5–v0.6 是否确无 piecewise/自定义 Pass。
- `InductorAdaptor` 最低支持 torch 版本。
- `VLLM_USE_BREAKABLE_CUDAGRAPH` 是否 v0.11+ 默认开启。
- `codegen.py` `tuple_return`/`donate_graph_module` 兼容点版本。
- `ir-ops.md`：`silu_and_mul`/`rotary_embedding` 迁入 IR plan、MLA `variance_size` 用例。
- `base-static-graph.md`：Protocol 是否显式声明 `cudagraph_wrapper`/`unwrap`。
- 多 stream 共享 graph pool 安全性 TODO 进度。
- ngram GPU kernel 临时禁 cache、`CUDAGraphMode.FULL_DECODE_ONLY/FULL_AND_PIECEWISE` 引入版本。

## 10-config
- `ModelArchitectureConfig`/`KernelConfig`/`IrOpPriorityConfig`/`OffloadConfig` 等 v0.8.x/v0.9 小版本（42 处）。
- `SpeechToTextConfig` 是否未在 `VllmConfig` 顶层聚合（由语音模型类内部持有）。
- `StructuredOutputsConfig.reasoning_*` 与 `ReasoningConfig` 过渡关系。
- `KVEventsConfig`/`WeightTransferConfig`/`ECTransferConfig` 精确引入小版本。

## 11-multimodal
- `v1-integration.md`：`v1/worker/gpu_model_runner.py` 与 `v1/worker/gpu/model_runner.py` 关系（两者 grep 都存在）。
- `gpu-ipc-memory.md`：`acquire` 在 `PyNvVideoCodecVideoBackendMixin` 内调用点。
- `evs.md`：EVS 剪枝率 `q` 配置来源、Qwen2.5-VL/Qwen3-VL 调用点。
- `media.md`：`extract_audio_from_video` 是否走 `load_audio_pyav`。
- `audio.md`：`target_sr`/`target_channels` 是否 CLI 暴露。
- `processing.md`：`_cached_apply_hf_processor` cache 接入点完整签名（1800 行未通读）。
- `video.md`：`PyNvVideoCodec` decoder surface 复用逻辑。

## 12-lora
- `resolver.md`：`LoRAResolver` 前端调用栈（待补充）、引入版本。
- `request.md`：`load_inplace` 引入版本。
- `lora-model.md`：`weights_mapper`/`skip_prefixes` 引入版本。
- `lora-weights.md`：`is_packed` 规范化版本。

## 13-entrypoints
- `openai/run-batch.md` 尾部 `main`/`run_batch` 行号。
- `serve/utils/ssl.py` 等小文件行号。
- gRPC proto schema 字段。
- Responses retrieve/cancel 持久化、realtime 事件族与 OpenAI Realtime API 完整对齐度。
- Rust frontend env 名；`fingerprint.py`/`orca_metrics.py` 稳定状态；generative scoring 端点确切路径；`bench/base.py` 接口。

## 14-tokenizers-transformers
- transformers_utils/`weights.md`、`detokenizer.md`、`tokenizer-group.md` 页缺失（源文件确实不存在对应模块，需调整或合并）。
- reasoning/ 已补 `reasoning.md`；renderers/ 已补 `renderers.md`。
- 各 `tool_parsers/` 子 parser 页未全部独立（README + abstract + mistral 现有，其余待补：hermes/pythonic/granite/granite-async/llama3-md）。

## 15-kv-cache-offload
- `base.md`：`CanonicalKVCaches` 衍生调用栈未 grep 到。
- `factory.md`：`create_spec` 在 `offloading/scheduler.py` 的调用行号。
- `cpu.md`：`swap_blocks_triton` `is_src_access_order_any` 落地 commit；`store_threshold` 调参案例。
- `tiering-fs.md`：`vllm.fs_io_C` C 扩展 build 配置入口。
- `tiering-obj.md`：Obj tier 落地 PR；Azure/GCS 原生 backend 计划。
- `tiering-p2p.md`：MNNVL/Mooncake backend 主线 commit；session 双向合并 PR 时间。
- `simple-kv-offload.md`：首次引入版本（v0.6?）。
- `sleep-mode.md`：EngineCore sleep 命令转 `reset_cache()` 调用顺序所在文件。

## 16-observability
- v0→v1 迁移细节版本归属。
- EngineCore 内 `dump_engine_exception` 调用位置。
- API server 中 `create_uvicorn_log_config` 与 `/metrics` ASGI 挂载位置。
- `usage_message.report_usage` 调用点、`enable_trace_function_call` 触发 env var 名。
- `_USAGE_ENV_VARS_TO_COLLECT` 添加时机。
- `[13-entrypoints/serve/instrumentator.md]` 目标页缺失（待补）。
- v1 `SamplingParams.logits_processors` 是否仍兼容 v0 `NoBadWordsLogitsProcessor` 签名。
- `enable_layerwise_nvtx_tracing`（ObservabilityConfig）与本目录 `layerwise_profile` 关系。
- uvicorn args 顺序兼容性（`UvicornAccessLogFilter` 硬编码 `record.args[2]`）。
- `NewLineFormatter` `\r\n` 设计动因。

## 17-utils-cross-cutting
- envs.md：每个 `VLLM_*` 变量引入版本未逐一核实；`VLLM_CPU_MOE_PREPACK` 等少量变量未源码确认。
- utils.md：`distributed_utils.py` 在当前版本**未发现**（实际散落 `vllm/distributed/`）；任务提示的子页面未拆。
- forward-context/parser/custom-ops/scalar-type/cute-utils：部分字段/算子 v0.9 vs v0.10 vs v0.11 版本归属。

## 18-build-ci-testing
- Rust 引入版本（v0.10?）；Rust frontend 功能边界。
- 各 build/CI 变更未细化版本号。

## 19-appendix
- 本清单为人工汇总，机械生成请用 `grep -rn '待核实\|待补充' develop/wiki/`。

[← 返回附录首页](../README.md)

## 参见

- [`conventions.md`](conventions.md)
- [`external-references.md`](external-references.md)
