# registry.py · 多模态注册中心

[← Wiki 首页](../README.md) > [多模态](../README.md) > registry

## 是什么

`vllm/multimodal/registry.py` 定义全局多模态注册中心 `MultiModalRegistry` 与单例 `MULTIMODAL_REGISTRY`（`vllm/multimodal/__init__.py:7`）。它把每个支持多模态的模型类（实现 `SupportsMultiModal`）与三个工厂函数 —— `ProcessingInfoFactory`、`MultiModalProcessorFactory`、`DummyInputsBuilderFactory` —— 绑定，并在运行时按 `ModelConfig` 懒构造出对应的 `BaseMultiModalProcessor` / `BaseProcessingInfo` / `BaseDummyInputsBuilder`。除派发外，它还封装"按 `VllmConfig` 决定使用哪种 `BaseMultiModalProcessorCache`"的逻辑。

## 为什么

vLLM 支持约 280 个模型架构，其中 VLM/ALM 数十个家族，每家的 HF Processor 行为、占位符语义、最大 token 数计算方式都不同。若在引擎核心里硬编码分支，会导致 `v1/engine` 与具体模型耦合死。注册中心模式把"数据如何被处理"的知识下沉到模型侧：模型作者在自己文件里用 `@MULTIMODAL_REGISTRY.register_processor(...)` 装饰类，引擎核心只持有一个 `MULTIMODAL_REGISTRY` 引用，调用 `create_processor(model_config, cache=...)` 即可获得一个统一接口的 processor。同时，缓存类型选择（`None` / `processor_only` / `lru` / `shm`）依赖 `parallel_config` 与 `mm_processor_cache_type`，集中在此处避免散落判断。

## 怎么做

### 工厂三元组

`_ProcessorFactories`（`registry.py:82`）是 frozen dataclass，持有 `info` / `processor` / `dummy_inputs` 三个 callable。`build_processor`（`:87`）按顺序：先用 `ctx` 构造 `info`，再用 `info` 构造 `dummy_inputs_builder`，最后用两者 + `cache` 构造 `processor`。这种顺序源于 `BaseDummyInputsBuilder` 依赖 `BaseProcessingInfo`，而 `BaseMultiModalProcessor` 依赖两者。

### 注册 API

`register_processor(processor, *, info, dummy_inputs)`（`:142`）返回一个装饰器，把 `_ProcessorFactories` 写到模型类的 `_processor_factory` 属性上。若已存在则 warning 后覆盖。模型类不需要继承特定基类，只要被装饰并实现 `SupportMultiModal` 即可。

### 模型定位

`_get_model_cls`（`:176`）通过 `vllm.model_executor.model_loader.get_model_architecture` 从 `ModelConfig` 解析架构类，并校验 `_processor_factory` 存在；缺失则抛 `"has no registered multimodal processor"`。`_create_processing_ctx`（`:188`）用 `cached_tokenizer_from_config` 复用 tokenizer，构造 `InputProcessingContext`。

### 入口方法

- `supports_multimodal_inputs(model_config)`（`:103`）：先看 `is_multimodal_model`，再尝试创建 `BaseProcessingInfo`（容忍 `ValueError` 转为 text-only），最后检查"所有支持模态的 `limit_per_prompt` 是否都为 0"。若全 0 但 `enable_mm_embeds=True`，仍返回 `True`（需要 MM 基础设施跑预计算 embedding）。
- `create_processor(model_config, *, tokenizer, cache)`（`:211`）：组合 ctx + factories + cache，返回 `BaseMultiModalProcessor`。
- `get_processing_info(model_config)`（`:208`）：只构造到 `info` 层，用于 `supports_multimodal_inputs` 与预算计算。
- `get_dummy_mm_inputs(model_config, mm_counts, *, cache, processor)`（`:232`）：用 `processor.dummy_inputs.get_dummy_processor_inputs` 造最大尺寸 dummy 数据，再 `processor.apply` 跑一遍得到 `mm_placeholders`；用于 profiling 和 `get_mm_max_toks_per_item` 的回退路径。

### 缓存类型决策

`_get_cache_type(vllm_config)`（`:268`）按下面顺序返回 `None | "processor_only" | "lru" | "shm"`：

1. 不支持 MM 输入 → `None`。
2. `mm_processor_cache_gb <= 0` → `None`（禁用缓存）。
3. IPC 缓存需要 `_api_process_count == 1` 且（`data_parallel_size == 1` 或 `data_parallel_external_lb`）；不满足 → `"processor_only"`（仅 P0 LRU，IPC 数据走 msgspec）。
4. 否则返回 `mm_config.mm_processor_cache_type`（用户可选 `lru` 或 `shm`）。

`processor_cache_from_config` / `processor_only_cache_from_config` / `engine_receiver_cache_from_config` / `worker_receiver_cache_from_config`（`:294`-`:347`）四个工厂分别构造 P0 sender 缓存与 P1 receiver 缓存，详见 [cache.md](cache.md)。

### 计时注册

`MultiModalTimingRegistry`（`:350`）按 `ObservabilityConfig.enable_mm_processor_stats` 决定是否记录 per-request 的 `TimingContext`，供可观测性子系统抽取阶段耗时。

## 与其它模块/系统配合

- **模型库**：每个 VLM 文件（如 `vllm/model_executor/models/llava.py`、`qwen2_5_vl.py`）通过装饰器向 `MULTIMODAL_REGISTRY` 注册，详见 [模型库-VLM](../04-model-zoo/architecture-families/llava.md)。
- **v1 InputProcessor**：`vllm/v1/engine/input_processor.py:58` 调用 `supports_multimodal_inputs` 判定走 MM 路径；`InputPreprocessor` 内部也用同一 registry 拿 processor（详见 [v1-integration.md](v1-integration.md)）。
- **Scheduler / MultiModalBudget**：`vllm/multimodal/encoder_budget.py` 与 `vllm/v1/core/sched/scheduler.py:205` 都通过 `MULTIMODAL_REGISTRY` 构造 `MultiModalBudget`，进而确定 `encoder_compute_budget`（详见 [encoder-budget.md](encoder-budget.md)）。
- **cache.py**：所有具体缓存类由本文件按 config 实例化，二者是"工厂-产品"关系（详见 [cache.md](cache.md)）。
- **processing/**：实际处理器实现位于 `vllm/multimodal/processing/`，本文件只负责"找到并构造它"（详见 [processing.md](processing.md)）。

## 历史版本演进

- **v0.5（LLaVA 初版）**：`MultiModalRegistry` 主要维护 `input_mapper` 与 `processing_plugin` 两类注册项，每个 modality 一组；模型侧分散实现。无统一 `BaseMultiModalProcessor`。
- **v0.6**：引入 `BaseMultiModalProcessor` 抽象，把"调用 HF Processor + 占位符展开"统一到框架；`register_processor` 取代旧 `register_input_mapper`。
- **v0.7（v1 化）**：新增 `ProcessingInfoFactory` / `DummyInputsBuilderFactory` 三元组，`get_dummy_mm_inputs` 用于 profiling；`_get_cache_type` 出现，区分 `processor_only` 与 IPC 路径。
- **v0.9（hash+cache）**：`processor_only_cache_from_config` / `engine_receiver_cache_from_config` / `worker_receiver_cache_from_config` 拆分，支持 P0/P1 镜像 LRU；`enable_mm_embeds` 路径合入 `supports_multimodal_inputs`。
- **v0.10**：`ShmObjectStoreSenderCache` / `ShmObjectStoreReceiverCache` 加入，`_get_cache_type` 引入 `shm` 分支与 `data_parallel_external_lb` 例外。
- **v0.11（EVS）**：`MultiModalTimingRegistry` 引入，配合 `TimingContext` 落地可观测性（`enable_mm_processor_stats`）。
- **main**：`enable_mm_embeds` 下"全 0 限流但需 MM 基础设施"的分支显式区分 `tower_modalities` 与 `embed_only_modalities`（影响 `encoder_budget`），`video_needs_metadata` / `expected_hidden_size` 透传到 `MultiModalDataParser`。

[← 返回多模态首页](../README.md)

## 参见

- [cache.md](cache.md)：被本文件实例化的四种缓存实现。
- [processing.md](processing.md)：被本文件构造的 processor 体系。
- [encoder-budget.md](encoder-budget.md)：使用 `get_dummy_mm_inputs` 计算预算。
- [v1-integration.md](v1-integration.md)：`MULTIMODAL_REGISTRY` 在 v1 引擎中的所有调用点。
