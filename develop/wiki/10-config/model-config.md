# ModelConfig（model.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ModelConfig

源码：`vllm/config/model.py`（约 2280 行）。`ModelConfig` 是描述"跑什么模型、怎么跑"的核心子配置：模型路径、dtype、分词、量化、最大长度、多模态、pooler、runner 类型、HF 配置覆写等。它是 `VllmConfig.model_config`，默认为 `None`（因构造 `ModelConfig` 会触发 HF 配置下载，昂贵），由 `EngineArgs.create_engine_config` 显式注入。

## 是什么

`@config(config=ConfigDict(arbitrary_types_allowed=True))` 装饰（`model.py:106`）。用 `InitVar` 传 `max_model_len`/`is_encoder_decoder` 给 `__post_init__`（与 `SchedulerConfig` 同模式）。关键字段（`model.py:110` 起）：

### 模型与权重

| 字段 | 默认 | 含义 |
|---|---|---|
| `model` | `"Qwen/Qwen3-0.6B"` | HF 模型名/路径，兼作 metrics `model_name` |
| `model_weights` | `""` | RunAI 对象存储原始 URI（`model` 指向本地解包后目录时保留原 URI） |
| `runner` | `"auto"` | `auto`/`generate`/`pooling`/`draft` |
| `convert` | `"auto"` | `auto`/`none`/`embed`/`classify`（用 `vllm.model_executor.models.adapters` 改装） |
| `tokenizer` | `None` | 分词器名/路径，未设则同 `model` |
| `tokenizer_mode` | `"auto"` | `auto`/`hf`/`slow`/`mistral`/`deepseek_v32`/`deepseek_v4` |
| `trust_remote_code` | `False` | 信任 HF 远程代码 |
| `dtype` | `"auto"` | `auto`/`half`/`float16`/`bfloat16`/`float`/`float32` 或 `torch.dtype` |
| `seed` | `0` | 全局随机种子（TP 各 rank 必须一致） |
| `hf_config` | init=False | 解析后的 `PretrainedConfig` |
| `hf_text_config` | init=False | 文本子配置（多模态时为 `config.text_config`） |
| `hf_config_path` | `None` | 显式 HF config 路径 |
| `revision`/`code_revision`/`tokenizer_revision` | `None` | 各类 revision |
| `config_format` | `"auto"` | `auto`/`hf`/`mistral` |
| `hf_token` | `None` | HTTP bearer token |
| `hf_overrides` | `{}` | dict 或 `Callable[[PretrainedConfig], PretrainedConfig]`，覆写 HF config |
| `model_class_overrides` | `{}` | `arch→"module:class"`，运行期注册到 `ModelRegistry`（仅调试用） |
| `generation_config` | `"auto"` | `auto`/`vllm`/路径，影响 `max_new_tokens` 服务级上限 |
| `override_generation_config` | `{}` | 与 generation config 合并的覆写 |

### 长度与量度

| 字段 | 默认 | 含义 |
|---|---|---|
| `max_model_len` | `None`(自动派生) | 上下文长度；支持 `1k`/`1K`/`25.6k`/`-1`(自动适配显存) |
| `spec_target_max_model_len` | `None` | 投机解码 draft 模型最大长度 |
| `quantization` | `None` | 量化方法名（如 `awq`/`gptq`/`fp8`...） |
| `quantization_config` | `None` | `QuantizationConfigArgs`，在线量化规范（见 [quantization-config.md](quantization-config.md)） |
| `allow_deprecated_quantization` | `False` | 允许过期量化方法 |
| `enforce_eager` | `False` | 强制 eager，关 torch.compile 与 cudagraph |
| `enable_return_routed_experts` | `False` | 返回 routed experts（与 PP/KV connector 互斥） |
| `max_logprobs` | `20` | 返回 logprobs 上限（`-1`=无上限，OOM 风险） |
| `logprobs_mode` | `"raw_logprobs"` | `raw_logprobs`/`processed_logprobs`/`raw_logits`/`processed_logits` |
| `use_fp64_gumbel` | `False` | Gumbel-max 用 FP64 保低尾 |
| `disable_sliding_window` | `False` | 关闭 sliding window |
| `disable_cascade_attn` | `True` | 关 cascade attention（默认关，需手动开） |
| `skip_tokenizer_init` | `False` | 跳过分词器初始化，只接 token_ids |
| `enable_prompt_embeds` | `False` | 允许文本 embedding 作为输入 |
| `served_model_name` | `None` | API 暴露的模型名（可多个） |
| `allowed_local_media_path` | `""` | 允许 API 读本地媒体目录 |
| `allowed_media_domains` | `None` | 多模态 URL 域名白名单 |
| `logits_processors` | `None` | 自定义 logits processor 列表/路径 |

### 子配置（内含）

- `multimodal_config: MultiModalConfig`（见 [multimodal-config.md](multimodal-config.md)）
- `pooler_config: PoolerConfig | None`（见 [pooler-config.md](pooler-config.md)）
- `model_arch_config: ModelArchitectureConfig`（见 [model-arch.md](model-arch.md)），由 `get_model_arch_config()` 从 `hf_config` 派生

### 关键派生属性（节选）

`is_moe`/`is_hybrid`/`is_attention_free`/`is_encoder_decoder`/`is_multimodal_model`/`is_diffusion`/`architecture`/`runner_type`/`use_mla`/`attention_chunk_size`/`get_hidden_size`/`get_head_size`/`get_num_kv_heads`/`get_sliding_window`/`get_vocab_size`/`get_num_attention_heads`/`get_layers`/`get_text_lengths`——大多代理到 `hf_text_config` 与 `model_arch_config`。

`get_model_arch_config()`（`model.py` 中段）经 `MODEL_ARCH_CONFIG_CONVERTORS` 把 `PretrainedConfig` 转 `ModelArchitectureConfig`，处理 hybrid、DeepSeek MLA、image-bidirectional 等特例。

`iter_architecture_defaults`/`try_match_architecture_defaults`/`str_dtype_to_torch_dtype` 为模块级辅助函数（从 `__init__.py` 导出）。

`verify_with_parallel_config(parallel_config)` 校验 TP 可整除注意力头数等；`verify_dual_chunk_attention_config(load_config)` 校验 DCA 兼容性。

## 为什么

- **单一模型描述面**：vLLM 支持 ~280 个架构 + 多任务（generate/pooling/classify/draft）+ 多量化 + 多模态。`ModelConfig` 把这些维度收口，避免散落各处。`hf_config`/`hf_text_config` 缓存避免重复解析。
- **延迟构造**：默认 `None` 让 `VllmConfig` 可在未指定模型时构造（测试/平台默认推导），真正下载在 `EngineArgs` 显式注入时发生。
- **派生量集中**：`is_moe`/`use_mla`/`get_hidden_size` 等派生属性让其它子系统（编译、内核、注意力）通过统一接口读模型规格，不直接碰 `hf_config`。
- **adapter 友好**：`convert`/`model_class_overrides`/`hf_overrides` 让 classify/embed 改装与调试 override 不改源码。
- **`compute_hash`**：`ModelConfig.compute_hash` 把架构/dtype/量化/multimodal/pooler 等纳入指纹（`VllmConfig.compute_hash` 中调用），量化配置 `quant_config` 因已被 `model_config.quantization` 覆盖而不再单列。

## 怎么做

- **CLI 设置**：`--model Qwen/Qwen3-0.6B --dtype bfloat16 --max-model-len 32768 --quantization fp8` 等，由 `EngineArgs` 映射。
- **派生长度**：`max_model_len=None` 时从 `hf_config.max_position_embeddings` 等派生；`-1`/`'auto'` 时按显存自动选最大可行长度。
- **runner 解析**：`runner="auto"` 时按 `convert`/`pooler_config`/`hf_config` 推断 `runner_type`；`convert="auto"` 时按 `pooling_config`/`sentence_transformer` config 决定是否走 `embed`/`classify` adapter。
- **多模态激活**：检测 `hf_config` 含视觉/音频子配置 → 设 `multimodal_config`，`is_multimodal_model=True`，影响 `SchedulerConfig.is_multimodal_model` 与编码器缓存预算。
- **架构默认匹配**：`try_match_architecture_defaults` 在支持列表内查 architectures，回填 `ModelArchitectureConfig` 的默认值。

## 与其它模块/系统配合

- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init__` 调 `verify_with_parallel_config`/`verify_dual_chunk_attention_config`/`try_verify_and_update_config`；`is_moe` 回写 `parallel_config.is_moe_model`；`enforce_eager` 触发 `compilation_config.mode=NONE`。
- **模型库（[`04-model-zoo/`](../04-model-zoo/README.md)）**：`architecture`/`architectures` + `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 决定加载哪个 vLLM 模型类；`model_class_overrides` 运行期注入。
- **量化（[quantization-config.md](quantization-config.md) 与 [`03-model-execution/layers/quantization/`](../03-model-execution/layers/quantization/README.md)）**：`quantization` + `quantization_config` + `LoadConfig` 经 `VllmConfig._get_quantization_config` 派生 `quant_config`，校验 dtype/capability 兼容。
- **多模态（[multimodal-config.md](multimodal-config.md) 与 [`11-multimodal/`](../11-multimodal/README.md)）**：`multimodal_config` 注入；`mm_encoder_*` 字段控制 ViT 编码器 TP/FP8。
- **pooling（[pooler-config.md](pooler-config.md)）**：`pooler_config` 挂在 `model_config` 上；`runner_type="pooling"` 关闭 `async_scheduling`（`VllmConfig` 中）。
- **分词器（[`14-tokenizers-transformers/`](../14-tokenizers-transformers/README.md)）**：`tokenizer`/`tokenizer_mode`/`tokenizer_revision` 驱动分词器加载。
- **`SpeechToTextConfig`（[speech-to-text-config.md](speech-to-text-config.md)）**：Whisper 等模型时配合用。
- **平台（[`08-platforms/`](../08-platforms/README.md)）**：`current_platform` 在 `apply_config_platform_defaults` 中按 `dtype`/`enforce_eager` 调整。

## 历史版本演进

- **v0.5/v0.6（v0）**：`ModelConfig` 已存在，字段较少；`dtype`/`quantization`/`max_model_len` 主要字段就位；`hf_config` 在 `__post_init__` 下载并解析。
- **v0.7（v1 落地）**：抽出 `MultiModalConfig`/`PoolerConfig` 子配置内嵌；`runner_type` 三态；`ModelArchitectureConfig` 概念初现（仍散在 ModelConfig 派生属性里）。
- **v0.8**：`config_format`（mistral 支持）；`hf_overrides` 支持 callable；`model_class_overrides` 调试钩子上线；`disable_cascade_attn` 默认改为 `True`（保守）。
- **v0.9**：`enable_return_routed_experts`；`logits_processors` plugin entry_points；`spec_target_max_model_len`；`quantization_config: QuantizationConfigArgs` 字段加入（在线量化）。
- **v0.10**：`logprobs_mode` 四态；`use_fp64_gumbel`；`allowed_media_domains` 安全字段；reasoning parser 集成；`mm_encoder_*` FP8 ViT 字段扩充。
- **v0.11 / v0.12 / main**：`ModelArchitectureConfig` 独立为 `model_arch.py`（`get_model_arch_config` 走 `MODEL_ARCH_CONFIG_CONVERTORS`）；DeepSeek V4 / Qwen3.5 / GLM4 / Bailing / Exaone / NemotronH / PanguUltra / Step3.5 等 MoE+MTP 架构批量接入；RunAI 对象存储 `ObjectStorageModel`；`is_diffusion` 属性（配合 `DiffusionConfig`）；`tokenizer_mode` 新增 `deepseek_v32`/`deepseek_v4`。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-arch.md](model-arch.md) — 运行时架构派生量。
- [quantization-config.md](quantization-config.md) — 在线量化规范。
- [multimodal-config.md](multimodal-config.md) — 多模态字段。
- [pooler-config.md](pooler-config.md) — pooling 模型输出聚合。
- [vllm-config.md](vllm-config.md) — `__post_init` 中对 ModelConfig 的校验。
- [../04-model-zoo/README.md](../04-model-zoo/README.md) — 模型注册表与架构分类。
