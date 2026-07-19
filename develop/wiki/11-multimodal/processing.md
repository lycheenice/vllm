# processing/ · 多模态处理器核心

[← Wiki 首页](../README.md) > [多模态](../README.md) > processing

## 是什么

`vllm/multimodal/processing/` 子目录是子系统的"大脑"——把"用户 prompt + 解析后的 `MultiModalDataItems`"转化为"LLM 可消费的 `prompt_token_ids` + `mm_kwargs` + `mm_hashes` + `mm_placeholders`"。包含五个文件：

- `context.py`：`TimingContext`、`InputProcessingContext`（持有 `ModelConfig`/`tokenizer`/HF processor 缓存）、`BaseProcessingInfo`（模型侧 info 抽象：`supported_mm_limits` / `allowed_mm_limits` / `get_data_parser` / `get_mm_max_tokens_per_item` / `validate_num_items`）。
- `inputs.py`：`ProcessorInputs` dataclass + `get_mm_hashes` 实现。
- `dummy_inputs.py`：`BaseDummyInputsBuilder` 抽象 + `_get_dummy_audios/images/videos` helper。
- `processor.py`（约 1800 行，最大）：`PromptIndex`/`PromptIndexTargets`/`PromptUpdateDetails`/`UpdateMode`/`PromptUpdate`/`PromptInsertion`/`PromptReplacement`/`ResolvedPromptUpdate`/`PlaceholderFeaturesInfo`/`MultiModalProcessingInfo`/`BaseMultiModalProcessor`/`EncDecMultiModalProcessor` 全套抽象与算法。
- `__init__.py` 统一导出。

每个具体 VLM（如 `vllm/model_executor/models/llava.py`）通过子类化 `BaseProcessingInfo` / `BaseMultiModalProcessor` / `BaseDummyInputsBuilder` 并向 `MULTIMODAL_REGISTRY` 注册来接入。

## 为什么

HF 自带的 `ProcessorMixin`（如 `LlavaProcessor`/`Qwen2VLProcessor`）做了"text + media → input_ids + tensors"的事，但 vLLM 在它之上还需要解决：

1. **占位符位置不确定性**：HF processor 输出的 `input_ids` 里图像占位符的位置随 prompt 长度、图像数量、aspect ratio 变化，而 vLLM 调度器/编码器 runner 需要**精确知道每个媒体 item 的 `[offset, length)`**。`PromptUpdate` 体系让模型作者声明"在哪里放多少个什么 token"，框架自动 match 并算出 `PlaceholderRange`。
2. **缓存复用粒度**：HF processor 一次性输出全 batch tensor，而 vLLM 需要按 per-item 缓存。`_cached_apply_hf_processor` + `cache.is_cached`/`get_and_update` 把 HF 输出拆成 per-item，命中即跳过。
3. **旁路 embedding**：用户传 `image_embeds` 时不应进 HF image processor；`_hf_processor_applies_updates` 判定后走 text-only + passthrough 路径。
4. **编码器-解码器分离**：Whisper 等模型 encoder 与 decoder 输入不同 prompt，需要 `EncDecMultiModalProcessor` 重写 `apply`。
5. **dummy 输入**：profiling 需要最大尺寸 dummy 数据，`BaseDummyInputsBuilder` 提供"造最大图/视频/音频"统一模板，模型只需声明尺寸上限。
6. **可观测性**：`TimingContext.record("stage")` 把每个阶段（call_hf_processor / apply_prompt_updates / ...）耗时收集，供 `MultiModalTimingRegistry` 上报。

## 怎么做

### InputProcessingContext（context.py）

`InputProcessingContext`（`context.py:90`，frozen dataclass）持 `model_config` + `tokenizer`。关键方法：

- `get_hf_config(typ=None)`（`:121`）：取 `model_config.hf_config`，类型校验。
- `get_hf_image_processor_config()` / `get_mm_config()`：透传。
- `get_hf_processor(typ=None, **kwargs)`（`:179`）：用 `cached_processor_from_config` + `get_merged_mm_kwargs` 构造/复用 HF Processor；Mistral tokenizer 走 `transformers_tokenizer` 适配。
- `init_processor(typ, **kwargs)`（`:212`）：构造 HF 风格 processor 但不缓存（用于自定义 processor）。
- `call_hf_processor(hf_processor, data, kwargs={}, *, num_tries=1, max_tries=5)`（`:244`）：合并 mm kwargs，`get_allowed_kwarg_only_overrides` 过滤不允许的 kwargs，调 `hf_processor(**data, **allowed_kwargs)`，把输出 `BatchFeature` 的浮点叶子 cast 到 `model_config.dtype`。
- `get_merged_mm_kwargs(kwargs)`（`:240`）：`mm_config.merge_mm_processor_kwargs(kwargs)`。

### TimingContext

`TimingContext`（`context.py:48`，dataclass）：`enabled` + `stage_secs: dict[str, float]`。`record(stage)` 是 contextmanager，记录耗时；`get_stats_dict()` 输出 `f"{stage}_secs"` + `preprocessor_total_secs`。被 `MultiModalTimingRegistry` 按 request_id 管理。

### BaseProcessingInfo

`BaseProcessingInfo`（`context.py:296`）是模型侧 info 抽象：

- `model_id` / `get_tokenizer` / `get_hf_config` / `get_hf_processor` 透传 ctx。
- `get_default_tok_params()`（`:321`）：构造 `TokenizeParams(max_total_tokens=model_config.max_model_len, do_lower_case=encoder_config.get("do_lower_case", False), add_special_tokens=True)`。
- `_get_expected_hidden_size()`（`:338`）：`enable_mm_embeds` 时返回 `model_config.get_inputs_embeds_size()`，用于 parser 校验 embedding 形状。
- `get_data_parser()`（`:353`）：默认构造 `MultiModalDataParser(expected_hidden_size=...)`，模型可 override 加 `target_sr`/`target_channels`/`video_needs_metadata`。
- `supported_mm_limits`（abstract cached_property）：模型声明每模态上限（None 表无限）。
- `allowed_mm_limits`（`:393`）：`min(user_limit, supported_limit)`。
- `validate_num_items(modality, num_items)`（`:409`）：超限 raise `VLLMValidationError(parameter=modality)`，提示 `--limit-mm-per-prompt`。
- `parse_mm_data(mm_data, *, validate=True)`（`:430`）：用 `data_parser` 解析，校验 embedding 需 `--enable-mm-embeds` 且 `limit>0` 跳过校验。
- `get_mm_max_tokens_per_item(seq_len, mm_counts)`（`:465`）：默认 None（走 dummy 回退）；模型可 override 给快速路径。
- `skip_prompt_length_check`：默认 False，模型（某些超长 prompt 的）可置 True 跳过校验。

### ProcessorInputs（inputs.py）

`ProcessorInputs`（`inputs.py:13`，dataclass）：`prompt: str | list[int]`、`mm_data_items: MultiModalDataItems`、`mm_uuid_items: MultiModalUUIDItems | None`、`hf_processor_mm_kwargs`、`tokenization_kwargs`。

`get_mm_hashes(model_id)`（`:25`）：

- 对每个 modality 的每个 item：
  - 若 `mm_uuid_items` 提供 UUID 且 `hf_processor_mm_kwargs` 为空 → 透传 UUID 不哈希。
  - 若 UUID None 或有 processor kwargs → `MultiModalHasher.hash_kwargs(model_id=model_id, **{modality: item}, **hf_processor_mm_kwargs)`。
- 返回 `dict[str, list[str]]`。

### BaseDummyInputsBuilder（dummy_inputs.py）

`BaseDummyInputsBuilder`（`dummy_inputs.py:28`，`Generic[_I]`）abstract：

- `get_dummy_text(mm_counts)` / `get_dummy_mm_data(seq_len, mm_counts, mm_options)` abstract。
- `get_dummy_processor_inputs(seq_len, mm_counts, mm_options)`（`:67`）：调上述二者，`info.parse_mm_data(validate=False)`，构造 `ProcessorInputs(prompt, mm_data_items, tokenization_kwargs={"truncation": False})`。
- `_get_dummy_audios/images/videos(*, length/w/h/num_frames, num_*, overrides)`（`:94`/`:115`/`:147`）：造全 0 / 全 255 的最大尺寸 dummy；支持 `BaseDummyOptions` override（但 override 超过模型上限时 warning 后 ignore）。

### PromptUpdate 体系（processor.py）

`UpdateMode`（`:292`）：`INSERT` / `REPLACE`。

`PromptUpdate`（`:298`，abstract dataclass）：`modality: str`、`target: PromptUpdateTarget`（callable 或固定）、`content`/`mode` 是 abstract property。

- `resolve(item_idx)`（`:338`）：把 callable target/content 解析成具体值，返回 `ResolvedPromptUpdate`。

`PromptInsertion`（`:354`）：`mode=INSERT`，在 target 位置**插入**占位符。
`PromptReplacement`（`:423`）：`mode=REPLACE`，把 target **替换**为占位符。

`PromptIndexTargets`（`:141`）提供 `start()` / `end()` / `prefix(text)` 等便利 target。

`PromptUpdateDetails`（`:206`）描述占位符内容（token ids 或 text，可含特殊语义）。

### BaseMultiModalProcessor

`BaseMultiModalProcessor`（`:972`，`Generic[_I]`）是处理器主入口：

- `__init__(info, dummy_inputs, *, cache=None)`（`:979`）：持 `info` / `dummy_inputs` / `cache` / `data_parser`。
- `__call__(prompt, mm_items, mm_uuid_items=None, hf_processor_mm_kwargs=None)`（`:994`）：构造 `ProcessorInputs` 调 `apply(inputs, TimingContext(enabled=False))`。
- abstract：`_get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]`（声明字段拆分）、`_get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptUpdate]`（声明占位符规则）。

核心流程 `apply(inputs, timing_ctx)`（`:1663`）：

1. `_cached_apply_hf_processor(inputs, timing_ctx)` 调 HF processor（被 cache 包裹），返回 `(prompt_ids, mm_info, is_update_applied)`。
2. `_maybe_apply_prompt_updates`：若 HF 未应用 prompt updates（如 embedding 旁路），手动在 token 序列里 match target 并 insert/replace 占位符；同时构造 `mm_placeholders`。
3. `mm_placeholder_ranges = {modality: [item.to_range() for item in placeholders]}`。
4. 返回 `mm_input(prompt_token_ids, mm_kwargs, mm_hashes, mm_placeholders)`。

HF processor 调用分支（`:1135`-`:1398`）：

- `_apply_hf_processor_text_mm`：text + media 一起进 HF，正常路径。
- `_apply_hf_processor_text_only`：仅文本（无 media 时）。
- `_apply_hf_processor_tokens_only`：tokens 已给（无 media）。
- `_apply_hf_processor_mm_only`：仅 media（embeddings 旁路），HF 跑 image processor 但不 tokenize。
- `_apply_hf_processor_main`：dispatch 分支选择 + `_hf_processor_applies_updates` 判定。
- `_apply_hf_processor`：带 cache 的入口，按 `mm_hashes` 查 `cache.is_cached`，命中 `cache.get_and_update_item` 取缓存，未命中跑 HF 后写缓存。

`_get_hf_mm_data(mm_items)`（`:1083`）：从 items 取 `get_processor_data()` + `get_passthrough_data()` 两 dict，前者进 HF，后者直接 attach 到 HF 输出。

### EncDecMultiModalProcessor

`EncDecMultiModalProcessor`（`:1710`）重写 `apply`：

- abstract `create_encoder_prompt(prompt, mm_items)`：模型实现，把用户 prompt 转成 encoder 输入（如 Whisper 把多语言 prompt 拼成 encoder 输入）。
- `create_decoder_prompt(prompt, mm_items)`：默认透传。
- `_get_enc_dec_inputs`：tokenize decoder prompt。
- `apply(inputs, timing_ctx)`（`:1756`）：构造 encoder `ProcessorInputs`，调父类 `apply` 得 encoder mm input，再拼 decoder prompt，返回 `mm_enc_dec_input`。
- `skip_decoder_start_token`：类属性，控制 decoder 是否加 `<s>`。

## 与其它模块/系统配合

- **registry.py**：`register_processor(info, processor, dummy_inputs)` 三元组指向此处的子类；`create_processor` 构造之。
- **inputs.py**：`_get_mm_fields_config` 返回的 `MultiModalFieldConfig` 决定 HF 输出如何拆；`mm_placeholders` 即 `PlaceholderRange`。
- **parse.py**：`info.get_data_parser` 提供 `MultiModalDataParser`，`apply` 的 `mm_items` 来自它。
- **cache.py**：`_apply_hf_processor` 是 sender cache 的唯一调用点；命中即跳过 HF 计算。
- **hasher.py**：`ProcessorInputs.get_mm_hashes` 是 hash 入口。
- **encoder_budget.py**：`info.get_mm_max_tokens_per_item` 是预算快速路径；`get_dummy_mm_inputs` 调 `dummy_inputs.get_dummy_processor_inputs` + `apply`。
- **v1/engine/input_processor.py**：`process_inputs` 通过 `InputPreprocessor` 间接调 processor `apply`；`mm_features` 由 `apply` 输出装配。
- **v1/worker/gpu**：`EncoderRunner` 消费 `mm_kwargs`、`PlaceholderRange`；模型侧 `embed_multimodal`/`embed_prompt_ids` 是 `apply` 输出的最终消费方（详见 [v1-integration.md](v1-integration.md)）。
- **04-model-zoo**：每个 VLM 文件实现 `_I`/`_P`/`_D` 三个子类，详见 [模型库-VLM](../04-model-zoo/architecture-families/llava.md)。
- **renderers / 14-tokenizers**：`Renderer` 调 processor `apply`；`TokenizeParams` 与 `cached_processor_from_config` 来自 `14-tokenizers-transformers`。

## 历史版本演进

- **v0.5（LLaVA 初版）**：每个 VLM 自己写 `process_inputs`，无统一抽象；占位符位置硬编码或正则替换。
- **v0.6**：`BaseMultiModalProcessor` 引入，统一"调 HF + 展开占位符"；`PromptReplacement`（仅 replace 模式）。
- **v0.7（v1 化）**：`PromptInsertion` 加入支撑 LLaVA 等"插入"语义模型；`BaseProcessingInfo` 与 `BaseDummyInputsBuilder` 三件套分离；`ProcessorInputs` 与 `get_mm_hashes` 抽出。
- **v0.8**：`EncDecMultiModalProcessor` 加入支撑 Whisper；`skip_decoder_start_token` 类属性。
- **v0.9（hash+cache）**：`_apply_hf_processor` 加入 cache 接入点；`_hf_processor_applies_updates` 判定让 embedding 旁路走 text-only 路径。
- **v0.10**：`BaseDummyOptions` / `AudioDummyOptions` / `ImageDummyOptions` / `VideoDummyOptions` 加入，让 dummy 输入可 override（用于降低 profiling 时的尺寸）；`PromptIndexTargets` 便利 target。
- **v0.11（EVS）**：`TimingContext` + `MultiModalTimingRegistry` 加入可观测性；`_cached_apply_hf_processor` 把 cache 包裹逻辑统一到此层。
- **main**：`PromptReplacement`/`PromptInsertion` 的 target/content 支持 callable（per-item 动态规则）；`get_mm_max_tokens_per_item` 被越来越多模型 override 启动加速；`DictEmbeddingItems` 路径让 HF 输出风格的 dict 直接进 processor。

[← 返回多模态首页](../README.md)

## 参见

- [registry.md](registry.md)：processor 的构造与缓存装配入口。
- [inputs.md](inputs.md)：`_get_mm_fields_config` 输出的字段配置。
- [parse.md](parse.md)：`mm_items` 的来源。
- [cache.md](cache.md)：`_apply_hf_processor` 的缓存命中逻辑。
- [hasher.md](hasher.md)：`ProcessorInputs.get_mm_hashes`。
- [encoder-budget.md](encoder-budget.md)：`get_mm_max_tokens_per_item` 快速路径。
- [v1-integration.md](v1-integration.md)：`apply` 输出如何被 v1 引擎消费。
- [04-model-zoo/architecture-families/llava.md](../04-model-zoo/architecture-families/llava.md)：模型子类化示例。
