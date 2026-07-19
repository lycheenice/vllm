# SpeculativeConfig（speculative.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > SpeculativeConfig

源码：`vllm/config/speculative.py`（约 1276 行）。`SpeculativeConfig` 描述投机解码的全部参数：方法、draft 模型/权重、num_speculative_tokens、draft TP、attention/MoE backend 覆写、ngram/suffix 配置、并行 drafting、动态 spec 调度、rejection sampling 策略。它是 `VllmConfig.speculative_config`（`None` 表示未启用），被 `vllm/v1/spec_decode/` 与 scheduler/model runner 消费。

## 是什么

`@config(config=ConfigDict(arbitrary_types_allowed=True))`（`speculative.py:79`）。

**方法类型**（`speculative.py:36`–74）

`SpeculativeMethod = Literal["ngram","medusa","mlp_speculator","draft_model","suffix","custom_class", EagleModelTypes, NgramGPUTypes, DSparkModelTypes]`。其中 `EagleModelTypes = Literal["eagle","eagle3","extract_hidden_states", MTPModelTypes, DFlashModelTypes]`。MTP 子类型极多（`deepseek_mtp`/`mimo_mtp`/`mimo_v2_mtp`/`glm4_moe_mtp`/`ernie_mtp`/`nemotron_h_mtp`/`qwen3_next_mtp`/`qwen3_5_mtp`/`longcat_flash_mtp`/`minimax_m3_mtp`/`bailing_hybrid_mtp`/`exaone_moe_mtp`/`exaone4_5_mtp`/`pangu_ultra_moe_mtp`/`step3p5_mtp`/`hy_v3_mtp`/`gemma4_mtp`/`mtp`/`glm_ocr_mtp`/`glm4_moe_lite_mtp`）。`NgramGPUTypes="ngram_gpu"`、`DFlashModelTypes="dflash"`、`DSparkModelTypes="dspark"`。

**核心字段**

| 字段 | 默认 | 含义 |
|---|---|---|
| `enforce_eager` | `None` | 覆写 `model_config.enforce_eager` |
| `num_speculative_tokens` | `None`(→draft config 或必填) | 投机 token 数 |
| `model` | `None` | draft 模型/eagle head/附加权重名 |
| `method` | `None`(自动检测) | `SpeculativeMethod` |
| `draft_tensor_parallel_size` | `None`(1 或同 target) | draft TP |
| `tensor_parallel_size` | `None` | （警告用，勿误传） |
| `quantization` | `None` | draft 量化方法 |
| `moe_backend` | `None`(继承 target) | draft MoE backend |
| `attention_backend` | `None` | draft attention backend（DFlash 需非 causal） |
| `max_model_len` | `None` | draft 最大长度（跳过 spec 用） |
| `revision`/`code_revision` | `None` | draft revision |

**高级控制**

| 字段 | 默认 | 含义 |
|---|---|---|
| `disable_padded_drafter_batch` | `False` | 禁 drafter 输入 padding（与 async 不兼容） |
| `use_local_argmax_reduction` | `False` | vocab-parallel local argmax，减通信 O(vocab)→O(2*tp) |
| `use_heterogeneous_vocab` | `False` | draft/target 词表不同（TLI 算法，需 `draft_model`） |
| `prompt_lookup_max`/`prompt_lookup_min` | `None`/`None` | ngram n-gram 窗口上下限 |
| `parallel_drafting` | `False` | 并行 drafting（所有 spec token 并行生成，需模型支持） |
| `num_speculative_tokens_per_batch_size` | `None` | 动态 spec：按 batch size 区间选 spec token 数 |

**draft 配置（`__post_init__` 派生）**：`target_model_config`/`target_parallel_config`/`draft_model_config`/`draft_parallel_config`/`draft_load_config`（均 `SkipValidation`，由 `from_model_parallel_configs` 静态方法填）。

**Suffix decoding**：`suffix_decoding_max_tree_depth=24`/`suffix_decoding_max_cached_requests=10000`/`suffix_decoding_max_spec_factor=1.0`/`suffix_decoding_min_token_prob=0.1`。

**Rejection sampling**：`rejection_sample_method`（`standard`/`synthetic`/`block`）、`synthetic_acceptance_rates`/`synthetic_acceptance_length`（互斥，`_resolve_synthetic_acceptance_rates` 把 length 转 rates）、`draft_sample_method`（`greedy`/`probabilistic`）。

**关键方法**：`compute_hash`（纳入 `uses_aux_hidden_states`——eagle3/extract_hidden_states/dflash/dspark 返回中间 hidden state 影响图；以及 `eagle_aux_hidden_state_layer_ids`）、`hf_config_override`（静态，把 DeepSeek V3/V4/Qwen3/GLM4/MiMo/Ernie/NemotronH/Bailing/Exaone/Pangu/Step3.5/HyV3/Gemma4 等的 CausalLM config 重写为对应 MTPModel config）、`use_eagle()`/`uses_dynamic_speculative_decoding()`/`max_num_new_slots_for_drafting` 等属性。

## 为什么

- **方法族庞大**：vLLM 投机解码涵盖 Eagle/Eagle3/MTP(20+ 子类)/ngram/ngram_gpu/draft_model/medusa/mlp_speculator/suffix/dflash/dspark/custom_class。`method` + `model` 配合自动检测（`_detect_method`），`hf_config_override` 把各家 CausalLM→MTPModel config 统一化。
- **draft 隔离**：draft 可有独立 TP/dtype/quantization/attention/MoE backend（如 DFlash draft 需非 causal attention backend，量化 target + 非量化 draft 需不同 MoE backend）。
- **async scheduling 兼容白名单**：`VllmConfig.__post_init` 限制 async scheduling 仅配 Eagle/MTP/draft/ngram_gpu/dspark；`disable_padded_drafter_batch` 与 async 互斥。
- **词表异构**：`use_heterogeneous_vocab` TLI 算法约束 draft logits 到交集 token，让 draft/target 词表不同也能 spec。
- **动态 spec**：`num_speculative_tokens_per_batch_size` 按 batch size 区间动态选 spec token 数，`VllmConfig._maybe_override_dynamic_sd_cudagraph_mode` 据此强制 PIECEWISE cudagraph（变长验证）。
- **rejection 策略**：`standard`(概率拒绝)/`synthetic`(衰减接受率，免 draft logits)/`block`(块验证)；`synthetic_acceptance_length` 把"目标平均接受长度"自动转 per-position rates（最小方差调度）。
- **`compute_hash` 精准**：仅 `uses_aux_hidden_states` 与 `eagle_aux_hidden_state_layer_ids` 影响图（额外返回中间层 hidden state），其余 spec 配置不改图形状。

## 怎么做

- **Eagle**：`--speculative-model model --speculative-method eagle --num-speculative-tokens 5`。
- **MTP**：`--speculative-model deepseek_v3 --speculative-method mtp`（`hf_config_override` 自动）。
- **ngram**：`--speculative-method ngram --prompt-lookup-max 4 --prompt-lookup-min 2`。
- **draft_model**：`--speculative-model /path/draft --speculative-method draft_model --draft-tensor-parallel-size 1`。
- **DSpark/DFlash**：`--speculative-method dspark`（V2 model runner，`use_v2_model_runner` 强制 True）。
- **动态 spec**：`--num-speculative-tokens-per-batch-size '[(0,15,5),(16,32,3),(33,999,1)]'`。
- **并行 drafting**：`--parallel-drafting`（V2 P-Eagle 除外）。
- **异步调度**：默认自动开（若 method 在白名单）；`--no-async-scheduling` 强制关。

## 与其它模块/系统配合

- **投机解码子系统（[`06-sampling-decoding/`](../06-sampling-decoding/README.md)）**：`method`/`num_speculative_tokens`/`draft_model_config` 驱动 `EagleProposer`/`MTPProposer`/`NgramProposer`/`DraftModelProposer`/`SuffixDecodingProposer` 等；`rejection_sample_method` 驱动采样器。
- **Scheduler（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`num_speculative_tokens` 影响 spec token 切片；`uses_dynamic_speculative_decoding()` 影响 cudagraph 模式；`update_draft_token_ids` 在 EngineCore.post_step。
- **SchedulerConfig（[scheduler-config.md](scheduler-config.md)）**：`max_num_scheduled_tokens` 由 `VllmConfig._set_max_num_scheduled_tokens` 按 `max_num_new_slots_for_drafting*max_num_seqs` 扣减。
- **ModelRunner（[`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md)）**：draft 模型加载与 drafter batch；`disable_padded_drafter_batch` 影响输入 padding；`parallel_drafting` 并行 token 生成。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：动态 spec → `cudagraph_mode=PIECEWISE`；`fast_moe_cold_start` 在 spec（draft 含 MoE）时关闭防静默错误；V2 不支持 ngram/ngram_gpu 等。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：async scheduling 白名单校验；`use_v2_model_runner` 在 dspark/dflash 时强制 True；cascade attention 在 async+spec 下关；`_validate_v2_model_runner` 限制 spec method 子集（V2 不支持 ngram/parallel_drafting EAGLE/eagle3+PP 等）。
- **DiffusionConfig（[diffusion-config.md](diffusion-config.md)）**：`num_speculative_tokens` 也可来自 `diffusion_config.canvas_length`（`VllmConfig.num_speculative_tokens` 优先 spec config）。

## 历史版本演进

- **v0.6**：speculative decoding 引入（ngram/Eagle v1/draft_model/medusa）。
- **v0.7（v1 落地）**：`SpeculativeConfig` v1 重写；Eagle/MTP/ngram 主线；`disable_padded_drafter_batch`；draft 独立 TP/quantization。
- **v0.8**：`parallel_drafting`；`use_local_argmax_reduction`；`use_heterogeneous_vocab`（TLI）。
- **v0.9**：Eagle3/extract_hidden_states（aux hidden states 进哈希）；dflash/dspark（并行 drafting 原生）；MTP 族扩充（GLM4/Ernie/NemotronH/Qwen3.Next/Bailing/Exaone/Pangu/Step3.5/HyV3 等）；suffix_decoding；`rejection_sample_method` 三态 + `synthetic_acceptance_length`。
- **v0.10**：`num_speculative_tokens_per_batch_size` 动态 spec；`moe_backend`/`attention_backend` draft 覆写；DSpark 上线（commit `f5a8d7337`）；V2 model runner 对 dense 默认开（`a2f713002`）影响 spec 兼容矩阵。
- **v0.11 / v0.12 / main**：MTP 族持续扩充（Gemma4/Qwen3.5/MiMoV2/LongcatFlash/MiniMaxM3 等）；`hf_config_override` 静态方法覆盖更多架构；V2 speculator 不支持特性集合细化。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [scheduler-config.md](scheduler-config.md) — `max_num_scheduled_tokens` 扣减与 async 白名单。
- [compilation-config.md](compilation-config.md) — 动态 spec 强制 PIECEWISE cudagraph。
- [diffusion-config.md](diffusion-config.md) — `canvas_length` 作为 `num_speculative_tokens` 备选源。
- [vllm-config.md](vllm-config.md) — `use_v2_model_runner`/cascade_attn/fast_moe_cold_start 联动。
- [../06-sampling-decoding/](../06-sampling-decoding/README.md) — 投机解码子系统消费方。
