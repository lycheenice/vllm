[← 分词与转换器](../README.md) > [转换器工具](README.md) > config.py

# config.py — 模型配置加载与修补

## 是什么

`vllm/transformers_utils/config.py`（1215 行）是 vLLM 的"模型配置中心"。它把 `config.json`（HF 格式）或 `params.json`（Mistral 格式）解析为 `PretrainedConfig`，并对约 60 个未上游化模型注册自定义 `PretrainedConfig` 子类。

核心组件：

### 1. 注册表（`vllm/transformers_utils/config.py:62`）

- `_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]]`：`LazyConfigDict`（自定义 dict 子类），首次取值时把字符串名解析为 `vllm.transformers_utils.configs` 中实际的 config 类。覆盖 `afmoe`/`bagel`/`chatglm`/`colpali`/`colqwen3`/`deepseek_v32`/`deepseek_v4`/`deepseek_vl_v2`/`diffusion_gemma`/`eagle`/`flex_olmo`/`granite4_vision`/`hunyuan_vl`/`hy_v3`/`kimi_*`/`minimax_m3_*`/`moondream3`/`nemotron`/`olmo_hybrid`/`openvla`/`ovis`/`qwen3_*`/`step3*`/`speculators`/`ultravox`/约 60 项。
- `_SPECULATIVE_DECODING_CONFIGS = {"eagle", "speculators", "medusa"}`（`:130`）：spec 解码家族走独立加载路径。
- `_PATCH_HF_VALIDATE_ROPE = {"sarvam_mla"}`（`:132`）：需 monkey-patch transformers v5 RoPE 校验。
- `_AUTO_CONFIG_KWARGS_OVERRIDES`（`:138`）：internvl/NVLVM_D/Llama_Nemotron 等需要特殊 kwargs。

### 2. Parser 体系

- `ConfigParserBase`（来自 `config_parser_base.py`）：ABC，`parse(model, trust_remote_code, revision, code_revision, **kwargs) -> (config_dict, PretrainedConfig)`。
- `HFConfigParser`（`:207`）：默认 parser。先 `PretrainedConfig.get_config_dict`，再按 model_type 决定走 `_CONFIG_REGISTRY` 注册类还是 `AutoConfig.from_pretrained`；trust_remote_code 错误给出友好提示。
- `MistralConfigParser`（`:304`）：下载 `params.json` → `_maybe_retrieve_max_pos_from_hf` 补全 `max_position_embeddings` → `adapt_config_dict` 转 vLLM 配置；从 `consolidated.safetensors` 推断 dtype。
- `get_config_parser(config_format)` / `register_config_parser(name)` 装饰器（`:373`/`:380`）：注册自定义 parser。
- `ConfigFormat = Literal["auto","hf","mistral"]`（`:366`）。

### 3. 主入口 `get_config(...)`（`:654`）

```python
get_config(model, trust_remote_code, revision=None, code_revision=None,
           config_format="auto", hf_overrides_kw=None, hf_overrides_fn=None, **kwargs) -> PretrainedConfig
```

流程：auto 决策 → parser.parse → 补 architectures → 注入 `quantization_config`（含 `scale_fmt=ue8m0` 自动开启 `VLLM_USE_DEEP_GEMM_E8M0`）→ 应用 hf_overrides → `patch_rope_parameters`（含 sub_configs）→ trust_remote_code 时 `maybe_register_config_serialize_by_value`。

### 4. RoPE 修补

- `patch_legacy_rope_type(rope_parameters)`（`:440`）：legacy `type`/`su`/`mrope` → 现代 `rope_type`/`longrope`/`default`，处理嵌套 vs 非嵌套两种。
- `patch_rope_parameters(config)`（`:491`）：兼容 `rotary_emb_base`/`rotary_pct` 等非标准字段名。
- `uses_mrope` / `uses_xdrope_dim` / `is_encoder_decoder` / `is_interleaved`（`:513`/`:543`/`:560`/`:569`）：上层用这些 helper 决定采样/位置编码分支。

### 5. Pooling/Sentence-Transformer

- `get_pooling_config(model, revision)`（`:783`）：从 `modules.json` 提取 Pooling/N Normalize 配置，返回 `{seq_pooling_type, tok_pooling_type, use_activation}`。被 `ModelConfig` 用于 pooling runner 默认。
- `get_sentence_transformer_tokenizer_config(model, revision)`（`:871`）：扫描 7 种 `sentence_*_config.json`，返回 `{max_seq_length, do_lower_case}` 给 tokenizer 加载器（见 [../tokenizers/hf.md](../tokenizers/hf.md)）。
- `parse_pooling_type(name)`（`:857`）：`pooling_mode_lasttoken_*` → `last`，统一为大写短名。
- `try_get_dense_modules(model, revision)`（`:1097`）：sentence-transformers `Dense` 模块配置，给 pooling head 用。

### 6. GenerationConfig / Safetensors / Tokenizer config

- `try_get_generation_config(model, ...)`（`:1038`）：先试 `GenerationConfig.from_pretrained`，失败回退到 `GenerationConfig.from_model_config(config)`。
- `try_get_safetensors_metadata(model, revision)`（`:1065`+retry，返回 `model.safetensors` 元信息）；`get_safetensors_params_metadata`（`:1142`，把所有 safetensors 文件的 tensor metadata 合并；local / remote / local-cache 三级回退）。
- `try_get_tokenizer_config(path, trust_remote_code, revision)`（`:1082`）：包装 `transformers.get_tokenizer_config`。

### 7. 其它

- `maybe_override_with_speculators(model, tokenizer, ...)`（`:598`）：当 `speculators_config` 存在时，把目标模型替换为 verifier、注入 `SpeculatorsConfig.extract_vllm_speculative_config` 结果。
- `set_default_rope_theta(config, default_theta)`（`:431`）：缺 `rope_theta` 时设默认。
- `get_hf_image_processor_config(model, ...)`（`:1007`）、`get_hf_text_config(config)`（`:1021`）、`_maybe_remap_hf_config_attrs`（`:588`，`llm_config`→`text_config`）。
- `maybe_register_config_serialize_by_value()`（`:931`）：trust_remote_code 时把 `transformers_modules` 通过 cloudpickle 按 value 序列化，让自定义 config 类跨进程/ray 可 pickle。
- 模块顶部硬性要求 `transformers >= 5.0.0`（`:55`），低于则报 "removed in vLLM v0.24.0"。
- `chat_templates/` 子包：`_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK`（blip-2/chameleon/clip/colpali/deepseek_ocr/deepseek_vl_v2/fuyu/minicpmv/paligemma/siglip 等）+ `register_chat_template_fallback_path`/`get_chat_template_fallback_path`（`vllm/transformers_utils/chat_templates/registry.py:25`）。

## 为什么

- **未上游化模型**：DeepSeek/Qwen3.5/Kimi-K25/Step3/Hunyuan 等上线节奏远快于 transformers 官方，vLLM 必须自带 config 才能正确解析 `interleaved attention`/`mrope_section`/`speculators_config`/`xdrope_section` 等字段，因此 `_CONFIG_REGISTRY` 持续增长。
- **Mistral 私有格式**：`params.json` 没有 `architectures`/`model_type`，直接喂给 `AutoConfig` 会失败；`MistralConfigParser.adapt_config_dict` 把它转成 vLLM 内部配置。
- **RoPE 字段混乱历史**：transformers v4→v5 把 `rope_scaling.type` 改为 `rope_type`、把 `su` 重命名为 `longrope`、把 `mrope` 折叠进 `default`；vLLM 必须兼容老 checkpoint。
- **trust_remote_code 序列化**：自定义 config 类来自 `transformers_modules.xxx.yyy.ZConfig`，在 worker 进程里不存在；用 cloudpickle by-value 序列化是社区唯一可靠 workaround。
- **speculators 自动展开**：speculators checkpoint 同时是"草稿+验证器"，vLLM 需要在加载前就把它拆开。

## 怎么做

注册一个新模型配置的两种方式：

1. **新建 `configs/<arch>.py`**：实现 `class XConfig(PretrainedConfig)`，在 `_CONFIG_REGISTRY` 加一行 `"model_type": "XConfig"`。`LazyConfigDict` 会在首次访问时 import。
2. **远程代码**：`trust_remote_code=True` 时 transformers 自动下载 `modeling_xxx.py`，`maybe_register_config_serialize_by_value` 保证可序列化。

注册新 parser：

```python
@register_config_parser("my_format")
class MyConfigParser(ConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, code_revision=None, **kw):
        ...
        return config_dict, config
```

注册 chat template fallback：

```python
register_chat_template_fallback_path("my_model_type", Path("/path/template.jinja"))
# 或 callable 形式，按 tokenizer_name_or_path 动态选模板
```

## 与其它模块/系统配合

- **`tokenizers/registry.py`**：`cached_tokenizer_from_config` 调 `_maybe_register_hf_config(hf_config)` 与 `get_config`（用于注册到 AutoConfig，使 tokenizer `from_pretrained` 内部 `AutoConfig.from_pretrained` 也用对类）。
- **`configs/`**：约 80 个 `PretrainedConfig` 子类，部分被 `_CONFIG_REGISTRY` 引用，部分仅由远程代码使用。
- **`model_arch_config_convertor.py`**：把 `PretrainedConfig` 进一步转成 vLLM 内部 `ModelArchitectureConfig`，给 ModelRunner V2 用。
- **`vllm/config/model.py`**：`ModelConfig.get_hf_config` 让其变懒加载；`hf_overrides` 在这里落地。
- **`vllm/model_executor/model_loader/weight_utils.py`**：用 `get_safetensors_params_metadata` 拿 dtype/tensor 信息以做权重加载决策。
- **`vllm/transformers_utils/repo_utils.py`**：所有下载/缓存逻辑的真实实现。
- **`vllm/transformers_utils/configs/speculators/`**：单独子包（`base.py`/`algos.py`），与 [`06-sampling-decoding/speculative-decoding/`](../../06-sampling-decoding/speculative-decoding/README.md) 协同。

## 历史版本演进

- **早期（v0.5 之前）**：`get_config` 仅调 `AutoConfig.from_pretrained`，无注册表。
- **v0.6（PR #7739）**：`MistralConfigParser` + `_mistral_patch_hf_hub_constants` 加入，`config_format` 概念出现。
- **v0.7–v0.9**：`_CONFIG_REGISTRY` 快速膨胀，`configs/` 子目录成型。
- **v0.10**：`patch_legacy_rope_type` 收敛 transformers RoPE 字段历史变更；DeepSeek/Kimi 等大量加入。
- **v0.11/main**：强制 transformers ≥ 5.0.0；`speculators` 配置族加入；`scale_fmt=ue8m0` → `VLLM_USE_DEEP_GEMM_E8M0` 自动开启（DeepGEMM 协同）；`get_safetensors_params_metadata` 加入三级回退；`chat_templates/` 成为带 registry 的子包。

---

[← 返回转换器工具首页](README.md)

## 参见

- [processor.md](processor.md) — 配置加载完后接着加载 processor。
- `../utils.md` — `repo_utils` / `utils` / `dynamic_module` helper 速查。
- [../tokenizers/registry.md](../tokenizers/registry.md) — `_maybe_register_hf_config` 的对端。
- [`../../04-model-zoo/registry.md`](../../04-model-zoo/registry.md) — 与模型注册表的协同。
