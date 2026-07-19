# ModelArchitectureConfig（model_arch.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > ModelArchitectureConfig

源码：`vllm/config/model_arch.py`（约 63 行）。`ModelArchitectureConfig` 是 vLLM **运行时**需要的"架构派生量"——把 `transformers.PretrainedConfig` 中分散且模型各异的字段，规范为一个固定 schema 的 dataclass，供编译/注意力/块池等子系统统一消费。由 `ModelConfig.get_model_arch_config()` 经 `MODEL_ARCH_CONFIG_CONVERTORS` 转换器从 `hf_config` 派生，挂在 `ModelConfig.model_arch_config`。

## 是什么

`@dataclass(config=ConfigDict(arbitrary_types_allowed=True))`（注意：**不**用 `@config`，因作为内嵌派生对象无需独立 CLI 入口）。字段（`model_arch.py:19` 起）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `architectures` | `list[str]` | 模型架构类名（如 `['LlamaForCausalLM']`）；`with_hf_config(config.text_config)` 时可为 `None` |
| `model_type` | `str` | 模型类型标识（如 `llama`/`gpt_oss`） |
| `text_model_type` | `str \| None` | 文本子模型类型（多模态用，如 `llama4_text`） |
| `hidden_size` | `int` | 隐藏维度 |
| `total_num_hidden_layers` | `int` | 总层数 |
| `total_num_attention_heads` | `int` | 总注意力头数 |
| `head_size` | `int` | 每头维度 |
| `vocab_size` | `int` | 词表大小 |
| `total_num_kv_heads` | `int` | KV 头数 |
| `num_experts` | `int` | 专家数 |
| `quantization_config` | `dict[str, Any] \| None` | 量化配置 dict |
| `is_deepseek_mla` | `bool` | 是否 DeepSeek MLA |
| `is_mm_prefix_lm` | `bool` | 多模态图像是否双向注意力 |
| `rswa_window` | `int \| None` | 参考 Sliding Window 窗口（None 关闭 R-SWA） |
| `derived_max_model_len_and_key` | `tuple[float, str \| None]` | 派生的最大长度及其来源 key |

## 为什么

- **屏蔽 HF config 差异**：不同模型的 `PretrainedConfig` 字段名/嵌套差异极大（`num_hidden_layers` vs `num_layers`、`text_config` 嵌套、`num_attention_heads` vs `num_key_value_heads`）。`ModelArchitectureConfig` 提供单一 schema，让块池/注意力/编译 pass 不写"if arch == ..."分支。
- **"total_" 前缀语义**：多层并行（PP/TP）下，`total_num_*` 表示整个模型的总量，而非本 rank 分片量，避免与 `get_num_attention_heads`(本 rank) 混淆。
- **派生量归一**：`is_deepseek_mla`/`rswa_window`/`is_mm_prefix_lm` 这些"语义判定"在转换器一处完成，下游直接读 bool。
- **可哈希**：作为 `ModelConfig` 子对象，其字段经 `normalize_value` 进 `ModelConfig.compute_hash`，从而参与编译缓存键。

## 怎么做

派生路径（`model.py: get_model_arch_config`）：

```mermaid
flowchart LR
    HF["hf_config<br/>(PretrainedConfig)"] -->|MODEL_ARCH_CONFIG_CONVERTORS<br/>[model_type/architecture]| CVT["ModelArchConfigConvertorBase<br/>.from_hf_config()"]
    CVT --> MAC["ModelArchitectureConfig"]
    MAC -->|挂载| MC["ModelConfig.model_arch_config"]
    MC --> MOM["VllmConfig.model_config<br/>供编译/注意力/块池消费"]
```

- 转换器从 `vllm/transformers_utils/model_arch_config_convertor.py` 的 `MODEL_ARCH_CONFIG_CONVERTORS` 注册表按 `model_type`/`architecture` 查找。
- `VllmConfig.with_hf_config`（`vllm.py:669`）在贴新 `hf_config` 后会重调 `get_model_arch_config()` 重建本对象。
- 子系统读法示例：`vllm_config.model_config.model_arch_config.is_deepseek_mla` 选择 MLA 块池 spec；`total_num_hidden_layers` 推导 PP 分层。

## 与其它模块/系统配合

- **`ModelConfig`（[model-config.md](model-config.md)）**：持有者与派生入口；`get_hidden_size`/`get_head_size` 等方法部分代理到本对象。
- **KV 缓存管理（[`01-engine-core/kv-cache-management/spec.md`](../01-engine-core/kv-cache-management/spec.md)）**：`is_deepseek_mla` 决定走 `MLAKVCacheSpec`；`total_num_kv_heads`/`head_size` 决定块容量。
- **注意力（[`05-attention/`](../05-attention/README.md)）**：MLA 后端选择、sliding window 行为依赖本对象。
- **编译（[`09-compilation-ir/`](../09-compilation-ir/README.md)）**：`hidden_size` 影响 SP 阈值（`get_sequence_parallelism_threshold`）与 allreduce 融合 compile range 推导。
- **块池（[`01-engine-core/kv-cache-management/block-pool.md`](../01-engine-core/kv-cache-management/block-pool.md)）**：`rswa_window` 影响滑动窗口块驱逐。

## 历史版本演进

- **v0.5–v0.8**：架构派生量散落在 `ModelConfig` 的派生属性 + 各模型类内部判断，无独立 dataclass；`is_deepseek_mla` 等通过 `hf_config` 字段实时检测。
- **v0.9（引入，待核实）**：抽出 `ModelArchitectureConfig`，初版字段聚焦 `hidden_size`/`total_num_hidden_layers`/`head_size`/`is_deepseek_mla` 等。
- **v0.10/v0.11**：`MODEL_ARCH_CONFIG_CONVERTORS` 转换器体系成形；`rswa_window`/`is_mm_prefix_lm`/`num_experts` 等字段补齐；`text_model_type` 支持多模态 text_config 提升。
- **v0.12 / main**：`derived_max_model_len_and_key` 元组字段引入（替代零散 `max_model_len` 派生）；与 `HybridAttentionMambaModelConfig.verify_and_update_config` 协同处理 hybrid SSM 架构。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — 持有本对象的 `ModelConfig`。
- [cache-config.md](cache-config.md) — `is_deepseek_mla` 影响 KV cache dtype/容量推导。
- [../01-engine-core/kv-cache-management/spec.md](../01-engine-core/kv-cache-management/spec.md) — KV cache spec 的架构判定消费方。
