# AttentionConfig（attention.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > AttentionConfig

源码：`vllm/config/attention.py`（约 117 行）。`AttentionConfig` 描述注意力机制选择：后端、flash-attn 版本、MLA prefill 后端、TRTLLM、flex-attn tile、quant 化等。它是 `VllmConfig.attention_config`，被 `vllm/v1/attention/backends/` 的 selector 与各后端消费。

## 是什么

`@config` 装饰（`attention.py:15`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `backend` | `None` | `AttentionBackendEnum` 或 `None`/`"auto"`（自动选） |
| `flash_attn_version` | `None` | 强制 flash-attn 版本 `2`/`3`/`4`（仅 flash-attn backend 有效） |
| `use_prefill_decode_attention` | `False` | 用分离的 prefill/decode kernel 而非统一 triton kernel |
| `flash_attn_max_num_splits_for_cuda_graph` | `32` | flash-attn cuda graph decode max splits |
| `tq_max_kv_splits_for_cuda_graph` | `32` | TurboQuant cuda graph decode 固定 split 数（保 grid 维度恒定，预分配缓冲） |
| `use_trtllm_attention` | `None`(三态) | flashinfer 中用/不用 TRTLLM backend；`None` 自动 |
| `disable_flashinfer_q_quantization` | `False` | fp8 kv 时不量化 Q |
| `mla_prefill_backend` | `None` | `MLAPrefillBackendEnum`：`FLASH_ATTN`(FA3/FA4)/`FLASHINFER`/`TRTLLM_RAGGED`；`None` 自动 |
| `use_prefill_query_quantization` | `False` | prefill 时量化 query |
| `use_fp4_indexer_cache` | `False` | dsv32 系列 fp4 indexer cache（未支持） |
| `indexer_kv_dtype` | `"bf16"` | 稀疏注意力 indexer K cache dtype：`bf16`/`fp8`/`mxfp4`/`nvfp4` |
| `use_non_causal` | `False` | 双向（非因果）注意力 |
| `flex_attn_block_m` | `None` | flex-attn Triton BLOCK_M（≥16 的 2 幂） |
| `flex_attn_block_n` | `None` | flex-attn Triton BLOCK_N |
| `flex_attn_q_block_size` | `None` | flex-attn 逻辑 Q block（须被 `block_m` 整除） |
| `flex_attn_kv_block_size` | `None` | flex-attn 逻辑 KV block（须被 `block_n` 整除） |

`IndexerKVDtype = Literal["bf16", "fp8", "mxfp4", "nvfp4"]`。

校验器：`validate_backend_before`（`backend` 字符串→枚举，`"auto"`→`None`）；`validate_mla_prefill_backend_before`（字符串→枚举）。

`compute_hash`：全部字段纳入（无 `ignored_factors`），因后端选择直接影响算子与图形状。

## 为什么

- **后端矩阵庞大**：vLLM 支持数十种注意力后端（flash-attn 2/3/4/flashinfer/trtllm/triton/cutlass/aiter/rocm/cpu/xpu/linear/flex-attention/MLA 子族），`backend`/`mla_prefill_backend` 让用户/平台/模型选最适。`"auto"` 让 selector 按模型/平台/dtype 自动决定。
- **MLA prefill 独立选**：DeepSeek MLA 在 prefill 与 decode 用不同后端最优（prefill 走 FA3/FA4/FlashInfer/TRTLLM_RAGGED，decode 走 flashmla/cutlass 等），故 `mla_prefill_backend` 单列。详见 [`05-attention/backends/mla/README.md`](../05-attention/backends/mla/README.md)。
- **cuda graph 固定 split**：`flash_attn_max_num_splits_for_cuda_graph`/`tq_max_kv_splits_for_cuda_graph` 固定 split 数让 grid 维度恒定，缓冲预分配，避免捕获时显存膨胀。
- **flex-attn tile**：`flex_attn_*` 字段控制 PyTorch flex-attention 的 Triton tile 与逻辑 block，供 `VLLM_BATCH_INVARIANT` 路径用，默认 `None` 时按 PyTorch 版本取默认。
- **fp4/fp8 indexer**：`indexer_kv_dtype`/`use_fp4_indexer_cache` 面向 DeepSeek V3.2 等稀疏注意力，量化格式需后端 indexer kernel 支持。

## 怎么做

- **强选后端**：`--attention-backend flash_attn`（或 `FLASH_ATTN`）；`"auto"` 走 selector。
- **MLA prefill**：`--mla-prefill-backend FLASH_ATTN`。
- **TRTLLM**：`--use-trtllm-attention`（flashinfer）。
- **flex-attn**：配合 `VLLM_BATCH_INVARIANT=1`，`--attention-config.flex-attn-block-m=64`。
- **flash-attn 版本**：`--flash-attn-version 3`。

## 与其它模块/系统配合

- **注意力子系统（[`05-attention/`](../05-attention/README.md)）**：`backend` 驱动 [`selector.md`](../05-attention/selector.md) 选后端；`mla_prefill_backend` 驱动 MLA prefill 后端选择；`flex_attn_*` 控制 flex-attention 后端。
- **CacheConfig（[cache-config.md](cache-config.md)）**：`cache_dtype` 与 `indexer_kv_dtype` 共同决定 KV 张量类型，影响后端兼容性。
- **ModelConfig（[model-config.md](model-config.md)）**：`use_mla`/`sliding_window`/`attention_chunk_size` 影响后端选择；`disable_cascade_attn` 在 async+spec 下被 `VllmConfig` 强制关。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`cudagraph_mode` 下后端须支持 cuda graph（`AttentionCGSupport`）；`splitting_ops` 默认含 attention op。
- **多模态（[multimodal-config.md](multimodal-config.md)）**：`mm_encoder_attn_backend`/`mm_encoder_attn_dtype` 独立控制 ViT 编码器注意力（与主干 `attention_config` 解耦）。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`attention_chunk_size` + spec → 关 HMA；`use_v2_model_runner` 不支持特性集合校验。

## 历史版本演进

- **v0.5/v0.6（v0）**：注意力后端选择散在 `EngineArgs`/`_Backend`；`flash_attn_version` 字段就位。
- **v0.7（v1 落地）**：`AttentionConfig` 引入；`backend` 枚举化（`AttentionBackendEnum`）；MLA 初步。
- **v0.8**：`mla_prefill_backend` 字段 + `MLAPrefillBackendEnum`；`use_trtllm_attention`；`flash_attn_version` 支持 `4`。
- **v0.9**：`indexer_kv_dtype`/`use_fp4_indexer_cache`（DeepSeek V3.2 稀疏）；`flex_attn_*` tile 字段；`use_non_causal`。
- **v0.10**：`tq_max_kv_splits_for_cuda_graph`（TurboQuant 固定 split）；`disable_flashinfer_q_quantization`；`use_prefill_query_quantization`。
- **v0.11 / v0.12 / main**：`flashinfer_cudnn`/`flashinfer_b12x` 等新 backend 枚举；SM12x 支持；MLA 后端矩阵持续扩充（cutlass/aiter/flashmla/triton）。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [cache-config.md](cache-config.md) — `cache_dtype`/`indexer_kv_dtype` 与后端兼容性。
- [vllm-config.md](vllm-config.md) — `attention_chunk_size`+spec 关 HMA 等校验。
- [multimodal-config.md](multimodal-config.md) — `mm_encoder_attn_*` 独立 ViT 注意力。
- [../05-attention/selector.md](../05-attention/selector.md) — 后端选择消费方。
- [../05-attention/backends/mla/README.md](../05-attention/backends/mla/README.md) — MLA prefill 后端矩阵。
