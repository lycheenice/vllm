# DeepSeek 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **DeepSeek**

> 代表文件：`deepseek_v2.py`、`deepseek_vl2.py`、`deepseek_mtp.py`、`deepseek_eagle.py`、`deepseek_eagle3.py`、`deepseek_ocr.py`、`deepseek_ocr2.py`；厂商隔离包：`vllm/models/deepseek_v32/`、`vllm/models/deepseek_v4/`（见 [`vendor-split-models.md`](../vendor-split-models.md)）。
> 厂商：DeepSeek。

---

## 是什么

DeepSeek 是 vLLM MoE + MLA 路线的核心家族，迭代非常快：

- **`deepseek_v2.py`**：基元。重要类：
  - `DeepseekV2Attention` / `DeepseekAttention`（普通 attention）
  - `DeepseekV2MLAAttention`（MLA：低秩 KV cache，吸收 q/k 投影）
  - `DeepseekV2MoE`（aux-loss-free 路由 + shared expert）
  - `DeepseekV2DecoderLayer`、`DeepseekV2Model`、`DeepseekV2ForCausalLM`
  - `DeepseekV2MixtureOfExperts(MixtureOfExperts)` 接口实现
  - `DeepseekForCausalLM(DeepseekV2ForCausalLM)`、`DeepseekV3ForCausalLM(DeepseekV2ForCausalLM)`、`GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`：三个族友直接子类化（`deepseek_v2.py:1906-1914`）
  - **DSA（V3.2 Sparse Attention）**：`DeepseekV32IndexerCache`（`:615`）+ `Indexer`（`:644`）+ `DeepseekV2FusedQkvAProjLinear`（`:906`），实现 MLA + "lightning indexer" 选 top-k 做 sparse MLA attend。
- **`vllm/models/deepseek_v32/`**：V3.2 在 SM100 上的特化（`nvidia/attention.py:DeepseekV32Attention(MLAAttention)`、`nvidia/model.py:DeepseekV32ForCausalLM(DeepseekV2ForCausalLM)`、`mtp.py:DeepseekV32MTP`）。其他平台报 `NotImplementedError`。
- **`vllm/models/deepseek_v4/`**：DeepSeek V4 三平台实现（NVIDIA/AMD/XPU）。核心新增：
  - `Sparse MLA` 后端（`sparse_mla.py`）+ `Compressor`（`compressor.py`，MLA KV 压缩 + 量化缓存）
  - `DeepseekV4MegaMoEExperts` + `DeepseekV4MoE`（MegaMoE 支持 EP/TP 混合）
  - `DSparkDeepseekV4ForCausalLM`（DSpark 投机 draft，仅 NVIDIA）
  - `DeepseekV4FP8Config`（按 `expert_dtype=fp4/fp8` 分派 MXFP4 / FP8 block）
  - `common/ops/` 跨平台 Triton kernel：`fused_compress_quant_cache`、`fused_indexer_q`、`fused_inv_rope_fp8_quant`、`fused_qk_rmsnorm`、`save_partial_states`
- **`deepseek_vl2.py`**：DeepSeek-VL2 视觉模型。
- **`deepseek_ocr.py` / `deepseek_ocr2.py`**：DeepSeek OCR。
- **spec draft**：`deepseek_mtp.py DeepSeekMTP`（MTP，含 `DeepseekV2MixtureOfExperts`）、`deepseek_eagle.py EagleDeepseekV3ForCausalLM`（EAGLE-1）、`deepseek_eagle3.py Eagle3DeepseekV2ForCausalLM`（EAGLE-3，覆盖 V2 与 V3 target）、`DeepSeekV4MTPModel`（vendor）。

---

## 为什么

- **MLA 是 vLLM 注意力栈的关键分支**：DeepSeek-V2 的 MLA（低秩 KV）压缩 KV cache 数十倍，是 vLLM 在 [`05-attention`](../../05-attention/README.md) 单独维护 MLA 通道的根本原因。
- **aux-loss-free MoE 范本**：DeepSeek-V2 的"无辅助损失路由 + shared expert"成为后续 Qwen3-MoE / GLM4-MoE / Kimi-Linear 等纷纷沿用的模式。
- **DSA 推动厂商隔离布局**：V3.2 的 indexer 强依赖 SM100 WGMMA/TMA，单文件无法兼容多平台；`vllm/models/` 顶层包由此诞生（见 [`vendor-split-models.md`](../vendor-split-models.md)）。
- **V4 引入 Compressor + Sparse MLA**：进一步把 MLA 通道做 sparse + KV 压缩 + FP8 量化，是当前 vLLM 性能前沿阵地。

---

## 怎么做

`DeepseekV2MLAAttention` 的关键吸收逻辑：在权重加载阶段把 `q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj` 合并到 fused 矩阵里（`DeepseekV2FusedQkvAProjLinear`），forward 时只算"压缩 latent"，decode 时把 q/k 投影吸收进 o_compact，避免展开 KV。

`Indexer`（DSA）从 latent KV 计算 query 与 key 的相似度，选 top-k token index，sparse MLA 后端只 attend 这些 index。`DeepseekV32Attention` 重写 `MLAAttention` 注入这一步。

V4 的 `Compressor` 在 prefill 后把 latent KV 做 RMSNorm + RoPE + FP8 量化 + 写入 cache，由 `sparse_mla.py` 的 metadata builder 在 decode 时读出。

---

## 与其它模块/系统配合

- **[注意力](../../05-attention/README.md)**：MLA / DSA / Sparse MLA 通道的源头。
- **[vendor-split-models.md](../vendor-split-models.md)**：V3.2 / V4 厂商隔离实现细节。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：MTP + EAGLE-1/3 + DSpark 投机。
- **[分布式](../../07-distributed/README.md)**：MegaMoE 走 EP/TP/专家迁移；`DeepseekV2MixtureOfExperts` + `DeepseekV4MixtureOfExperts` 实现 `MixtureOfExperts` 给 EPLB。
- **[模型执行-层库](../../03-model-execution/layers/README.md)**：`FusedMoE` 路由层在 DeepSeek-V2 上首发 aux-loss-free 模式。
- **[08 硬件平台](../../08-platforms/README.md)**：V3.2/V4 平台分支由 `current_platform.verify_model_arch` 守卫。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.5 | DeepSeek-V2 首发，引入 MLA + aux-loss-free MoE。 |
| v0.6 | DeepSeek-V3（同文件子类化）；MTP draft `DeepSeekMTP`。 |
| v0.8 | DeepSeek-VL2 + DeepSeek-OCR；EAGLE-1/3 draft 落地。 |
| v0.10 | DeepSeek-V3.2 引入 DSA（Indexer）；`vllm/models/deepseek_v32/` SM100 隔离实现上线。 |
| v0.11 | GLM-5.2 DSA 复用 `GlmMoeDsaForCausalLM`；DeepSeek-OCR2。 |
| v0.12 | DeepSeek-V4：Sparse MLA + Compressor + MegaMoE + FP4/FP8 expert 分派；三平台隔离；DSpark 投机 draft。 |
| main | common/ops 持续补 fused Triton kernel；与 GLM/Kimi/MiniMax 共享 sparse attention 工具栈。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [vendor-split-models](../vendor-split-models.md) · [注意力](../../05-attention/README.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [glm 家族](./glm.md)
