[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > TurboQuant KV Cache

# turboquant — TurboQuant KV Cache 量化

> 源码目录：`vllm/model_executor/layers/quantization/turboquant/`
>
> 不是 `QuantizationMethods` 中的独立 `quant_method`——它通过 `--kv-cache-dtype` 的命名预设触发（如 `turboquant_k8v4`），属于 **KV cache 维度**的量化，与 `BaseKVCacheMethod`（weight/KV scale 装载）是不同机制：TurboQuant 在 cache 写入前对 key 做 Hadamard 旋转 + Lloyd-Max 标量量化、对 value 做均匀量化。

---

## 是什么

`TurboQuantConfig`（`turboquant/config.py:45`，dataclass）描述 TurboQuant KV-cache 压缩：对 key 施加 Hadamard 旋转后做 per-coordinate Lloyd-Max（MSE）标量量化，对 value 做均匀量化，可选 norm correction（反旋转前对 centroid 向量单位化）。`turboquant/centroids.py` 提供 Lloyd-Max centroid 计算/查表。

预设（`TQ_PRESETS`，`:20`，经 `--kv-cache-dtype` 选择）：

| 预设 | key bits | value bits | norm correction | 压缩比 | PPL 变化（文档值） |
|---|---|---|---|---|---|
| `turboquant_k8v4` | 8 (FP8 key) | 4 | 否 | ~2.6× | +1.17% |
| `turboquant_4bit_nc` | 4 (MSE) | 4 | 是 | ~3.8× | +2.71% |
| `turboquant_k3v4_nc` | 3 (MSE) | 4 | 是 | ~3.5× | +10.63% |
| `turboquant_3bit_nc` | 3 (MSE) | 3 | 是 | ~4.9× | +20.59% |

`TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim)`（`:215`）按预设构造。`get_boundary_skip_layers`（`:176`）返回首末各 N 层 attention 以跳过压缩（边界保护，dense 模型必需，hybrid 模型禁用）。

算法溯源（`config.py:52` docstring）：Hadamard+确定性标量量化+再归一化模式源自 DRIVE（NeurIPS 2021）/EDEN（ICML 2022），与 HIGGS 标量情形（NAACL 2025）数学等价；KV-cache 应用见 "Cache Me If You Must"（ICML 2025）；TurboQuant 论文 ICLR 2026。

---

## 为什么

- **激进 KV cache 压缩**。长上下文场景 KV cache 显存主导，TurboQuant 提供 2.6×–4.9× 压缩，代价是可接受的 PPL 上升。
- **Hadamard 旋转降方差**。旋转使 key 各坐标能量均匀化，标量量化损失最小化；norm_correction 修复量化引入的范数畸变（4-bit 下 +0.8% PPL，`:85`）。
- **FP8 key 快速档**。`turboquant_k8v4` 用 FP8 key 跳过旋转/MSE，最cheap 的压缩档（2.6×，+1.17%）。
- **边界保护**。首末层 attention 对预设 `k3v4_nc`/`3bit_nc` 敏感，`get_boundary_skip_layers` 自动跳过（dense n=2），避免 GSM8K 大幅下降；hybrid 模型（仅 8-12 层 full attention）禁用边界保护（`:193`）。
- **与 `BaseKVCacheMethod` 正交**。TurboQuant 不读 checkpoint 的 k/v_scale，而是改写 cache 存储格式（packed key+value slot），故不复用 `BaseKVCacheMethod`。

---

## 怎么做

### 配置创建

`TurboQuantConfig.from_cache_dtype(cache_dtype, head_dim)`（`:215`）查 `TQ_PRESETS`，按 `key_quant_bits`/`value_quant_bits`/`norm_correction` 构造 dataclass。

### 关键属性（`:94` 起）

- `key_fp8`（`:94`）：`key_quant_bits==8` → FP8 key，无旋转/MSE。
- `mse_bits`/`key_mse_bits`/`centroid_bits`/`n_centroids`：MSE 量化器位数与 centroid 数（2^bits）。
- `key_packed_size`（`:128`）：FP8 = `head_dim` 字节；MSE = `ceil(head_dim*key_mse_bits/8)` + 2 字节 `vec_norm`(fp16)。
- `value_packed_size`（`:150`）：`ceil(head_dim*value_quant_bits/8)` + 4 字节（scale+zero fp16）。
- `slot_size`（`:159`）：`key_packed_size + value_packed_size`（每 head 每位置）；`slot_size_aligned`（`:167`）round up to even（使 `effective_head_size = slot_size_aligned // 2` 为整数）。

### 边界保护（`:176`）

`get_boundary_skip_layers(model_config, n=2)`：hybrid 模型（`model_config.is_hybrid`）→ 用 `_get_full_attention_layer_indices` 找 full-attention 层并返回 `[]`（禁用保护）；dense → 返回首末各 n 层索引（`min(n, num_layers//2)`）。

`_get_full_attention_layer_indices`（`:235`）兼容三种约定：`layer_types`（Qwen3.5/Next）、`layers_block_type`（Jamba/Zamba2）、`attn_type_list`（Minimax）。

### 运行期

TurboQuant cache 量化/解量化发生在 attention backend 的 cache 写入/读出 kernel（不在 `quantization/` 层）。`TurboQuantConfig` 主要提供配置与 packed layout 规格；`centroids.py` 提供 Lloyd-Max centroid（按 head_dim/bits 预计算或查表）。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：FP8 key 路径需 FP8 硬件支持；Hadamard 旋转 kernel 平台依赖。
- **[分布式](../../../07-distributed/README.md)**：KV cache 在 TP 下分头，TurboQuant slot layout 需与分头对齐。
- **[编译-IR](../../../09-compilation-ir/README.md)**：cache 量化 kernel 需编译兼容；`slot_size_aligned` 对齐支持 TMA。
- **Attention 层**（`05-attention`、`03-model-execution/layers/`）：cache 写入/读出 kernel 消费 `slot_size`/`effective_head_size`。
- **KV cache 存储**（`15-kv-cache-offload`、`01-engine-core/kv-cache-management`）：TurboQuant 改变 slot 物理布局，与 paged attention 的 block/slot 管理对接。
- **`BaseKVCacheMethod`**（[kv_cache.md](kv_cache.md)）：正交关系；TurboQuant 不通过 `quant_config.get_quant_method(Attention)`，而通过 `kv_cache_dtype` 字符串进入。

---

## 历史版本演进

- **v0.12 / main**（待核实）：TurboQuant KV cache 路径引入，4 个预设 + Hadamard+Lloyd-Max + norm correction；边界保护 hybrid 适配；算法溯源 docstring 完整（DRIVE/EDEN/HIGGS/"Cache Me If You Must"）。
- **后续**（待核实）：3-bit/2-bit 更激进档、与 per-token-head scale 的交互、更多 hybrid layer_types 约定支持。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [kv_cache.md](kv_cache.md) · [fp8.md](fp8.md) · [平台](../../../08-platforms/README.md) · [05-attention](../../../05-attention/README.md) · [15-kv-cache-offload](../../../15-kv-cache-offload/README.md)
