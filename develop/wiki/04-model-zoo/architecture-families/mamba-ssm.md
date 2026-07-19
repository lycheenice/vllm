# Mamba / SSM 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **Mamba/SSM**

> 代表文件：`mamba.py`、`mamba2.py`、`jamba.py`、`zamba2.py`、`lfm2.py`、`lfm2_moe.py`、`lfm2_siglip2.py`、`lfm2_vl.py`、`olmo_hybrid.py`、`granitemoehybrid.py`（同时见 [granite](./granite.md)）、`nemotron_h.py`。
> 另有 `FalconMambaForCausalLM` 别名指向 `mamba.py:MambaForCausalLM`（`registry.py:102`）。

---

## 是什么

- **`mamba.py` / `mamba2.py`**：纯 SSM 实现。`MambaForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP, SupportsMambaPrefixCaching)` 与 `Mamba2ForCausalLM` 同款接口。`FalconMambaForCausalLM` 直接复用 `MambaForCausalLM`。
- **`jamba.py`**：Jamba（AI21），混合 Mamba+Attention+MoE。`JambaForCausalLM(nn.Module, HasInnerState, IsHybrid, ...)`，`JambaMoE` 是 MoE 实现。
- **`zamba2.py`**：Zamba2（Zyphra）`Zamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMambaPrefixCaching)`，混合模式与 Jamba 同门。
- **`lfm2.py` / `lfm2_moe.py`**：Liquid Foundation Model 2，`Lfm2ForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid, SupportsQuant)`；MoE 版 `Lfm2MoeForCausalLM` 同样 `HasInnerState` + `IsHybrid`。
- **`lfm2_siglip2.py` / `lfm2_vl.py`**：LFM2 视觉变体（Siglip2 视觉塔 / VLM），保留 hybrid 标签。
- **`olmo_hybrid.py`**：Olmo-Hybrid（AllenAI），`OlmoHybridForCausalLM(nn.Module, HasInnerState, SupportsPP, SupportsLoRA, IsHybrid)`。
- **`nemotron_h.py`**：Nemotron-H（NVIDIA），`NemotronHForCausalLM` 含 hybrid + MTP（`NemotronHMTPModel`）。
- **`granitemoehybrid.py`**：Granite-MoE-Hybrid（IBM）见 [granite](./granite.md)。

`colbert.py:ColBERTLfm2Model` 也用 LFM2 backbone（late-interaction 检索，`IsHybrid` + `HasInnerState`，见 `colbert.py:389`）。

---

## 为什么

- **三个接口标签的发源地**：`HasInnerState` / `IsAttentionFree` / `IsHybrid` 都因 Mamba 系而引入。它们让调度器为 SSM 层单独分配 conv state + temporal state（不是 PagedAttention 的 KV cache），决定 attention 后端能否跳过。
- **prefix caching 实验场**：`SupportsMambaPrefixCaching` 在 Mamba/Jamba/Zamba2 等开启，让 SSM 状态也参与前缀缓存（与传统 KV cache 不同路径）。
- **混合架构主流化**：Jamba/Zamba2/LFM2/Olmo-Hybrid/Nemotron-H 都走"部分层 Mamba + 部分层 Attention"的混合模式（`hf_config.layers_block_type` 标记每层类型），`IsHybrid.get_mamba_state_shape_from_config` 给出 conv state + ssm state shape 供调度器分配。

---

## 怎么做

`MambaForCausalLM` 的 `backbone = MambaModel` 把每层替换为 `MambaDecoderLayer`（含 Mamba2 layer / Mamba mixer）。`forward` 不走 attention metadata，改用 `MambaStateCopyFunc` 在 prefill/decode 间维护 conv + ssm state。混合模型在 `JambaModel` 内按 `layers_block_type` 在 `JambaMambaDecoderLayer` 与 `JambaAttentionDecoderLayer` 间切换。

`supports_mamba_prefix_caching=True` 的模型让调度器在 prefill 复用历史 SSM 状态——通过 `get_mamba_state_copy_func` 给出每个 state 的 `MambaCopySpec`。

`has_inner_state=True` 让调度器在初始化时分配大小为 `max_num_seqs * state_shape` 的 SSM 状态缓存。

---

## 与其它模块/系统配合

- **[interfaces.md](../interfaces.md)**：`HasInnerState`/`IsAttentionFree`/`IsHybrid`/`SupportsMambaPrefixCaching` 是本家族专属接口。
- **[注意力](../../05-attention/README.md)**：`IsAttentionFree` 让模型走 attention-free 后端；混合模型在 attention 层仍走标准 backend。
- **[KV 缓存卸载](../../15-kv-cache-offload/README.md)**：SSM 状态与 KV cache 分轨卸载。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：`NemotronHMTPModel` 提供 MTP draft。
- **[07 分布式](../../07-distributed/README.md)**：`get_mamba_state_copy_func` 与 prefix caching 协作。
- **[embedding-col](./embedding-col.md)**：`ColBERTLfm2Model` 用 LFM2 backbone 做 late-interaction。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.4 | Mamba 接入；`HasInnerState`/`IsAttentionFree` 引入。 |
| v0.5 | Mamba2 + Jamba（hybrid）；`IsHybrid` 接口。 |
| v0.6 | `SupportsMambaPrefixCaching` 实验性落地。 |
| v0.7 | Zamba2 + LFM2 + LFM2-MoE。 |
| v0.8 | Olmo-Hybrid + Nemotron-H + MTP；Granite-MoE-Hybrid。 |
| v0.9 | LFM2-VL / LFM2-Siglip2 视觉变体。 |
| main | Mamba prefix caching 持续优化；colbert Lfm2 检索变体。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [interfaces](../interfaces.md) · [注意力](../../05-attention/README.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [granite](./granite.md)
