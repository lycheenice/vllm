# Mamba / SSM 块（mamba/）

[← Wiki 首页](../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > Mamba/SSM

`vllm/model_executor/layers/mamba/` 实现 vLLM 全部状态空间模型（SSM）/Mamba 系列算子层，包括经典 Mamba（Mamba1）、Mamba2（SSD）、线性注意力变体（GDN/short_conv/bailing_linear_attn 等）。这些块在 v1 引擎里通过 `AttentionLayerBase` 抽象与 KV-cache 管理器协作，被当作"另一种 attention 后端"看待。

## 是什么

`mamba/` 目录主要文件：

| 文件 | 内容 |
|---|---|
| `abstract.py` | `MambaBase(AttentionLayerBase)`：定义 `get_state_shape` / `mamba_type` / `get_state_dtype` / `get_kv_cache_spec` / `get_attn_backend` |
| `mamba_mixer.py` | `MambaMixer(MambaBase, PluggableLayer)`（`:51`）—— Mamba1 经典实现；`split_batch_to_prefill_and_decode`（`:497`）与 `mamba_mixer` custom op（`:523`） |
| `mamba_mixer2.py` | `MambaMixer2`（`:234`）、`Mixer2RMSNormGated`（`:66`）—— Mamba2/SSD 实现；`mamba_v2_sharded_weight_loader`（`:171`） |
| `short_conv.py` | `ShortConv(MambaBase, CustomOp)`（`:35`）—— 短卷积 SSM 变体 |
| `mamba_utils.py` | `MambaStateDtypeCalculator` / `MambaStateShapeCalculator` / `is_conv_state_dim_first` |
| `ops/` | SSM 算子族：`causal_conv1d.py`、`mamba_ssm.py`（selective_scan）、`ssd_combined.py`、`ssd_chunk_scan.py`、`ssd_chunk_state.py`、`ssd_bmm.py`、`ssd_state_passing.py`、`ssu_dispatch.py`、`layernorm_gated.py`、`triton_helpers.py`，以及 `cpu/` 与 `configs/` |
| `ops/gdn_chunk_cutedsl/` | FLA 路径的 GDN chunked CUDA DSL 实现 |
| `gdn/` | GDN（gated normalization）：`base.py` 与 `kimi_gdn_linear_attn.py` / `olmo_gdn_linear_attn.py` / `qwen_gdn_linear_attn.py` |
| `linear/` | 线性注意力变体：`bailing_linear_attn.py`、`minimax_linear_attn.py`、`base.py` |

外部依赖：`mamba_ssm`、`causal_conv1d`（若安装）；否则走 `ops/` 内部 Triton 实现。

## 为什么

把 Mamba 单独成层库子模块，是为了：

1. **状态空间而非 KV cache**：Mamba/SSM 的"状态"是 `(conv_state, ssm_state)` 元组而非传统的 K/V cache。`MambaBase.get_kv_cache_spec` 返回 `MambaSpec`（`abstract.py:44-60`），告诉 `KVCacheManager` 这个层的缓存形态——`block_size = mamba_block_size`、`page_size_padded`、`mamba_type`、`mamba_cache_mode`、`num_speculative_blocks` 等，由 v1 attention 后端在 prefill/decode 时分配与重置。
2. **`AttentionLayerBase` 复用**：Mamba 层被 v1 看作"另一种 attention"——通过 `get_attn_backend()` 返回 `get_mamba_attn_backend(self.mamba_type)`（`abstract.py:62-64`），让 `ModelRunner` 统一走 attention backend 选择路径。`mamba_type` 由 `MambaAttentionBackendEnum` 决定（Mamba1 / Mamba2 / ShortConv / Linear 等）。
3. **prefill/decode 分流**：`split_batch_to_prefill_and_decode`（`mamba_mixer.py:497`）把 batch 内 prefill（长序列）与 decode（单步）分流到不同 kernel 路径——SSM 的 prefill 是一次性算 whole-seq state，decode 是 state update + 单步输出，工程上必须分离 hiding 隐藏的复杂度。
4. **不透明 custom op**：`mamba_mixer`（`mamba_mixer.py:523`）通过 `direct_register_custom_op` 注册（与 `moe_forward` 同套路），让 Mamba 的复杂算子图对 `torch.compile` 黑盒，避免 cudagraph 捕获失败。`MambaMixer2` 同理用 `mamba_mixer2` custom op（`mamba_mixer2.py:1074`）。
5. **多种 SSM 算子族统一调度**：Mamba2 引入 SSD（State Space Duality），把 attention 与 SSM 视为同一对偶结构；`ops/ssd_combined.py` / `ssd_chunk_scan.py` / `ssd_chunk_state.py` 实现分块 SSD，可与 attention backend 共享部分基础设施。GDN（Grouped Dense Normalization）则给 Mamba2 提供门控归一化。
6. **PluggableLayer 让整层可 OOT 替换**：`MambaMixer` 与 `MambaMixer2` 都 `PluggableLayer.register(...)`，方便厂商提供自己的 SSM kernel。

## 怎么做

### `MambaMixer.__init__`（Mamba1）

`mamba_mixer.py:50-200` 关键步骤：

1. 创建 `conv1d = ColumnParallelLinear(conv_kernel_size → intermediate)` 并把 weight `unsqueeze(1)` 改造成 conv1d 权重形状（`mamba_mixer.py:91-101`）。
2. `in_proj = MergedColumnParallelLinear(hidden → [intermediate, intermediate])`——对应原 Mamba 的 `x = in_proj(x); x = z, x_branch`。
3. `x_proj = RowParallelLinear(intermediate → time_step_rank + 2*ssm_state_size)`：算 `dt, B, C`。
4. `dt_proj = ColumnParallelLinear(time_step_rank → intermediate)`：把 `dt` 投影回 intermediate。
5. `A_log`、`D`（input-independent 参数）按模型定义创建。
6. `out_proj = RowParallelLinear(intermediate → hidden)`。
7. 可选 RMSNorm 在 in_proj 之后（`use_rms_norm`）。

### 前向链

`mamba_mixer.py:523` 起的 custom op 把以下逻辑黑盒：

```
forward(hidden_states)
  → split_batch_to_prefill_and_decode
  → for prefill tokens:  selective_scan_fn(x, dt, A, B, C, z, D)  # ops/mamba_ssm.py
  → for decode tokens:   selective_state_update(...)               # ops/ssu_dispatch.py
  → causal_conv1d_fn / causal_conv1d_update                         # ops/causal_conv1d.py
  → out_proj + residual
```

所有上述算子都通过 `direct_register_custom_op` 注册，并在 `forward_context` 中由 `mamba_layer_index` 反查到本层实例（参见 [fused-moe.md](fused-moe.md) 中类似机制）。

### `MambaMixer2`（Mamba2/SSD）

`mamba_mixer2.py:234` 起的 `MambaMixer2`：

- 使用 SSD（State Space Duality）算子族：`ops/ssd_combined.py` 的分块 SSD 内核，结合 `ssd_chunk_scan` 与 `ssd_chunk_state` 处理超长序列。
- `Mixer2RMSNormGated`（`mamba_mixer2.py:66`）是 Mamba2 的门控 RMSNorm，与 [norm.md](norm.md) 中的 `RMSNormGated` 不同——它继承自 `CustomOp` 而非 `layernorm.RMSNormGated`，因为参数布局不同。
- `mamba_v2_sharded_weight_loader`（`:171`）实现 Mamba2 模型在权重加载时的分片重组，比 v1 复杂——`A_log`、`D`、`dt`、`B`/`C` 都有特殊 layout。
- 通过 `SharedWeightParameter`（[parameter.md](parameter.md)）支持 in_proj 两段的内存共享。

### ShortConv

`short_conv.py:35`：`ShortConv(MambaBase, CustomOp)`——短上下文卷积 SSM 变体，`mamba_type = ShortConv`。状态空间相比 Mamba1 更简单，主要服务"Cohere Command A"等模型的特定 block。

### GDN（gated normalization）

`gdn/` 子目录：`base.py` 提供共享接口，`qwen_gdn_linear_attn.py` / `kimi_gdn_linear_attn.py` / `olmo_gdn_linear_attn.py` 是各模型专用 GDN——本质是"线性注意力 + 门控 RMSNorm"，作为 SSM 的轻量替代。`mamba_mixer2.py` 内部的 `Mixer2RMSNormGated` 是这条路径的官方 Mamba2 实现。

### 与 KV cache 的协作

`MambaBase.get_kv_cache_spec`（`abstract.py:44-60`）返回：

```python
MambaSpec(
    shapes=tuple(self.get_state_shape()),              # ((conv_state), (ssm_state))
    dtypes=self.get_state_dtype(),
    block_size=mamba_block_size,
    page_size_padded=page_size_padded,
    mamba_type=self.mamba_type,
    mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
    num_speculative_blocks=vllm_config.speculative_config.num_speculative_tokens or 0,
)
```

`KVCacheManager` 会按此 spec 分配 per-layer `(conv_state, ssm_state)`；prefill 时填满 state，decode 时单步更新；spec decode 时复制额外 `num_speculative_blocks` 份 state 用于 draft。

## 与其它模块/系统配合

- [attention #05](../../05-attention/README.md)：`MambaBase` 继承 `AttentionLayerBase`，让 `ModelRunner` 用同一套 backend 选择机制；`get_mamba_attn_backend(mamba_type)` 返回 `Mamba1AttentionMetadata`/`Mamba2`/`ShortConv`/`Linear` 等后端。
- [KV cache offload #15](../../15-kv-cache-offload/README.md)：Mamba 的 state 比 KV cache 紧凑，但仍可能被算子层 offload；`mamba_cache_mode` 控制 `last_state` / `full` 等策略 `(待核实)`。
- [linear.md](linear.md)：Mamba block 内部全部投影都复用 `ColumnParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear`；SSM 在 TP 下不切状态空间维（`ssm_state_size` 不除 `tp_size`），只切 `intermediate` / `hidden`。
- [norm.md](norm.md)：`Mixer2RMSNormGated` 与 `layernorm.RMSNormGated`、`fla/ops/layernorm_guard.rmsnorm_fn` 的关系 `(待核实)`——两者均做 gated RMSNorm，但参数与平台 dispatch 路径不同。
- [compilation-ir #09](../../09-compilation-ir/README.md)：`mamba_mixer` / `mamba_mixer2` 是不透明 custom op，与 `moe_forward` 同样注册到 `compilation_config.static_forward_context`；并由 `_resolve_layer_name(layer_name)` 解析 `LayerName` opaque object（torch >= 2.11）。
- [model-zoo #04](../../04-model-zoo/README.md)：Mamba、Mamba2、Jamba、Bamba、Falcon-Mamba、Cohere Atrium 等模型在 `__init__` 中实例化 `MambaMixer`/`MambaMixer2`/`ShortConv`/`Mixer2RMSNormGated`。
- [platforms #08](../../08-platforms/README.md)：CUDA 路径优先用外部 `mamba_ssm`/`causal_conv1d` 包；缺失时 fallback 到 `ops/mamba_ssm.py`/`causal_conv1d.py` 的 Triton 实现；CPU 走 `ops/cpu/`。

## 历史版本演进

- **v0.5 末**：Mamba1 模型接入；`MambaMixer` 在 `vllm/model_executor/layers/mamba/`，仍依赖外部 `mamba_ssm` 包。
- **v0.6（PR #10909 `Add Bamba Model`）**：`mamba_mixer2.py` 引入服务 Bamba；Mamba2 SSD 算子族（`ops/ssd_*.py`）落地。
- **v0.7–v0.8**：`AttentionLayerBase` 抽象引入，`MambaBase` 改为继承之，让 v1 引擎把 Mamba 当作 attention backend 调度；`MambaSpec` 与 `KVCacheManager` 协作成熟。
- **v0.8–v0.9**：`split_batch_to_prefill_and_decode` 引入，把 prefill/decode 分流到不同 kernel；`mamba_mixer` / `mamba_mixer2` 注册为不透明 custom op。
- **v0.9**：`ShortConv` 引入服务 Command A 等模型；`gdn/` 子目录成形，包含 Kimi/Qwen/Olmo 的 GDN 线性注意力变体。
- **v0.10（PR #32744）**：`PluggableLayer` 落地，`MambaMixer`/`MambaMixer2` 改为 `PluggableLayer.register(...)` 支持 OOT 替换；`Mixer2RMSNormGated` 改继承 `CustomOp`。
- **v0.10末–v0.11**：`mamba_v2_sharded_weight_loader` 与 `SharedWeightParameter` 协同，让 Mamba2 的 in_proj 两段在内存中共享张量；`mamba_cache_mode` 引入支持 spec decode 的多 state 复制。
- **v0.11–v0.12**：`fla/` 子目录成形，FLA（Flash Linear Attention）路径与 Mamba GDN 共享部分 layernorm 算子；`ops/gdn_chunk_cutedsl/` 引入 CUDA DSL 实现。
- **v0.12 / main**：`linear/bailing_linear_attn.py`、`linear/minimax_linear_attn.py` 引入支持 Bailing-M3 与 MiniMax-M3 的线性注意力变体；`mamba_cache_mode` 与 `num_speculative_blocks` 协同 spec decode；与 [rejection-sampler-layer.md](rejection-sampler-layer.md) 的 spec pipeline 协同成熟。

[← 返回层库首页](../README.md)

## 参见

- [attention #05](../../05-attention/README.md)：Mamba 后端如何作为 attention backend 被调度。
- [norm.md](norm.md)：`RMSNormGated` 与 GDN 的差异。
- [custom-op.md](custom-op.md)：`PluggableLayer.register` + `direct_register_custom_op` 的注册机制。
- [linear.md](linear.md)：Mamba block 内部的投影层复用。
