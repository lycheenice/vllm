# MambaConfig + MambaBackendEnum（mamba.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > MambaConfig

源码：`vllm/config/mamba.py`（约 76 行）。`MambaConfig` 描述 Mamba SSM（选择性状态空间）后端选择与缓存随机舍入。`MambaBackendEnum` 列举 SSU（selective state update）后端。它是 `VllmConfig.mamba_config`，被 Mamba 层（`vllm/model_executor/layers/mamba/`）与缓存管理消费。

## 是什么

### `MambaBackendEnum`（`mamba.py:25`，`Enum` + `_MambaBackendEnumMeta`）

| 成员 | 值 | 含义 |
|---|---|---|
| `TRITON` | `"triton"` | Triton SSU 后端 |
| `FLASHINFER` | `"flashinfer"` | FlashInfer SSU 后端 |

`_MambaBackendEnumMeta`（`mamba.py:12`）重写 `__getitem__`，未知名 raise 含合法选项列表的 `ValueError`（比默认 `KeyError` 更友好）。

### `MambaConfig`（`mamba.py:32`，`@config`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `backend` | `MambaBackendEnum.TRITON` | SSU 后端 |
| `enable_stochastic_rounding` | `False` | 写 SSM 状态到 fp16 cache 时用随机舍入（无偏，长序列稳） |
| `stochastic_rounding_philox_rounds` | `0` | Philox PRNG 轮数；0=Triton 默认；大则随机性质量更高但算力更贵 |

校验器：`validate_backend_before`（字符串→枚举）。

`__post_init__`：
- `enable_stochastic_rounding=True` 须 CUDA 平台。
- `TRITON` backend + stochastic rounding 须 compute capability 10.0（Blackwell data center），因 `cvt.rs` PTX 指令仅 Blackwell 支持；否则建议 `flashinfer` backend。

> 无 `compute_hash`（`MambaConfig` 未定义），`VllmConfig.compute_hash` 中 **未** 纳入 `mamba_config`——SSU 后端选择不改前向图形状（与采样层配置一致）。`cache_config` 的 `mamba_cache_dtype`/`mamba_ssm_cache_dtype`/`mamba_cache_mode` 已进 `CacheConfig.compute_hash` 覆盖图形状面。

## 为什么

- **SSU 后端两选**：Triton（默认，广泛可用）vs FlashInfer（优化，特定硬件）。`backend` 字段让用户/平台选最优。
- **随机舍入**：fp16 SSM state 长序列累积舍入误差致数值不稳。随机舍入用随机 bit 无偏化舍入误差，提升稳定性。但 `cvt.rs` PTX 仅 Blackwell 支持（Triton 路径），其它硬件须用 FlashInfer。
- **Philox 轮数**：`stochastic_rounding_philox_rounds` 让用户在随机性质量与算力间权衡，0 用 Triton 默认。
- **友好错误**：`_MambaBackendEnumMeta` 重写 `__getitem__` 把 `KeyError` 变 `ValueError` 含合法选项，降低用户试错成本。
- **图形状由 cache 覆盖**：Mamba cache 的 dtype/mode（在 `CacheConfig`）影响图形状并已进哈希；SSU 后端本身是算子选择，不改图，故本配置不进哈希。

## 怎么做

- **FlashInfer SSU**：`--mamba-backend flashinfer`。
- **随机舍入**：`--enable-mamba-cache-stochastic-rounding`（须 Blackwell + Triton，或任意 CUDA + FlashInfer）。
- **Philox 轮数**：`--mamba-config.stochastic-rounding-philox-rounds 4`。

## 与其它模块/系统配合

- **Mamba 层（[`03-model-execution/layers/mamba-ssm.md`](../03-model-execution/layers/mamba-ssm.md)）**：`backend` 驱动 `MambaMixer2`/SSU kernel 选择；`enable_stochastic_rounding` 在写 state 时启用。
- **CacheConfig（[cache-config.md](cache-config.md)）**：`mamba_cache_dtype`/`mamba_ssm_cache_dtype` 决定 cache 张量类型；`mamba_cache_mode`（`all`/`align`/`none`）决定状态缓存策略。`VllmConfig.__post_init` 校验 `enable_stochastic_rounding=True` 须 `mamba_ssm_cache_dtype=="float16"`。
- **Mamba attention 后端（[`05-attention/backends/mamba.md`](../05-attention/backends/mamba.md)）**：Mamba 作为"注意力后端"注册（`AttentionBackendEnum.MAMBA`），与本配置的 SSU backend 协同。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`__post_init` 校验 stochastic rounding + `mamba_ssm_cache_dtype` 须 float16；`validate_block_size` 校验 `mamba_cache_mode="align"` 下 `block_size <= max_num_batched_tokens`。
- **平台（[`08-platforms/`](../08-platforms/README.md)）**：`current_platform.is_cuda()`/`is_device_capability_family(100)` 判定 stochastic rounding 可用性。

## 历史版本演进

- **v0.7（v1 落地）**：Mamba/Jamba 接入；`MambaConfig` 初版；Triton SSU。
- **v0.8**：`MambaBackendEnum` + FlashInfer SSU；`_MambaBackendEnumMeta` 友好错误。
- **v0.9**：`enable_stochastic_rounding`/`stochastic_rounding_philox_rounds`（Blackwell `cvt.rs` PTX）；与 `CacheConfig.mamba_ssm_cache_dtype` float16 约束。
- **v0.10–main**：`mamba_cache_mode="align"`（Marconi APC）与 `validate_block_size` 联动；hybrid SSM 模型（Jamba/Bamba）HMA 协同。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [cache-config.md](cache-config.md) — `mamba_cache_*` 字段（图形状面，进哈希）。
- [vllm-config.md](vllm-config.md) — stochastic rounding + float16 校验；align 模式 block_size 校验。
- [../03-model-execution/layers/mamba-ssm.md](../03-model-execution/layers/mamba-ssm.md) — Mamba 层消费方。
- [../05-attention/backends/mamba.md](../05-attention/backends/mamba.md) — Mamba 注意力后端。
