# CacheConfig（cache.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > CacheConfig

源码：`vllm/config/cache.py`（约 293 行）。`CacheConfig` 描述 KV cache 的物理与逻辑规格：块大小、存储 dtype、前缀缓存、Mamba cache、KV 卸载。它是 `VllmConfig.cache_config`，被 `KVCacheManager`/`BlockPool`/`KVCacheSpec` 体系与 `Worker` 内存 profiling 消费。

## 是什么

`@config` 装饰（`cache.py:43`）。`DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`。

### 物理块与内存

| 字段 | 默认 | 含义 |
|---|---|---|
| `block_size` | `None`(→16) | 单块 token 数；`None` 表"用默认"，构造后恒为 int |
| `user_specified_block_size` | `False`(init=False) | 用户是否显式设了 `block_size` |
| `hash_block_size` | `None` | 前缀缓存哈希粒度（可细于物理块，须能整除各 KV cache group 的 block_size） |
| `gpu_memory_utilization` | `0.92` | 单实例 GPU 显存占用上限（0,1] |
| `num_gpu_blocks_override` | `None` | 覆盖 profiling 出的 `num_gpu_blocks`（测试/抢占用） |
| `sliding_window` | `None` | 滑动窗口大小（主要在 `ModelConfig` 设，此处镜像） |
| `is_attention_free` | `False` | 是否 attention-free（`ModelConfig` 镜像） |
| `kv_cache_memory_bytes` | `None` | 手动指定 KV cache 字节数（设定后忽略 `gpu_memory_utilization`） |

### KV cache dtype（`CacheDType`，`cache.py:19`）

`auto`/`float16`/`bfloat16`/`fp8`/`fp8_e4m3`/`fp8_e5m2`/`fp8_inc`/`fp8_ds_mla`/`turboquant_k8v4`/`turboquant_4bit_nc`/`turboquant_k3v4_nc`/`turboquant_3bit_nc`/`int4_per_token_head`/`int8_per_token_head`/`fp8_per_token_head`/`nvfp4`。`cache_dtype` 字段默认 `"auto"`（随 model dtype）。

- `kv_cache_dtype_skip_layers: list[str]`：跳过 KV 量化的层（按索引或注意力类型名，如 `'sliding_window'`）。
- `skip_page_size_padded: int | None`：被跳过层的页大小对齐。
- `calculate_kv_scales: bool`（**deprecated，v0.19 移除**）：fp8 时动态算 `k_scale`/`v_scale`。

### 前缀缓存

| 字段 | 默认 | 含义 |
|---|---|---|
| `enable_prefix_caching` | `True` | 是否启用前缀缓存 |
| `prefix_caching_hash_algo` | `"sha256"` | `sha256`/`sha256_cbor`/`xxhash`/`xxhash_cbor`（xxhash 非密码学安全，多租户慎用） |

### Mamba cache

| 字段 | 默认 | 含义 |
|---|---|---|
| `mamba_page_size_padded` | `None` | hybrid mamba/attention 页大小对齐覆盖 |
| `mamba_block_size` | `None` | Mamba cache 块大小（仅 `enable_prefix_caching` 时可设；须为 8 的倍数） |
| `user_specified_mamba_block_size` | `False`(init=False) | 派生标记 |
| `mamba_cache_dtype` | `"auto"` | Mamba cache（conv+ssm）dtype |
| `mamba_ssm_cache_dtype` | `"auto"` | 仅 ssm state dtype |
| `mamba_cache_mode` | `"none"` | `none`/`all`/`align`（`align`=仅缓存 step 末且对齐 `i*block_size` 的 token，Marconi APC） |

### 卸载与共享

| 字段 | 默认 | 含义 |
|---|---|---|
| `kv_sharing_fast_prefill` | `False` | YOCO 等场景 prefill 跳过（WIP）；与 EAGLE 互斥 |
| `kv_offloading_size` | `None` | KV 卸载到 CPU 的缓冲 GiB（TP>1 时为所有 TP rank 总和）；设则触发 `KVTransferConfig` |
| `kv_offloading_backend` | `"native"` | `native`/`lmcache` |

### 运行期派生（init=False）

`num_gpu_blocks`/`num_cpu_blocks`（profiling 后填）、`kv_cache_size_tokens`（per-DP 容量，混合模型 group-aware）、`kv_cache_max_concurrency`（per-DP 最大并发）、`_block_size_resolved`（防 pydantic 重跑 validator）。

### 校验器

- `_skip_none_validation`（`block_size` wrap）：`None` 跳过 `gt=0` 校验。
- `_apply_block_size_default`（model_validator after）：填默认 16，置 `user_specified_*` 标记，幂等保护。
- `_warn_deprecated_calculate_kv_scales`/`_validate_cache_dtype`：过期告警 + 量化 dtype 信息日志。

`compute_hash`（`cache.py:193`）：排除运行期/派生项（`gpu_memory_utilization`/`num_gpu_blocks`/`enable_prefix_caching`/`hash_block_size`/`kv_sharing_fast_prefill` 等），把图形状相关项（`block_size`/`cache_dtype`/`mamba_*`/`kv_cache_dtype_skip_layers`/`kv_cache_memory_bytes`/`kv_offloading_*`）纳入哈希。

## 为什么

- **块大小双轨**：`block_size`（物理块）与 `hash_block_size`（哈希粒度）分离，允许混合块池（多种 KV cache group 块大小不同）在最细公共粒度算前缀哈希后合并。
- **dtype 丰富性**：`CacheDType` 涵盖 fp8/turboquant/per-token-head/nvfp4 等压缩格式，`_validate_cache_dtype` 对 per-token-head 与量化格式打 info 日志提示精度/性能权衡。
- **Mamba cache 模式**：`all`（缓存所有 `i*block_size` 位置）vs `align`（仅 step 末对齐位置）决定 hybrid SSM 模型的状态缓存策略，直接影响块池与调度器对齐切分（`VllmConfig.validate_block_size` 中校验）。
- **卸载桥接**：`kv_offloading_size` 是用户侧简便开关，`VllmConfig._post_init_kv_transfer_config` 把它翻译成 `KVTransferConfig`（`native`→`OffloadingConnector`/`SimpleCPUOffloadConnector`，`lmcache`→`LMCacheMPConnector`），让前缀卸载复用 KV connector 体系。
- **`compute_hash` 精粒度**：物理块/dtype 影响编译图（kv cache 张量形状与算子），前缀缓存开关不影响图形状故排除。

## 怎么做

- **设块大小**：`--block-size 16`（默认）；混合模型由 `KVCacheConfig` 推导各 group 的 `block_size`，`hash_block_size` 留 `None` 取最细粒度。
- **fp8 KV**：`--kv-cache-dtype fp8`，`_validate_cache_dtype` 打 info；`calculate_kv_scales` 弃用，scale 从 checkpoint 读或默认 1.0。
- **Mamba align**：`--enable-prefix-caching --mamba-cache-mode align`，`VllmConfig.validate_block_size` 校验 `block_size <= max_num_batched_tokens` 且 `long_prefill_token_threshold >= block_size`。
- **KV 卸载**：`--kv-offloading-size 8`（GiB）+ 可选 `--kv-offloading-backend lmcache`；`VllmConfig` 自动生成 `kv_transfer_config`。
- **手动显存**：`--kv-cache-memory-bytes 5368709120` 跳过 `gpu_memory_utilization` 推导。

## 与其它模块/系统配合

- **KV 缓存管理（[`01-engine-core/kv-cache-management/`](../01-engine-core/kv-cache-management/README.md)）**：`block_size`/`cache_dtype`/`mamba_cache_mode` 驱动 `KVCacheSpec` 与 `BlockPool` 容量；`kv_cache_size_tokens` 是 `Scheduler` 准入门控依据。
- **Worker / ModelRunner（[`02-execution/`](../02-execution/README.md)）**：`gpu_memory_utilization` + profiling 决定 `num_gpu_blocks`；`cache_dtype` 决定 attention kernel 的 KV 张量类型。
- **调度器（[`01-engine-core/scheduler/scheduler.md`](../01-engine-core/scheduler/scheduler.md)）**：`enable_prefix_caching` 影响 `get_computed_blocks`/`allocate_slots`；`scheduler_reserve_full_isl` 与 `kv_cache_size_tokens` 配合防过度 admit。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`validate_nvfp4_kv_cache_with_mla`（nvfp4 与 MLA 互斥）、`validate_mamba_block_size`、`_post_init_kv_transfer_config`（卸载转 connector）。
- **KV 迁移（[kv-transfer-config.md](kv-transfer-config.md)）**：`kv_offloading_*` 经 `VllmConfig` 翻译为 `kv_transfer_config`。
- **注意力（[`05-attention/`](../05-attention/README.md)）**：`cache_dtype` 选择支持该 dtype 的后端；`sliding_window` 影响后端选择与块驱逐。
- **平台（[`08-platforms/`](../08-platforms/README.md)）**：`block_size` 由 `Platform.update_block_size_for_backend` 在 `VllmConfig.validate_block_size` 前最终确定。

## 历史版本演进

- **v0.5/v0.6（v0）**：`CacheConfig` 已存在，`block_size`/`gpu_memory_utilization`/`cache_dtype`（仅 auto/fp8）/`enable_prefix_caching`（默认 `False`）；`num_gpu_blocks` profiling 后填。
- **v0.7（v1 落地）**：`enable_prefix_caching` 默认改为 `True`；`hash_block_size` 引入支持混合块大小；`CacheDType` 扩充 `fp8_e4m3`/`fp8_e5m2`。
- **v0.8**：`mamba_cache_dtype`/`mamba_block_size`（hybrid SSM 支持）；`kv_cache_dtype_skip_layers`；`prefix_caching_hash_algo` 多算法（xxhash/cbor）。
- **v0.9**：`kv_offloading_size`/`kv_offloading_backend` 字段加入，桥接 `KVTransferConfig`；`mamba_cache_mode`（`all`/`align`/`none`）；`kv_sharing_fast_prefill`（WIP）；`kv_cache_memory_bytes` 手动显存。
- **v0.10**：`kv_cache_size_tokens`/`kv_cache_max_concurrency` per-DP group-aware 派生；`turboquant_*`/`per_token_head`/`nvfp4` dtype 加入；`calculate_kv_scales` 标记 v0.19 弃用。
- **v0.11 / v0.12 / main**：`mamba_cache_mode="align"`（Marconi APC）与 `validate_block_size` 联动校验；`nvfp4` 与 MLA 互斥校验（`validate_nvfp4_kv_cache_with_mla`）；`skip_page_size_padded` 配合 `kv_cache_dtype_skip_layers`。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `_post_init_kv_transfer_config`/`validate_*` 校验。
- [kv-transfer-config.md](kv-transfer-config.md) — `kv_offloading_*` 翻译为 connector。
- [mamba-config.md](mamba-config.md) — `mamba_cache_*` 与 Mamba SSU 后端配合。
- [../01-engine-core/kv-cache-management/README.md](../01-engine-core/kv-cache-management/README.md) — 消费方：块池/spec/coordinator。
- [../02-execution/worker/gpu-worker.md](../02-execution/worker/gpu-worker.md) — profiling `num_gpu_blocks`。
