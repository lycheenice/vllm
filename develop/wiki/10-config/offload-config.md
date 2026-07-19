# OffloadConfig + PrefetchOffloadConfig + UVAOffloadConfig + OffloadBackend（offload.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > OffloadConfig

源码：`vllm/config/offload.py`（约 153 行）。`OffloadConfig` 描述模型**权重** CPU 卸载以减 GPU 显存：`uva`（Unified Virtual Addressing 零拷贝）与 `prefetch`（异步预取分组）两种 backend，各自有子配置。它是 `VllmConfig.offload_config`，被 `vllm/model_executor/offloader.py` 的 `OffloadLoader`/`PrefetchOffloader` 消费。注意：这是**权重**卸载，与 KV cache 卸载（`CacheConfig.kv_offloading_*`→`KVTransferConfig`）正交。

## 是什么

`OffloadBackend = Literal["auto","uva","prefetch"]`。

### `UVAOffloadConfig`（`offload.py:15`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `cpu_offload_gb` | `0` | 每 GPU 卸载到 CPU 的 GiB；0=不卸载。直观上"虚拟扩显存"（24GB GPU + 10GB ≈ 34GB，可装 13B BF16） |
| `cpu_offload_params` | `set()` | 按参数名段匹配定向卸载（空=非选择性地卸到限额）；`"experts"`/`"experts.w2_weight"` 匹配 `"mlp.experts.w2_weight"`，`"expert"`/`"w2"` 不匹配（须精确段） |

### `PrefetchOffloadConfig`（`offload.py:47`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `offload_group_size` | `0`(禁用) | 每 N 层一组，卸载每组末 `offload_num_in_group` 层；0=禁用。如 group=8,num=2 卸载 6,7,14,15,... |
| `offload_num_in_group` | `1` | 每组卸载层数，须 ≤ `offload_group_size` |
| `offload_prefetch_step` | `1` | 预取提前层数；大则隐藏更多延迟但耗显存 |
| `offload_params` | `set()` | 预取卸载定向参数段（空=卸整层全部参数） |

### `OffloadConfig`（`offload.py:79`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `offload_backend` | `"auto"` | `auto`(按子配置非默认值选: prefetch>0→prefetch，uva>0→uva)/`uva`/`prefetch` |
| `uva` | `UVAOffloadConfig()` | UVA 子配置 |
| `prefetch` | `PrefetchOffloadConfig()` | prefetch 子配置 |

校验（`validate_offload_config`）：prefetch 启用时 `offload_num_in_group ≤ offload_group_size` 且 `offload_prefetch_step ≥ 1`；backend 与子配置不匹配时 warning（如 `uva` backend 但 prefetch 字段非默认→prefetch 被忽略）。

`compute_hash`：**全字段纳入**（无 `ignored_factors`）——`PrefetchOffloader` 会 patch module forward 并插入 `wait_prefetch`/`start_prefetch` custom op 进计算图，改任何卸载设置都可能改变哪些层被 hook 与预取索引，故编译缓存须区分。

## 为什么

- **两种卸载策略**：
  - **UVA**：CPU pinned 内存零拷贝 GPU 访问，简单但需快 CPU-GPU 互连。每个 forward 实时从 CPU 读权重，无预取。
  - **prefetch**：按层分组，异步 H2D 预取隐藏传输延迟。`offload_group_size=8, offload_num_in_group=2` 把每 8 层的末 2 层卸载，forward 时提前 `offload_prefetch_step` 层预取。比 UVA 更激进，需要 careful 索引管理。
- **定向卸载**：`cpu_offload_params`/`offload_params` 用段匹配让用户只卸载大参数（如 MoE expert 权重），保留小/热参数在 GPU。段匹配（非子串）区分 `w2_weight` 与 `w2_weight_scale`。
- **`auto` backend**：按子配置非默认值自动选，避免用户同时设 backend 与子配置的矛盾。
- **`compute_hash` 全纳入**：因 `PrefetchOffloader` 改图（插 custom op + hook forward），缓存键须反映卸载拓扑。UVA 虽不改图，但与 prefetch 共用 `OffloadConfig` 哈希以简化。
- **与权重迁移区分**：权重卸载是**单实例内** CPU↔GPU；权重迁移（`WeightTransferConfig`）是**跨实例** RL 训练热更新。二者正交。

## 怎么做

- **UVA**：`--offload-backend uva --cpu-offload-gb 10`（13B BF16 装 24GB GPU）。
- **定向 UVA**：`--cpu-offload-gb 10 --cpu-offload-params '["experts"]'`（仅卸 expert）。
- **prefetch**：`--offload-backend prefetch --offload-group-size 8 --offload-num-in-group 2 --offload-prefetch-step 2`。
- **auto**：`--cpu-offload-gb 10`（auto→uva）或 `--offload-group-size 8`（auto→prefetch）。

## 与其它模块/系统配合

- **Offloader（[`03-model-execution/offloader.md`](../03-model-execution/offloader.md)）**：`offload_backend` 驱动 `OffloadLoader`(UVA)/`PrefetchOffloader` 选择；后者 patch module forward 插 `wait_prefetch`/`start_prefetch`。
- **LoadConfig（[load-config.md](load-config.md)）**：`PrefetchOffloader` 在加载阶段决定哪些层卸载、参数分桶；与 `load_format`/`safetensors_load_strategy` 协同。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`PrefetchOffloader` 插入的 custom op 进编译图，故 `offload_config.compute_hash` 纳入 `VllmConfig.compute_hash`。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`offload_config.compute_hash()` 进顶层哈希；`_validate_v2_model_runner` 当前未把 offload 列为 V2 不支持（UVA/prefetch 与 V2 兼容性 `(待核实)`）。
- **KV 卸载（[cache-config.md](cache-config.md) 与 [kv-transfer-config.md](kv-transfer-config.md)）**：KV cache 卸载走 `CacheConfig.kv_offloading_*`→`KVTransferConfig`，与权重卸载正交，可共存。

## 历史版本演进

- **v0.5–v0.8**：仅 `--cpu-offload-gb`（UVA 零拷贝），字段在 `ModelConfig`/`ParallelConfig`。
- **v0.9（待核实）**：`OffloadConfig` + `UVAOffloadConfig`/`PrefetchOffloadConfig` 独立子配置；`offload_backend` 三态；`PrefetchOffloader` 成形。
- **v0.11/v0.12（UVA offload 重引入，待核实）**：`cpu_offload_params`/`offload_params` 段匹配定向卸载（MoE expert 场景）；`compute_hash` 全纳入；与 MRv2 pooling 兼容性。
- **main**：prefetch 与 cudagraph 兼容性细化；定向卸载对 quantized 参数（`w2_weight` vs `w2_weight_scale`）的段匹配精确化。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [load-config.md](load-config.md) — 加载阶段与 offload 协同。
- [compilation-config.md](compilation-config.md) — `PrefetchOffloader` 插入的 custom op 进图。
- [kv-transfer-config.md](kv-transfer-config.md) — KV cache 卸载（与权重卸载正交）。
- [weight-transfer-config.md](weight-transfer-config.md) — 跨实例权重迁移（与单实例卸载正交）。
- [../03-model-execution/offloader.md](../03-model-execution/offloader.md) — Offloader 消费方。
