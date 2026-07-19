# LoRAConfig（lora.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > LoRAConfig

源码：`vllm/config/lora.py`（约 131 行）。`LoRAConfig` 描述 LoRA 适配器的部署参数：rank 上限、批内 LoRA 数、sharding、dtype、目标模块、多模态 LoRA、MoE LoRA 格式等。它是 `VllmConfig.lora_config`（`None` 表示未启用 LoRA），被 `vllm/lora/` 子系统与 model runner 消费。

## 是什么

`@config(config=ConfigDict(arbitrary_types_allowed=True))` 装饰（`lora.py:30`）。`LoRADType = Literal["auto","float16","bfloat16"]`、`MaxLoRARanks = Literal[1,8,16,32,64,128,256,320,512]`、`LoRAExtraVocabSize = Literal[256,512]`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `max_lora_rank` | `16` | 最大 LoRA rank（受限于 `MaxLoRARanks`） |
| `max_loras` | `1` | 单批最多 LoRA 数 |
| `fully_sharded_loras` | `False` | TP 下全分片 LoRA（高 seq/rank/TP 更快） |
| `max_cpu_loras` | `None`(→`max_loras`) | CPU 内存存 LoRA 上限，须 ≥ `max_loras` |
| `lora_dtype` | `"auto"` | LoRA dtype，`auto`=随基模型 |
| `target_modules` | `None` | 限制 LoRA 应用模块后缀（如 `["o_proj","qkv_proj"]`） |
| `default_mm_loras` | `None` | 多模态：`{modality: lora_path}`，该模态总激活某 LoRA |
| `enable_tower_connector_lora` | `False` | 多模态 tower（视觉编码器）+connector LoRA（实验，Qwen VL 等） |
| `specialize_active_lora` | `False` | 按活跃 LoRA 数（2 幂）各捕一份 cudagraph（需 `cudagraph_specialize_lora`） |
| `enable_mixed_moe_lora_format` | `False` | 强制用 2D MoE LoRA wrapper，让 2D/3D 格式 MoE LoRA 同部署 |

校验：`_validate_lora_config`（`max_cpu_loras` 默认设 `max_loras`，<则 raise；`VLLM_LORA_ENABLE_DUAL_STREAM` 仅 CUDA；与 `fully_sharded_loras` 冲突时关 dual stream）。`verify_with_model_config`（`lora_dtype` 从 `"auto"` 解析为基模型 `torch.dtype`）。

`compute_hash`：纳入 `max_lora_rank`/`max_loras`/`fully_sharded_loras`/`lora_dtype`/`enable_tower_connector_lora`/`enable_mixed_moe_lora_format`/`target_modules`（排序元组），因这些影响 LoRA 层 buffer 形状与编译图。

## 为什么

- **静态 buffer 形状**：LoRA 按 `max_lora_rank`/`max_loras` 预分配 Punica kernel buffer，这些尺寸被 torch.compile 捕获进图，故进哈希。`max_num_batched_tokens` 也因此进 `SchedulerConfig.compute_hash`。
- **rank 离散化**：`MaxLoRARanks` 限定 rank 取值，让 cudagraph 按 rank 分桶捕获，避免无限尺寸组合。
- **sharding 策略**：`fully_sharded_loras` 在高 TP/seq/rank 时更优（全分片 vs 半分片），但与 dual stream 不兼容。
- **多模态 LoRA**：`default_mm_loras` 让某模态永远激活指定 LoRA（自动分配 ID）；`enable_tower_connector_lora` 扩展到 ViT tower + connector。
- **MoE LoRA 混合格式**：`enable_mixed_moe_lora_format` 用 `FusedMoEWithLoRA` 2D wrapper，让 2D 与 3D 格式 MoE LoRA 同部署（模型驱动行为被覆盖）。
- **specialize_active_lora**：按活跃 LoRA 数分桶捕 cudagraph，减少无 adapter 时的 LoRA op 开销（代价是启动时间/显存）。

## 怎么做

- **启用 LoRA**：`--enable-lora --max-lora-rank 64 --max-loras 4 --fully-sharded-loras`。
- **dtype**：`--lora-dtype bfloat16`（默认随基模型）。
- **目标模块**：`--target-modules '["q_proj","v_proj"]'`。
- **多模态**：`--default-mm-loras '{"image":"/path/img_lora"}'` + `--enable-tower-connector-lora`。
- **MoE**：`--enable-mixed-moe-lora-format` 让 2D/3D MoE LoRA 共存。
- **cudagraph 专门化**：`--lora-specialize-active-lora`（配合 `-cc.cudagraph-specialize-lora`，V1 默认 `cudagraph_specialize_lora=True`）。

## 与其它模块/系统配合

- **LoRA 子系统（[`12-lora/`](../12-lora/README.md)）**：`max_lora_rank`/`max_loras` 驱动 `LRUCacheWorkerLoRAManager` 容量；`fully_sharded_loras` 选 LoRA 层实现；`target_modules` 过滤。
- **Worker / ModelRunner（[`02-execution/worker/lora-mixin.md`](../02-execution/worker/lora-mixin.md)）**：`cudagraph_specialize_lora`/`specialize_active_lora` 决定 cudagraph 捕获桶；`VLLM_LORA_ENABLE_DUAL_STREAM` 双流。
- **SchedulerConfig（[scheduler-config.md](scheduler-config.md)）**：`max_num_batched_tokens` 决定 LoRA 静态 buffer 形状（故进 sched 哈希）。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`cudagraph_specialize_lora` 控制是否按有无 LoRA 各捕一份图；`specialize_active_lora` 按活跃数细分。
- **ModelConfig（[model-config.md](model-config.md)）**：`verify_with_model_config` 读基模型 dtype；MoE 模型与 `enable_mixed_moe_lora_format` 协同。
- **多模态（[multimodal-config.md](multimodal-config.md)）**：`default_mm_loras` 与 `mm_processor_kwargs` 配合。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`lora_config` 非 None 时 `__post_init` 调 `verify_with_model_config`。

## 历史版本演进

- **v0.5/v0.6（v0）**：`LoRAConfig` 已存在，`max_lora_rank`/`max_loras`/`lora_dtype`/`target_modules`；v0 Punica 内核。
- **v0.7（v1 落地）**：v1 LoRA 路径；`fully_sharded_loras`；`default_mm_loras` 多模态 LoRA；`MaxLoRARanks` 离散化。
- **v0.8**：`enable_tower_connector_lora`（ViT tower+connector）；`specialize_active_lora` cudagraph 分桶。
- **v0.9**：`enable_mixed_moe_lora_format`（2D/3D MoE LoRA 共存）；`VLLM_LORA_ENABLE_DUAL_STREAM` 双流。
- **v0.10–main**：`compute_hash` 纳入 `target_modules`（排序元组）；与 MRv2 的 LoRA mixin 协同；dual stream 与 fully_sharded 互斥处理。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [scheduler-config.md](scheduler-config.md) — `max_num_batched_tokens` 影响 LoRA buffer。
- [compilation-config.md](compilation-config.md) — `cudagraph_specialize_lora`。
- [multimodal-config.md](multimodal-config.md) — `default_mm_loras`。
- [../12-lora/README.md](../12-lora/README.md) — LoRA 子系统消费方。
- [../02-execution/worker/lora-mixin.md](../02-execution/worker/lora-mixin.md) — Worker LoRA mixin。
