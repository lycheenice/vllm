# LoadConfig（load.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > LoadConfig

源码：`vllm/config/load.py`（约 149 行）。`LoadConfig` 描述模型权重加载方式：加载格式、safetensors 策略、下载目录、ignore patterns、进度条、`map_location` 等。它是 `VllmConfig.load_config`，被 `vllm/model_executor/model_loader/` 的各 loader 消费。它是"不参与图形状"的配置——`compute_hash` 返回空 factors。

## 是什么

`@config` 装饰（`load.py:26`）。`DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS=8`、`DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE=16*1024*1024`。

| 字段 | 默认 | 含义 |
|---|---|---|
| `load_format` | `"auto"` | 加载格式，校验器转小写。见下表 |
| `download_dir` | `None` | 权重下载目录，默认 HF cache |
| `safetensors_load_strategy` | `None` | `lazy`/`eager`/`prefetch`/`torchao`；`None`=mmap 懒加载，NFS+内存够时自动 prefetch |
| `safetensors_prefetch_num_threads` | `8` | prefetch 线程数 |
| `safetensors_prefetch_block_size` | `16MiB` | prefetch 读块大小 |
| `model_loader_extra_config` | `{}` | loader 专用额外配置（如 `TensorizerConfig`） |
| `device` | `None` | 加载目标设备，默认回退 `device_config.device` |
| `ignore_patterns` | `["original/**/*"]` | 下载时忽略的路径模式 |
| `use_tqdm_on_load` | `True` | 加载进度条 |
| `pt_load_map_location` | `"cpu"` | `torch.load` 的 `map_location`，支持 `"cuda"`/`{"cuda:1":"cuda:0"}` |

**`load_format` 取值**（`load.py:30` docstring）：`auto`/`pt`/`safetensors`/`instanttensor`/`npcache`/`dummy`(profiling)/`tensorizer`/`runai_streamer`/`runai_streamer_sharded`/`bitsandbytes`/`sharded_state`/`mistral`/`modelexpress`/自定义插件。

`safetensors_load_strategy`：
- `None`（默认）：mmap 懒加载；NFS 且 checkpoint < 90% RAM 时自动 prefetch。
- `lazy`：mmap，禁用 NFS 自动 prefetch。
- `eager`：整个文件读入 CPU 内存（网络盘推荐）。
- `prefetch`：OS page cache 预读。
- `torchao`：加载后重建为 torchao tensor 子类（需 `torchao >= 0.14.0`）。

`compute_hash`（`load.py:117`）：返回空 factors——加载方式不影响编译图形状。

校验器：`_lowercase_load_format`（转小写）、`_validate_ignore_patterns`（非默认时打 info）。

## 为什么

- **加载与图解耦**：加载方式（pt/safetensors/runai/...）只影响"如何把权重放进显存"，不影响"前向图结构"。故 `compute_hash` 空，加载策略变更不致编译缓存失效。
- **网络盘优化**：`safetensors_load_strategy` 的 `eager`/`prefetch` 针对 Lustre/NFS 的随机读低效，整文件预读大幅提速初始化（代价是 CPU RAM）。
- **loader 插件化**：`load_format` 支持自定义值，配合 `model_loader_extra_config` 让外部 loader（如 `modelexpress`）无需改核心。
- **量化协同**：`bitsandbytes` load_format 与 `ModelConfig.quantization` 协同；`VllmConfig.try_verify_and_update_config` 检测 RunAI URI 时把 `auto` 改写为 `runai_streamer`。

## 怎么做

- **默认**：`--model ...` 不设 `--load-format`，走 `auto`（优先 safetensors，回退 pt）。
- **网络盘**：`--load-format safetensors --safetensors-load-strategy eager` 加速。
- **profiling**：`--load-format dummy` 随机初始化（不读盘）。
- **RunAI 对象存储**：`model` 为 S3/GCS URI 时 `VllmConfig` 自动改 `runai_streamer`；或显式 `--load-format runai_streamer_sharded`。
- **ignore**：`--ignore-patterns '["original/**/*","*.msgpack"]'`。

## 与其它模块/系统配合

- **Model Loader（[`03-model-execution/model-loader/`](../03-model-execution/model-loader/README.md)）**：`load_format` dispatch 到 `DefaultModelLoader`/`TensorizerLoader`/`RunaiModelLoader`/`ShardedStateLoader`/`ModelExpressLoader`/`BnbModelLoader` 等；`safetensors_load_strategy` 控制 `SafetensorsModelLoader` 行为。
- **`ModelConfig`（[model-config.md](model-config.md)）**：`quantization` + `load_format` 决定是否走量化 loader；`model_weights`（RunAI）保留原 URI。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`try_verify_and_update_config` 检测 RunAI URI 做 `load_format` 自动改写；`_get_quantization_config` 用 `load_config` 取 quant config。
- **量化（[quantization-config.md](quantization-config.md)）**：`get_quant_config(model_config, load_config)` 读量化参数。
- **`DeviceConfig`（[device-config.md](device-config.md)）**：`load_config.device` 默认回退 `device_config.device`。

## 历史版本演进

- **v0.5/v0.6（v0）**：加载相关字段散在 `ModelConfig`/`EngineArgs`（`load_format`/`download_dir`/`use_tqdm_on_load`）。
- **v0.7（v1 落地）**：抽出独立 `LoadConfig`，集中加载参数；`safetensors_load_strategy` 字段引入。
- **v0.8**：`runai_streamer`/`runai_streamer_sharded` load_format；`sharded_state`；`mistral`。
- **v0.9**：`instanttensor`（InstantTensor 分布式加载）；`prefetch` 策略 + `safetensors_prefetch_*` 参数；`pt_load_map_location` 暴露。
- **v0.10**：`modelexpress` load_format；`torchao` 策略（torchao 量化 checkpoint）；`ignore_patterns` 默认 `["original/**/*"]`。
- **v0.11 / v0.12 / main**：`assigned_physical_gpu_ids` 协同加载设备映射；`tensorizer`/`bitsandbytes` 兼容性维护。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — `quantization`/`model_weights` 与加载协同。
- [device-config.md](device-config.md) — `load_config.device` 回退源。
- [quantization-config.md](quantization-config.md) — `get_quant_config` 消费 `load_config`。
- [../03-model-execution/model-loader/README.md](../03-model-execution/model-loader/README.md) — loader dispatch 消费方。
