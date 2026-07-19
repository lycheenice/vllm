# weight_utils：HF 下载与权重迭代器

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **weight_utils**

> 源码：`vllm/model_executor/model_loader/weight_utils.py`

---

## 是什么

`weight_utils.py` 是默认加载路径的"工具箱"，向 `DefaultModelLoader` / `ShardedStateLoader` / `RunaiModelStreamerLoader` / `BitsAndBytesModelLoader` 等提供：

- **下载**：`download_weights_from_hf`、`download_safetensors_index_file_from_hf`、`maybe_download_from_modelscope`。
- **文件筛选**：`filter_duplicate_safetensors_files`、`filter_files_not_needed_for_inference`。
- **权重迭代器**（产出 `(name, torch.Tensor)`）：`safetensors_weights_iterator`、`multi_thread_safetensors_weights_iterator`、`pt_weights_iterator`、`multi_thread_pt_weights_iterator`、`np_cache_weights_iterator`、`fastsafetensors_weights_iterator`、`instanttensor_weights_iterator`、`runai_safetensors_weights_iterator`。
- **量化配置读取**：`get_quant_config`、`get_sparse_attention_config`。
- **weight_loader 回调工厂**：`default_weight_loader`、`row_parallel_weight_loader`、`sharded_weight_loader`、`composed_weight_loader`。
- **dummy 初始化**：`initialize_dummy_weights`、`initialize_single_dummy_weight`。
- **公共工具**：`get_lock`（跨用户 filelock）、`atomic_writer`、`enable_xet_high_performance`。

---

## 为什么

把"如何从一种文件格式产出张量流"集中到一个文件，让 loader 只做格式选择。不同存储介质（本地 SSD / NFS / Lustre / S3 / GCS / Azure Blob）与不同 checkpoint 布局（单文件 / 分片 index / 预分片 / 序列化）各有最优读法：

- NFS/Lustre 上随机 mmap 读极慢，故有 `eager`/`prefetch` 策略与 page-cache 预取。
- 大模型分片多，多线程并发读 shard 能显著缩短加载时间（`multi_thread_*`）。
- safetensors 的 `weight_map` index 能避免下载 `original/` 子目录里的重复文件。
- EP 下专家权重占比 ~85–90%，读前 `should_skip_weight` 跳过非本 rank 专家张量。

---

## 怎么做

### 下载链路

`download_weights_from_hf`（`weight_utils.py:431`）：

1. 先 `hf_fs().ls(...)` 列远程文件，若 `*.safetensors` 且存在 `model.safetensors.index.json`，下载 index 并把 `allow_patterns` 收紧为 index 中实际出现的文件名集合（`weight_utils.py:474`），避免下载 `original/` 等冗余目录。
2. 否则在 `allow_patterns` 里挑第一个匹配远程文件列表的 pattern。
3. `get_lock(model, cache_dir)` 跨进程/跨用户互斥（lock 文件名 `sha256(model)+model.lock`，`mode=0o666`）。
4. `hf_api().snapshot_download(...)` 下载，命中即 break。

`download_safetensors_index_file_from_hf`（`weight_utils.py:538`）单独下载 index 文件，容错 `LocalEntryNotFoundError`/`EntryNotFoundError`。`enable_xet_high_performance()`（`weight_utils.py:74`）在导入时自动开 HF Xet 高性能模式。

### safetensors 迭代器（`weight_utils.py:820`）

`_prepare` 阶段先探测文件系统类型（`_get_fs_type` 读 `/proc/mounts` 最长前缀匹配，`weight_utils.py:695`），判断是否 NFS/Lustre；再比对 checkpoint 总大小与可用 RAM（`_get_checkpoints_size_bytes` / `_get_available_ram_bytes`），决定是否自动 prefetch（默认阈值 90% RAM）。`safetensors_load_strategy`：

| 策略 | 行为 |
|---|---|
| `None`（默认） | NFS/Lustre 且 fitting-in-RAM 时自动 prefetch；否则 lazy `safe_open` |
| `lazy` | 纯 `safe_open` 内存映射，不做 prefetch |
| `eager` | `load(f.read())` 整文件读入 CPU |
| `prefetch` | `_prefetch_all_checkpoints` 后台线程把文件读进 OS page cache（`weight_utils.py:745`） |
| `torchao` | `safe_open` 读 + `unflatten_tensor_state_dict` 重组 torchao 子类（需 torchao≥0.15） |

迭代器内部每条 tensor 都过 `should_skip_weight(name, local_expert_ids)`（EP 过滤，`weight_utils.py:916`）。

### 多线程迭代器

`multi_thread_safetensors_weights_iterator`（`weight_utils.py:957`）用 `ThreadPoolExecutor` 并发 `load_file`，`as_completed` 顺序产出，控制内存不一次性持有所有 shard。`multi_thread_pt_weights_iterator` 同理。

### pt / npcache / fastsafetensors / instanttensor / runai

- `pt_weights_iterator`（`weight_utils.py:1133`）：`torch.load(..., map_location=pt_load_map_location)` 逐 shard。
- `np_cache_weights_iterator`（`weight_utils.py:635`）：首次把 `.bin` 转 numpy 落盘 `np/weight_names.json`，后续直接 `np.load`。
- `fastsafetensors_weights_iterator` / `instanttensor_weights_iterator`：依赖可选包 `fastsafetensors` / `InstantTensor`（`weight_utils.py:1024`/`1093`）。
- `runai_safetensors_weights_iterator`（`weight_utils.py:987`）：用 `SafetensorsStreamer` 流式产出，支持分布式直接读到 cuda。

### weight_loader 回调

- `default_weight_loader(param, loaded_weight)`（`weight_utils.py:1198`）：标量 reshape 后 `copy_`，否则断言 shape 一致后 `copy_`。
- `row_parallel_weight_loader`：按 `tp_rank` 在 dim 0（或 None）`narrow` 后调 default。
- `sharded_weight_loader(shard_axis)`：返回闭包，按指定轴 narrow。
- `composed_weight_loader(loader, fn)`：先 `loader`，再 `param.data.copy_(fn(param))`。

### 量化配置读取

`get_quant_config`（`weight_utils.py:240`）按优先级：

1. `model_config.quantization_config`（`QuantizationConfigArgs`）→ `OnlineQuantizationConfig`（在线量化）。
2. `hf_config.quantization_config`（或 `compression_config`，compressed-tensors）→ `quant_cls.from_config(...)`。
3. `hf_overrides["quantization_config_file"]` / `"quantization_config_dict_json"` → `from_config_file` / `from_config_dict_json`。
4. 否则下载 `*.json`，按 `quant_cls.get_config_filenames()` 找配置文件并 `from_config`。

### dummy 初始化

`initialize_dummy_weights`（`weight_utils.py:1265`）对每个参数调 `initialize_single_dummy_weight`；后者用**每参数独立随机种子**（依赖 numel + dtype），保证 TP/PP 分片下各 rank 上同一逻辑参数产生相同值（`weight_utils.py:1288`）。meta tensor 跳过（留给 `finalize_layerwise_processing`）。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `config/load.py::LoadConfig` | `safetensors_load_strategy`/`safetensors_prefetch_*`/`pt_load_map_location`/`download_dir`/`ignore_patterns`/`use_tqdm_on_load` 都来自这里 |
| `ep_weight_filter.py` | `should_skip_weight` 被 `safetensors_weights_iterator` 调用 |
| `vllm/transformers_utils/repo_utils` | `hf_api`/`hf_fs`/`list_filtered_repo_files` |
| `vllm/transformers_utils/s3_utils` | `is_s3`/`s3_glob`（供 ShardedStateLoader 用，非本文件） |
| `vllm/distributed` | `row_parallel_weight_loader`/`sharded_weight_loader` 用 `get_tensor_model_parallel_rank` |
| `layers/quantization` | `get_quant_config` 读 quant config 类 |
| `vllm/platforms` | `current_platform` 决定 runai 迭代器目标 device；`is_pin_memory_available` 间接影响 dummy |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 早期 | `download_weights_from_hf` + `safetensors_weights_iterator` + `pt_weights_iterator` |
| 中期 | 引入 `safetensors_load_strategy` 三策略（lazy/eager/prefetch）与 page-cache 预取；ModelScope 下载支持 |
| 中期 | `multi_thread_*` 迭代器；`npcache` 保留为兼容路径 |
| main（#37136） | `safetensors_weights_iterator` 增加 `local_expert_ids` 参数，逐 tensor 调 `should_skip_weight` |
| main | NFS/Lustre 文件系统探测 + RAM 阈值自动决定 prefetch；index 文件收紧下载 pattern |
| main | `torchao` 策略 + `unflatten_tensor_state_dict` 支持 |
| main | HF Xet 高性能模式自动启用（`weight_utils.py:74`） |

---

## 参见

- [`default.md`](default.md) —— 谁在调用这些迭代器
- [`ep-weight-filter.md`](ep-weight-filter.md) —— `should_skip_weight` 的来源
- [`../README.md`](../README.md) —— 返回模型执行首页
