# RunaiModelStreamerLoader：对象存储流式加载

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **Run:ai**

> 源码：`vllm/model_executor/model_loader/runai_streamer_loader.py`

---

## 是什么

`RunaiModelStreamerLoader`（`runai_streamer_loader.py:21`）对应 `load_format="runai_streamer"`，使用 [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) 库流式加载 safetensors，支持**本地 FS / S3 / GCS / Azure Blob Storage**。相比 vLLM 内置的 `safe_open` 逐 tensor 读取，Run:ai streamer 用多流并发与对象存储原生 API，对云存储冷启动更友好。

`load_format="runai_streamer_sharded"` 走的是 `ShardedStateLoader`（见 [`sharded-state.md`](sharded-state.md)），但迭代器复用本 loader 同源的 `runai_safetensors_weights_iterator`。

---

## 为什么

云上部署时权重常存在 S3/GCS/Azure Blob，而 `huggingface_hub.snapshot_download` 要先把权重全量下载到本地盘再读，冷启动慢且占磁盘。Run:ai streamer 直接对对象存储做流式读，边读边喂给模型，省掉本地落盘。对大模型（几十到上百 GB）能显著缩短首次加载。

---

## 怎么做

### `__init__`（`runai_streamer_loader.py:27`）

`model_loader_extra_config` 仅允许三个 key：

| key | 类型 | 作用 |
|---|---|---|
| `distributed` | bool | 是否分布式直接读到 cuda（多 rank 各读一部分） |
| `concurrency` | int >0 | 写入 `RUNAI_STREAMER_CONCURRENCY` 环境变量 |
| `memory_limit` | int ≥-1 | 写入 `RUNAI_STREAMER_MEMORY_LIMIT`（-1 为不限制） |

先把所有要写的 env 收集到 `env_updates` 校验通过后再 `os.environ.update`（避免部分应用）。若 `RUNAI_STREAMER_S3_ENDPOINT` 未设但 `AWS_ENDPOINT_URL` 已设，则把后者复制给前者（兼容自定义 S3 endpoint）。

### `_prepare_weights`（`runai_streamer_loader.py:80`）

- `is_runai_obj_uri(path)` 或本地目录 → 直接用 path。
- 否则 `download_weights_from_hf` 下载 `*.safetensors`（非对象存储路径走 HF）。
- `list_safetensors(path=hf_folder)` 列出 safetensors 文件。
- 远程非对象存储时补下载 `model.safetensors.index.json`。

### `_get_weights_iterator` / `load_weights`

`runai_safetensors_weights_iterator(files, use_tqdm, distributed)`（实现在 `weight_utils.py:987`）：用 `SafetensorsStreamer` 流式产出 `(name, tensor)`；`distributed=True` 且 CUDA 平台时直接读到 `cuda:{current_device}`。`load_weights` 支持 `model_config.model_weights` 覆盖路径（`runai_streamer_loader.py:135`）。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `runai_model_streamer`（外部包） | `SafetensorsStreamer`；缺失时 `PlaceholderModule` 占位 |
| `vllm/transformers_utils/runai_utils` | `is_runai_obj_uri`、`list_safetensors` |
| `weight_utils.py` | `runai_safetensors_weights_iterator`、`download_weights_from_hf` |
| `ShardedStateLoader` | `runai_streamer_sharded` 复用同一迭代器 |
| `vllm/platforms` | `current_platform.is_cuda_alike()` 决定 distributed 模式目标 device |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| 中期（待核实） | 引入 `RunaiModelStreamerLoader`，支持 S3/GCS/Azure |
| main（#45291） | 校验 `model_loader_extra_config` 合法 key，避免误配静默 |
| main（#45308） | revision 按 name 传给 index 下载 |
| main（#47337） | 允许 `memory_limit` 哨兵值 |

---

## 参见

- [`sharded-state.md`](sharded-state.md) —— `runai_streamer_sharded` 共用迭代器
- [`weight-utils.md`](weight-utils.md) —— `runai_safetensors_weights_iterator`
- [`../README.md`](../README.md) —— 返回模型执行首页
