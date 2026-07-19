# MultiModalConfig（multimodal.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > MultiModalConfig

源码：`vllm/config/multimodal.py`（约 348 行）。`MultiModalConfig` 描述多模态模型的行为控制：每 prompt 各模态限额、编码器 TP 模式、IPC 方式、processor cache、ViT FP8 注意力、视频剪枝等。它是 `ModelConfig.multimodal_config`（内嵌于 `model_config`，非 `VllmConfig` 顶层字段），被 `vllm/multimodal/` 与 scheduler/encoder cache 消费。

## 是什么

`@config` 装饰（`multimodal.py:73`）。辅助 dataclass：`BaseDummyOptions`/`ImageDummyOptions`/`VideoDummyOptions`/`AudioDummyOptions`（profiling dummy 数据生成选项）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `language_model_only` | `False` | True 则所有模态限额置 0（等价 `--limit-mm-per-prompt` 全 0） |
| `limit_per_prompt` | `{}` | 每 prompt 每模态上限与选项；支持 `{"image":16}` 或 `{"video":{"count":1,"num_frames":32,...}}` 混合格式，校验器规范化为 `DummyOptions` 子类 |
| `enable_mm_embeds` | `False` | 允许传多模态 embedding（tensor/`*_embeds` 消息） |
| `media_io_kwargs` | `{}` | 媒体 IO 额外参数，按模态分桶（如 `{"video":{"num_frames":40}}`） |
| `mm_processor_kwargs` | `None` | 传给 `AutoProcessor` 的覆写（如 `{"num_crops":4}`） |
| `mm_processor_cache_gb` | `4` | processor 缓存 GiB（API+EngineCore 各一份，总量 = `×(api_count+dp_size)`） |
| `mm_processor_cache_type` | `"lru"` | `shm`(共享内存 FIFO)/`lru`(镜像 LRU) |
| `mm_shm_cache_max_object_size_mb` | `128` | shm 缓存单对象上限（仅 `shm` 有效） |
| `mm_encoder_only` | `False` | 仅跑编码器（disagg Encoder 进程） |
| `mm_encoder_tp_mode` | `"weights"` | `weights`(层内 TP 分权重)/`data`(批内 DP 分数据，每 rank 全权重) |
| `mm_encoder_attn_backend` | `None` | ViT 编码器注意力后端覆写 |
| `mm_encoder_attn_dtype` | `None` | ViT 注意力 dtype 覆写，`"fp8"` 启用 FlashInfer cuDNN FP8 |
| `mm_encoder_fp8_scale_path` | `None` | 静态 FP8 scale JSON 路径 |
| `mm_encoder_fp8_scale_save_path` | `None` | 动态 scale 校准后保存路径 |
| `mm_encoder_fp8_scale_save_margin` | `1.5` | 自动保存 scale 安全裕度 |
| `interleave_mm_strings` | `False` | `--chat-template-content-format=string` 下全交错多模态 prompt |
| `skip_mm_profiling` | `False` | 跳过多模态内存 profiling（仅 profile 语言骨干） |
| `video_pruning_rate` | `None` | 视频剪枝率 [0,1)（Efficient Video Sampling） |
| `mm_tensor_ipc` | `"direct_rpc"` | 多模态张量 IPC：`direct_rpc`(msgspec)/`torch_shm`(零拷贝，需 `VLLM_WORKER_MULTIPROC_METHOD=spawn`) |
| `mm_ipc_gpu_memory_gb` | `0` | 前端 GPU 多模态工作显存预留（硬件视频解码等），从 KV cache 显存切出 |

`MMEncoderTPMode = Literal["weights","data"]`、`MMCacheType = Literal["shm","lru"]`、`MMTensorIPC = Literal["direct_rpc","torch_shm"]`。

校验器/方法：`_validate_limit_per_prompt`（混合格式→`DummyOptions`）、`_validate_mm_encoder_attn_backend`（`XFORMERS` 已移除）、`_validate_multimodal_config`（shm + 对象大小约束、FP8 scale 路径组合与文件存在性）、`compute_hash`（`mm_encoder_attn_backend`/`mm_encoder_tp_mode`/`mm_encoder_attn_dtype`/`mm_encoder_fp8_scale_path` 纳入；`mm_encoder_*` 编码器层影响编译图）、`get_limit_per_prompt(modality)`、`merge_mm_processor_kwargs(inference_kwargs)`、`is_multimodal_pruning_enabled()`。

> `MultiModalDummyOptionsBuiltins` TypedDict 与 `MMDummyOptions = dict[str, BaseDummyOptions]` 类型别名供外部注释。

## 为什么

- **限额双格式**：`limit_per_prompt` 支持纯计数（`{"image":16}`）与带选项（`{"video":{"count":1,"num_frames":32}}`）混合，校验器统一规范化，兼容旧 CLI。
- **编码器 TP 双模**：`mm_encoder_tp_mode="data"` 让 ViT 编码器走批内 DP（每 rank 全权重处理部分 batch），适合大 ViT；`"weights"` 是默认层内 TP。仅支持模型显式声明的编码器。
- **IPC 双模**：`direct_rpc`（msgspec 序列化，简单）vs `torch_shm`（零拷贝，大张量优，需 spawn）。`mm_ipc_gpu_memory_gb` 为前端 GPU 硬件视频解码等预留显存（从 KV cache 切出，物理保证 headroom）。
- **ViT FP8**：`mm_encoder_attn_dtype="fp8"` 经 FlashInfer cuDNN 量化 ViT 注意力；静态/动态 scale 二选一，动态可自动保存校准 scale 供后续复用，`mm_encoder_fp8_scale_save_margin` 留安全裕度防溢出。
- **processor cache 独立**：`mm_processor_cache_gb` 在 API 与 EngineCore 各一份，`shm` 模式跨进程共享避免重处理；`lru` 镜像 LRU 简单但重复处理。
- **`compute_hash` 选择性**：`mm_encoder_attn_backend`/`mm_encoder_tp_mode`/`mm_encoder_attn_dtype`/`mm_encoder_fp8_scale_path` 影响编码器图形状（V1 `compile_mm_encoder` 开启时 `VllmConfig.compute_hash` 才纳入 multimodal hash）；其余限额/IPC 字段不改变图。

## 怎么做

- **限额**：`--limit-mm-per-prompt '{"image":5,"video":{"count":1,"num_frames":32}}'`。
- **编码器 TP**：`--mm-encoder-tp-mode data`。
- **IPC**：`--mm-tensor-ipc torch_shm`（须 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，`VllmConfig` 校验）。
- **ViT FP8**：`--mm-encoder-attn-dtype fp8 --mm-encoder-fp8-scale-path /path/scales.json`；或动态 + 自动保存 `--mm-encoder-fp8-scale-save-path /path/save.json`。
- **缓存**：`--mm-processor-cache-gb 8 --mm-processor-cache-type shm --mm-shm-cache-max-object-size-mb 256`。
- **视频剪枝**：`--video-pruning-rate 0.3`。
- **前端 GPU 显存**：`--mm-ipc-gpu-memory-gb 2`（硬件视频解码预留）。

## 与其它模块/系统配合

- **多模态子系统（[`11-multimodal/`](../11-multimodal/README.md)）**：`limit_per_prompt` 驱动 `MultiModalProcessor`/`EncoderCacheManager`；`mm_processor_kwargs` 传给 `AutoProcessor`；`mm_processor_cache_*` 控制 processor cache 后端。
- **SchedulerConfig（[scheduler-config.md](scheduler-config.md)）**：`is_multimodal_model`/`encoder_cache_size`/`disable_chunked_mm_input` 影响 `EncoderCacheManager` 与调度切分。
- **ModelConfig（[model-config.md](model-config.md)）**：`multimodal_config` 内嵌于 `model_config`；`is_multimodal_model` 派生影响 `VllmConfig` 决策。
- **CompilationConfig（[compilation-config.md](compilation-config.md)）**：`compile_mm_encoder`/`cudagraph_mm_encoder`/`encoder_cudagraph_*` 控制 ViT 编译/捕获；`compute_hash` 在 `compile_mm_encoder=True` 时纳入本配置哈希。
- **AttentionConfig（[attention-config.md](attention-config.md)）**：`mm_encoder_attn_backend` 独立于主干 `attention_config.backend`。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`mm_tensor_ipc="torch_shm"` 须 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（`__post_init__` 校验）；`compile_mm_encoder` 时把 `multimodal_config.compute_hash()` 加入 `VllmConfig.compute_hash`。

## 历史版本演进

- **v0.5（多模态正式化）**：`MultiModalConfig` 初版，`limit_per_prompt`/`mm_processor_kwargs`；LLaVA 系列。
- **v0.7（v1 落地）**：`EncoderCacheManager` 抽出；`mm_processor_cache_gb`；`mm_encoder_tp_mode`。
- **v0.8**：`mm_tensor_ipc`（`direct_rpc`/`torch_shm`）；`mm_processor_cache_type`（shm/lru）；`interleave_mm_strings`。
- **v0.9**：`mm_encoder_attn_dtype`/`mm_encoder_fp8_scale_*`（ViT FP8）；`enable_mm_embeds`；`video_pruning_rate`；`skip_mm_profiling`。
- **v0.10**：`mm_ipc_gpu_memory_gb`（前端 GPU 显存预留）；`language_model_only`；`mm_encoder_only`（disagg Encoder）；XFORMERS 后端移除校验。
- **v0.11 / v0.12 / main**：`mm_shm_cache_max_object_size_mb` shm 对象上限校验；`mm_encoder_attn_backend` 字段；ViT cudagraph token budgets 配合 `encoder_cudagraph_*`。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — 内嵌于 `ModelConfig`。
- [scheduler-config.md](scheduler-config.md) — `is_multimodal_model`/`encoder_cache_size`。
- [compilation-config.md](compilation-config.md) — `compile_mm_encoder` 触发本配置进哈希。
- [attention-config.md](attention-config.md) — `mm_encoder_attn_backend`。
- [../11-multimodal/README.md](../11-multimodal/README.md) — 多模态子系统消费方。
