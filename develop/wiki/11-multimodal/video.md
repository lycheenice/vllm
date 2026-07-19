# video.py · 视频加载注册表与多后端解码器

[← Wiki 首页](../README.md) > [多模态](../README.md) > video

## 是什么

`vllm/multimodal/video.py`（约 1900 行，本子系统最大文件）实现视频侧的全部解码与帧采样逻辑。核心组件：

- `VideoLoaderRegistry`（`ExtensionManager` 子类）+ 全局 `VIDEO_LOADER_REGISTRY` 单例：按 `name` + `video_processor`（HF processor 类名）注册 backend，支持查询 `get_backend_for_video_processor` 与 `backend_requires_gpu`。
- 元数据 NamedTuple：`VideoTargetMetadata` / `VideoSourceMetadata` / `PyNvVideoCodecSourceMetadata`。
- `VideoLoader` 抽象基类：`compute_frames_index_to_sample` + `load_bytes` + `create_hf_metadata`。
- 四个 backend mixin：`OpenCVVideoBackendMixin` / `PyAVVideoBackendMixin` / `TorchCodecVideoBackendMixin` / `PyNvVideoCodecVideoBackendMixin`。
- `VideoBackend` 多继承 mixin 的"统一采样"实现；以及多个模型专属 backend：`PyNvVideoCodecVideoBackend` / `Qwen3VLVideoBackend` / `Qwen2VLVideoBackend` / `DynamicVideoBackend` / `GLM46VVideoBackend` / `GLMGAVideoBackend` / `Molmo2VideoBackend` / `NemotronVLVideoBackend` / `OpenCVDynamicOpenPanguVideoBackend` / `PyNvVideoCodecDecoderSlot`。
- helper：`_check_frame_pixel_limit` / `resize_video` / `rescale_video_size` / `sample_frames_from_video` / `get_video_loader_backend_for_processor`。

## 为什么

视频是多模态里最复杂的 ingest：

- **解码器乱**：OpenCV（`cv2.VideoCapture`，无需 FFmpeg 但跨平台兼容差）、PyAV（libav）、TorchCodec（PyTorch 官方，依赖 FFmpeg）、PyNvVideoCodec（NVDEC 硬解，NVIDIA 专有）各有优劣。需要可插拔。
- **采样策略乱**：不同模型对帧数 / fps / 帧索引算法要求不同——Qwen2-VL 按 fps+时长算帧索引并要求 metadata；Qwen3-VL 在帧间插 timestamp token；GLM-4.6V / GLM-GA 有自己的算法；Dynamic 系列根据视频长度动态决定帧数。
- **元数据需求**：HF `VideoProcessor`（如 Qwen2-VL 的）需要 `total_num_frames` / `fps` / `duration` / `frames_indices` / `do_sample_frames` 元数据，否则无法正确处理。
- **GPU 显存管控**：PyNvVideoCodec 把解码 surface 留在 GPU，需通过 `gpu_ipc_memory.py` 的 pool 准入，且 decoder 自身有持久 surface 占用（`PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES = 128 MiB`，cache size 2）。
- **安全**：解码炸弹——小压缩文件解出巨量帧。`_check_frame_pixel_limit` + `VLLM_MAX_IMAGE_PIXELS` + `max_duration` 限制。

`VideoLoaderRegistry` 的 `processor2backend` 映射让"按 HF `video_processor` 类名自动选 backend"成为可能，用户无需手动指定；`backend_requires_gpu` 让 `VideoMediaIO.merge_kwargs` 能在请求级 strip 未配置的 GPU backend（防止用户在请求里偷偷开 GPU 解码耗显存）。

## 怎么做

### VideoLoaderRegistry

`VideoLoaderRegistry.register(name, *, video_processor=None, requires_gpu=False)`（`:68`）：

- `_normalize_registered_video_processors` 把 str / tuple 规范成 tuple。
- 装饰器把 `cls` 写入 `name2class[name]`、`_requires_gpu[name]`、对每个 `processor_name` 写 `processor2backend[processor_name] = name`。

`get_backend_for_video_processor(video_processor)`（`:86`）：查 `processor2backend`，None 输入返 None。
`backend_requires_gpu(name)`（`:95`）：查 `_requires_gpu`（默认 False）。

### 元数据 NamedTuple

- `VideoTargetMetadata(num_frames, fps, max_duration)`：模型目标。
- `VideoSourceMetadata(total_frames_num, original_fps, duration)`：源视频实测。
- `PyNvVideoCodecSourceMetadata(source, width, height)`：硬解前需要的源信息。

### VideoLoader 抽象

`VideoLoader`（`:172`）：

- `compute_frames_index_to_sample(source, target, **kwargs) -> list[int]`：abstract，决定采哪些帧索引。
- `load_bytes(data, **kwargs) -> (ndarray, metadata_dict)`：abstract，主入口。
- `create_hf_metadata(source, valid_frame_indices, video_backend)`（`:193`）：构造 HF processor 期望的 metadata dict，含 `do_sample_frames = (len(indices) == total_frames_num)` 标志（让 processor 知道帧已被采过还是需要自己再采）。

### Backend Mixin

四个 mixin 各自实现"用特定 codec 读取指定索引的帧"：

- `OpenCVVideoBackendMixin`（`:264`）：`cv2.VideoCapture`，跨平台无 FFmpeg 依赖；帧索引 seek。
- `PyAVVideoBackendMixin`（`:508`）：`av.open(BytesIO(data))` + `container.decode(video=0)`，支持精确 seek 与 frame recovery。
- `TorchCodecVideoBackendMixin`（`:578`）：`torchcodec.decoders.VideoDecoder`，PyTorch 原生 API，输出直接是 tensor。
- `PyNvVideoCodecVideoBackendMixin`（`:625`）：NVDEC 硬解，输出 GPU tensor；最复杂，含 `PyNvVideoCodecDecoderSlot` 管理 decoder surface 复用（`PYNVVIDEOCODEC_DECODER_CACHE_SIZE=2`）。

### VideoBackend（统一采样）

`VideoBackend(VideoLoader, OpenCVMixin, PyAVMixin, TorchCodecMixin, PyNvVideoCodecMixin)`（`:830`）：默认 backend，按"uniform sampling"算法采样——`num_frames_to_sample = min(num_frames, total)` 再 `min(.., floor(duration*fps))`，用 `np.linspace` 取索引。`load_bytes` 支持 `backend="opencv|pyav|torchcodec|pynvvideocodec"` kwarg 切换底层 codec。

### 模型专属 backend

| backend | 算法/特点 | GPU |
|---|---|---|
| `Qwen2VLVideoBackend`（`:1093`） | Qwen2-VL：按 fps+时长算索引，需 metadata | 否 |
| `Qwen3VLVideoBackend`（`:1044`） | Qwen3-VL：帧间插 timestamp，5 通道 mrope | 否 |
| `DynamicVideoBackend`（`:1172`） | 动态帧数（视频长则多采） | 否 |
| `GLM46VVideoBackend`（`:1264`） | GLM-4.6V 采样算法 | 否 |
| `GLMGAVideoBackend`（`:1389`） | GLM-GA 采样算法 | 否 |
| `Molmo2VideoBackend`（`:1492`） | Molmo2，继承 `VideoLoader + OpenCVMixin` | 否 |
| `NemotronVLVideoBackend`（`:1782`） | Nemotron-VL | 否 |
| `OpenCVDynamicOpenPanguVideoBackend`（`:1812`） | OpenPangu 动态 + OpenCV | 否 |
| `PyNvVideoCodecVideoBackend`（`:1007`） | 硬解专用（非 mixin 复用） | 是 |
| `PyNvVideoCodecDecoderSlot`（`:222`） | decoder surface 复用管理 | 是 |

### helper 函数

- `_check_frame_pixel_limit(w, h)`（`:105`）：`w*h > VLLM_MAX_IMAGE_PIXELS` raise，防大帧。
- `resize_video(frames, size)`（`:117`）：逐帧 `cv2.resize`，保持 dtype 与 channel。
- `rescale_video_size(frames, factor)`（`:130`）：按 factor 缩放。
- `sample_frames_from_video(frames, num_frames)`（`:138`）：`num_frames==-1` 全留；否则 `np.linspace(0, total-1, num_frames, dtype=int)` 索引。
- `get_video_loader_backend_for_processor(video_processor)`（`:99`）：暴露给 `MediaConnector` 用。

### 与 MediaIO 的衔接

`media/video.py` 的 `VideoMediaIO.__init__` 从 kwargs 取 `video_backend`（默认 `VLLM_VIDEO_LOADER_BACKEND`），用 `VIDEO_LOADER_REGISTRY.load(backend)` 实例化 loader；`load_bytes` 调 loader 的 `load_bytes(data, num_frames=..., **kwargs)`。`merge_kwargs` 在请求级 strip 掉未配置的 GPU backend（见 [media.md](media.md)）。

## 与其它模块/系统配合

- **media/video.py**：`VideoMediaIO` 是 IO 入口；`VIDEO_LOADER_REGISTRY` 在 `media/__init__.py` 导出供 connector 使用。
- **media/connector.py**：`MediaConnector.fetch_video` 通过 `video_processor` 参数自动选 backend（`get_video_loader_backend_for_processor`），优先级低于显式 `video_backend` kwarg。
- **parse.py**：`_parse_video_data` 接收 `(ndarray, metadata)` 元组，构造 `VideoProcessorItems(metadata=metadata_lst)`；`video_needs_metadata` 决定是否强制要求 metadata。
- **gpu_ipc_memory.py**：`PyNvVideoCodecVideoBackend` 与 `PyNvVideoCodecDecoderSlot` 是 GPU pool 的主要消费方（`(待核实)` acquire 调用点应在 `PyNvVideoCodecVideoBackendMixin` 内）。
- **processing/dummy_inputs.py**：`_get_dummy_videos` 造 `np.full((num_frames, w, h, 3), 255, uint8)`，绕过本文件。
- **evs.py**：EVS 剪枝发生在视频 embedding 之后，与采样阶段正交，但 `frames_indices` 元数据影响后续 token 计数。
- **环境变量**：`VLLM_VIDEO_LOADER_BACKEND`（默认 backend）、`VLLM_MAX_IMAGE_PIXELS`（帧像素上限）。
- **配置**：`--media-io-kwargs '{"video": {"num_frames":40, "video_backend":"torchcodec"}}'`；`mm_ipc_gpu_memory_gb` 控制 PyNvVideoCodec 是否可用。

## 历史版本演进

- **v0.6**：`video.py` 加入，仅 OpenCV + PyAV 两个 mixin，无注册表，硬编码 backend 选择。
- **v0.7（v1 化）**：`VideoLoaderRegistry` + `VIDEO_LOADER_REGISTRY` 出现，按 `video_processor` 类名自动选 backend；`Qwen2VLVideoBackend` 等模型专属采样器加入。
- **v0.8**：`TorchCodecVideoBackendMixin` 加入（torchcodec 进入 PyTorch 主线）；`metadata` 透传机制稳定，`create_hf_metadata` 标准化。
- **v0.9（hash+cache）**：`_check_frame_pixel_limit` 加入，防大帧 OOM；`VideoMediaIO.merge_kwargs` strip 请求级 GPU backend。
- **v0.10**：`PyNvVideoCodecVideoBackendMixin` + `PyNvVideoCodecDecoderSlot` 加入，NVDEC 硬解上线，配合 `gpu_ipc_memory.py`；`PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES` / `CACHE_SIZE` 常量固定。
- **v0.11（EVS）**：`Qwen3VLVideoBackend` 加入，5 通道 mrope 与 timestamp token 配套；`DynamicVideoBackend` 系列扩展（GLM-4.6V / GLM-GA / Molmo2 / Nemotron-VL / OpenPangu）。
- **main**：`VideoBackend.load_bytes` 默认 `backend="opencv"`，全部 backend 由 `Literal` 枚举校验；`frame_recovery` 参数处理损坏帧；`num_ffmpeg_threads` 参数调优多线程解码。

[← 返回多模态首页](../README.md)

## 参见

- [media.md](media.md)：`VideoMediaIO` 与 `MediaConnector` 的调用层。
- [gpu-ipc-memory.md](gpu-ipc-memory.md)：GPU 解码的显存准入。
- [parse.md](parse.md)：`VideoProcessorItems` 接收 `(ndarray, metadata)`。
- [evs.md](evs.md)：视频 embedding 剪枝，与采样正交。
- [04-model-zoo/architecture-families/llava.md](../04-model-zoo/architecture-families/llava.md)：使用本文件的 VLM 家族。
