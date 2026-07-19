# media/ · 媒体 I/O 抽象与连接器

[← Wiki 首页](../README.md) > [多模态](../README.md) > media

## 是什么

`vllm/multimodal/media/` 子目录定义"原始字节 → 内存对象"的统一 I/O 层：

- `base.py`：`MediaIO` 抽象基类（`load_bytes` / `load_base64` / `load_file` + `merge_kwargs`）、`MediaWithBytes` 包装类。
- `image.py`：`ImageMediaIO`（PIL.Image）、`ImageEmbeddingMediaIO`（torch.Tensor）。
- `audio.py`：`AudioMediaIO`（`(ndarray, sr)`）、`AudioEmbeddingMediaIO`；`load_audio_pyav` / `load_audio_soundfile` / `load_audio` 三个模块级函数。
- `video.py`：`VideoMediaIO`（`(ndarray, metadata)`），委托给 `video.py` 的 `VIDEO_LOADER_REGISTRY`。
- `connector.py`：`MediaConnector`（HTTP/data/file URL → media 对象）、`MEDIA_CONNECTOR_REGISTRY` 扩展点、模块级 `global_thread_pool`、`merge_media_io_kwargs`、`_wrap_media_fetch_error`。
- `__init__.py` 统一导出。

它处在 [media.md 流水的最前段]——用户提交的 `image_url` 等在此被解码为 PIL/ndarray 后，才进 `parse.py` 与 processor。

## 为什么

多模态输入到达 vLLM 时形态极乱：HTTP URL、`data:image/png;base64,...`、`file:///abs/path`、bytes。每种媒体又可能有多种格式与目标类型（image vs image_embeds）。若让 API server 直接调 PIL/soundfile/cv2，会出现：

- **安全问题**：`file://` 可读任意本地文件；HTTP URL 可指向内网；需统一 `allowed_local_media_path` / `allowed_media_domains` 校验。
- **错误归一**：4xx 应转 `VLLMUnprocessableEntityError`（422），5xx/408/429 应保留重试，需统一 `_wrap_media_fetch_error`。
- **缓存复用**：相同 URL 被多个请求重复下载浪费带宽，需 LRU + TTL 的媒体字节缓存（`VLLM_MEDIA_CACHE`）。
- **kwargs 合并**：`--media-io-kwargs`（服务级）与每请求 `media_io_kwargs` 要 modality-specific 合并（如 video 的 fps/num_frames 互斥），需 per-IO `merge_kwargs`。
- **异步下载**：API server 是 async，但解码是同步 CPU 密集，需 `ThreadPoolExecutor` 隔离。
- **防解码炸弹**：图片/音频/视频都可能被构造为压缩比极大的炸弹，需 `VLLM_MAX_IMAGE_PIXELS` / `VLLM_MAX_AUDIO_DECODE_DURATION_S` 限制。
- **tensor embedding 安全**：`torch.load` 反序列化恶意 tensor 可导致 OOB 写，需 `torch.sparse.check_sparse_tensor_invariants()` 校验。

`MediaIO` 抽象把这些差异封装；`MediaConnector` 把 URL/缓存/线程池/错误归一串成一条线。

## 怎么做

### MediaIO（base.py）

`MediaIO(ABC, Generic[_T])`（`base.py:46`）：

- `merge_kwargs(defaults, overrides)`（classmethod）：默认 shallow merge，runtime 覆盖 defaults；子类可 override（如 `VideoMediaIO` 跨字段处理）。
- `load_bytes(data: bytes) -> _T`：abstract。
- `load_base64(media_type, data: str) -> _T`：abstract。
- `load_file(filepath: Path) -> _T`：abstract。

### MediaWithBytes

`MediaWithBytes`（`base.py:14`，`Generic[_T]`，`@dataclass`）：`media: _T` + `original_bytes: bytes`。

- `__array__` 委托给 media，让 `np.array(wrapper)` 等价 `np.array(media)`。
- `__getattr__` 委托给 media，让 wrapper 像 PIL 一样用（`wrapper.size` 等）。
- `__getstate__` / `__setstate__` 支持 pickle 跨进程。
- **仅 image modality 使用**（docstring 明确），让 hasher 走原始字节快速路径。

### ImageMediaIO / ImageEmbeddingMediaIO（image.py）

`ImageMediaIO(image_mode="RGB", **kwargs)`（`image.py:21`）：

- 持 `image_mode`、自定义 `kwargs`、`rgba_background_color`（默认 `(255,255,255)`，list 转 tuple，校验 3 个 0-255 int）。
- `_convert_image_mode`：`MediaWithBytes` 解包；RGBA→RGB 走 `rgba_to_rgb`；其它走 `convert_image_mode`。
- `load_bytes`（`:73`）：`Image.open(BytesIO)`；查 `VLLM_MAX_IMAGE_PIXELS`；`normalize_image` + `load()` + `_convert_image_mode`；包成 `MediaWithBytes(image, data)`（`data` 即原始 bytes）。
- `load_base64`：`pybase64.b64decode(validate=True)` → `load_bytes`。
- `load_file`：`filepath.read_bytes()` → `load_bytes`。
- `encode_base64(media, *, image_format="PNG")`：转 mode 后 save 到 BytesIO，pybase64 编码。

`ImageEmbeddingMediaIO`（`:113`）：`load_bytes` 区分 numpy（`MAGIC_NUMPY_PREFIX`）与 pickle torch；`torch.load(weights_only=True)` + `torch.sparse.check_sparse_tensor_invariants()` + `to_dense()`，防恶意 sparse 张量 OOB。`load_file` 按 `.npy` 后缀分支。

### AudioMediaIO / AudioEmbeddingMediaIO（audio.py）

模块级 `load_audio_pyav(path, *, sr=22050, mono=True, max_duration_s=None)`（`audio.py:46`）：FFmpeg 解码 + 重采样 + mono 合并 + 时长上限校验（先查 metadata，再逐帧累计 samples）。

`load_audio_soundfile(path, *, sr=22050, mono=True, max_duration_s=None)`（`:156`）：libsndfile 路径，`f.read(dtype="float32", always_2d=False).T`，必要时 `resample_audio_pyav`。

`load_audio(path, *, sr, mono, max_duration_s)`（`:187`）：先 soundfile，`ImportError`（无 soundfile）或 `LibsndfileError.code in _BAD_SF_CODES={0,1,3,4}`（格式识别失败）时回退 PyAV；其它 LibsndfileError（corrupt 但识别）re-raise。BytesIO seek 重置后传给 PyAV。

`AudioMediaIO(**kwargs)`（`:221`）：`load_bytes` → `load_audio(BytesIO(data), sr=None, max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S)`。`encode_base64` 用 soundfile 写 WAV。

`AudioEmbeddingMediaIO`（`:274`）：同 image 版本，sparse invariant 校验。

### VideoMediaIO（video.py）

`VideoMediaIO(image_io, num_frames=32, **kwargs)`（`video.py:22`）：

- `merge_kwargs`（classmethod）：runtime kwargs 中 `video_backend`/`backend` 若指向 GPU backend 且 defaults 未配置 → strip + warning；runtime 设了 `num_frames` 没 `fps` → wipe defaults 的 `fps`（反之亦然），防跨字段互斥。
- `__init__`：`video_loader_backend = kwargs.pop("video_backend") or envs.VLLM_VIDEO_LOADER_BACKEND`；`VIDEO_LOADER_REGISTRY.load(backend)` 实例化。
- `load_bytes(data)` → `video_loader.load_bytes(data, num_frames=num_frames, **kwargs)`。
- `load_base64(media_type, data)`：`video/jpeg` 走 JPEG sequence（按 `,` 分帧逐个 base64 decode，校验 `frames_indices`/`total_num_frames`/`duration` 一致性，构造 metadata dict 含 `video_backend="jpeg_sequence"`）；其它走 `load_bytes(pybase64.b64decode(data))`。
- `encode_base64(media, *, video_format="JPEG")`：每帧 `Image.fromarray` + `ImageMediaIO.encode_base64`，逗号拼接。

### MediaConnector（connector.py）

`MEDIA_CONNECTOR_REGISTRY = ExtensionManager()`（`:45`）+ `MODALITY_IO_MAP: dict[str, type[MediaIO]]`（`:47`）映射 `"audio"|"image"|"video"` 到 IO 类，供 `merge_media_io_kwargs` 找对应 `merge_kwargs`。

`global_thread_pool = ThreadPoolExecutor(max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT)`（`:40`），`atexit.register(global_thread_pool.shutdown)`。

`MediaConnector.__init__(media_io_kwargs, connection=global_http_connection, *, allowed_local_media_path="", allowed_media_domains=None)`（`:143`）：

- 校验 `allowed_local_media_path` 存在且是目录（否则 raise）。
- 解析 `allowed_media_domains`。
- 可选启用 `VLLM_MEDIA_CACHE`（路径存在 + 可写 → 设 `_media_cache_dir`/`_media_cache_max_bytes`/`_media_cache_ttl_secs`，否则 warning 禁用）。

`load_from_url(url, media_io, *, fetch_timeout)`（`:348`）：

1. `data:` URL → `_load_data_url`（base64 → `media_io.load_base64`）。
2. HTTP/HTTPS：`_assert_url_in_allowed_media_domains` → `_get_cached_bytes` 命中即 `media_io.load_bytes(cached)`；否则 `connection.get_bytes(timeout, allow_redirects=VLLM_MEDIA_URL_ALLOW_REDIRECTS)`，异常经 `_wrap_media_fetch_error`（4xx→VLLMUnprocessableEntityError，5xx/408/429 原样），`_put_cached_bytes` 写缓存 + LRU evict，返 `media_io.load_bytes(data)`。
3. `file://`：`_load_file_url`，校验 path 在 `allowed_local_media_path` 子树内。
4. 其它 scheme raise。

`load_from_url_async`（`:389`）：把上述同步步骤通过 `loop.run_in_executor(global_thread_pool, ...)` 调度，HTTP 用 `connection.async_get_bytes`。

`fetch_audio/fetch_image/fetch_video`（sync/async 各一对）：构造对应 `MediaIO`（带 `media_io_kwargs[modality]`），`fetch_video` 还从 `video_processor` 参数反查 backend 注入 kwargs。`fetch_image` 把 `UnidentifiedImageError` 转 `ValueError` 便于上游处理。

`fetch_image_embedding` / `fetch_audio_embedding`（`:579`/`:590`）：直接 `ImageEmbeddingMediaIO` / `AudioEmbeddingMediaIO().load_base64`。

### 错误归一

`_wrap_media_fetch_error(url, exc)`（`:54`）把 `aiohttp.ClientResponseError` / `requests.HTTPError` / `InvalidURL` / `ValueError` 转 `VLLMUnprocessableEntityError`（422，parameter="image_url"）；408/429/5xx/DNS 失败等 transient 原样返回。

### 缓存

`_get_cached_bytes`（`:218`）/`_put_cached_bytes`（`:238`）/`_maybe_evict`（`:260`）/`_media_cache_path`（`:294`）：

- 路径 = `<cache_dir>/<sha256(url)[:20]><ext>`。
- TTL：`VLLM_MEDIA_CACHE_TTL_HOURS` 小时过期，过期 unlink。
- LRU：超 `VLLM_MEDIA_CACHE_MAX_SIZE_MB` 时按 mtime 升序驱逐（exclude 刚写的）。
- 原子写：temp file + rename。

`merge_media_io_kwargs(defaults, overrides)`（`:113`）：per-modality 调对应 `MediaIO.merge_kwargs`，让 video 的 fps/num_frames 互斥逻辑生效。

## 与其它模块/系统配合

- **inputs.py**：`MediaWithBytes` 让 hasher 走原始字节（详见 [inputs.md](inputs.md) / [hasher.md](hasher.md)）。
- **parse.py**：`ProcessorBatchItems._unwrap` 把 `MediaWithBytes` 解包为底层 media；`get_item_for_hash` 保留 wrapper。
- **image.py / audio.py**：被 `ImageMediaIO` / `AudioMediaIO` 复用做归一与重采样。
- **video.py**：`VideoMediaIO` 委托给 `VIDEO_LOADER_REGISTRY`，并 strip 请求级 GPU backend。
- **utils.py**：`encode_*` / `fetch_*` 系列是本目录功能的对外门面。
- **13-entrypoints / renderers**：API server 的 `Renderer` 持有受控 `MediaConnector`；`merge_media_io_kwargs` 在每请求合并服务级与请求级 kwargs。
- **vllm.connections / global_http_connection**：HTTP 下载与重试由 `HTTPConnection` 提供。
- **vllm.exceptions**：`VLLMUnprocessableEntityError` 由 `_wrap_media_fetch_error` 抛出。
- **环境变量**：`VLLM_MAX_IMAGE_PIXELS`、`VLLM_MAX_AUDIO_DECODE_DURATION_S`、`VLLM_AUDIO/IMAGE/VIDEO_FETCH_TIMEOUT`、`VLLM_MEDIA_URL_ALLOW_REDIRECTS`、`VLLM_MEDIA_LOADING_THREAD_COUNT`、`VLLM_MEDIA_CACHE` / `VLLM_MEDIA_CACHE_MAX_SIZE_MB` / `VLLM_MEDIA_CACHE_TTL_HOURS`、`VLLM_VIDEO_LOADER_BACKEND`。
- **配置**：`--media-io-kwargs`、`--allowed-local-media-path`、`--allowed-media-domains`。

## 历史版本演进

- **v0.5（LLaVA 初版）**：`MediaIO` 抽象 + `ImageMediaIO`/`AudioMediaIO` 基础；`MediaConnector` 仅支持 HTTP 与 data URL。
- **v0.6**：`VideoMediaIO` 加入；`fetch_video` 上线；`PyAV` audio 解码路径。
- **v0.7（v1 化）**：`MediaWithBytes` 引入支撑 hash stability；`ImageEmbeddingMediaIO`/`AudioEmbeddingMediaIO` 旁路加入；`allowed_media_domains` 安全控制。
- **v0.8**：`VLLM_MAX_IMAGE_PIXELS` / `VLLM_MAX_AUDIO_DECODE_DURATION_S` 解码炸弹防护；`torch.sparse.check_sparse_tensor_invariants` 校验 embedding tensor 防 OOB 攻击；`global_thread_pool` async 解耦。
- **v0.9（hash+cache）**：`VLLM_MEDIA_CACHE` LRU+TTL 媒体字节缓存加入；`_wrap_media_fetch_error` 区分 4xx/5xx。
- **v0.10**：`merge_media_io_kwargs` + per-modality `merge_kwargs` 体系成型，配合 `--media-io-kwargs`；`VideoMediaIO.merge_kwargs` strip 请求级 GPU backend；`rgba_background_color` 可配。
- **v0.11（EVS）**：`video/jpeg` JPEG sequence 路径稳定（多帧 base64 拆分）；`frames_indices`/`total_num_frames`/`duration` 一致性校验。
- **main**：`VLLM_MEDIA_LOADING_THREAD_COUNT` 线程池可配；`VLLM_MEDIA_URL_ALLOW_REDIRECTS` 重定向可控；`soundfile.LibsndfileError.code` 精细化回退 PyAV（`_BAD_SF_CODES`）；`memoryview` 零拷贝优化在 hasher 路径。

[← 返回多模态首页](../README.md)

## 参见

- [image.md](image.md) / [audio.md](audio.md) / [video.md](video.md)：被 `MediaIO` 复用的底层函数与 backend。
- [hasher.md](hasher.md)：`MediaWithBytes.original_bytes` 的消费方。
- [parse.md](parse.md)：`MediaWithBytes` 解包时机。
- [utils.md](utils.md)：`encode_*` / `fetch_*` 对外门面。
- [13-entrypoints](../13-entrypoints/README.md)：API server 注入受控 `MediaConnector` 的位置。
