# audio.py · 音频规格 / 重采样 / 声道归一化 / 切分

[← Wiki 首页](../README.md) > [多模态](../README.md) > audio

## 是什么

`vllm/multimodal/audio.py` 提供音频侧的纯函数工具集：`get_audio_duration`、`ChannelReduction` 枚举、`AudioSpec` 数据类、`normalize_audio`、三种重采样后端（`resample_audio_pyav` / `resample_audio_scipy` / `resample_audio_soxr`）、`AudioResampler` 封装、`split_audio` / `find_split_point`（长音频智能切分）。这些函数既被 `parse.py`（解析期重采样 + 声道归一）调用，也被 `media/audio.py`（IO 层 `load_audio_pyav` / `load_audio_soundfile`）复用。

## 为什么

音频输入在 vLLM 上的乱度仅次于视频：

- **采样率乱**：用户可能传 8kHz/16kHz/22.05kHz/44.1kHz/48kHz；模型期望固定采样率（如 Whisper 16kHz）。必须重采样。
- **声道乱**：立体声、单声道、多声道混存；多数模型只要 mono。
- **格式乱**：`(samples,)` 1D、`(channels, samples)` 2D（torchaudio 风格）、`(samples, channels)` 2D（soundfile 风格）。
- **解码炸弹**：小压缩文件解出巨大 PCM，撑爆内存。

重采样需要后端可切换：PyAV（libswresample，FFmpeg 内置，默认）、scipy（`scipy.signal.resample_poly`，纯 Python 生态）、soxr（高质量，需 `pip install soxr`）。`AudioResampler` 抽象让 `MultiModalDataParser` 不关心后端，由 `--audio-resample-method` 或模型 config 决定。

长音频（>30s）一次性喂 Whisper 等会超上下文，`split_audio` 在低能量处切分以最小化语音中断。

## 怎么做

### get_audio_duration

`get_audio_duration(*, y, sr=22050)`（`:32`）：`n_samples = y.shape[-1]; return n_samples / sr`，支持 1D/2D（按最后一维算样本数），与 `librosa.get_duration` 对齐。

### AudioSpec 与 ChannelReduction

`ChannelReduction`（`:46`）：`MEAN` / `FIRST` / `MAX` / `SUM` 四种降声道策略。

`AudioSpec`（`:55`，dataclass）：

- `target_channels: int | None = 1`（None 表示 passthrough 不归一）。
- `channel_reduction: ChannelReduction = MEAN`。
- `needs_normalization` property：`target_channels is not None`。
- `__repr__` 区分 passthrough 与归一化。

预定义实例：`MONO_AUDIO_SPEC`（mono+MEAN）、`PASSTHROUGH_AUDIO_SPEC`。

### normalize_audio

`normalize_audio(audio, spec)`（`:91`）：

1. `spec.needs_normalization` 为 False 直接返回。
2. 1D audio：`target_channels==1` 直接返回；否则 raise（不可 mono→stereo 扩展）。
3. 2D audio：若 `shape[0] > shape[1]` 视作 `(time, channels)`（soundfile 风格）转置。
4. `num_channels == target_channels` 直接返回；`num_channels < target_channels` raise（不可扩展）。
5. `target_channels == 1`：按 `channel_reduction` 取 `MEAN` / `FIRST` / `MAX` / `SUM`。
6. 多通道目标：`audio[:target_channels]`（取前 N，不做混合）。

支持 numpy 与 torch（用 `isinstance` 分支选 `np.mean` 或 `audio.mean(dim=0)`）。

### 重采样后端

`resample_audio_pyav(audio, *, orig_sr, target_sr)`（`:174`）：

- 相等直接返回。
- 2D 分通道递归后 `np.stack`。
- 1D：算 `expected_len`，padding 到 `_MIN_SAMPLES=1024`（libswresample 最小输入要求），构造 `av.AudioResampler(format="fltp", layout="mono", rate=target_sr_int)`，`AudioFrame.from_ndarray` + 设 `sample_rate=orig_sr`，`resampler.resample(frame)` + flush（`resample(None)`），concat 后 squeeze 并 trim 到 `expected_len`。

`resample_audio_scipy`（`:232`）：`scipy.signal.resample_poly(audio, up=target/gcd, down=orig/gcd, axis=-1)`。

`resample_audio_soxr`（`:253`）：2D 分通道递归；1D `soxr.resample(audio, orig_sr_int, target_sr_int)`。

`AudioResampler`（`:277`，class）：

- `__init__(target_sr=None, method="pyav")`。
- `resample(audio, *, orig_sr)`：`target_sr is None` raise；`math.isclose` 直接返回；按 method dispatch。
- 默认 PyAV 因 FFmpeg 普遍可用；soxr 质量最高但需额外装；scipy 后备。

### 长音频切分

`split_audio(audio_data, sample_rate, max_clip_duration_s, overlap_duration_s, min_energy_window_size)`（`:325`）：

- `chunk_size = int(sr * max_clip_duration_s)`，`overlap_size = int(sr * overlap_duration_s)`。
- 主循环：到末尾直接收尾；否则在 `[i+chunk_size-overlap_size, i+chunk_size)` 重叠区找 `find_split_point`，选最低能量点切。
- `split_point <= i` 时回退硬边界，保证前进不卡死。
- 沿最后一维切片，保留前面所有维度。

`find_split_point(wav, start_idx, end_idx, min_energy_window)`（`:393`）：在 `[start, end)` 用滑窗算 RMS 能量，取最低能量窗起点作切点（最安静处切入，减少语音中断）。

## 与其它模块/系统配合

- **parse.py**：`MultiModalDataParser.__init__` 持 `AudioResampler(target_sr, method)`，`_parse_audio_data` 对每个带 `orig_sr` 的 item 调 `resample`，再用 `AudioSpec(target_channels)` 调 `normalize_audio`。
- **media/audio.py**：`load_audio_pyav` / `load_audio_soundfile` 也使用 `resample_audio_pyav`（soundfile 路径降采样到目标 sr）；IO 层还用 `VLLM_MAX_AUDIO_DECODE_DURATION_S` 防解码炸弹。
- **processing/context.py**：`BaseProcessingInfo.get_data_parser` 子类化时透传 `target_sr` / `target_channels` / `audio_resample_method`，源自模型 HF config。
- **video.py**：`media/audio.py` 的 `load_audio_pyav` 也用于视频抽取音轨（`extract_audio_from_video` 场景，`(待核实)`）。
- **环境变量**：`VLLM_MAX_AUDIO_DECODE_DURATION_S`（解码时长上限）、`av` / `scipy` / `soxr` 任一缺失则该后端降级为 `PlaceholderModule`。
- **配置**：`target_sr` / `target_channels` / `audio_resample_method` 来自模型实现（`BaseProcessingInfo.get_data_parser`），`--media-io-kwargs` 不直接覆盖这些（`(待核实)` 是否有 CLI 暴露）。

## 历史版本演进

- **v0.6**：`audio.py` 随 audio 模态支持加入；仅 `PyAV` 后端；`AudioResampler` 类。
- **v0.7（v1 化）**：`AudioSpec` / `normalize_audio` / `ChannelReduction` 加入，统一处理立体声→mono；`target_channels` 配置化。
- **v0.8**：`resample_audio_scipy` / `resample_audio_soxr` 加入，`audio_resample_method` 可选；解决 FIPS / 无 FFmpeg 环境。
- **v0.9（hash+cache）**：`_MIN_SAMPLES=1024` padding 解决短音频 PyAV 无输出问题。
- **v0.10**：`split_audio` / `find_split_point` 加入，服务长音频（>30s）模型，如 Whisper 长转录。
- **main**：`normalize_audio` 同时支持 numpy 与 torch；格式自动检测 `(time, channels)` vs `(channels, time)`；`get_audio_duration` 与 `librosa.get_duration` 行为对齐。

[← 返回多模态首页](../README.md)

## 参见

- [parse.md](parse.md)：解析期调用本文件做重采样与归一。
- [media.md](media.md)：IO 层 `load_audio_*` 复用 `resample_audio_pyav`。
- [10-config/multimodal-config.md](../10-config/multimodal-config.md)：`target_sr` 等通过模型 config 透传。
