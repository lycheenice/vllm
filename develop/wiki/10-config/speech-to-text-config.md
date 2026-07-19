# SpeechToTextConfig + SpeechToTextParams（speech_to_text.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > SpeechToTextConfig

源码：`vllm/config/speech_to_text.py`（约 85 行）。`SpeechToTextConfig` 描述语音转写模型的音频处理参数：采样率、单片段上限、分块重叠、能量分窗。`SpeechToTextParams` 是每请求的转写参数包（语言、hotwords、task_type 等），由 `TranscriptionRequest.build_stt_params` 构造。`SpeechToTextConfig` 当前**不是** `VllmConfig` 顶层字段（未在 `vllm.py` 聚合），而是由语音模型（如 Whisper）在模型类内部持有 `(待核实)`。

## 是什么

### `SpeechToTextConfig`（`speech_to_text.py:54`，`@config`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `sample_rate` | `16000` | 重采样目标采样率（Hz），多数语音模型需 16kHz |
| `max_audio_clip_s` | `30` | 单音频片段上限秒数；超过则按 `allow_audio_chunking` 分块或拒绝；`None`=无限不分块 |
| `overlap_chunk_second` | `1` | 分块间重叠秒数，保持跨块上下文 |
| `min_energy_split_window_size` | `1600` | 找低能量（静音）分割点的窗口样本数（≈100ms@16kHz）；`None`=不分块 |

属性：`allow_audio_chunking` = `min_energy_split_window_size is not None and max_audio_clip_s is not None`。

### `SpeechToTextParams`（`speech_to_text.py:16`，`@dataclass`，非 `@config`）

`get_generation_prompt()` 的全部入参。由 `TranscriptionRequest.build_stt_params()` 构造，把 API 字段映射为类型化属性。模型只接收此对象，故新参数可在此加而不改 `get_generation_prompt` 签名。

| 字段 | 含义 |
|---|---|
| `audio: np.ndarray` | 重采样后的单片段波形 |
| `stt_config: SpeechToTextConfig` | 服务级配置 |
| `model_config: ModelConfig` | 模型配置 |
| `language: str | None` | ISO 639-1 语言码（校验/自动检测） |
| `hotwords: str | None` | 重点词列表 |
| `task_type: str` | `"transcribe"`/`"translate"` |
| `request_prompt: str` | 引导文本 prompt |
| `to_language: str | None` | 翻译目标语言（模型相关） |

> 无 `compute_hash`——本配置不进 `VllmConfig` 顶层聚合，也不影响编译图形状。

## 为什么

- **音频分块**：语音模型（如 Whisper）有最大片段限制（30s），长音频须分块。`max_audio_clip_s` + `min_energy_split_window_size` + `overlap_chunk_second` 实现"在静音处切分、带重叠"的分块策略，最小化切断语音。
- **能量分窗**：`min_energy_split_window_size` 在窗口内找最静时刻切分，避免切在词中。`None` 关闭分块（短音频或模型无上限）。
- **`SpeechToTextParams` 参数包**：把 API 级字段与模型 `get_generation_prompt` 解耦——新 API 参数加在 `SpeechToTextParams`，模型只读此对象，签名稳定。
- **`allow_audio_chunking` 派生**：让分块逻辑单点判定（两条件同时满足），避免散落判断。

## 怎么做

- **服务级配置**：模型类内部设 `SpeechToTextConfig`（如 Whisper 设 `max_audio_clip_s=30`）。
- **请求级**：API `POST /v1/audio/transcriptions` 字段（`language`/`hotwords`/`prompt`/`response_format` 等）经 `TranscriptionRequest.build_stt_params` 构造 `SpeechToTextParams` 传模型。

## 与其它模块/系统配合

- **语音模型（`vllm/model_executor/models/whisper.py` 等）**：模型类持 `SpeechToTextConfig`，`get_generation_prompt(SpeechToTextParams)` 生成 prompt。
- **API server（[`13-entrypoints/`](../13-entrypoints/README.md)）**：`/v1/audio/transcriptions` 端点把请求映射为 `TranscriptionRequest`，`build_stt_params` 构造 params。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：Whisper 模型时 `__post_init` 检查 `VLLM_WORKER_MULTIPROC_METHOD=spawn`（forked worker 有问题）；本配置未顶层聚合。
- **ModelConfig（[model-config.md](model-config.md)）**：`SpeechToTextParams.model_config` 携带模型配置给 `get_generation_prompt`。

## 历史版本演进

- **v0.5–v0.9**：无独立 STT 配置；Whisper 等模型硬编码 30s/16kHz。
- **v0.10/v0.11（待核实）**：`SpeechToTextConfig` + `SpeechToTextParams` 抽出；`/v1/audio/transcriptions` API；`allow_audio_chunking` 派生属性；能量分窗分块策略。
- **v0.12 / main**：与 MRv2 的兼容性；hotwords/language 等请求级字段扩充。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [model-config.md](model-config.md) — `SpeechToTextParams.model_config`。
- [vllm-config.md](vllm-config.md) — Whisper 的 `VLLM_WORKER_MULTIPROC_METHOD=spawn` warning。
- [../13-entrypoints/](../13-entrypoints/README.md) — `/v1/audio/transcriptions` API 消费方。
