[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [Speech-to-text](README.md) > base

# speech_to_text/base/（SpeechToTextBaseServing）

> `SpeechToTextBaseServing` 是转录/翻译 handler 的基类，继承 `GenerateBaseServing`，封装音频文件→多模态输入→`generate`→文本输出的共性，含流式 chunk 分隔处理。

## 是什么

| 成员 | 位置 | 职责 |
|---|---|---|
| `asr_inter_chunk_separator` | `vllm/entrypoints/speech_to_text/base/serving.py:79` | 流式 chunk 间分隔符处理 |
| `SpeechToTextBaseServing` | `:90` | ASR 基类，`GenerateBaseServing` 子类 |
| `base/protocol.py`/`base/utils.py` | — | ASR 协议与工具 |

`SpeechToTextBaseServing` 持音频处理参数、response_format（text/json/verbose_json/srt/vtt）、`asr_inter_chunk_separator` 决定流式 chunk 间拼接方式（空格/换行/无）。

## 为什么

- **生成基建复用**：ASR 模型本质是"音频→token"生成，复用 `GenerateBaseServing` 的 engine 调用/timing/logprobs。
- **chunk 分隔**：流式 ASR 每个输出 chunk 是一段文本，分隔符影响可读性（srt/vtt 格式需换行，纯文本需空格）。
- **多 response_format**：text/json/verbose_json/srt/vtt 同一推理结果不同序列化，基类统一组装。

## 怎么做

见 [README](README.md)。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| asr_inter_chunk_separator | `vllm/entrypoints/speech_to_text/base/serving.py:79` |
| SpeechToTextBaseServing | `vllm/entrypoints/speech_to_text/base/serving.py:90` |

## 与其它模块/系统配合

- [generate/base-serves.md](../generate/base-serves.md)：父类。
- [transcription.md](transcription.md)/[translation.md](translation.md)：子类。

## 历史版本演进

- **v0.10（引入）**：`SpeechToTextBaseServing` + `asr_inter_chunk_separator`。
- **main**：srt/vtt 格式（待核实）。

## 参见

- [← 返回 Speech-to-text 首页](README.md)
- [generate/base-serves.md](../generate/base-serves.md)
