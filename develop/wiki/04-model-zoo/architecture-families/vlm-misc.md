# 杂系 VLM/多模态家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **VLM 杂系**

> 代表文件：`aria.py`、`bagel.py`、`bee.py`、`blip.py`/`blip2.py`、`cheers.py`、`cosmos3.py`（NVIDIA，亦见 [gpt-classic](./gpt-classic.md)）、`deepseek_vl2.py`（亦见 [deepseek](./deepseek.md)）、`deepseek_ocr.py`/`deepseek_ocr2.py`、`dots_ocr.py`、`eagle2_5_vl.py`、`ernie45_vl*.py`（见 [asian-vendor](./asian-vendor.md)）、`exaone4_5.py`、`fireredlid.py`、`fuyu.py`（见 [llava](./llava.md)）、`gemma3_mm.py`/`gemma3n_mm.py`/`gemma4_mm.py`（见 [gemma](./gemma.md)）、`glm4_1v.py`/`glm4v.py`/`glm_ocr*.py`（见 [glm](./glm.md)）、`granite4_vision.py`/`granite_speech*.py`（见 [granite](./granite.md)）、`h2ovl.py`（见 [llava](./llava.md)）、`hunyuan_vision.py`/`hyperclovax_vision*.py`、`idefics3.py`（见 [llava](./llava.md)）、`internvl.py`/`interns1*.py`/`interns2_preview.py`（见 [internlm](./internlm.md)）、`isaac.py`、`jina_vl.py`、`kanana_v.py`/`keye.py`/`keye_vl1_5.py`、`kimi_*.py`（见 [kimi](./kimi.md)）、`lfm2_vl.py`（见 [mamba-ssm](./mamba-ssm.md)）、`llama4.py`/`mllama4.py`（见 [llama](./llama.md)）、`llava*.py`/`molmo*.py`/`moondream3.py`/`smolvlm.py`（见 [llava](./llava.md)）、`mistral3.py`/`pixtral.py`（见 [mistral](./mistral.md)）、`moss_audio.py`、`nano_nemotron_vl.py`、`nemotron_vl.py`/`nemotron_parse.py`、`nvlm_d.py`、`openvla.py`/`ovis.py`/`ovis2_5.py`、`paddleocr_vl.py`/`qianfan_ocr.py`、`paligemma.py`（见 [gemma](./gemma.md)）/`phi3v.py`/`phi4mm*.py`/`phi4siglip.py`（见 [phi](./phi.md)）、`qianfan_ocr.py`、`qwen2_5_vl.py`/`qwen2_vl.py`/`qwen3_vl*.py`/`qwen3_5.py`（见 [qwen](./qwen.md)）、`radio.py`、`rvl.py`、`skyworkr1v.py`、`step3_vl.py`/`step3p7.py`/`step_vl.py`（见 [asian-vendor](./asian-vendor.md)）、`terratorch.py`、`unlimited_ocr.py`、`whisper.py`（见 [speech-audio](./speech-audio.md)）、`voxtral*.py`/`ultravox.py`（见 [speech-audio](./speech-audio.md)）。

---

## 是什么

本页是"未被前述家族页收纳"的 VLM/多模态模型聚合，按用途分组：

### 视觉对话 VLM
- **`aria.py`**：Aria（Rhymes AI）混合模态。
- **`bagel.py`**：Bytedance Bagel VLM。
- **`bee.py`**：NVIDIA Bee。
- **`blip.py`/`blip2.py`**：Salesforce BLIP/BLIP-2，经典 VLM，含 Q-Former connector。
- **`chameleon.py`/`cheers.py`**：Meta Chameleon（混合模态 token 化）/ Cheers。
- **`cosmos3.py`**：NVIDIA Cosmos3 物理世界 VLM。
- **`hunyuan_vision.py`**：腾讯 Hunyuan-VL。
- **`hyperclovax_vision.py`/`hyperclovax_vision_v2.py`**：Naver HCX-Vision。
- **`internvl.py`/`interns1*.py`/`interns2_preview.py`**：见 [internlm](./internlm.md)。
- **`mllama4.py`/`llama4.py`**：见 [llama](./llama.md)。
- **`mistral3.py`/`pixtral.py`**：见 [mistral](./mistral.md)。
- **`nvlm_d.py`**：NVIDIA NVLM-D。
- **`ovis.py`/`ovis2_5.py`**：Ovis（视觉 token 化变体）。
- **`skyworkr1v.py`**：Skywork-R1V（VLM 推理模型）。
- **`keye.py`/`keye_vl1_5.py`/`kanana_v.py`**：阶跃 Keye / Kakao Kanana-V。
- **`isaac.py`**：Isaac（.robotics ?）。
- **`jina_vl.py`**：Jina CLIP-style VLM（rerank/embed）。
- **`rvl.py`**：R-vL 检索 VLM。

### OCR / 文档理解
- **`deepseek_ocr.py`/`deepseek_ocr2.py`**：DeepSeek OCR。
- **`dots_ocr.py`**：Dots OCR。
- **`lightonocr.py`**：LightOn OCR。
- **`unlimited_ocr.py`**：Unlimited OCR。
- **`paddleocr_vl.py`**：百度 PaddleOCR-VL。
- **`qianfan_ocr.py`**：百度千帆 OCR。
- **`glm_ocr.py`/`glm_ocr_mtp.py`**：智谱 GLM-OCR + MTP。
- **`nemotron_parse.py`**：NVIDIA Nemotron-Parse（encoder-decoder，文档解析）。
- **`phi3v.py`/`phi4mm*.py`**：见 [phi](./phi.md)。

### Agent / Robotics / Geo
- **`openvla.py`**：OpenVLA（机器人 action prediction）。
- **`openpangu_vl.py`**：见 [asian-vendor](./asian-vendor.md)。
- **`terratorch.py`**：IBM Terratorch，地理空间 MAE（注册到 embedding）。
- **`radio.py`**：NVIDIA RADIO。
- **`eagle2_5_vl.py`**：Eagle2.5-VL（VLM + 可作 EAGLE target）。

### Omni/Nano
- **`nano_nemotron_vl.py`**：Nemotron-H-Nano-VL-V2（含 omni reasoning 别名）。
- **`exaone4_5.py`**：LG ExaOne 4.5（多模态 + MTP）。
- **`ernie45_vl*.py`**：见 [asian-vendor](./asian-vendor.md)。

### 编码器/视觉塔
- **`aimv2.py`**：Apple AIMv2（视觉/语音编码器）。
- **`conformer_encoder.py`**：Conformer encoder（CTC，被 ASR/audio 模型复用，见 [speech-audio](./speech-audio.md)）。
- **`deepencoder.py`/`deepencoder2.py`**：DeepEncoder 系（可能用于检索/embedding 的视觉塔，待核实）。
- **`intern_vit.py`/`interns1_vit.py`/`lfm2_siglip2.py`/`moonvit.py`**：各家自研视觉塔。

---

## 为什么

- **vLLM 多模态面扩张的最直接体现**：Owl/Bee/Bagel/Chameleon/Cosmos3/Isaac/OpenVLA 等大量杂系 VLM 的接入让 vLLM 的多模态覆盖面成为开源推理框架里最广之一。
- **OCR 子赛道**：6+ 个独立 OCR 模型说明文档理解成为 LLM 推理重要场景，vLLM 通过标准 VLM 接口收编它们。
- **视觉塔多样化**：除 CLIP/Siglip/Pixtral 之外，自研视觉塔（InternViT/AIMv2/MoonViT/Siglip2 等）成为常态，`vision.py:VisionEncoderInfo` 体系只覆盖三家主流，其他走模型内部自定义。

---

## 怎么做

绝大多数 VLM 共用如下模式：

1. 视觉塔（自家或 CLIP/Siglip/Pixtral）+ connector（projector / Q-Former / Perceiver）
2. `SupportsMultiModal` + `MultiModalRegistry.register_processor`
3. `embed_input_ids` 把视觉 embedding scatter 进 input embedding
4. LM backbone 前向（多继承自家或 Llama/Qwen）

OCR 系通常在 connector 加 OCR-specialized head（bbox/text 输出）；nemotron_parse 走 encoder-decoder 路径。

OpenVLA 的 `OpenVLAForActionPrediction` 输出 action token（机器人控制），与文本生成 LM 共用 backbone 但 head 替换。

Eagle2.5-VL 实现 `supports_eagle`/`supports_eagle3`，可作 EAGLE target。

---

## 与其它模块/系统配合

- **[多模态](../../11-multimodal/README.md)**：本家族直接消费多模态 processor + encoder budget。
- **[vision.md](../vision.md)**：visual encoder 选择/合并/DP 分片。
- **[interfaces.md](../interfaces.md)**：`SupportsMultiModal`/`SupportsEncoderCudaGraph`/`SupportsMRoPE`/`SupportsEagle*`。
- **[embedding-col](./embedding-col.md)**：`colpali.py`/`rvl.py`/`jina_vl.py` 走 VLM + late-interaction/cross-encoder。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：`Eagle3Qwen3vlForCausalLM`、`Eagle3Qwen2_5vlForCausalLM`、`eagle2_5_vl.py` 等。
- **[13-entrypoints](../../13-entrypoints/README.md)**：VLM API（`/v1/chat/completions` 含图像/视频/音频）由本家族供能。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | BLIP/BLIP-2 + Llava 系。 |
| v0.6 | Chameleon + Idefics + Fuyu + PaliGemma。 |
| v0.7 | Pixtral/Llama4 系；Ovis；NVLM-D。 |
| v0.8 | Aria/Bagel/Bee/Cosmos3/Isaac/OpenVLA 等大举接入。 |
| v0.9 | 多家 OCR（DeepSeek-OCR、PaddleOCR-VL、Qianfan-OCR、LightOn、Unlimited）。 |
| v0.10 | Nano-Nemotron-VL；Eagle2.5-VL；支持 encoder CUDA graph。 |
| v0.11 | Dots-OCR、Deepseek-OCR2、Step-VL、Pangu-VL、Keye/Kanana-V。 |
| main | 视觉塔多样化（MoonViT/Siglip2/AIMv2）；与 encoder CUDA graph 协同加深。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [多模态](../../11-multimodal/README.md) · [vision](../vision.md) · [embedding-col](./embedding-col.md) · [13-entrypoints](../../13-entrypoints/README.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md)
