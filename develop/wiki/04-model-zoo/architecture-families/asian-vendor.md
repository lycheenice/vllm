# 亚洲厂商模型家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **亚洲厂商**

> 代表文件：`minimax_m2.py`、`bailing_moe.py`/`bailing_moe_linear.py`/`bailing_moe_mtp.py`、`hy_v3.py`/`hy_v3_mtp.py`、`hunyuan_v1.py`/`hunyuan_vision.py`、`mimo.py`/`mimo_audio.py`/`mimo_mtp.py`/`mimo_v2*.py`、`longcat_flash.py`/`longcat_flash_mtp.py`、`ernie45.py`/`ernie45_moe.py`/`ernie45_vl*.py`/`ernie_mtp.py`、`openpangu.py`/`openpangu_mtp.py`/`openpangu_vl.py`、`telechat2.py`/`teleflm.py`、`exaone.py`/`exaone4*`/`exaone_moe*`/`exaone4_5_mtp.py`/`exaone_moe_mtp.py`、`plamo2.py`/`plamo3.py`、`step1.py`/`step3_*`/`step3p5_mtp.py`、`skyworkr1v.py`、`keye.py`/`keye_vl1_5.py`、`kanana_v.py`、`AXK1.py`、`rnj1.py`、`jais2.py`、`lightonocr.py`、`unlimited_ocr.py`、`deepseek_ocr.py`/`deepseek_ocr2.py`、`dots_ocr.py`、`paddleocr_vl.py`、`qianfan_ocr.py`、`midashenglm.py`。
> 厂商隔离包：`vllm/models/minimax_m3/`（见 [`vendor-split-models.md`](../vendor-split-models.md)）。

---

## 是什么

亚洲厂商系是 vLLM 模型库扩张最快的板块，多数带 MTP/nextn 投机机制：

- **MiniMax**：`minimax_m2.py MiniMaxM2ForCausalLM(SupportsLoRA, SupportsPP, SupportsEagle3)`；MiniMax M3（稀疏注意力）走 `vllm/models/minimax_m3/` 厂商隔离包，与 DeepSeek V4 同模式（nvidia/amd/common 三分）。
- **字节豆包系**：`bailing_moe.py`（BailingMoe V1）、`bailing_moe_linear.py`（BailingMoe V2.5），含独立 MTP（`bailing_moe_mtp.py`）。
- **腾讯 Hunyuan**：`hunyuan_v1.py`（dense + MoE 共用类，`HunYuanMoEV1`/`HunYuanDenseV1`）、`hunyuan_vision.py`（VL）。
- **腾讯 HY-V3**：`hy_v3.py HYV3ForCausalLM` + `hy_v3_mtp.py` MTP。
- **小米 MiMo**：`mimo.py`、`mimo_v2.py`/`mimo_v2_omni.py`、`mimo_audio.py`、`mimo_mtp.py`/`mimo_v2_mtp.py`（多个 MTP draft）。
- **美团 Longcat**：`longcat_flash.py LongcatFlashForCausalLM` + `longcat_flash_mtp.py`。
- **百度 ERNIE/Pangu**：`ernie45.py`/`ernie45_moe.py`/`ernie45_vl*.py`/`ernie_mtp.py`（旧 ERNIE 退役，v0.23.0）、`openpangu.py`/`openpangu_mtp.py`/`openpangu_vl.py`（PanguEmbedded/PanguProMoEV2/PanguUltraMoE）。
- **TeleChat/TeleFLM**：`telechat2.py`（`TeleChat2ForCausalLM`，旧 `TeleChatForCausalLM` 别名）、`teleflm.py`；`TeleChat3ForCausalLM` 别名指向 Llama。
- **LG ExaOne**：`exaone.py`/`exaone4.py`/`exaone4_5.py`/`exaone_moe.py`/`exaone_moe_mtp.py`/`exaone4_5_mtp.py`（多个 MTP）。
- **Preferred Networks Plamo**：`plamo2.py`/`plamo3.py`（数值稳定性需 bfloat16）。
- **昆仑 StepFun**：`step1.py`、`step3_text.py`/`step3_vl.py`/`step3p5.py`/`step3p5_mtp.py`/`step3p7.py`/`step_vl.py`。
- **SF/Kunlun Skywork**：`skyworkr1v.py`（VLM 推理模型）。
- **阶跃 Keye**：`keye.py`/`keye_vl1_5.py`。
- **Kanaan V**：`kanana_v.py`（韩国 NA Kakao）。
- **OCR 系**：`deepseek_ocr.py`/`deepseek_ocr2.py`、`dots_ocr.py`、`lightonocr.py`、`unlimited_ocr.py`、`paddleocr_vl.py`（百度）、`qianfan_ocr.py`（百度千帆）、`midashenglm.py`（小米 MiDashengLM 语音）。
- **杂系**：`AXK1.py`（Ant AXK1）、`rnj1.py`、`jais2.py`（G42 Jais）、`iquest_loopcoder.py`、`cheers.py`。

---

## 为什么

- **MTP 成为主流**：上述十几个家族几乎都自带 MTP，反映"在线训练 nextn 投机解码"在中国厂商中已成标配。vLLM 对应的 `_SPECULATIVE_DECODING_MODELS` 字典中 MTP 类目大部分来自亚洲厂商。
- **稀疏注意力跟进**：MiniMax M3 与 DeepSeek V3.2/V4 一起把 sparse attention 推进主流，触发 `vllm/models/` 厂商隔离布局扩展。
- **OCR/VLM 多样**：百度千帆、PaddleOCR、DeepSeek-OCR、Dots-OCR、LightOnOCR、Unlimited-OCR 等多家 OCR 模型接入，反映 vLLM 在文档理解场景的扩展。

---

## 怎么做

每个家族基本沿用"自家 attention + FusedMoE + (可选 MLA/hybrid) + MTP"模板，差异主要在路由细节、norm 顺序、视觉塔选择。MTP 文件实现单层 DeepSeekMultiTokenPredictor 风格的 nextn 结构（见 `deepseek_mtp.py:130 DeepSeekMultiTokenPredictor`），各家族写自己的版本以适配 backbone 隐维度。

MiniMax M3 走 `vllm/models/minimax_m3/`，`common/` 提供 sparse_attention / indexer / vision_tower / mm_preprocess，`nvidia/` 与 `amd/` 各自实现平台 kernel——与 DeepSeek V4 同构（见 [`vendor-split-models.md`](../vendor-split-models.md)）。

---

## 与其它模块/系统配合

- **[vendor-split-models](../vendor-split-models.md)**：MiniMax M3 隔离实现。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：数十个 MTP draft + `Eagle3MiniMaxM2ForCausalLM`。
- **[多模态](../../11-multimodal/README.md)**：Hunyuan-VL / Step-VL / Pangu-VL / Keye-VL / Exaone4.5 等 VLM。
- **[speech-audio](./speech-audio.md)**：MiMo-Audio / MiDashengLM。
- **[vlm-misc](./vlm-misc.md)**：OCR 系（DeepSeek-OCR、Dots-OCR、LightOn、Unlimited、Paddle、Qianfan）。
- **[配置](../../10-config/README.md)**：`plamo2`/`glm4` 等列入 FP16 禁用；`SequenceClassificationConfig`/`VerifyAndUpdateConfig` 部分被本家族使用（如 `Ernie4_5_VLMoeForConditionalGenerationConfig` 在 `config.py:42`）。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| v0.6–v0.7 | Qwen-VL/Audio / Yi 等亚洲系先发；旧 InternLM/Qwen/QWen 清退（v0.23.0）。 |
| v0.8 | TeleChat2 / Hunyuan-V1 / Step1 / Plamo2 / Jais2 / Skywork-R1V。 |
| v0.9 | MiniMaxM2 + EAGLE-3；ERNIE-4.5 全家；BailingMoe；HY-V3 + MTP。 |
| v0.10 | Longcat-Flash + MTP；MiMo + MiMo-Audio + 多 MTP；AXK1 / RNJ1 / Keye / Kanana-V。 |
| v0.11 | ExaOne 4/4.5/Moe/OCR + MTP；Pangu 全家 + MTP/VL；Step3.x + MTP；MiMo-V2/Omni。 |
| v0.12 | MiniMax M3（稀疏注意力 VLM，厂商隔离）；DeepSeek-OCR2 / Dots-OCR / Unlimited-OCR。 |
| main | 持续新增 MTP/VLM 变体；MTP draft 字典大幅扩张。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [vendor-split-models](../vendor-split-models.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [vlm-misc](./vlm-misc.md) · [speech-audio](./speech-audio.md)
