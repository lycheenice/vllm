# GPT 经典与杂系 decoder 家族

[← Wiki 首页](../../README.md) > [模型库](../README.md) > [家族分组](./README.md) > **GPT 经典**

> 代表文件：`gpt2.py`、`gpt_j.py`、`gpt_neox.py`、`bloom.py`、`falcon.py`、`falcon_h1.py`、`opt.py`、`mpt.py`、`dbrx.py`、`gpt_oss.py`、`persimmon.py`、`orion.py`、`solar.py`、`stablelm.py`、`arctic.py`、`jais2.py`、`telechat2.py`、`teleflm.py`、`nemotron.py`、`nemotron_nas.py`、`nemotron_vl.py`、`plamo2.py`、`plamo3.py`、`sarvam.py`、`seed_oss.py`、`flex_olmo.py`、`olmo.py`、`olmo2.py`、`olmoe.py`、`arcee.py`、`apertus.py`、`afmoe.py`、`cosmos3.py`、`hrm_text.py`、`laguna.py`、`laguna_dflash.py`、`ouro.py`、`param2moe.py`、`step1.py` 等。
> 另：`GPTBigCodeForCausalLM`/`SmolLM3ForCausalLM`/`Starcoder2ForCausalLM` 走 [Transformers 后端](../transformers-backend.md)（`_TRANSFORMERS_SUPPORTED_MODELS`）。

---

## 是什么

本页收录"经典 decoder + 杂系厂商"的早期或独立架构家族：

- **GPT 系**：`gpt2.py GPT2LMHeadModel`、`gpt_j.py GPTJForCausalLM`、`gpt_neox.py GPTNeoXForCausalLM`——vLLM 最早期支持的架构；`gpt2.py` 还含 `GPT2ForSequenceClassification`。`gpt_oss.py GptOssForCausalLM` 是 OpenAI gpt-oss 系列（与 GPT2 无关）。
- **Bloom / Falcon / OPT / MPT**：经典开源 decoder，独立实现，与 Llama 不同 attention/Norm 设计。
- **`falcon_h1.py`**：Falcon-H1（hybrid），后期版本。
- **`dbrx.py`**：Databricks DBRX，MoE。
- **`olmo*.py`**：AllenAI OLMo 系列——`olmo.py`、`olmo2.py`（与 Llama 接近）、`olmoe.py`（MoE）、`flex_olmo.py`（FlexOlmo）、`olmo_hybrid.py`（hybrid，归 [mamba-ssm](./mamba-ssm.md)）。
- **`nemotron*.py`**：NVIDIA 系——`nemotron.py`（Nemotron-4）、`nemotron_nas.py`（DeciLM/NAS，`DeciLMForCausalLM` 同时注册到 embedding）、`nemotron_vl.py`、`nemotron_h.py`（hybrid，归 mamba-ssm）、`nemotron_parse.py`（encoder-decoder）、`nano_nemotron_vl.py`。
- **IBM/厂商系**：`plamo2.py`/`plamo3.py`（Preferred Networks）、`sarvam.py`（Sarvam MoE/MLA）、`seed_oss.py`（ByteDance SeedOss）、`arctic.py`（Snowflake Arctic MoE）、`jais2.py`（Jais2）、`apertus.py`、`arcee.py`、`afmoe.py`、`cosmos3.py`（NVIDIA Cosmos3 VLM）、`hrm_text.py`（HRM）、`laguna.py` + `laguna_dflash.py`（Laguna DFlash draft）、`ouro.py`、`param2moe.py`、`telechat2.py`/`teleflm.py`（TeleChat）、`step1.py`（StepFun Step1）、`step3_text.py`/`step3p5.py`/`step3p7.py`/`step3_vl.py`/`step_vl.py`（Step 系列，详见 [asian-vendor](./asian-vendor.md)）。

`gpt_oss.py` 的 `GptOssForCausalLM` 是后期加入的 OpenAI gpt-oss 系列实现。

---

## 为什么

- **早期支持奠定兼容面**：GPT2/Bloom/Falcon/OPT/MPT 是 vLLM 早期兼容 HF 生态的"必答题"，几乎都独立实现，与 Llama 子类化不同路径。
- **杂系厂商聚合**：Plamo/Sarvam/Seed/Arctic/Jais 等各自只 1-2 个架构，统一在本页概述，避免单独开页。详细 spec_decode/能力差异需查源码。
- **Transformers 后端分流**：`GPTBigCode`/`Starcoder2`/`SmolLM3` 不写原生实现，全部走 HF 后端——意味着"小众/稳定但不算性能关键"的架构倾向后端路径。

---

## 怎么做

GPT2/J/NeoX 在 attention 后端选型上无特殊；Bloom 用 ALiBi 位置编码（`RotaryEmbedding` 替换为 ALiBi bias）；Falconattention fuse qkv 形式独特（单 `query_key_value` 矩阵）；DBRX/Arctic/Olmoe 走 `FusedMoE`。

`nemotron_nas.py DeciLMForCausalLM` 同时注册到生成与 embedding（因 DeciLM 架构兼容双向池化）。

不少文件含 DFlash/DSpark/EAGLE/MTP 等 spec draft（`laguna_dflash.py`、`param2moe_mtp` 等），与 [采样-投机](../../06-sampling-decoding/speculative-decoding/README.md) 字典对应。

---

## 与其它模块/系统配合

- **[transformers-backend](../transformers-backend.md)**：Starcoder/SmolLM3/GPTBigCode 走 HF 后端。
- **[mamba-ssm](./mamba-ssm.md)**：Falcon-H1 / Olmo-Hybrid / Nemotron-H 跨家族。
- **[采样-投机](../../06-sampling-decoding/speculative-decoding/README.md)**：Laguna DFlash、各 MTP。
- **[embedding-col](./embedding-col.md)**：DeciLM 双向 embedding。

---

## 历史版本演进

| 版本 | 变更 |
|---|---|
| 早期 | GPT2/GPT-J/GPT-NeoX/Bloom/Falcon/OPT/MPT 接入。 |
| v0.5 | DBRX + Arctic MoE；Snowflake/AI21 等。 |
| v0.6–v0.7 | OLMo* 系列；Nemotron/DeciLM。 |
| v0.8 | Plamo2/3、Sarvam、SeedOss。 |
| v0.9 | GPT-OSS、Falcon-H1。 |
| v0.10 | 多 MoE 厂商模型；部分早期模型清退（`BaiChuan`/`Aquila`/`Grok1` 等入 `_PREVIOUSLY_SUPPORTED_MODELS`）。 |
| main | 持续小幅扩张，部分转入 HF 后端降维护成本。 |

---

## 参见

- [← 返回家族分组](./README.md)
- [transformers-backend](../transformers-backend.md) · [mamba-ssm](./mamba-ssm.md) · [采样-投机解码](../../06-sampling-decoding/speculative-decoding/README.md) · [asian-vendor](./asian-vendor.md)
