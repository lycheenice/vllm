[← Wiki 首页](../../../README.md) > [模型执行](../../README.md) > [层库](../README.md) > [量化](README.md) > bitsandbytes

# bitsandbytes — BNB NF4/INT8 在线量化

> 源码（扁平 `.py`，**不存在 `bitsandbytes/` 子目录**）：`vllm/model_executor/layers/quantization/bitsandbytes.py`
>
> 方法名 `"bitsandbytes"`（`__init__.py:155`）。

---

## 是什么

`BitsAndBytesConfig`（`bitsandbytes.py:49`）消费 HuggingFace `bitsandbytes` 量化 checkpoint（`load_in_8bit` / `load_in_4bit`，[论文](https://arxiv.org/abs/2305.14314)）。BNB 是**在线/运行期**量化——checkpoint 通常仍是 fp16/bf16 原始权重或 BNB 格式，vLLM 在加载期调用 bitsandbytes 库做 NF4/FP4/INT8 量化与 kernel。

关键字段（`bitsandbytes.py:55`）：`load_in_8bit`、`load_in_4bit`(默认True)、`bnb_4bit_compute_dtype`、`bnb_4bit_quant_storage`(`uint8`)、`bnb_4bit_quant_type`(`fp4`/`nf4`)、`bnb_4bit_use_double_quant`、`llm_int8_*`（INT8 系：`threshold`、`has_fp16_weight`、`skip_modules`、`enable_fp32_cpu_offload`）。

版本要求：`bitsandbytes>=0.48.1`，ROCm 需 `>=0.49.2`（`_check_bitsandbytes_version`，`:31`）。min capability 70（`:104`）。

| 类 | 角色 |
|---|---|
| `BitsAndBytesConfig` | Config，`from_config` 读 HF 字段；不实现 `override_quantization_method`（走 `quant_method=="bitsandbytes"` 直配） |
| `BitsAndBytesLinearMethod` | 4-bit NF4/FP4 Linear（`create_weights` 建 `weight` uint8 packed + `weight_scale`/`zeros`/`quant_state`） |
| `BitsAndBytesMoEMethod` | 4-bit BNB MoE (`FusedMoEMethodBase`) |
| (INT8 path) | 8-bit LLM.int8 Linear (待核实类名行号) |

---

## 为什么

- **生态兼容**。BNB 是 HF 生态默认的"开箱即用"量化，用户量大。vLLM 直接消费 HF BNB checkpoint，降低迁移门槛。
- **双量化与 NF4**。`bnb_4bit_use_double_quant` 对 scale 再量化，`nf4` dtype 针对 LLM 权重分布优化，精度/显存比好。
- **MoE 支持**。`BitsAndBytesMoEMethod` 把 BNB 4-bit 扩展到 FusedMoE。
- **CPU offload**。`llm_int8_enable_fp32_cpu_offload` 支持显存不足时部分权重卸载到 CPU（待核实当前实现完整度）。

---

## 怎么做

### 配置接入

`from_config`（`bitsandbytes.py:112`）用 `get_safe_value` 容错读取可选键。`get_config_filenames` 返回 `[]`（无额外 json）。`get_quant_method` 按 layer 类型返回对应 method。

### 权重创建与加载

4-bit：`weight` 为 `uint8` packed（每元素 4 bit，`pack_factor=2`），`packed_dim` 属性供 TP 切分；`quant_state`（BNB 内部分层 scale/zero 状态）由 BNB 库管理。`process_weights_after_loading` 触发 BNB 量化（若 checkpoint 非 BNB 格式）或反序列化 `quant_state`。

### 前向

`apply` 调用注册的 custom op（`direct_register_custom_op`，`bitsandbytes.py:28` import），BNB kernel 做 dequant+matmul（4-bit）或 LLM.int8（8-bit，按 `llm_int8_threshold` 分离 outlier 通道）。

### MoE

`BitsAndBytesMoEMethod`（引用 `FusedMoEConfig`/`FusedMoEQuantConfig`，`bitsandbytes.py:10`）为每个 expert 建 BNB 4-bit weight，路由后分组 dequant+GEMM。

### CUDA graph 限制

`ModelConfig._verify_bnb_config`（`vllm/config/model.py:1131`）：8-bit BNB 不支持 CUDA graph，自动 `enforce_eager=True`（待 BNB 修复）。

---

## 与其它模块/系统配合

- **[平台](../../../08-platforms/README.md)**：`current_platform.is_rocm()` 决定 BNB 最低版本；BNB 主要 CUDA，XPU/CPU 受限（待核实当前支持矩阵）。
- **[分布式](../../../07-distributed/README.md)**：packed weight TP 切分；`llm_int8_has_fp16_weight` 影响权重保留形态。
- **[编译-IR](../../../09-compilation-ir/README.md)**：BNB kernel 经 custom op 注册以兼容；8-bit + CUDA graph 不兼容（见上）。
- **HF transformers**：checkpoint 字段命名直接对齐 HF `BitsAndBytesConfig`。
- **MoE**（`layers/fused_moe/`）：复用 `FusedMoEConfig`。

---

## 历史版本演进

- **v0.5 及之前**：4-bit NF4/FP4 BNB 支持（`load_in_4bit`）。
- **v0.6–v0.7**（待核实）：8-bit LLM.int8 支持；`bnb_4bit_compute_dtype`/`double_quant` 完善。
- **v0.8–v0.9**（待核实）：BNB MoE（`BitsAndBytesMoEMethod`）加入。
- **v0.10–v0.12 / main**（待核实）：CUDA graph 限制附带警告；版本门槛提升到 0.48.1/0.49.2；custom op 注册路径统一。

---

[← 返回量化首页](README.md)

## 参见

- [量化首页](README.md) · [online.md](online.md) · [schemes.md](schemes.md) · [utils.md](utils.md)
