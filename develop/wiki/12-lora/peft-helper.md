[← Wiki 首页](../README.md) > [LoRA](README.md) > PEFTHelper

# PEFTHelper

> 解析 HuggingFace PEFT 的 `adapter_config.json`，校验 LoRA 特性兼容性，并计算缩放因子的轻量 dataclass。

## 是什么

`PEFTHelper`（`vllm/lora/peft_helper.py:20`）是 `@dataclass`，字段直接对应 PEFT `LoraConfig`：

| 字段 | 默认 | 说明 |
|---|---|---|
| `r` | 必填 | LoRA rank |
| `lora_alpha` | 必填 | 缩放分子 |
| `target_modules` | 必填 | PEFT 目标模块列表/字符串 |
| `bias` | `"none"` | 偏置训练模式（vLLM 仅支持 `none`） |
| `modules_to_save` | `None` | 全量保存模块（不支持） |
| `use_rslora` | `False` | Rank-Stabilized LoRA |
| `use_dora` | `False` | DoRA（不支持） |
| `vllm_lora_scaling_factor` | `1.0` | 计算出的缩放（vllm 前缀避免冲突） |
| `vllm_max_position_embeddings` | `False` | 基座模型 max positions，用于校验上下文 |

## 为什么

- **解耦 PEFT 依赖**：直接 `json.load` 读 `adapter_config.json` 而非 import `peft`，避免把 PEFT 作为运行期硬依赖。
- **校验前置**：`validate_legal`（`vllm/lora/peft_helper.py:116`）在加载权重之前就拒绝不支持的特性（`modules_to_save`、`use_dora`、`bias≠none`、`r > max_lora_rank`），快速失败。
- **缩放统一**：rslora 用 `alpha/sqrt(r)`，普通 LoRA 用 `alpha/r`（`vllm/lora/peft_helper.py:53-58`），后续 `LoRALayerWeights.optimize` 把缩放融进 `lora_b`，前向无需再乘。
- **tensorizer 支持**：`from_local_dir`（`vllm/lora/peft_helper.py:81`）可从 tensorizer 流读配置，匹配序列化适配器分发。

## 怎么做

### 解析入口

```python
peft_helper = PEFTHelper.from_local_dir(
    lora_path,
    max_position_embeddings=model_max_pos,
    tensorizer_config_dict=None,  # 或 tensorizer 配置
)
peft_helper.validate_legal(lora_config)  # 对照 vllm LoRAConfig
```

`from_local_dir`（`vllm/lora/peft_helper.py:80`）：
1. 拼 `adapter_config.json` 路径；若 `tensorizer_config_dict` 给定，改从 tensorizer 目录流式读取（`vllm/lora/peft_helper.py:89`）。
2. `json.load` 得 dict，注入 `vllm_max_position_embeddings`。
3. `from_dict`（`vllm/lora/peft_helper.py:60`）校验必填字段（`r`/`lora_alpha`/`target_modules`），过滤未定义字段后构造。

### 校验规则（`_validate_features` + `validate_legal`）

- `modules_to_save` 非空 → 不支持。
- `use_dora` True → 不支持。
- `r > lora_config.max_lora_rank` → 超限。
- `bias != "none"` → 不支持。
任一不满足即拼错误串抛 `ValueError`。

### 缩放计算

`__post_init__` 里：

```
use_rslora: vllm_lora_scaling_factor = lora_alpha / sqrt(r)
otherwise:  vllm_lora_scaling_factor = lora_alpha / r
```

该值经 `LoRALayerWeights.from_config`（`vllm/lora/lora_weights.py:56`）传入，再被 `optimize()` 融进 `lora_b`。

## 与其它模块/系统配合

- **WorkerLoRAManager._load_adapter**：`vllm/lora/worker_manager.py:120` 调 `PEFTHelper.from_local_dir` + `validate_legal`，通过后才 `LoRAModel.from_local_checkpoint`。
- **LoRALayerWeights**：消费 `r`/`lora_alpha`/`vllm_lora_scaling_factor`；见 [lora-weights.md](lora-weights.md)。
- **LoRAConfig**（[配置-LoRA](../10-config/lora-config.md)）：`validate_legal` 用其 `max_lora_rank` 做上限校验。
- **TensorizerConfig**：序列化适配器加载路径，见 `vllm/model_executor/model_loader/tensorizer.py`。

## 历史版本演进

- **v0.5（首版）**：`PEFTHelper` 引入，从 PEFT 源码改编（注释 `Adapted from peft/tuners/lora/config.py`，`vllm/lora/peft_helper.py:4`），支持基础 `r`/`lora_alpha`/`bias`/`modules_to_save`。
- **v0.6（rsLoRA）**：加入 `use_rslora`，缩放改为 `alpha/sqrt(r)` 分支（论文 https://arxiv.org/abs/2312.03732）。
- **v0.7（DoRA 检测）**：加入 `use_dora` 字段并显式拒绝（vLLM 暂不支持）。
- **v0.8+（tensorizer）**：`from_local_dir` 增加 tensorizer 流式读取分支。
- **main**：保持稳定；`vllm_lora_scaling_factor`/`vllm_max_position_embeddings` 用 `vllm_` 前缀避免与 PEFT 字段冲突。

## 参见

- [← 返回 LoRA 首页](README.md)
- [lora-weights.md](lora-weights.md)
- [lora-model.md](lora-model.md)
- [worker-manager.md](worker-manager.md)
- [配置-LoRA](../10-config/lora-config.md)
