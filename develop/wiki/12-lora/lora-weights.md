[← Wiki 首页](../README.md) > [LoRA](README.md) > LoRA Weights

# LoRALayerWeights & PackedLoRALayerWeights

> 描述"一层 LoRA"的两个低秩矩阵 A/B 及缩放，是 `LoRAModel` 的最小组成单元；打包版本用于 qkv/gate_up/专家等融合层。

## 是什么

### LoRALayerWeights

`vllm/lora/lora_weights.py:13`，表示单层 LoRA：

| 属性 | 形状 | 说明 |
|---|---|---|
| `module_name` | `str` | 适配器内模块全名 |
| `rank` | `int` | LoRA rank |
| `lora_alpha` | `int` | 缩放分子 |
| `lora_a` | `(rank, input_dim)` | 降维矩阵 |
| `lora_b` | `(output_dim, rank)` | 升维矩阵 |
| `scaling` | `float` | `alpha/rank` 或构造时传入 |

属性 `input_dim`/`output_dim`（`vllm/lora/lora_weights.py:44-50`）取自 A/B 形状；`is_packed` 为 `False`。

### PackedLoRALayerWeights

`vllm/lora/lora_weights.py:99`，继承 `LoRALayerWeights`，用于 qkv_proj、gate_up_proj、专家等"一个运行时模块由多个子模块融合"的场景。把 N 个子层 LoRA 打包成 list 形式的 A/B：

- `lora_alphas: list[int|None]`、`lora_a: list[torch.Tensor|None]`、`lora_b: list[torch.Tensor|None]`、`scaling: list[float|None]`。
- `is_packed` 为 `True`；`input_dim`/`output_dim` 抛 `NotImplementedError`（打包后无单一维度）。

## 为什么

- **缩放前置融合**：`optimize()`（`vllm/lora/lora_weights.py:36`）把 `scaling` 乘进 `lora_b`，使前向只需 `x @ A @ B`，内核无需每次乘标量，减少算子开销。
- **打包减少算子**：融合层（如 `qkv_proj`）原本 3 个独立 LoRA，打包后由 Punica `add_lora_linear` 一次处理 3 slice，避免多次 kernel launch。
- **None 占位语义**：`PackedLoRALayerWeights.pack` 允许 list 中元素为 `None`，表示该子模块未加 LoRA（如 qkv 中只训了 q），`set_lora` 时跳过对应 buffer。
- **MoE 专家堆叠**：`pack_moe` 把每专家的 w1/w2/w3 按维度 0 stack 成 `(num_experts, rank, in)`，直接喂给 `FusedMoEWithLoRA.set_lora`。

## 怎么做

### 构造路径

1. **配置占位**：`LoRALayerWeights.from_config(module_name, peft_helper)`（`vllm/lora/lora_weights.py:56`）只建空壳（A/B=None），填入 rank/alpha/scaling，用于 `LoRAModel.from_lora_tensors` 先占位。
2. **填值**：`LoRAModel.from_lora_tensors`（`vllm/lora/lora_model.py:116`）遍历 safetensors，按 `parse_fine_tuned_lora_name` 判 lora_A/lora_B，赋给对应 `LoRALayerWeights.lora_a/b`，并按需 `pin_memory`。
3. **dummy**：`create_dummy_lora_weights`（`vllm/lora/lora_weights.py:72`）建零张量，用于 warmup/cudagraph 捕获。

### 打包

- **普通融合**：`PackedLoRALayerWeights.pack(loras)`（`vllm/lora/lora_weights.py:126`）— 先对每个非 None 元素 `optimize()`，再收集 alphas/A/B/scaling 成 list，scaling 统一置 1（已 optimize）。
- **MoE**：`pack_moe(loras, module_name, is_non_gated_moe)`（`vllm/lora/lora_weights.py:154`）：
  - 断言 `len(loras) % 3 == 0`，按专家分组取 w1/w2/w3。
  - 非 gated MoE 缺 w3 时复用 w1（`vllm/lora/lora_weights.py:185`），并设 w3 scaling=1 避免双重缩放。
  - `torch.stack` 生成 `(num_experts, rank, in)` / `(num_experts, out, rank)`，传入 `[lora_alpha]*3` 与 `[scaling, scaling, last_scaling]`。

由 `LoRAModelManager._create_merged_loras_inplace`（`vllm/lora/model_manager.py:734`）调用：对每个 packed module 收集子层 LoRA，按 `.experts` 后缀走 `pack_moe`，否则 `pack`，随后 `optimize()` + `pin_memory()`。

### optimize

```python
def optimize(self):
    if self.scaling == 1: return self
    self.lora_b *= self.scaling
    self.scaling = 1
    return self
```

`PackedLoRALayerWeights.optimize`（`vllm/lora/lora_weights.py:230`）逐 slice 处理，跳过 None。

## 与其它模块/系统配合

- **LoRAModel**：持有 `dict[module_name, LoRALayerWeights]`；见 [lora-model.md](lora-model.md)。
- **模型构建**：`from_config` 借 `PEFTHelper.vllm_lora_scaling_factor`；见 [peft-helper.md](peft-helper.md)。
- **LoRAModelManager.activate_adapter**：调 `module.set_lora(index, lora_a, lora_b)` 把 A/B 拷进 GPU stacked buffer；见 [model-manager.md](model-manager.md)。
- **FusedMoE LoRA 层**：`pack_moe` 产物喂 `FusedMoEWithLoRA.set_lora`（`vllm/lora/layers/fused_moe.py:355`）；见 [layers.md](layers.md)。
- **Punica wrapper**：消费 stacked 形式的 A/B；见 [punica.md](punica.md)。

## 历史版本演进

- **v0.5（首版）**：`LoRALayerWeights` + `PackedLoRALayerWeights.pack` 支持普通融合层（qkv/gate_up）。
- **v0.7（待核实）**：`is_packed` 属性 + `input_dim`/`output_dim` 在打包版本上显式抛 `NotImplementedError`，规范接口。
- **v0.10/v0.11（MoE）**：新增 `pack_moe`，按专家堆叠 3D 张量；引入非 gated MoE 复用 w1 逻辑与双重缩放规避。
- **main**：`create_dummy_lora_weights` 增加 `pin_memory` 支持；`optimize` 逻辑稳定。

## 参见

- [← 返回 LoRA 首页](README.md)
- [lora-model.md](lora-model.md)
- [model-manager.md](model-manager.md)
- [peft-helper.md](peft-helper.md)
- [layers.md](layers.md)
