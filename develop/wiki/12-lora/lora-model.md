[← Wiki 首页](../README.md) > [LoRA](README.md) > LoRAModel

# LoRAModel & MoEEPLoadSpec

> 一个 LoRA 适配器在 CPU 侧的完整权重集合（module_name → LoRALayerWeights），由 safetensors/bin/pt/tensorizer 加载，是 `LoRAModelManager` 缓存的基本单位。

## 是什么

`LoRAModel`（`vllm/lora/lora_model.py:60`）持有：

| 属性 | 说明 |
|---|---|
| `id` | `lora_model_id`，对应 `LoRARequest.lora_int_id` |
| `rank` | 适配器 rank（来自 `PEFTHelper.r`） |
| `loras` | `dict[str, LoRALayerWeights]`，模块名 → 单层权重 |
| `is_3d_lora_weight` | 磁盘 MoE 布局声明，透传自 `LoRARequest`（`vllm/lora/lora_model.py:88`） |

配套 `MoEEPLoadSpec`（`vllm/lora/lora_model.py:25`，`@dataclass(frozen=True)`）：`ep_rank`/`local_num_experts`/`global_num_experts`，用于专家并行下加载期裁剪非本地专家。

## 为什么

- **CPU 驻留 + 按需上 GPU**：`LoRAModel` 常驻 CPU（`device="cpu"` + `pin_memory`），仅当 `activate_adapter` 时把 A/B 拷进 GPU slot，让 `max_cpu_loras > max_loras` 成为可能（CPU 缓存多，GPU slot 少）。
- **多检查点格式**：`from_local_checkpoint` 支持 `adapter_model.safetensors`/`.bin`/`.pt`/tensorizer，兼容 PEFT 训练产物与序列化分发。
- **模块名校验**：`check_unexpected_modules` 用 safetensors key 作真相源，发现磁盘上有但 `expected_lora_modules` 不含的模块即报错，避免静默错配。
- **EP 早裁剪**：`MoEEPLoadSpec` + `_is_remote_expert_key`（`vllm/lora/lora_model.py:41`）在 `safe_open` 读张量前就跳过非本 rank 专家，省 CPU 内存与拷贝。
- **权重名映射**：通过 `weights_mapper`（如 Qwen2VL 的 `hf_to_vllm_mapper`）把 HF 名映回 vLLM 模块名，保证跨模型架构正确装填。

## 怎么做

### from_local_checkpoint（`vllm/lora/lora_model.py:167`）

参数：`lora_dir`、`expected_lora_modules`（来自 `LoRAModelManager.supported_lora_modules` + packed mapping）、`peft_helper`、`lora_model_id`、`device`、`dtype`、`model_vocab_size`、`weights_mapper`、`tensorizer_config_dict`、`skip_prefixes`、`moe_ep_spec`。

流程：

1. 定位检查点文件，优先 safetensors（`vllm/lora/lora_model.py:261`），次 `.bin`/`.pt`，再 tensorizer。
2. `check_unexpected_modules`（`vllm/lora/lora_model.py:212`）：对 expert 与普通模块分别校验 membership；`base_layer`（PEFT 3D 专家的 gate_up）显式跳过。
3. 若 `moe_ep_spec` 给定：safetensors 用 `continue` 跳过 remote expert key（`vllm/lora/lora_model.py:273`）；`.bin`/`.pt` 读后 dict 过滤（`vllm/lora/lora_model.py:286`）。
4. 委托 `from_lora_tensors`（`vllm/lora/lora_model.py:116`）：
   - 跳过 base embedding 权重（`is_base_embedding_weights`）。
   - `skip_prefixes`（如 MTP 层）按 `_should_skip_module` 跳过。
   - `parse_fine_tuned_lora_name` 解析模块名 + lora_A/B 标志，经 `weights_mapper` 映射。
   - 首次见到模块用 `LoRALayerWeights.from_config` 占位，再填 A 或 B。
   - embedding LoRA 的 A 维度需等于 `model_vocab_size`，否则 `RuntimeError`（`vllm/lora/lora_model.py:146`）。
   - 按 device/dtype 转换 + `pin_memory`。

### from_lora_tensors 的产物

返回 `LoRAModel(id, peft_helper.r, loras)`。注意此时 A/B 仍是原始（未 optimize、未打包），打包由 `LoRAModelManager._create_merged_loras_inplace` 后续完成。

### clone

`clone(lora_model_id)`（`vllm/lora/lora_model.py:90`）返回共享底层张量、换 id 的副本，用于 dummy LoRA 复用（`WorkerLoRAManager.add_dummy_lora`，`vllm/lora/worker_manager.py:174`）。

### EP 裁剪细节

`_is_remote_expert_key`（`vllm/lora/lora_model.py:41`）：定位 `.experts.` 分隔符，解析专家 idx，若不在 `[ep_rank*local, ep_rank*local+local)` 区间返回 True。`MoEEPLoadSpec` 由 `LoRAModelManager._build_moe_ep_load_spec`（`vllm/lora/model_manager.py:1094`）从首个 2D `FusedMoEWithLoRA` 模块构建。

## 与其它模块/系统配合

- **WorkerLoRAManager._load_adapter**：`vllm/lora/worker_manager.py:142` 调 `from_local_checkpoint`，并把 `lora_request.is_3d_lora_weight` 盖到 `lora.is_3d_lora_weight`（`vllm/lora/worker_manager.py:158`）。见 [worker-manager.md](worker-manager.md)。
- **LoRAModelManager**：`add_adapter`→`_create_merged_loras_inplace` 做打包/EP 切片/stack；`activate_adapter` 把 A/B 上 GPU。见 [model-manager.md](model-manager.md)。
- **PEFTHelper**：提供 rank/alpha/scaling；见 [peft-helper.md](peft-helper.md)。
- **WeightsMapper**：来自 `vllm/model_executor/models/utils.py`，模型层定义 HF↔vLLM 名映射。
- **FusedMoE LoRA 层**：消费 `MoEEPLoadSpec` 并在 `set_lora` 里断言 `num_experts == A.shape[0]`；见 [layers.md](layers.md)。

## 历史版本演进

- **v0.5（首版）**：`LoRAModel.from_local_checkpoint` 支持 safetensors/.bin/tensorizer；`from_lora_tensors` 基本流程；`is_base_embedding_weights` 跳过 base embedding。
- **v0.7（待核实）**：加入 `weights_mapper` 支持多模态/重命名模型；`skip_prefixes` 让模型声明跳过 MTP 等推理不用的层。
- **v0.10/v0.11（MoE + EP）**：新增 `MoEEPLoadSpec` 与 `_is_remote_expert_key`，加载期裁剪非本地专家；`is_3d_lora_weight` 字段透传磁盘 3D 布局；`check_unexpected_modules` 处理 PEFT `base_layer`（gate_up）特殊情况。
- **main**：`_should_skip_module` 让模型自定义跳过前缀；EP 裁剪对 `.bin`/`.pt` 路径补 dict 过滤；`max_position_embeddings` 取 `get_text_config()` 与 encoder-decoder 的 `max_target_positions`（在 `WorkerLoRAManager` 里）。

## 参见

- [← 返回 LoRA 首页](README.md)
- [lora-weights.md](lora-weights.md)
- [model-manager.md](model-manager.md)
- [worker-manager.md](worker-manager.md)
- [peft-helper.md](peft-helper.md)
