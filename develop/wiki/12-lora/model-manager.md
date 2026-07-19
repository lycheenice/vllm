[← Wiki 首页](../README.md) > [LoRA](README.md) > Model Manager

# LoRAModelManager / LRUCacheLoRAModelManager / AdapterLRUCache

> GPU 侧 LoRA 适配器的"激活管理器"：管 slot、替换原模型层、把 CPU 权重拷进 GPU stacked buffer、维护 LRU。是 LoRA 子系统在 worker 内的核心枢纽。

## 是什么

### LoRAModelManager（`vllm/lora/model_manager.py:71`）

管理一个基座模型上的多个 LoRA 适配器 GPU 激活态。关键字段：

| 字段 | 说明 |
|---|---|
| `model` | 被 LoRA 化的 `SupportsLoRAModel` |
| `supported_lora_modules` | 模型支持的 LoRA 模块后缀（`get_supported_lora_modules`） |
| `packed_modules_mapping` | 融合模块名→子模块名映射（含 MoE 专家展开） |
| `modules: dict[str, BaseLayerWithLoRA]` | 已替换的 LoRA 层，按完整模块名索引 |
| `_registered_adapters: AdapterLRUCache[LoRAModel]` | 已加载适配器缓存（容量=`max_cpu_loras`） |
| `_active_adapters: AdapterLRUCache[None]` | 已激活进 GPU slot 的集合（容量=`max_loras`） |
| `lora_index_to_id: list[int|None]` | slot index→adapter id 映射 |
| `punica_wrapper_mapping: dict[str, PunicaWrapperBase]` | 前缀→Punica wrapper（语言/塔/connector） |

属性 `capacity`=`max_cpu_loras`，`lora_slots`=`max_loras`，`adapter_slots`=`lora_slots`（`vllm/lora/model_manager.py:282-293`）。

### LRUCacheLoRAModelManager（`vllm/lora/model_manager.py:1168`）

子类，重写 `add_adapter`/`activate_adapter`/`remove_oldest_adapter`/`pin_adapter` 增加 LRU touch 与淘汰。

### AdapterLRUCache（`vllm/lora/model_manager.py:60`）

`LRUCache[int, T]` 子类，覆写 `_on_remove`：被驱逐时调 `deactivate_fn(key)`，确保从 GPU slot 卸载后再丢弃 CPU 权重。

## 为什么

- **两级缓存**：CPU `max_cpu_loras`（多）+ GPU `max_loras` slot（少），用 LRU 在两者间调度，平衡显存与重载开销。
- **slot 化前向**：`lora_a_stacked`/`lora_b_stacked` 预分配 `(max_loras, ...)` 形状，前向只按 index 取，图静态、cudagraph 可捕获。
- **层替换一次性**：构造期 `_create_lora_modules` 遍历模型所有模块，命中 target 的换成 `BaseLayerWithLoRA` 子类，之后前向无需再改图。
- **多模态多 wrapper**：语言/tower/connector 各自独立 Punica wrapper 与 token 预算，避免互相挤占（`_maybe_init_mm`，`vllm/lora/model_manager.py:173`）。
- **MoE 布局自适应**：`_is_3d_moe_model`/`_enable_mixed_moe_lora_format` 决定用 `FusedMoE3DWithLoRA` 还是通用 `FusedMoEWithLoRA`，并支持 3D→2D 转换（`_convert_3d_to_2d_moe_lora`）。
- **EP 切片**：`_restrict_to_local_experts`/`_slice_moe_lora_ep` 让每 rank 只持有本地专家，省内存。

## 怎么做

### 构造期层替换（`_create_lora_modules`，`vllm/lora/model_manager.py:385`）

1. 遍历 `model.named_modules`，跳过 `PPMissingLayer`。
2. `_match_target_modules`：先验是否在 `supported_lora_modules`（regex 后缀），再过 `LoRAConfig.target_modules` 部署过滤（含 packed 父子互匹配，`vllm/lora/utils.py:271`）。
3. `_get_punica_wrapper(module_name)` 选 wrapper（非 MM 直接语言 wrapper；MM 按最长前缀匹配 tower/connector/language）。
4. non-gated MoE 的 `mixer.gate` 显式跳过（peft 限制）。
5. 同一底层 module 已被包装（别名路径，如 MoE gate 同时挂在 block 与 runner）→ rewire 别名属性指向同一 wrapper，避免重复注册导致 `reset_lora` 误清（`vllm/lora/model_manager.py:425`）。
6. `MoERunner` 按是否 3D 决定 `packed_moduled_lst`（`["w13"]` vs `["w1","w3"]`），`from_layer` 选 `FusedMoE3DWithLoRA`/`FusedMoEWithLoRA`。
7. `replace_submodule` 把新 LoRA 层挂回模型；`lm_head` 额外包装 `LogitsProcessorWithLoRA`。
8. `register_module` + `_register_packed_modules` + `new_module.set_mapping(punica_wrapper)`。

### add_adapter → 激活

- `add_adapter`（`vllm/lora/model_manager.py:1140`）：超容量抛错；`_add_adapter` 调 `_create_merged_loras_inplace`（打包/EP 切片/stack/pin_memory）后入 `_registered_adapters`。
- `activate_adapter`（`vllm/lora/model_manager.py:295`）：找空 slot index，对该 LoRA 每个注册模块调 `set_lora(index, A, B)`，无权重的调 `reset_lora(index)`。LRU 版本超 slot 先 `remove_oldest`。
- `deactivate_adapter`（`vllm/lora/model_manager.py:1133`）：清 `lora_index_to_id` slot。

### _create_merged_loras_inplace（`vllm/lora/model_manager.py:734`）

对每个 packed module：
- MoE `.experts` 走 `_restrict_to_local_experts` 收窄到本地专家，`_pad_lora_pairs_to_triplets`（非 gated）补 None，`PackedLoRALayerWeights.pack_moe`。
- 普通融合 `PackedLoRALayerWeights.pack`。
- 删除参与打包的子模块条目（含非本地专家）释放内存。
- 全体 `optimize()`。
- 遍历 `modules`：`FusedMoE3DWithLoRA` 调 `_stack_moe_lora_weights`（3D reshape+EP slice+permute）；`FusedMoEWithLoRA` 在混合模式下调 `_convert_3d_to_2d_moe_lora`，否则 `_slice_moe_lora_ep`。
- 最后统一 `pin_memory`（合并后做，避免 pack 失效）。

### 映射与 Punica 更新

`set_adapter_mapping`（`vllm/lora/model_manager.py:1149`）去重后 `_set_adapter_mapping`（`vllm/lora/model_manager.py:354`）：按 `mapping.type`（LANGUAGE/TOWER/CONNECTOR）选 punica wrapper，调 `update_metadata(mapping, lora_index_to_id, lora_slots+1, vocab_size)`。

### dummy / warmup

- `create_dummy_lora`（`vllm/lora/model_manager.py:536`）：零张量填充所有支持模块，3D MoE 单独走 w2/w13 双权重；非 gated MoE 补三元组后 `pack_moe`。
- `get_dummy_lora_warmup_rank`（`vllm/lora/model_manager.py:650`）：fully_sharded 时按各模块 `tp_size` LCM 对齐 rank，避免 MoE rank 轴切片失配。

### LRU / pin

- `LRUCacheLoRAModelManager.add_adapter`（`vllm/lora/model_manager.py:1171`）：已存在 touch；不存在则 `_add_adapter`。
- `LRUCacheLoRAModelManager.activate_adapter`（`vllm/lora/model_manager.py:1183`）：slot 满先 `remove_oldest`，激活后 touch。
- `pin_adapter`（`vllm/lora/model_manager.py:1203`）：CPU cache `pin` + GPU `activate` + active `pin`，锁定不被淘汰。

## 与其它模块/系统配合

- **WorkerLoRAManager**：持有 `LoRAModelManager`，做 CPU 加载后委托激活；见 [worker-manager.md](worker-manager.md)。
- **BaseLayerWithLoRA 系**：`set_lora`/`reset_lora`/`set_mapping` 的被调用方；见 [layers.md](layers.md)。
- **Punica wrapper**：`update_metadata` 与 `add_lora_*` 的驱动者；见 [punica.md](punica.md)。
- **LoRARequest/LoRAModel**：身份与权重来源；见 [request.md](request.md)、[lora-model.md](lora-model.md)。
- **utils.from_layer**：层选型工厂；见 [utils.md](utils.md)。
- [模型执行-FusedMoE](../03-model-execution/layers/fused-moe.md)：MoE 层替换与 `MoELoRAContext`。
- [多模态](../11-multimodal/README.md)：`_maybe_init_mm` 建 tower/connector wrapper。

## 历史版本演进

- **v0.5（首版）**：`LoRAModelManager` + `LRUCacheLoRAModelManager` + `AdapterLRUCache`；`_create_lora_modules` 遍历替换；`activate_adapter` slot 写入。
- **v0.7（v1 mixin 接入）**：`create_lora_manager` 工厂接受 `vllm_config`；多模态 `supports_mm`/`supports_tower_connector_lora` 分支引入。
- **v0.9（多模态 LoRA）**：`punica_wrapper_mapping` 多前缀；`_maybe_init_mm` 按 mm_mapping 分配 tower/connector wrapper 与各自 token 预算。
- **v0.11（MoE LoRA 3D + EP）**：`_is_3d_moe_model`/`_enable_mixed_moe_lora_format`；`_stack_moe_lora_weights`/`_convert_3d_to_2d_moe_lora`/`_slice_moe_lora_ep`/`_restrict_to_local_experts`/`_build_moe_ep_load_spec`/`MoEEPLoadSpec` 全套上线；非 gated MoE padding。
- **main**：`load_inplace` 热替换；别名 module rewire 防误清；`VLLM_LORA_ENABLE_DUAL_STREAM` 双流（在层里）；`pin_adapter` 落地 GPU pin。

## 参见

- [← 返回 LoRA 首页](README.md)
- [worker-manager.md](worker-manager.md)
- [layers.md](layers.md)
- [lora-model.md](lora-model.md)
- [utils.md](utils.md)
- [punica.md](punica.md)
