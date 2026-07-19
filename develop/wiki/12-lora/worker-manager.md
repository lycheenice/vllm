[← Wiki 首页](../README.md) > [LoRA](README.md) > Worker Manager

# WorkerLoRAManager & LRUCacheWorkerLoRAManager

> Worker 进程侧的 LoRA 总管：把 `LoRARequest` 落成 `LoRAModel`、按 LRU 调度 CPU/GPU、设激活映射。是 `LoRAModelManager` 的上层外壳，对接 v1 ModelRunner Mixin。

## 是什么

`WorkerLoRAManager`（`vllm/lora/worker_manager.py:26`）每个 worker 一实例，包装一个 `LoRAModelManager`（`_adapter_manager`）。关键字段：

| 字段 | 说明 |
|---|---|
| `_lora_model_cls` | 默认 `LoRAModel`，可替换 |
| `embedding_modules` | 模型 embedding 模块映射（供 dummy 用） |
| `_cached_dummy_lora` | `False`/`None`/`LoRAModel`，warmup dummy 复用 |
| `max_num_seqs`/`max_num_batched_tokens`/`vocab_size` | 取自 `vllm_config.scheduler_config`/`model_config` |
| `max_position_embeddings` | 取 `get_text_config()`；encoder-decoder 用 `max_target_positions`（`vllm/lora/worker_manager.py:61`） |
| `lora_config`/`device` | 取自 `vllm_config` |

`LRUCacheWorkerLoRAManager`（`vllm/lora/worker_manager.py:241`）：`_manager_cls=LRUCacheLoRAModelManager`，重写 `_apply_adapters`/`add_adapter` 做 LRU + `load_inplace`。

## 为什么

- **职责分层**：CPU 加载/路径解析/PEFT 校验在 worker manager；GPU 激活/slot/层替换在 model manager。前者偏 IO，后者偏算子。
- **LRU 按需加载**：`max_cpu_loras` 可远大于 `max_loras`，LRU 让高频适配器留 CPU、热适配器上 GPU，冷的下盘重载。
- **dummy warmup 复用**：`dummy_lora_cache` 上下文 + `clone(lora_int_id)` 让 cudagraph 捕获多次复用同一零权重，省构造开销。
- **热替换**：`load_inplace` 先装新权重再卸旧，保证替换瞬间不空槽；并允许临时超 `max_cpu_loras`（`vllm/lora/worker_manager.py:297`）。
- **跨模型通用**：通过 `hf_to_vllm_mapper.get_unstacked_mapper()`、`lora_skip_prefixes`、`moe_ep_spec` 适配 Qwen2VL/MoE/encoder-decoder 等多种架构。

## 怎么做

### 初始化与 model manager 创建

`create_lora_manager(model, vllm_config)`（`vllm/lora/worker_manager.py:85`）委托 `create_lora_manager` 工厂（`vllm/lora/model_manager.py:1225`），按 `_manager_cls` 实例化 `LoRAModelManager`，并 `self._adapter_manager = lora_manager`，返回被 LoRA 化后的 `model`。

### _load_adapter（`vllm/lora/worker_manager.py:105`）

1. 从 `_adapter_manager` 取 `supported_lora_modules` + `packed_modules_mapping`，展开成 `expected_lora_modules`（含 `experts` 自身）。
2. `get_adapter_absolute_path`（`vllm/lora/utils.py:314`）把 HF/ModelScope repo id 下载成本地快照，绝对路径原样返回。
3. `PEFTHelper.from_local_dir(path, max_position_embeddings, tensorizer_config_dict)` + `validate_legal`。
4. 取模型的 `hf_to_vllm_mapper` 并 `get_unstacked_mapper()`（丢弃 QKV/MLP 融合 substr，保留真实 rename/prefix，让 q_proj 等子名存活供打包）。
5. 取 `lora_skip_prefixes`（`vllm/lora/worker_manager.py:140`）。
6. `LoRAModel.from_local_checkpoint(... weights_mapper, skip_prefixes, moe_ep_spec=_adapter_manager.moe_ep_load_spec)`。
7. `lora.is_3d_lora_weight = lora_request.is_3d_lora_weight`（透传磁盘布局）。
8. `FileNotFoundError` → `LoRAAdapterNotFoundError`。

### set_active_adapters / _apply_adapters

- 基类 `_apply_adapters`（`vllm/lora/worker_manager.py:204`）：算 requested vs existing，差集卸载、新增 `add_adapter`（装+激活）；超 `adapter_slots` 报错。
- LRU 版 `_apply_adapters`（`vllm/lora/worker_manager.py:270`）：超 `lora_slots` 报错；对每个 requested `add_adapter`，已存在只 touch。
- LRU `add_adapter`（`vllm/lora/worker_manager.py:285`）：未加载或 `load_inplace` 时 `_load_adapter` → `remove_adapter` 旧的 → 超 capacity 淘汰最旧 → `_add_adapter`；最后 `activate_adapter`。

`set_active_adapters`（`vllm/lora/worker_manager.py:193`）：先 `_apply_adapters`，再 `set_adapter_mapping(mapping)` 注入 Punica。

### Add/Remove/Pin/List

- `add_adapter`（基类 `vllm/lora/worker_manager.py:223`）：load + model manager add + activate。
- `remove_adapter` / `remove_all_adapters` / `list_adapters`：委托 model manager。
- `pin_adapter`：委托 model manager.pin。
- `add_dummy_lora`（`vllm/lora/worker_manager.py:174`）：复用 `_cached_dummy_lora.clone(id)` 或新建。
- `dummy_lora_cache` 上下文（`vllm/lora/worker_manager.py:73`）：进入置 `None` 允许缓存，退出置 `False`。

### supports_tower_connector_lora（`vllm/lora/worker_manager.py:198`）

返回 `_adapter_manager.supports_mm and supports_tower_connector_lora`，供 v1 多模态路径判是否路由 TOWER/CONNECTOR mapping。

## 与其它模块/系统配合

- **LoRAModelManager**：`_adapter_manager`，承接激活/slot；见 [model-manager.md](model-manager.md)。
- **LoRAModel/PEFTHelper/LoRARequest**：加载三方；见 [lora-model.md](lora-model.md)、[peft-helper.md](peft-helper.md)、[request.md](request.md)。
- **utils.get_adapter_absolute_path**：HF/ModelScope 下载；见 [utils.md](utils.md)。
- **v1 LoRAModelRunnerMixin**：`LRUCacheWorkerLoRAManager` 实例由 `load_lora_model` 创建，`set_active_loras`/`add_lora`/`remove_lora`/`pin_lora` 全部委托它；见 [v1-integration.md](v1-integration.md)、[执行层-LoRA Mixin](../02-execution/worker/lora-mixin.md)。
- [执行层-Worker](../02-execution/worker/README.md)：worker 持有 manager 实例。

## 历史版本演进

- **v0.5（首版）**：`WorkerLoRAManager` + `LRUCacheWorkerLoRAManager`，`_load_adapter` 读 safetensors/tensorizer；`_apply_adapters` 增删差集。
- **v0.7（v1 接入）**：`create_lora_manager` 接 `vllm_config`；`max_position_embeddings` 用 `get_text_config()`；encoder-decoder 用 `max_target_positions`。
- **v0.9（PEFT mapper + skip）**：引入 `hf_to_vllm_mapper.get_unstacked_mapper()` 与 `lora_skip_prefixes`，支持 Qwen2VL 等多模态/重命名模型。
- **v0.10/main（load_inplace）**：`LRUCacheWorkerLoRAManager.add_adapter` 支持 `load_inplace`：先装后卸、允许临时超 capacity；`LoRAAdapterNotFoundError` 前置 FileNotFoundError 处理。
- **v0.11（EP）**：`from_local_checkpoint` 透传 `moe_ep_spec`；`is_3d_lora_weight` 盖章。

## 参见

- [← 返回 LoRA 首页](README.md)
- [model-manager.md](model-manager.md)
- [lora-model.md](lora-model.md)
- [peft-helper.md](peft-helper.md)
- [utils.md](utils.md)
- [v1-integration.md](v1-integration.md)
- [执行层-LoRA Mixin](../02-execution/worker/lora-mixin.md)
