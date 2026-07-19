[← Wiki 首页](../README.md) > [LoRA](README.md) > v1 集成

# v1 LoRA 集成

> LoRA 子系统接入 v1 执行路径的三个接入点：ModelRunner Mixin（激活/加载钩子）、GPU lora_utils（cudagraph 捕获/派发）、GPU mm/lora（多模态 tower/connector 路由），以及 scheduler/前端对 `LoRARequest` 的路由。

## 是什么

| 文件 | 角色 |
|---|---|
| `vllm/v1/worker/lora_model_runner_mixin.py:30` | `LoRAModelRunnerMixin`：`GPUModelRunner` 的 LoRA 钩子混入 |
| `vllm/v1/worker/gpu/lora_utils.py:1` | cudagraph 捕获 cases、dispatch 计数、`LoraState`、capture hook |
| `vllm/v1/worker/gpu/mm/lora.py:13` | `set_active_mm_loras`：多模态 tower/connector LoRA 路由 |
| `vllm/v1/request.py:86` | `Request.lora_request` 字段 |
| `vllm/v1/engine/input_processor.py:146` | `_validate_lora` 校验启用 |
| `vllm/v1/core/sched/scheduler.py:630`+ | 调度期收集 `scheduled_loras`，限流 `max_loras` |
| `vllm/v1/engine/core.py:824` | `add_lora` 经 executor 广播 |

## 为什么

- **Mixin 复用**：LoRA 逻辑以 `LoRAModelRunnerMixin` 注入 `GPUModelRunner`，多后端（GPU/CPU/XPU/TPU）共享同一份激活/warmup 接口，详见 [执行层-LoRA Mixin](../02-execution/worker/lora-mixin.md)。
- **cudagraph 分桶**：`get_lora_capture_cases` 按是否 `cudagraph_specialize_lora` 生成 `[0] + active_counts`，dispatch 用 `num_active_loras` 选图，兼顾内存与变 LoRA 吞吐。
- **多模态独立路由**：vision tower/connector 有自己的 token 预算与 punica wrapper，不能与语言模型共用 mapping，故 `set_active_mm_loras` 按 `LoRAMappingType.TOWER/CONNECTOR` 单独下推。
- **调度期限流**：scheduler 在每步收集 `scheduled_loras`，超 `max_loras` 的请求延后，保证 batch 内 adapter 数 ≤ GPU slot 数。
- **前端校验前置**：`input_processor._validate_lora` 在入引擎前拒绝"未启用 --enable-lora 却带 lora_request"，快速失败。

## 怎么做

### 1. 请求绑定与校验

- API 层把 `LoRARequest` 传给 `LLM.generate`/`LLMEngine`（`vllm/v1/engine/llm_engine.py:224`），`async_llm.py:289` 异步入口同理。
- `InputProcessor.process_request`→`_validate_lora`（`vllm/v1/engine/input_processor.py:146`）：`lora_request` 非 None 时必须有 `lora_config`，否则 `ValueError`。
- `Request` 持有 `lora_request`（`vllm/v1/request.py:86`），`SchedulerOutput` 的 `CachedRequestData` 透传 `request.lora_request`（`vllm/v1/core/sched/output.py:61`）。

### 2. 调度期路由（`vllm/v1/core/sched/scheduler.py`）

- `:630`：构造 `request_lora_int_ids`（`req.lora_request.lora_int_id if >0 else -1`）。
- `:670-673`：调度时若某请求的 `lora_int_id` 不在已调度集合且 batch 内 `scheduled_loras` 满，则延迟该请求（避免超 slot）。
- `:981-982`：成功调度的请求把 `lora_int_id` 加入 `scheduled_loras`。
- `block_pool.py:346`：atomic block 复用时按 `lora_id`/`lora_name` 区分缓存前缀。

### 3. ModelRunner 钩子（`LoRAModelRunnerMixin`，`lora_model_runner_mixin.py:30`）

- `load_lora_model(model, vllm_config, device)`（`:31`）：建 `LRUCacheWorkerLoRAManager` 并 `create_lora_manager`，返回 LoRA 化后的 model。`GPUModelRunner.load_model` 在 `supports_lora(model)` 时调它。
- `set_active_loras(input_batch, num_scheduled_tokens, num_sampled_tokens, mapping_type)`（`:73`）：每步前调用。`input_batch.make_lora_inputs` 生成 `prompt_lora_mapping`/`token_lora_mapping`/`lora_requests`，`_set_active_loras`（`:48`）建 `LoRAMapping(is_prefill=True, type=...)` 调 `lora_manager.set_active_adapters`。`is_prefill=True` 强制非 CUDA 平台用 SGMV；CUDA 上 prefill/decode 同 kernel。
- `add_lora`/`remove_lora`/`pin_lora`/`list_loras`/`maybe_remove_all_loras`（`:269-287`）：委托 `lora_manager`。
- warmup 上下文管理器：`maybe_setup_dummy_loras`（`:93`）、`maybe_select_dummy_loras`（`:132`，按 `num_active_loras` 选 dummy 数量与是否含 -1 no-LoRA）、`maybe_dummy_run_with_lora`（`:236`，组合前两者）。

### 4. cudagraph 捕获与派发（`vllm/v1/worker/gpu/lora_utils.py`）

- `get_lora_capture_cases(lora_config, compilation_config)`（`:20`）：LoRA 关闭→`[0]`；`cudagraph_specialize_lora` 且 `specialize_active_lora`→`[0] + captured_counts`（2 的幂+max+1）； specialize 但非 active→`[0, max_loras+1]`；不 specialize→`[0, max_loras+1]`。
- `get_num_active_loras_for_dispatch(lora_config, lora_state, req_ids, dummy_run)`（`:39`）：非 dummy 取 `len(lora_state.get_activate_loras(req_ids))`；dummy 取 `max_loras+1`；关闭取 0。该值作 cudagraph 选择键。
- `create_lora_capture_hook(lora_config, runner)`（`:53`）：返回 hook，在每图捕获前用 `maybe_select_dummy_loras` 装填对应数量的 dummy LoRA，使捕获到的图与运行期 active 数一致。
- `LoraState`（`:72`）：持 `lora_ids` 数组 + `lora_requests` 字典；`add_request`/`remove_request`/`make_lora_inputs`/`get_activate_loras`。`make_lora_inputs` 由 `GPUModelRunner` 在 `prepare_inputs` 调用，产出 mapping。
- `NO_LORA_ID=0`（`:17`）：未绑定 LoRA 的请求占位。

### 5. 多模态 LoRA 路由（`vllm/v1/worker/gpu/mm/lora.py:13`）

`set_active_mm_loras(model, lora_manager, encoder_cache, req_id_to_index, lora_state, scheduled_encoder_inputs)`：
- 若无 encoder 输入 / 无 cache / 不支持 tower connector LoRA → 直接返回。
- 遍历 `scheduled_encoder_inputs`（每个图像/视频），取 `model.get_num_mm_encoder_tokens(pos_info.get_num_embeds())` 算该 modality token 数，按 `lora_id` 填 `prompt_lora_mapping`/`token_lora_mapping`，收集 `lora_requests`。
- `lora_manager.set_active_adapters(lora_requests, LoRAMapping(..., type=TOWER))`（`:58`）。
- 若模型有 connector 且 `get_num_mm_connector_tokens`：再用 `np.repeat(prompt_mapping, connector_tokens)` 构造 `LoRAMapping(type=CONNECTOR)`（`:86`），让 connector 层也按对应 adapter 跑 LoRA。

由 `GPUModelRunner` 在处理多模态 encoder 前调用（与语言模型 `set_active_loras` 分开），两套 mapping 经 `LoRAModelManager._set_adapter_mapping` 选不同 punica wrapper。

## 与其它模块/系统配合

- **WorkerLoRAManager**：mixin 持有的 `lora_manager` 就是 `LRUCacheWorkerLoRAManager`；见 [worker-manager.md](worker-manager.md)。
- **LoRAModelManager**：`set_active_adapters`→`set_adapter_mapping`→`update_metadata`；见 [model-manager.md](model-manager.md)。
- **LoRARequest**：身份载体；见 [request.md](request.md)。
- [执行层-Worker](../02-execution/worker/README.md)：`GPUModelRunner` 持有 mixin。
- [执行层-LoRA Mixin](../02-execution/worker/lora-mixin.md)：mixin 角色综述。
- [引擎-InputProcessor](../01-engine-core/input-processor.md)：`_validate_lora` 与请求构造。
- [09-编译与 IR](../09-compilation-ir/README.md)：`cudagraph_specialize_lora` 与 capture hook。
- [多模态](../11-multimodal/README.md)：`EncoderCache`/`mm_features`/tower/connector。
- [配置-LoRA](../10-config/lora-config.md)：`specialize_active_lora`/`enable_tower_connector_lora`。

## 历史版本演进

- **v0.5（v0 路径）**：LoRA 经 `vllm/engine/llm_engine.py` 老 runner 接入，单后端，无 cudagraph 分桶。
- **v0.7（v1 mixin 首版）**：引入 `LoRAModelRunnerMixin` + `LRUCacheWorkerLoRAManager`，把激活逻辑与 `GPUModelRunner` 解耦；`set_active_loras` 每步调。
- **v0.9（cudagraph specialize）**：`get_lora_capture_cases`/`get_num_active_loras_for_dispatch`/`create_lora_capture_hook`/`LoraState` 落地，支持按 active 数分桶捕获；`maybe_select_dummy_loras` 含 no-LoRA 模拟。
- **v0.10（多模态 LoRA）**：`v1/worker/gpu/mm/lora.py` + `set_active_mm_loras` + `LoRAMappingType.TOWER/CONNECTOR`；`enable_tower_connector_lora` 实验性。
- **v0.11/main**：`is_prefill=True` 统一 CUDA kernel；`load_inplace` 经 `add_lora`；scheduler 限流细化；dummy warmup rank 对齐 fully_sharded MoE。

## 参见

- [← 返回 LoRA 首页](README.md)
- [worker-manager.md](worker-manager.md)
- [model-manager.md](model-manager.md)
- [request.md](request.md)
- [punica.md](punica.md)
- [执行层-Worker](../02-execution/worker/README.md)
- [执行层-LoRA Mixin](../02-execution/worker/lora-mixin.md)
- [引擎-InputProcessor](../01-engine-core/input-processor.md)
- [多模态](../11-multimodal/README.md)
- [配置-LoRA](../10-config/lora-config.md)
