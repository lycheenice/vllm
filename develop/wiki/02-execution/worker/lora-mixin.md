[← Wiki 首页](../../README.md) > [执行层](../README.md) > [Worker](./README.md) > LoRA Mixin

# LoRA Model Runner Mixin

源码：`vllm/v1/worker/lora_model_runner_mixin.py`（288 行）+ `vllm/v1/worker/gpu/lora_utils.py`（109 行，V2 专用）

## 是什么

`LoRAModelRunnerMixin` (`lora_model_runner_mixin.py:30`) 是注入到 `GPUModelRunner` 的横向 Mixin，把 LoRA 相关能力（加载 LoRA 模型、设置活跃 adapter、dummy LoRA 捕获、add/remove/pin/list adapter）从主 runner 中剥离。`GPUModelRunner` 通过 `class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin)` 多继承获得。

V2 路径下，`gpu/lora_utils.py` 提供 `LoraState` 状态对象 + 工厂函数（`get_lora_capture_cases` / `get_num_active_loras_for_dispatch` / `create_lora_capture_hook`），跑时调度由 `ModelCudaGraphManager` 与 `gpu/model_runner.py:execute_model` 主动调，不再依赖 mixin 里的 `set_active_loras`。

## 为什么

- **横向复用**：LoRA 在 GPU/TPU ModelRunner 行为一致， независим от KV/EC/具体模型，做成 Mixin 让任意 `GPUModelRunner` 子类（含 CPU/XPU 借父类继承）都能用同一份代码。
- **dummy 捕获**：cudagraph 捕获需要在各种 `num_active_loras` 下都预录一张图；mixin 提供 `maybe_dummy_run_with_lora` / `maybe_select_dummy_loras` 上下文管理器，自动构造 dummy LoRA 请求 + 循环分配 lora_id，覆盖 capture cases。
- **LRU 管理**：`LRUCacheWorkerLoRAManager` 实例化在 mixin 里，统一管理"加载/换出/激活"。
- **SGMV kernel 路径**：`_set_active_loras` 强制 `is_prefill=True`，让非 CUDA 平台也走 SGMV 内核（CUDA 上 prefill/decode 同内核，flag 被忽略）。

## 怎么做

### load_lora_model（`lora_model_runner_mixin.py:31`）

```python
if not supports_lora(model):
    raise ValueError(...)
self.lora_manager = LRUCacheWorkerLoRAManager(vllm_config, device, model.embedding_modules)
return self.lora_manager.create_lora_manager(model, vllm_config)
```

返回包了 LoRA 层的新 model，赋给 `self.model`。`LRUCacheWorkerLoRAManager` 由 [LoRA 子系统](../../12-lora/README.md) 提供。

### set_active_loras（`:73`）

入口（V1 路径）：

1. 从 `input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)` 拿到 `prompt_lora_mapping`/`token_lora_mapping`/`active_lora_requests`。
2. `_set_active_loras(prompt, token, requests, mapping_type=LANGUAGE)`：构造 `LoRAMapping(token_lora_mapping, prompt_lora_mapping, is_prefill=True, type=mapping_type)`，调 `lora_manager.set_active_adapters(requests, mapping)`。

### maybe_setup_dummy_loras（`:93`）

`contextmanager`：构造 `lora_config.max_loras` 个 dummy `LoRARequest`（`lora_path="/not/a/real/path"`），进入 `lora_manager.dummy_lora_cache()`，逐个 `add_dummy_lora(lr, rank=lora_warmup_rank)`（`max_lora_rank<8` 用 `max_lora_rank`，否则 8，但 `get_dummy_lora_warmup_rank` 可下调）。`__exit__` 时若 `remove_lora=True` 调 `remove_all_adapters`。

### maybe_select_dummy_loras（`:132`）

`contextmanager`，决定捕获时具体激活多少 LoRA：

- `num_active_loras == 0`：全 0 mapping。
- `num_active_loras > max_loras`：cycling `-1, 1, 2, ..., max_loras`（含 no-LoRA token，让 `prepare_tensors` 看到正确的 `effective_num_loras = max_loras+1`）。
- 否则：`1..effective_num_loras` 循环分配。

构造 `prompt_lora_mapping`（per request）、`token_lora_mapping`（per token）、`sample_lora_mapping`（per sampled token）；构造 dummy `lora_requests`；`_set_active_loras`。

### maybe_dummy_run_with_lora（`:236`）

把 `maybe_setup_dummy_loras` 与 `maybe_select_dummy_loras` 串起来：`with maybe_setup_dummy_loras(...), maybe_select_dummy_loras(...): yield`。供 `_dummy_run` 在每个 capture case 包一层。

### add/remove/pin/list_lora（`:274-288`）

透传给 `lora_manager`，每个先 `_ensure_lora_enabled()`（检查 `hasattr(self, "lora_manager")` 否则抛错）。

---

### V2：gpu/lora_utils.py（109 行）

V2 把状态收纳进 `LoraState`，逻辑改成纯函数：

- **`LoraState(max_num_reqs)`** (`:72`)：`lora_ids = np.zeros(max_num_reqs, int32)`（填 `NO_LORA_ID=0`）+ `lora_requests: dict[str, LoRARequest]`。方法：`add_request` / `remove_request` / `make_lora_inputs` / `get_activate_loras`。
- **`NO_LORA_ID = 0`** (`:17`)。
- **`get_lora_capture_cases(lora_config, compilation_config)`** (`:20`)：
  - `lora_config is None` → `[0]`。
  - `cudagraph_specialize_lora=True` + `specialize_active_lora=False` → `[0] + [c for c in get_captured_lora_counts(max_loras) if c > 0]`（2 的幂次 + `max_loras+1`）。
  - `cudagraph_specialize_lora=False` → `[0, max_loras+1]`（只捕两图：无 LoRA 与 满载）。
- **`get_num_active_loras_for_dispatch(lora_config, lora_state, req_ids, dummy_run)`** (`:39`)：dummy_run 时返回 `max_loras+1`，否则 `len(lora_state.get_activate_loras(req_ids))`。
- **`create_lora_capture_hook(lora_config, runner)`** (`:53`)：返回 hook，在每次 cudagraph 捕获前调 `runner.maybe_select_dummy_loras(lora_config, num_scheduled, num_active_loras=...)`，让 ModelCudaGraphManager 驱动 LoRA 特化捕获。

V2 `execute_model` 用法（`gpu/model_runner.py:1141`）：

- `num_active_loras = get_num_active_loras_for_dispatch(...)` 算 dispatch 维度。
- `dispatch_cg_and_sync_dp(..., num_active_loras=...)` → `BatchExecutionDescriptor(num_active_loras)`。
- 若 LoRA 启用：`lora_inputs = lora_state.make_lora_inputs(...)` → `self._set_active_loras(*lora_inputs)`（mixin 方法）。

## 与其它模块/系统配合

- [GPU Model Runner V1](gpu-model-runner.md) / [V2](model-runner-v2.md)：多继承注入；`set_active_loras` 在每步 `execute_model` 准备输入后调用。
- [cudagraph 捕获/重放](cudagraph-capture.md)：`maybe_dummy_run_with_lora` 包 `_dummy_run` 让每个 `num_active_loras` 都有一张图；`get_lora_capture_cases` 给 V2。
- [LoRA 子系统](../../12-lora/README.md)：`LRUCacheWorkerLoRAManager`、`LoRAMapping`、SGMV/Punica 内核。
- [引擎核心](../../01-engine-core/README.md)：`Executor.add_lora`/`remove_lora`/`pin_lora`/`list_loras` 最终透传到 mixin。
- [Worker 基类](worker-base.md)：`WorkerWrapperBase.__getattr__` 让 `add_lora` 等直接打到 model_runner。
- [多模态](../../11-multimodal/README.md)：`gpu/mm/lora.py:set_active_mm_loras` 在 V2 中给 encoder 也设 LoRA。
- [编译子系统](../../09-compilation-ir/README.md)：`cudagraph_specialize_lora` 配置项决定捕获 cases。

## 历史版本演进

- **v0.5–v0.6**：V0 已有 LoRA manager，但与 ModelRunner 紧耦合。
- **v0.7.0**：V1 引入 `LoRAModelRunnerMixin`，把 LoRA 逻辑从主 runner 抽出；`LRUCacheWorkerLoRAManager` 落地。
- **v0.8.0**：`maybe_setup_dummy_loras` / `maybe_select_dummy_loras` 支持各种 `num_active_loras` 捕获。
- **v0.9.0**：`is_prefill=True` 强制让非 CUDA 平台走 SGMV；`LoRAMappingType.LANGUAGE` 与图文 LoRA 区分。
- **v0.10.0**：`get_captured_lora_counts` + `cudagraph_specialize_lora` 控制 capture cases；`specialize_active_lora` 细粒度选项。
- **v0.11–v0.12**：V2 `LoraState` + `get_lora_capture_cases` 抽到 `gpu/lora_utils.py`；`create_lora_capture_hook` 让 `ModelCudaGraphManager` 驱动捕获。
- **main**：`get_dummy_lora_warmup_rank` 让小 rank warmup 更快；`mm/lora.py` 让多模态 encoder 也参与 LoRA 切换。

[← 返回执行层首页](../README.md)

## 参见

- [GPU Model Runner V1](gpu-model-runner.md)
- [Model Runner V2](model-runner-v2.md)
- [cudagraph 捕获/重放](cudagraph-capture.md)
- [KV Connector Mixin](kv-connector-mixin.md)
- [EC Connector Mixin](ec-connector-mixin.md)
- [LoRA 子系统](../../12-lora/README.md)
