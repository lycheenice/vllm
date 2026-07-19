[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [OpenAI](README.md) > models

# models/serving.py（/v1/models + LoRA 适配器管理）

> `OpenAIServingModels` 是各 serving handler 共享的"模型注册表"：维护 base model 路径、静态/动态 LoRA 适配器列表、LoRA resolver，对外提供 `/v1/models`、`/v1/load_lora_adapter`、`/v1/unload_lora_adapter`。只读变体 `OpenAIModelRegistry` 供无引擎的 render-only server 使用。

## 是什么

| 类 | 位置 | 职责 |
|---|---|---|
| `OpenAIModelRegistry` | `vllm/entrypoints/openai/models/serving.py:31` | 只读：`base_model_paths`、`is_base_model`、`check_model`、`show_available_models`（仅 base） |
| `OpenAIServingModels` | `:83` | 带 engine_client + LoRA：`init_static_loras`、`load_lora_adapter`、`unload_lora_adapter`、`resolve_lora`、`show_available_models`（base+LoRA） |

`OpenAIServingModels.__init__`（`:92`）：
- 内含 `self.registry = OpenAIModelRegistry(...)`（`:101`）做只读部分。
- `self.static_lora_modules`、`self.lora_requests: dict[str, LoRARequest]`、`self.lora_id_counter = AtomicCounter(0)`。
- `self.lora_resolvers`：从 `LoRAResolverRegistry.get_supported_resolvers()` 取所有注册 resolver（`:114`）。
- `self.lora_resolver_lock: dict[str, Lock]`（`:118`）按 name 串行化 load。

`init_static_loras`（`:124`）：遍历 `--lora-modules` 列表，逐个 `LoadLoRAAdapterRequest` → `load_lora_adapter`，失败 raise `ValueError`（启动期硬失败）。

`load_lora_adapter`（`:167`）流程：校验请求（`_check_load_lora_adapter_request`，`:237`）→ 分配 `lora_int_id`（counter++）→ 构造 `LoRARequest` → `engine_client.collective_rpc("load_lora", ...)` 真正加载 → 进 `lora_requests` dict。

`unload_lora_adapter`（`:221`）：校验 → `engine_client.collective_rpc("unload_lora", lora_int_id)` → 从 dict 移除。

`resolve_lora`（`:282`）：当 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 开启，请求 `model` 命中某 resolver 名则按需 load（带 lock）。

`show_available_models`（`:149`）：base cards + 已加载 LoRA cards（`parent` 指向 base_model_name）。

## 为什么

- **base 与 adapter 一视同仁**：`/v1/models` 同时列出 base 与 LoRA adapter，客户端按 `model` 字段直接选 adapter，无需额外 API；`ModelCard.parent` 表述从属关系。
- **静态/动态双加载**：静态（启动期 `--lora-modules`，硬失败）保证部署确定性；动态（`/v1/load_lora_adapter` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING`）支持运行期增删，配合 resolver 按名解析。
- **只读 registry 复用**：render-only server 无 engine_client、不能加 LoRA，但需要 `is_base_model`/`show_available_models`，故抽 `OpenAIModelRegistry` 注入 `OpenAIServingModels`（`:101`），避免条件分支散落。
- **resolver 链**：`LoRAResolverRegistry` 提供多 resolver（如本地路径、HF repo、自定义），`resolve_lora` 顺序尝试，第一个成功即返回，便于多源 adapter 仓库。
- **load 串行化**：`lora_resolver_lock[name]` 防止同 adapter 并发 load 产生两个 int_id，浪费 slot。
- **集体 RPC 下发**：加载/卸载通过 `engine_client.collective_rpc("load_lora"/"unload_lora")` 广播到所有 worker，保证 TP/PP 一致（见 [LoRA-worker-manager](../../12-lora/worker-manager.md)）。

## 怎么做

### `/v1/models` 路由

`models/api_router.py:attach_router` 注册：

- `GET /v1/models` → `show_available_models`（base + LoRA cards）。
- `POST /v1/load_lora_adapter` → `load_lora_adapter`（body: `LoadLoRAAdapterRequest{lora_path,lora_name,is_3d_lora_weight}`）。
- `POST /v1/unload_lora_adapter` → `unload_lora_adapter`（body: `UnloadLoRAAdapterRequest{lora_name|lora_int_id}`）。

### 静态 LoRA 配置

CLI `--lora-modules name=path` 或 JSON `{"name":"...","path":"...","base_model_name":"...","is_3d_lora_weight":false}`（由 `LoRAParserAction` 解析，见 [cli-args.md](cli-args.md)）。`init_app_state` 调 `process_lora_modules` 合并 `default_mm_loras`，再 `await init_static_loras()`（`api_server.py:355`）。

### 请求路由

各 serving handler `_check_model`（`serve/engine/serving.py:40`）：先 `is_base_model` → 否则查 `lora_requests` dict → 否则（`VLLM_ALLOW_RUNTIME_LORA_UPDATING`）`resolve_lora` → 否则 404。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| OpenAIModelRegistry | `vllm/entrypoints/openai/models/serving.py:31` |
| check_model | `vllm/entrypoints/openai/models/serving.py:53` |
| OpenAIServingModels | `vllm/entrypoints/openai/models/serving.py:83` |
| init_static_loras | `vllm/entrypoints/openai/models/serving.py:124` |
| show_available_models | `vllm/entrypoints/openai/models/serving.py:149` |
| load_lora_adapter | `vllm/entrypoints/openai/models/serving.py:167` |
| unload_lora_adapter | `vllm/entrypoints/openai/models/serving.py:221` |
| _check_load | `vllm/entrypoints/openai/models/serving.py:237` |
| _check_unload | `vllm/entrypoints/openai/models/serving.py:261` |
| resolve_lora | `vllm/entrypoints/openai/models/serving.py:282` |
| models api_router | `vllm/entrypoints/openai/models/api_router.py:16` |

## 与其它模块/系统配合

- [api-server.md](api-server.md)：`init_app_state` 实例化并 `init_static_loras`。
- [engine-protocol.md](engine-protocol.md)：用 `ModelCard`/`ModelList`/`ModelPermission`。
- [chat-completion.md](chat-completion.md)/[responses.md](responses.md)：`_check_model` 路由 LoRA。
- [serve/lora-serves.md](../serve/lora-serves.md)：`/v1/load_lora_adapter` 路由 + `LoadLoRAAdapterRequest` 协议。
- [LoRA 子系统](../../12-lora/README.md)：`LoRARequest`、`LoRAResolver`/`LoRAResolverRegistry`、worker 端 `collective_rpc("load_lora")`。
- [serve/engine-serve.md](../serve/engine-serve.md)：`BaseServing._check_model` 复用 `models.lora_requests`。

## 历史版本演进

- **v0.5（LoRA 首版）**：静态 `--lora-modules`，`/v1/models` 内联在 api_server。
- **v0.7（动态 load/unload）**：加 `/v1/load_lora_adapter`、`/v1/unload_lora_adapter` + `VLLM_ALLOW_RUNTIME_LORA_UPDATING`；`AtomicCounter` 分配 int_id。
- **v0.9（resolver）**：引入 `LoRAResolver`/`LoRAResolverRegistry`，按 name 解析多源 adapter；resolver lock 串行化。
- **v0.10（OpenAIModelRegistry）**：拆只读 registry 支持无引擎 render-only server；`is_3d_lora_weight` 字段透传 MoE 适配器布局。
- **v0.11/main**：`default_mm_loras`（`lora_config.default_mm_loras`）合并进静态列表（`api_server.py:343`）；`process_lora_modules` 在 api_utils 统一处理；3D MoE LoRA 与 `enable_mixed_moe_lora_format` 协同（见 [LoRA-lora-model](../../12-lora/lora-model.md)）。

## 参见

- [← 返回 OpenAI 首页](README.md)
- [serve/lora-serves.md](../serve/lora-serves.md)
- [LoRA 子系统](../../12-lora/README.md)
- [engine-protocol.md](engine-protocol.md)
- [api-server.md](api-server.md)
