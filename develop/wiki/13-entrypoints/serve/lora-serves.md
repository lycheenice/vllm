[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > lora

# lora/（/v1/load_lora_adapter & /v1/unload_lora_adapter）

> `serve/lora/` 暴露运行期 LoRA 适配器增删 HTTP 端点，委托 `OpenAIServingModels.load_lora_adapter`/`unload_lora_adapter` 通过 `engine_client.collective_rpc` 广播到所有 worker。配合 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 使用。

## 是什么

| 组件 | 位置 | 职责 |
|---|---|---|
| `api_router.attach_router` | `vllm/entrypoints/serve/lora/api_router.py:26` | 注册 `/v1/load_lora_adapter`、`/v1/unload_lora_adapter` |
| `protocol.py` | `vllm/entrypoints/serve/lora/protocol.py` | `LoadLoRAAdapterRequest{lora_path,lora_name,is_3d_lora_weight}`、`UnloadLoRAAdapterRequest{lora_name|lora_int_id}` |

`load` handler 从 `app.state.openai_serving_models` 取对象，调 `load_lora_adapter(request, base_model_name)`；`unload` 类似。具体加载逻辑见 [openai/models.md](../openai/models.md)。

## 为什么

- **运行期热增删**：启动期 `--lora-modules` 之外，允许 HTTP 动态加/卸，支持"一基座多适配器"动态租户场景，无需重启。
- **协议独立**：把 LoRA 管理端点从 chat/completion decouple，单独路由，便于权限/审计控制。
- **与 `OpenAIServingModels` 解耦**：router 只做参数解析 + 委托，重逻辑在 models serving，避免重复。

## 怎么做

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_path":"/data/adapters/x","lora_name":"x"}'
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -d '{"lora_name":"x"}'
```

需服务启动时设 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`。

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| attach_router | `vllm/entrypoints/serve/lora/api_router.py:26` |
| LoadLoRAAdapterRequest | `vllm/entrypoints/serve/lora/protocol.py`（待核实行号） |
| load 实现 | `vllm/entrypoints/openai/models/serving.py:167` |
| unload 实现 | `vllm/entrypoints/openai/models/serving.py:221` |

## 与其它模块/系统配合

- [openai/models.md](../openai/models.md)：`OpenAIServingModels.load/unload_lora_adapter`。
- [LoRA 子系统](../../12-lora/README.md)：`LoRARequest`、worker `collective_rpc("load_lora")`。
- [cli-args.md](../openai/cli-args.md)：`--lora-modules` 静态加载互补。

## 历史版本演进

- **v0.7（动态 load/unload）**：HTTP 端点 + `VLLM_ALLOW_RUNTIME_LORA_UPDATING`。
- **v0.9（serve/lora 子包）**：从 api_server 抽到 `serve/lora/`。
- **v0.11/main**：`is_3d_lora_weight` 字段支持 MoE 3D 适配器；`UnloadLoRAAdapterRequest` 支持按 int_id 卸载。

## 参见

- [← 返回 serve/ 首页](README.md)
- [openai/models.md](../openai/models.md)
- [LoRA 子系统](../../12-lora/README.md)
