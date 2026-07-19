[← Wiki 首页](../../README.md) > [API 入口](../README.md) > [serve/](README.md) > dev

# dev/（实验端点：cache/rlhf/rpc/server_info/sleep）

> `serve/dev/` 是一组实验/运维端点，仅在 `VLLM_SERVER_DEV_MODE` 下挂载（`serve/__init__.py:35`，启动时 logger.warning 提醒安全风险）。提供缓存重置、RLHF pause/resume/weight transfer、collective RPC 透传、server 环境信息、sleep/wake 等能力，主要服务 RL 训练、调试、节能场景。

## 是什么

| 子包 | 端点 | 位置 | 职责 |
|---|---|---|---|
| `cache/` | `POST /reset_prefix_cache`/`/reset_mm_cache`/`/reset_encoder_cache` | `vllm/entrypoints/serve/dev/cache/api_router.py:21/47/58` | 清 KV 前缀缓存/MM 缓存/编码器缓存 |
| `rlhf/` | `POST /pause_generation`/`/resume_generation`/`/is_paused` | `vllm/entrypoints/serve/dev/rlhf/api_router.py:30/75/95` | RLHF 训练期暂停/恢复推理 |
| `rlhf/` | `POST /init_weight_transfer_engine`/`/start_weight_update`/`/update_weights`/`/finish_weight_update`/`GET /get_world_size` | `:113/131/137/155/161` | 权重热更新（RLHF 训练同步权重） |
| `rpc/` | `POST /collective_rpc` | `vllm/entrypoints/serve/dev/rpc/api_router.py:24` | 透传任意 collective RPC 到 worker |
| `server_info/` | `GET /server_info` | `vllm/entrypoints/serve/dev/server_info/api_router.py:44` | 打印 vLLM env 与系统信息 |
| `sleep/` | `POST /sleep`/`/wake_up`/`GET /is_sleeping` | `vllm/entrypoints/serve/dev/sleep/api_router.py:22/33/46` | 让引擎睡眠（释放显存）/唤醒 |

所有端点都经 `engine_client` 依赖注入（各 `api_router.py:engine_client(request)`），调 `engine_client.collective_rpc(...)` 或专用方法（`pause_generation`/`sleep`/`wake_up`/`reset_*_cache`）广播到 EngineCore/worker。

`rlhf` 重量级流程（`:113` 起）：

1. `init_weight_transfer_engine`：建立权重传输引擎（`WeightTransferInitRequest`）。
2. `start_weight_update` → `update_weights`（`WeightTransferUpdateRequest`，可多次）→ `finish_weight_update` 完成一次权重迁移。
3. `get_world_size` 返回训练世界大小，便于对齐。

## 为什么

- **RL 训练闭环**：RLHF 需要 vLLM 在 roll-out 与训练间切换：`pause_generation` 停推理、`update_weights` 把新策略权重灌进 vLLM、`resume_generation` 继续采样。dev 端点把这套编排暴露给训练侧脚本。
- **权重大小协商**：`get_world_size` 让训练侧知道 vLLM 视角的世界大小，避免切分不一致。
- **调试便利**：`reset_*_cache` 在排查 KV/MM 缓存异常时可一键清；`collective_rpc` 透传让运维执行任意 worker 端方法；`server_info` 一键拿环境。
- **节能/多租**：`sleep`/`wake_up` 让空闲 vLLM 释放显存给同机其他负载，`is_sleeping` 供探针判断。
- **危险隔离**：这些端点会改引擎状态/权重/显存，生产误开极危险，故 dev mode + warning 双保险。

## 怎么做

```bash
# 启动时
VLLM_SERVER_DEV_MODE=1 vllm serve <model>

# RLHF 权重更新
curl -X POST .../pause_generation
curl -X POST .../init_weight_transfer_engine -d '{...}'
curl -X POST .../start_weight_update
curl -X POST .../update_weights -d '{...}'   # 多次
curl -X POST .../finish_weight_update
curl -X POST .../resume_generation

# 调试
curl -X POST .../collective_rpc -d '{"method":"...","args":...}'
curl -X POST .../reset_prefix_cache
curl .../server_info
curl -X POST .../sleep && curl .../is_sleeping && curl -X POST .../wake_up
```

### 关键代码导航

| 关注点 | 位置 |
|---|---|
| dev 路由聚合 | `vllm/entrypoints/serve/dev/__init__.py`（待核实） + `serve/__init__.py:35` |
| reset_prefix_cache | `vllm/entrypoints/serve/dev/cache/api_router.py:21` |
| pause/resume | `vllm/entrypoints/serve/dev/rlhf/api_router.py:30/75` |
| init_weight_transfer_engine | `vllm/entrypoints/serve/dev/rlhf/api_router.py:113` |
| update_weights | `vllm/entrypoints/serve/dev/rlhf/api_router.py:137` |
| collective_rpc | `vllm/entrypoints/serve/dev/rpc/api_router.py:24` |
| server_info | `vllm/entrypoints/serve/dev/server_info/api_router.py:44` |
| sleep/wake_up | `vllm/entrypoints/serve/dev/sleep/api_router.py:22/33` |

## 与其它模块/系统配合

- [openai/api-server.md](../openai/api-server.md)：`build_app` 在 `VLLM_SERVER_DEV_MODE` 调 `register_vllm_dev_api_routers`。
- [引擎核心-EngineCore](../../01-engine-core/engine-core-process.md)：`pause_generation`/`sleep`/`weight transfer` 由 EngineCore 执行。
- [07-distributed](../../07-distributed/README.md)：`WeightTransferInitRequest`/`WeightTransferUpdateRequest`（`vllm/distributed/weight_transfer/base.py`）。
- [15-kv-cache-offload](../../15-kv-cache-offload/README.md)：`reset_prefix_cache`。
- [多模态](../../11-multimodal/README.md)：`reset_mm_cache`/`reset_encoder_cache`。

## 历史版本演进

- **v0.8（sleep/wake + rpc）**：节能与调试端点首发。
- **v0.9（cache reset + server_info）**：缓存重置与环境信息。
- **v0.10（RLHF weight transfer）**：`init_weight_transfer_engine`/`start_weight_update`/`update_weights`/`finish_weight_update` 系列，支持 RL 训练在线权重同步；`pause/resume/is_paused`。
- **v0.10.x（elastic_ep 协同）**：与弹性 EP 扩缩容共用 weight transfer 基建（待核实）。
- **main**：`VLLM_SERVER_DEV_MODE` warning；server_info 缓存（`_get_system_env_info_cached`，`:39`）。

## 参见

- [← 返回 serve/ 首页](README.md)
- [openai/api-server.md](../openai/api-server.md)
- [07-distributed](../../07-distributed/README.md)
- [引擎核心-EngineCore](../../01-engine-core/engine-core-process.md)
