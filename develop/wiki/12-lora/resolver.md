[← Wiki 首页](../README.md) > [LoRA](README.md) > Resolver

# LoRAResolver & 注册表

> 按适配器名解析出 `LoRARequest` 的可扩展抽象，让 vLLM 从任意远端（S3、对象存储、内部仓库）按需拉取适配器。

## 是什么

`vllm/lora/resolver.py` 定义两件事：

1. `LoRAResolver`（`vllm/lora/resolver.py:14`）：ABC，唯一抽象方法 `async resolve_lora(base_model_name, lora_name) -> LoRARequest | None`。
2. `_LoRAResolverRegistry`（`vllm/lora/resolver.py:43`）：dataclass，持有 `resolvers: dict[str, LoRAResolver]`，提供 `register_resolver`/`get_resolver`/`get_supported_resolvers`。模块级单例 `LoRAResolverRegistry = _LoRAResolverRegistry()`（`vllm/lora/resolver.py:88`）对外暴露。

## 为什么

- **解耦命名与定位**：调用方只给 `lora_name`，resolver 负责把它解析成带 `lora_path`/`lora_int_id` 的 `LoRARequest`，使 vLLM 不必硬编码 HuggingFace 下载逻辑。
- **多源支持**：注册多个 resolver（按名区分），可对接企业内部模型仓库、私有 S3、缓存服务等。
- **异步**：`resolve_lora` 是 `async`，允许 resolver 内部做网络 IO 不阻塞引擎。
- **插件化**：第三方通过 `LoRAResolverRegistry.register_resolver(name, instance)` 注入实现，无需改 vLLM 主干。

## 怎么做

### 自定义 resolver

```python
class MyResolver(LoRAResolver):
    async def resolve_lora(self, base_model_name, lora_name):
        path = await fetch_from_blob(lora_name)
        if path is None:
            return None
        return LoRARequest(lora_name, alloc_id(), path, base_model_name)

LoRAResolverRegistry.register_resolver("my-store", MyResolver())
```

### 注册表行为

- `register_resolver`（`vllm/lora/resolver.py:51`）：重名会 `logger.warning` 并覆盖。
- `get_resolver`（`vllm/lora/resolver.py:71`）：未注册抛 `KeyError`，列出可用名。
- `get_supported_resolvers`（`vllm/lora/resolver.py:47`）：返回所有注册名。

### 调用点（待核实）

resolver 的具体调用接入点位于前端/引擎层；`vllm/lora/` 内部不直接调用 `resolve_lora`，而是由上层在收到请求时按 `base_model_name`/`lora_name` 触发解析，再把生成的 `LoRARequest` 注入请求。具体调用栈`（待补充）`。

## 与其它模块/系统配合

- **LoRARequest**：resolver 的产出物；见 [request.md](request.md)。
- **WorkerLoRAManager**：消费 resolver 产出的 `LoRARequest.lora_path` 加载；见 [worker-manager.md](worker-manager.md)。
- [引擎-InputProcessor](../01-engine-core/input-processor.md)：自然接入点是请求预处理阶段。

## 历史版本演进

- **v0.8（待核实）**：`LoRAResolver` ABC 与注册表引入，满足多租户按名解析需求。此前 `LoRARequest` 由调用方直接构造并指定 path。
- **main**：API 形态稳定；`_LoRAResolverRegistry` 用 dataclass + 模块单例，与 `vllm.config` 的插件式注册风格一致。具体引入版本`（待核实）`。

## 参见

- [← 返回 LoRA 首页](README.md)
- [request.md](request.md)
- [worker-manager.md](worker-manager.md)
