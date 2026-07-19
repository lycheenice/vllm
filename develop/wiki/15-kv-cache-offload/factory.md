# factory.py — OffloadingSpecFactory 注册表

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > factory

源码：`vllm/v1/kv_offload/factory.py`（73 行）。极简单的 spec 工厂：维护 `name → lazy loader` 字典，把 `VllmConfig` 转成具体的 `OffloadingSpec` 子类实例。

---

## 是什么

```python
class OffloadingSpecFactory:
    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(cls, name, module_path, class_name): ...
    @classmethod
    def get_spec_cls(cls, config) -> type[OffloadingSpec]: ...
    @classmethod
    def create_spec(cls, config, kv_cache_config) -> OffloadingSpec: ...
```

- `register_spec` 把 `name` 映射到一个**懒加载 loader**：调用 loader 时才 `importlib.import_module(module_path)` 并 `getattr(class_name)`，避免未使用 spec 的依赖被强制 import。
- `get_spec_cls` 解析顺序：先看 `extra_config["spec_name"]`（默认 `"CPUOffloadingSpec"`）是否在 registry；若不在再看 `extra_config["spec_module_path"]`，否则 raise。
- `create_spec` 直接 `spec_cls(config, kv_cache_config)` 实例化。

模块末尾预注册两个内置 spec（`factory.py:66`）：

```python
OffloadingSpecFactory.register_spec("CPUOffloadingSpec", "vllm.v1.kv_offload.cpu.spec", "CPUOffloadingSpec")
OffloadingSpecFactory.register_spec("TieringOffloadingSpec", "vllm.v1.kv_offload.tiering.spec", "TieringOffloadingSpec")
```

---

## 为什么

- **延迟 import**：`tiering/p2p/` 依赖 NIXL、`tiering/obj/` 依赖 S3 client——把这些放在显式 register_spec 的字符串路径里，避免单 spec 启动时强制拉全部依赖。
- **第三方扩展点**：插件只需调 `register_spec` 或在 `extra_config` 里塞 `spec_module_path` + `spec_name`（不需改 vLLM 源码）就能接入。
- **与 KVConnectorFactory 对齐**：vLLM 的其他可插拔子系统（attention backend、sleep mode backend、kv connector）都用同样的" name → lazy loader "模式。

---

## 怎么做

### 启用 tiering

`kv_connector_extra_config` 里设：

```json
{
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 10737418240,
  "secondary_tiers": [
    {"type": "fs", "root_dir": "/mnt/ssd/kv"},
    {"type": "p2p", "host": "0.0.0.0", "port": 7777}
  ]
}
```

### 自定义 spec

```python
# my_pkg/offload_spec.py
class MySpec(OffloadingSpec): ...

# 方式 A：在 vLLM 加载早期调用（plugin entry point）
OffloadingSpecFactory.register_spec("MySpec", "my_pkg.offload_spec", "MySpec")
# 然后 extra_config: {"spec_name": "MySpec"}

# 方式 B：完全外部，不注册
# extra_config: {"spec_name": "MySpec", "spec_module_path": "my_pkg.offload_spec"}
```

`register_spec` 不允许重复注册同名（`factory.py:23` raise `ValueError`）。

---

## 与其它模块/系统配合

- 调用方：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`（待核实具体行号，约在 spec 创建路径上）的 `OffloadingConnectorRoleBase` 创建 spec。
- 二级层工厂对照：[tiering.md](tiering.md) 的 `SecondaryTierFactory`——tier 自身有独立注册表，用于 `secondary_tiers` 列表里的 `type` 解析。
- 配置入口：[10-config/kv-transfer-config](../10-config/kv-transfer-config.md)。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.9 | 引入 `OffloadingSpecFactory`，仅注册 `CPUOffloadingSpec`。早期 spec 解析逻辑写在 connector 而非独立 factory |
| v0.10 | 注册 `TieringOffloadingSpec`；`get_spec_cls` 加入 `spec_module_path` fallback 支持外部 spec |
| main | 与 `SecondaryTierFactory`、`SleepModeBackendFactory` 保持同构；预期 plugin entry point `vllm.general_plugins` 将提供自动注册路径（待核实） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [base.md](base.md)：`OffloadingSpec` 抽象。
- [tiering.md](tiering.md)：`SecondaryTierFactory`（结构对称的二级层工厂）。
