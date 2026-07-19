[← Wiki 首页](../README.md) > [LoRA](README.md) > Request

# LoRARequest

> 一次推理请求所携带的 LoRA 适配器身份与定位信息，是整个 LoRA 子系统的"入场券"。

## 是什么

`LoRARequest` 定义于 `vllm/lora/request.py:8`，是一个 `msgspec.Struct`（`omit_defaults=True`、`array_like=True`），用于把"这个请求要用哪个适配器"从 API 前端一路传到 worker。

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `lora_name` | `str` | 适配器名，跨引擎唯一标识（用于相等性/哈希） |
| `lora_int_id` | `int` | 全局唯一整数 id，必须 > 0（`__post_init__` 校验） |
| `lora_path` | `str` | 本地路径或 HuggingFace/ModelScope repo id，不能为空 |
| `base_model_name` | `str \| None` | 基座模型名，供 resolver 使用 |
| `tensorizer_config_dict` | `dict \| None` | tensorizer 序列化加载配置 |
| `load_inplace` | `bool` | True 时强制重载，替换同 id 旧适配器（热更新） |
| `is_3d_lora_weight` | `bool` | MoE 适配器是否 3D 融合布局，仅在 `enable_mixed_moe_lora_format=True` 时生效 |

属性别名：`adapter_id`→`lora_int_id`、`name`→`lora_name`、`path`→`lora_path`（见 `vllm/lora/request.py:46-56`）。

## 为什么

- **msgspec 而非 dataclass**：`array_like=True` 让结构体以紧凑数组形式序列化，跨进程（EngineCore↔Worker）传递零拷贝、低开销；`omit_defaults=True` 省略默认字段减少传输量。
- **按 name 做相等/哈希**：`__eq__`/`__hash__`（`vllm/lora/request.py:58-73`）只看 `lora_name`，使 `LoRARequest` 可作集合/字典键，跨引擎识别同一适配器，即便 `lora_int_id` 不同。
- **整数 id 才是 slot 键**：`lora_int_id` 与 GPU slot、Punica 索引张量一一对应；name 用于人/解析层，int id 用于内核层，二者解耦。
- **热更新语义**：`load_inplace` 让同 id 适配器可在不重启引擎下替换权重（`LRUCacheWorkerLoRAManager.add_adapter` 据此先卸后装，见 `vllm/lora/worker_manager.py:285`）。
- **3D 布局声明**：`is_3d_lora_weight` 把磁盘布局透传给 `LoRAModelManager`，决定走 3D→2D 转换还是直接 2D 装载（`vllm/lora/model_manager.py:794`）。

## 怎么做

### 构造与校验

```python
req = LoRARequest(
    lora_name="my-adapter",
    lora_int_id=1,
    lora_path="/path/to/lora_or_hf_repo",
)
```

`__post_init__`（`vllm/lora/request.py:39`）会校验 `lora_int_id >= 1` 且 `lora_path` 非空，否则抛 `ValueError`/`AssertionError`。

### 提交路径

1. API 层把 `LoRARequest` 绑定到请求对象的 `lora_request` 属性。
2. 调度器把活跃请求的 `LoRARequest` 集合下传给 worker。
3. `WorkerLoRAManager.set_active_adapters`（`vllm/lora/worker_manager.py:193`）→ `_apply_adapters` 比对已加载集合，按 `lora_int_id` 增删。
4. `lora_int_id` 最终进入 `LoRAMapping.index_mapping`（token 级）与 `prompt_mapping`（请求级），经 `convert_mapping` 转成 slot index 灌入 Punica wrapper（`vllm/lora/punica_wrapper/utils.py:54`）。

### 相等性用途

因 `__eq__`/`__hash__` 按 name，`set[LoRARequest]` 会把同名请求去重；而 `lora_int_id` 由用户保证全局唯一（注释明确"currently not enforced in vLLM"，见 `vllm/lora/request.py:16`）。

## 与其它模块/系统配合

- **WorkerLoRAManager**：消费 `lora_int_id`/`lora_path` 做 load/apply；见 [worker-manager.md](worker-manager.md)。
- **PEFTHelper**：用 `lora_path` 读 `adapter_config.json`；见 [peft-helper.md](peft-helper.md)。
- **LoRAModel**：`lora_int_id` 成为 `LoRAModel.id`；见 [lora-model.md](lora-model.md)。
- **LoRAModelManager**：按 `lora_int_id` 查/激活/移除；见 [model-manager.md](model-manager.md)。
- **v1 cudagraph**：`LoraState` 用 `lora_int_id` 数组算 `num_active_loras` 选 cudagraph；见 [v1-integration.md](v1-integration.md)。
- [引擎-InputProcessor](../01-engine-core/input-processor.md)：前端绑定入口。

## 历史版本演进

- **v0.5（首版）**：`LoRARequest` 作为 dataclass 风格类出现，含 `lora_name`/`lora_int_id`/`lora_path`/`base_model_name`/`tensorizer_config_dict`。
- **v0.6+**：迁到 `msgspec.Struct`，获得 `array_like`/`omit_defaults`，按 name 的 `__eq__`/`__hash__` 落地，支持跨引擎去重。
- **v0.10（待核实）**：新增 `load_inplace` 字段，支持同 id 热替换，注释见 `vllm/lora/request.py:18`。
- **v0.11/main**：新增 `is_3d_lora_weight` 字段（`vllm/lora/request.py:31`），配合 `enable_mixed_moe_lora_format` 声明 MoE 适配器磁盘布局。

## 参见

- [← 返回 LoRA 首页](README.md)
- [worker-manager.md](worker-manager.md)
- [peft-helper.md](peft-helper.md)
- [lora-model.md](lora-model.md)
- [v1-integration.md](v1-integration.md)
