# Platform 抽象基类与枚举

[← Wiki 首页](../README.md) > [硬件平台](README.md) > Platform 抽象

源码：`vllm/platforms/interface.py`（约 1279 行）

## 是什么

`interface.py` 定义了 vLLM 的**硬件平台抽象根类型**，是整个平台子系统的"契约层"。所有具体厂商平台（CUDA/ROCm/TPU/XPU/CPU/Zen）都必须继承自这里的 `Platform`，并通过覆盖 `@classmethod` 注入各自的硬件特性。该模块不依赖任何厂商库，只 `import torch`，因此可以在任何环境中无副作用导入。

核心成员：

- `PlatformEnum`（`interface.py:67`）：枚举 `CUDA / ROCM / TPU / XPU / CPU / OOT / UNSPECIFIED` 七种平台身份。`is_cuda()`/`is_rocm()` 等 `is_*()` 谓词均据此判别。
- `CpuArchEnum`（`interface.py:79`）：CPU 架构枚举 `X86 / ARM / POWERPC / S390X / RISCV / OTHER / UNKNOWN`，由 `get_cpu_architecture()` 通过 `platform.machine()` 推断。
- `DeviceCapability`（`interface.py:89`）：`NamedTuple(major, minor)`，实现完整比较运算符（`< <= == >= >`、`__hash__`），并提供 `as_version_str()`（`"9.0"`）与 `to_int()`（`90`）两个换算方法，被 `has_device_capability` / `is_device_capability` / `is_device_capability_family` 用来做能力判断。
- `Platform`（`interface.py:134`）：抽象根类。约 80 个 `@classmethod`/属性，覆盖设备识别、能力探测、dtype/attention/编译/通信/内存/numa/sleep 等全部跨子系统 hook。`__getattr__`（`interface.py:1109`）做兜底转发：对未覆写的方法，自动代理到 `torch.<device_type>` 模块的同名属性。
- `UnspecifiedPlatform`（`interface.py:1277`）：当所有探针都失败时的占位实现，`_enum=UNSPECIFIED`、`device_type=""`，几乎所有方法走基类默认（多数是 `raise NotImplementedError`）。
- 模块级辅助：`set_assigned_physical_gpu_ids` / `get_assigned_physical_gpu_ids`（`interface.py:36`/`57`，支持不在 `CUDA_VISIBLE_DEVICES` 下的显式 logical→physical 映射，幂等、冲突即报错）；`in_wsl()`（`interface.py:61`，`functools.cache` 缓存）。

## 为什么

- **一处抽象，N 处适配**：vLLM 需要同时支持 NVIDIA/AMD/Intel/Google/纯 CPU，且每家又有不同架构代际。把这些差异集中到 `Platform` 上，让上层（编译、attention 选择、worker、distributed、KV 卸载）只通过 `current_platform.xxx()` 调用，避免在按厂商 `if/elif` 散落整个代码库。`__getattr__` 对未覆写方法的转发，让"先有最小实现、后续再补"成为可能：例如某平台只覆写了 `device_type`，`torch.cuda.is_available` 这类调用仍可通过代理拿到。
- **Stateless（无上下文初始化）**：`Platform` 的设计原则是**所有探针在初始化 CUDA/HIP context 之前就能工作**。`get_device_capability` / `get_device_name` 等返回 `classmethod`，让 Ray worker 在 `fork` 前就能查询设备信息（CUDA 走 NVML、ROCm 走 amdsmi，见各自页）。这点直接体现在 `set_assigned_physical_gpu_ids` 的注释"expected to run during single-threaded worker initialization"。
- **单一能力模型**：`DeviceCapability` 用 `(major, minor)` 表达架构代际，统一表达 Hopper=9.0、Blackwell=10.0/12.0、MI300X=gfx942→(9,4) 等异质信息（ROCm 通过 `_capability_from_gcn_arch` 把 `gfx942` 映射到 `(9,4)`，见 [rocm.md](rocm.md)）。`is_device_capability_family`（`interface.py:466`）特化"CUDA 13 family semantics"——例如 `family == 100` 匹配所有 `10.x`。
- **跨子系统注入点**：通过 `get_compile_backend` / `get_pass_manager_cls` / `get_static_graph_wrapper_cls` / `get_attn_backend_cls` / `get_device_communicator_cls` / `get_punica_wrapper` 等返回**字符串 qualname**，让上层用 `resolve_obj_by_qualname` 延迟导入，避免循环依赖和**导入期副作用**（参考 [`plugin-resolver.md`](plugin-resolver.md) 的懒加载说明）。
- **`__getattr__` 解耦**：基类默认未实现的方法不会立刻 `ImportError`，而是转发到 `torch.cuda`/`torch.xpu` 等模块。对 `__dunder__` 显式 `raise AttributeError`，避免 pickle 与 copy 把 `None` 当成真实方法。

## 怎么做

### 1. Platform 类的字段维度

`Platform` 的类属性构成"平台身份证"。子类必须覆盖前 4 项，其余有默认：

| 字段 | 含义 | 默认值 | 典型覆盖 |
|---|---|---|---|
| `_enum: PlatformEnum` | 平台身份 | —（必须） | `PlatformEnum.CUDA` |
| `device_name: str` | 短名（日志/配置） | — | `"cuda"` / `"rocm"` |
| `device_type: str` | `torch.<device_type>` 的 namespace | — | `"cuda"` / `"xpu"` / `"cpu"` |
| `dispatch_key: str` | PyTorch dispatch key | `"CPU"` | `"CUDA"` / `"XPU"` |
| `ray_device_key: str` | Ray accelerator key | `""`（不支持 ray） | `"GPU"` |
| `device_control_env_var: str` | 可见性环境变量 | 占位符 | `CUDA_VISIBLE_DEVICES` 等 |
| `ray_noset_device_env_vars: list[str]` | 阻止 Ray 设置可见设备的 env | `[]` | `["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]` |
| `simple_compile_backend: str` | 简单函数编译后端 | `"inductor"` | — |
| `dist_backend: str` | torch.distributed 后端 | `""` | `"nccl"` / `"gloo"` / `"xccl"` |
| `supported_quantization: list[str]` | 白名单 quant 方法 | `[]` | ROCm 列出 22 种 |
| `additional_env_vars: list[str]` | 该平台要继承给子进程的额外 env | `[]` | — |
| `_global_graph_pool: Any\|None` | 平台级 graph pool 缓存 | `None` | — |

### 2. 设备 ID 三层映射

`Platform` 把"逻辑设备 → 物理设备 → 可见设备"做成显式 API，避免子类各写一套：

```mermaid
flowchart LR
    L["vLLM 逻辑 ID<br/>(local_rank)"] -->|"device_id_to_physical_device_id<br/>(interface.py:275)"| P["物理设备 ID"]
    P -->|"logical_device_id_to_visible_device_id<br/>(interface.py:305)"| V["torch.device 可见序号"]
    V -.->|"visible_device_id_to_physical_device_id<br/>(interface.py:331) 反函数"| P
    E["device_control_env_var<br/>(CUDA_VISIBLE_DEVICES 等)"] -.-> P
    A["set_assigned_physical_gpu_ids()<br/>显式映射（最高优先）"-.-> P
```

优先级：`set_assigned_physical_gpu_ids` 显式映射 > `device_control_env_var` 环境变量 > 直接返回 `device_id`。三者一致性由调用方保证。

### 3. 配置生命周期 hook

`Platform` 暴露 4 个由 `VllmConfig` 初始化流水调用的钩子（顺序固定）：

1. `pre_register_and_update(parser=None)`（`interface.py:531`）：CLI 解析前。OOT 平台用来动态注册 quant config 等需要"先存在再解析"的资源。
2. `apply_config_platform_defaults(vllm_config)`（`interface.py:547`）：CLI 解析后、`VllmConfig` 校验前。注入平台默认（如 ROCm 的 `+sparse_attn_indexer` custom_ops）。
3. `check_and_update_config(vllm_config)`（`interface.py:561`）：校验并就地修改配置。例如 CUDA 把 `worker_cls: "auto"` 解析为 `vllm.v1.worker.gpu_worker.Worker`。
4. `update_block_size_for_backend(vllm_config)`（`interface.py:594`）：根据 attention backend 的 `get_preferred_block_size` 与 mamba/nvfp4 hybrid page 对齐调整 `block_size`、`mamba_page_size_padded`、`skip_page_size_padded`。内部根据 `model_config.is_hybrid`、`kv_cache_dtype_skip_layers` 分别走 `_align_hybrid_block_size`（`interface.py:750`）、`_align_heterogeneous_kv_block_size`（`interface.py:639`）。

### 4. 能力探测 API 矩阵

| 方法 | 返回 | 默认 |
|---|---|---|
| `get_device_capability(device_id=0)` | `DeviceCapability \| None` | `None` |
| `has_device_capability(cap, device_id=0)` | `bool`（`>=` 语义，可传 `(major, minor)` 或 int） | 基于 `get_device_capability` |
| `is_device_capability(cap, device_id=0)` | `bool`（`==` 语义） | 同上 |
| `is_device_capability_family(cap, device_id=0)` | `bool`（`major.x` 任意） | 同上 |
| `get_device_name(device_id=0)` | `str` | `raise NotImplementedError` |
| `get_device_uuid(device_id=0)` | `str`（PCI bus ID / UUID） | `raise NotImplementedError` |
| `get_device_total_memory(device_id=0)` | `int`（bytes） | `raise NotImplementedError` |
| `get_all_gpu_pci_bus_ids()` | `dict[int,str]` | `raise NotImplementedError`（用于 RDMA NIC 选择） |
| `num_compute_units(device_id=0)` | `int`（SM / CU / EU / 线程数） | `raise NotImplementedError` |

### 5. 子系统注入点一览

`Platform` 把跨子系统的注入点全部以"返回 qualname 字符串"形式表达，避免 import 期循环：

- **编译**：`get_compile_backend()`（`interface.py:248`，默认 `"inductor"`）、`get_pass_manager_cls()`（`interface.py:240`，默认 `PostGradPassManager`）、`get_static_graph_wrapper_cls()`（`interface.py:1140`，默认 `AbstractStaticGraphWrapper`）、`pass_key`（`interface.py:177`，默认 `"post_grad_custom_post_pass"`）、`get_default_ir_op_priority(vllm_config)`（`interface.py:1260`，默认 `["native"]`）、`support_static_graph_mode()` / `is_arch_support_pdl()`。
- **Attention**：`get_attn_backend_cls(selected, cfg, num_heads)`（`interface.py:363`）、`get_supported_vit_attn_backends()`（`interface.py:373`，默认 `[TORCH_SDPA]`）、`get_vit_attn_backend(...)`（`interface.py:379`）、`opaque_attention_op()`（`interface.py:1094`，是否把 attention 注册为单个不透明 op）。
- **通信**：`get_device_communicator_cls()`（`interface.py:1024`，默认 `DeviceCommunicatorBase`）、`use_all_gather()`、`use_custom_allreduce()`、`use_custom_op_collectives()`、`stateless_init_device_torch_dist_pg(...)`（`interface.py:1147`，无 CUDA init 的 PG 构造，给 Ray stateless worker 用）。
- **LoRA**：`get_punica_wrapper()`（`interface.py:996`）、`get_lora_vocab_padding_size()`（`interface.py:1017`，默认 `256`）。
- **Sleep / 内存**：`is_sleep_mode_available()`（`interface.py:224`，默认对 CUDA/ROCm/XPU `True`）、`is_cumem_allocator_available()`（`interface.py:231`，try import `vllm.device_allocator.cumem.cumem_available`）、`is_integrated_gpu()`（UMA，GH200/Jetson）、`get_current_memory_usage()`、`can_update_inplace()`、`get_nixl_supported_devices()` / `get_nixl_memory_type()`（KV 迁移/NIXL 集成）。
- **dtype**：`supported_dtypes`（属性，默认 `[bf16, fp16, fp32]`，**顺序即优先级**：第一个是 `auto` dtype 的回退值）、`supports_mx()` / `supports_fp8()` / `is_fp8_fnuz()` / `fp8_dtype()`、`check_if_supports_dtype(dtype)`、`get_infinity_values(dtype)`。
- ** kernels / IR**：`import_kernels()`（`interface.py:353`，默认 `import vllm._C` + `_moe_C_stable_libtorch`）、`import_ir_kernels()`（`interface.py:255`，默认 `import vllm.kernels`，OOT 平台覆盖以注入私有 kernel）。

### 6. `__getattr__` 兜底

```python
# interface.py:1109 简化
def __getattr__(self, key):
    if key.startswith("__") and key.endswith("__"):
        raise AttributeError(key)             # pickle 友好
    device = getattr(torch, self.device_type, None)
    if device is not None and hasattr(device, key):
        attr = getattr(device, key)
        if attr is not None:
            return attr                        # 代理到 torch.cuda / torch.xpu ...
    logger.warning_once(...)
    return None
```

## 与其它模块/系统配合

- **[plugin-resolver](plugin-resolver.md)**：`vllm/platforms/__init__.py` 通过 `resolve_current_platform_cls_qualname()` 返回的 qualname 实例化 `Platform` 子类，得到全局单例 `current_platform`；所有上层都 `from vllm.platforms import current_platform`。
- **[编译](../09-compilation-ir/README.md)**：`VllmBackend` 通过 `current_platform.get_pass_manager_cls()` / `get_static_graph_wrapper_cls()` / `pass_key` 注入厂商 Pass 管理器与 cudagraph 包装类（见 [`backends.md`](../09-compilation-ir/backends.md)）；`get_default_ir_op_priority()` 决定 `vllm_c` / `native` / `oink` / `aiter` / `xpu_kernels` 的优先级。
- **[分布式-通信](../07-distributed/device-communicators/README.md)**：`get_device_communicator_cls()` 返回 `CudaCommunicator` / `CpuCommunicator` / `XpuCommunicator`；`stateless_init_device_torch_dist_pg()` 让 Ray stateless worker 在无 init hook 下构造 NCCL PG。
- **[执行层](../02-execution/README.md)**：`check_and_update_config` 把 `worker_cls: "auto"` 解析为 `Worker` / `CPUWorker` / `XPUWorker`；`update_block_size_for_backend` 调整 `block_size`。
- **[模型执行-内核](../03-model-execution/kernels.md)**：`import_kernels()` 加载 `vllm._C` / `vllm._C_stable_libtorch` / `vllm._rocm_C` / `vllm._C_AVX512` 等扩展；`import_ir_kernels()` 注册 IR op 实现（vLLM IR 命名空间见 [`ir-op.md`](../09-compilation-ir/ir-op.md)）。
- **[配置-device](../10-config/device-config.md)**：`uses_host_device_handling()` 决定 `DeviceConfig.device` 是否留空（仅 TPU `True`）；`apply_config_platform_defaults` + `check_and_update_config` 是平台写回配置的唯二入口。
- **[KV 卸载-sleep](../15-kv-cache-offload/README.md)**：`is_sleep_mode_available()` / `is_cumem_allocator_available()` 决定能否走 [`device-allocator.md`](device-allocator.md) 的 sleep/wake 路径；`get_nixl_supported_devices()` / `get_nixl_memory_type()` 给 NIXL KV 迁移提供平台 hint。
- **[注意力](../05-attention/README.md)**：`get_attn_backend_cls` 与 `AttentionSelector` 配合做 backend 自动选择；`opaque_attention_op()` 决定 attention 是否注册成单个不透明 op（影响 torch.compile 的图切分）。
- **Ray**：`ray_device_key` / `device_control_env_var` / `ray_noset_device_env_vars` 决定 Ray 调度与可见设备控制；`set_assigned_physical_gpu_ids` 给 Ray DP / 弹性 EP 等显式映射逻辑设备编号。

## 历史版本演进

- **v0.5–v0.6（早期）**：`Platform` 抽象已在 `vllm/platforms/__init__.py`，但与具体厂商实现混在一起；`current_platform` 在模块导入期即解析（非懒加载），导致 OOT 插件无法 `from vllm.platforms import Platform` 后再被检测（待核实）。
- **v0.7（平台抽象独立化）**：`interface.py` 从 `__init__.py` 拆出，定义 `PlatformEnum`、`CpuArchEnum`、`DeviceCapability`；`__getattr__` 代理到 `torch.<device_type>` 让"最小实现可用"成为可能；`get_attn_backend_cls`/`get_device_communicator_cls` 改为返回 qualname 字符串以解耦 import。
- **v0.8（IR / sleep / NUMA 集成）**：引入 `import_ir_kernels()`（#38807）支持 OOT 平台私有 IR；`is_cumem_allocator_available()` 出现为 sleep mode 的前置检查；`num_compute_units()`（#35042）成为统一接口；`get_all_gpu_pci_bus_ids` 为 RDMA NIC 选择（#42083）铺路。
- **v0.9（多模型/hybrid/层重构）**：`pre_register_and_update` / `apply_config_platform_defaults` 拆分以适配 OOT 与多模型；`update_block_size_for_backend` + `_align_hybrid_block_size` 处理 attention+mamba hybrid page 对齐；`use_custom_op_collectives()`（#34760）让 platform 决定是否注册 `torch.ops.vllm.*` 自定义 collective。
- **v0.10（nvfp4/异质 KV/sleep 抽象）**：`_align_heterogeneous_kv_block_size` 处理 nvfp4 primary + skip layers 共享 block pool；`Plugable KVCacheSpec`（`register_custom_kv_cache_specs`，#37505）；`is_device_capability_family` 引入 CUDA 13 family semantics；`uses_host_device_handling` hook（#42313）让 DeviceConfig 留空。
- **v0.11 / v0.12 / main**：`SleepModeBackend` 抽象（见 [`device-allocator.md`](device-allocator.md) RFC #34303）让平台与 sleep 机制解耦；`get_default_ir_op_priority` 扩展支持 `oink` / `aiter` / `xpu_kernels` provider；`is_integrated_gpu` 通过 `torch.cuda.get_device_properties(...).is_integrated` 处理 GH200/GB200/Jetson（#35356）；`set_assigned_physical_gpu_ids` 显式映射支持 Ray DP 弹性 EP（#45026 后续）。部分版本归属（待核实）。

[← 返回硬件平台首页](README.md)

## 参见

- [plugin-resolver.md](plugin-resolver.md) — `current_platform` 如何懒加载与 OOT 插件机制。
- [cuda.md](cuda.md) / [rocm.md](rocm.md) / [tpu.md](tpu.md) / [xpu.md](xpu.md) / [cpu.md](cpu.md) / [zen-cpu.md](zen-cpu.md) — 各厂商 `Platform` 子类。
- [device-allocator.md](device-allocator.md) — `is_sleep_mode_available` 落地到 `MemAllocator` / `SleepModeBackend`。
- [platform-vs-allocator.md](platform-vs-allocator.md) — `Platform` 与 `device_allocator/` 的边界划分。
- [../09-compilation-ir/backends.md](../09-compilation-ir/backends.md) — `get_pass_manager_cls` 等注入点被 `VllmBackend` 消费。
- [../10-config/device-config.md](../10-config/device-config.md) — `DeviceConfig.device` 受 `uses_host_device_handling()` 控制。
