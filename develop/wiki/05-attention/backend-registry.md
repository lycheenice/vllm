# backend 注册表

[← Wiki 首页](../README.md) > [注意力](../README.md) > backend 注册表

> 源码：`vllm/v1/attention/backends/registry.py`

## 是什么

`registry.py` 用两个 Enum 把"backend 名字 → 实现类的全限定路径" centrally 登记起来，并提供运行时 override 机制：

- `AttentionBackendEnum`（`registry.py:34`）—— 标准注意力 + MLA + Sparse MLA + 杂项（CPU/Flex/HPC/TurboQuant），约 35 个成员。
- `MambaAttentionBackendEnum`（`registry.py:166`）—— SSM 族：MAMBA1 / MAMBA2 / SHORT_CONV / LINEAR / GDN_ATTN。

每个枚举成员的 value 是默认类路径字符串（如 `FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"`），可用 `register_backend()` 在运行时覆盖，支持装饰器与直接注册两种用法。

## 为什么

- **懒加载**：枚举 value 只是字符串，调用 `get_class()` 时才 `resolve_obj_by_qualname` 真正 import。这样启动期 selector 校验配置不会拖入所有 GPU kernel，也避免循环依赖。
- **可 override**：插件/三方 backend 通过 `register_backend(AttentionBackendEnum.CUSTOM, "my.mod.Backend")` 注入；`get_path()` 优先读 `_ATTN_OVERRIDES` 表。已存在的 backend 也能被子类覆盖（如 ROCm unified 复用 RocmAttentionBackend）。
- **友好报错**：`_AttentionBackendEnumMeta`（`registry.py:18`）重写 `__getitem__`，对未知名字列出全部合法成员。
- **两套独立 override 表**：`_ATTN_OVERRIDES` 与 `_MAMBA_ATTN_OVERRIDES` 分开，SSM 与标准注意力互不干扰；`is_mamba` 参数控制写入哪张表。

## 怎么做

### AttentionBackendEnum 成员分类

> 完整列表见 `registry.py:44-120`。下表按用途聚合。

| 分类 | 成员 | 默认实现路径 |
|------|------|------|
| 标准注意力 | `FLASH_ATTN`, `FLASH_ATTN_DIFFKV` | `backends/flash_attn.py`, `backends/flash_attn_diffkv.py` |
| 标准注意力 | `TRITON_ATTN`, `TRITON_ATTN_DIFFKV` | `backends/triton_attn.py`, `backends/triton_attn_diffkv.py` |
| 标准注意力 | `FLASHINFER`, `FLEX_ATTENTION`, `HPC_ATTN`, `TURBOQUANT` | `backends/flashinfer.py` 等 |
| 平台专用 | `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `CPU_ATTN` | `backends/rocm_attn.py` 等 |
| ML¬A dense | `TRITON_MLA`, `CUTLASS_MLA`, `FLASHMLA`, `FLASH_ATTN_MLA`, `FLASHINFER_MLA`, `TOKENSPEED_MLA`, `ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA` | `backends/mla/*.py` |
| MLA sparse | `FLASHMLA_SPARSE`, `FLASHINFER_MLA_SPARSE`, `FLASHINFER_MLA_SPARSE_SM120`, `FLASH_ATTN_MLA_SPARSE`, `ROCM_AITER_MLA_SPARSE`, `XPU_MLA_SPARSE` | `backends/mla/*_sparse*.py` |
| DeepSeek V4 模型驱动 | `FLASHMLA_SPARSE_DSV4`, `FLASHINFER_MLA_SPARSE_DSV4`, `ROCM_FLASHMLA_SPARSE_DSV4`, `MINIMAX_M3_SPARSE` | `vllm/models/deepseek_v4/...`, `vllm/models/minimax_m3/...` |
| 特殊 | `TORCH_SDPA`（仅 ViT，value 为空串）, `NO_ATTENTION`, `CUSTOM`（None，需注册） | — |

> 注：`NO_ATTENTION` 指向 `backends/no_attention.py`，但实际仓库中该文件**不存在**（待核实），可能由下游打包/插件提供，使用前需确认。

### MambaAttentionBackendEnum

| 成员 | 默认路径 |
|------|------|
| `MAMBA1` | `backends/mamba1_attn.Mamba1AttentionBackend` |
| `MAMBA2` | `backends/mamba2_attn.Mamba2AttentionBackend` |
| `SHORT_CONV` | `backends/short_conv_attn.ShortConvAttentionBackend` |
| `LINEAR` | `backends/linear_attn.LinearAttentionBackend` |
| `GDN_ATTN` | `backends/gdn_attn.GDNAttentionBackend` |
| `CUSTOM` | `None`（需注册） |

### 三大方法

```
backend.get_path(include_classname=True) -> str   # 类全限定路径，含 override
backend.get_class() -> type[AttentionBackend]     # 上面路径 resolve 出的类
backend.is_overridden() / clear_override()
```

`CUSTOM` 成员 value 为 `None`，`get_path` 遇到空串/None 会 `raise ValueError` 提示先 `register_backend`。

### register_backend 用法

`register_backend(backend, class_path=None, is_mamba=False)` 返回装饰器或 no-op（`registry.py:233`）：

```python
# 覆盖已有 backend
@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MyFlashAttn: ...

# 注册三方自定义
register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyBackend")

# Mamba 族
@register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
class MyLinear: ...
```

底层写 `_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"`。

## 与其它模块/系统配合

- **selector**：`get_attn_backend_cls` 返回的字符串本质是 `AttentionBackendEnum` 的类路径，selector 再 `resolve_obj_by_qualname`，见 [selector](selector.md)。
- **platform get_valid_backends**：遍历的就是 `AttentionBackendEnum` 列表，每个成员 `get_class()` 后 `validate_configuration`，见各平台文件。
- **MLA prefill registry**：MLA prefill 有自己**独立**的一套 enum/override（`backends/mla/prefill/registry.py:MLAPrefillBackendEnum` + `_MLA_PREFILL_OVERRIDES`），与本文件解耦，见 [MLA 总览](backends/mla/README.md)。
- **MLA indexer**：`DeepseekV32IndexerBackend` / `DeepseekV4IndexerBackend`（`backends/mla/indexer.py`）是独立 backend 但**未**进 `AttentionBackendEnum`，由模型层直接引用（待核实其注册路径）。
- **DeepSeek V4 模型驱动 backend**：`FLASHMLA_SPARSE_DSV4` 等成员 value 指向 `vllm/models/deepseek_v4/...`，说明 sparse MLA 的最终实现可由模型自带，而非通用 backends 目录。

## 历史版本演进

- **v0.6.x**：初版 `AttentionBackendEnum`，只有 FlashAttn/Triton/FlashInfer 等少量成员，硬编码路径。
- **v0.7.x**：抽 `register_backend` 装饰器 + override 表；CUSTOM 占位。
- **v0.8.x**：大批 MLA 成员加入（TRITON_MLA / CUTLASS_MLA / FLASHMLA / FLASH_ATTN_MLA / FLASHINFER_MLA / ROCM_AITER_MLA / ROCM_AITER_TRITON_MLA）。
- **v0.9.x**：Sparse MLA 成员（FLASHMLA_SPARSE / FLASHINFER_MLA_SPARSE / *_SM120 / ROCM_AITER_MLA_SPARSE / XPU_MLA_SPARSE）；DSV4 模型驱动成员；`MambaAttentionBackendEnum` 独立（之前与标准 backend 混在一起）。
- **v0.10.x（当前）**：`TOKENSPEED_MLA`；`HPC_ATTN`（腾讯 hpc-ops）；`TURBOQUANT`；`ROCM_AITER_UNIFIED_ATTN`；`_AttentionBackendEnumMeta` 友好报错；显式标注 value 为 `None` 避免 CUSTOM 与空串 backend 别名冲突。

---

[← 返回注意力首页](../README.md)

## 参见

- [selector 选择器](selector.md)
- [backend 抽象层](backend-abstraction.md)
- [backends 导航](backends/README.md)
- [MLA 总览](backends/mla/README.md)
