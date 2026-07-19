# 配置基础设施（utils.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > 配置基础设施

源码：`vllm/config/utils.py`（约 487 行）。本模块是整个配置子系统的"地基"，提供 `@config` 装饰器、`replace`/`update_config` 不可变更新工具、`SupportsHash`/`SupportsMetricsInfo` 协议、以及用于 `compute_hash` 的一致性哈希工具链。所有 `vllm/config/*.py` 中的 dataclass 都经由这里的 `@config` 装饰为 Pydantic dataclass。

## 是什么

### `@config` 装饰器（`utils.py:52`）

```python
def config(cls=None, *, config: ConfigDict | None = None, **kwargs):
    """Decorator to create a pydantic dataclass with default config.
    The default config for the dataclass forbids extra fields.
    All config classes in vLLM should use this decorator.
    """
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)
    ...
    return dataclass(cls, config=merged_config, **kwargs)
```

- 既可裸用 `@config`，也可带参 `@config(config=ConfigDict(arbitrary_types_allowed=True))`。
- 默认 `extra="forbid"`：传入未声明字段时直接报错，避免配置漂移。
- 底层走 `pydantic.dataclasses.dataclass`，因此支持 `Field(default_factory=...)`、`model_validator`、`field_validator` 等 Pydantic 能力，同时仍保留 dataclass 的 `__post_init__`、`InitVar` 语义。
- 带 `@dataclass_transform(field_specifiers=(PydanticField,))`，让静态检查器把被装饰类识别为 dataclass，支持 `字段:` 自动补全。

### `replace`（`utils.py:119`）

```python
def replace(dataclass_instance, /, **kwargs) -> ConfigT:
    """Like dataclasses.replace, but compatible with Pydantic dataclasses
    which use pydantic.fields.Field instead of dataclasses.field"""
```

与标准库 `dataclasses.replace` 的差异：只取 **init 字段**（通过 `is_init_field`），再合并 `kwargs` 重建实例。这是因为 Pydantic 的 `Field(init=False)` 字段不能在构造时传入。`update_config` 与 `VllmConfig.with_hf_config` 都依赖它。

### `update_config`（`utils.py:230`）

```python
def update_config(config, overrides: dict[str, Any]) -> ConfigT:
    ...
    if is_dataclass(current_value) and not is_dataclass(value):
        value = update_config(current_value, value)  # 递归
    return replace(config, **processed_overrides)
```

支持嵌套字典递归覆盖：当某字段当前值是 dataclass 而传入值是 dict 时，递归下钻而非整体替换。这是 `-cc.pass_config.fuse_norm_quant=true` 这类"点分嵌套覆盖"得以成立的关键。

### 哈希工具链

| 名称 | 位置 | 作用 |
|---|---|---|
| `SupportsHash` | `utils.py:201` | `Protocol`，定义 `compute_hash(self) -> str`。每个影响编译图形状的配置类都实现它 |
| `SupportsMetricsInfo` | `utils.py:226` | `Protocol`，定义 `metrics_info() -> dict[str, str]`，供 Prometheus 指标打标 |
| `compute_hash_cached` | `utils.py:209` | 按 `id(config)` 缓存 `compute_hash()` 结果，避免每次 forward 重算 |
| `normalize_value` | `utils.py:250` | 把任意值规范为 JSON 可哈希形式（Enum→`(FQN, value)`、torch.dtype→字符串、dataclass→`(FQN, sorted items)`、Path→绝对路径、PretrainedConfig→`to_json_string()`，否则硬报错） |
| `get_hash_factors` | `utils.py:344` | 遍历 dataclass 字段（排除 `ignored_factors`），对每个值调 `normalize_value`，返回 `dict[str, object]` |
| `hash_factors` | `utils.py:365` | `sha256(json.dumps(items, sort_keys=True))`，生成最终 hex 摘要 |

### 其他工具

- `get_field`（`utils.py:83`）：从 dataclass 取某字段的 `default`/`default_factory`/`init`，供 `EngineArgs` 复用默认值；兼容 `pydantic.Field`。
- `is_init_field`（`utils.py:115`）：判断字段是否参与 `__init__`。
- `getattr_iter`（`utils.py:130`）：从对象上按多个候选名取属性，支持过期别名告警（用于读 `PretrainedConfig` 的别名）。
- `get_attr_docs`（`utils.py:160`）：用 AST 解析类源码，提取"字段赋值后紧跟的字符串字面量"作为字段 docstring，供 CLI 帮助/文档生成。
- `Range`（`utils.py:370`）：闭区间数据类，含 `is_single_size`/`__contains__`，被 `CompilationConfig.cudagraph_capture_sizes` 与动态投机调度区间使用。
- `handle_deprecated`/`get_from_deprecated_env_if_set`/`set_from_deprecated_env_if_set`（`utils.py:402` 起）：统一的"旧字段→新字段"与"旧环境变量→新字段"迁移工具，带版本化告警。

## 为什么

- **统一 Pydantic 校验 + dataclass 语义**：vLLM 配置既要走 CLI（JSON/dot-notation 覆盖），又要在 worker 间 pickle 传递。`@config` 让每个子配置都获得 Pydantic 的类型校验、`model_validator`/`field_validator` 钩子、`extra="forbid"` 防错，同时保留 `__post_init__`/`InitVar` 供"构造后派生"（如 `SchedulerConfig.__post_init__(max_model_len, is_encoder_decoder)`）。
- **不可变更新**：`replace`/`update_config` 让配置以"复制-改字段"方式演进（如 `VllmConfig.with_hf_config` 深拷贝 `model_config` 后 `replace`），避免共享可变状态导致的跨 worker 漂移。
- **编译缓存键**：`VllmConfig.compute_hash` 聚合各子配置的 `compute_hash()`，作为 `torch.compile` 缓存目录与 DP worker 配置一致性校验的键。`normalize_value` 硬报错策略（"宁可失败也不欠哈希"）确保缓存不会因未识别类型而误命中。
- **性能**：`compute_hash_cached` 按 `id()` 缓存，因 config 对象构造后不可变、长生命周期，可避免每个 forward 的 SHA-256 开销。
- **弃用治理**：`handle_deprecated` 系列把"旧名→新名"的迁移集中化，配合 `--show-hidden-metrics-for-version` 做 metrics 退役。

## 怎么做

### 定义新配置类

```python
from vllm.config.utils import config, get_hash_factors, hash_factors

@config(config=ConfigDict(arbitrary_types_allowed=True))  # 需 torch.dtype 等非标量时
class MyConfig:
    field_a: int = 0
    field_b: torch.dtype = torch.float16
    _derived: int = field(default=0, init=False)   # 派生字段

    def __post_init__(self):
        self._derived = self.field_a * 2

    def compute_hash(self) -> str:
        return hash_factors(get_hash_factors(self, ignored_factors={"_derived"}))
```

要点：
1. 类上加 `@config`，需要 `torch.dtype`/`PretrainedConfig` 等非 Pydantic 原生类型时加 `arbitrary_types_allowed=True`。
2. 用 `Field(default=..., gt=0)` 做约束；用 `field(default=False, init=False)` 声明派生字段。
3. 重的校验/派生放 `__post_init__`；跨字段校验用 `@model_validator(mode="after")`；单字段规范用 `@field_validator(mode="before"|"after"|"wrap")`。
4. 凡是影响编译图形状的字段，要在 `compute_hash` 中纳入（或显式加入 `ignored_factors` 排除）。

### 嵌套覆盖

```python
vllm_config = update_config(vllm_config, {
    "compilation_config": {"pass_config": {"fuse_norm_quant": True}}
})
# 等价于：递归 replace，保留 pass_config 其余字段
```

### 取字段的默认值（EngineArgs 复用）

```python
sched_field = get_field(SchedulerConfig, "max_num_seqs")
# 返回 dataclasses.field(default=128, ge=1)，供 EngineArgs 透传
```

### 哈希一个配置

```python
from vllm.config.utils import compute_hash_cached
key = compute_hash_cached(vllm_config.cache_config)
```

## 与其它模块/系统配合

- **所有 `vllm/config/*.py` 子配置**：都经 `@config` 装饰，并在 `VllmConfig` 中聚合（见 [vllm-config.md](vllm-config.md)）。
- **`VllmConfig.compute_hash`（`vllm.py:383`）**：逐项调各子配置的 `compute_hash()`，再 `safe_hash` 聚合，作为编译缓存键。本模块的 `SupportsHash`/`normalize_value` 是其底层支撑。
- **`EngineArgs`（`vllm/engine/arg_utils.py`）**：用 `get_field`/`is_init_field` 把子配置默认值"提升"为 CLI 参数，`create_engine_config` 用 `replace`/`update_config` 把 CLI 覆盖打进 `VllmConfig`。
- **`set_current_vllm_config`（`vllm.py:2232`）**：把 `VllmConfig` 放入进程级全局变量，供 CustomOp/IR op 在 forward 时通过 `get_current_vllm_config()` 读到当前配置（见 [vllm-config.md](vllm-config.md)）。
- **编译子系统（`vllm/compilation/`）**：`CompilationConfig.compute_hash` 用 `get_hash_factors`/`hash_factors`，`Range` 被 `cudagraph_capture_sizes` 使用。
- **Prometheus 指标**：`CacheConfig.metrics_info()` 等 `SupportsMetricsInfo` 实现，被 `vllm/v1/metrics/` 读取。

## 历史版本演进

- **v0.5/v0.6（v0 时代）**：配置散落在 `EngineArgs` 与若干独立 dataclass（`ModelConfig`/`CacheConfig`/`ParallelConfig`/`SchedulerConfig`）中，无统一基类，用标准库 `@dataclass`。
- **v0.7（v1 落地）**：引入 `VllmConfig` 聚合体；开始用 Pydantic dataclass 提升校验；`compute_hash` 概念引入以支持 `torch.compile` 缓存（早期实现直接 `str(factors)` + sha256，无 `normalize_value` 规范化）。
- **v0.8**：`@config` 装饰器正式化，统一 `extra="forbid"`；`replace`/`update_config` 抽到 `utils.py`；`SupportsHash` Protocol 上线。
- **v0.9**：`normalize_value` 引入，对 Enum/dataclass/PretrainedConfig/Path 等统一规范，改为"未识别类型硬报错"以避免欠哈希；`compute_hash_cached` 加入以降低每步 forward 的哈希开销；`get_hash_factors`/`hash_factors` 取代散落的内联哈希逻辑。
- **v0.10/v0.11**：`handle_deprecated` 系列与 `get_from_deprecated_env_if_set` 加入，迁移治理（如 `dcp_kv_cache_interleave_size`→`cp_kv_cache_interleave_size`、`calculate_kv_scales` 弃用）。
- **v0.12 / main**：`IrOpPriorityConfig.compute_hash` 显式把 IR op 实现的 `uuid()` 纳入哈希因子（`utils.py` 的 `normalize_value` 已支持 `uuid()` 路径）；`PerformanceMode`/`OptimizationLevel` 体系成形；`Range` 用于动态投机解码批次区间。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — `VllmConfig` 复合体如何聚合所有子配置并驱动 `compute_hash`。
- [compilation-config.md](compilation-config.md) — `Range` 与 `get_hash_factors` 的主要消费方。
- [../17-utils-cross-cutting/README.md](../17-utils-cross-cutting/README.md) — `vllm/utils/hashing.py` 的 `safe_hash` 是本模块哈希的底层。
