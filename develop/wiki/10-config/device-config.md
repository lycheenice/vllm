# DeviceConfig（device.py，已 deprecated）

[← Wiki 首页](../README.md) > [配置](../README.md) > DeviceConfig

源码：`vllm/config/device.py`（约 78 行）。`DeviceConfig` 描述 vLLM 执行所用设备类型。**该配置已 deprecated**：`device` 字段现由当前平台（`vllm/platforms/`）自动推导，用户无需也不应显式设置。保留仅为向后兼容。它是 `VllmConfig.device_config`。

## 是什么

`@config(config=ConfigDict(arbitrary_types_allowed=True))` 装饰（`device.py:16`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `device` | `"auto"` | 设备类型，`SkipValidation[Device | torch.device | None]`。**deprecated**，将自动按平台推导 |
| `device_type` | init=False | 由 `__post_init__` 从 `device`/平台推导出的字符串设备类型 |

`Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]`。

`__post_init__`（`device.py:49`）：
- `device == "auto"`：取 `current_platform.device_type`，失败抛 RuntimeError 提示开 `VLLM_LOGGING_LEVEL=DEBUG`。
- 否则按 `device` 是 `str`/`torch.device` 取 `device_type`。
- 若平台 `uses_host_device_handling()` 且 `device_type == current_platform.device_type`（如 CPU 平台需在 CPU 上处理输入），把 `device` 置 `None`；否则 `device = torch.device(device_type)`。

`compute_hash`（`device.py:30`）：返回空 factors 哈希——设备/平台信息由 torch/vllm 自动汇总，不进图形状指纹。

## 为什么

- **历史遗留**：v0 时代用户需要 `--device cuda` 显式选设备。v1 平台层（`vllm/platforms/`）成熟后，设备由 `current_platform` 单一推导，`DeviceConfig` 退化为派生容器。
- **保留理由**：`VllmConfig.device_config` 仍被部分代码读 `device_type`/`device`，删字段会触发大范围改动；故字段标 deprecated 但保留派生。
- **`SkipValidation`**：因 `torch.device` 非 Pydantic 原生，用 `SkipValidation` 关类型校验，`arbitrary_types_allowed=True` 容纳。

## 怎么做

- **不要显式设**：让 `device="auto"`，由 `current_platform` 推导。
- **读 `device_type`**：`vllm_config.device_config.device_type`（如 `"cuda"`/`"cpu"`/`"tpu"`/`"xpu"`）。
- **读 `device`**：通常为 `torch.device(device_type)`；CPU host-device 平台为 `None`。

## 与其它模块/系统配合

- **平台层（[`08-platforms/`](../08-platforms/README.md)）**：`current_platform.device_type` 是 `device_type` 的真相源；`uses_host_device_handling()` 决定是否置 `None`。
- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`current_platform.apply_config_platform_defaults`/`check_and_update_config` 可能进一步调整；`compute_hash` 中 `device_config.compute_hash()` 返回空（平台信息已由 torch 汇总）。
- **Executor / Worker（[`02-execution/`](../02-execution/README.md)）**：`device_type` 影响选 `GPUWorker`/`CPUWorker`/`XPUWorker`/`TPUWorker`；`device` 用于张量放置。
- **LoadConfig（[load-config.md](load-config.md)）**：`load_config.device` 默认回退到 `device_config.device`。

## 历史版本演进

- **v0.5/v0.6（v0）**：`DeviceConfig` 显式 `--device cuda/cpu/tpu`，用户必填或 `"auto"`。
- **v0.7/v0.8**：平台层引入并逐步接管设备推导；`device` 字段标注 deprecated；`device_type` 由 `current_platform.device_type` 派生。
- **v0.9–main**：保持 deprecated 状态；`compute_hash` 返回空；新增 `xpu` 到 `Device` Literal；部分平台（CPU host-device）置 `device=None` 行为成形。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [vllm-config.md](vllm-config.md) — 平台默认注入与 `compute_hash` 聚合。
- [load-config.md](load-config.md) — `load_config.device` 回退到本配置。
- [../08-platforms/README.md](../08-platforms/README.md) — `current_platform` 是设备真相源。
