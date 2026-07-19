# DiffusionConfig（diffusion.py）

[← Wiki 首页](../README.md) > [配置](../README.md) > DiffusionConfig

源码：`vllm/config/diffusion.py`（约 26 行，最小配置之一）。`DiffusionConfig` 描述离散扩散语言模型（dLLM，discrete diffusion language model）的 canvas 与去噪参数。dLLM 通过迭代去噪生成固定长度 canvas，而非左到右自回归；它**复用投机解码数据通路**（draft token ids / scheduled spec decode tokens）以块为单位生成。它是 `VllmConfig.diffusion_config`（`None` 表示非 dLLM），被 `VllmConfig.num_speculative_tokens` 属性与 scheduler/model runner 消费。

## 是什么

`@config` 装饰（`diffusion.py:10`）。

| 字段 | 默认 | 含义 |
|---|---|---|
| `canvas_length` | `None`(必填) | 去噪 canvas（块）长度；同时决定每步调度的"投机 token"数 |
| `max_denoising_steps` | `None`(→model `generation_config.json`) | 每 canvas 块最大去噪迭代数 |

> 无 `compute_hash`（`DiffusionConfig` 未定义，且 `VllmConfig.compute_hash` 中 **未** 纳入 `diffusion_config`，因 dLLM 去噪在采样层做事，不改前向图形状）。

`VllmConfig.num_speculative_tokens`（`vllm.py:504`）：若 `speculative_config` 未设但 `diffusion_config.canvas_length` 设，则返回 `canvas_length`——即 dLLM 把 canvas 长度当作"投机 token 数"复用调度通路。

`VllmConfig.use_v2_model_runner`：`model_config.is_diffusion=True` 时强制 V2 model runner（`vllm.py:534`）。

## 为什么

- **复用投机通路**：dLLM 的"块去噪"与投机解码的"批量 draft token + 验证"在数据结构上同构——都是"一次产出 N 个 token 槽位再校验/修正"。故 `DiffusionConfig` 复用 `num_speculative_tokens` 调度通路，避免为 dLLM 单建调度路径。
- **canvas_length = spec token 数**：`canvas_length` 既是去噪块长度，也是每步 token 槽位数，直接进入 `SchedulerConfig.max_num_scheduled_tokens` 推导（`VllmConfig._set_max_num_scheduled_tokens`）。
- **`max_denoising_steps` 模型默认**：从 `generation_config.json` 读，避免用户必填。
- **V2 强制**：dLLM 仅 V2 model runner 实现，故 `use_v2_model_runner` 检测 `is_diffusion` 强制 True。
- **不进哈希**：去噪迭代在采样层，不改前向图形状（与 `structured_outputs`/`reasoning` 等采样层配置一致）。

## 怎么做

- **启用 dLLM**：`--diffusion-config '{"canvas_length":128,"max_denoising_steps":64}'` 配合 dLLM 模型（如 MDLM/SunDAE 等）。
- **仅 canvas_length**：`--diffusion-config.canvas-length 128`，`max_denoising_steps` 从模型读。

## 与其它模块/系统配合

- **`VllmConfig`（[vllm-config.md](vllm-config.md)）**：`num_speculative_tokens` 把 `canvas_length` 作备选源；`use_v2_model_runner` 强制 V2；`_set_max_num_scheduled_tokens` 扣 drafter 槽。
- **SpeculativeConfig（[speculative-config.md](speculative-config.md)）**：二者互斥（dLLM 不用 spec decode）但共用 `num_speculative_tokens` 通路；`diffusion_config` 优先级低于 `speculative_config`。
- **SchedulerConfig（[scheduler-config.md](scheduler-config.md)）**：`num_speculative_tokens` 影响 `max_num_scheduled_tokens` 与调度预算切分。
- **ModelRunner（[`02-execution/worker/gpu-model-runner.md`](../02-execution/worker/gpu-model-runner.md)）**：V2 实现 dLLM 去噪 forward；`canvas_length` 决定 token 槽位形状。
- **ModelConfig（[model-config.md](model-config.md)）**：`is_diffusion` 属性触发本配置生效路径。

## 历史版本演进

- **v0.9/v0.10（待核实）**：`DiffusionConfig` 引入；dLLM 模型接入（MDLM 等）；"canvas = spec token"复用通路设计。
- **v0.11 / v0.12 / main**：`use_v2_model_runner` 强制 V2；`num_speculative_tokens` 备选源；与投机解码互斥关系明确。具体版本归属（待核实）。

[← 返回配置首页](../README.md)

## 参见

- [speculative-config.md](speculative-config.md) — 共用 `num_speculative_tokens` 通路（互斥）。
- [scheduler-config.md](scheduler-config.md) — `max_num_scheduled_tokens` 受 `canvas_length` 影响。
- [vllm-config.md](vllm-config.md) — `num_speculative_tokens`/`use_v2_model_runner` 联动。
- [../06-sampling-decoding/](../06-sampling-decoding/README.md) — 复用投机解码数据通路。
