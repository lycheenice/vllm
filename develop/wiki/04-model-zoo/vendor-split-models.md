# 厂商分流模型（vllm/models/）

[← Wiki 首页](../README.md) > [模型库](../README.md) > **厂商分流**

> 源码：`vllm/models/`（顶层包，含 `deepseek_v32/`、`deepseek_v4/`、`minimax_m3/` 三个子包）。

---

## 是什么

`vllm/models/`（注意：不在 `model_executor/models/` 下）是 vLLM 为"重度依赖平台特化算子"的模型开辟的第二条实现路径。当一个模型架构（DeepSeek V3.2 DSA、DeepSeek V4 Sparse MLA、MiniMax M3 稀疏注意力）需要为 NVIDIA / AMD ROCm / Intel XPU 分别写不同的 attention、indexer、fused op 时，单文件无法承载，于是按厂商分目录拆开。

每个子包的 `__init__.py` 在 import 期通过 `vllm.platforms.current_platform` 判断硬件，挑出对应厂商的实现类 re-export。registry 把架构名指向 `vllm.models.<name>` 全限定路径（`registry.py:1393` `_resolve_module_name` 识别 `vllm.` 前缀）。

### 当前三个子包

| 子包 | 架构 | 平台分支 | 公开类 |
|---|---|---|---|
| `deepseek_v32/` | DeepSeek V3.2 DSA（含 GLM-5.2 复用） | 仅 `nvidia/`（SM100）；ROCm/XPU 直接 `NotImplementedError` | `DeepseekV32ForCausalLM`、`DeepseekV32MTP` |
| `deepseek_v4/` | DeepSeek V4 Sparse MLA + Compressor | `nvidia/`、`amd/`、`xpu/` | `DeepseekV4ForCausalLM`、`DeepSeekV4MTP`、`DSparkDeepseekV4ForCausalLM`（仅 NVIDIA）、`DeepseekV4FP8Config`（量化配置） |
| `minimax_m3/` | MiniMax M3 稀疏注意力（多模态版） | `nvidia/`、`amd/`，`common/` 共享 | `MiniMaxM3SparseForCausalLM`、`MiniMaxM3SparseForConditionalGeneration`、`MiniMaxM3MTP` |

### deepseek_v4 的内部布局（最复杂的样板）

```
vllm/models/deepseek_v4/
├── __init__.py            # 按 current_platform 选 nvidia/amd/xpu
├── quant_config.py        # DeepseekV4FP8Config：expert_dtype=fp4/fp8 分派
├── attention.py           # 公共 attention 结构
├── sparse_mla.py          # FlashMLA sparse 后端 metadata
├── compressor.py          # MLA Compressor（compress + quant + cache）
├── common/                # 跨平台共享
│   ├── rope.py            # build_deepseek_v4_rope（按 compress_ratio 选 theta）
│   └── ops/               # cache_utils、fused_compress_quant_cache、
│                          #   fused_indexer_q、fused_inv_rope_fp8_quant、
│                          #   fused_qk_rmsnorm、save_partial_states …
├── nvidia/
│   ├── model.py           # DeepseekV4ForCausalLM（SM100 MegaMoE + sparse MLA）
│   ├── mtp.py             # DeepSeekV4MTP
│   ├── dspark.py          # DSparkDeepseekV4ForCausalLM（DSpark 投机 draft）
│   ├── flashmla.py / flashinfer_sparse.py
│   └── ops/               # cutedsl 算子：dequant_gather_k、fused_indexer_q、
│                          #   o_proj、prepare_megamoe、sparse_attn_compress
├── amd/  (model.py、mtp.py、rocm.py)
└── xpu/  (model.py、mtp.py、xpu_sparse*.py、xpu_qnorm_rope_kv_fp8_insert.py)
```

---

## 为什么

- **平台特化算子无法跨平台复用**：DeepSeek V3.2 的 indexer 写到 SM100 的 WGMMA/TMA，DeepSeek V4 的 sparse MLA 用 cutedsl + FlashInfer/FlashMLA，AMD 则走 Triton/ROCM hip kernel，XPU 走 IPEX。一份代码加 `if platform:` 分支会让文件膨胀到无法维护，分目录才能让每个平台只 import 自己的算子（避免在 ROCm 上 import cutedsl 报错）。
- **避开 `vllm.model_executor.models.*` 扁平布局的导入副作用**：扁平目录里所有模型在 import 顶层 `__init__.py` 时会一同暴露，但 `vllm/models/` 顶层 `__init__.py` 是空的（见源码），主流程只在 registry 命中时才 import `vllm.models.deepseek_v4`，副作用最小。
- **共享部分仍可跨平台**：`common/` 子目录（rope 构造、fused triton ops、cache utils）跨厂商共用，避免重复。`common/ops/cache_utils.py` 这类纯 Triton kernel 在 NVIDIA/AMD 上都能跑。
- **量化配置类也要平台感知**：`DeepseekV4FP8Config`（`quant_config.py`）根据 `expert_dtype` 在 MXFP4 与 FP8 block 间分派，本身放在包顶层（不在任何厂商目录下），供 `quant_config` 解析路径直接 import。

---

## 怎么做

### 平台分支模板（deepseek_v4/__init__.py）

```python
from vllm.platforms import current_platform
from .quant_config import DeepseekV4FP8Config

if current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM
    from .amd.mtp import DeepSeekV4MTP
    DSparkDeepseekV4ForCausalLM = None               # DSpark 仅 NVIDIA
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM
    from .xpu.mtp import DeepSeekV4MTP
    DSparkDeepseekV4ForCausalLM = None
else:
    from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
```

mypy 看到的静态类型是 NVIDIA 分支；其他分支用 `# type: ignore[assignment]` 保持类型兼容。

### minimax_m3 的 TYPE_CHECKING 技巧

```python
if TYPE_CHECKING or not current_platform.is_rocm():
    from .nvidia.model import MiniMaxM3SparseForCausalLM, ...
else:
    from .amd.model import MiniMaxM3SparseForCausalLM, ...
```

静态检查器始终走 NVIDIA 分支（保证类型推断稳定），运行时 ROCm 自动切到 AMD。

### registry 的对接

```python
# registry.py:94 / :154 / :480 / :596 / :617
"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),
"MiniMaxM3SparseForCausalLM": ("vllm.models.minimax_m3", "MiniMaxM3SparseForCausalLM"),
"DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
"DeepSeekV4MTPModel": ("vllm.models.deepseek_v4", "DeepSeekV4MTP"),
"MiniMaxM3MTP": ("vllm.models.minimax_m3", "MiniMaxM3MTP"),
```

注意 `DeepseekV32ForCausalLM`（registry.py:93）别名指向 `deepseek_v2` 模块的 `DeepseekV3ForCausalLM`——因为 V3.2 在非 SM100 平台直接沿用 V3 实现；SM100 时由 `vllm.models.deepseek_v32` 提供特化版（需通过其他途径切换，见 `deepseek_v32/__init__.py` 的 `NotImplementedError` 守卫）。

### 缓存友好性

`_LazyRegisteredModel.inspect_model_cls`（`registry.py:929`）对 `vllm.model_executor.models.` 之外的模块用 `importlib.util.find_spec` 解析文件路径，把整个包内所有 `.py` 哈希进 cache key——保证 `vllm/models/deepseek_v4/` 下任何子文件改动都让 `modelinfos` 缓存失效。

---

## 与其它模块/系统配合

- **[registry.md](registry.md)**：`_resolve_module_name` 把 `vllm.` 前缀路径透传；`_LazyRegisteredModel` 的 find_spec 解析保证有效。
- **[08 硬件平台](../08-platforms/README.md)**：`current_platform.is_rocm()`/`is_xpu()` 决定分支；`current_platform.verify_model_arch` 在 `_try_load_model_cls` 里被调，例如 `deepseek_v32/__init__.py` 主动对非 SM100 抛 `NotImplementedError`。
- **[模型执行-层库](../03-model-execution/layers/README.md)**：`common/ops/` 里的 Triton kernel 与 `layers/fused_moe`、`layers/quantization.fp8/mxfp4` 协作；`DeepseekV4FP8Config` 继承 `Fp8Config` 并按 `expert_dtype` 切换 MoE method。
- **[注意力](../05-attention/README.md)**：`sparse_mla.py` 实现 `AttentionBackend` 子类（FlashMLA sparse），`compressor.py` 实现 MLA Compressor；`nvidia/flashmla.py` 与 `flashinfer_sparse.py` 是平台 attention kernel 入口。
- **[采样-投机](../06-sampling-decoding/speculative-decoding/README.md)**：`DSparkDeepseekV4ForCausalLM` 是 NVIDIA 专属 draft 模型（DSpark 投机解码）；`DeepSeekV4MTP` / `MiniMaxM3MTP` 是 MTP draft。
- **[分布式](../07-distributed/README.md)**：`DeepseekV4MixtureOfExperts`（`nvidia/model.py:1319`）实现 `MixtureOfExperts` 接口，供 EPLB / 专家迁移使用；MegaMoE 走 EP/TP 混合。
- **[多模态](../11-multimodal/README.md)**：`MiniMaxM3SparseForConditionalGeneration` 是 VLM 版本，`common/vision_tower.py` + `common/mm_preprocess.py` 提供多模态前处理；`common/indexer.py` 与 `common/sparse_attention.py` 在两个厂商分支里被复用。

---

## 历史版本演进

| 版本 | 变更 | 动机 |
|---|---|---|
| v0.10 以前 | 所有模型放在 `vllm/model_executor/models/` 扁平目录；DeepSeek V2/V3 单文件实现。 | 模型体量与算子复杂度尚可控。 |
| v0.10–v0.11 | DeepSeek V3.2 引入 DSA（MLA + lightning indexer），强依赖 SM100 WGMMA/TMA；仅 NVIDIA 可用，但平台分支开始出现。 | 单文件内 `if platform` 难维护。 |
| v0.11 | 引入 `vllm/models/` 顶层包；`deepseek_v32/__init__.py` 加 ROCm/XPU `NotImplementedError` 守卫；GLM-5.2（`GlmMoeDsaForCausalLM`）直接复用 V3.2 实现（registry.py:116 指向 `deepseek_v2`）。 | "一个 DSA 实现喂多个模型"。 |
| v0.11.5–v0.12 | DeepSeek V4（Sparse MLA + Compressor + MegaMoE + DSpark 投机）落地，三平台分支齐全；`common/ops/` 抽出跨平台 Triton kernel；`DeepseekV4FP8Config` 按 `expert_dtype=fp4/fp8` 分派 MXFP4 / FP8 block。 | V4 模型规模与算子复杂度上一个台阶，平台分支必备。 |
| main | MiniMax M3 入场（稀疏注意力 VLM），结构对齐 `deepseek_v4` 的 `nvidia`+`amd`+`common` 三分；`TYPE_CHECKING or not is_rocm()` 静态分支技巧定型。 | 稀疏注意力成为新主流，VLM 版本与纯文本版本共用 backend。后续新平台模型预计延续此模板（待核实）。 |

---

## 参见

- [← 返回模型库首页](../README.md)
- [`registry.md`](registry.md) — `vllm.` 全限定路径的解析
- [`architecture-families/deepseek.md`](architecture-families/deepseek.md) — DeepSeek V2/V3/V3.2/V4 家族文件清单
- [`architecture-families/asian-vendor.md`](architecture-families/asian-vendor.md) — MiniMax M2/M3、Kimi、Step3 等亚洲厂商模型
- [08 硬件平台](../08-platforms/README.md) · [05 注意力](../05-attention/README.md) · [采样-投机](../06-sampling-decoding/speculative-decoding/README.md)
