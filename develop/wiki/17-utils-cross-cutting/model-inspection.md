# 模型检查打印（model_inspection）

[← Wiki 首页](../README.md) > [工具与横切](README.md) > 模型检查

本页覆盖 `vllm/model_inspection.py`（136 行），提供把 `torch.nn.Module` 渲染成 transformers 风格层级树、并折叠同构层的工具。

## 是什么

公开入口：

- `format_model_inspection(model: nn.Module) -> str`（`vllm/model_inspection.py:134`）：返回模型的多行树形描述，类似 `print(model)` 但**折叠同构层**，例：
  ```
  (layers): ModuleList(
    (0-27, 29-47): 47 x LlamaDecoderLayer(...)
    (28, 48): 2 x DifferentDecoderLayer(...)
  )
  ```

内部辅助：

- `_get_module_info(module)`（`:8`）：单模块信息——类名 + 量化信息（`quant_method`/`scheme`，跳过 `Unquantized`）+ `extra_repr()`。
- `_get_child_signature(child)`（`:36`）：递归 `named_modules()` 拼签名，用于判定两子模块是否结构相同。
- `_format_index_ranges(indices)`（`:44`）：把索引列表折叠为区间串（`0-2, 4-6`）。
- `_format_module_tree(module, name, indent)`（`:61`）：核心递归——叶子输出 info；非叶子先输出 info 再递归子节点；把数字命名子节点（`"0"`/`"1"`）与命名子节点（`"norm"`）分流，数字子节点按结构签名分组，相同结构的连续/离散索引合并为 `N x` 行。

## 为什么

- **大模型可读打印**：上百层的 LLM 直接 `print(model)` 产生数千行噪声；折叠同构层后一眼看到"哪些层是普通 decoder、哪些被量化/替换"，便于排查权重加载与量化配置问题。
- **量化信息显式**：把 `quant_method`/`scheme` 类名嵌入描述，方便核对 GPTQ/AWQ/FP8 等是否生效。
- **零副作用**：纯只读，不改模型；可直接喂 logger。
- **服务于 `VLLM_LOG_MODEL_INSPECTION`**：引擎启动后可触发打印（见 [envs.md](envs.md)），帮助调试"为什么这个层没被量化 / 为什么 MoE 层路由异常"。

## 怎么做

```python
from vllm.model_inspection import format_model_inspection
logger.info(format_model_inspection(model))
```

- 签名基于 `named_modules` 全树，对超大模型有成本；仅在启动或调试时调用，不进热路径。
- 折叠粒度依赖"结构签名"完全一致——参数 shape 不同但模块结构相同的层会被合并（因其描述是结构而非数值）。

## 与其它模块/系统配合

- [模型执行](../03-model-execution/README.md)/[模型库](../04-model-zoo/README.md)：模型加载完成后调用，诊断权重映射/量化。
- [envs.md](envs.md)：`VLLM_LOG_MODEL_INSPECTION` 控制是否在启动期打印。
- [可观测性](../16-observability/README.md)：作为启动日志的一部分。
- 不参与编译/执行，纯展示用途。

## 历史版本演进

- **v0.9–v0.10**：随量化方案增多与模型层数膨胀，引入本工具以可读地诊断"哪些层被量化"。
- **v0.11**：加入 `VLLM_LOG_MODEL_INSPECTION` env 钩子，启动期自动打印。
- **main**：折叠/签名逻辑稳定；针对 CompressedTensors 的 `scheme` 显示与 `Unquantized` 跳过等细节完善（具体版本待核实）。

---

[← 返回工具与横切首页](README.md)

## 参见

- [envs.md](envs.md)（`VLLM_LOG_MODEL_INSPECTION`）
- [模型执行 · 量化](../03-model-execution/README.md)
- [可观测性](../16-observability/README.md)
