# ep_weight_filter：专家并行权重过滤

[← Wiki 首页](../../README.md) > [模型执行](../README.md) > [模型加载器](./README.md) > **EP 权重过滤**

> 源码：`vllm/model_executor/model_loader/ep_weight_filter.py`

---

## 是什么

`ep_weight_filter.py` 实现专家并行（Expert Parallelism, EP）下的**加载期权重过滤**：在 MoE 模型权重从磁盘读出之前，先判断每条权重 tensor 是否属于本 rank 该拥有的专家，不属于则直接跳过 `safe_open`/`get_tensor`，避免无用的存储 I/O。它不是独立 loader，而是被 `DefaultModelLoader` 与 `weight_utils.safetensors_weights_iterator` 复用的工具模块。

三个对外函数：

- `parse_expert_id(weight_name) -> int | None`：从权重名解析 expert id。
- `compute_local_expert_ids(num_experts, ep_size, ep_rank, placement) -> set[int] | None`：算本 rank 拥有的专家集合。
- `should_skip_weight(weight_name, local_expert_ids) -> bool`：决定是否跳过。

---

## 为什么

MoE 模型的专家权重通常占总权重 85–90%（文件头注释 `ep_weight_filter.py:8`）。EP 下每个 rank 只需自己的那份专家，但默认加载会读全量再丢，I/O 浪费严重。在读前过滤可把存储 I/O 近似降到 `1/ep_size`，对大 EP（如 EP=8/16）的加载时间是数量级提升。

不读 scale/metadata 张量（`should_skip_weight` 末尾 `if not weight_name.endswith(".weight"): return False`），因为 scale 极小且某些后端（如 FlashInfer NVFP4 全局 activation max）需要所有专家的 scale。

不处理 3D fused-expert 名（无数字 id，如 `.experts.gate_proj.weight`），因为这种布局下所有专家在同一 tensor 里，需整体加载后由 `RoutedExperts.weight_loader` 切，不能在这里跳。

---

## 怎么做

### expert id 解析

`_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")`（`ep_weight_filter.py:17`），匹配 `.experts.42.gate_proj.weight` 提取 `42`。不匹配 `.experts.gate_proj.weight`（3D fused）。

### 本地专家集合

`compute_local_expert_ids`（`ep_weight_filter.py:31`）镜像 `layers/fused_moe/layer.py::determine_expert_map` 的分布逻辑：

- `ep_size <= 1` → 返回 `None`（不过滤）。
- `placement="linear"`：连续分块，`base = num_experts//ep_size`，余数前几个 rank 各多 1。
- `placement="round_robin"`：交错，`set(range(ep_rank, num_experts, ep_size))`。

### 跳过判定

`should_skip_weight`（`ep_weight_filter.py:64`）：

```python
def should_skip_weight(weight_name, local_expert_ids) -> bool:
    if local_expert_ids is None:
        return False
    eid = parse_expert_id(weight_name)
    if eid is None:
        return False              # 非专家权重 / shared / 3D fused → 保留
    if not weight_name.endswith(".weight"):
        return False              # scale/metadata → 保留
    return eid not in local_expert_ids
```

### 接入点

1. `DefaultModelLoader._init_ep_weight_filter`（`default_loader.py:351`）在 `load_weights` 开头算 `self.local_expert_ids`，条件：`is_moe` + `enable_expert_parallel` + `enable_ep_weight_filter`，且 **`enable_eplb=False`**（EPLB 下冗余物理槽映射到别 rank 的逻辑专家，需全量权重）。
2. `ep_size = dp_size*pcp_size*tp_size`、`ep_rank = dp_rank*pcp_size*tp_size + pcp_rank*tp_size + tp_rank`（`default_loader.py:390`），与 `FusedMoEParallelConfig.make()` 一致。
3. `local_expert_ids` 传给 `safetensors_weights_iterator(..., local_expert_ids=...)`（`default_loader.py:291`），迭代器在 `safe_open`/`get_tensor` 前 `should_skip_weight`（`weight_utils.py:916`/`933`/`951`）。

---

## 与其它模块/系统配合

| 协作方 | 关系 |
|---|---|
| `DefaultModelLoader` | 调 `_init_ep_weight_filter` 计算 `local_expert_ids` |
| `weight_utils.safetensors_weights_iterator` | 逐 tensor 调 `should_skip_weight` |
| `layers/fused_moe/layer.py::determine_expert_map` | 分布逻辑需保持一致 |
| `vllm/distributed`（dp/tp/pcp group） | 提供各 rank |
| `config/ParallelConfig` | `enable_expert_parallel`/`enable_ep_weight_filter`/`enable_eplb`/`expert_placement_strategy` |
| `#07 分布式` | EP/EPLB 语义见 [`../../07-distributed/`](../../07-distributed/README.md) |

---

## 历史版本演进

| 时间锚 | 变更要点 |
|---|---|
| main（#37136） | 引入 EP weight filter，`should_skip_weight` 接入 safetensors 迭代器 |
| main（#37322） | Bugfix：EPLB 与 NVFP4 精度问题——EPLB 开启时跳过过滤；保留所有非 `.weight` 的 scale 张量 |
| main（#41184） | FusedMoE/MoERunner inversion refactor，分布逻辑与 `determine_expert_map` 对齐 |

---

## 参见

- [`default.md`](default.md) —— `_init_ep_weight_filter` 调用点
- [`weight-utils.md`](weight-utils.md) —— `should_skip_weight` 在迭代器中的位置
- [`../README.md`](../README.md) —— 返回模型执行首页
