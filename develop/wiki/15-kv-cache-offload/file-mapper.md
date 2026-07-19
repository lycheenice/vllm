# file_mapper.py — block-id ↔ 文件/对象路径

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > file_mapper

源码：`vllm/v1/kv_offload/file_mapper.py`（139 行）。把 `OffloadKey`（block hash + group_idx）映射成稳定的文件路径或对象 key，被 [tiering-fs.md](tiering-fs.md) 与 [tiering-obj.md](tiering-obj.md) 共享。映射设计目标是：**同一 model 配置、同一 token 内容 → 同一路径**，从而允许多个 vLLM 实例通过共享目录/PVC/S3 bucket 复用 KV。

---

## 是什么

### 路径布局

`FileMapper.get_file_name(key)` 输出（`file_mapper.py:112`）：

```
<base_path>_r<rank>/<hash_hex[:3]>/<hash_hex[3:5]>_g<group_idx>/<hash_hex>.bin
```

- 三级 hash 分桶 (`hash_hex[:3]` / `hash_hex[3:5]` / 完整 hex) 限制单目录 fan-out，避免百万级文件挤在一个目录里影响 `readdir` 与 `faccessat`。
- `_r<rank>` 让 multi-rank worker 各自写自己的目录（避免争用），但 `config.json` 共享。
- hash hex 来自 block 的内容 hash（scheduler 侧 `Request.block_hashes`）；内容由 `model_name + block layout + parallelism + dtype + ...` 共同决定。

### `<base_path>` 计算

`_compute_base_path(root_dir, fields)`（`file_mapper.py:128`）：

```
<root_dir>/<safe_model_name>_<sha256(canonical(fields))[:12]>
```

- `safe_model_name = model_name.replace("/", "_")`，让 HuggingFace ID（`org/model`）不变成嵌套目录。
- `canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))` → `sha256[:12]`。任意字段变化（block_size / dtype / tp / pp / pcp / dcp / kv_cache_groups）都会换目录，防止布局不兼容的实例互相污染。
- `_BASE_PATH_HASH_LEN = 12`：12 字符 hex 提供 48 bit 熵，对单 model 千实例级别足够避免碰撞。

### `from_offloading_spec` 与 `parallel_agnostic`

`FileMapper.from_offloading_spec(...)` 默认会把每个 worker 的 rank 等并行参数写进 fields。但 `parallel_agnostic=True` 时强制 `tp=pp=pcp=dcp=1, rank=0`，让不同并行度的实例共用同一目录。

只有满足**所有条件**才允许 `parallel_agnostic=True` 真正生效（`file_mapper.py:90`）：

1. 用户传 `parallel_agnostic=True`；
2. 未启用 `use_v2_model_runner`（V2 的 KV 布局尚未证明并行无关）；
3. 仅单个 KV cache group；
4. 该 group 的 spec 是 `FullAttentionSpec`（且**不是** `MLAAttentionSpec`——MLA 的 latent KV 是 per-rank 复制而非 head-shard，所以跨并行度不兼容）。

FS tier 与 Obj tier 都默认 `parallel_agnostic=True`（见 [tiering-fs.md](tiering-fs.md)、[tiering-obj.md](tiering-obj.md)）。`FileMapper.from_offloading_spec` 还接受 `gpu_blocks_per_file`（默认 1），决定一个 ".bin" 文件含几个 GPU 块——若 `block_size_factor>1`（offloaded 块 > GPU 块），`gpu_blocks_per_file` 一般设为 `block_size_factor` 让一个文件正好对应一个 offloaded 块。

### `config.json` 与跨进程一致性

`get_config_file_path()` 返回 `<base_path>/config.json`，内容是 `get_run_config()`（即 fields dict）。FS tier 在启动时若文件不存在则写入（`fs/manager.py:131`），让后续启动的实例能读到当前目录对应的 model 配置以便校验。

### 跨实例 hash 一致性

`store_block` 路径里的 hash 来自 vLLM 的 block content hash。Python 默认 `hash()` 带 `PYTHONHASHSEED` 随机化会影响某些哈希路径——`FileSystemTierManager` 的 docstring（`fs/manager.py:86-92`）明确要求多实例共享 `root_dir` 时必须固定 `PYTHONHASHSEED` 环境变量（如 `PYTHONHASHSEED=0`），否则 `NONE_HASH` 链式哈希种子会因进程而异，导致同样 token 在不同实例生成不同 `OffloadKey`，从而文件名不匹配。

---

## 为什么

- **跨实例共享 KV**：分布式推理、多副本 deployment、PVC 挂载同一目录的场景都需要"内容相同 → 路径相同"的稳定映射；hash + sha256(root-config) 双重 ID 兼顾内容寻址与配置版本隔离。
- **目录 fan-out**：百万级 block 全放在 `<base_path>_r<rank>/` 下会拖慢文件系统；3+2 字符两级 hash 分桶让单目录条目数控制在数千以内。
- **`parallel_agnostic` 安全阀**：跨并行度共享代价是必须保证 KV 字节布局完全一致——所以排除 MLA、排除 V2 model runner、要求 single group。
- **`config.json` 写盘**：跨实例 KV 共享时的 sanity check，避免不同模型实例往同一 hash 路径写不同字节。

---

## 怎么做

### 典型使用

```python
mapper = FileMapper.from_offloading_spec(
    root_dir="/mnt/ssd/kv",
    offloading_spec=spec,
    gpu_blocks_per_file=spec.block_size_factor,  # 一个文件 = 一个 offloaded 块
    parallel_agnostic=True,
)
# mapper.get_file_name(key) -> "/mnt/ssd/kv/Qwen2___5-7B_a1b2c3d4e5f6_r0/abc/de_g0/abcdef....bin"
# mapper.get_config_file_path() -> "/mnt/ssd/kv/Qwen2___5-7B_a1b2c3d4e5f6/config.json"
```

### 启用跨实例 FS 共享

1. 所有实例设置相同 `PYTHONHASHSEED`（如 `0`）。
2. 所有实例 `kv_connector_extra_config` 一致（`block_size` / `cpu_bytes_to_use` / 等），且 model、dtype、tp 等决定 fields 的字段相同。若 tp 不同则需 `parallel_agnostic=True` 且 model 满足前述条件。
3. `root_dir` 指向共享存储（NFS/PVC/shared host path）。

### 启用对象存储

`ObjStoreConfig`（[tiering-obj.md](tiering-obj.md)）的 `prefix` 字段会让 `FileMapper` 的 `root_dir` 变成 `{prefix}/`，最终对象 key 为 `<base_path>_r0/abc/de_g0/abcdef....bin`——S3 兼容存储以"前缀 + /"模拟目录。

---

## 与其它模块/系统配合

- 调用方：[tiering-fs.md](tiering-fs.md) 的 `FileSystemTierManager.__init__`、[tiering-obj.md](tiering-obj.md) 的 `ObjectStoreSecondaryTierManager.__init__`、[tiering-p2p.md](tiering-p2p.md) 的 `P2PSecondaryTierManager.__init__`（用 `FileMapper.get_run_config()` 计算 config fingerprint）。
- 上游：`KVCacheConfig`（提供 group / spec）、`VllmConfig`（提供 model/parallel/dtype）。
- block hash 来源：scheduler 侧 `Request.block_hashes` 经 `make_offload_key(block_hash, group_idx)` 拼成 `OffloadKey`（见 [base.md](base.md)）。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.10 | `FileMapper` 随 `tiering/fs/` 一起引入，原仅 FS tier 使用；hash hex 三级分桶 + `config.json` 写盘 |
| v0.10.x | 加入 `parallel_agnostic` 安全阀（最初实现较宽松，后续收紧到排除 MLA/V2） |
| v0.11 | P2P tier 复用 `FileMapper.from_offloading_spec(...).get_run_config()` 计算 `config_fingerprint`，作为 ZMQ 握手时的兼容性校验（`p2p/data/base.py:142`） |
| main | Obj tier 同样基于 `FileMapper` 形成 `{prefix}/.../_r0/.../<hash>.bin` 的对象 key 布局 |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [tiering-fs.md](tiering-fs.md)、[tiering-obj.md](tiering-obj.md)：实际使用 `FileMapper` 的两类 secondary tier。
- [tiering-p2p.md](tiering-p2p.md)：用 `get_run_config()` 做 config fingerprint。
