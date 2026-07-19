# tiering/obj/ — 对象存储二级层

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > tiering-obj

源码目录：`vllm/v1/kv_offload/tiering/obj/`（2 文件 + `__init__.py`）。把 KV 块作为 S3 兼容对象存储 key 推/拉，用 [NIXL](../07-distributed/nixl-utils.md) 的 OBJ backend 做实际传输（CPU DRAM ↔ S3）。适用于跨地域 / 跨 K8s 集群共享 KV、单实例级别存储成本最低的场景。

```
obj/
├── __init__.py
├── config.py   # ObjStoreConfig dataclass
└── manager.py  # ObjectStoreSecondaryTierManager + ObjAsyncLookupManager
```

---

## 是什么

### 1. `ObjStoreConfig`（`config.py`）

```python
@dataclass
class ObjStoreConfig:
    bucket: str            # S3 bucket 名
    endpoint_override: str # 自定义 endpoint（MinIO/R2/CEPH 等）
    access_key: str
    secret_key: str
    scheme: str = "http"
    ca_bundle: str = ""

    def to_nixl_params(self) -> dict[str, str]: ...
```

`to_nixl_params` 把字段展开为 NIXL OBJ backend 期望的 string dict（`bucket/endpoint_override/scheme/access_key/secret_key/ca_bundle`），其中 `ca_bundle` 仅非空时带。

### 2. `ObjectStoreSecondaryTierManager`（`manager.py`）

继承 [tiering.md](tiering.md) `SecondaryTierManager`。构造参数（tier_config 给）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `store_config` | 必填 | dict，被 `ObjStoreConfig(**store_config)` 解析 |
| `prefix` | `""` | 对象 key 前缀（模拟"目录"路径） |
| `io_threads` | 4 | NIXL OBJ backend 工作线程数 |

#### 构造逻辑（`manager.py:93`）

1. `NixlWrapper("ObjAgent", nixl_agent_config(backends=[]))`：创建一个 NIXL agent，**不**注册额外 backend，仅用 `OBJ` 一个 backend。
2. `agent.create_backend("OBJ", {**obj_config.to_nixl_params(), "num_threads": str(io_threads)})`：实例化 OBJ backend。
3. `FileMapper.from_offloading_spec(root_dir=prefix+"/" or "", spec, parallel_agnostic=True)`：复用 [file-mapper.md](file-mapper.md) 的路径算法生成对象 key。
4. `_probe_connectivity()`：用 `__nixl_probe__/connectivity_test` 这个永不存在 key 跑一次 `_exists`，确认 bucket 可达；任何异常 raise `RuntimeError(...)`，让用户在启动时立即看到配置错。
5. 用 `ctypes.addressof` 把 `primary_kv_view` 基址转成数值，`agent.register_memory([(base_addr, nbytes, 0, "")], "DRAM")` 注册 CPU DRAM 内存。
6. `block_size_bytes = primary_kv_view.strides[0]`。
7. 一次性构造 `all_blocks = [(base_addr + i*stride, stride, 0) for i in range(len(view))]` 并 `agent.prep_xfer_dlist("NIXL_INIT_AGENT", all_blocks, "DRAM")`——这是 **local-side prepped descriptor list**，之后所有 transfer 复用它。

> NIXL `_dev_id`：CPU DRAM 永远 `0`；OBJ backend 用 devId 区分不同 obj_key，所以每次 transfer 用全局递增 `_next_obj_dev_id`（从 1 开始，0 留给 `_exists` 探针）。

#### `lookup`（`:225`）

委托给 `ObjAsyncLookupManager`：返回 `True/False/None` → `HIT/MISS/RETRY`。

#### `ObjAsyncLookupManager.batch_lookup`

构造探针 descriptor `(_PROBE_ADDR=0, _PROBE_LEN=1, _PROBE_DEV_ID=0, obj_key)`（即占位 addr/len），调 `agent.query_memory(descs, "OBJ", "OBJ")`：返回非 None 视为存在。一次 batch_lookup = 一次 NIXL 调用 = 一个 S3 RTT。

#### `submit_store` / `submit_load`（`:231` / `:237`）

转 `obj_keys = (file_mapper.get_file_name(k) for k in keys)` 然后 `_submit_transfer(job_id, block_ids, obj_keys, op)`，op 是 `"WRITE"` 或 `"READ"`。

#### `_submit_transfer`（`:170`）

1. `nixl_files = [(0, block_size_bytes, dev_id, key) for dev_id, key in enumerate(obj_keys, _next_obj_dev_id)]`：每个对象一个唯一 devId（OBJ backend 用 devId → obj_key 映射，重复会覆盖）。
2. `agent.register_memory(nixl_files, "OBJ")`：注册 OBJ 端的源/目标 descriptor。失败 push `JobResult(success=False)` 到 `_pending_results`。
3. `agent.prep_xfer_dlist("ObjAgent", files_desc.trim())`：构造 remote-side prepped descriptor list（peer agent name 是 `"ObjAgent"`）。
4. `agent.make_prepped_xfer(op, self._dram_prepped_handle, block_ids_list, obj_handle, list(range(len(nixl_files))))`：把 DRAM 端的 block_ids 与 OBJ 端的 descriptor 索引对齐成 transfer handle。
5. `agent.transfer(xfer_handle)`：发起异步传输；返 `"ERR"` 则失败入 `_pending_results`。
6. 成功则 `self._transfers[job_id] = TransferEntry(xfer_handle, files_desc, obj_handle)`。

#### 完成检测 `get_finished_jobs`（`:275`）

调 `_poll_active_transfers()`：遍历 `self._transfers`，对每个 entry 调 `agent.check_xfer_state(xfer_handle)`：

- `"PROC"` → 仍在飞，跳过。
- `"DONE"` → 成功。
- 其它 → 失败 + warning log。

完成的 entry 释放三件套（`release_xfer_handle` / `release_dlist_handle` / `deregister_memory`）并 append `JobResult` 到 `_pending_results`。最后返回并清空 `_pending_results`。

NIXL 只暴露 poll-based `check_xfer_state`，所以无 callback 机制——周期性 `get_finished_jobs` 是唯一驱动。

#### `drain_jobs`（`:282`）

`reset_cache` 调用，必须真正阻塞。实现：

```python
while self._transfers:
    self._poll_active_transfers()
    if not self._transfers: break
    if not warned and time.monotonic() - start > 5.0:
        logger.warning("still draining after 5s ...")
        warned = True
    time.sleep(0.001)
```

5 秒后 warning（不 abort——abort 会破坏 primary memoryview 一致性，由 [tiering.md](tiering.md) `reset_cache` 强约束）。

#### `shutdown`（`:306`）

释放所有 in-flight transfer handle + prepped DRAM handle + primary 注册，warning 容错每个释放异常。

#### `_probe_connectivity` + `_exists`

启动时跑一次 `_exists("__nixl_probe__/connectivity_test")`——`agent.query_memory` 探一个永不存在 key，能产 True/False 说明 bucket 可达；raise 说明凭据/endpoint 错。

---

## 为什么

- **极低存储成本 + 跨地域**：相比 FS（共享 PVC/本地盘）容量受限且单可用域；Obj tier 用 S3/MinIO/R2/CEPH 让 KV 跨地域、跨集群共享，单 GB 月成本最低。
- **NIXL OBJ backend 复用**：NIXL 已抽象 S3 客户端连接池、并发、断点续传；不重新发明轮子，且与 P2P tier 的 NIXL data transport 共享同一 wrapper（`vllm.distributed.nixl_utils.NixlWrapper`）。
- **batch_lookup 探针合并**：每 step 每 key 一次 S3 HEAD 请求太贵；NIXL `query_memory` 一次发整批 descriptor = 一次 RTT。
- **`_probe_connectivity` 立即失败**：Obj tier 配置错（bucket 拼写 / access_key 失效）只会在第一次 store 时显错；启动时探一下让用户得到早反馈。
- **DRAM prepped handle 一次构造**：local-side DRAM descriptor 复用避免每 transfer 重新注册大块内存。
- **`_next_obj_dev_id` 全局递增**：OBJ backend 用 `devId` 区分 obj_key；若复用 ID 后注册覆盖前注册会丢 descriptor——简单递增 ID 取代释放再分配的 ID 池管理。

---

## 怎么做

### 配置示例（MinIO）

```json
{
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 8589934592,
  "secondary_tiers": [
    {
      "type": "obj",
      "prefix": "prod-cluster-1",
      "io_threads": 8,
      "store_config": {
        "bucket": "vllm-kv",
        "endpoint_override": "minio.internal:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "scheme": "http"
      }
    }
  ]
}
```

### 跨集群共享

1. 同 model/`block_size`/dtype 的集群用同一 bucket + 不同 `prefix`（避免不同部署互踩）或同 `prefix`（共享）。
2. `PYTHONHASHSEED` 固定（同 [tiering-fs.md](tiering-fs.md)）。
3. 凭据通过 env / secret 注入而非明文 extra_config（推荐）。

### 调优

- `io_threads` 取 4–16——S3 client 通常自身已有连接池并发，过高反而引发限流。
- 跨区域Obj tier 必用 `ca_bundle` + `scheme=https`。
- Obj tier 适合"小时级温 KV 共享"场景（latency 10–100ms 级），不适合 prompt cache 的"每请求都查"热路径——把 P2P 或 FS 放在前面。

---

## 与其它模块/系统配合

- 编排：[tiering.md](tiering.md) `TieringOffloadingManager`。
- 对象 key 布局：[file-mapper.md](file-mapper.md) `FileMapper`，`prefix` 作为 `root_dir`。
- NIXL wrapper：[07-distributed/nixl-utils](../07-distributed/nixl-utils.md) `NixlWrapper` / `nixl_agent_config`。
- 与 P2P tier 共用 NIXL API（[tiering-p2p.md](tiering-p2p.md) 用 `DRAM`↔`DRAM`，本 tier 用 `DRAM`↔`OBJ`）。
- primary memoryview：[cpu.md](cpu.md) `SharedOffloadRegion.create_kv_memoryview()`。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| main | Obj tier 引入：`ObjectStoreSecondaryTierManager` + `ObjStoreConfig` + `ObjAsyncLookupManager`。`_probe_connectivity` 在启动时强校验。早期版本仅有 FS 与 P2P，无法满足跨地域 / 跨集群持久共享需求（待核实具体落地 PR） |
| 待核实 | 是否计划支持非 S3 兼容 backend（如 Azure Blob / GCS 原生）—— NIXL OBJ backend 当前接口偏 S3 API |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [tiering.md](tiering.md)：编排器与本 tier 的关系。
- [tiering-fs.md](tiering-fs.md)：本地盘二级层，设计选择对照。
- [tiering-p2p.md](tiering-p2p.md)：同样基于 NIXL，但是 DRAM↔DRAM RDMA。
