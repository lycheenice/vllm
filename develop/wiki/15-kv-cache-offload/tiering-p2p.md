# tiering/p2p/ — P2P RDMA 二级层

[← Wiki 首页](../README.md) > [KV 卸载](README.md) > tiering-p2p

源码目录：`vllm/v1/kv_offload/tiering/p2p/`（control/、data/、session/ 三个子包）。让一个 vLLM 实例（prefiller）把自己刚算出的 KV 块通过 RDMA 推送到另一个实例（decoder）的 CPU primary tier，使 decoder 无需重算 prefill。ZMQ 当控制面交换消息，NIXL（UCX/Mooncake/LIBFABRIC）当数据面做 RDMA 块拷贝。

```
p2p/
├── __init__.py
├── manager.py            # P2PSecondaryTierManager
├── control/
│   ├── __init__.py
│   ├── base.py           # ControlTransport / ControlConnection ABC
│   └── zmq.py            # ZmqTransport / ZmqConnection
├── data/
│   ├── __init__.py
│   ├── base.py           # DataTransport ABC + PollResult
│   └── nixl.py           # NixlTransport（基于 vllm.distributed.nixl_utils）
└── session/
    ├── __init__.py
    ├── protocol.py       # ConnectMsg/FetchMsg/TransferDoneMsg/Abort* 消息定义
    ├── session.py        # P2PSession（合并 client+server 双向）
    ├── client.py         # ClientRole（请求块拉取）
    └── server.py         # ServerRole（被请求块推送）
```

---

## 是什么

### 1. `P2PSecondaryTierManager`（`manager.py`）

继承 `SecondaryTierManager`。单 scheduler 线程驱动，所有 public 方法在 scheduler 进程调用。构造参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `host` | `"0.0.0.0"` | ZMQ ROUTER 绑定地址 |
| `port` | 7777 | ZMQ ROUTER 端口，**必须可被 peer 直连** |
| `backends` | None（→ `["UCX"]`） | NIXL transport backend 列表；含非 UCX 时走 `nixl_agent_config(backends=...)`，否则走 UCX-only + `num_threads` |
| `num_threads` | 4 | UCX-only 分支的 NIXL worker 线程数 |

构造（`manager.py:109`）：

1. `_local_id = f"{host}:{port}"`。
2. **数据面**：用 `FileMapper.from_offloading_spec(...).get_run_config()` 算出 `config_fields` 喂给 `NixlTransport`（用于 `config_fingerprint` 兼容性校验），实例化 `NixlTransport(local_id, primary_kv_view, config_fields=..., backends=..., num_threads=...)`。
3. **控制面**：`ZmqTransport(local_id, host, port)`。
4. `_sessions: dict[peer_id, P2PSession]` —— 一个 peer 一个 session。
5. `_kv_to_session: dict[kv_request_id, P2PSession]` —— FetchMsg 到达时绑定。
6. `_unbound_stores: dict[kv_request_id, list[_UnboundStoreBatch]]` —— prefiller 收到 store 但还没 peer 来 fetch 时暂存。
7. `_failed_req_ids: set[kv_request_id]` —— 失败的 ID，lookup 时直接 MISS。
8. `has_pending_work()` 永远返 `True`：保证 engine 持续 tick 控制 fd 与 session poll。

#### `lookup`（`manager.py:189`）

- 从 `req_context.kv_transfer_params["prefill"]` 取 `remote_host/remote_port/kv_request_id`。
- 任一字段缺失 → `MISS`（说明这个请求不是 P2P load 场景）。
- 在 `_failed_req_ids` → `MISS`。
- 否则 → `HIT`（P2P 假设 peer 一定有，由上层 affirm）。

#### `on_new_request`

若请求带 `prefill` 参数（decoder 侧），预创建一个 outbound session 指向 producer。prefiller 侧不主动建 session，等 decoder 连入。

#### `submit_store`（`manager.py:259`）

prefiller 侧用（`kv_transfer_params["decode"]` 标记）。

- 没有 `decode` 子字典→这不是远程 decode 请求，立即 `JobResult(success=True)`。
- 有 `decode.kv_request_id`：
  - 若 `_kv_to_session[id]` 已绑定（peer 已 fetch），直接 `session.add_stored_blocks(id, keys, block_ids, job_id)` 把块登记到 ServerRole 并发起 RDMA write。
  - 否则把 batch 暂存到 `_unbound_stores[id]`，等 FetchMsg 来时 replay。`_UNBOUND_STORE_TIMEOUT_S = 60s` 后由 `_reap_unbound_stores` 转 `JobResult(success=False)`。

#### `submit_load`（`:318`）

decoder 侧用（`prefill` 子字典）。

- 校验 `prefill.remote_host/port/kv_request_id`，缺则 fail。
- `keys` 空 → 短路成功。
- `_sessions[peer_id]` 不存在 → fail + 加 `_failed_req_ids`。
- 否则 `session.request_blocks(job_id, kv_request_id, keys, block_ids)`——通过 ClientRole 发 FetchMsg。

#### `get_finished_jobs`（`:383`）

每 step 调一次：`_poll_once()` → 收集 `_finished_jobs` → 清空并返回。

#### `_poll_once`（`:568`）

scheduler 线程一轮 polling：

1. `new_connections = self._control.poll()`：接 inbound ZMQ 连接。
2. `_accept_new_peers(new_connections)`：为每个新连接 build `P2PSession`。
3. 对每个 session `session.poll()` 拿 `SessionPollResult{loads, stores, new_fetch_ids}`：
   - `loads`/`stores` 翻译成 `JobResult` 加 `_finished_jobs`，`lr.success=False` 时把 `kv_request_id` 加 `_failed_req_ids`。
   - `new_fetch_ids`：把每个 id 绑定到当前 session，并 replay `_unbound_stores[id]` 中暂存的 batch（`session.add_stored_blocks`）。
4. `_reap_dead_sessions()`：连接断了的 session close + 释放 `_data.remove_remote_peer` + 把未完成的 load/store job 全 fail。
5. `_reap_unbound_stores()`：超 60s 仍无 peer fetch 的 prefiller batch 转 fail。

#### `drain_jobs`（`:401`）

`reset_cache` 用，阻塞至每个 session 没在飞的 inbound/outbound。`_poll_once` 时间长于 5s 后 warning。**不** abort 中途 transfer——`reset_cache` 需要的是 primary memoryview 静默，而非中断。

#### `shutdown`（`:618`）

`_drain_inflight_for_shutdown`（最多 3s 等 NIXL handle 自然完成，超时 force-cancel）→ 关所有 session → 把残留 unbound store 标 fail → 关控制面 + 数据面。

### 2. `control/` —— ZMQ 控制面

#### `ControlTransport` / `ControlConnection`（`base.py`）

抽象：

- `ControlConnection.send(msg: dict)`：非阻塞 enqueue；serialization 内部完成。
- `ControlConnection.recv() -> Sequence[dict]`：返回自上次以来收到的所有消息；只读。
- `ControlConnection.alive` / `mark_dead()` / `close()`。
- `ControlTransport.connect(peer_id) -> ControlConnection`：outbound 连接。
- `ControlTransport.poll() -> Sequence[ControlConnection]`：处理所有 pending I/O，返回**新接受**的 inbound 连接（已返回过的不重复）。同时已存在的连接的 `recv()` buffer 也被刷新。
- `ControlTransport.close()`：关监听 socket + 所有连接。

**无后台线程**——所有 I/O 经调用方周期 `poll()` 驱动。

#### `ZmqTransport` / `ZmqConnection`（`zmq.py`）

ROUTER/DEALER 模式实现：

- ROUTER socket 监听 `tcp://host:port`，接受 inbound DEALER 连接。
- `connect(peer_id="host:port")` 创建 DEALER socket 连到 peer ROUTER。
- 心跳：`HEARTBEAT_IVL=2s` / `HEARTBEAT_TIMEOUT=10s` / `HEARTBEAT_TTL=10s`。
- ZMQ monitor socket 跟踪连接事件（`disconnect`/`closed` 等）让 `alive` 反映真实状态。
- 序列化用 `msgspec.msgpack`（紧凑、快、可选 schema 校验）。

### 3. `data/` —— NIXL 数据面

#### `DataTransport` ABC（`base.py`）

```
view: 2D memoryview (num_blocks × block_len bytes)
base_addr, num_blocks, block_len
config_fingerprint: str  # sha256(canonical(config_fields))[:16]，无 config_fields 时为 ""
```

抽象方法：

- `get_agent_metadata() -> bytes`：NIXL agent 元数据，握手里发给 peer。
- `add_remote_peer(peer_id, agent_metadata, base_addr, num_blocks, block_len)`：注册远端。
- `remove_remote_peer(peer_id)`。
- `write_blocks(peer_id, local_idxs, remote_idxs) -> int | None`：发 WRITE transfer，返 transfer_id。
- `poll() -> PollResult(done, failed)`：轮询完成情况。
- `cancel(transfer_ids, mode="immediate"|"wait") -> list[int]`：best-effort 取消；`mode="wait"` 返回仍 inflight 的 ID 让调用方继续 poll。
- `close()`。

**无后台线程**——`poll()` 由调用方驱动。

`config_fingerprint` 用 `json.dumps(config_fields, sort_keys=True, separators=(",",":"))` 之后 sha256 前 16 hex；peer 握手时双方比对，不一致拒绝——防止不同 model/dtype/block_size 的节点互相写脏数据。

#### `NixlTransport`（`nixl.py`）

`NixlWrapper`（vLLM 对 NIXL C API 的 Python 绑定，见 [07-distributed/nixl-utils](../07-distributed/nixl-utils.md)）的子类实现。

构造（`nixl.py:38`）：

- `_backends = list(backends) or ["UCX"]`；`_num_threads = num_threads`。
- 若含非 UCX backend：`cfg = NixlAgentConfig(backends=self._backends, capture_telemetry=True)`。
- 否则：`cfg = NixlAgentConfig(num_threads=self._num_threads, capture_telemetry=True)`。
- `agent.register_memory([(base_addr, total_size, 0, "")], "DRAM")`：整片注册为单块。
- 为每个 block 算单独 descriptor，`agent.get_xfer_descs + agent.prep_xfer_dlist("NIXL_INIT_AGENT", ...)` 得复用 `_local_dlist`。

`add_remote_peer`：`agent.add_remote_agent(metadata)` 拿到 peer NIXL name，把 peer 暴露的每块注册成 remote descriptor 并 prep 为 `_remote_dlists[peer_id]`。

`write_blocks`：从 `_local_dlist` 选 `local_idxs`、从 `_remote_dlists[peer_id]` 选 `remote_idxs`，`make_prepped_xfer("WRITE", ...)` → `agent.transfer(...)` → `_inflight[transfer_id] = handle`。

`poll`：遍历 `_inflight`，`check_xfer_state` 为 `DONE` 加 `done` list，`ERR` 加 `failed` list，回收 handle。

`cancel(mode="wait")`：调 `agent.release_xfer_handle`；若 transfer 仍 `PROC/PEND` 不能 release，保留在 `_inflight` 并把 id 返给调用方让它继续 poll。`mode="immediate"` 直接释放。

### 4. `session/` —— 双向 P2P session

#### 协议消息（`protocol.py`）

| 消息 | 方向 | 字段 |
|---|---|---|
| `ConnectMsg` | C→S | `peer_id, agent_metadata(bytes), base_addr, num_blocks, block_len, config_fingerprint` |
| `ConnectAckMsg` | S→C | `peer_id` |
| `DisconnectMsg` | 任一方 | (仅 type) |
| `FetchMsg` | C→S | `kv_request_id, block_hashes, block_indexes` |
| `TransferDoneMsg` | S→C | `kv_request_id, success` |
| `AbortFetchMsg` | C→S | `kv_request_id`（取消未完 fetch） |
| `AbortAckMsg` | S→C | `kv_request_id` |

每条消息类带 `validate(msg)` 静态方法做类型/值校验，遇非法字段 raise `ValueError`。

#### 安全

- 来自 peer 的所有消息包 try/except，非法消息 log 后丢弃，不 crash session。
- `ConnectMsg` 阶段比对 `config_fingerprint` 与 `block_len`，不符拒绝握手。

#### `P2PSession`（`session.py`）

合并 `ClientRole` 与 `ServerRole` 的 thin coordinator。每个 peer 一个 session。

构造（`session.py:90`）：

- `conn=None` → pending：可调 `add_stored_blocks` 但 send 队列等到 connection 建立后 flush。
- `conn!=None` → connected：立即发自身 `ConnectMsg`；peer 的 `ConnectMsg` 到达时触发 `transport.add_remote_peer`、回 `ConnectAckMsg`；自身收到 `ConnectAckMsg` 后置 `_send_ready=True` 并 flush 队列。

`poll()` 一轮：

- `conn.recv()` 收消息。
- 按 `TYPE_KEY` 分发到 `ClientRole` 或 `ServerRole` 处理，错误重置 `_dispatch_error_count`，连续 5 次非协议错误 close session（`_MAX_CONSECUTIVE_DISPATCH_ERRORS`）。
- 返 `SessionPollResult{loads, stores, new_fetch_ids}`——后两者由各 role 内部累计。

`add_stored_blocks(kv_request_id, keys, block_ids, job_id)`：`ServerRole` 把块登记，若 ClientRole 之前已 `FetchMsg` 过此 id 则立即 RDMA write。

`request_blocks(job_id, kv_request_id, keys, block_ids)`：`ClientRole` 发 `FetchMsg`，匹配 block_hashes 把 peer 已有的立即拉、还没的等 peer 之后 `add_stored_blocks` 触发 write。

`finish_request(kv_request_id)`：取消该 id 的所有未完 load，`ServerRole` 也清掉对应 inflight store。

`close()`：发 `DisconnectMsg`，cancel 所有 inflight transfer，返 `(failed_loads, failed_stores)` 让 manager 翻译成 `JobResult(success=False)`。

#### `ClientRole` / `ServerRole`

- `ClientRole`：管理 outbound `FetchMsg` 与 inbound `TransferDoneMsg`/`AbortAckMsg`；维护 per-`kv_request_id` 的"已请求但未完"block 集。超时未到 `TransferDoneMsg` 时发 `AbortFetchMsg`，超时收 `AbortAck` 视作失败。
- `ServerRole`：管理 inbound `FetchMsg` 与 outbound `TransferDoneMsg`；维护"被 demand 但还没存"的 block 索引。`add_stored_blocks` 时若有 demand 立即发 RDMA write。

---

## 为什么

- **prefiller↔decoder disaggregation 性能关键**：P2P tier 让 prefiller 把 KV 推给 decoder，是 PD disagg 场景避免重算 prefill 的唯一可行手段——CPU 内存经 RDMA 直拷远端 CPU 内存，避免 4 倍 PCIe + NIC round trip。
- **单 session 双向**：早期实现每方向一个 connection；现合并为单 ZMQ 连接承载双向协议，简化握手与心跳。
- **数据面/控制面分离**：ZMQ 慢但能跨网段且支持 ROUTER/DEALER 多对多；NIXL/RDMA 块传输快但需直连。两者解耦让 prefiller↔decoder 部署拓扑灵活。
- **`config_fingerprint`**：peer model 不一致时静默写脏数据代价巨大；握手阶段强制比对 16-hex 内容指纹。
- **`_unbound_stores` + 60s 超时**：原 prefiller 收到 store 时就试图绑 session，但 decoder 可能晚一步连入；改为 prefiller 先暂存、等 decoder 的 FetchMsg 来时再 replay。若 decoder 始终不来（崩溃/网络分区），超时让 store job 失败。
- **`has_pending_work()` 恒 True**：engine tick 是控制面 poll 的唯一驱动；不持续 tick 会漏接新连接与 inbound 消息。
- **`drain_jobs` 不 abort transfer**：NIXL 中途 abort 会留下"半写"的 primary memoryview；等待自然完成更安全。
- **MNNVL / Mooncake / LIBFABRIC**：UCX 不是所有 RDMA 网络的最优（如 NVIDIA MNNVL multi-node NVLink 用 Mooncake），`backends` 配置项允许切换非 UCX NIXL backend。

---

## 怎么做

### 配置示例（prefiller 与 decoder 对连）

prefiller（被 fetch 方）：

```json
{
  "spec_name": "TieringOffloadingSpec",
  "cpu_bytes_to_use": 8589934592,
  "secondary_tiers": [
    {"type": "p2p", "host": "0.0.0.0", "port": 7777, "backends": ["UCX"], "num_threads": 4}
  ]
}
```

decoder（fetch 方）：同上 `secondary_tiers`，请求时 `kv_transfer_params` 携带：

```json
{
  "prefill": {
    "remote_host": "prefiller-host",
    "remote_port": 7777,
    "kv_request_id": "req-uuid-xyz"
  }
}
```

上层 [07-distributed/kv-transfer](../07-distributed/kv-transfer/README.md) 负责在 prefiller/decoder 间路由这些参数。

### MNNVL 场景

```json
{"type": "p2p", "backends": ["MOONCAKE"], "host": "0.0.0.0", "port": 7777}
```

Non-UCX backend 让 `NixlTransport` 走 `NixlAgentConfig(backends=..., capture_telemetry=True)` 分支，忽略 `num_threads`。

### 调优要点

- `port` 必须可被 peer 直连，K8s 部署注意 Service/NetworkPolicy。
- `num_threads` 仅 UCX-only 分支生效；ROCm/CUDA 跨节点 RDMA 通常 4–8 足够。
- 心跳间隔（2s）与 TTL（10s）控制 peer 崩溃检测窗口；过短易误判，过长导致 `_unbound_stores` 超时 60s 等不够。
- `_UNBOUND_STORE_TIMEOUT_S = 60s` 必须**长于**单 store 的内部超时，否则 prefiller 会先 fail 而 decoder 才到——见 `manager.py:46` 注释。

---

## 与其它模块/系统配合

- 编排：[tiering.md](tiering.md) `TieringOffloadingManager`（本 tier 由 `SecondaryTierFactory` 注册）。
- 数据面：[07-distributed/nixl-utils](../07-distributed/nixl-utils.md) `NixlWrapper` / `nixl_agent_config`。
- 上游 `kv_transfer_params`：[07-distributed/kv-transfer](../07-distributed/kv-transfer/README.md) 在 disagg scheduler 中生成 `prefill`/`decode` 子字典并透传到 `ReqContext`。
- `config_fingerprint`：用 [file-mapper.md](file-mapper.md) `get_run_config()` 的内容。
- primary memoryview：[cpu.md](cpu.md) `SharedOffloadRegion.create_kv_memoryview()`。
- 跨子系统锚：[执行层-kv_connector mixin](../02-execution/worker/kv-connector-mixin.md)（worker 侧 connector metadata 注入），[分布式-KV transfer](../07-distributed/kv-transfer/README.md)（session disagg 决策）。

---

## 历史版本演进

| 版本 | 变化 |
|---|---|
| v0.11 | **P2P tier 首发**：原版本为单向 client/server 两 session；引入 `kv_request_id` 绑定、`_unbound_stores` 暂存路径、`has_pending_work` 恒 True 强制 tick |
| v0.11.x | session 合并为双向 `P2PSession`：单 ZMQ 连接承载 client/server 双协议；`ConnectMsg`/`ConnectAckMsg` 双向握手；`config_fingerprint` 加入握手校验 |
| v0.11.x | `_reap_dead_sessions` + `_reap_unbound_stores` 引入：peer 崩溃/无 peer fetch 时显式失败管理；之前 `SessionPollResult.new_fetch_ids` 重构：让 manager 而非 session 回调驱动 `_kv_to_session` 绑定 |
| main | `_drain_inflight_for_shutdown` 加 3s 超时 + force-cancel fallback：避免 wedged peer hang shutdown；`cancel(mode="wait")` API 引入让 drain 期不破坏 primary memoryview；MNNVL/Mooncake backend 经 `backends` 配置项支持（待核实具体 commit） |

---

[← 返回 KV 卸载首页](README.md)

## 参见

- [tiering.md](tiering.md)：编排与本 tier 的协作。
- [tiering-obj.md](tiering-obj.md)：同样基于 NIXL，但目标是 OBJ backend 而非 DRAM↔DRAM。
- [07-distributed/nixl-utils](../07-distributed/nixl-utils.md)：NIXL wrapper。
