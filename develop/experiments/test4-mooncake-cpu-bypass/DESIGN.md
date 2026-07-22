# test4 — MooncakeConnector CPU 中转(host-staging)设计文档

**日期**: 2026-07-22 · **vLLM**: v0.25.0 · **状态**: 设计完成 + 代码已实现(py_compile 通过),
**待 on-device 验证**(当前被 h200-2 的 mooncake RoCE fabric 阻塞,见 §4)。
**目标**: 给上游 `MooncakeConnector`(P2P GPU 直传)加 `kv_buffer_device=cpu` 语义 —— KV 先 D2H
落宿主 pinned DRAM,经 mooncake 传输,对端 H2D 回显存。用于与 **test3**(nixl + CPU 中转)横向对比,
并补齐"connector × 传输设备(GPU 直传/CPU 中转)"矩阵的最后一格。

---

## 1. 现状(基于实际代码调研)

### MooncakeConnector 是纯 GPU 直传
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`:
- `register_kv_caches`(L1653):对每层取 `cache.data_ptr()`(**GPU 显存**),用
  `self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)`(L1725)注册 GPU storage;
  存 `self.device_kv_caches = kv_caches`(L1729)、`self.kv_caches_base_addr`(GPU 基址)。
- 传输(P 侧 send / D 侧 recv)用 mooncake TransferEngine 的 `batch_transfer_sync_write`,
  src/dst 都是**注册过的 GPU 地址**;按 block 粒度(`kv_block_len_per_layer`)构造 descriptor。
- **没有 `kv_buffer_device` 分支**(config 里字段存在,但 mooncake worker 未读取)。

### NixlConnector 的 CPU 中转模板(要移植的范式)
`vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`:
- `use_host_buffer = (kv_buffer_device == "cpu")`;`self.host_xfer_buffers: dict[str,Tensor]`。
- `_allocate_host_xfer_buffers`:每层 `torch.empty(kv_shape, dtype, device="cpu")`(pinned)。
- `register_kv_caches`:`use_host_buffer` 时注册 **host buffer 基址**(而非 GPU)到传输后端。
- D2H:`save_kv_to_host` → `copy_blocks(device_kv_caches, host_xfer_buffers, ids, ids, "d2h")`。
- H2D:`sync_recved_kv_to_device` → `copy_blocks(host_xfer_buffers, device_kv_caches, ids, ids, "h2d")`。
- `copy_blocks` 由模型运行器经 `set_host_xfer_buffer_ops(copy_operation)` 注入
  (接口在 `base.py`,类型 `CopyBlocksOp = Callable[[src_dict, dst_dict, src_ids, dst_ids, "h2d"|"d2h"], None]`)。

### 接口契约(base.py,worker 侧)
`register_kv_caches` / `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save`
/ `get_finished`;基类已提供 `set_host_xfer_buffer_ops(copy_operation: CopyBlocksOp)` 默认实现
(存到 `self.copy_blocks`),子类复用即可。

### 路线 B(MooncakeStore + DRAM)结论:不可行
`mooncake/store/connector.py` 的 `MooncakeStoreConnector` 是集中式共享存储、无 `kv_buffer_device`
开关、架构与 P2P 直传不同;要用它做 CPU 中转须改 mooncake 库底层,成本过高。**故走路线 A。**

---

## 2. 设计(路线 A:给 MooncakeConnectorWorker 加 host-staging)

### 数据流
```
P(producer):  attn 写 GPU KV ──D2H──▶ host pinned buf ──mooncake send──▶
D(consumer):  ──mooncake recv──▶ host pinned buf ──H2D──▶ GPU KV ──▶ attn 读
```
对比:GPU 直传(test2)是 GPU──mooncake──GPU;CPU 中转(test4)在两端各加一次 D2H/H2D,
并把 mooncake 注册/传输的地址从 GPU 改为 host buffer。等价于 nixl 的 `kv_buffer_device=cpu`(test3)。

### 改动点(全部在 `mooncake_connector.py` 的 `MooncakeConnectorWorker`)
1. **`__init__`**:读 `kv_buffer_device`,置
   `self.use_host_buffer = (self.kv_buffer_device == "cpu")`;`self.host_xfer_buffers = {}`;
   `self.copy_blocks = None`。
2. **`register_kv_caches`**:`use_host_buffer` 时,为每个 `cache` 分配等形状 `device="cpu"` pinned
   buffer 存入 `self.host_xfer_buffers[layer_name]`;把注册用的 `base_addr/storage_addr` 改成
   **host buffer 的地址**;仍存 `self.device_kv_caches = kv_caches`(D2H/H2D 的 GPU 端)。
   其余 block_len/topology 计算不变(host buffer 形状与 device 一致)。
3. **send 前(P 侧,构造 transfer ops 处)**:若 `use_host_buffer`,对本次要传的 block ids 先
   `self.copy_blocks(self.device_kv_caches, self.host_xfer_buffers, ids, ids, "d2h")`,
   再走原 mooncake send(此时注册地址已是 host,transfer 自然从 host 发出)。
4. **recv 完成后(D 侧,KV 落地后、通知 ready 前)**:若 `use_host_buffer`,
   `self.copy_blocks(self.host_xfer_buffers, self.device_kv_caches, ids, ids, "h2d")`。
5. **`MooncakeConnector` 代理类**:透传 `set_host_xfer_buffer_ops` 到 worker(基类默认实现即可,
   若代理类未继承则显式转发)。

### 关键风险与验证点
- **R1 mooncake 是否支持 host(DRAM)内存注册/传输**:`batch_register_memory` 能否注册 host 地址、
  `batch_transfer_sync_write` 能否 host→host 传。RDMA 本身支持注册 host pinned 内存(日志见
  `Using default malloc/free for protocol: rdma`),预期可行;实现后**首先单独验证一次 host↔host 传输**。
- **R2 H2D 注入点**:mooncake recv 是异步(sender listener + ZMQ 通知),需精确定位"某 req 的 KV
  已全部 recv"事件,在其后、标记 finished_recving 前插 H2D。见 `fetch_finished_recving_reqs`
  (L1746)与 `_start_load_kv` 路径。
- **R3 copy_blocks 注入**:确认 gpu_model_runner 对 MooncakeConnector 也调用
  `set_host_xfer_buffer_ops`(nixl 走此路径;需确认 mooncake 未被特殊跳过)。若未调用,退化方案:
  在 worker 内自建一个 `copy_blocks`(直接用 `torch` 按 block 索引 D2H/H2D,不依赖 runner 注入)。

### 落地文件(最小改动,单文件覆盖)
```
develop/experiments/test4-mooncake-cpu-bypass/code/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
```
经 `config.env` 的 `VLLM_CODE_OVERRIDE=1` + `VLLM_PKG_PATH=/usr/local/lib/python3.12/dist-packages/vllm`
bind-mount 单文件覆盖进容器(免整包挂载)。

> ⚠️ **挂载方式**:当前 `common/serve_pd.sh` 的 `VLLM_CODE_OVERRIDE` 是**整目录**挂
> `code/vllm:$VLLM_PKG_PATH`,会遮蔽整个 vllm 包。因 code/ 只含单文件,运行前须改为
> **单文件挂载**:`-v code/vllm/.../mooncake_connector.py:$VLLM_PKG_PATH/.../mooncake_connector.py`。
> 故 config.env 暂留 `VLLM_CODE_OVERRIDE=0`,待改好单文件挂载 + fabric 通后再置 1。

## 已实现改动(code/vllm/.../mooncake/mooncake_connector.py,py_compile 通过,+72 行)
1. `MooncakeConnectorWorker.__init__`:`use_host_buffer`/`host_xfer_buffers`/`copy_blocks` 标志。
2. `register_kv_caches`:`use_host_buffer` 时分配每层 `device="cpu" pin_memory=True` 镜像,
   对 host 地址注册;`device_kv_caches` 仍存 GPU 张量供 D2H/H2D。
3. send 路径(`_send_blocks` 调用前):对 `ok_ready_reqs` 各 `send_meta.local_block_ids` D2H。
4. recv 完成(`process_pulling_result`,`pull_tasks_count==0`):对 `pull_meta.local_block_ids` H2D。
5. `_stage_host_blocks(direction, block_ids_by_group)` helper:复用注入的 `copy_blocks`。
6. `MooncakeConnector.set_host_xfer_buffer_ops`:把运行器注入的 copy_blocks 转发给 worker。

**待验证点**(on-device):copy_blocks 的多 KV 组 block-id 展平是否正确;mooncake 对 host pinned
内存的注册/传输是否 OK(R1);H2D 时机(pull 完成即 H2D)与 layerwise 语义是否一致。

---

## 3. 实验方法
1. 前置:mooncake 镜像在 h200-2(已传)、**test2(GPU 直传)先跑通**(验证 mooncake 传输链路本身)。
2. `config.env` 置 `BYPASS=cpu`、`VLLM_CODE_OVERRIDE=1`;起服务先跑内置冒烟。
3. `correct_check`:与 base1 greedy 输出一致(语义正确)。
4. bench(统一口径),与 **test3(nixl+cpu)** 横向对比、与 **test2(mooncake+gpu)** 纵向对比。

## 4. 前置阻塞与关键问答(2026-07-22)

### Q1:单机 TP4+TP4 的 PD,为什么 mooncake 还要 RDMA?不能走 NVLink 吗?
- **两个 connector 传输机制不同**:
  - **NIXL(test1/test3)**用 UCX,我们设了 `UCX_TLS=cuda_ipc,cuda_copy,tcp`。同机 GPU↔GPU 时 UCX 选
    **`cuda_ipc`= CUDA IPC 点对点(走 NVLink/PCIe P2P),不经网卡**。所以 nixl 单机 PD 天然用 NVLink,无需 RDMA。
  - **Mooncake(test2/test4)**的 TransferEngine 在 vLLM v0.25.0 里只暴露 `rdma` / `tcp` 两种协议
    (`mooncake_protocol`,默认 rdma),**没有 CUDA-IPC/NVLink 本地传输**。故即便单机 P/D,inter-instance
    的 KV 传输也走**网卡(RDMA)或 TCP**,不会走 NVLink。
- **关键区分**:NVLink 在这里只用于**实例内** TP4 的 NCCL all-reduce;**跨实例 P→D 的 KV 搬运**是另一条
  连接器传输路径——nixl 能把它走 NVLink(cuda_ipc),mooncake 在此集成里不能。
- **结论**:mooncake 现状下,单机 PD 仍需 RDMA(或可用的 TCP);它不会自动用 NVLink 做 PD 这一跳。这也说明
  **mooncake 在结构上比 nixl 更不适合单机 PD**(nixl 白拿 NVLink)。(mooncake 某些版本或有 local/shm 传输,
  但本连接器未选用,亦未跑通。)

### Q2:h2↔h6 RDMA 实测没问题,是不是容器挂载参数的问题?有没有用 host network?
- **用了 host network**:`serve_pd.sh` 的 `docker run` 一直带 `--network host --ipc=host`。
- **容器参数确有过一处缺口且已补**:最初容器有 rdma 库但**没透传 `/dev/infiniband` 字符设备** →
  mooncake `topology: No RDMA devices found`。已加 `--device=/dev/infiniband/* --cap-add=IPC_LOCK --ulimit memlock=-1`,
  之后容器能看到 RDMA 设备。
- **补齐后仍失败于 GID**:`Failed to open device mlx5_* … GID -1 / No available RNIC`。实测各 mlx5(含
  mlx5_bond_0)**GID index 0 = link-local `fe80`(RoCEv1)**,mooncake 需 RoCEv2 GID。
- **修正此前结论**:鉴于你实测 h2↔h6 RDMA 正常,**fabric 本身没问题**;更可能的原因是:
  (a) **GID index 选择**——RoCEv2 GID 在更高 index(如 1/3),mooncake 默认取到 index 0 的 link-local;
      可能需给 mooncake 指定 GID index / 正确的 `device_name:port`;
  (b) **单机 RDMA 回环**——你的 RDMA 测试大概率是跨机(h2↔h6),而 test2 是 h2→h2(P/D 同机、都绑
      `get_ip()=10.119.195.74`),同网卡 QP-to-self 回环行为可能与跨机不同;
  (c) 仍可能有我没完全配对的容器/GID 细节。
- **老实说**:我把"RDMA 设备可见"这步搞定了,但**没在交回机器前解决 RoCEv2-GID 选择**,故之前"fabric 需配
  RoCEv2 GID"的说法**过强**。准确表述:容器已能看到 RDMA 设备,但 mooncake 只找到 link-local(RoCEv1)GID
  就把所有 RNIC 禁用了;下一步应让 mooncake 用上 RoCEv2 GID(指定 GID index / 设备端口),并处理单机回环。

### 对 test4 的影响
test4 实验依赖 mooncake 传输可用,与 test2 同被上述 GID 问题阻塞。**代码已实现**(见 §落地约定 + 上方实现清单),
待 mooncake 传输在单机跑通后 on-device 验证。注:test3 已证 KV 传输方式非 PD 瓶颈,test4/test2 预期结论与 test1 同量级。
