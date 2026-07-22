# 夜间自主执行日志 (2026-07-21 夜)

用户切走 h200-2 流量、授权停 SGLang 后开始逐个实验。全程自主:遇错自修,修不了跳过。
**抗中断**:bench 一律 detach 到 h200-6(nohup+setsid),结果落 h200-6 再回拉;服务用 setsid 起在 h200-2。
恢复会话时先读本文件 + 各 result 目录判断进度。

## 环境实测修正(详见 memory project-vllm-minimax-pd)
1. h200-2 /ceph 无 User/E01223 → 同步到 **h200-2:/data1/pd-exp/vllm**,`REPO_ON_EXEC=/data1/pd-exp/vllm`。
2. 镜像 ENTRYPOINT=["vllm","serve"] → CMD 只写 model+args(已修 serve_pd.sh/verify.sh)。
3. MiniMax FP8 blockquant 纯 TP8 崩(192%128)→ base1 加 `--enable-expert-parallel`。TP4 不受影响。
4. 发压 IP:h200-2 用 **10.119.195.74**(bond1),非 10.118.89.32(已修 common.env)。
5. h200-6 无 pip/numpy → bench 走 CPU-only docker(vllm 镜像);ramp `_count_traces` "dataset has 1" 是无害 bug。
6. MAX_MODEL_LEN 32768→**65536**(32768 下 codex 多轮超长 ~15% HTTP400;模型支持 196608)。
7. **PD proxy 启动 bug**(影响 test1/2/3):serve_pd.sh launch_proxy 用 `python`(容器只有 `python3`)
   → `python: command not found`,proxy 从不启动,冒烟卡死。已改 python3。
8. **PD proxy 只绑 127.0.0.1**:toy_proxy 默认 host 127.0.0.1,h200-6 发压打不到。已加 `--host 0.0.0.0`。

## 执行顺序与状态
| # | 实验 | 状态 | 结果目录 / 备注 |
|---|------|------|----------------|
| 0 | 停 SGLang 腾 8 卡 | ✅ 完成 | 两容器已删,8 卡释放 |
| 1 | base1-tp8 (TP8attn+EP8) | ✅ 完成 | 65536正式: results/base1-tp8_20260721_212609(32768旧跑_200520有截断,弃用) |
| 2 | base2-tp4 | ✅ 完成 | results/base2-tp4_20260721_214626 |
| 3 | verify-kvrole | ✅ 完成 | results/verify_20260721_222125;**pc/both 输出均与golden一致,KV均传输(证据66/70)→用默认 pc** |
| 4 | test1-nixl-pd | ✅ 完成 | results/test1-nixl-pd_20260721_224311 |
| 5 | test3-nixl-cpu-bypass | ✅ 完成 | results/test3-nixl-cpu-bypass_20260721_231532 |
| 6 | test2-mooncake-pd | ⛔ 阻塞 | 镜像已直传h2(RoCE 62MB/s);服务/proxy/import 均OK,但 mooncake TransferEngine 起不来:h2 RoCE NIC 只有 link-local(fe80/RoCEv1)GID,mooncake 需 RoCEv2 GID → `No available RNIC`。TCP 路径传输也失败。需 fabric 配 RoCEv2 GID |
| - | test4-mooncake-cpu-bypass | 🧩 代码完成/实验阻塞 | DESIGN.md + code/vllm/.../mooncake_connector.py(+72行,py_compile通过);实验与 test2 同被 fabric 阻塞 |

## 2026-07-22 追加(mooncake test2/test4)
- **镜像直传解决**:h6→h2 直连(h2 已有到 h6 免密 ssh),从 h2 pull 走 RoCE eth10 **62MB/s**(a100-2 中转仅 3.4MB/s)。
- **serve_pd 修复**:mooncake proxy 加 `--host 0.0.0.0`;mooncake 容器透传 `/dev/infiniband/*` + `--cap-add=IPC_LOCK --ulimit memlock=-1`;
  kv_config 加 `mooncake_protocol`/`device_name`(MOONCAKE_PROTOCOL/MOONCAKE_DEVICE 覆盖)。
- **test2 阻塞根因**:mooncake RDMA 需 RoCEv2 GID,h2 各 mlx5 只有 fe80 link-local GID(RoCEv1)→ 全设备被禁用 `No available RNIC`。属 fabric/GID 配置,非代码 bug。
- **test4 代码**:按设计路线A实现(host pinned 镜像注册 + send前D2H + recv后H2D,复用 SupportsHMA 注入的 copy_blocks),py_compile 通过,待 on-device 验证(依赖 test2 传输通)。
- git:develop 已提交推送 fork(lycheenice/vllm v0.25.0);本轮 mooncake 改动待再次提交。

## 2026-07-22 交回 + 问答修正
- **h200-2 已交回生产**:实验容器/进程全清,SGLang 生产已自行恢复满载。
- **Q1(为何单机还要 RDMA)**:mooncake TransferEngine 只有 rdma/tcp,无 NVLink/cuda_ipc 本地传输;
  nixl 用 UCX cuda_ipc 走 NVLink 才在单机免 RDMA。故 mooncake 单机 PD 仍走网卡,详见 test4/DESIGN.md §4。
- **Q2(是否容器参数/host network)**:用了 `--network host`;缺 `/dev/infiniband` 透传已补;
  剩余卡在 mooncake 只取到 link-local(RoCEv1)GID index0 → `No available RNIC`。鉴于 fabric 实测正常,
  **修正**:非 fabric 问题,而是 RoCEv2 GID index 选择 / 单机回环,未在交回前解决。详见 DESIGN.md §4。

## 2026-07-22 mooncake 根因定位 + CPU 验证(已解决,GPU 端待跑)
- 用户 `ib_write_bw -d mlx5_1` 通 → fabric/RoCEv2 正常。CPU 端(挂 libcuda 不占 GPU)定位:
- **根因**:h200-2 各 mlx5(含 mlx5_bond_0)**都有 RoCEv2 GID,在 index 3**;mooncake **不自动选**,不设
  `MC_GID_INDEX` 就报 `GID is NULL / GID -1 / No available RNIC`(engine.so 里有此提示串)。get_ip()=bond1
  更让它绑到 mlx5_bond_0。**非 fabric、非 host-network 问题**——之前判断已修正。
- **验证**:`MC_GID_INDEX=3` 时 `initialize` 对 mlx5_1/mlx5_bond_0/auto 全 ret=0;两进程 host-buffer RDMA
  自测**数据正确送达(0xAB 全中)**。`transfer ret=-1` 仅纯 CPU 无 GPU 的 `cudaPointerGetAttributes` 假象。
- **修复**:serve_pd.sh mooncake 容器加 `-e MC_GID_INDEX=3`(可 env 覆盖)。test2/test4 完整 PD bench 待 GPU 空闲。
| - | test4-mooncake-cpu-bypass | ❌ 跳过 | code/vllm 未开发 |

## ★核心对比报告 → results_report/PD_COMPARISON.md + pd_ramp.png(make_pd_report.py 生成)
**归因(C=16 output tok/s)**:TP8→TP4 −19%(238→192)· PD 分离本身 **−38%**(192→118)· CPU中转vsGPU直传 +8%(噪声内≈持平)。
**结论**:PD 对多轮 agent 负载净负收益,主因 TTFT 爆炸(15637ms vs base2 489ms);TPOT 是 PD 唯一优势;
connector/传输方式不是瓶颈(test3≈test1),PD 结构性开销(重复prefill/排队/proxy)才是;kv_role 用默认 pc 即可。

## base1-tp8 初步结果(32768 旧跑,有 ~15% 上下文截断 400,已改 65536 重跑)
| C | E2E mean | TTFT p50 | TPOT mean | tok/s | 失败 |
|---|---|---|---|---|---|
| 1 | 14.7s | 3425ms | 52.6ms | 15 | 3/13 (400超长) |
| 4 | 15.9s | 365ms | 63.4ms | 47 | 8/58 |
| 8 | 15.2s | 391ms | 60.7ms | 107 | 20/109 |
| 16 | 15.0s | 354ms | 60.4ms | 188 | 34/227 |

## base1-tp8 正式结果 @65536 (TP8 attn + EP8 MoE, results/base1-tp8_20260721_212609)
| C | E2E mean | tok/s | compl_tok | 失败 |
|---|---|---|---|---|
| 1 | 12.0s | 19 | 3564 | 0 |
| 4 | 13.3s | 71 | 15428 | 0 |
| 8 | 13.9s | 133 | 30573 | 1 |
| 16 | 15.0s | 241 | 61189 | 1 |
65536 后错误从 ~15% 降至 ~0;吞吐 19→71→133→241 tok/s 线性扩展,E2E 稳定 12→15s,整机余量充足。
