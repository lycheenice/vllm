# HANDOFF —— MiniMax-M2.5 vLLM v0.25.0 PD 分离实验(交接)

**更新**: 2026-07-22 · **接手先读**: 本文件 → [`NIGHT_RUN_LOG.md`](./NIGHT_RUN_LOG.md)(全程执行日志+踩坑)
→ [`results_report/PD_COMPARISON.md`](./results_report/PD_COMPARISON.md)(核心结论)→ 目标实验 README。
记忆指针:`project-vllm-minimax-pd`(状态最全)、`feedback_vllm_pd_experiment_workflow`。

## 0. 一句话现状
**base1/base2/verify-kvrole/test1/test3 五个实验已实跑完成+出报告**(数据在各 `experiments/*/results/`,
跨实验对比在 `results_report/`)。**test2(mooncake GPU直传)/ test4(mooncake CPU中转)未跑完**:
根因已定位并 CPU 端验证修复(`MC_GID_INDEX=3`),代码/脚本已就绪,**只差 h200-2 GPU 空闲跑完整 PD bench**。

## 1. 机器与账户
| 机器 | 角色 | 关键点 |
|------|------|--------|
| a100-2(Claude 所在) | 撰写+监控 | `/ceph/User/E01223/mycode/vllm` 权威副本(在此写);`ssh -l root h200-2/h200-6` |
| h200-2 `10.119.195.74`(bond1) | 执行 | 8×H200;**/ceph 只挂 teamFPA,看不到仓库** → 仓库同步在 **`/data1/pd-exp/vllm`**,跑时设 `REPO_ON_EXEC=/data1/pd-exp/vllm`;模型 `/data1/models/MiniMax-M2.5`;**GPU 归生产(SGLang 满载),CPU/网络可用**;h2→h6 免密 ssh 已通 |
| h200-6 | 发压+镜像+git | kvcache-benchmarks 在 `/home/lychee/mycode/vllm`(有 `fork` 远程);发压走 CPU-docker;git push 用 **lychee 身份** |

**同步**:本地 `develop/` 改完 → `tar czf - develop | ssh -l root h200-6 "tar xzf - -C /home/lychee/mycode/vllm"`(留档+提交);
h200-2 执行侧 → `tar ... | ssh -l root h200-2 "tar xzf - -C /data1/pd-exp/vllm"`。

## 2. 已完成实验结果(C=16 output tok/s;max-model-len 65536)
| 实验 | 切分 | tok/s | 结果目录 |
|------|------|-------|---------|
| base1-tp8 | TP8 attn + **EP8**(纯TP8因FP8 blockquant不可行) | 238 | `experiments/base1-tp8/results/base1-tp8_20260721_212609/` |
| base2-tp4 | 单实例 TP4 | 192 | `experiments/base2-tp4/results/base2-tp4_20260721_214626/` |
| test1-nixl-pd | PD nixl GPU直传(NVLink) | 118 | `experiments/test1-nixl-pd/results/test1-nixl-pd_20260721_224311/` |
| test3-nixl-cpu-bypass | PD nixl CPU中转 | 127 | `experiments/test3-nixl-cpu-bypass/results/test3-nixl-cpu-bypass_20260721_231532/` |
| verify-kvrole | pc vs both | — | `experiments/verify-kvrole/results/verify_20260721_222125/`(结论:**pc 即可**) |

**结论**:PD 对多轮 agent 负载净负收益(-38% vs base2,主因 TTFT 爆炸 ~32×);TPOT 是 PD 唯一优势;
**KV 传输方式非瓶颈**(test3≈test1)。跨实验图表 `results_report/PD_COMPARISON.md` + `pd_ramp.png`。

## 3. 待办:跑 test2 / test4(GPU 空闲后)
harness 已修好并固化(见 §5)。**mooncake 根因已解决**:`MC_GID_INDEX=3`(RoCEv2 GID index)已进 serve_pd.sh,
CPU 端实测 initialize ret=0 且两进程 RDMA 数据实测送达(详见 `experiments/test4-mooncake-cpu-bypass/DESIGN.md §4`)。

```bash
# 前置:h200-2 8 卡空闲(需先停 SGLang 生产,得用户确认)。仓库已在 /data1/pd-exp/vllm。
# ---- test2(mooncake GPU 直传)----
ssh -l root h200-2 'cd /data1/pd-exp/vllm/develop/experiments/test2-mooncake-pd && \
  REPO_ON_EXEC=/data1/pd-exp/vllm IMAGE=docker.1ms.run/vllm/vllm-openai:v0.25.0-mooncake \
  setsid bash run.sh > results/run.out 2>&1 < /dev/null &'
# 就绪判据:results/logs/prefill.log 有 "KV Transfer metrics ... successful transfers>0"、无 ret=-1
( cd experiments/test2-mooncake-pd && bash bench.sh )   # a100-2 发压(内部 ssh h200-6)
ssh -l root h200-2 'bash /data1/pd-exp/vllm/develop/common/stop.sh test2-mooncake-pd'

# ---- test4(mooncake CPU 中转,代码已实现,py_compile 通过)----
# ⚠ 单文件覆盖:serve_pd 的 VLLM_CODE_OVERRIDE 是整目录挂载会遮蔽整个 vllm 包,
#   test4 只改单文件,须改成单文件 bind-mount(见 DESIGN.md §落地约定),再置 config.env VLLM_CODE_OVERRIDE=1。
#   要挂的文件:code/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
#   -> 容器内 /usr/local/lib/python3.12/dist-packages/vllm/.../mooncake_connector.py
# 然后同 test2 起服务(config.env 已 BYPASS=cpu),对比 test3(nixl+cpu)。
```
建议:跑前先用 serve_pd 内置冒烟确认 KV 真传输(prefill.log successful transfers>0),再 bench。

## 4. mooncake 关键事实(排查已用掉大量时间,勿重踩)
- mooncake TransferEngine **只有 rdma/tcp,无 NVLink/cuda_ipc**;单机 PD 也经 CX NIC 的 RoCEv2 RDMA
  (不像 nixl 用 cuda_ipc 走 NVLink)。已验证 rdma 路径通,tcp 路径此前失败,协议默认已设 rdma。
- **必须 `MC_GID_INDEX=3`**(h200-2 各 mlx5 的 RoCEv2 GID 在 index 3;换机器用 `show_gids` 确认)。已进 serve_pd。
- mooncake 容器还需:`--network host`、透传 `/dev/infiniband/*`、`--cap-add=IPC_LOCK --ulimit memlock=-1`
  (均已进 serve_pd 的 mooncake 分支)。
- `get_ip()` 取默认路由源 10.119.195.74(bond1=CX-6 Dx mlx5_bond_0);auto device 会发现所有 CX-7,可用。
- `import mooncake.engine` 需 libcuda:CPU 端调试可 `-v /usr/lib/x86_64-linux-gnu/libcuda.so.1(+.595.58.03):同路径:ro`
  不加 `--gpus`(不碰生产 GPU)。注:纯 CPU 下 transfer 完成 ack 会因 `cudaPointerGetAttributes` 报 ret=-1,
  但 RDMA 数据仍送达——有 GPU 时不会。
- 硬件拓扑图:`experiments/test4-mooncake-cpu-bypass/topology_h200-2.jpg`(NVSwitch 全互联、每GPU+CX-7 同PCIe switch)。

## 5. harness 已修复(全部固化在 develop/,勿重复踩)
1. h200-2 /ceph 无仓库 → `/data1/pd-exp/vllm` + `REPO_ON_EXEC`。
2. 镜像 ENTRYPOINT=`["vllm","serve"]` → docker CMD 只写 model+args。
3. FP8 blockquant 纯 TP8 崩 → base1 用 `--enable-expert-parallel`(TP4 不受影响)。
4. 发压 IP=`10.119.195.74`(bond1);`common.env` KVBENCH_PY 走 CPU-docker(h6 无 pip/numpy)。
5. MAX_MODEL_LEN=65536(32768 下 codex 多轮 ~15% HTTP400)。
6. PD proxy:`python3`(非 python)+ `--host 0.0.0.0`(nixl & mooncake)。
7. mooncake:`MC_GID_INDEX=3` + IB 设备透传 + IPC_LOCK,协议默认 rdma。

## 6. Git
develop/ 已全部提交并推送 **fork `git@github.com:lycheenice/vllm.git` 分支 v0.25.0**
(在 h200-6 用 lychee 身份:`sudo -u lychee git push fork refs/heads/v0.25.0:refs/heads/v0.25.0`,
避开同名 tag;root 的 key 未注册 github)。截至交接最新 commit `67bdf2c1a`;本次改动(协议默认 rdma + 本文件)待再提交。

## 7. 阻塞 / 待用户决策
- 🔴 h200-2 8 卡在 SGLang 生产手里;跑 test2/test4 需先停生产(用户确认后)。
- 🟡 test4 单文件挂载改造(见 §3 ⚠)。
- 🟡(可选)NUMA 亲和:P(GPU0-3)用 mlx5_0-3、D(GPU4-7)用 mlx5_4/5/6/9,按实例 `MOONCAKE_DEVICE` pin。
