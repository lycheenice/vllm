# HANDOFF —— MiniMax-M2.5 vLLM PD 分离实验课题(交接文档)

**更新**: 2026-07-21 · **接手请先读**: 本文件 → [`EXPERIMENTS.md`](./EXPERIMENTS.md)(总纲)→ 目标实验的 `README.md`。
**背景分析(勿重推)**: [`vllm_pd_analysis_and_optimization_20260721.md`](./vllm_pd_analysis_and_optimization_20260721.md)。

---

## 0. 一句话现状
6 个实验 + 1 个 kv_role 验证实验的**框架/脚本/文档已就绪并通过 `bash -n`,尚未实跑**
(h200-2 的 8 卡还被 SGLang 生产占用,等用户指示停)。mooncake 派生镜像已在 h200-6 构建完成。

## 1. 机器与账户
| 机器 | 角色 | 关键点 |
|------|------|--------|
| a100-2(当前) | 撰写+监控 | `/ceph/.../vllm/develop` 权威副本;`ssh -l root h200-2 / h200-6`(默认 lychee 无权限,**必须 -l root**);无 rsync,用 `tar over ssh` |
| h200-2 (`10.118.89.32`) | 执行 | 8×H200;挂 `/ceph`;容器 bind-mount `/ceph` 仓库直接跑;模型 `/data1/models/MiniMax-M2.5`;**无外网** |
| h200-6 | 发压+拉/建镜像 | kvcache-benchmarks + `codex_swebenchpro.json` 在此;GPU 满载生产(**只用 CPU,勿碰 GPU**);无 `/ceph` |

## 2. 目录结构(develop/)
```
EXPERIMENTS.md  HANDOFF.md(本文件)
common/         common.env(口径) serve_pd.sh bench_ramp.sh stop.sh status.sh lib.sh
docker/mooncake/  Dockerfile  build.sh
experiments/
  base1-tp8/  base2-tp4/  test1-nixl-pd/  test2-mooncake-pd/
  test3-nixl-cpu-bypass/  test4-mooncake-cpu-bypass/(code/ 待开发)  verify-kvrole/
每个实验: config.env(唯一差异点) run.sh bench.sh stop.sh README.md results/
```

## 3. 统一发压口径(已按用户最新要求)
`--levels 1,4,8,16` · `--max-turns 4` · `--trials-per-user 4` · **`--max-tokens 256`** · 目标每组 <10min。
固化在 `common/common.env`,覆盖用 `LEVELS=.. bash bench.sh`。

## 4. 已完成 ✅ / 未完成 ⬜
- ✅ 本地 `develop/` 与 h200-6 同步;6 实验 + verify-kvrole 脚手架;总纲 + 各 README;`bash -n` 全过。
- ✅ 统一口径(含 max-tokens=256);serve_pd 支持 `CONNECTOR×BYPASS×MODE` 及 `KV_ROLE_MODE` 开关。
- ✅ mooncake 派生镜像 `docker.1ms.run/vllm/vllm-openai:v0.25.0-mooncake` 在 **h200-6 已构建**(纯 CPU)。
- ⬜ 该镜像 **import 未验证**:`import mooncake.engine` 需 `libcuda.so.1`(要 `--gpus`);为不碰 h200-6 生产 GPU,
     推迟到 **h200-2 实跑 test2 时**用带 `--gpus` 的容器验证:
     `docker run --rm --entrypoint python3 --gpus '"device=0"' <image> -c "import mooncake.engine;print('ok')"`。
- ⬜ 该镜像 **未传 h200-2**(h200-2 无外网)。需要时:
     `ssh -l root h200-6 "docker save docker.1ms.run/vllm/vllm-openai:v0.25.0-mooncake" | ssh -l root h200-2 "docker load"`(~28GB,不占 GPU)。
- ⬜ **所有实验均未实跑**(等 h200-2 腾 GPU)。
- ⬜ test4 的 CPU 绕行**代码未开发**(设计见 `experiments/test4-mooncake-cpu-bypass/code/README.md`)。

## 5. 阻塞 / 待用户决策
- 🔴 **停 h200-2 的 SGLang 生产**(`sglang-glm-sglang-1`)腾出 8 卡 —— 用户说"再晚一点会停,等指示"。停法:
  `ssh -l root h200-2 "docker stop sglang-glm-sglang-1"` 后 `nvidia-smi` 确认清零。**得到指示再动**。

## 6. 下一步建议顺序(GPU 腾出后)
```bash
cd /ceph/User/E01223/mycode/vllm/develop/experiments
# ① 先验证 kv_role(决定后续所有 PD 实验的默认 kv_role)
ssh -l root h200-2 "bash $(pwd)/verify-kvrole/verify.sh"   # 读 results/verify_*/VERDICT.md
# ② 基线
ssh -l root h200-2 "bash $(pwd)/base1-tp8/run.sh"   # h200-2 起服务
( cd base1-tp8 && bash bench.sh )                    # a100-2 发压(内部 ssh h200-6)
tail -f base1-tp8/results/logs/*.log                 # a100-2 监控
ssh -l root h200-2 "bash $(pwd)/base1-tp8/stop.sh"
# ③ 依次 base2 → test1(用 ① 的 kv_role 结论)→ test3 →(装/传 mooncake 后)test2 →(开发后)test4
```
每个实验**先看 serve_pd 内置冒烟是否通过,再 bench**;首个 ramp 用 C=1 校准 <10min。

## 7. 关键坑(避免重踩)
- `vllm-openai` 镜像有 vllm ENTRYPOINT:跑 python 要 `--entrypoint python3`,否则被当 `vllm serve` 参数。
- `import mooncake.engine` 需 `libcuda`(要 `--gpus`),纯 CPU 容器会 `ImportError: libcuda.so.1`。
- `kv_role`:pc vs kv_both 未定,先跑 verify-kvrole;结论若为 both,改各 config 的 `KV_ROLE_MODE=both`。
- 发压连通:h200-6 需能访问 h200-2:port(`--network host` + 内网 IP);bench_ramp.sh 已内置探活报错。
- 同步:改完本地 `develop/` 用 `tar czf - -C develop <子目录> | ssh -l root h200-6 'tar xzf - -C /home/lychee/mycode/vllm/develop/'`;h200-2 走 `/ceph` 免同步。
- `ssh -l root`(两台 h200 默认 lychee 无权限)。

## 8. 记忆指针
`project_vllm_minimax_pd`(拓扑/框架/状态)· `feedback_vllm_pd_experiment_workflow`(本地写→推 h200-6、每实验一目录)·
`feedback_respond_in_chinese` · `feedback_p2p_nccl_startup`(PD 启动 6 坑)· `project_vllm_glm52_test_init`(姊妹 GLM 实验)。
