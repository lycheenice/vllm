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
| 6 | test2-mooncake-pd | ⏭️ 跳过 | 27.8GB 镜像经 a100-2 中转仅 3.4MB/s(需1h+);且 test3≈test1 已证 connector 非瓶颈,mooncake 极可能≈test1;边际价值低,深夜中止 |
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
