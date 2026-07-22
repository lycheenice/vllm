# MiniMax-M2.5 vLLM v0.25.0 PD 分离实验对比报告

模型 MiniMax-M2.5 (FP8 blockquant, 256 experts) · 8×H200 · 数据集 codex_swebenchpro · levels [1, 4, 8, 16] · max-turns 4 · trials/user 4 · max-tokens 256 · max-model-len 65536

数据来源(每实验最新 summary.json):
- base1 TP8+EP8 (整机上界): `experiments/base1-tp8/results/base1-tp8_20260721_212609/summary.json`
- base2 TP4 (算力减半): `experiments/base2-tp4/results/base2-tp4_20260721_214626/summary.json`
- test1 PD nixl GPU直传: `experiments/test1-nixl-pd/results/test1-nixl-pd_20260721_224311/summary.json`
- test3 PD nixl CPU中转: `experiments/test3-nixl-cpu-bypass/results/test3-nixl-cpu-bypass_20260721_231532/summary.json`

## 吞吐 (output tok/s)

| 实验 | C=1 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| base1 TP8+EP8 (整机上界) | 18 | 70 | 131 | 238 |
| base2 TP4 (算力减半) | 14 | 55 | 109 | 192 |
| test1 PD nixl GPU直传 | 12 | 47 | 83 | 118 |
| test3 PD nixl CPU中转 | 12 | 46 | 80 | 127 |

## E2E 延迟 mean

| 实验 | C=1 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| base1 TP8+EP8 (整机上界) | 12.0 | 13.3 | 13.9 | 15.0 |
| base2 TP4 (算力减半) | 17.1 | 16.4 | 16.8 | 19.4 |
| test1 PD nixl GPU直传 | 18.5 | 19.6 | 22.0 | 29.9 |
| test3 PD nixl CPU中转 | 19.4 | 19.6 | 22.3 | 27.5 |

## TTFT p50

| 实验 | C=1 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| base1 TP8+EP8 (整机上界) | 280 | 310 | 373 | 380 |
| base2 TP4 (算力减半) | 1255 | 452 | 494 | 489 |
| test1 PD nixl GPU直传 | 4190 | 4301 | 6146 | 15637 |
| test3 PD nixl CPU中转 | 5227 | 3254 | 3881 | 6913 |

## TPOT mean

| 实验 | C=1 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| base1 TP8+EP8 (整机上界) | 53 | 54 | 56 | 61 |
| base2 TP4 (算力减半) | 59 | 66 | 67 | 77 |
| test1 PD nixl GPU直传 | 60 | 60 | 61 | 63 |
| test3 PD nixl CPU中转 | 61 | 66 | 73 | 85 |

## 错误率

| 实验 | C=1 | C=4 | C=8 | C=16 |
|---|---|---|---|---|
| base1 TP8+EP8 (整机上界) | 0.0 | 0.0 | 0.8 | 0.4 |
| base2 TP4 (算力减半) | 0.0 | 0.0 | 0.8 | 0.4 |
| test1 PD nixl GPU直传 | 0.0 | 0.0 | 1.6 | 0.8 |
| test3 PD nixl CPU中转 | 0.0 | 0.0 | 1.6 | 0.8 |

## 归因(按 C=16 output tok/s)

- **TP8→TP4 算力代价** (base2 vs base1): -19%  (238→192 tok/s)
- **PD 分离本身开销** (test1 vs base2): -38%  (192→118 tok/s)
- **CPU 中转 vs GPU 直传** (test3 vs test1): +8%  (118→127 tok/s)

## 结论

1. **PD 分离对该多轮 agent 负载是净负收益**:test1(PD)相对 base2(同 TP4 算力)C=16 吞吐 -38%、E2E +54%。主因是 **TTFT 爆炸**(test1 C=16 TTFT p50 15637ms vs base2 489ms,~32×):prefill 集中在 P 实例排队 + KV 传输 + proxy 转发。
2. **TPOT 是 PD 唯一优势**:专用 decode 实例使 test1 TPOT 保持平稳(~63ms),而 base2 在 C=16 涨到 77ms。即 PD 换来了平滑的每 token 延迟,却牺牲了首 token 与整体吞吐。
3. **CPU 中转 ≈ GPU 直传**:test3 vs test1 在噪声内(C=16 +8%)。decode 日志实测 KV xfer ~1.1s/次,但相对 20-30s 的多轮 E2E 占比极低,故 kv_buffer_device=cpu 不是瓶颈——**PD 分离的结构性开销(重复 prefill/排队/proxy)才是**。
4. **kv_role**:verify-kvrole 实测 pc(producer/consumer)与 both 输出均与 golden 一致、KV 均真实传输(证据 66/70),v0.25.0+Nixl 用默认 pc 即可,无需 kv_both。
5. **整机最优仍是 base1**(TP8 attn + EP8 MoE):C=16 238 tok/s / E2E 15s,全面优于任何 TP4/PD 方案。纯 TP8 因 FP8 blockquant(192%128)不可行,EP 是整机跑该 MoE 的必需项。

> 与既有分析 `vllm_pd_analysis_and_optimization_20260721.md` 结论一致并给出量化。后续优化方向(P0):MultiConnector 加 L2 OffloadingConnector + 双向 D→P KV 回传,消除多轮场景下 decode 端的重复 prefill。