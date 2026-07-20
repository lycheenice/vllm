# vLLM NIXL PD Disaggregation Benchmark Report

## MiniMax-M2.5 on 8×H200 SXM 141GB (h200-2)

**Date**: 2026-07-19 ~ 2026-07-20  
**vLLM version**: v0.25.0  
**Model**: MiniMax-M2.5 (230GB, 125 shards, fp8 dynamic w8a8, 256 experts top-8, 62 layers, GQA 48 heads / 8 kv_heads)  
**Benchmark tool**: kvcache-benchmarks `ramp_test.py`, dataset `codex_swebenchpro.json` (610 records, ~30 turns/trial, multi-turn agent traces)  

---

## 1. Experiment Setup

### 1.1 Hardware
- 8× H200 SXM 141GB, NVLink/NVSwitch interconnect
- No external internet/DNS — all artifacts local

### 1.2 FP8 + TP8 Compatibility Fix
MiniMax-M2.5 has `intermediate_size=1536`. Under TP8, each rank gets `1536/8 = 192`, which is not divisible by FP8 block size 128. This causes:
```
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128
```
**Fix**: `--enable-expert-parallel` (EP). Experts are distributed across ranks (32 experts/rank), keeping `intermediate_size=1536` intact (divisible by 128). All phases use `--enable-expert-parallel`.

### 1.3 Configurations

| Phase | Config | GPUs | Description |
|-------|--------|------|-------------|
| Phase 1: Baseline | TP8 single instance | 8 GPU | Single vLLM instance, TP=8, inline prefill+decode |
| Phase 2: PD Basic | P4 + D4 (no KV return) | 4+4 GPU | Prefill on 4 GPUs, decode on 4 GPUs, NIXL P→D KV transfer, no D→P return |

### 1.4 PD Basic Architecture
- **Prefill node** (`minimax-prefill`): TP=4, port 8100, `--kv-transfer-config` with NixlConnector
- **Decode node** (`minimax-decode`): TP=4, port 8200, NixlConnector (pull mode)
- **External proxy** (`minimax-proxy`): FastAPI `disagg_proxy_demo.py`, port 8000
  - Turn 1: sends to prefill → KV transferred to decode → decode generates
  - Turn 2+: sends to decode directly (no KV return to prefill, no prefix cache reuse on P)

### 1.5 Benchmark Parameters
- Concurrency levels: C = 1, 4, 16, 32, 64
- Trials per concurrency: scales with C (5, 20, 80, 160, 320)
- Streaming enabled, max_tokens=4096
- Dataset: multi-turn coding agent traces (context up to 100k+ tokens)

---

## 2. Results

### 2.1 Summary Table

| C | Phase | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | Lat P50 | Throughput | OK | Fail | Duration |
|---|-------|----------|----------|----------|----------|---------|------------|----|------|----------|
| 1 | Baseline | 237ms | 6,222ms | 7.7ms | 8.5ms | 5,843ms | 112 tok/s | 112 | 3 | 896s |
| 1 | PD Basic | 561ms | 14,275ms | 8.5ms | 9.3ms | 4,448ms | 90 tok/s | 145 | 0 | 1,040s |
| 4 | Baseline | 205ms | 1,769ms | 9.2ms | 11.9ms | 4,683ms | 321 tok/s | 638 | 2 | 1,556s |
| 4 | PD Basic | 437ms | 4,968ms | 11.1ms | 13.9ms | 6,092ms | 276 tok/s | 641 | 0 | 1,779s |
| 16 | Baseline | 233ms | 1,118ms | 16.0ms | 23.9ms | 8,097ms | 803 tok/s | 2,255 | 22 | 2,489s |
| 16 | PD Basic | 523ms | 3,992ms | 28.3ms | 36.2ms | 11,316ms | 488 tok/s | 2,645 | 0 | 3,947s |
| 32 | Baseline | 255ms | 1,093ms | 29.4ms | 39.0ms | 12,706ms | 883 tok/s | 4,591 | 39 | 4,400s |
| 32 | PD Basic | 578ms | 6,675ms | 50.3ms | 62.1ms | 23,401ms | 615 tok/s | 1,881 | 3,469 | 2,469s |
| 64 | Baseline | 304ms | 1,196ms | 49.6ms | 62.2ms | 23,166ms | 1,085 tok/s | 9,081 | 85 | 7,332s |
| 64 | PD Basic | — | — | — | — | — | — | 0 | 10,638 | 3.5s |

### 2.2 Throughput Comparison

```
Throughput (tok/s)
1200 ┤                          ██ 1085
1000 ┤
 800 ┤    ██ 803    ██ 883
 600 ┤                      ██ 615
 400 ┤              ██ 488
 300 ┤  ██ 321  ██ 276
 100 ┤██ 112  ██ 90
     └─────┬────┬─────┬─────┬─────
          C=1  C=4  C=16  C=32  C=64

         ■ Baseline   ■ PD Basic
```

### 2.3 TTFT Comparison

```
TTFT P50 (ms)
600 ┤  ██ 561          ██ 578
500 ┤      ██ 523
400 ┤  ██ 437
300 ┤                          ██ 304
200 ┤  ██ 237  ██ 205  ██ 233  ██ 255
    └─────┬────┬─────┬─────┬─────
         C=1  C=4  C=16  C=32  C=64

         ■ Baseline   ■ PD Basic
```

### 2.4 TPOT Comparison

```
TPOT P50 (ms)
 60 ┤                      ██ 50.3
 50 ┤                          ██ 49.6
 30 ┤              ██ 28.3  ██ 29.4
 20 ┤      ██ 16.0
 10 ┤  ██ 7.7  ██ 9.2
  8 ┤  ██ 8.5  ██ 11.1
    └─────┬────┬─────┬─────┬─────
         C=1  C=4  C=16  C=32  C=64

         ■ Baseline   ■ PD Basic
```

---

## 3. Analysis

### 3.1 TTFT: PD Basic is 2-3× Slower

At all concurrency levels, PD Basic has significantly higher TTFT than baseline:

| C | Baseline TTFT P50 | PD Basic TTFT P50 | Overhead |
|---|-------------------|-------------------|----------|
| 1 | 237ms | 561ms | +137% |
| 4 | 205ms | 437ms | +113% |
| 16 | 233ms | 523ms | +124% |
| 32 | 255ms | 578ms | +127% |

**Root cause**: PD Basic adds overhead from:
1. **NIXL KV transfer latency**: Prefill computes KV cache, then transfers it to decode node via NIXL (RDMA/NVLink), before decode can start generating.
2. **Proxy overhead**: External FastAPI proxy adds an extra hop and serialization.
3. **No prefill batching optimization**: With only 4 GPUs for prefill (vs 8 in baseline), prefill compute is slower for long prompts.

### 3.2 TPOT: PD Basic is 50-70% Slower at High Concurrency

| C | Baseline TPOT P50 | PD Basic TPOT P50 | Overhead |
|---|-------------------|-------------------|----------|
| 1 | 7.7ms | 8.5ms | +10% |
| 4 | 9.2ms | 11.1ms | +21% |
| 16 | 16.0ms | 28.3ms | +77% |
| 32 | 29.4ms | 50.3ms | +71% |

**Root cause**: 
- **TP4 decode has half the throughput of TP8**: With 4 GPUs vs 8, decode batch processing is slower per step.
- At C=16+ decode node saturates: 4 GPUs handling all decode requests, while baseline spreads across 8 GPUs.
- PD Basic throughput peaks at ~615 tok/s (C=32), while baseline peaks at ~1,085 tok/s (C=64) — **43% lower peak throughput**.

### 3.3 Error Rates: PD Basic Catastrophic at C=32+

| C | Baseline Fail% | PD Basic Fail% |
|---|----------------|----------------|
| 1 | 2.6% | 0% |
| 4 | 0.3% | 0% |
| 16 | 1.0% | 0% |
| 32 | 0.8% | 64.8% (3,469/5,350) |
| 64 | 0.9% | 100% (10,638/10,638) |

**C=32 PD Basic**: 65% failure rate — decode node overwhelmed, requests timing out.  
**C=64 PD Basic**: 100% failure in 3.5 seconds — decode node immediately rejects all requests (likely OOM or connection refused).

Baseline errors were all HTTP 400 (context length >100k), not performance-related.  
PD Basic errors at C=32+ are performance-related (timeouts, connection errors).

### 3.4 Cache Hit Rate

Both phases show 0% client cache rate — vLLM does not return a `cached_tokens` field that the benchmark tool can detect. However, server-side logs show:
- **Baseline**: prefix cache hit rate 91-96% (for multi-turn, turn 2+ reuses prefill cache)
- **PD Basic**: prefill prefix cache hit rate 92.8% on P node, but this doesn't help because:
  - Turn 1: P does prefill → KV transferred to D → D generates
  - Turn 2+: Proxy sends to D directly. D has the KV from previous turn, but P does NOT have the new prompt's prefix. **No prefix cache reuse on P for subsequent turns.**
  - This means each new turn's prompt prefix is re-computed on D (decode node), which is inefficient.

### 3.5 Latency

| C | Baseline Lat P50 | PD Basic Lat P50 |
|---|------------------|------------------|
| 1 | 5,843ms | 4,448ms (-24%) |
| 4 | 4,683ms | 6,092ms (+30%) |
| 16 | 8,097ms | 11,316ms (+40%) |
| 32 | 12,706ms | 23,401ms (+84%) |

Interestingly, at C=1 PD Basic has lower latency than baseline. This is because PD offloads decode to a dedicated node, reducing per-request contention. But as concurrency increases, the decode node bottleneck dominates.

---

## 4. Key Findings

### 4.1 PD Basic (P→D only, no KV return) is Net Negative for This Workload

- **Peak throughput**: 615 tok/s (PD) vs 1,085 tok/s (baseline) — **43% worse**
- **TTFT**: 2-3× worse across all concurrency levels
- **TPOT**: 10-77% worse, especially at high concurrency
- **Scalability**: Collapses at C=32 (65% errors) and C=64 (100% errors)
- **Only advantage**: Slightly lower latency at C=1 (single user, no contention)

### 4.2 Root Causes

1. **No D→P KV return**: The basic proxy (`disagg_proxy_demo.py`) doesn't return KV cache from D to P after decode. Subsequent turns in a multi-turn conversation lose prefix cache on P, forcing re-prefill on D (which is slow at decode).

2. **TP4 vs TP8**: Splitting 8 GPUs into 4+4 halves the compute power of each phase. For a model this large (230GB), neither P4 nor D4 has enough compute to match TP8.

3. **NIXL transfer overhead**: KV transfer adds latency to every first-turn request, with no benefit for subsequent turns (KV is on D, not P).

4. **No prefix cache continuity**: In baseline multi-turn, turn 2+ hits prefix cache (92-96% hit rate). In PD Basic, P only sees turn 1 — turns 2+ go to D, which has the KV but must compute new prompt tokens without prefill-optimized batching.

### 4.3 What Would Help (Phase 3+ Recommendations)

1. **Bidirectional PD (Phase 3)**: Use `disagg_proxy_multiturn_autoid.py` with D→P KV return. After decode, return KV to P so turn 2+ can hit prefix cache on P. This is the critical missing piece.

2. **Asymmetric GPU allocation**: With EP, consider P2+D6 or P6+D2 depending on workload (prefill-heavy vs decode-heavy). For multi-turn agent traces, prefill dominates, so more GPUs on P may help.

3. **Chunked prefill tuning**: Increase `--max-num-batched-tokens` on P to batch more prefill requests, improving prefill throughput.

4. **Max num seqs**: Increase `--max-num-seqs` on D to handle more concurrent decode streams.

5. **KV cache lease**: Tune NIXL `kv_lease_seconds` to prevent premature KV eviction during long multi-turn conversations.

---

## 5. Configuration Details

### 5.1 Common vLLM Arguments (All Phases)
```
--model /data1/models/MiniMax-M2.5/
--enable-expert-parallel
--tensor-parallel-size <TP>
--quantization fp8
--kv-cache-dtype fp8
--max-model-len 102400
--max-num-seqs 256
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.95
--use-prefix-cache
--no-enable-prefix-caching n/a (use-prefix-cache is the v0.25.0 flag)
```

### 5.2 PD-Specific Arguments
**Prefill node (TP=4)**:
```
--port 8100
--kv-transfer-config '{"role": "producer", "local_port": 55555, "nixl_connector": true}'
```

**Decode node (TP=4)**:
```
--port 8200
--kv-transfer-config '{"role": "consumer", "local_port": 55556, "remote_host": "127.0.0.1", "remote_port": 55555, "nixl_connector": true}'
```

### 5.3 Benchmark Result Locations (h200-2)
- **Baseline**: `/home/lychee/mycode/kvcache-benchmarks/results/MiniMax-M2.5_vllm_baseline_20260719_174503/`
- **PD Basic**: `/home/lychee/mycode/kvcache-benchmarks/results/MiniMax-M2.5_vllm_pd-basic_20260719_230622/`

Each directory contains: `c01/`, `c04/`, `c16/`, `c32/`, `c64/` subdirs with `result.json` + `run.log`, plus `summary.json` and `manifest.json`.

---

## 6. Conclusion

PD Basic disaggregation (P→D only, no KV return) is **net negative** for multi-turn agent workloads on MiniMax-M2.5 with 8×H200:

- **Throughput**: 43% lower peak (615 vs 1,085 tok/s)
- **TTFT**: 2-3× worse due to NIXL transfer + proxy overhead
- **TPOT**: 10-77% worse due to TP4 decode (half the GPUs)
- **Scalability**: Catastrophic failure at C≥32 (decode node overwhelmed)
- **Only advantage**: Lower per-request latency at C=1 (no contention)

The fundamental issue is that **without D→P KV return**, multi-turn conversations lose prefix cache reuse on the prefill node, eliminating the primary benefit of disaggregation. Phase 3 (bidirectional PD with `disagg_proxy_multiturn_autoid.py`) is needed to properly evaluate PD's potential.
