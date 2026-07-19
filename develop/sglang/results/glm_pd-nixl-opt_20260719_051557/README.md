# GLM-5.2-W4AFP8 sglang NIXL PD-opt benchmark (2026-07-19 05:15)

## Environment

- Model: GLM-5.2-W4AFP8 (MLA, 78 layers, w4afp8, 40 shards, model_impl=GlmMoeDsaForCausalLM, served_model_name=glm)
- Hardware: h200-2, 8xH200 SXM 141GB (1P+1D, each TP4)
- Container: br-harbor01.birentech.com/sucloud_test/h200-serving/lmsysorg/sglang:v0.5.15.post1-cu129 (sglang 0.5.15.post1, nixl 1.3.1)
- PD backend: NIXL (intra-node NVLink), `disaggregation_transfer_backend=nixl`
- Bench: kvcache-benchmarks, codex_swebenchpro_traces (610 records, p50=30 turns, p90=57 turns, gpt output p50=246 tok / p90=1133 tok), all turns, max_tokens=4096, trials_per_user=5, levels 8/16/32/64/128

## PD-opt config (exp1)

- prefill: `SGLANG_DISAGGREGATION_QUEUE_SIZE=8`, `THREAD_POOL_SIZE=12`, `BOOTSTRAP_TIMEOUT=600`, `WAITING_TIMEOUT=600`, `chunked_prefill_size=32768`, `max_prefill_tokens=16384`, `max_running_requests=64`, `mem_fraction_static=0.85`, `context_length=300000`, `kv_cache_dtype=fp8_e4m3`, `disable_cuda_graph=True`
- decode: same + `disaggregation_decode_enable_radix_cache=True` (PD radix cache on decode side)
- router: `python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:8001 --decode http://127.0.0.1:8002 --host 0.0.0.0 --port 8000`

## L1 / L2 cache state (from server_args dump)

- `enable_hierarchical_cache=False` → **hicache (L2) NOT enabled**
- `hicache_ratio=2.0`, `hicache_size=0` (defaults, inactive)
- `disable_radix_cache=False` → L1 radix cache enabled
- L1 KV pool size per GPU: ~17.9 GB (weights occupy 94.5 GB/GPU; `avail mem` after weight load = 43.5 GB; after warmup + pool allocation = 17.9 GB)
- `max_total_tokens=None` (auto-computed from pool)
- `page_size=64`, `radix_eviction_policy=lru`
- `prefill_only_disable_kv_cache=False` (prefill keeps L1)

## Results

| level | trials | returncode | succeeded | ttft p50/p90 (s) | tpot p50/p90 (ms) | cache_rate (mean) | duration (s) | status |
|---|---|---|---|---|---|---|---|---|
| c8  | 40  | 0 | full | 4.95 / 13.73 | 13.48 / 14.84 | **0.92** (p50 0.99) | 1953 | OK |
| c16 | 80  | 0 | full | 94.80 / 184.08 | 18.52 / 54.13 | **0.22** (p50 0.03) | 2723 | cache collapsed |
| c32 | 160 | 0 | ?   | n/a (no lat fields) | n/a | n/a | 133 | **abnormal** — 160 trials in 133s is implausible; likely all errored/empty |
| c64 | —   | — | —   | stuck at lat=301s | — | 0% | >2h44m | **halted manually** (was burning time on 5-min-per-request timeouts) |
| c128 | —   | — | —   | — | — | — | — | not run |

## Failure analysis

1. **c08 OK, c16 cache_rate collapse (0.92 → 0.22)**: L1 radix cache eviction under concurrency. With only ~17.9 GB/GPU L1 pool and no L2 hicache to fall back on, doubling concurrency from 8 to 16 immediately evicts prior turns' KV.
2. **c64 mass 301 s timeouts**: `lat=301s` matches the bench client's 5-min timeout. Under c64 concurrent load the prefill side shows `#bootstrap-req: 7` backlog with `#inflight-req: 1-2` — the prefill cannot keep up, and `chunked_prefill_size=32768` lets a single large prefill monopolize a batch. The router health-check for :8001 (prefill) TimedOut, which the router reports misleadingly as "No available decode workers (all circuits open or unhealthy)".
3. **c32 abnormal summary**: summary.json lacks `ttft_ms`/`tpot_ms`/`cache_rate_dist` fields but claims `total_trials=160, returncode=0, duration=133.55s`. Likely a bench-client fast-fail mode where all requests 503'd quickly. Needs separate inspection of `result.json`.
4. **Early 6× c08 attempts at 02:43–05:08**: all 1343/0 failed with HTTP 503 "No available decode workers". Root cause same as above: router circuit breaker opened after prefill health-check TimedOut during cold-start; misleading error message.

## Next-move hypotheses (not yet applied)

- Open L2 hicache (`--enable-hierarchical-cache --hicache-ratio 2.0` or higher) to relieve L1 eviction.
- Drop `chunked_prefill_size` from 32768 to 8192–16384 so a single long prefill does not monopolize the batch.
- Add `--health-check-interval-secs 30` (or disable circuit breaker) on the PD router to avoid false-negative eviction during prefill warmup/slow batches.
- Consider raising L1 pool by lowering `mem_fraction_static` is not viable (weights already 94.5 GB). Reducing `context_length` from 300000 to 131072 would tighten radix cache accounting and possibly alleviate pressure.
