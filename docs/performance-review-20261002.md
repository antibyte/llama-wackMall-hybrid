# GTX 1660 Ti performance review, 2026-10-02

Baseline: commit `e8d475e79` with `build-main-sm75` (DP4A MMQ). All GPU runs used a stopped agollm service, a warm page cache, and the GTX 1660 Ti Mobile (TU116, 6 GiB) with the Ryzen 7 4800H. Qwen runs use `benchmark-results/perf-review-20260929/run_server.py` (3x1024 tokens, greedy, 498-token prompt); small-model chat runs use each launcher's own sampling defaults with three 400-token chat requests.

## Findings that do not need code

### agollm memory pressure

With agollm running, the Vega iGPU workers held up to 9.6 GiB of GTT system memory. The 22.6 GB Qwen GGUF then no longer fits into the page cache and cold experts are read from NVMe during decode: the same Qwen benchmark ran at 27.8-32.0 TPS instead of 42-43 TPS, with IO pressure up to 13%. Nex (19.7 GB), Ornith (18.7 GB) and Hy-MT2 (17.0 GB) are affected the same way.

### CPU expert path is at the DRAM limit

A standalone harness using the real ggml Q4_K/Q5_K CPU dot products with the cold-expert work split reached 38.3-39.7 GB/s with 8 threads; a pure AVX2 streaming read reached 37-41 GB/s. 2 MiB pages changed the kernel result by less than 1%. Faster CPU expert kernels cannot raise Qwen decode; only fewer cold bytes or CPU/GPU overlap can.

## Code changes

### Slow MMA on Turing without tensor cores

GTX 16xx and MX450/550 report SM 7.5 but execute HMMA far below FP32 SIMT rate. `ggml_cuda_device_info::slow_mma` is set for the device names llama.cpp already lists as lacking tensor cores (`GGML_CUDA_SLOW_MMA=0/1` overrides). With it:

- cuBLAS FP16 GEMMs run as FP32 SGEMM. nsys showed `volta_h884gemm`/`cutlass ... tensorop_h884gemm` at about 0.75 TFLOPS taking 77% of an LFM2.5-VL image request.
- Flash Attention uses the tile kernel instead of the MMA kernel for batches.

| Workload | MMA | slow_mma | Change |
| --- | ---: | ---: | ---: |
| LFM2.5-VL-3B, 2327-token image request | 14.45 s | 4.85 s | 3.0x |
| Spark-X2.5-4B pp512 / pp2048 | 782 / 564 | 889 / 824 | +14% / +46% |
| MiniCPM5-2B pp512 / pp2048 | 1423 / 1115 | 1619 / 1501 | +14% / +35% |
| LFM2.5-1.2B pp512 / pp2048 | 3335 / 2990 | 3454 / 3360 | +4% / +12% |
| Ling-3.0-tiny pp512 / pp2048 (ub 1024) | 1615 / 1411 | 1782 / 1833 | +10% / +30% |
| Qwen3.6 DFlash decode, 3x1024 median | 43.26 | 42.73 | within run spread |

tg128 is unchanged on all dense models (decode uses the vector kernel). `test-backend-ops` passed FLASH_ATTN_EXT 2880/2880 and F16 MUL_MAT 270/270 with the flag active.

### Bucketed exact top-p

With `top_k` disabled (Spark, MiniCPM model-card sampling, or GGUF defaults), top-p ran a softmax plus partial sort over the full vocabulary for every token: 1.23-1.56 ms on 131072 entries, 8.5 ms for flat distributions. The new path computes the same probabilities in the same order, histograms their mass by distance to the maximum logit, and sorts only the buckets that can contain the cut: 0.62 ms. 393/400 random distributions produced identical nucleus sets; the 7 others had identical sizes and differ only in tied logits at the cut, whose order the old partial sort does not define either.

Backend (GPU) top-p argsorts the whole vocabulary and added 2.3 ms per token on Spark, so CPU sampling is now faster for these launchers:

| Launcher | Backend sampling | CPU sampling | Prompt latency |
| --- | ---: | ---: | --- |
| `startspark.sh` chat | 66.0 TPS | 73.2 TPS (+11%) | 0.85 s -> 0.30 s |
| `start-minicpm5-2b.sh` chat | 96.0 TPS | 112.2 TPS (+17%) | 1.1 s -> 0.14 s |
| `start-lfm25-1.2b.sh` chat (top_k 50) | 234 TPS | 235 TPS | 150 ms -> 66 ms |
| `start-ling-tiny.sh` chat (top_k 20) | 133 TPS | 130 TPS | kept backend sampling |

### Power profile switching

`llama-server --power-busy-cmd CMD --power-idle-cmd CMD --power-idle-delay MS` (`LLAMA_ARG_POWER_*`). The busy command runs synchronously before a completion, embedding, rerank or decision task starts; the idle command runs after the delay once no server of the same user is working. Servers coordinate through `flock` and per-process marker files in `/tmp/llama-power-$UID` (`LLAMA_POWER_DIR`); stale markers of dead processes are removed. The state is released when the server loop ends, so a shutdown inside the idle delay still runs the idle command.

The launchers use the system76-power D-Bus methods (`Performance` / `Battery`, about 20-120 ms per switch). Qwen 3x1024 with q4_0 KV: Performance 45.1-45.2 TPS, Balanced 45.21 TPS, Battery without switching 33.29 TPS (-26%, prefill -12%). The switch therefore preserves full speed while the idle state saves power. The Battery profile also caps the CPU at 1.45 GHz and dims the panel to 10%; use `Balanced` as idle command to keep the brightness.

## Launcher changes

| Launcher | Change | Measured |
| --- | --- | --- |
| `start1660.sh` | target KV q4_0/q4_0 instead of Turbo4 | 6 runs each: 44.98 vs 42.76 TPS (+5.2%); same S=20/W=8 fit. The fork's own quality table has q4/q4 at PPL +0.96%, KLD 0.015 vs Turbo4/Turbo4 +1.58%, 0.028 |
| `start1660.sh` | `LLAMA_EXPERT_CPU_DOWN_PREFETCH=0` | +0.5% to +1.2% on identical token streams |
| `start1660.sh` | final configuration | 44.85 TPS median (44.85, 46.12, 44.18) |
| `start-lfm25-1.2b.sh` | SM75 build, Q4_K/Q6_K MMVQ rows 2 | tg128 275.2 vs 222.3 (+23.8%) |
| `start-lfm25-vl-3b.sh` | SM75 build, KVFlash off | tg128 126.8 vs 120.5; image request 5.1 s vs 14.8 s |
| `start-hy-mt2-30b.sh` | SM75 build, `profiles/hy-mt2-translate.csv` | holdout translation 27.5 vs 20.8 TPS (+32%); `PROFILE_KIND=none` had pinned no experts |
| `start-nex-n25-mini*.sh` | `profiles/nex-n25-mini-code.csv` | holdout coding prompt 44.3 vs 41.8 TPS (+5.9%) |
| all CUDA launchers | `DECISION_SEQS=0` | agollm sends `/v1/decision` to its own worker; Nex +0-4% on identical tokens |
| all launchers | power profile switching | see above |

## Rejected

| Experiment | Result |
| --- | --- |
| DFlash draft KV q4_0 instead of Turbo4 | 42.42 vs 42.73 TPS, no acceptance gain |
| Hy-MT2 KV q8_0 | auto-fit drops to S=4, 24.3 TPS |
| Hy-MT2 KV q4_0 | S=16, 26.4 TPS vs Turbo4 27.5 |
| Performance instead of Balanced during work | no TPS difference on Qwen |

## Open items

- Qwen verify rebuilds the graph on about 40% of steps (`reuse/rebuild = 299/201`, 4.2 ms each) because adaptive DFlash alternates block sizes; keeping one graph per block size would save about 1.7 ms per verify.
- CPU cold experts and GPU hot experts of the same layer still run serially: the cold inputs are read back after the hot kernels in stream order. An event-based early readback with a deferred cold upload could hide the hot-expert time, estimated at 5-10% of a verify step.
- The Ling and Nex servers segfault during shutdown after "cleaning up before exit"; the vendored binary from 2026-09-29 behaves the same.
- Gemma4 and Ornith were not re-measured.
