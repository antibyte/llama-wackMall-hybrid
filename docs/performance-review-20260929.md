# GTX 1660 Ti performance review, 2026-09-29

The baseline is commit `e6c3be1e3` plus the working tree present before this review. Existing uncommitted changes were preserved. Binaries, source snapshots, raw results, launcher copies, and experimental patches are in `benchmark-results/perf-review-20260929/`.

The local `build-main-sm75` has DP4A MMQ enabled. `start1660.sh` enables adaptive DFlash and explicitly sets `DECISION_SEQS=0` for its 6-GiB chat configuration. The generic build and server options remain opt-in.

## Conditions

- GTX 1660 Ti, 6 GiB; Ryzen 7 4800H; eight CPU threads; native SM75 Release build.
- GPU benchmarks ran sequentially with agollm stopped. Compilation and CPU tests ran separately from performance measurements.
- Frozen binaries use `LD_LIBRARY_PATH` pointing to their own directory, so their embedded build-directory search paths cannot select another version's libraries.
- MiniCPM5-2B Q4_K_M: full GPU offload, Flash Attention, Q8 KV, ubatch 1024, five repetitions. The table uses llama-bench's mean tokens/s after its internal warmup.
- Qwen3.6-35B-A3B UD-Q4_K_M: `start1660.sh` settings, DFlash 4/0.75, S20/W8, frequency admission with window 200 and ratio 2.5, Turbo4 KV, context 65536, prefill 1856, decode 64. Completion tests use greedy sampling, seed 42, no prompt reuse, a 64-token warmup, and three measured repetitions. The normal prompt has 498 tokens; the repeated prompt has 1496.
- All Qwen runs explicitly set `LLAMA_ARG_DECISION_SEQS=0`. The default additional decision context exhausted VRAM during the first prefill with this 6-GiB configuration. Results therefore describe the chat context with the decision endpoint disabled.

## Retained changes

### DP4A MMQ with native SM75 decode

`-DGGML_CUDA_TURING_MMQ_DP4A=ON` selects the existing DP4A MMQ kernels and their activation layout for compiled SM75 kernels. MMVQ and Flash Attention retain native SM75 implementations. Host dispatch follows the compiled architecture so its data layout agrees with the device kernel, including PTX execution on a newer GPU.

The option defaults to OFF. It is intended for GTX 16-series hardware without tensor cores; it also affects SM75 builds running on tensor-core Turing hardware, where it needs a separate benchmark.

| MiniCPM test | Native baseline | DP4A MMQ | Change |
| --- | ---: | ---: | ---: |
| pp64 | 487.45 PPS | 1236.11 PPS | +153.6% |
| pp512 | 519.63 PPS | 1447.11 PPS | +178.5% |
| pp2048 | 475.35 PPS | 1143.85 PPS | +140.6% |
| tg128 | 126.12 TPS | 126.33 TPS | +0.2% |

The Qwen 3x256 screen improved median prompt throughput from 87.42 to 136.23 PPS and median generation throughput from 35.84 to 40.75 TPS. The longer DP4A-only comparison was 39.60 versus 39.34 TPS, so the short-run decode gain was not sustained. Adaptive draft length provides the long-run TPS gain in the combined preset.

| Qwen comparison | Original baseline | Final candidate | Change |
| --- | ---: | ---: | ---: |
| Prefill, 1496 tokens, median of 3 | 162.77 PPS | 320.99 PPS | +97.2% |
| Prefill, 498 tokens in 3x1024 run | 87.42 PPS | 136.31 PPS | +55.9% |
| Decode, 3x1024 tokens, median | 39.60 TPS | 43.10 TPS | +8.8% |
| Full request, 3x1024, median wall time | 31.75 s | 27.54 s | -13.2% |

The 1496-token prefill comparison isolates DP4A with adaptation off. Its short 64-token decode phase was slower: 36.99 versus 32.64 TPS, while median complete request time fell from 11.03 to 6.72 seconds. These results are workload-specific; a faster prefill does not imply a faster continuation for every prompt.

### MoE verification fusion

The existing opt-in `GGML_CUDA_MOE_MULTI_FUSION=1` now uses the architecture/type-specific MMVQ batch limit, up to eight tokens. Four DFlash draft tokens plus the last sampled token fit in the fused path. Host limits now follow the compiled NVIDIA architecture, matching kernel launch bounds.

The isolated 3x256 screen changed median TPS from 35.84 to 36.13 (+0.8%). The 3x1024 comparison was 39.60 versus 39.80 TPS (+0.5%). Corresponding measured token IDs matched in both comparisons. These small differences should not be interpreted as a substantial independent speedup.

### Adaptive DFlash

`LLAMA_DFLASH_ADAPTIVE=1` compares wall time per emitted token for draft limits two and up to four, using 32-token probes and 256-token hold windows with 5% hysteresis. It includes draft, verification, rollback, and host overhead without adding a GPU synchronization. It requires one slot and classic DFlash; model block size, batch capacity, context capacity, and the configured maximum bound the actual draft. Context shifts and clipped limits invalidate incomplete probes.

An independent 3x1024 run measured 42.47 median TPS before the final preset repeated the gain at 43.10 TPS. Both exceed the original 39.60 TPS baseline; the range is approximately +7% to +9% on this prompt.

The persistent three-turn conversation has a different tradeoff. The original baseline generated at 44.20 aggregate TPS and took 45.56 seconds of client wall time. The DP4A candidate with fixed limit four generated at 41.83 TPS. Adaptive DFlash recovered this to 43.20 TPS, with a second conversation run at 43.02 TPS. Complete conversation time fell to 43.00 and 43.17 seconds: about 5% faster overall despite 2% to 3% lower decode TPS than the original baseline. Fixed limit two reached 42.31 TPS on the same conversation scenario. Generated answers and subsequent histories differ, so this is not a comparison of identical token sequences.

The GTX 1660 preset opts in with `DFLASH_ADAPTIVE=1`. Set that value to 0 in `start1660.sh` for fixed-limit comparisons; the server-wide default remains off.

### MTP and warm-cache corrections

MTP catch-up now offsets `batch.embd`, a `float *`, by `n_embd` rather than a byte count. The old offset skipped three extra embedding rows and could exceed the allocation. Fixed and original 128-token benchmark outputs matched, but the fix is required for correctness regardless of throughput.

Warm-cache accounting classifies a completed graph using its last published lookup table. Async completions and new admissions update the next table, which is uploaded once per layer after the graph's counts are consumed. The isolated screen had matching token IDs and no meaningful throughput change.

### Benchmark coverage

`tools/bench_hybrid_client.py --chat-turns-file` accepts a JSON array of user prompts and sends consecutive `/v1/chat/completions` requests through slot 0. It preserves the full assistant message, including reasoning, and records processed and cached prompt tokens separately. Chat output includes requests, responses, timings, and token hashes. Nonstream chat does not report TTFT.

The client verifies that the received token-ID count matches `timings.predicted_n`. If the existing SSE endpoint omits IDs while buffering incomplete UTF-8, `token_count_complete` is false and `token_sha256` is null. This does not change server timing counters. Older captured runs with incomplete IDs are flagged by the summary tool rather than treated as complete hash comparisons.

DP4A and native matrix multiplication need not produce identical floating-point results or greedy continuations. CUDA/CPU numerical tests are the kernel correctness check; token hashes only establish equivalence for the specific complete outputs that match.

## Rejected experiments

| Experiment | Comparison | Decision |
| --- | --- | --- |
| Pinned whole-tensor staging | 1496-token Qwen prompt: 323.02 to 264.78 PPS, -18.0%; corresponding outputs match | Removed from production code; patch retained with artifacts |
| Frequency-based warm eviction | Qwen 3x1024: 39.34 to 38.06 median TPS, -3.3% | Removed; existing LRU eviction retained |
| MTP catch-up chunks 8 to 64, native MMQ | 118.38 to 112.94 median PPS, -4.6%; corresponding outputs match | Keep the measured eight-token default |

The first adaptive-DFlash run was disabled by an overly strict type-list check and is a control run, not an adaptive performance result. The regular parser includes a `NONE` entry alongside `DFLASH`; the corrected gate ignores that placeholder. Subsequent measured runs explicitly logged activation and both selected limits.

The DP4A MTP comparison also favored keeping eight-token catch-up chunks: 143.23 PPS versus 140.97 PPS with 64-token chunks, with identical corresponding output IDs.

## Validation

- Release builds completed with the DP4A option enabled.
- 140 quantized matrix-multiplication and 181 MoE/fusion cases passed CUDA-versus-CPU checks, including Q2_K, Q4_K, Q5_K, Q6_K, IQ formats, FP4 formats, negative expert IDs, and multi-token batches.
- Seven benchmark-client tests passed, including preserved multi-turn reasoning/history, partial-result handling, SSE errors, and incomplete token-ID detection.
- Expert warm-cache, expert adaptation, and argument-parser tests passed. The argument-parser download checks required network access outside the sandbox.
- MTP tests cover exact prompt lengths 63, 64, 65, and 129, with complete 64-token responses and no prompt caching. Native catch-up experiments passed with target ubatch 64 and 256; the final DP4A build was also checked with ubatch 64 and the retained eight-token catch-up limit.
- The broad architecture test passed its Qwen checks but stops on a pre-existing BailingMoE3 test fixture missing `bailingmoe3.attention.key_length_mla`. The frozen baseline reproduces exactly the same failure with the same seed. This unrelated fixture was not changed.

The initial Qwen run with the extra decision context is preserved as a failed VRAM experiment. `adaptive-chat-512` is preserved as the inactive-regulator control. Neither is counted as a successful optimization trial.

## Reproduction

Build the SM75 candidate as shown in the README. Run the dense benchmark with the same settings:

```bash
GGML_CUDA_MMVQ_Q4_K_NCOLS1_ROWS=2 GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS=2 \
  build-main-sm75/bin/llama-bench \
  -m /home/andi/models/minicpm5-2b/MiniCPM5-2B-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -t 8 -r 5 -p 64,512,2048 -n 128 -ub 1024 -o jsonl
```

The artifact runner creates a local launcher on port 18091, records overrides, disables the additional decision context, runs the client, and terminates only its own server:

```bash
python3 benchmark-results/perf-review-20260929/run_server.py new-run \
  --bin build-main-sm75/bin --tokens 1024 --repeats 3

python3 benchmark-results/perf-review-20260929/run_server.py new-chat-run \
  --bin build-main-sm75/bin --tokens 512 \
  --chat benchmark-results/perf-review-20260929/chat-turns.json

python3 benchmark-results/perf-review-20260929/run_server.py fixed-limit-control \
  --bin build-main-sm75/bin --set DFLASH_ADAPTIVE=0 --tokens 1024 --repeats 3

python3 benchmark-results/perf-review-20260929/summarize_results.py \
  --output benchmark-results/perf-review-20260929/summary.md
```

Use a new label for each run. Artifacts are local benchmark data and are not required to build the project.
