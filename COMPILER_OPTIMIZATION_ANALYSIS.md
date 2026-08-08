# Compiler and remaining optimization analysis

Date: 2026-08-07

This note separates compiler experiments from algorithmic and CUDA-kernel
work. No throughput claim is made for a compiler variant until it has passed a
fresh-server, hash-checked A/B on the target host.

## Existing compiler state

The production `build-hybrid` is already more optimized than a plain generic
Release build:

- C and C++ use `-O3 -DNDEBUG`.
- `GGML_NATIVE=ON` gives the complete ggml CPU backend `-march=native`.
  Consequently the Q4_K/Q5_K cold-expert AVX2 dot-product code already uses
  the Ryzen 7 4800H instruction set, including AVX2, FMA, F16C, and BMI2.
- CUDA is compiled as native sm_75 SASS and ggml CUDA already supplies
  `-use_fast_math`.
- `GGML_CUDA_FORCE_MMQ` is off, as required for the measured Turing winner.
- LTO is not enabled in the production build.

Global `-march=native` can still specialize graph construction, scheduling,
sampling, server, and expert-tier management, but it cannot newly unlock AVX2
inside the main ggml CPU kernels because those kernels already have it.

## Prepared non-production builds

Three isolated runtime copies were built from the same dirty source revision.
The production build directory was not overwritten.

| Runtime | Additional options | Status |
|---|---|---|
| `/tmp/compiler-lto-runtime-20260807/bin` | `GGML_LTO=ON` | built |
| `/tmp/compiler-native-lto-runtime-20260807/bin` | LTO plus global `-march=native -mtune=native` | built |
| `/tmp/compiler-native-lto-nointerpose-runtime-20260807/bin` | previous plus `-fno-semantic-interposition` | built |

The working build directory is `build-compiler-lto-sm75`, which is ignored by
Git and was returned to the controlled O3 configuration after the experiment.
Six sampling/expert/MoE component tests pass with both the most aggressive
safe-math candidate and the final controlled O3 build.

Static library sizes do not establish runtime speed, but they show where LTO
could plausibly matter:

| Library | O3 bytes | LTO bytes | LTO+native bytes | LTO+native+no-interpose bytes |
|---|---:|---:|---:|---:|
| `libggml-cpu.so` | 1,169,928 | 1,113,168 | 1,113,168 | 1,182,792 |
| `libggml-cuda.so` | 190,946,528 | 190,946,528 | 190,946,528 | 190,946,528 |
| `libllama.so` | 4,449,928 | 4,449,928 | 4,511,832 | 4,620,272 |
| `libllama-server-impl.so` | 7,553,480 | 4,450,792 | 4,538,832 | 4,576,936 |

LTO does not alter the compiled CUDA kernels and leaves `libllama` the same
size. Its opportunity is therefore CPU backend and host/server overhead, not
the dominant GPU matrix work. Global native flags and disabled semantic
interposition increase code size, so instruction-cache effects must be
measured rather than assumed beneficial.

## Throughput gate on the GTX 1660 Ti

The GPU was made exclusive and the compiler variants were screened with three
fresh-server 256-token runs each. The controlled baseline and candidates use
the same source revision, CMake feature configuration, power profile, and CUDA
library. The inference configuration was the established S=33, W=0, MTP-2,
376/376, q4_0/q4_0 target and draft KV, target backend sampling, eight CPU and
draft workers, shared-hot IDs, multi-token MoE fusion, Q8 rows=4, Q6 rows=2/4,
flat CONCAT, static no-sync, and profile SHA-256 `351cf93b...`.

| Host compiler configuration | Median decode | Difference from O3 | Hashes |
|---|---:|---:|---|
| controlled O3 | 45.771 tok/s | baseline | identical |
| O3 plus LTO | 45.749 tok/s | -0.05% | identical |
| O3 plus global native | 45.760 tok/s | -0.03% | identical |
| LTO plus global native | 45.687 tok/s | -0.18% | identical |
| previous plus `-fno-semantic-interposition` | 45.671 tok/s | -0.22% | identical |

Every run within and across the controlled variants has the same output hash,
token hash, MTP acceptance (0.672811), and mean accepted length (2.34). The
small differences are measurement noise or slight regressions; no variant
cleared the one-percent promotion gate, so no 2,000-token or q8_0/q4_0 follow-up
is justified. Production remains O3 without LTO or global native flags.

An earlier run using the pre-existing production binary produced a different
token stream and is excluded from this compiler comparison because it was not
built under the otherwise identical controlled CMake configuration. The first
attempt made while that server still occupied 5.67 GiB also failed allocation
before inference and remains explicitly not a result.

Artifacts are in:

- `benchmark-results/compiler-o3-controlled-screen-20260807`
- `benchmark-results/compiler-lto-screen-20260807`
- `benchmark-results/compiler-o3-native-screen-20260807`
- `benchmark-results/compiler-native-lto-screen-20260807`
- `benchmark-results/compiler-nointerpose-screen-20260807`

The System76 profile was restored to `balanced` after the tests.

## Compiler options worth considering

### Reasonable candidates

1. **LTO (`GGML_LTO=ON`)**: measured neutral/slightly negative on the Ryzen plus
   GTX 1660 Ti and therefore rejected for this production build. It can still
   be screened independently on the older i7 host; no result is transferable.
2. **Per-host `-march=native -mtune=native` for all host code**: measured neutral
   on the Ryzen because ggml-cpu was already native. Binaries would have to be
   rebuilt independently for the Ryzen and older GTX 1080/i7 host.
3. **`-fno-semantic-interposition`**: measured slightly negative and not adopted.
   It also changes symbol-interposition behavior for dynamically loaded backends
   and `LD_PRELOAD` diagnostics.
4. **Representative PGO**: now lower priority because neither LTO nor global
   native flags exposed meaningful host headroom. If revisited, train on a
   balanced prompt corpus, prompt processing, MTP-2 decode, request reset, and
   prefix reuse; never train only on the benchmark prompt. Use a separate
   profile directory and reject missing/mismatched profile warnings.
5. **A newer CUDA compiler/ptxas**: build the exact same sm_75 source with CUDA
   12.0 and a separately installed 12.4-or-newer toolkit. Compare cubins,
   registers, occupancy, hashes, and throughput. Compiler version alone is not
   a reason to replace the working toolkit.

### Low-priority candidates

- CUDA device LTO (`-dlto`) is unlikely to help much because the hot kernels
  are header-defined templates instantiated within their CUDA translation
  units. It requires a full CUDA rebuild and should follow a cubin/call-site
  audit.
- `--extra-device-vectorization` may help individual loops but can increase
  register pressure. It belongs in a per-kernel experiment, not a global
  production flag.
- Clang versus GCC can be screened for CPU cold kernels, but the current custom
  path is memory-heavy and already native AVX2. A whole-toolchain migration is
  not justified without a focused microbenchmark win.
- A different linker (`mold` or `lld`) improves build time, not token throughput.

### Rejected as global production flags

- Do not add host `-ffast-math`, `-Ofast`, or reassociation globally. ggml uses
  infinities in masks, and altered reductions can change routing, logits,
  greedy tokens, MTP acceptance, and reasoning quality. The HIP build files
  explicitly warn about finite-math-only for the same reason.
- Do not set a global CUDA register cap. The Q4_K, Q5_K, Q6_K, attention, and
  fused MoE kernels have different occupancy/register tradeoffs.
- Do not force MMQ on Turing; that direction was already measured slower.

## Remaining optimization priorities

The path to 50 tok/s is more likely to be a collection of exact CUDA-kernel
improvements than a single compiler switch. Moving from the 48.599 tok/s
q4_0/q4_0 reference to 50 requires approximately another 2.9 percent.

1. **Capture a decode-only Nsight trace after all current winners.** The latest
   available aggregate trace includes prompt work and two server threads, so
   its CUDA API totals cannot be read as per-token critical-path time. Add an
   NVTX decode range or time-window filter and separate target from draft.
2. **Retune the dominant one- and three-column Q4_K/Q5_K/Q6_K paths.** Q8 and
   Q6 row grouping already won. Next experiments should be kernel-specific
   warp/tile/unroll choices, with ptxas register/spill reporting, rather than
   global flags. Q5_K Down and Q4_K Gate/Up remain particularly relevant.
3. **Re-evaluate graph/API overhead only inside the decode range.** Aggregate
   traces contain many `cudaStreamSynchronize` and small transfers, but the
   static no-sync upper-bound test gained only about 0.6 percent. Further
   synchronization work needs critical-path evidence.
4. **Keep placement and profiles workload-specific.** A compiler cannot recover
   time spent on badly placed cold experts. Layer-variable profiles and a small
   profile bank remain portable to GPUs with more VRAM.
5. **Treat the quality and absolute-speed configurations separately.** The
   48.599 reference uses q4_0/q4_0 KV, while production deliberately uses
   q8_0/q4_0 after a measurable code-quality improvement. Any new winner must
   be confirmed with the quality prompt and reasoning-budget transition.
6. **Run all kernel and compiler screens independently on sm_61.** Pascal lacks
   the Turing fused path and has different register, occupancy, and launch
   optima. The GTX 1080 should not inherit sm_75 geometry or Ryzen-native host
   flags.
