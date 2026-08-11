# TurboQuant integration design

Date: 2026-08-07

## Phase 1 reference boundary

The reference codec lives below `tools/turboquant-ref` and is compiled into a
standalone tool and a C++ test. It is deliberately not part of `ggml-base` or
the llama runtime.

Properties:

- fixed 128-value storage blocks;
- zero padding for logical widths such as 64;
- independent blocks for widths such as 256;
- deterministic fixed-sign randomized WHT;
- forward and inverse transforms;
- corrected per-block FP16 norm;
- Turbo3 3-bit and Turbo4 4-bit centroid packing;
- reconstruction and rotated-domain decode;
- normalized attention dot-product error;
- explicit rejection of empty, truncated, null, and non-finite input;
- no input file overwrite by the row analysis tool.

The tool reads raw little-endian float32 rows:

```bash
./build-turbo-ref/bin/llama-turboquant-ref \
    --input /absolute/path/kv-keys.f32 \
    --row-size 256 \
    --format turbo3 \
    --output /tmp/turbo3-k-metrics.json
```

Rows are paired as query/key only for the dot-product metric: row 0 is the
query for key row 1, row 2 is the query for key row 3, and so on. All rows are
also measured independently for reconstruction error.

For model captures, use the manifest-aware wrapper rather than analyzing the
pair file directly:

```bash
python3 tools/turboquant-ref/analyze-capture.py \
    --capture-dir /tmp/turboquant-qk-capture-01 \
    --analyzer build-hybrid/bin/llama-turboquant-ref \
    --output /tmp/turboquant-qk-analysis.json
```

The wrapper validates the manifest, raw file lengths, head dimensions, and GQA
head mapping. It invokes causal-attention mode with the unique K file, so K
reconstruction is not polluted by query rows. For every captured Q head it
compares exact and quantized logits against all keys at earlier or equal token
positions and reports normalized dot error, scaled-logit error, softmax KL,
and maximum attention-probability deviation.

`llama-turboquant-capture` uses the existing evaluation callback only inside a
fresh measurement process. It recognizes post-RoPE tensors by name and
producer operation, copies at most the configured token cap, refuses an
existing output directory, and records no prompt text. A no-capture control
with the same model, prompt, seed, and generation settings is mandatory. Token
IDs and the generated-token hash must match before the captured rows are used.

## Phase 2 SM75 K-only runtime

Only after real K rows pass the offline quality gate. The first implementation
candidate is Turbo4 K with Q8 V; Turbo3 stays behind a later quality gate:

1. Allocate an unused `ggml_type` ID without changing existing IDs.
2. Add only the Turbo4-K type trait and raw state serialization.
3. Port CUDA `SET_ROWS` quantization matching the phase-1 golden codec.
4. Add the graph-side query WHT.
5. Add only the `turbo4 K x q8 V` FlashAttention vector kernel.
6. Validate SM75 first; keep SM61 an explicit, separately measured target.
7. Keep all defaults and existing Q4/Q8 paths unchanged.
8. Permit Q8 K fallback per layer; do not hard-code the ten-layer pattern seen
   in the current Qwen3.6 model.

InnerQ, Turbo2, TriAttention, Metal, HIP, and layer-adaptive policies remain
out of scope. Turbo4 V is now a separate experimental extension described
below; it does not change the K-only default or draft-cache support.

The implemented command-line gate accepts `turbo4_k` only for target `-ctk`.
It requires target Q8 V, FlashAttention, KQV offload, and a non-speculative
context. Target V and draft K/V parsers do not accept the type. Selecting
`-ctk turbo4_k` is itself the default-off experimental opt-in.

Two compile-time experiments are independently default-off:

- `GGML_CUDA_TURBO4_KQ_DP4A` selects an int8-centroid DP4A vector dot. SM75
  measurements rejected it; it is retained only as an experimental reference.
- `GGML_CUDA_TURBO4_F16_PREFILL` keeps one/two-column decode on the float
  centroid vector path and permits larger prompt batches to use existing FP16
  tile/MMA attention after dequantizing rotated Turbo4 K into transient
  scratch. `GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH` controls the crossover and
  defaults to 32 so short prompt fragments stay on the native Turbo4 path. It
  does not expand the stored KV cache.

The prefill conversion must never apply the inverse WHT: query input is already
transformed, so the temporary K values must remain in the rotated centroid
space. This is distinct from the generic CPU dequantizer, which reconstructs
keys in the original space.

## Phase 3 model quality gate

Use no MTP first. For identical prompts and seeds record:

- maximum and mean logit deviation;
- perplexity on a small fixed corpus;
- reasoning tokens and time to final answer;
- repeated verification and self-contradiction markers;
- output and token hashes within each configuration;
- Prompt TPS, Decode TPS, TTFT, VRAM, and RAM.

A cache type is rejected on the target model if it reproduces the known
Q4/Q4 reasoning loop, creates material perplexity regression, or loses more
than 3 percent throughput without a measured quality benefit.

The first one-chunk quality matrix is inconclusive: Turbo4/Q8 has better PPL
than Q4/Q8 and Q4/Q4, but worse KL divergence and top-token agreement. Before
MTP, expand the corpus and test a model-independent per-layer Q8 fallback. The
fallback policy must be externally configured or generated from measured
layer metrics; observed Qwen layer IDs must not be compiled into the runtime.
Mixed storage must preserve per-layer state type validation, report its actual
memory/type breakdown, and reject incompatible shared-cache layouts.

The first default-off implementation uses
`LLAMA_TURBO4_Q8_FALLBACK_LAYERS` with validated comma-separated IDs and
ranges. It changes only the selected K storage tensors to Q8. The actual K
tensor type continues to drive query transform and FlashAttention dispatch.
Cache sharing and layer reuse are rejected until dedicated compatibility tests
exist.

## Phase 4 MTP

First compress only the target K cache while the draft cache remains Q8/Q4.
Then test MTP-1, MTP-2, and MTP-3, including rollback, state save/load, verify
batches, draft acceptance, and first-logit divergence. Turbo draft KV remains
behind an experimental guard until these tests pass.

The initial target-only gate is `LLAMA_TURBO4_MTP_EXPERIMENTAL=1`. It permits
only `draft-mtp`; it does not permit other speculative methods or change draft
cache types. The process-wide per-layer fallback policy is applied only when a
context actually uses Turbo4 K, so the Q8/Q4 MTP context ignores it. Direct API
creation of an MTP context with Turbo4 K remains rejected by the context-level
guard.

When the experimental FP16 prefill build is used,
`GGML_CUDA_TURBO4_F16_PREFILL_MIN_BATCH=2` is the current MTP test setting.
It keeps one-column decode on native Turbo4 while routing multi-column verify
attention through the same FP16 tile/MMA implementation independent of MTP-2
or MTP-3 width. The runtime override is deliberately not a production default:
MTP-1 can still exercise a different effective geometry, and long-run speed is
sensitive to cache-induced token and acceptance changes.

## Phase 5 SM61

GTX 1080 support is a separate experiment. Build native SM61 without forced
MMQ, benchmark its VEC attention path, and retain Q8/Q4 fallback. SM75 results
must not be extrapolated to Pascal.

## Experimental target V extension

`LLAMA_TURBO4_V_EXPERIMENTAL=1` is required in addition to selecting
`-ctv turbo4_k`. The accepted target combinations are Turbo4/Q8, Q8/Turbo4,
and Turbo4/Turbo4. FlashAttention and KQV offload are mandatory. Turbo4 draft
K/V remains rejected, and speculative use also requires
`LLAMA_TURBO4_MTP_EXPERIMENTAL=1` with draft-mtp only.

The stored V rows use the same randomized-WHT and centroid format as K. Since
the attention-weighted sum is linear, FlashAttention accumulates in rotated
space and the graph applies one inverse WHT to the final attention output.
The inverse operation must occur exactly once; individual V rows are never
inverse-transformed on the hot attention path.

The native vector kernel has a dedicated Turbo4 V schedule. A generic
quantized-V dequantizer is functionally insufficient for performance. Larger
prefill and MTP verify batches may use the existing transient FP16 conversion
when `GGML_CUDA_TURBO4_F16_PREFILL` is built. The runtime crossover of 2 is a
measured MTP setting, not a universal default.

Turbo4/Turbo4 reduces cache memory sufficiently to retain additional fixed
experts at long reserved contexts, but it also changes logits and MTP token
acceptance. It must be judged on sustained end-to-end timing and quality, not
raw attention throughput. The feature remains absent from stable start
scripts until active long-context quality and SM61 tests pass.

## Later TriAttention work

TriAttention begins as trace and oracle simulation only. The calibration must
fingerprint the model and include only target full-attention layers. Production
eviction additionally requires request-local state, persistent CUDA scratch,
event-based publication, prefix-cache semantics, and explicit MTP guards.

The offline gate can optionally capture a dedicated materialized V tensor and
measure the renormalized attention-output error. This is deliberately separate
from the aliasing runtime `Vcur` view and default-off. It still does not model
cache quantization, downstream residual amplification, logits, or MTP rollback.

The next gate is a replay-only additive attention mask, not KV deletion. A
baseline capture exports immutable keep sets for a bounded prompt and pruning
event. A separate no-MTP, single-sequence replay may apply those sets to
layer-specific attention masks while retaining all K/V storage. The ordinary
causal mask remains authoritative; replay can only add `-inf` entries. The tool
must dump raw logits for identical evaluation positions and compare maximum
and mean logit error, KL divergence, top-token identity, and perplexity. It
must support a no-prune mask whose logits are bit-identical to the baseline.

The first implemented gate uses an even narrower attention-output delta
replay. Offline F32 Q/K/V reconstruct the full and pruned pre-gate attention
outputs for one layer. Their difference is added to the actual Q8/Q8 runtime
attention output, so the target graph then measures downstream gate, output
projection, residual, FFN, and final-logit amplification. This is valid for a
single-layer sensitivity test and leaves the KV cache untouched. It is not a
replacement for a future live mask when multiple earlier layers interact.

The second diagnostic gate now applies the exported fixed keep set to a
layer-local duplicate of the real causal mask. It adds `-inf` only for evicted
old positions and leaves the shared mask and Q8/Q8 KV cache untouched. A graph
callback performs the bounded mutation after the duplicate is computed and
before FlashAttention consumes it. The manifest permits one layer and binds
the exact prompt, graph, token window, context, batch geometry, and cache
types. This synchronous callback is intentionally unsuitable for production;
it exists to compare actual cache-quantized attention against the F32 offline
proxy.

For autoregressive sensitivity only, the capture executable can keep applying
the same fixed old-prefix keep set to later one-token graphs. It counts every
continuation application and aborts if the number differs from successful
decode calls. This deliberately excludes MTP, cache shifts, multiple
sequences, prefix reuse, and physical cache reclamation.

Replay configuration must stay outside the public server path and carry a
model fingerprint, prompt-token hash, full-attention layer IDs, absolute token
positions, budget, and event generation. It is invalid with MTP, multiple
sequences, cache shifting, rollback, prefix reuse, or a mismatched prompt. No
physical cache compaction is permitted by this diagnostic. A runtime eviction
prototype is considered only if multi-prompt replay passes the quality gate
and predicts a material KV or attention-time benefit after its metadata and
scoring costs.
