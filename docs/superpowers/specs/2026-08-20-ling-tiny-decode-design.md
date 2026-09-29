# Ling-tiny decode +10% design

Date: 2026-08-20

Raise Ling-3.0-tiny decode throughput on the GTX 1660 Ti from the current
measured winner (~108 tok/s) to about 119 tok/s (+10%). Parameter knobs in
`start-ling-tiny.sh` stay frozen. Work is code and algorithm only.

Small quality drift is allowed after the prune stage. Bench-prompt quality and
reasoning must remain usable. CUDA hangs, Xid, OOM, NaN/Inf, or a broken
reasoning sample reject the candidate even if tok/s rises.

## Baseline

- GPU: GTX 1660 Ti (sm_75, 6 GiB)
- Model: `Ling-3.0-tiny-Q4_K_M.gguf` (24 layers, 1536 d_model, 128 routed
  experts, 8 used, 1 shared, vocab 157184)
- Layer mix: 18 KDA + 6 MLA (`layer_group_size=4`), layer 0 dense, layers 1-23 MoE
- Router: sigmoid + expert bias, 8 groups, top-2 sum per group, 4 groups kept,
  top-8 experts, L1-norm, scale 2.5
- Launcher: `start-ling-tiny.sh` (KVFlash 8192, phase 2048/64, Q8 KV, ngram-simple
  n_max 4, MoE multi/combine fusion on, Q8 MMVQ rows=4)
- Known leftover: grouped router cannot use the flat `topk_moe` fusion; KDA
  `causal_conv1d` inserts `RESHAPE` between `SSM_CONV` and `SILU`, so the
  existing CUDA silu fusion does not fire
- Recent exact code wins: +2.1% (Q8 rows + ordered warp `TOP_K`)

Primary metric: median decode tok/s over three fresh-server 512-token runs,
then one 2000-token confirmation if the median is at least +8%. Compare against
a new baseline captured on the same host and binary family, not against the
2026-08-19 numbers alone.

## Phases

### Phase 0: measure

Capture a decode-only Nsight Systems trace of the current launcher (NVTX or
time-window on the decode range). Record share of:

- grouped-router small kernels (`TOP_K`, `GET_ROWS`, `SET_ROWS`, `FILL`, `SUM_ROWS`)
- KDA conv chain (`MUL_MAT`, `CONCAT`, `CPY`, `SSM_CONV`, `SILU`)
- routed + shared MoE GEMMs
- Q8_0 output projection

Establish the numeric baseline (3 x 512) and keep the token SHA-256. Phase 0
does not change code behavior.

### Phase 1: grouped router fusion (exact)

Keep the existing `build_moe_ffn` graph. Extend `ggml/src/ggml-cuda/topk-moe.cu`
with a grouped path and teach `ggml_cuda_topk_moe_fusion` in
`ggml/src/ggml-cuda/ggml-cuda.cu` to match the BailingMoE3 chain:

1. sigmoid (optional) + expert bias
2. reshape to `[n_exp_per_group=16, n_groups=8, n_tokens]`
3. top-2 per group, gather, sum -> group scores `[8, n_tokens]`
4. top-4 groups
5. mask other groups to `-inf` via get/set-rows
6. reshape to `[128, n_tokens]`
7. top-8
8. gather weights from the unbiased sigmoid probs
9. L1-norm + clamp + scale 2.5

Do not add a new ggml op. If the matcher misses, the unfused graph runs
unchanged. Token hash must match the Phase 0 baseline.

Specialized first for 128/8/4/8 (Ling-tiny). Other grouped shapes are out of
scope unless they match for free.

### Phase 2: KDA conv collapse (exact)

In `src/models/bailingmoe3.cpp` (`bailingmoe3_causal_conv1d`):

- apply `SILU` directly to `SSM_CONV` output, then reshape
- elementwise SiLU commutes with the view/reshape, so the existing
  `{SSM_CONV, UNARY(SILU)}` fusion in `ggml-cuda.cu` can fire
- 18 KDA layers x 3 convs = 54 fewer standalone SiLU launches

If a decode-only trace after the SiLU fusion still shows `CONCAT` plus `CPY`
in the KDA conv path at 2% or more of decode GPU-kernel time, add a
decode-only fused kernel in `ggml/src/ggml-cuda/ssm-conv.cu` for `n_t=1`,
`d_conv=4`, `d_inner=2048`: roll the conv state, conv, SiLU, write the new
state. Prefill keeps the existing path. Token hash must still match Phase 0.

### Phase 3: weight prune (only if Phase 1+2 are under +10%)

Default off. Enable with `GGML_CUDA_MOE_WEIGHT_EPS` (float). Weights at this
point are L1-normalized then multiplied by 2.5, so they sum to 2.5. Start the
screen at `0.02` (about 0.8% of that mass). Apply prune in the same grouped
router kernel when fusion is active, otherwise as a tiny follow-up on the
8-wide id/weight tensors:

- if `weight[i] < EPS`, write `id[i] = -1` and `weight[i] = 0`
- keep at least the largest-weight expert
- graph shape stays 8 slots so CUDA graphs stay valid

`skip_slot` on `MUL_MAT_ID` stays the existing zero-sentinel mechanism. Do not
reuse it. Extend mmvq (and the fused MoE gate/up path that already reads ids)
to skip negative expert ids: no weight load, no accumulate. Combine fusion
already multiplies by weight, so a zero weight contributes 0.

Token hash may change. Quality gate applies.

## Data flow (decode, n_tokens=1)

```text
layer input
  -> KDA or MLA
  -> MoE router logits
  -> [fused grouped router | unfused chain]
  -> 8 ids + 8 weights
  -> optional prune (id=-1, weight=0)
  -> gate/up/down mul_mat_id (skip id<0)
  -> shared expert FFN
  -> weighted combine (existing GGML_CUDA_MOE_COMBINE_FUSION)
  -> residual
```

No host readback of ids. No graph rebuild. `LLAMA_EXPERT_S` stays unset.

## Error handling

- Fusion mismatch: run the unfused nodes. Log once behind an existing or new
  opt-in trace env, not on every token.
- Prune env unset or `<= 0`: no prune.
- All 8 weights below EPS: keep argmax, zero the rest.
- CUDA error, Xid, hang, OOM: candidate is dead. Stop the screen.
- NaN/Inf in logits or empty completion: dead.
- Reasoning sample unusable or bench-prompt quality clearly worse: dead, even
  at +10% tok/s.

## Tests

Backend (must stay green on sm_75):

- `test-backend-ops` grouped-router cases: 128 experts, 8 groups, top-2 group
  score, 4 groups, top-8, bias, norm, scale, including ties
- existing `SSM_CONV+SILU` fusion cases, plus the Ling reshape-after-silu order
- mmvq `MUL_MAT_ID` with one or more `id=-1` (output matches zeroed expert)

Throughput A/B against `start-ling-tiny.sh`:

- 3 x 512 decode tokens, temperature 0, `ignore_eos`, same prompt as the
  2026-08-19 Ling screens
- Phase 1 and 2: token SHA-256 must equal the new baseline
- Phase 3: hash may differ; run the quality prompt with `--reasoning on`.
  Reject on empty output, repeat-loop, or a reasoning trace that fails the
  same smoke checks used for the 2026-08-19 Ling screens
- if median >= +8% vs baseline, one 2000-token run
- promote only if median >= +10% and the quality gate holds

Do not retune batch, KV type, threads, KVFlash, or ngram. Those are frozen.

## Out of scope

- GTX 1080 / sm_61 unless a change is backend-generic and costs nothing extra
- MTP (this GGUF has no NextN weights)
- new ggml ops, new subsystems, expert-tier / CPU MoE
- launcher parameter search
- compiler flags / LTO
- replacing ngram with another speculative decoder

## Promotion rule

Ship Phase 1+2 if they are exact and help. Ship Phase 3 only when it is required
to reach +10% and the quality gate holds. Record median tok/s, hashes, prune
EPS, Nsight shares, and the quality verdict under
`benchmark-results/ling-tiny-hybrid/`.
