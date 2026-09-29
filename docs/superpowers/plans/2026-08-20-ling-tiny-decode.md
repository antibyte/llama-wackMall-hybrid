# Ling-tiny decode +10% Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Ling-3.0-tiny decode on the GTX 1660 Ti from a fresh `start-ling-tiny.sh` baseline (~108 tok/s) to about 119 tok/s (+10%) without changing launcher knobs.

**Architecture:** Keep the existing `build_moe_ffn` and KDA graphs. Fuse the BailingMoE3 grouped router (sigmoid, 8 groups, top-2 sum, top-4 groups, top-8, norm, scale 2.5) into one CUDA kernel. Reorder KDA `SILU` onto `SSM_CONV` so the existing fusion fires. If Phase 1+2 miss +10%, prune experts with weight below `GGML_CUDA_MOE_WEIGHT_EPS` by writing `id=-1` and skipping those slots in mmvq. CUDA graph shape stays 8 experts.

**Tech Stack:** C++ / CUDA in ggml and llama.cpp, `test-backend-ops`, `start-ling-tiny.sh`, `tools/bench_hybrid_client.py`, Nsight Systems via `LLAMA_NSYS_TRACE=1`.

## Global Constraints

- Target: GTX 1660 Ti, sm_75, binary `build-main-sm75/bin/llama-server`
- Model: `$HOME/models/ling-3.0-tiny/Ling-3.0-tiny-Q4_K_M.gguf` (128 experts, 8 used, 8 groups, 4 groups kept, scale 2.5)
- Launcher knobs stay frozen: `start-ling-tiny.sh` as checked in
- Phase 1 and 2 token SHA-256 must match the Phase 0 baseline
- Phase 3 may change the hash; quality prompt + reasoning must stay usable
- No new ggml op, no new subsystem, no MTP, no 1080-only work
- ASCII only in code and comments (`-`, `->`, `x`, `...`)
- Do not commit unless the human asks; suggested `git add` lines are for the human
- `LLAMA_EXPERT_S` stays unset (CUDA graphs stay available)

## File map

- Modify: `tests/test-backend-ops.cpp` - grouped-router graph test, reshape-after-silu SSM test, negative-id MUL_MAT_ID test
- Modify: `ggml/src/ggml-cuda/topk-moe.cuh` - grouped args + entry points
- Modify: `ggml/src/ggml-cuda/topk-moe.cu` - grouped kernel + prune tail
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu` - grouped fusion matcher in `ggml_cuda_try_fuse`
- Modify: `src/models/bailingmoe3.cpp` - `SILU` before reshape in `bailingmoe3_causal_conv1d`
- Modify: `ggml/src/ggml-cuda/common.cuh` - `ggml_cuda_mul_mat_id_is_skipped`
- Modify: `ggml/src/ggml-cuda/mmvq.cu` - skip `id < 0` in both decode kernels
- Modify: `ggml/src/ggml-cuda/mmvf.cu` - same skip if that path reads ids
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c` - skip `id < 0` when grouping MUL_MAT_ID rows (CPU reference for tests; zero that dst slot)
- Optional: `ggml/src/ggml-cuda/ssm-conv.cu` - decode-only state+conv+silu only if CONCAT+CPY >= 2% of decode kernel time
- Create: `benchmark-results/ling-tiny-hybrid/20260820-decode10/NOTES.md` plus prompt/result JSON from the screens
- Read: `docs/superpowers/specs/2026-08-20-ling-tiny-decode-design.md`, `src/llama-graph.cpp` lines 2090-2192, `start-ling-tiny.sh`

---

### Task 1: Phase 0 baseline and decode-only Nsight

**Files:**
- Create: `benchmark-results/ling-tiny-hybrid/20260820-decode10/prompt-decode.txt`
- Create: `benchmark-results/ling-tiny-hybrid/20260820-decode10/NOTES.md`
- Test: no code change; numbers become the promotion baseline

**Interfaces:**
- Consumes: `start-ling-tiny.sh`, `tools/bench_hybrid_client.py`, NVTX range `wackmall.target.decode` (already emitted when `LLAMA_NSYS_TRACE` is set)
- Produces: `BASELINE_MEDIAN_TPS`, `BASELINE_TOKEN_SHA256`, kernel-share table in `NOTES.md`

- [ ] **Step 1: Write the frozen decode prompt**

Create `benchmark-results/ling-tiny-hybrid/20260820-decode10/prompt-decode.txt` with this exact text (no trailing blank line after the last sentence):

```text
Write a short technical briefing about hybrid linear attention. Cover KDA versus MLA, why a 3:1 stack helps decode, and one risk when the KV cache is quantized to Q8_0. Use plain sentences.
```

- [ ] **Step 2: Capture three 512-token baseline runs**

From the repo root, with the 1660 Ti free:

```bash
mkdir -p benchmark-results/ling-tiny-hybrid/20260820-decode10
./start-ling-tiny.sh --port 18080 > /tmp/ling-tiny-base.server.log 2>&1 &
sleep 20
for i in 1 2 3; do
  python3 tools/bench_hybrid_client.py \
    --url http://127.0.0.1:18080 \
    --prompt-file benchmark-results/ling-tiny-hybrid/20260820-decode10/prompt-decode.txt \
    --n-predict 512 \
    --output benchmark-results/ling-tiny-hybrid/20260820-decode10/base-run${i}.json
done
```

Expected: three JSON files with `timings.predicted_n == 512`. Record median `predicted_per_second` and the token hash field (`token_sha256` if present, else hash `token_ids`). All three hashes must match. Kill the server after the three runs.

- [ ] **Step 3: Capture a decode-only Nsight trace**

```bash
LLAMA_NSYS_TRACE=1 nsys profile --stats=true \
  -o benchmark-results/ling-tiny-hybrid/20260820-decode10/base-decode \
  -t cuda,nvtx \
  ./start-ling-tiny.sh --port 18081
```

In another shell, one 128-token request with the same prompt (`--n-predict 128`), then stop the server. Filter stats to NVTX `wackmall.target.decode`.

In `NOTES.md` write four shares of decode GPU-kernel time:

1. grouped router small ops (`TOP_K`, `GET_ROWS`, `SET_ROWS`, `FILL`, `SUM_ROWS`)
2. KDA conv chain (`MUL_MAT` Q/K/V proj, `CONCAT`, `CPY`, `SSM_CONV`, `SILU`)
3. routed plus shared MoE GEMMs
4. Q8_0 output projection

- [ ] **Step 4: Lock the baseline numbers**

Append to `NOTES.md`:

```text
baseline_median_tps: <median>
baseline_token_sha256: <hash>
```

Do not start Task 2 until those two lines exist.

---

### Task 2: Grouped-router whole-graph test (unfused first)

**Files:**
- Modify: `tests/test-backend-ops.cpp` (add `test_topk_moe_grouped` next to `test_topk_moe` around line 6017, register cases next to the existing TOPK_MOE loop around line 9653)

**Interfaces:**
- Consumes: same graph ops as `src/llama-graph.cpp` lines 2115-2192 for `LLM_ARCH_BAILINGMOE3`
- Produces: `test_topk_moe_grouped` with `run_whole_graph() == true` and `fusion_test_nodes()` returning `{selected_experts, weights}`

- [ ] **Step 1: Add the test type**

Insert this struct after `test_topk_moe`:

```cpp
struct test_topk_moe_grouped : public test_case {
    const int n_expert;
    const int n_tokens;
    const int n_groups;
    const int n_group_used;
    const int group_top;
    const int n_expert_used;
    const bool with_norm;
    const bool bias_probs;
    const float scale_w;
    ggml_tensor * weights {};
    ggml_tensor * selected_experts {};

    test_topk_moe_grouped(int n_expert = 128, int n_tokens = 1, int n_groups = 8,
                          int n_group_used = 4, int group_top = 2, int n_expert_used = 8,
                          bool with_norm = true, bool bias_probs = true, float scale_w = 2.5f)
        : n_expert(n_expert), n_tokens(n_tokens), n_groups(n_groups),
          n_group_used(n_group_used), group_top(group_top), n_expert_used(n_expert_used),
          with_norm(with_norm), bias_probs(bias_probs), scale_w(scale_w) {
        GGML_ASSERT(n_expert % n_groups == 0);
        GGML_ASSERT(n_group_used < n_groups);
        GGML_ASSERT(n_expert_used <= n_expert);
    }

    std::string vars() override {
        return VARS_TO_STR9(n_expert, n_tokens, n_groups, n_group_used, group_top,
                            n_expert_used, with_norm, bias_probs, scale_w);
    }
    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "TOPK_MOE_GROUPED";
    }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_exp_per_group = n_expert / n_groups;
        ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
        ggml_tensor * probs = ggml_sigmoid(ctx, logits);
        ggml_set_name(probs, "probs");

        ggml_tensor * selection_probs = probs;
        if (bias_probs) {
            ggml_tensor * exp_probs_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_expert);
            ggml_set_name(exp_probs_b, "exp_probs_b");
            selection_probs = ggml_add(ctx, probs, exp_probs_b);
        }

        ggml_tensor * selection_groups = ggml_reshape_3d(ctx, selection_probs,
                n_exp_per_group, n_groups, n_tokens);
        ggml_tensor * group_ids = ggml_top_k(ctx, selection_groups, group_top);
        ggml_tensor * group_vals = ggml_get_rows(ctx,
                ggml_reshape_4d(ctx, selection_groups, 1, selection_groups->ne[0],
                    selection_groups->ne[1], selection_groups->ne[2]),
                group_ids);
        ggml_tensor * group_scores = ggml_sum_rows(ctx,
                ggml_reshape_3d(ctx, group_vals, group_vals->ne[1], group_vals->ne[2], group_vals->ne[3]));
        group_scores = ggml_reshape_2d(ctx, group_scores, group_scores->ne[1], group_scores->ne[2]);

        ggml_tensor * expert_groups = ggml_top_k(ctx, group_scores, n_group_used);
        selection_probs = ggml_get_rows(ctx, selection_groups, expert_groups);
        selection_probs = ggml_set_rows(ctx, ggml_fill(ctx, selection_groups, -INFINITY),
                selection_probs, expert_groups);
        selection_probs = ggml_reshape_2d(ctx, selection_probs, n_expert, n_tokens);

        selected_experts = ggml_top_k(ctx, selection_probs, n_expert_used);
        ggml_set_name(selected_experts, "selected_experts");

        weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts);
        if (with_norm) {
            weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            ggml_tensor * weights_sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5, INFINITY);
            weights = ggml_div(ctx, weights, weights_sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        }
        if (scale_w != 0.0f && scale_w != 1.0f) {
            weights = ggml_scale(ctx, weights, scale_w);
        }
        ggml_set_name(weights, "weights");
        return weights;
    }

    std::vector<ggml_tensor *> fusion_test_nodes() override { return { selected_experts, weights }; }

    double err(const float * a, const float * b, size_t n) override {
        std::vector<float> a2(a, a + n);
        std::vector<float> b2(b, b + n);
        std::sort(a2.begin(), a2.end());
        std::sort(b2.begin(), b2.end());
        return nmse(a2.data(), b2.data(), n);
    }
};
```

Register at least these cases next to the existing TOPK_MOE loop:

```cpp
test_cases.emplace_back(new test_topk_moe_grouped(128, 1, 8, 4, 2, 8, true, true, 2.5f));
test_cases.emplace_back(new test_topk_moe_grouped(128, 4, 8, 4, 2, 8, true, true, 2.5f));
test_cases.emplace_back(new test_topk_moe_grouped(128, 2, 8, 4, 2, 8, true, false, 2.5f));
```

- [ ] **Step 2: Build and run the new cases (unfused CUDA vs CPU)**

```bash
cmake --build build-main-sm75 -j8 --target test-backend-ops
./build-main-sm75/bin/test-backend-ops -o TOPK_MOE_GROUPED
```

Expected: PASS on CUDA vs CPU. If this fails, the test graph does not match ggml semantics; fix the test before any kernel work.

- [ ] **Step 3: Human commit (optional)**

```bash
git add tests/test-backend-ops.cpp
# human commit: cuda : add grouped topk moe backend test
```

---

### Task 3: Grouped-router kernel and fusion matcher

**Files:**
- Modify: `ggml/src/ggml-cuda/topk-moe.cuh`
- Modify: `ggml/src/ggml-cuda/topk-moe.cu`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_try_fuse`, after the existing topk-moe block around line 4617)

**Interfaces:**
- Consumes: graph built by `test_topk_moe_grouped` / `build_moe_ffn` for BAILINGMOE3
- Produces:

```cpp
struct ggml_cuda_topk_moe_grouped_args {
    bool sigmoid = true;
    bool prob_bias = true;
    bool norm = true;
    bool scale = true;
    int n_groups = 8;
    int n_exp_per_group = 16;
    int n_group_used = 4;
    int group_top = 2;
    float weight_eps = 0.0f;
};

float ggml_cuda_moe_weight_eps();

bool ggml_cuda_topk_moe_grouped_fusion(const struct ggml_cgraph * cgraph, int node_idx,
                                       ggml_cuda_topk_moe_grouped_args & args);

void ggml_cuda_op_topk_moe_grouped(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * logits,
                                   ggml_tensor * weights,
                                   ggml_tensor * ids,
                                   const ggml_tensor * clamp,
                                   const ggml_tensor * scale,
                                   const ggml_tensor * bias,
                                   const ggml_cuda_topk_moe_grouped_args & args);
```

- [ ] **Step 1: Add the header API**

Append to `ggml/src/ggml-cuda/topk-moe.cuh` the struct and three declarations above. Implement `ggml_cuda_moe_weight_eps()` in `topk-moe.cu`:

```cpp
float ggml_cuda_moe_weight_eps() {
    static const float eps = []() {
        const char * v = getenv("GGML_CUDA_MOE_WEIGHT_EPS");
        if (!v || v[0] == '\0') {
            return 0.0f;
        }
        return std::strtof(v, nullptr);
    }();
    return eps > 0.0f ? eps : 0.0f;
}
```

Leave prune unused until Task 7 (`weight_eps` stays 0).

- [ ] **Step 2: Write the matcher against the live test graph**

`ggml_cuda_topk_moe_grouped_fusion` must start at `GGML_OP_UNARY` with `GGML_UNARY_OP_SIGMOID` and walk the BAILINGMOE3 / test graph:

```text
SIGMOID
optional ADD (bias; src[0] == sigmoid)
RESHAPE to [16, 8, T]
TOP_K k=2
RESHAPE + GET_ROWS + RESHAPE + SUM_ROWS + RESHAPE -> [8, T]
TOP_K k=4
GET_ROWS
FILL(-inf) + SET_ROWS
RESHAPE to [128, T]
TOP_K k=8          -> ids output node
RESHAPE of unbiased sigmoid to [1,128,T] + GET_ROWS
optional RESHAPE + SUM_ROWS + CLAMP + DIV + RESHAPE
optional SCALE
```

If any step mismatches, return false (unfused path). On success fill `args` and remember the ids node (`TOP_K k=8`) and the final weights node.

Do not require `ARGSORT`. Ling uses `GGML_OP_TOP_K`.

If the walk is off by a VIEW/RESHAPE, dump `cgraph->nodes[i]->op` for the test graph once (compile-time or a one-shot `fprintf` behind `GGML_CUDA_TOPK_MOE_TRACE=1`) and adjust the matcher. Do not invent a new ggml op.

- [ ] **Step 3: Implement the specialized kernel**

In `topk-moe.cu`, add a template specialized for Ling-tiny:

```cpp
template <int n_experts, int n_groups, int n_group_used, int n_expert_used, int group_top, bool has_bias>
__launch_bounds__(WARP_SIZE, 1)
__global__ void topk_moe_grouped_cuda(const float * logits, float * weights, int32_t * ids,
                                      const float * bias, int n_rows, float clamp_val,
                                      float scale_val, float weight_eps);
```

Launch: one warp per token (`blockDim.x = WARP_SIZE`, `grid.x = n_rows`).

Per row:

1. Each lane holds `n_experts / WARP_SIZE` logits (4 values when `n_experts=128`).
2. Sigmoid in registers. Keep a copy as `prob` (unbiased). If `has_bias`, `sel = prob + bias[e]`, else `sel = prob`.
3. For each group `g` in `0..7`, compute the top-2 of the 16 `sel` values in that group (lane-local plus warp shuffles). `group_score[g] = top1 + top2`.
4. Top-4 group indices among the 8 scores. Ties: smaller index wins (same as existing `topk_moe_cuda`).
5. Set `sel[e] = -inf` when `e` is not in a kept group.
6. Iterative top-8 on `sel`, same unique-argmax / `-inf` mark as `topk_moe_cuda`. Write compact `ids[row * n_expert_used + k]`.
7. `w[k] = prob[ids[k]]`. If `norm`, divide by `max(sum(w), clamp_val)`. Multiply by `scale_val`.
8. If `weight_eps > 0`, apply Task 7 logic (no-op while eps is 0).

`ggml_cuda_op_topk_moe_grouped` reads:

```cpp
const int n_experts = (int) logits->ne[0];
const int n_rows    = (int) logits->ne[1];
GGML_ASSERT(n_experts == 128);
GGML_ASSERT(weights->ne[1] == 8);
GGML_ASSERT(ids->ne[0] == 8);
```

Abort (fallback already happened if matcher refused) if the shape is not 128/8/4/8.

- [ ] **Step 4: Hook the matcher in `ggml_cuda_try_fuse`**

Before the existing flat topk-moe block, try grouped fusion:

```cpp
if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID) {
    ggml_cuda_topk_moe_grouped_args gargs;
    if (ggml_cuda_topk_moe_grouped_fusion(cgraph, i, gargs)) {
        // resolve logits, bias, ids, weights, clamp, scale from the matched nodes
        // ggml_can_fuse_subgraph + ggml_cuda_check_fusion_memory_ranges
        // gargs.weight_eps = ggml_cuda_moe_weight_eps();
        ggml_cuda_op_topk_moe_grouped(*cuda_ctx, logits, weights, ids, clamp, scale, bias, gargs);
        return n_fused - 1;
    }
}
```

On mismatch, fall through to the existing flat matcher. Log once if `GGML_CUDA_TOPK_MOE_TRACE=1`.

- [ ] **Step 5: Rebuild and run the grouped tests**

```bash
cmake --build build-main-sm75 -j8 --target test-backend-ops
GGML_CUDA_TOPK_MOE_TRACE=1 ./build-main-sm75/bin/test-backend-ops -o TOPK_MOE_GROUPED
```

Expected: PASS, and the trace prints that grouped fusion was used. Also run the old suite so flat fusion did not break:

```bash
./build-main-sm75/bin/test-backend-ops -o TOPK_MOE
```

Expected: PASS (existing 445-class TOP_K / TOPK_MOE cases stay green).

- [ ] **Step 6: Human commit (optional)**

```bash
git add ggml/src/ggml-cuda/topk-moe.cuh ggml/src/ggml-cuda/topk-moe.cu ggml/src/ggml-cuda/ggml-cuda.cu
# human commit: cuda : fuse bailingmoe3 grouped moe router
```

---

### Task 4: Phase 1 decode A/B (exact)

**Files:**
- Create: `benchmark-results/ling-tiny-hybrid/20260820-decode10/phase1-run{1,2,3}.json`
- Modify: `benchmark-results/ling-tiny-hybrid/20260820-decode10/NOTES.md`

**Interfaces:**
- Consumes: Task 1 baseline hash and median; Task 3 binary
- Produces: Phase 1 median and hash verdict

- [ ] **Step 1: Rebuild the server**

```bash
cmake --build build-main-sm75 -j8 --target llama-server
```

- [ ] **Step 2: Repeat the three 512-token runs from Task 1 Step 2** against the new server. Same port pattern, same prompt, `n-predict 512`.

Expected: `token_sha256` equals `baseline_token_sha256`. If it differs, stop and fix the kernel (ties or group-score math). Do not continue to Task 5.

- [ ] **Step 3: Record median tok/s and percent vs baseline in `NOTES.md`.**

If median is already >= +10% and hashes match, skip Task 7-8 later. Still do Task 5 (KDA silu is exact and cheap).

---

### Task 5: KDA SILU-before-reshape (exact)

**Files:**
- Modify: `src/models/bailingmoe3.cpp` (`bailingmoe3_causal_conv1d`, lines 202-205)
- Modify: `tests/test-backend-ops.cpp` (add one reshape-after-silu case beside `test_ssm_conv_bias_silu`)

**Interfaces:**
- Consumes: existing `{SSM_CONV, UNARY(SILU)}` fusion in `ggml-cuda.cu` lines 4381-4394 and 5311-5313
- Produces: Ling KDA graph that fusion can match; token hash still equal to Phase 0

- [ ] **Step 1: Add a reshape-after-silu test**

Next to `test_ssm_conv_bias_silu::build_graph`, add `test_ssm_conv_silu_reshape` that does:

```cpp
ggml_tensor * out = ggml_ssm_conv(ctx, a, b);
out = ggml_silu(ctx, out);
out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1] * out->ne[2]);
```

Register Ling decode shape: `a = {4, 2048, 1, 1}`, `b = {4, 2048, 1, 1}` (n_t=1, d_inner=2048, d_conv=4).

```bash
cmake --build build-main-sm75 -j8 --target test-backend-ops
./build-main-sm75/bin/test-backend-ops -o SSM_CONV_BIAS_SILU
./build-main-sm75/bin/test-backend-ops -o SSM_CONV_SILU_RESHAPE
```

Expected: existing SILU fusion cases PASS. New case PASSes against CPU (fusion optional).

- [ ] **Step 2: Reorder the production graph**

In `bailingmoe3_causal_conv1d` replace:

```cpp
ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
out = ggml_silu(ctx0, ggml_reshape_2d(ctx0, out, d_inner, n_tokens));
return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_seq_tokens, n_seqs);
```

with:

```cpp
ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
out = ggml_silu(ctx0, out);
out = ggml_reshape_2d(ctx0, out, d_inner, n_tokens);
return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_seq_tokens, n_seqs);
```

SiLU is elementwise, so this is bit-exact vs reshape-then-silu for finite values.

- [ ] **Step 3: Rebuild server and repeat three 512-token runs**

Expected: hash == Phase 0. Record Phase 2a median in `NOTES.md`.

- [ ] **Step 4: Decide on fused state+conv**

If you still have a decode-only Nsight after Step 3, sum `CONCAT` + `CPY` in the KDA conv path. If that share is < 2% of decode GPU-kernel time, skip Task 6 entirely. If >= 2%, do Task 6.

- [ ] **Step 5: Human commit (optional)**

```bash
git add src/models/bailingmoe3.cpp tests/test-backend-ops.cpp
# human commit: bailingmoe3 : fuse kda ssm_conv silu
```

---

### Task 6: Optional decode-only KDA conv+state kernel

**Files:**
- Modify: `ggml/src/ggml-cuda/ssm-conv.cu` and `ssm-conv.cuh` only if Task 5 Step 4 said >= 2%

**Interfaces:**
- Consumes: decode `n_t=1`, `d_conv=4`, `d_inner=2048`, existing conv state view + `SSM_CONV` + `SILU`
- Produces: one kernel that rolls `d_conv-1` state, convolves, applies SiLU, writes new state
- Prefill (`n_t != 1`) stays on the current path

- [ ] **Step 1: Write a backend-ops case** that builds the current `concat(state, x) -> ssm_conv -> silu` graph for `n_t=1`, `d_conv=4`, `d_inner=2048` and compares CPU vs CUDA. Name it `SSM_CONV_DECODE1`. Run it; it must PASS before fusion.

- [ ] **Step 2: Add a fusion in `ggml_cuda_try_fuse`** only for that decode shape. Do not change the ggml graph API. If the matcher misses, unfused remains correct.

- [ ] **Step 3: Three 512-token A/B.** Hash must equal Phase 0. If hash drifts or tok/s regresses, revert Task 6 and keep Task 5 only.

---

### Task 7: Negative expert id skip (needed before prune)

**Files:**
- Modify: `ggml/src/ggml-cuda/common.cuh`
- Modify: `ggml/src/ggml-cuda/mmvq.cu` (both skip sites, ~611 and ~855)
- Modify: `ggml/src/ggml-cuda/mmvf.cu` (same pattern if ids are read)
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c` (`ggml_compute_forward_mul_mat_id`, id grouping at line 1853)
- Modify: `tests/test-backend-ops.cpp`

**Interfaces:**
- Consumes: `fusion.skip_slot` (unchanged meaning)
- Produces: `ggml_cuda_mul_mat_id_is_skipped(int32_t expert_id, int32_t skip_slot)`

```cpp
static inline bool ggml_cuda_mul_mat_id_is_skipped(int32_t expert_id, int32_t skip_slot) {
    return expert_id < 0 || (skip_slot >= 0 && expert_id == skip_slot);
}
```

- [ ] **Step 1: Add `test_mul_mat_id_neg`**

Copy `test_mul_mat_id` and override `initialize_tensors` so that after `init_mul_mat_id_tensors(ctx, n_mats)` the first id of each row is set to `-1`. Use `GGML_TYPE_Q4_K` x `GGML_TYPE_F32`, `n_mats=8`, `n_used=8`, `m=32`, `n=1`, `k=32` (decode-like).

CPU must skip `i02 < 0` or the reference will assert. In `ggml_compute_forward_mul_mat_id` replace the grouping loop body with:

```cpp
const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
if (i02 < 0) {
    float * dst_col = (float *) ((char *) dst->data + id*dst->nb[1] + iid1*dst->nb[2]);
    memset(dst_col, 0, (size_t) dst->ne[0] * sizeof(float));
    continue;
}
assert(i02 >= 0 && i02 < n_as);
MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
matrix_row_counts[i02] += 1;
```

- [ ] **Step 2: Run the new test before the CUDA skip (expect FAIL or CUDA illegal access)**

```bash
cmake --build build-main-sm75 -j8 --target test-backend-ops
./build-main-sm75/bin/test-backend-ops -o MUL_MAT_ID_NEG
```

Expected: FAIL or device error. That is the red bar.

- [ ] **Step 3: Skip negative ids in mmvq (and mmvf if it reads ids)**

Replace both `fusion.skip_slot` checks with:

```cpp
if (ids && ggml_cuda_mul_mat_id_is_skipped((int32_t) channel_x, fusion.skip_slot)) {
    // existing zero-store + return
}
```

Do not change the meaning of `skip_slot`.

- [ ] **Step 4: Re-run**

```bash
./build-main-sm75/bin/test-backend-ops -o MUL_MAT_ID_NEG
./build-main-sm75/bin/test-backend-ops -o MUL_MAT_ID
```

Expected: both PASS.

- [ ] **Step 5: Human commit (optional)**

```bash
git add ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvf.cu \
        ggml/src/ggml-cpu/ggml-cpu.c tests/test-backend-ops.cpp
# human commit: cuda : skip negative mul_mat_id experts
```

---

### Task 8: Weight prune (only if Phase 1+2 median < +10%)

**Files:**
- Modify: `ggml/src/ggml-cuda/topk-moe.cu` (grouped kernel tail)
- Modify: `tests/test-backend-ops.cpp` only if you add a prune-off case (default tests stay eps=0)
- Modify: `start-ling-tiny.sh` is **not** allowed. Export `GGML_CUDA_MOE_WEIGHT_EPS` only in the bench shell.

**Interfaces:**
- Consumes: `ggml_cuda_moe_weight_eps()`, normalized then scaled weights (sum == 2.5 when scale is 2.5)
- Produces: some `ids[k] == -1` and `weights[k] == 0`, at least one expert kept

- [ ] **Step 1: Add the prune tail to `topk_moe_grouped_cuda`**

After scale, if `weight_eps > 0`:

```cpp
// lane 0 holds the 8 weights/ids in shared or via warp
int keep = 0;
int argmax = 0;
float best = -INFINITY;
for (int k = 0; k < n_expert_used; k++) {
    if (w[k] > best) { best = w[k]; argmax = k; }
    if (w[k] >= weight_eps) keep++;
}
if (keep == 0) {
    // keep argmax only
} else {
    for (int k = 0; k < n_expert_used; k++) {
        if (w[k] < weight_eps && k != argmax) {
            w[k] = 0.f;
            ids[k] = -1;
        }
    }
}
```

If every weight is below eps, keep `argmax` and zero the others. Never emit eight `-1`s.

- [ ] **Step 2: Backend test remains eps=0** so Task 2 hashes stay comparable. Do not enable prune inside `test-backend-ops`.

- [ ] **Step 3: Screen EPS on the live server**

Rebuild `llama-server`. For `EPS` in `0.02 0.05 0.10`:

```bash
GGML_CUDA_MOE_WEIGHT_EPS=$EPS ./start-ling-tiny.sh --port 18082
# 3 x 512 with prompt-decode.txt
```

Record median tok/s, token hash, and a reasoning smoke:

```bash
curl -s http://127.0.0.1:18082/completion -d '{
  "prompt": "User: What is 17*19? Think step by step.\nAssistant:",
  "n_predict": 256,
  "temperature": 0.6,
  "reasoning": true
}'
```

Reject an EPS on empty output, a repeat loop, or a reasoning trace that never answers. Start at `0.02`. Promote the smallest EPS that reaches +10% and passes quality. If none reach +10% without failing quality, record that in `NOTES.md` and stop; do not raise EPS further as a hidden quality trade.

- [ ] **Step 4: If an EPS wins, one 2000-token confirmation** with `prompt-decode.txt`, `n-predict 2000`. Quality prompt once more.

---

### Task 9: Promotion writeup

**Files:**
- Modify: `benchmark-results/ling-tiny-hybrid/20260820-decode10/NOTES.md`
- Modify: `benchmark-results/ling-tiny-hybrid/OPT-20260818.md` only if a phase ships (append a 2026-08-20 section, do not rewrite history)

**Interfaces:**
- Consumes: all `NOTES.md` numbers
- Produces: ship / no-ship decision

- [ ] **Step 1: Fill the promotion table**

```text
phase0 median / hash
phase1 median / hash / delta
phase2a median / hash / delta
phase2b (skipped | median / hash / delta)
phase3 EPS / median / hash / quality
```

- [ ] **Step 2: Apply the spec rule**

- Ship Phase 1+2 if they are exact and help.
- Ship Phase 3 only if it is required for +10% and the quality gate holds.
- Promote only if median >= +10% vs the Task 1 baseline and no CUDA/Xid/OOM/NaN.

- [ ] **Step 3: If Phase 3 ships, document the env** in `NOTES.md` and in a comment at the top of `start-ling-tiny.sh` only after the human agrees to change the launcher. The spec freezes knobs; adding `GGML_CUDA_MOE_WEIGHT_EPS` is a launcher change and needs an explicit yes.

---

## Self-review

**Spec coverage:**

| Spec item | Task |
| --- | --- |
| Phase 0 Nsight + 3x512 baseline + hash | Task 1 |
| Grouped router fusion 128/8/4/8, no new ggml op, fallback unfused | Tasks 2-3 |
| Phase 1 hash-identical A/B | Task 4 |
| KDA SILU before reshape, existing SSM_CONV+SILU fusion | Task 5 |
| Optional state+conv if CONCAT+CPY >= 2% | Task 6 |
| Prune via `GGML_CUDA_MOE_WEIGHT_EPS`, id=-1, min 1 expert, graphs stay 8-wide | Tasks 7-8 |
| mmvq skip id<0, do not reuse skip_slot | Task 7 |
| Quality gate, 2000-token confirm at +8% | Tasks 8-9 |
| Out of scope: 1080, MTP, new ops, launcher search | Global constraints |

**Placeholder scan:** no TBD/TODO left in task steps.

**Type consistency:** `ggml_cuda_topk_moe_grouped_args`, `ggml_cuda_topk_moe_grouped_fusion`, `ggml_cuda_op_topk_moe_grouped`, `ggml_cuda_moe_weight_eps`, `ggml_cuda_mul_mat_id_is_skipped` are named the same in Tasks 3, 7, and 8.
