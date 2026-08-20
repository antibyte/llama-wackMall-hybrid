#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <climits>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float *         logits,
                                                                  float *               weights,
                                                                  int32_t *             ids,
                                                                  float *               bias,
                                                                  const int             n_rows,
                                                                  const int             n_expert_used,
                                                                  const float           clamp_val,
                                                                  const float           scale_val,
                                                                  const topk_moe_config config) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = 4;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else     if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

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

static bool ggml_cuda_topk_moe_trace() {
    static const bool on = getenv("GGML_CUDA_TOPK_MOE_TRACE") != nullptr &&
                           getenv("GGML_CUDA_TOPK_MOE_TRACE")[0] != '\0';
    return on;
}

static void ggml_cuda_dump_grouped_moe_graph(const struct ggml_cgraph * cgraph, int node_idx) {
    static std::atomic<bool> dumped{false};
    if (dumped.exchange(true)) {
        return;
    }
    const int last = std::min(cgraph->n_nodes, node_idx + 40);
    fprintf(stderr, "ggml_cuda: TOPK_MOE_GROUPED graph dump from node %d (n_nodes=%d):\n",
            node_idx, cgraph->n_nodes);
    for (int i = node_idx; i < last; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        fprintf(stderr, "  [%d] op=%s", i, ggml_op_name(t->op));
        if (t->op == GGML_OP_UNARY) {
            fprintf(stderr, "/%s", ggml_unary_op_name(ggml_get_unary_op(t)));
        }
        fprintf(stderr, " name=%s ne=[%lld,%lld,%lld,%lld]\n",
                t->name,
                (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3]);
    }
}

static bool ggml_cuda_is_sigmoid(const ggml_tensor * t) {
    return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID;
}

static bool ggml_cuda_is_neg_inf_fill(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_FILL) {
        return false;
    }
    const float v = ggml_get_op_params_f32(t, 0);
    return !std::isfinite(v) && v < 0.0f;
}

static const ggml_tensor * ggml_cuda_unwrap1(const ggml_tensor * t) {
    if (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
        return t->src[0];
    }
    return t;
}

static bool ggml_cuda_from_tensor(const ggml_tensor * t, const ggml_tensor * src) {
    return t == src || ggml_cuda_unwrap1(t) == src;
}

static bool ggml_cuda_grouped_whitelist(enum ggml_op op) {
    switch (op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_ADD:
        case GGML_OP_TOP_K:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_FILL:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CLAMP:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
            return true;
        default:
            return false;
    }
}

static bool ggml_cuda_tensor_in_range(const struct ggml_cgraph * cgraph, int start, int end, const ggml_tensor * t) {
    for (int i = start; i < end; ++i) {
        if (cgraph->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

static int ggml_cuda_find_tensor(const struct ggml_cgraph * cgraph, int start, int end, const ggml_tensor * t) {
    for (int i = start; i < end; ++i) {
        if (cgraph->nodes[i] == t) {
            return i;
        }
    }
    return -1;
}

static bool ggml_cuda_src_in_range(const struct ggml_cgraph * cgraph, int start, int end, const ggml_tensor * node) {
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = node->src[s];
        if (src && ggml_cuda_tensor_in_range(cgraph, start, end, src)) {
            return true;
        }
    }
    return false;
}

static void ggml_cuda_grouped_fail(const char * why) {
    static std::atomic<bool> logged{false};
    if (ggml_cuda_topk_moe_trace() && !logged.exchange(true)) {
        fprintf(stderr, "ggml_cuda: grouped matcher fail: %s\n", why);
    }
}

static bool ggml_cuda_topk_moe_grouped_validate(
        const struct ggml_cgraph * cgraph, int start, int end,
        ggml_cuda_topk_moe_grouped_args & args,
        ggml_cuda_topk_moe_grouped_match & match) {
    ggml_tensor ** nodes = cgraph->nodes;
    const ggml_tensor * sigmoid = nodes[start];
    if (!ggml_cuda_is_sigmoid(sigmoid)) {
        ggml_cuda_grouped_fail("start is not sigmoid");
        return false;
    }

    const ggml_tensor * logits = sigmoid->src[0];
    if (!logits || logits->type != GGML_TYPE_F32 || logits->ne[0] != 128 || !ggml_is_contiguous(logits)) {
        ggml_cuda_grouped_fail("logits shape/type/contig");
        return false;
    }
    const int64_t n_tokens = logits->ne[1];
    if (n_tokens < 1) {
        ggml_cuda_grouped_fail("n_tokens < 1");
        return false;
    }

    const ggml_tensor * bias_add = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_ADD && nodes[i]->src[0] == sigmoid &&
            nodes[i]->src[1] && nodes[i]->src[1]->ne[0] == 128 &&
            ggml_nrows(nodes[i]->src[1]) == 1) {
            bias_add = nodes[i];
            break;
        }
    }

    const ggml_tensor * groups = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_RESHAPE &&
            nodes[i]->ne[0] == 16 && nodes[i]->ne[1] == 8 && nodes[i]->ne[2] == n_tokens &&
            (nodes[i]->src[0] == sigmoid || nodes[i]->src[0] == bias_add)) {
            groups = nodes[i];
            break;
        }
    }
    if (!groups) {
        ggml_cuda_grouped_fail("no groups reshape [16,8,T]");
        return false;
    }

    const ggml_tensor * topk2 = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_TOP_K && nodes[i]->ne[0] == 2 && nodes[i]->src[0] == groups) {
            topk2 = nodes[i];
            break;
        }
    }
    if (!topk2) {
        ggml_cuda_grouped_fail("no TOP_K k=2 on groups");
        return false;
    }

    const ggml_tensor * group_vals = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_GET_ROWS && nodes[i]->src[1] == topk2 &&
            ggml_cuda_from_tensor(nodes[i]->src[0], groups)) {
            group_vals = nodes[i];
            break;
        }
    }
    if (!group_vals) {
        ggml_cuda_grouped_fail("no GET_ROWS of group top-2");
        return false;
    }

    const ggml_tensor * group_sum = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_SUM_ROWS &&
            ggml_cuda_from_tensor(nodes[i]->src[0], group_vals)) {
            group_sum = nodes[i];
            break;
        }
    }
    if (!group_sum) {
        ggml_cuda_grouped_fail("no SUM_ROWS of group vals");
        return false;
    }

    const ggml_tensor * group_scores = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_RESHAPE &&
            nodes[i]->ne[0] == 8 && nodes[i]->ne[1] == n_tokens &&
            ggml_cuda_from_tensor(nodes[i], group_sum)) {
            group_scores = nodes[i];
            break;
        }
    }
    if (!group_scores) {
        ggml_cuda_grouped_fail("no group scores reshape [8,T]");
        return false;
    }

    const ggml_tensor * topk4 = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_TOP_K && nodes[i]->ne[0] == 4 && nodes[i]->src[0] == group_scores) {
            topk4 = nodes[i];
            break;
        }
    }
    if (!topk4) {
        ggml_cuda_grouped_fail("no TOP_K k=4 on group scores");
        return false;
    }

    const ggml_tensor * kept = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_GET_ROWS && nodes[i]->src[0] == groups && nodes[i]->src[1] == topk4) {
            kept = nodes[i];
            break;
        }
    }
    if (!kept) {
        ggml_cuda_grouped_fail("no GET_ROWS of kept groups");
        return false;
    }

    const ggml_tensor * fill = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (ggml_cuda_is_neg_inf_fill(nodes[i]) && ggml_cuda_from_tensor(nodes[i]->src[0], groups)) {
            fill = nodes[i];
            break;
        }
    }
    if (!fill) {
        ggml_cuda_grouped_fail("no FILL(-inf) of groups");
        return false;
    }

    const ggml_tensor * set_rows = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_SET_ROWS &&
            nodes[i]->src[1] == topk4 &&
            ggml_cuda_from_tensor(nodes[i]->src[2], fill) &&
            ggml_cuda_from_tensor(nodes[i]->src[0], kept)) {
            set_rows = nodes[i];
            break;
        }
    }
    if (!set_rows) {
        ggml_cuda_grouped_fail("no SET_ROWS mask");
        return false;
    }

    const ggml_tensor * flat = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_RESHAPE &&
            nodes[i]->ne[0] == 128 && nodes[i]->ne[1] == n_tokens &&
            ggml_cuda_from_tensor(nodes[i], set_rows)) {
            flat = nodes[i];
            break;
        }
    }
    if (!flat) {
        ggml_cuda_grouped_fail("no reshape [128,T]");
        return false;
    }

    ggml_tensor * ids = nullptr;
    int ids_idx = -1;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_TOP_K && nodes[i]->ne[0] == 8 && nodes[i]->src[0] == flat) {
            ids = nodes[i];
            ids_idx = i;
            break;
        }
    }
    if (!ids || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids)) {
        ggml_cuda_grouped_fail("no TOP_K k=8 ids");
        return false;
    }

    const ggml_tensor * probs_rows = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_RESHAPE &&
            nodes[i]->ne[0] == 1 && nodes[i]->ne[1] == 128 && nodes[i]->ne[2] == n_tokens &&
            nodes[i]->src[0] == sigmoid) {
            probs_rows = nodes[i];
            break;
        }
    }
    if (!probs_rows) {
        ggml_cuda_grouped_fail("no unbiased sigmoid reshape [1,128,T]");
        return false;
    }

    ggml_tensor * gathered = nullptr;
    for (int i = start + 1; i < end; ++i) {
        if (nodes[i]->op == GGML_OP_GET_ROWS && nodes[i]->src[0] == probs_rows && nodes[i]->src[1] == ids) {
            gathered = nodes[i];
            break;
        }
    }
    if (!gathered) {
        ggml_cuda_grouped_fail("no GET_ROWS of unbiased probs");
        return false;
    }

    ggml_tensor * weights = gathered;
    const ggml_tensor * clamp = nullptr;
    const ggml_tensor * scale = nullptr;

    int widx = ggml_cuda_find_tensor(cgraph, start, end, gathered);
    if (widx < 0) {
        return false;
    }

    if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_RESHAPE &&
        nodes[widx + 1]->src[0] == weights &&
        nodes[widx + 1]->ne[0] == 8 && nodes[widx + 1]->ne[1] == n_tokens) {
        weights = nodes[++widx];
        if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_SUM_ROWS &&
            nodes[widx + 1]->src[0] == weights) {
            ggml_tensor * sum = nodes[++widx];
            if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_CLAMP &&
                nodes[widx + 1]->src[0] == sum) {
                clamp = nodes[++widx];
                if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_DIV &&
                    nodes[widx + 1]->src[0] == weights && nodes[widx + 1]->src[1] == clamp) {
                    weights = nodes[++widx];
                    if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_RESHAPE &&
                        nodes[widx + 1]->src[0] == weights &&
                        nodes[widx + 1]->ne[0] == 1 && nodes[widx + 1]->ne[1] == 8 &&
                        nodes[widx + 1]->ne[2] == n_tokens) {
                        weights = nodes[++widx];
                    } else {
                        ggml_cuda_grouped_fail("norm missing final reshape [1,8,T]");
                        return false;
                    }
                } else {
                    ggml_cuda_grouped_fail("norm missing DIV");
                    return false;
                }
            } else {
                ggml_cuda_grouped_fail("norm missing CLAMP");
                return false;
            }
        }
    }

    if (widx + 1 < end && nodes[widx + 1]->op == GGML_OP_SCALE &&
        nodes[widx + 1]->src[0] == weights) {
        scale = nodes[++widx];
        weights = nodes[widx];
    }

    if (widx != end - 1) {
        ggml_cuda_grouped_fail("trailing nodes after weights");
        return false;
    }
    if (weights->type != GGML_TYPE_F32 || weights->ne[1] != 8 || !ggml_is_contiguous(weights)) {
        ggml_cuda_grouped_fail("weights shape/type/contig");
        return false;
    }

    args.sigmoid         = true;
    args.prob_bias       = bias_add != nullptr;
    args.norm            = clamp != nullptr;
    args.scale           = scale != nullptr;
    args.n_groups        = 8;
    args.n_exp_per_group = 16;
    args.n_group_used    = 4;
    args.group_top       = 2;
    args.weight_eps      = 0.0f;

    match.logits      = logits;
    match.weights     = weights;
    match.ids         = ids;
    match.clamp       = clamp;
    match.scale       = scale;
    match.bias        = bias_add ? bias_add->src[1] : nullptr;
    match.n_nodes     = end - start;
    match.ids_idx     = ids_idx;
    match.weights_idx = widx;
    return true;
}

bool ggml_cuda_topk_moe_grouped_match_nodes(const struct ggml_cgraph * cgraph, int node_idx,
                                            ggml_cuda_topk_moe_grouped_args & args,
                                            ggml_cuda_topk_moe_grouped_match & match) {
    match = {};
    args = {};

    if (!cgraph || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return false;
    }
    if (!ggml_cuda_is_sigmoid(cgraph->nodes[node_idx])) {
        return false;
    }

    const int n_nodes = cgraph->n_nodes;
    int end = node_idx + 1;
    const int limit = std::min(n_nodes, node_idx + 31);
    while (end < limit) {
        const ggml_tensor * node = cgraph->nodes[end];
        if (!ggml_cuda_grouped_whitelist(node->op) ||
            !ggml_cuda_src_in_range(cgraph, node_idx, end, node)) {
            break;
        }
        if (node->op == GGML_OP_ADD && node->src[0] != cgraph->nodes[node_idx]) {
            break;
        }
        if (node->op == GGML_OP_TOP_K &&
            node->ne[0] != 2 && node->ne[0] != 4 && node->ne[0] != 8) {
            break;
        }
        ++end;
    }

    if (end - node_idx < 16) {
        if (ggml_cuda_topk_moe_trace()) {
            bool saw_topk = false;
            for (int i = node_idx; i < end; ++i) {
                if (cgraph->nodes[i]->op == GGML_OP_TOP_K) {
                    saw_topk = true;
                    break;
                }
            }
            if (saw_topk) {
                fprintf(stderr, "ggml_cuda: grouped matcher short walk end-start=%d\n", end - node_idx);
                ggml_cuda_dump_grouped_moe_graph(cgraph, node_idx);
            }
        }
        return false;
    }

    if (!ggml_cuda_topk_moe_grouped_validate(cgraph, node_idx, end, args, match)) {
        if (ggml_cuda_topk_moe_trace()) {
            ggml_cuda_dump_grouped_moe_graph(cgraph, node_idx);
        }
        return false;
    }
    return true;
}

bool ggml_cuda_topk_moe_grouped_fusion(const struct ggml_cgraph * cgraph, int node_idx,
                                       ggml_cuda_topk_moe_grouped_args & args) {
    ggml_cuda_topk_moe_grouped_match match;
    return ggml_cuda_topk_moe_grouped_match_nodes(cgraph, node_idx, args, match);
}

template <int n_experts, int n_groups, int n_group_used, int n_expert_used, int group_top, bool has_bias>
__launch_bounds__(WARP_SIZE, 1)
__global__ void topk_moe_grouped_cuda(const float * logits, float * weights, int32_t * ids,
                                      const float * bias, int n_rows, float clamp_val,
                                      float scale_val, float weight_eps, bool with_norm) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    static_assert(n_experts == 128 && n_groups == 8 && n_group_used == 4 &&
                  n_expert_used == 8 && group_top == 2, "Ling-tiny grouped router only");
    static_assert(n_experts % WARP_SIZE == 0, "experts must be a multiple of warp size");

    logits  += n_experts * row;
    weights += n_expert_used * row;
    ids     += n_expert_used * row;

    constexpr int experts_per_thread = n_experts / WARP_SIZE;
    constexpr int n_exp_per_group    = n_experts / n_groups;

    float prob[experts_per_thread];
    float sel[experts_per_thread];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int e = lane + i * WARP_SIZE;
        float p = 1.f / (1.f + expf(-logits[e]));
        if (__isnanf(p)) {
            p = -FLT_MAX;
        }
        prob[i] = p;
        if constexpr (has_bias) {
            sel[i] = p + bias[e];
        } else {
            sel[i] = p;
        }
        if (__isnanf(sel[i])) {
            sel[i] = -FLT_MAX;
        }
    }

    float group_score[n_groups];
#pragma unroll
    for (int g = 0; g < n_groups; ++g) {
        const int  slot     = g / 2;
        const int  half     = g & 1;
        const bool in_group = (lane >> 4) == half;
        const int  e        = lane + slot * WARP_SIZE;

        float taken_idx[group_top];
        float score = 0.f;
#pragma unroll
        for (int t = 0; t < group_top; ++t) {
            bool taken = false;
#pragma unroll
            for (int p = 0; p < t; ++p) {
                taken = taken || (e == (int) taken_idx[p]);
            }
            float v = (in_group && !taken) ? sel[slot] : -INFINITY;
            int   idx = (in_group && !taken) ? e : INT_MAX;
#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float ov  = __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE);
                const int   oi  = __shfl_xor_sync(0xFFFFFFFF, idx, mask, WARP_SIZE);
                if (oi != INT_MAX && (idx == INT_MAX || ov > v || (ov == v && oi < idx))) {
                    v   = ov;
                    idx = oi;
                }
            }
            taken_idx[t] = (float) idx;
            score += v;
        }
        group_score[g] = score;
    }

    int   kept[n_group_used];
    float gs[n_groups];
#pragma unroll
    for (int g = 0; g < n_groups; ++g) {
        gs[g] = group_score[g];
    }
#pragma unroll
    for (int k = 0; k < n_group_used; ++k) {
        float max_val = gs[0];
        int   max_g   = 0;
#pragma unroll
        for (int g = 1; g < n_groups; ++g) {
            if (gs[g] > max_val || (gs[g] == max_val && g < max_g)) {
                max_val = gs[g];
                max_g   = g;
            }
        }
        kept[k]   = max_g;
        gs[max_g] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int e = lane + i * WARP_SIZE;
        const int g = e / n_exp_per_group;
        bool keep = false;
#pragma unroll
        for (int k = 0; k < n_group_used; ++k) {
            keep = keep || (kept[k] == g);
        }
        if (!keep) {
            sel[i] = -INFINITY;
        }
    }

    ggml_cuda_pdl_lc();

    float out_w = 0.f;
    for (int k = 0; k < n_expert_used; ++k) {
        float max_sel    = sel[0];
        float max_prob   = prob[0];
        int   max_expert = lane;
#pragma unroll
        for (int i = 1; i < experts_per_thread; ++i) {
            const int e = lane + i * WARP_SIZE;
            if (sel[i] > max_sel || (sel[i] == max_sel && e < max_expert)) {
                max_sel    = sel[i];
                max_prob   = prob[i];
                max_expert = e;
            }
        }
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float vs = __shfl_xor_sync(0xFFFFFFFF, max_sel, mask, WARP_SIZE);
            const float vp = __shfl_xor_sync(0xFFFFFFFF, max_prob, mask, WARP_SIZE);
            const int   ve = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (vs > max_sel || (vs == max_sel && ve < max_expert)) {
                max_sel    = vs;
                max_prob   = vp;
                max_expert = ve;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            sel[max_expert / WARP_SIZE] = -INFINITY;
        }
        if (lane == k) {
            ids[k] = max_expert;
            out_w  = max_prob;
        }
    }

    if (with_norm) {
        float sum = (lane < n_expert_used) ? out_w : 0.f;
        sum = warp_reduce_sum(sum);
        sum = fmaxf(sum, clamp_val);
        if (lane < n_expert_used) {
            out_w /= sum;
        }
    }

    const float w = (lane < n_expert_used) ? out_w * scale_val : 0.f;

    if (weight_eps > 0.0f) {
        int   keep   = 0;
        int   argmax = 0;
        float best   = -INFINITY;
#pragma unroll
        for (int k = 0; k < n_expert_used; ++k) {
            const float wk = __shfl_sync(0xFFFFFFFF, w, k, WARP_SIZE);
            if (wk > best) {
                best   = wk;
                argmax = k;
            }
            if (wk >= weight_eps) {
                ++keep;
            }
        }
        if (lane < n_expert_used) {
            if (keep == 0) {
                if (lane != argmax) {
                    weights[lane] = 0.f;
                    ids[lane]     = -1;
                } else {
                    weights[lane] = w;
                }
            } else if (w < weight_eps && lane != argmax) {
                weights[lane] = 0.f;
                ids[lane]     = -1;
            } else {
                weights[lane] = w;
            }
        }
    } else if (lane < n_expert_used) {
        weights[lane] = w;
    }
}

void ggml_cuda_op_topk_moe_grouped(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * logits,
                                   ggml_tensor * weights,
                                   ggml_tensor * ids,
                                   const ggml_tensor * clamp,
                                   const ggml_tensor * scale,
                                   const ggml_tensor * bias,
                                   const ggml_cuda_topk_moe_grouped_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = (int) logits->ne[0];
    const int n_rows    = (int) logits->ne[1];
    GGML_ASSERT(n_experts == 128);
    GGML_ASSERT(weights->ne[1] == 8);
    GGML_ASSERT(ids->ne[0] == 8);
    GGML_ASSERT(args.n_groups == 8);
    GGML_ASSERT(args.n_exp_per_group == 16);
    GGML_ASSERT(args.n_group_used == 4);
    GGML_ASSERT(args.group_top == 2);

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    const float * bias_d    = bias ? (const float *) bias->data : nullptr;

    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const bool  with_norm = args.norm;
    // mmq down-proj still faults on id < 0; only prune decode-sized (mmvq) rows
    const float weight_eps = n_rows <= 8 ? args.weight_eps : 0.0f;

    const dim3 grid_dims(n_rows, 1, 1);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if (bias) {
        ggml_cuda_kernel_launch(topk_moe_grouped_cuda<128, 8, 4, 8, 2, true>, launch_params,
            logits_d, weights_d, ids_d, bias_d, n_rows, clamp_val, scale_val, weight_eps, with_norm);
    } else {
        ggml_cuda_kernel_launch(topk_moe_grouped_cuda<128, 8, 4, 8, 2, false>, launch_params,
            logits_d, weights_d, ids_d, bias_d, n_rows, clamp_val, scale_val, weight_eps, with_norm);
    }
}
