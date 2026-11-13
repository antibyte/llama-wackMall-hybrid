#pragma once

#include "common.cuh"

#include <unordered_map>
#include <vector>

// Fusions whose first node comes earlier in the graph than the kernel that takes over its work.
// The early node is skipped out of order; the plan is rebuilt for every graph evaluation (and
// therefore for every CUDA graph capture) by ggml_cuda_graph_fusion_plan in ggml-cuda.cu.

#define GGML_CUDA_CONV_CONCAT_MAX_CPY 8

// concat(conv_state, x) -> ssm_conv, plus the cpy of the new conv state into the cache
struct ggml_cuda_ssm_conv_concat {
    const char * a;     // conv state, concat columns [0, ne0_a)
    int64_t      a_nb0, a_nb1, a_nb2;
    int          ne0_a;
    const char * b;     // new tokens, concat columns [ne0_a, ne0_a + n_t)
    int64_t      b_nb0, b_nb1, b_nb2;
    int          n_cpy;
    int          cpy_col[GGML_CUDA_CONV_CONCAT_MAX_CPY];        // first concat column of the copied window
    char *       cpy_dst[GGML_CUDA_CONV_CONCAT_MAX_CPY];        // window rows of one sequence are contiguous
    int64_t      cpy_seq_stride[GGML_CUDA_CONV_CONCAT_MAX_CPY]; // bytes between sequences
};

// l2_norm of q/k and the get_rows gather of the recurrent state, done by gated_delta_net
struct ggml_cuda_gdn_prologue {
    const ggml_tensor * q_src = nullptr;
    const ggml_tensor * k_src = nullptr;
    float               q_eps = 0.0f;
    float               k_eps = 0.0f;
    const float *       state_rows       = nullptr; // state of sequence s: state_rows + state_ids[s]*state_row_stride
    const int32_t *     state_ids        = nullptr;
    int64_t             state_row_stride = 0;       // elements
};

#define GGML_CUDA_Q8_1_REUSE_BYTES (1u << 20)

struct ggml_cuda_graph_fusion {
    std::vector<char>                                                   skip_idx; // by node index
    int                                                                 n_skip = 0;
    std::unordered_map<const ggml_tensor *, ggml_cuda_ssm_conv_concat> conv; // keyed by the SSM_CONV node
    std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_prologue>    gdn;  // keyed by the GATED_DELTA_NET node

    // q8_1 copy of the last mmvq src1, reused by later matvecs on the same data until a node writes it
    bool         q8_1_active = false;
    bool         q8_1_valid  = false;
    void *       q8_1_buf    = nullptr;
    const char * q8_1_lo     = nullptr;
    const char * q8_1_hi     = nullptr;
    int64_t      q8_1_key[8] = {};

    ~ggml_cuda_graph_fusion() {
        if (q8_1_buf != nullptr) {
            CUDA_CHECK(cudaFree(q8_1_buf));
        }
    }
};

static inline const ggml_cuda_ssm_conv_concat * ggml_cuda_graph_fusion_conv(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.graph_fusion == nullptr || ctx.graph_fusion->conv.empty()) {
        return nullptr;
    }
    auto it = ctx.graph_fusion->conv.find(node);
    return it == ctx.graph_fusion->conv.end() ? nullptr : &it->second;
}

static inline const ggml_cuda_gdn_prologue * ggml_cuda_graph_fusion_gdn(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.graph_fusion == nullptr || ctx.graph_fusion->gdn.empty()) {
        return nullptr;
    }
    auto it = ctx.graph_fusion->gdn.find(node);
    return it == ctx.graph_fusion->gdn.end() ? nullptr : &it->second;
}
