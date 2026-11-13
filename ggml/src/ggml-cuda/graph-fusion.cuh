#pragma once

#include "common.cuh"

#include <algorithm>
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

// l2_norm of q/k, the get_rows gather of the recurrent state and the gate activations, done by gated_delta_net
struct ggml_cuda_gdn_prologue {
    const ggml_tensor * q_src = nullptr;
    const ggml_tensor * k_src = nullptr;
    float               q_eps = 0.0f;
    float               k_eps = 0.0f;
    const float *       state_rows       = nullptr; // state of sequence s: state_rows + state_ids[s]*state_row_stride
    const int32_t *     state_ids        = nullptr;
    int64_t             state_row_stride = 0;       // elements
    // KDA gate g = g_scale * sigmoid(g_raw * g_a[head]), same layout as g
    const float *       g_raw   = nullptr;
    const float *       g_a     = nullptr;
    float               g_scale = 0.0f;
    const float *       beta_raw = nullptr;         // beta = sigmoid(beta_raw), same layout as beta
};

// a one-column matvec that also does the work of the nodes behind it and writes their output (ggml_cuda_mm_epilogue)
struct ggml_cuda_mm_takeover {
    ggml_tensor *                 dst = nullptr;
    ggml_cuda_mm_fusion_args_host fusion;
    // conv epilogue: the state view it reads and the copies it replaces
    const ggml_tensor *           conv_a      = nullptr;
    const ggml_tensor *           conv_cpy[2] = { nullptr, nullptr };
};

#define GGML_CUDA_Q8_1_REUSE_BYTES     (1u << 20)
#define GGML_CUDA_Q8_1_REUSE_MAX_SLOTS 8

// q8_1 copy of a matvec input, reused by later matvecs on the same data until a node writes the source
struct ggml_cuda_q8_1_slot {
    bool         valid  = false;
    bool         fresh  = false; // written by the kernel that produces the source, whose own write does not count
    void *       buf    = nullptr;
    const char * lo     = nullptr;
    const char * hi     = nullptr;
    int64_t      key[8] = {}; // src1 data, ne[0..3], strides 1..3 in elements
    int64_t      used   = 0;
};

// one row y = rms_norm(x) * w: work the norm kernel does on top (rms_norm_row_f32)
struct ggml_cuda_norm_row {
    // x = a + b (the residual add in front of the norm), written to sum
    const float * add_a   = nullptr;
    const float * add_b   = nullptr;
    float *       sum     = nullptr;
    int           add_idx = -1;    // node index of the add: folded only if no other fusion ran it
    bool          q8_1    = false; // also write the q8_1 copy of y for the matvecs that read it
};

struct ggml_cuda_graph_fusion {
    std::vector<char>                                                   skip_idx; // by node index
    std::vector<char>                                                   ran_idx;  // by node index: run as part of an adjacent fusion
    int                                                                 n_skip = 0;
    std::unordered_map<const ggml_tensor *, ggml_cuda_ssm_conv_concat> conv; // keyed by the SSM_CONV node
    std::unordered_map<const ggml_tensor *, ggml_cuda_gdn_prologue>    gdn;  // keyed by the GATED_DELTA_NET node
    std::unordered_map<const ggml_tensor *, ggml_cuda_mm_takeover>     mm;   // keyed by the MUL_MAT / MUL_MAT_ID node

    std::unordered_map<const ggml_tensor *, ggml_cuda_norm_row>        norm; // keyed by the RMS_NORM node
    // rms_norm * weight that reads its input here instead of from a dropped cont, keyed by the RMS_NORM node
    std::unordered_map<const ggml_tensor *, const float *>             norm_src;

    bool                q8_1_active = false;
    int                 q8_1_n      = 1; // slots in use (GGML_CUDA_Q8_1_REUSE_SLOTS)
    int64_t             q8_1_clock  = 0;
    ggml_cuda_q8_1_slot q8_1[GGML_CUDA_Q8_1_REUSE_MAX_SLOTS];

    ggml_cuda_q8_1_slot * q8_1_find(const int64_t * key) {
        for (int i = 0; i < q8_1_n; ++i) {
            if (q8_1[i].valid && std::equal(key, key + 8, q8_1[i].key)) {
                q8_1[i].used = ++q8_1_clock;
                return &q8_1[i];
            }
        }
        return nullptr;
    }

    // slot for a new copy: a free one, else the least recently used
    ggml_cuda_q8_1_slot & q8_1_claim(const int64_t * key, const char * lo, const char * hi) {
        int best = 0;
        for (int i = 0; i < q8_1_n; ++i) {
            if (!q8_1[i].valid) {
                best = i;
                break;
            }
            if (q8_1[i].used < q8_1[best].used) {
                best = i;
            }
        }
        ggml_cuda_q8_1_slot & slot = q8_1[best];
        if (slot.buf == nullptr) {
            CUDA_CHECK(cudaMalloc(&slot.buf, GGML_CUDA_Q8_1_REUSE_BYTES));
        }
        std::copy(key, key + 8, slot.key);
        slot.valid = true;
        slot.fresh = false;
        slot.lo    = lo;
        slot.hi    = hi;
        slot.used  = ++q8_1_clock;
        return slot;
    }

    // a write to [data, data + n) invalidates the copies of that memory
    void q8_1_written(const void * data, size_t n) {
        const char * lo = (const char *) data;
        for (int i = 0; i < q8_1_n; ++i) {
            if (q8_1[i].valid && lo < q8_1[i].hi && q8_1[i].lo < lo + n) {
                if (!(q8_1[i].fresh && lo == q8_1[i].lo)) {
                    q8_1[i].valid = false;
                }
            }
        }
    }

    // the producing node of a fresh copy is done
    void q8_1_settle() {
        for (int i = 0; i < q8_1_n; ++i) {
            q8_1[i].fresh = false;
        }
    }

    void q8_1_reset() {
        for (ggml_cuda_q8_1_slot & slot : q8_1) {
            slot.valid = false;
        }
    }

    ~ggml_cuda_graph_fusion() {
        for (ggml_cuda_q8_1_slot & slot : q8_1) {
            if (slot.buf != nullptr) {
                CUDA_CHECK(cudaFree(slot.buf));
            }
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

static inline const ggml_cuda_norm_row * ggml_cuda_graph_fusion_norm(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.graph_fusion == nullptr || ctx.graph_fusion->norm.empty()) {
        return nullptr;
    }
    auto it = ctx.graph_fusion->norm.find(node);
    return it == ctx.graph_fusion->norm.end() ? nullptr : &it->second;
}

static inline const ggml_cuda_gdn_prologue * ggml_cuda_graph_fusion_gdn(const ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (ctx.graph_fusion == nullptr || ctx.graph_fusion->gdn.empty()) {
        return nullptr;
    }
    auto it = ctx.graph_fusion->gdn.find(node);
    return it == ctx.graph_fusion->gdn.end() ? nullptr : &it->second;
}
