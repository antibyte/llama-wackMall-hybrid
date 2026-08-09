#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static int get_rows_override(const char * name) {
    const char * env = std::getenv(name);
    if (!env || !env[0]) {
        return 0;
    }
    const int parsed = std::atoi(env);
    if (parsed == 0 || parsed == 1 || parsed == 2 || parsed == 4) {
        return parsed;
    }
    std::fprintf(stderr, "ggml_cuda: ignoring invalid %s='%s' (expected 0, 1, 2, or 4)\n", name, env);
    return 0;
}

static int get_q8_0_ncols1_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_Q8_NCOLS1_ROWS");
    }();
    return value;
}

static int get_q8_0_ncols2_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_Q8_NCOLS2_ROWS");
    }();
    return value;
}

static int get_q8_0_ncols3_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_Q8_NCOLS3_ROWS");
    }();
    return value;
}

static int get_q6_k_ncols1_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_Q6_K_NCOLS1_ROWS");
    }();
    return value;
}

static int get_q6_k_ncols3_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_Q6_K_NCOLS3_ROWS");
    }();
    return value;
}

static int get_moe_fused_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_MOE_FUSED_ROWS");
    }();
    return value;
}

static int get_moe_plain_rows_override() {
    static const int value = [] {
        return get_rows_override("GGML_CUDA_MMVQ_MOE_PLAIN_ROWS");
    }();
    return value;
}

// Bit-identical multi-token MoE weight staging: when every token maps to the
// same expert, warp 0 copies each thread's weight block into shared memory and
// all token warps reuse it. The per-token K assignment and reduction order are
// unchanged, so float results match the independent-load path.
static bool get_moe_share_weights() {
    static const bool value = [] {
        const char * env = std::getenv("GGML_CUDA_MMVQ_MOE_SHARE_WEIGHTS");
        return env && env[0] && std::atoi(env) != 0;
    }();
    return value;
}

// Compact hot-only MMVQ launches for expert-tier skip_slot:
//   0 = off (default): full grid + per-channel sentinel early-exit
//   1 = on: memset dst, pack non-sentinel (channel[,token]) work, launch only
//           active channels. Bit-identical to skip-sentinel. Falls back to the
//           full grid while a CUDA graph is being captured (variable grid is
//           not capture-safe). Requires skip_slot >= 0 on the MUL_MAT_ID path.
static int get_mmvq_compact_skip() {
    static const int value = [] {
        const char * env = std::getenv("GGML_CUDA_MMVQ_COMPACT_SKIP");
        if (!env || !env[0]) {
            return 0;
        }
        const int parsed = std::atoi(env);
        if (parsed == 0 || parsed == 1) {
            return parsed;
        }
        std::fprintf(stderr,
                "ggml_cuda: ignoring invalid GGML_CUDA_MMVQ_COMPACT_SKIP='%s' (expected 0 or 1)\n",
                env);
        return 0;
    }();
    return value;
}

// Pack non-sentinel expert channels for single-token MUL_MAT_ID (ncols_dst == 1).
// compact_ch[i] = original channel_dst for the i-th active work item.
static __global__ void mmvq_compact_channels_kernel(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ compact_ch,
        int * __restrict__ n_active,
        const int nchannels,
        const int skip_slot) {
    const int ch = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= nchannels) {
        return;
    }
    if (ids[ch] == skip_slot) {
        return;
    }
    const int slot = atomicAdd(n_active, 1);
    compact_ch[slot] = ch;
}

// Pack non-sentinel (channel_dst, token) pairs for multi-token MoE MMVQ.
static __global__ void mmvq_compact_pairs_kernel(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ compact_ch,
        int32_t * __restrict__ compact_tok,
        int * __restrict__ n_active,
        const int nchannels,
        const int ncols_dst,
        const int ids_stride,
        const int skip_slot) {
    const int work = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int n_work = nchannels * ncols_dst;
    if (work >= n_work) {
        return;
    }
    const int tok = work / nchannels;
    const int ch  = work - tok * nchannels;
    if (ids[ch + tok * ids_stride] == skip_slot) {
        return;
    }
    const int slot = atomicAdd(n_active, 1);
    compact_ch[slot]  = ch;
    compact_tok[slot] = tok;
}

static bool mmvq_stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

// Optional remapping from launch blockIdx.y to the original channel/token.
// When ch is null the launch grid uses the full channel axis and identity map.
struct mmvq_compact_args {
    const int32_t * ch  = nullptr;
    const int32_t * tok = nullptr;
    int             n_launch = 0; // grid.y when ch != nullptr
};

// Maps quantized MMVQ types to their on-device block struct for shared staging.
template <ggml_type type>
struct mmvq_weight_block {
    static constexpr bool supported = false;
    static constexpr int  nbytes    = 0;
};

#define MMVQ_DEFINE_WEIGHT_BLOCK(TYPE_ENUM, BLOCK_T)          \
    template <>                                               \
    struct mmvq_weight_block<TYPE_ENUM> {                     \
        static constexpr bool supported = true;               \
        using type_t = BLOCK_T;                               \
        static constexpr int nbytes = (int) sizeof(BLOCK_T);  \
    }

MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q4_0,    block_q4_0);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q4_1,    block_q4_1);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q5_0,    block_q5_0);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q5_1,    block_q5_1);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q8_0,    block_q8_0);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q2_K,    block_q2_K);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q3_K,    block_q3_K);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q4_K,    block_q4_K);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q5_K,    block_q5_K);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_Q6_K,    block_q6_K);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ2_XXS, block_iq2_xxs);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ2_XS,  block_iq2_xs);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ2_S,   block_iq2_s);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ3_XXS, block_iq3_xxs);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ3_S,   block_iq3_s);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ1_S,   block_iq1_s);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ4_NL,  block_iq4_nl);
MMVQ_DEFINE_WEIGHT_BLOCK(GGML_TYPE_IQ4_XS,  block_iq4_xs);

#undef MMVQ_DEFINE_WEIGHT_BLOCK

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
#ifdef GGML_CUDA_PASCAL_MMVQ_TUNING
    MMVQ_PARAMETERS_PASCAL_DP4A,
#endif
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(GGML_CUDA_PASCAL_MMVQ_TUNING) && defined(__CUDA_ARCH__) && \
        __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A && __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    return MMVQ_PARAMETERS_PASCAL_DP4A;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    const int arch = ggml_cuda_highest_compiled_arch(cc);
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && arch >= GGML_CUDA_CC_TURING && arch < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
#ifdef GGML_CUDA_PASCAL_MMVQ_TUNING
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && arch >= GGML_CUDA_CC_DP4A && arch < GGML_CUDA_CC_VOLTA) {
        return MMVQ_PARAMETERS_PASCAL_DP4A;
    }
#endif
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC
#ifdef GGML_CUDA_PASCAL_MMVQ_TUNING
            || table_id == MMVQ_PARAMETERS_PASCAL_DP4A
#endif
            ) {
#ifdef GGML_CUDA_PASCAL_MMVQ_TUNING
        if (table_id == MMVQ_PARAMETERS_PASCAL_DP4A && ncols_dst == 1) {
            return 2;
        }
#endif
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC
#ifdef GGML_CUDA_PASCAL_MMVQ_TUNING
            || table_id == MMVQ_PARAMETERS_PASCAL_DP4A
#endif
            || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, int rows_per_block_override = 0>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const int32_t * compact_ch_ptr) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int32_t * GGML_CUDA_RESTRICT compact_ch = compact_ch_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = rows_per_block_override > 0 ? rows_per_block_override :
            calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    // Compact hot-only launches remap blockIdx.y -> original channel_dst.
    const uint32_t channel_dst = compact_ch ? (uint32_t) compact_ch[blockIdx.y] : blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    // Load ids before pdl_sync so sentinel work can exit without the barrier.
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : 0;
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    // Expert-tier sentinel: cold experts map to a zeroed weight slot. The math
    // result is zero, so skip the quantized load/dot and write zeros.
    // Compact launches pre-zero dst and only schedule active channels, so the
    // zero-store is unnecessary when compact_ch is set.
    if (ids && fusion.skip_slot >= 0 && (int32_t) channel_x == fusion.skip_slot) {
        if (!compact_ch && threadIdx.y == 0) {
            float * dst_row = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                        dst_row[j*stride_col_dst + i] = 0.0f;
                    }
                }
            }
        }
        return;
    }

    ggml_cuda_pdl_sync();
    if (!(ncols_dst == 1 && ids)) {
        channel_x  = fastdiv(channel_dst, channel_ratio);
        channel_y  = channel_dst;
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
//
// Optional bit-identical weight staging (GGML_CUDA_MMVQ_MOE_SHARE_WEIGHTS=1):
// when every token maps to the same expert, warp 0 copies each thread's weight
// block into shared memory and every token warp reuses it. The kbx schedule and
// warp reduction order are unchanged, so results match the independent path.
// A previous attempt that reassigned warps onto K was rejected for hash drift.
template <ggml_type type, int c_rows_per_block, bool has_fusion>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int share_weights,
        const int32_t * compact_ch_ptr, const int32_t * compact_tok_ptr) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int32_t * GGML_CUDA_RESTRICT compact_ch  = compact_ch_ptr;
    const int32_t * GGML_CUDA_RESTRICT compact_tok = compact_tok_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    using weight_block_info = mmvq_weight_block<type>;
    constexpr bool stage_supported = weight_block_info::supported;
    constexpr int  stage_nbytes    = weight_block_info::nbytes;
    constexpr int  stage_bytes_needed =
            warp_size * c_rows_per_block * stage_nbytes * (has_fusion ? 2 : 1);
    // Keep a safety margin under the sm_75 48 KiB shared-memory limit.
    constexpr bool can_stage = stage_supported && stage_bytes_needed > 0 &&
            stage_bytes_needed <= 40 * 1024;

    // Pair-compact launches one work item per (channel, token) with block.y == 1.
    // The default path keeps one warp per token (threadIdx.y).
    const uint32_t token_idx   = compact_tok ? (uint32_t) compact_tok[blockIdx.y] : threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = compact_ch ? (uint32_t) compact_ch[blockIdx.y] : blockIdx.y;

    if (!compact_tok && token_idx >= ncols_dst) {
        return;
    }

    const uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    // share_mode: 0 = independent loads, 1 = all sentinel skip, 2 = shared stage
    // Pair-compact schedules only active (channel,token) work, so staging that
    // compares all tokens in a channel block is disabled for that path.
    int share_mode = 0;
    if constexpr (can_stage) {
        if (share_weights && !compact_ch) {
            __shared__ int32_t sh_channel[MMVQ_MAX_BATCH_SIZE];
            __shared__ int     sh_mode;
            if (threadIdx.x == 0) {
                sh_channel[token_idx] = (int32_t) channel_x;
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                const int32_t c0 = sh_channel[0];
                bool all_same = true;
                for (uint32_t t = 1; t < ncols_dst; ++t) {
                    if (sh_channel[t] != c0) {
                        all_same = false;
                        break;
                    }
                }
                if (all_same && fusion.skip_slot >= 0 && c0 == fusion.skip_slot) {
                    sh_mode = 1;
                } else if (all_same) {
                    sh_mode = 2;
                } else {
                    sh_mode = 0;
                }
            }
            __syncthreads();
            share_mode = sh_mode;
        }
    }

    if (share_mode == 1 ||
            (share_mode == 0 && fusion.skip_slot >= 0 && (int32_t) channel_x == fusion.skip_slot)) {
        // Compact path pre-zeros dst; skip the redundant store.
        if (!compact_ch && threadIdx.x < c_rows_per_block &&
                (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = 0.0f;
        }
        return;
    }

    ggml_cuda_pdl_sync();

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;
    const void * vgate = has_fusion ? fusion.gate : nullptr;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    if constexpr (can_stage) {
        if (share_mode == 2) {
            using block_t = typename weight_block_info::type_t;
            __shared__ block_t sh_w[warp_size][c_rows_per_block];
            __shared__ block_t sh_g[has_fusion ? warp_size : 1][c_rows_per_block];

            for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                const int kby = kbx * (qk/QK8_1);
                const int kqs = vdr * (threadIdx.x % (qi/vdr));

                // Warp 0 stages the weight tile that every token warp needs for
                // this thread's kbx. Other warps wait, then reuse it.
                if (threadIdx.y == 0) {
#pragma unroll
                    for (int i = 0; i < c_rows_per_block; ++i) {
                        const block_t * src_w =
                                (const block_t *) vx + kbx_offset + i*stride_row_x + kbx;
                        // memcpy: block types with ggml_half have deleted copy assignment.
                        memcpy(&sh_w[threadIdx.x][i], src_w, sizeof(block_t));
                        if constexpr (has_fusion) {
                            const block_t * src_g =
                                    (const block_t *) vgate + kbx_offset + i*stride_row_x + kbx;
                            memcpy(&sh_g[threadIdx.x][i], src_g, sizeof(block_t));
                        }
                    }
                }
                __syncthreads();

#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    // kbx=0: vbq already points at the staged block for this thread.
                    tmp[i] += vec_dot_q_cuda(&sh_w[threadIdx.x][i], &y[kby], 0, kqs);
                    if constexpr (has_fusion) {
                        tmp_gate[i] += vec_dot_q_cuda(&sh_g[threadIdx.x][i], &y[kby], 0, kqs);
                    }
                }
                __syncthreads();
            }
        } else {
            for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                const int kby = kbx * (qk/QK8_1);
                const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    } else {
        for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
            for (int i = 0; i < c_rows_per_block; ++i) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const float gate = tmp_gate[threadIdx.x];
            switch (fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    result *= ggml_cuda_op_silu_single(gate);
                    break;
                case GGML_GLU_OP_GEGLU:
                    result *= ggml_cuda_op_gelu_single(gate);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    result = ggml_cuda_op_swiglu_oai_single(gate, result);
                    break;
                default:
                    result *= gate;
                    break;
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, int rows_per_block_override = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, const int32_t * compact_ch, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, rows_per_block_override>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact_ch);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, rows_per_block_override>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact_ch);
}

template <ggml_type type, bool has_fusion, int rows_per_block>
static void mul_mat_vec_q_moe_launch_rows(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_launch, const int share_weights,
        const int32_t * compact_ch, const int32_t * compact_tok, cudaStream_t stream) {

    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    // Pair-compact: one token per block (block.y == 1). Default: one warp per token.
    const int block_tokens = compact_tok ? 1 : (int) ncols_dst;
    const dim3 block_nums(nblocks_rows, nchannels_launch);
    const dim3 block_dims(warp_size, block_tokens);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, has_fusion>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride, share_weights, compact_ch, compact_tok);
}

template <ggml_type type, bool has_fusion>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, const mmvq_compact_args compact, cudaStream_t stream) {

    const int rows_override = has_fusion ? get_moe_fused_rows_override() : get_moe_plain_rows_override();
    const int share_weights = get_moe_share_weights() ? 1 : 0;
    if (rows_override != 0 && rows_override != 2) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::fprintf(stderr,
                    "ggml_cuda: %s multi-token MoE MMVQ rows/block override active: %d (default 2)\n",
                    has_fusion ? "fused" : "plain", rows_override);
        }
    }
    if (share_weights) {
        static std::atomic<bool> logged_share{false};
        if (!logged_share.exchange(true)) {
            std::fprintf(stderr,
                    "ggml_cuda: multi-token MoE MMVQ bit-identical weight staging active "
                    "(GGML_CUDA_MMVQ_MOE_SHARE_WEIGHTS=1)\n");
        }
    }

    const int nchannels_launch = compact.ch ? compact.n_launch : nchannels_dst;
    if (nchannels_launch <= 0) {
        return;
    }

    switch (rows_override) {
        case 1:
            mul_mat_vec_q_moe_launch_rows<type, has_fusion, 1>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_launch, share_weights,
                compact.ch, compact.tok, stream);
            break;
        case 4:
            mul_mat_vec_q_moe_launch_rows<type, has_fusion, 4>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_launch, share_weights,
                compact.ch, compact.tok, stream);
            break;
        case 0:
        case 2:
        default:
            mul_mat_vec_q_moe_launch_rows<type, has_fusion, 2>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_launch, share_weights,
                compact.ch, compact.tok, stream);
            break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const mmvq_compact_args compact, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    // When compact remaps blockIdx.y, grid.y is the packed active count.
    const int nchannels_launch = compact.ch ? compact.n_launch : nchannels_dst;
    if (nchannels_launch <= 0) {
        return;
    }

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
        constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
        constexpr int vdr                   = get_vdr_mmvq(type);
        const int     blocks_per_row_x      = ncols_x / qk;
        const int     blocks_per_iter_1warp = vdr * warp_size / qi;
        const int     nwarps                = calc_nwarps(type, c_ncols_dst, table_id);
        bool          use                   = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    if (has_ids && ncols_dst > 1) {
        const bool has_fusion = fusion.gate != nullptr;
        if (has_fusion) {
            GGML_ASSERT(fusion.x_bias == nullptr && fusion.gate_bias == nullptr &&
                    fusion.x_scale == nullptr && fusion.gate_scale == nullptr);
            mul_mat_vec_q_moe_launch<type, true>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_dst, compact, stream);
        } else {
            mul_mat_vec_q_moe_launch<type, false>(
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_dst, compact, stream);
        }
        return;
    }

    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;

            bool use_small_k = should_use_small_k(c_ncols_dst);

            if constexpr (type == GGML_TYPE_Q6_K) {
                const int rows_override = !ids && !has_fusion && !use_small_k ? get_q6_k_ncols1_rows_override() : 0;
                if (rows_override == 2 || rows_override == 4) {
                    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
                    const dim3 block_nums((nrows_x + rows_override - 1)/rows_override,
                            nchannels_launch, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps, 1);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        std::fprintf(stderr,
                                "ggml_cuda: Q6_K ncols=1 MMVQ rows/block override active: %d (default 1)\n",
                                rows_override);
                    }
                    if (rows_override == 2) {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 2>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    } else {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 4>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    }
                    break;
                }
            }

            if constexpr (type == GGML_TYPE_Q8_0) {
                // Default rows/block for ncols=1 is 1; allow 2 or 4 like Q6_K.
                const int rows_override = !ids && !has_fusion && !use_small_k ? get_q8_0_ncols1_rows_override() : 0;
                if (rows_override == 2 || rows_override == 4) {
                    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
                    const dim3 block_nums((nrows_x + rows_override - 1)/rows_override,
                            nchannels_launch, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps, 1);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        std::fprintf(stderr,
                                "ggml_cuda: Q8_0 ncols=1 MMVQ rows/block override active: %d (default 1)\n",
                                rows_override);
                    }
                    if (rows_override == 2) {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 2>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    } else {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 4>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    }
                    break;
                }
            }

            if (use_small_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch,
                                                                        nsamples_dst, warp_size, table_id, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    compact.ch, stream);
            } else {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch,
                                                                        nsamples_dst, warp_size, table_id);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    compact.ch, stream);
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            if constexpr (type == GGML_TYPE_Q8_0) {
                // Default rows/block for ncols=2 is 2; allow 1 or 4.
                const int rows_override = !ids ? get_q8_0_ncols2_rows_override() : 0;
                if (rows_override == 1 || rows_override == 4) {
                    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
                    const dim3 block_nums((nrows_x + rows_override - 1)/rows_override,
                            nchannels_launch, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps, 1);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        std::fprintf(stderr,
                                "ggml_cuda: Q8_0 ncols=2 MMVQ rows/block override active: %d (default 2)\n",
                                rows_override);
                    }
                    if (rows_override == 1) {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 1>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    } else {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 4>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    }
                    break;
                }
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            if constexpr (type == GGML_TYPE_Q6_K) {
                const int rows_override = !ids && !has_fusion ? get_q6_k_ncols3_rows_override() : 0;
                if (rows_override == 1 || rows_override == 4) {
                    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
                    const dim3 block_nums((nrows_x + rows_override - 1)/rows_override,
                            nchannels_launch, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps, 1);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        std::fprintf(stderr,
                                "ggml_cuda: Q6_K ncols=3 MMVQ rows/block override active: %d (default 2)\n",
                                rows_override);
                    }
                    if (rows_override == 1) {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 1>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    } else {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 4>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    }
                    break;
                }
            }
            if constexpr (type == GGML_TYPE_Q8_0) {
                const int rows_override = !ids ? get_q8_0_ncols3_rows_override() : 0;
                if (rows_override == 1 || rows_override == 4) {
                    const int nwarps = calc_nwarps(type, c_ncols_dst, table_id);
                    const dim3 block_nums((nrows_x + rows_override - 1)/rows_override,
                            nchannels_launch, nsamples_dst);
                    const dim3 block_dims(warp_size, nwarps, 1);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        std::fprintf(stderr,
                                "ggml_cuda: Q8_0 ncols=3 MMVQ rows/block override active: %d (default 2)\n",
                                rows_override);
                    }
                    if (rows_override == 1) {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 1>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    } else {
                        mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, 4>(
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd,
                            stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                            block_nums, block_dims, 0, ids_stride, compact.ch, stream);
                    }
                    break;
                }
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_launch, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, compact.ch, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const mmvq_compact_args compact, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, compact, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    // Zero-init would leave skip_slot=0 (a valid expert slot). Disable until set.
    fusion_local.skip_slot = -1;

    if (fusion) {
        GGML_ASSERT( !ids || (dst->ne[2] >= 1 && dst->ne[2] <= 4));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (ids && dst->ne[2] > 1) {
            GGML_ASSERT(fusion->gate != nullptr && fusion->x_bias == nullptr && fusion->gate_bias == nullptr &&
                    fusion->x_scale == nullptr && fusion->gate_scale == nullptr);
        }
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.skip_slot = fusion->skip_slot;
    }

    // Non-fused MUL_MAT_ID carries the skip slot on the destination op_params.
    // Fused GLU destinations pass it through fusion->skip_slot above.
    if (ids && fusion_local.skip_slot < 0) {
        fusion_local.skip_slot = ggml_cuda_mul_mat_id_skip_slot(dst);
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mmvq_compact_args compact{};
    // Compact hot-only: pack non-sentinel expert channels / (channel,token)
    // pairs and launch only those. Requires skip_slot from expert tiering.
    // Variable grid is not CUDA-graph capture-safe; fall back while capturing.
    const bool want_compact = get_mmvq_compact_skip() != 0 &&
            ids_d && fusion_local.skip_slot >= 0 &&
            !mmvq_stream_is_capturing(stream);

    // CUDA VMM pool is LIFO: construct/alloc in order n_active -> ch -> tok so
    // destructors free tok -> ch -> n_active -> src1_q8_1.
    ggml_cuda_pool_alloc<int>     n_active_alloc(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> compact_ch_alloc(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> compact_tok_alloc(ctx.pool());

    if (want_compact) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            std::fprintf(stderr,
                    "ggml_cuda: MMVQ compact-skip active (GGML_CUDA_MMVQ_COMPACT_SKIP=1); "
                    "hot-only packed launches for expert-tier sentinel slots\n");
        }

        // Pre-zero the full destination so skipped (cold) channels stay exact 0
        // without per-block zero stores in the main kernels.
        CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

        n_active_alloc.alloc(1);
        CUDA_CHECK(cudaMemsetAsync(n_active_alloc.get(), 0, sizeof(int), stream));

        if (ncols_dst == 1) {
            compact_ch_alloc.alloc((size_t) nchannels_dst);
            const int threads = 256;
            const int blocks  = (int) ((nchannels_dst + threads - 1) / threads);
            mmvq_compact_channels_kernel<<<blocks, threads, 0, stream>>>(
                    ids_d, compact_ch_alloc.get(), n_active_alloc.get(),
                    (int) nchannels_dst, fusion_local.skip_slot);
        } else {
            const int n_work = (int) (nchannels_dst * ncols_dst);
            compact_ch_alloc.alloc((size_t) n_work);
            compact_tok_alloc.alloc((size_t) n_work);
            const int threads = 256;
            const int blocks  = (n_work + threads - 1) / threads;
            mmvq_compact_pairs_kernel<<<blocks, threads, 0, stream>>>(
                    ids_d, compact_ch_alloc.get(), compact_tok_alloc.get(), n_active_alloc.get(),
                    (int) nchannels_dst, (int) ncols_dst, (int) ids_stride, fusion_local.skip_slot);
        }
        CUDA_CHECK(cudaGetLastError());

        // One int D2H: shrink grid.y to the packed active count. This is the
        // structural win over skip-sentinel's full-grid early-exit.
        int n_active_host = 0;
        CUDA_CHECK(cudaMemcpyAsync(&n_active_host, n_active_alloc.get(), sizeof(int),
                cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (n_active_host <= 0) {
            // All channels were sentinel; dst already zeroed. Destructors free
            // pool buffers in LIFO order before returning.
            return;
        }

        compact.ch       = compact_ch_alloc.get();
        compact.tok      = ncols_dst > 1 ? compact_tok_alloc.get() : nullptr;
        compact.n_launch = n_active_host;
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, compact, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    fusion_local.skip_slot = -1;
    const mmvq_compact_args compact{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, compact, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
