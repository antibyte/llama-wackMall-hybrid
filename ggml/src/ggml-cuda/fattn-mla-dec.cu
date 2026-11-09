#include "fattn-mla-dec.cuh"

#include <cfloat>
#include <cstdlib>

// Decode attention for MLA models (DKQ 576, DV 512, V = the first 512 values of each K row) on GPUs
// without tensor cores. All query heads of a KV head read the same K rows; the tile kernel moves K
// and the scores through shared memory and runs few blocks for short contexts. Here each warp
// computes 4 heads of one token: the warps of a block read the same K rows (the 2nd and later
// through L1), the V product reuses the K registers, and the KV range is split over one wave of
// blocks. Products are summed in half2 over chains of 9 per lane (like the tile kernel's half
// products), then in float across lanes; V is summed in half2 for 32 rows, then in float.
// flash_attn_mla_dec_combine merges the partial results, stored as normalized half.

static constexpr int mla_dkq2       = 576/2;               // half2 per K row
static constexpr int mla_dv2        = 512/2;               // half2 per V row
static constexpr int mla_nk         = mla_dkq2/WARP_SIZE;  // half2 per lane and K row
static constexpr int mla_nv         = mla_dv2/WARP_SIZE;   // the first 8 of those are the V part
static constexpr int mla_hpw        = 4;                   // query heads per warp
static constexpr int mla_rows       = 8;                   // KV rows per step
static constexpr int mla_nvals      = mla_rows*mla_hpw;    // scores per step and warp, one per lane
static constexpr int mla_flush_rows = 32;                  // rows between flushing the half2 V sums into float
static constexpr int mla_max_warps  = 8;
static constexpr int mla_sh_warp    = mla_hpw*mla_nk*WARP_SIZE + 2*mla_nvals; // Q and 2 P buffers, half2

static constexpr int mla_combine_threads = 64;
static constexpr int mla_combine_groups  = 4;

static_assert(mla_nv == 8 && mla_nk == 9, "slot layout below");
static_assert(mla_nvals == WARP_SIZE, "the transposed reduction leaves one score per lane");

static int ggml_cuda_fa_mla_dec_env(const char * name, const int def) {
    const char * env = getenv(name);
    return env == nullptr || env[0] == '\0' ? def : atoi(env);
}

// 1 = MLA decode kernel for F16 K/V, 0 = tile kernel as before.
static bool ggml_cuda_fa_mla_dec_enabled() {
    static const bool value = ggml_cuda_fa_mla_dec_env("GGML_CUDA_FA_MLA_DEC", 0) != 0;
    return value;
}

// Largest number of query tokens (decode plus draft) that takes the MLA decode kernel. On a GTX
// 1660 Ti it beats the tile kernel up to at least 32 (1.4x at 32, 2.5x at 1).
static int ggml_cuda_fa_mla_dec_max_tokens() {
    static const int value = ggml_cuda_fa_mla_dec_env("GGML_CUDA_FA_MLA_DEC_MAX_TOKENS", 32);
    return value;
}

// Fewest KV rows per block: short contexts use fewer blocks instead of tiny ones.
static int ggml_cuda_fa_mla_dec_min_rows() {
    static const int value = std::max(mla_rows, ggml_cuda_fa_mla_dec_env("GGML_CUDA_FA_MLA_DEC_MIN_ROWS", 32));
    return value;
}

// K/V slot i of a lane: 4 consecutive half2 per 16-byte load for the V part, then one of the rope part.
static __device__ __forceinline__ int mla_dec_j(const int i, const int lane) {
    return i < mla_nv ? (i/4)*(4*WARP_SIZE) + 4*lane + i%4 : mla_dv2 + lane;
}

// Index of slot i of head h in the per-warp Q buffer, same 16-byte grouping.
static __device__ __forceinline__ int mla_dec_q_idx(const int h, const int i, const int lane) {
    return i < mla_nv ? ((h*(mla_nv/4) + i/4)*WARP_SIZE + lane)*4 + i%4 : mla_hpw*mla_nv*WARP_SIZE + h*WARP_SIZE + lane;
}

// One step of the transposed reduction: lanes with bit off set keep values n..2n-1 of the other
// half of the warp, the others keep 0..n-1, both now at v[0..n-1]. Constant indices only: a select
// on array elements in a loop the compiler does not unroll becomes an indexed stack access.
template <int off, int n>
static __device__ __forceinline__ void mla_dec_butterfly(float * v, const int lane) {
    const bool upper = lane & off;
#pragma unroll
    for (int k = 0; k < n; ++k) {
        const float lo   = v[k];
        const float hi   = v[k + n];
        const float send = upper ? lo : hi;
        const float keep = upper ? hi : lo;
        v[k] = keep + __shfl_xor_sync(0xFFFFFFFF, send, off, WARP_SIZE);
    }
}

// grid: x = KV part, y = head blocks x sequences, z = token groups; block: 32 x (hg_per_block x tokens).
__launch_bounds__(mla_max_warps*WARP_SIZE, 1)
static __global__ void flash_attn_mla_dec(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        half2      * __restrict__ part,
        float2     * __restrict__ part_meta,
        const float q_scale,
        const int hg_per_block,
        const int rows_per_block,
        const int ne01, const int ne02, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int ne11, const int gqa_ratio, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int ne33, const int64_t nb31, const int64_t nb33) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tok  = blockIdx.z*(blockDim.y/hg_per_block) + warp/hg_per_block;
    if (tok >= ne01) {
        return;
    }

    const int heads_per_block = hg_per_block*mla_hpw;
    const int nhb   = ne02/heads_per_block;
    const int seq   = blockIdx.y/nhb;
    const int head0 = (blockIdx.y - seq*nhb)*heads_per_block + (warp % hg_per_block)*mla_hpw;

    const half2 * K_h2      = (const half2 *) (K + seq*nb13 + (head0/gqa_ratio)*nb12);
    const int64_t stride_K2 = nb11/sizeof(half2);
    const half  * mask_row  = (const half *) (mask + (seq % ne33)*nb33 + tok*nb31);

    // Q in shared memory leaves the registers to the K rows of a step. q_scale holds the softmax
    // scale, log2(e) and 1/4 against half overflow in the chains.
    extern __shared__ __align__(16) half2 mla_sh[]; // per warp: mla_sh_warp half2
    half2 * Q_sh = mla_sh + warp*mla_sh_warp;      // [h][i/4][lane][4] for i < 8, then [h][lane] for i = 8
    half2 * P_sh = Q_sh + mla_hpw*mla_nk*WARP_SIZE; // [2][rows][hpw]
#pragma unroll
    for (int h = 0; h < mla_hpw; ++h) {
        const float * Qf = (const float *) (Q + seq*nb03 + (head0 + h)*nb02 + tok*nb01);
#pragma unroll
        for (int i0 = 0; i0 < mla_nv; i0 += 4) {
            float q[8];
            ggml_cuda_memcpy_1<sizeof(q)/2>(q + 0, Qf + 2*mla_dec_j(i0, lane) + 0);
            ggml_cuda_memcpy_1<sizeof(q)/2>(q + 4, Qf + 2*mla_dec_j(i0, lane) + 4);
            half2 tmp[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                tmp[c] = __floats2half2_rn(q[2*c + 0]*q_scale, q[2*c + 1]*q_scale);
            }
            ggml_cuda_memcpy_1<sizeof(tmp)>(Q_sh + mla_dec_q_idx(h, i0, lane), tmp);
        }
        const float2 q = *((const float2 *) (Qf + 2*mla_dec_j(mla_nv, lane)));
        Q_sh[mla_dec_q_idx(h, mla_nv, lane)] = __floats2half2_rn(q.x*q_scale, q.y*q_scale);
    }
    __syncwarp();

    // After the reduction lane v holds the score of row v/4 and head v%4 of the step; lanes
    // xor 4, 8, 16 hold the other rows of the same head, lane h holds head h.
    const int r_own = lane / mla_hpw;

    float m_own = -FLT_MAX/2.0f; // running max of head lane%4, log2 units
    float s_own = 0.0f;          // running sum of the probabilities of that head in row r_own

    half2  VKQ_h[mla_hpw][mla_nv];
    float2 VKQ_f[mla_hpw][mla_nv];
#pragma unroll
    for (int h = 0; h < mla_hpw; ++h) {
#pragma unroll
        for (int i = 0; i < mla_nv; ++i) {
            VKQ_h[h][i] = make_half2(0.0f, 0.0f);
            VKQ_f[h][i] = make_float2(0.0f, 0.0f);
        }
    }

    const int k_begin = blockIdx.x*rows_per_block;
    const int k_end   = min(k_begin + rows_per_block, ne11);

    for (int k0 = k_begin, step = 1; k0 < k_end; k0 += mla_rows, ++step) {
        half2 Kr[mla_rows][mla_nk];
#pragma unroll
        for (int r = 0; r < mla_rows; ++r) {
            const half2 * row = K_h2 + (k0 + r)*stride_K2;
            ggml_cuda_memcpy_1<4*sizeof(half2)>(&Kr[r][0], row + 4*WARP_SIZE*0 + 4*lane);
            ggml_cuda_memcpy_1<4*sizeof(half2)>(&Kr[r][4], row + 4*WARP_SIZE*1 + 4*lane);
            Kr[r][8] = row[mla_dv2 + lane];
        }
        const float mk = __half2float(mask_row[k0 + r_own]);

        half2 acc[mla_rows][mla_hpw];
#pragma unroll
        for (int i0 = 0; i0 < mla_nv; i0 += 4) {
#pragma unroll
            for (int h = 0; h < mla_hpw; ++h) {
                half2 q[4];
                ggml_cuda_memcpy_1<sizeof(q)>(q, Q_sh + mla_dec_q_idx(h, i0, lane));
#pragma unroll
                for (int c = 0; c < 4; ++c) {
#pragma unroll
                    for (int r = 0; r < mla_rows; ++r) {
                        acc[r][h] = i0 + c == 0 ? __hmul2(q[c], Kr[r][i0 + c]) : __hfma2(q[c], Kr[r][i0 + c], acc[r][h]);
                    }
                }
            }
        }
#pragma unroll
        for (int h = 0; h < mla_hpw; ++h) {
            const half2 q = Q_sh[mla_dec_q_idx(h, mla_nv, lane)];
#pragma unroll
            for (int r = 0; r < mla_rows; ++r) {
                acc[r][h] = __hfma2(q, Kr[r][mla_nv], acc[r][h]);
            }
        }
        float v[mla_nvals];
#pragma unroll
        for (int r = 0; r < mla_rows; ++r) {
#pragma unroll
            for (int h = 0; h < mla_hpw; ++h) {
                const float2 f = __half22float2(acc[r][h]);
                v[r*mla_hpw + h] = f.x + f.y;
            }
        }

        // Transposed butterfly: each step halves the values a lane keeps.
        mla_dec_butterfly<16, 16>(v, lane);
        mla_dec_butterfly< 8,  8>(v, lane);
        mla_dec_butterfly< 4,  4>(v, lane);
        mla_dec_butterfly< 2,  2>(v, lane);
        mla_dec_butterfly< 1,  1>(v, lane);
        const float s = 4.0f*v[0] + 1.4426950408889634f*mk;

        float m_new = s;
#pragma unroll
        for (int off = mla_hpw; off < WARP_SIZE; off *= 2) {
            m_new = fmaxf(m_new, __shfl_xor_sync(0xFFFFFFFF, m_new, off, WARP_SIZE));
        }
        m_new = fmaxf(m_new, m_own);
        if (__any_sync(0xFFFFFFFF, m_new > m_own)) {
            const float f = exp2f(m_own - m_new);
            m_own  = m_new;
            s_own *= f;
#pragma unroll
            for (int h = 0; h < mla_hpw; ++h) {
                const float fh  = __shfl_sync(0xFFFFFFFF, f, h, WARP_SIZE);
                const half2 fh2 = __float2half2_rn(fh);
#pragma unroll
                for (int i = 0; i < mla_nv; ++i) {
                    VKQ_h[h][i]    = __hmul2(VKQ_h[h][i], fh2);
                    VKQ_f[h][i].x *= fh;
                    VKQ_f[h][i].y *= fh;
                }
            }
        }
        const float p = exp2f(s - m_own);
        s_own += p;

        // Probabilities to all lanes; two buffers so that the next step's write needs no 2nd sync.
        half2 * P_buf = P_sh + (step & 1)*mla_nvals;
        P_buf[lane] = __float2half2_rn(p);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < mla_rows; ++r) {
            half2 P4[mla_hpw];
            ggml_cuda_memcpy_1<sizeof(P4)>(P4, P_buf + r*mla_hpw);
#pragma unroll
            for (int h = 0; h < mla_hpw; ++h) {
#pragma unroll
                for (int i = 0; i < mla_nv; ++i) {
                    VKQ_h[h][i] = __hfma2(P4[h], Kr[r][i], VKQ_h[h][i]);
                }
            }
        }

        if (step % (mla_flush_rows/mla_rows) == 0 || k0 + mla_rows >= k_end) {
#pragma unroll
            for (int h = 0; h < mla_hpw; ++h) {
#pragma unroll
                for (int i = 0; i < mla_nv; ++i) {
                    const float2 f = __half22float2(VKQ_h[h][i]);
                    VKQ_f[h][i].x += f.x;
                    VKQ_f[h][i].y += f.y;
                    VKQ_h[h][i] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    // Sum over the rows of a step; lane h then holds head h.
    float s_tot = s_own;
#pragma unroll
    for (int off = mla_hpw; off < WARP_SIZE; off *= 2) {
        s_tot += __shfl_xor_sync(0xFFFFFFFF, s_tot, off, WARP_SIZE);
    }

    const int64_t row0 = (int64_t(seq)*ne01 + tok)*ne02 + head0; // dst row of head0, layout [DV, ne02, ne01, ne03]

    if (gridDim.x == 1) {
#pragma unroll
        for (int h = 0; h < mla_hpw; ++h) {
            const float S   = __shfl_sync(0xFFFFFFFF, s_tot, h, WARP_SIZE);
            const float inv = S > 0.0f ? 1.0f/S : 0.0f;
            float2 * d = (float2 *) (dst + (row0 + h)*(2*mla_dv2));
#pragma unroll
            for (int i = 0; i < mla_nv; ++i) {
                d[mla_dec_j(i, lane)] = make_float2(VKQ_f[h][i].x*inv, VKQ_f[h][i].y*inv);
            }
        }
        return;
    }

#pragma unroll
    for (int h = 0; h < mla_hpw; ++h) {
        const float S   = __shfl_sync(0xFFFFFFFF, s_tot, h, WARP_SIZE);
        const float inv = S > 0.0f ? 1.0f/S : 0.0f;
        half2 * pp = part + ((row0 + h)*gridDim.x + blockIdx.x)*mla_dv2;
#pragma unroll
        for (int i0 = 0; i0 < mla_nv; i0 += 4) {
            half2 tmp[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                tmp[c] = __floats2half2_rn(VKQ_f[h][i0 + c].x*inv, VKQ_f[h][i0 + c].y*inv);
            }
            ggml_cuda_memcpy_1<sizeof(tmp)>(pp + mla_dec_j(i0, lane), tmp);
        }
    }
    if (lane < mla_hpw) {
        part_meta[(row0 + lane)*gridDim.x + blockIdx.x] = make_float2(m_own, s_tot);
    }
}

// One dst row (head of a token) per blockIdx.x, mla_combine_threads half2 columns per blockIdx.y,
// the parts split over threadIdx.y so that the loads of a thread are few and independent.
static __global__ void flash_attn_mla_dec_combine(
        const half2  * __restrict__ part,
        const float2 * __restrict__ part_meta,
        float        * __restrict__ dst,
        const int nparts) {
    constexpr int nthreads = mla_combine_groups*mla_combine_threads;
    constexpr int nw       = nthreads/WARP_SIZE;

    const int64_t row  = blockIdx.x;
    const int     j    = blockIdx.y*mla_combine_threads + threadIdx.x;
    const int     tid  = threadIdx.y*mla_combine_threads + threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int     wid  = tid / WARP_SIZE;

    part      += row*nparts*mla_dv2;
    part_meta += row*nparts;

    extern __shared__ float w_sh[]; // nparts weights
    __shared__ float2 red[mla_combine_groups][mla_combine_threads];
    __shared__ float  red_w[nw];

    // Rounds of 16 independent loads, predicated: an unroll pragma on a plain loop leaves typical
    // counts (12 parts per thread) to the serial remainder loop. The first round is issued before
    // the weights are known.
    auto load_round = [&](half2 (&x)[16], const int p0) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int p = p0 + k*mla_combine_groups;
            x[k] = p < nparts ? part[p*mla_dv2 + j] : make_half2(0.0f, 0.0f);
        }
    };
    half2 x0[16];
    load_round(x0, threadIdx.y);

    // Block-wide max and sum of the part weights.
    const float2 meta0 = tid < nparts ? part_meta[tid] : make_float2(-FLT_MAX/2.0f, 0.0f);
    float m = meta0.x;
    for (int p = tid + nthreads; p < nparts; p += nthreads) {
        m = fmaxf(m, part_meta[p].x);
    }
    m = warp_reduce_max(m);
    if (lane == 0) {
        red_w[wid] = m;
    }
    __syncthreads();
    m = red_w[0];
#pragma unroll
    for (int w = 1; w < nw; ++w) {
        m = fmaxf(m, red_w[w]);
    }
    float den = 0.0f;
    if (tid < nparts) {
        den = exp2f(meta0.x - m)*meta0.y;
        w_sh[tid] = den;
    }
    for (int p = tid + nthreads; p < nparts; p += nthreads) {
        const float2 meta = part_meta[p];
        const float  w    = exp2f(meta.x - m)*meta.y;
        w_sh[p] = w;
        den += w;
    }
    den = warp_reduce_sum(den);
    __syncthreads(); // the max reads of red_w are done
    if (lane == 0) {
        red_w[wid] = den;
    }
    __syncthreads();

    float2 num = make_float2(0.0f, 0.0f);
    for (int p0 = threadIdx.y; p0 < nparts; p0 += 16*mla_combine_groups) {
        half2 x[16];
        if (p0 == (int) threadIdx.y) {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                x[k] = x0[k];
            }
        } else {
            load_round(x, p0);
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int    p  = p0 + k*mla_combine_groups;
            const float  w  = p < nparts ? w_sh[p] : 0.0f;
            const float2 xf = __half22float2(x[k]);
            num.x += w*xf.x;
            num.y += w*xf.y;
        }
    }
    red[threadIdx.y][threadIdx.x] = num;
    __syncthreads();
    if (threadIdx.y != 0) {
        return;
    }
    den = 0.0f;
#pragma unroll
    for (int w = 0; w < nw; ++w) {
        den += red_w[w];
    }
#pragma unroll
    for (int g = 1; g < mla_combine_groups; ++g) {
        num.x += red[g][threadIdx.x].x;
        num.y += red[g][threadIdx.x].y;
    }
    const float inv = den > 0.0f ? 1.0f/den : 0.0f;
    ((float2 *) (dst + row*(2*mla_dv2)))[j] = make_float2(num.x*inv, num.y*inv);
}

bool ggml_cuda_flash_attn_ext_mla_dec_applies(const ggml_tensor * dst) {
    if (!ggml_cuda_fa_mla_dec_enabled()) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (K->ne[0] != 2*mla_dkq2 || V->ne[0] != 2*mla_dv2 || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || sinks) {
        return false;
    }
    // V must be the first DV values of the K rows, the kernel reads them from the K registers.
    if (V->data != K->data || V->nb[1] != K->nb[1] || V->nb[2] != K->nb[2] || V->nb[3] != K->nb[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (Q->ne[1] > ggml_cuda_fa_mla_dec_max_tokens() || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2]/K->ne[2]) % mla_hpw != 0) {
        return false;
    }
    if (K->ne[1] % mla_rows != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    if ((uintptr_t) Q->data % 16 != 0 || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || Q->nb[3] % 16 != 0) {
        return false;
    }
    if ((uintptr_t) K->data % 16 != 0 || K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || K->nb[3] % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_mla_dec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();
    const int    id     = ggml_cuda_get_device();
    const int    nsm    = ggml_cuda_info().devices[id].nsm;

    const int ntok    = Q->ne[1];
    const int gqa     = Q->ne[2]/K->ne[2];
    const int ngroups = gqa/mla_hpw; // warps per token and KV head

    // Two tokens per block share the K reads; more go to further blocks.
    const int tpb = ntok >= 2 && ngroups <= mla_max_warps/2 ? 2 : 1;
    int hg = 1;
    while (2*hg*tpb <= mla_max_warps && ngroups % (2*hg) == 0) {
        hg *= 2;
    }
    const int nwarps = hg*tpb;
    const int ntg    = (ntok + tpb - 1)/tpb;

    const size_t nbytes_shared = nwarps*mla_sh_warp*sizeof(half2);

    static int occupancy[GGML_CUDA_MAX_DEVICES][mla_max_warps + 1] = {{0}};
    int & max_blocks_per_sm = occupancy[id][nwarps];
    if (max_blocks_per_sm == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, flash_attn_mla_dec, nwarps*WARP_SIZE, nbytes_shared));
        GGML_ASSERT(max_blocks_per_sm > 0);
    }

    // One wave of blocks over the KV range.
    const int ne11  = K->ne[1];
    const int nhbs  = (Q->ne[2]/(hg*mla_hpw))*Q->ne[3];
    int nparts = std::max(1, nsm*max_blocks_per_sm/(nhbs*ntg));
    nparts = std::min(nparts, std::max(1, ne11/ggml_cuda_fa_mla_dec_min_rows()));
    int rows_per_block = (ne11 + nparts - 1)/nparts;
    rows_per_block = (rows_per_block + mla_rows - 1)/mla_rows*mla_rows;
    nparts = (ne11 + rows_per_block - 1)/rows_per_block;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const float q_scale = scale*1.4426950408889634f*0.25f;

    const int64_t nrows = ggml_nrows(dst);
    ggml_cuda_pool_alloc<half2>  part(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_meta(ctx.pool());
    if (nparts > 1) {
        part.alloc(nrows*nparts*mla_dv2);
        part_meta.alloc(nrows*nparts);
    }

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    const dim3 grid_dim(nparts, nhbs, ntg);
    flash_attn_mla_dec<<<grid_dim, block_dim, nbytes_shared, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) mask->data,
        (float *) dst->data, part.ptr, part_meta.ptr,
        q_scale, hg, rows_per_block,
        ntok, Q->ne[2], Q->nb[1], Q->nb[2], Q->nb[3],
        ne11, gqa, K->nb[1], K->nb[2], K->nb[3],
        mask->ne[3], mask->nb[1], mask->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    if (nparts > 1) {
        const dim3 grid_combine(nrows, mla_dv2/mla_combine_threads, 1);
        const dim3 block_combine(mla_combine_threads, mla_combine_groups, 1);
        flash_attn_mla_dec_combine<<<grid_combine, block_combine, nparts*sizeof(float), stream>>>(
            part.ptr, part_meta.ptr, (float *) dst->data, nparts);
        CUDA_CHECK(cudaGetLastError());
    }
}
