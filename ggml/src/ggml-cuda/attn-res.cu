#include "attn-res.cuh"
#include "graph-fusion.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <type_traits>
#include <cstdlib>
#include <vector>

#define ATTN_RES_BLOCK          512
#define ATTN_RES_MAX_PER_THREAD 8
#define ATTN_RES_MAX_STATES     64
#define ATTN_RES_MAX_Q          4
#define ATTN_RES_MAX_SLICES     8 // slices of E per logit

#define LOOP_GATE_DOWN_THREADS 256 // one block per down row
#define LOOP_GATE_OUT_WARPS    8   // warps per block of the output kernel

static bool attn_res_env_flag(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && atoi(v) != 0;
}

static bool attn_res_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE;
}

// root of a view chain is a weight: constant for this graph
static bool attn_res_is_weight_view(const ggml_tensor * t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t->op == GGML_OP_NONE && t->buffer != nullptr &&
           ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static bool attn_res_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * pa = (const char *) a->data;
    const char * pb = (const char *) b->data;
    return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
}

// nodes the graph fusion plan already handles out of order. Mark 2 on the output (an add folded into the rms_norm
// behind it) is fine: the norm skips the add once a fusion has run the node (ran_idx).
static bool attn_res_planned(const ggml_backend_cuda_context & ctx, const std::vector<int> & idx) {
    const ggml_cuda_graph_fusion * gfu = ctx.graph_fusion;
    if (gfu == nullptr || gfu->skip_idx.empty()) {
        return false;
    }
    for (size_t k = 0; k < idx.size(); ++k) {
        const char mark = gfu->skip_idx[idx[k]];
        if (mark == 1 || (mark != 0 && k + 1 != idx.size())) {
            return true;
        }
    }
    return false;
}

static void attn_res_trace(const char * what, const ggml_cgraph * cgraph, int i, const char * why) {
    static const bool trace = attn_res_env_flag("GGML_CUDA_ATTN_RES_TRACE");
    static std::atomic<int> n_logged[2] = {{0}, {0}};
    if (!trace || n_logged[what[0] == 'a' ? 0 : 1].fetch_add(1) >= 16) {
        return;
    }
    fprintf(stderr, "ggml_cuda: %s fusion at node %d (%s): %s; next:", what, i, cgraph->nodes[i]->name, why);
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 14; ++j) {
        fprintf(stderr, " %s", ggml_op_desc(cgraph->nodes[j]));
    }
    fprintf(stderr, "\n");
}

// s = [blocks; rms_norm(cur)], out[:, q] = sum_b softmax_b(scale*<s_b, query_q>) s_b. Blocks (t, y): every block of
// token t computes the logits, block y writes channels [y*chunk, (y + 1)*chunk). vec4: E % 4 == 0, 16-byte rows.
template <bool vec4>
static __global__ void __launch_bounds__(ATTN_RES_BLOCK) attn_res_mix_f32(
        const float * __restrict__ blocks, const int64_t blocks_s1, const int64_t blocks_s2,
        const float * __restrict__ cur, const int64_t cur_s,
        const float * __restrict__ query, const int64_t query_s1, const int64_t query_s2,
        float * dst, const int64_t dst_s1, const int64_t dst_s2,
        const int E, const int nb, const int nq, const float eps, const float scale, const int chunk) {
    constexpr int n_warps = ATTN_RES_BLOCK / WARP_SIZE;

    const int t    = blockIdx.x;
    const int e0   = blockIdx.y*chunk;
    const int e1   = min(E, e0 + chunk);
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    extern __shared__ float xn[]; // E: the normalized cur
    __shared__ float red[n_warps];
    __shared__ float prob[ATTN_RES_MAX_Q][ATTN_RES_MAX_STATES];
    __shared__ float part[ATTN_RES_MAX_Q*ATTN_RES_MAX_STATES*ATTN_RES_MAX_SLICES];

    ggml_cuda_pdl_sync();

    blocks += t*blocks_s2;
    cur    += t*cur_s;
    query  += t*query_s2;
    dst    += t*dst_s2;

    float sum = 0.0f;
    for (int e = tid; e < E; e += ATTN_RES_BLOCK) {
        const float v = cur[e];
        xn[e] = v;
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        red[warp] = sum;
    }
    __syncthreads();
    sum = warp_reduce_sum(lane < n_warps ? red[lane] : 0.0f);
    const float rs = rsqrtf(sum/E + eps);
    for (int e = tid; e < E; e += ATTN_RES_BLOCK) {
        xn[e] = rs*xn[e];
    }
    __syncthreads();

    // logits: warps take (state, query, slice of E) units; the slices are summed in a fixed order
    const int ns      = nb + 1;
    const int n_pairs = ns*nq;
    const int n_slice = min(ATTN_RES_MAX_SLICES, max(1, (2*n_warps + n_pairs - 1)/n_pairs));
    const int n4      = vec4 ? E/4 : E;
    const int len     = (n4 + n_slice - 1)/n_slice;
    for (int u = warp; u < n_pairs*n_slice; u += n_warps) {
        const int     pq = u / n_slice;
        const int     sl = u % n_slice;
        const int     b  = pq % ns;
        const int     q  = pq / ns;
        const float * sv = b < nb ? blocks + b*blocks_s1 : xn;
        const float * qv = query + q*query_s1;
        const int     i0 = sl*len;
        const int     i1 = min(n4, i0 + len);
        float d = 0.0f;
        if (vec4) {
            const float4 * sv4 = (const float4 *) sv;
            const float4 * qv4 = (const float4 *) qv;
            for (int e4 = i0 + lane; e4 < i1; e4 += WARP_SIZE) {
                const float4 x = sv4[e4];
                const float4 c = qv4[e4];
                d += x.x*c.x + x.y*c.y + x.z*c.z + x.w*c.w;
            }
        } else {
            for (int e = i0 + lane; e < i1; e += WARP_SIZE) {
                d += sv[e]*qv[e];
            }
        }
        d = warp_reduce_sum(d);
        if (lane == 0) {
            part[u] = d;
        }
    }
    __syncthreads();
    if (tid < n_pairs) {
        float d = 0.0f;
        for (int sl = 0; sl < n_slice; ++sl) {
            d += part[tid*n_slice + sl];
        }
        prob[tid / ns][tid % ns] = scale*d;
    }
    __syncthreads();

    if (warp < nq) {
        float m = -INFINITY;
        for (int b = lane; b < ns; b += WARP_SIZE) {
            m = fmaxf(m, prob[warp][b]);
        }
        m = warp_reduce_max(m);
        float s = 0.0f;
        for (int b = lane; b < ns; b += WARP_SIZE) {
            const float v = expf(prob[warp][b] - m);
            prob[warp][b] = v;
            s += v;
        }
        s = warp_reduce_sum(s);
        const float inv = 1.0f/s;
        for (int b = lane; b < ns; b += WARP_SIZE) {
            prob[warp][b] *= inv;
        }
    }
    __syncthreads();

    float acc[ATTN_RES_MAX_PER_THREAD][ATTN_RES_MAX_Q];
#pragma unroll
    for (int k = 0; k < ATTN_RES_MAX_PER_THREAD; ++k) {
        const int e = e0 + tid + k*ATTN_RES_BLOCK;
#pragma unroll
        for (int q = 0; q < ATTN_RES_MAX_Q; ++q) {
            acc[k][q] = 0.0f;
        }
        if (e < e1) {
            for (int b = 0; b < ns; ++b) {
                const float v = b < nb ? blocks[b*blocks_s1 + e] : xn[e];
#pragma unroll
                for (int q = 0; q < ATTN_RES_MAX_Q; ++q) {
                    if (q < nq) {
                        acc[k][q] += prob[q][b]*v;
                    }
                }
            }
        }
    }
    // every read of this block is done: with one block per token dst may reuse the inputs' memory
    __syncthreads();
#pragma unroll
    for (int k = 0; k < ATTN_RES_MAX_PER_THREAD; ++k) {
        const int e = e0 + tid + k*ATTN_RES_BLOCK;
        if (e < e1) {
#pragma unroll
            for (int q = 0; q < ATTN_RES_MAX_Q; ++q) {
                if (q < nq) {
                    dst[q*dst_s1 + e] = acc[k][q];
                }
            }
        }
    }
}

int ggml_cuda_try_attn_res_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = attn_res_env_flag("GGML_CUDA_ATTN_RES_FUSION");
    ggml_tensor * rms = cgraph->nodes[i];
    if (!enabled || rms->op != GGML_OP_RMS_NORM || rms->type != GGML_TYPE_F32) {
        return 0;
    }

    ggml_tensor * cat = nullptr;
    ggml_tensor * tr  = nullptr;
    ggml_tensor * ct  = nullptr;
    ggml_tensor * rep = nullptr;
    ggml_tensor * s   = nullptr;
    ggml_tensor * p   = nullptr;
    ggml_tensor * o   = nullptr;

    std::vector<int>     idx = { i };
    std::vector<ggml_op> ops = { GGML_OP_RMS_NORM };
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 16 && o == nullptr; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (cat == nullptr && t->op == GGML_OP_CONCAT && t->src[1] == rms) {
            cat = t;
        } else if (cat != nullptr && tr == nullptr && t->op == GGML_OP_TRANSPOSE && t->src[0] == cat) {
            tr = t;
        } else if (tr != nullptr && ct == nullptr && t->op == GGML_OP_CONT && t->src[0] == tr) {
            ct = t;
        } else if (cat != nullptr && s == nullptr && t->op == GGML_OP_MUL_MAT && t->src[0] == cat) {
            s = t;
        } else if (s != nullptr && p == nullptr && t->op == GGML_OP_SOFT_MAX && t->src[0] == s) {
            p = t;
        } else if (ct != nullptr && p != nullptr && t->op == GGML_OP_MUL_MAT && t->src[0] == ct && t->src[1] == p) {
            o = t;
        } else if (cat != nullptr && s == nullptr && rep == nullptr && t->op == GGML_OP_REPEAT && attn_res_is_weight_view(t->src[0])) {
            rep = t;
        } else if (cat != nullptr && s == nullptr && attn_res_is_view(t) && attn_res_is_weight_view(t)) {
            // the queries: a view of a weight
        } else {
            if (cat != nullptr) {
                attn_res_trace("attn-res", cgraph, i, "unexpected node in the subgraph");
            }
            return 0;
        }
        idx.push_back(j);
        ops.push_back(t->op);
    }
    if (o == nullptr) {
        return 0;
    }

    const ggml_tensor * blocks = cat->src[0];
    const ggml_tensor * cur    = rms->src[0];
    const ggml_tensor * qs     = s->src[1];
    const ggml_tensor * query  = qs->op == GGML_OP_REPEAT ? qs->src[0] : qs;
    if (qs->op == GGML_OP_REPEAT ? qs != rep : (rep != nullptr || !attn_res_is_weight_view(qs))) {
        attn_res_trace("attn-res", cgraph, i, "queries");
        return 0;
    }

    const int64_t E  = rms->ne[0];
    const int64_t T  = rms->ne[2];
    const int64_t nb = blocks->ne[1];
    const int64_t nq = qs->ne[1];
    float eps, scale, max_bias;
    memcpy(&eps,      rms->op_params, sizeof(float));
    memcpy(&scale,    (const float *) p->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) p->op_params + 1, sizeof(float));

    const bool shape_ok =
        rms->ne[1] == 1 && rms->ne[3] == 1 && ggml_get_op_params_i32(cat, 0) == 1 &&
        cur->type == GGML_TYPE_F32 && blocks->type == GGML_TYPE_F32 && query->type == GGML_TYPE_F32 && o->type == GGML_TYPE_F32 &&
        cur->nb[0] == sizeof(float) && blocks->nb[0] == sizeof(float) && query->nb[0] == sizeof(float) &&
        ggml_is_contiguous(o) && cur->ne[0] == E && blocks->ne[0] == E && blocks->ne[2] == T && blocks->ne[3] == 1 &&
        query->ne[0] == E && query->ne[1] == nq && (query->ne[2] == 1 || query->ne[2] == T) && query->ne[3] == 1 &&
        qs->ne[2] == T && o->ne[0] == E && o->ne[1] == nq && o->ne[2] == T &&
        p->src[1] == nullptr && p->src[2] == nullptr && max_bias == 0.0f &&
        nb + 1 <= ATTN_RES_MAX_STATES && nq <= ATTN_RES_MAX_Q && E <= ATTN_RES_BLOCK*ATTN_RES_MAX_PER_THREAD &&
        E*sizeof(float) <= 32*1024;
    if (!shape_ok) {
        attn_res_trace("attn-res", cgraph, i, "shapes or types");
        return 0;
    }
    // several blocks read what others write when the output shares memory with an input: one token, one block
    const bool overlap = attn_res_overlap(o, cur) || attn_res_overlap(o, blocks);
    if (T > 1 && overlap) {
        attn_res_trace("attn-res", cgraph, i, "output overlaps an input");
        return 0;
    }
    const int out_idx = idx.back();
    if (attn_res_planned(ctx, idx)) {
        attn_res_trace("attn-res", cgraph, i, "planned by the graph fusion");
        return 0;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx.data(), (int) idx.size(), ops.data(), &out_idx, 1)) {
        attn_res_trace("attn-res", cgraph, i, "uses outside the subgraph");
        return 0;
    }

    static std::atomic<bool> logged{false};
    if (attn_res_env_flag("GGML_CUDA_ATTN_RES_TRACE") && !logged.exchange(true)) {
        fprintf(stderr, "ggml_cuda: attn-res fusion active: E=%lld states=%lld queries=%lld tokens=%lld nodes=%d\n",
                (long long) E, (long long) nb + 1, (long long) nq, (long long) T, (int) idx.size());
    }

    // few tokens: split the outputs over more blocks (GGML_CUDA_ATTN_RES_SPLIT, default 4) to use more SMs
    static const int split_env = [] {
        const char * v = getenv("GGML_CUDA_ATTN_RES_SPLIT");
        return v != nullptr ? std::max(1, atoi(v)) : 4;
    }();
    const int n_split = !overlap && T*split_env <= 4*ggml_cuda_info().devices[ctx.device].nsm ? split_env : 1;
    const int chunk   = (int) ((E + n_split - 1)/n_split);
    auto aligned = [](const void * ptr, int64_t stride_bytes) {
        return ((uintptr_t) ptr % 16) == 0 && stride_bytes % 16 == 0;
    };
    const bool vec4 = E % 4 == 0 && aligned(blocks->data, blocks->nb[1]) && aligned(blocks->data, blocks->nb[2]) &&
                      aligned(query->data, query->nb[1]) && (query->ne[2] == 1 || aligned(query->data, query->nb[2]));
    const dim3 grid((unsigned) T, (unsigned) n_split, 1);
    const size_t smem = E*sizeof(float);
    if (vec4) {
        attn_res_mix_f32<true><<<grid, ATTN_RES_BLOCK, smem, ctx.stream()>>>(
            (const float *) blocks->data, blocks->nb[1]/sizeof(float), blocks->nb[2]/sizeof(float),
            (const float *) cur->data, cur->nb[2]/sizeof(float),
            (const float *) query->data, query->nb[1]/sizeof(float), query->ne[2] == 1 ? 0 : query->nb[2]/sizeof(float),
            (float *) o->data, o->nb[1]/sizeof(float), o->nb[2]/sizeof(float),
            (int) E, (int) nb, (int) nq, eps, scale, chunk);
    } else {
        attn_res_mix_f32<false><<<grid, ATTN_RES_BLOCK, smem, ctx.stream()>>>(
            (const float *) blocks->data, blocks->nb[1]/sizeof(float), blocks->nb[2]/sizeof(float),
            (const float *) cur->data, cur->nb[2]/sizeof(float),
            (const float *) query->data, query->nb[1]/sizeof(float), query->ne[2] == 1 ? 0 : query->nb[2]/sizeof(float),
            (float *) o->data, o->nb[1]/sizeof(float), o->nb[2]/sizeof(float),
            (int) E, (int) nb, (int) nq, eps, scale, chunk);
    }
    CUDA_CHECK(cudaGetLastError());

    return out_idx - i;
}

// h[r] = <w_r, [rms_norm(cur); rms_norm(msg)]> = r0*<w_r_lo, cur> + r1*<w_r_hi, msg>: one block per (row, token),
// one pass over cur, msg and the row
template <bool vec4>
static __global__ void __launch_bounds__(LOOP_GATE_DOWN_THREADS) loop_gate_down_f32(
        const float * __restrict__ cur, const int64_t cur_s, const float * __restrict__ msg, const int64_t msg_s,
        const float * __restrict__ w, const int64_t w_s, float * __restrict__ h, const int E, const int R, const float eps) {
    constexpr int n_warps = LOOP_GATE_DOWN_THREADS / WARP_SIZE;

    const int r    = blockIdx.x;
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    __shared__ float2 red[2][n_warps];

    ggml_cuda_pdl_sync();

    cur += t*cur_s;
    msg += t*msg_s;
    const float * wlo = w + r*w_s;
    const float * whi = wlo + E;

    float2 ss = make_float2(0.0f, 0.0f); // sum of squares of cur, msg
    float2 dd = make_float2(0.0f, 0.0f); // <w_lo, cur>, <w_hi, msg>
    if (vec4) {
        for (int e4 = tid; e4 < E/4; e4 += LOOP_GATE_DOWN_THREADS) {
            const float4 c = ((const float4 *) cur)[e4];
            const float4 m = ((const float4 *) msg)[e4];
            const float4 a = ((const float4 *) wlo)[e4];
            const float4 b = ((const float4 *) whi)[e4];
            ss.x += c.x*c.x + c.y*c.y + c.z*c.z + c.w*c.w;
            ss.y += m.x*m.x + m.y*m.y + m.z*m.z + m.w*m.w;
            dd.x += a.x*c.x + a.y*c.y + a.z*c.z + a.w*c.w;
            dd.y += b.x*m.x + b.y*m.y + b.z*m.z + b.w*m.w;
        }
    } else {
        for (int e = tid; e < E; e += LOOP_GATE_DOWN_THREADS) {
            const float c = cur[e];
            const float m = msg[e];
            ss.x += c*c;
            ss.y += m*m;
            dd.x += wlo[e]*c;
            dd.y += whi[e]*m;
        }
    }
    ss = warp_reduce_sum(ss);
    dd = warp_reduce_sum(dd);
    if (lane == 0) {
        red[0][warp] = ss;
        red[1][warp] = dd;
    }
    __syncthreads();
    if (warp == 0) {
        ss = warp_reduce_sum(lane < n_warps ? red[0][lane] : make_float2(0.0f, 0.0f));
        dd = warp_reduce_sum(lane < n_warps ? red[1][lane] : make_float2(0.0f, 0.0f));
        if (lane == 0) {
            h[t*R + r] = rsqrtf(ss.x/E + eps)*dd.x + rsqrtf(ss.y/E + eps)*dd.y;
        }
    }
}

// dst = (1 - tanh(<wf_e, h>))*cur + tanh(<wg_e, h>)*msg, one warp per channel e
static __global__ void __launch_bounds__(LOOP_GATE_OUT_WARPS*WARP_SIZE) loop_gate_out_f32(
        const float * __restrict__ h, const float * __restrict__ wf, const int64_t wf_s,
        const float * __restrict__ wg, const int64_t wg_s,
        const float * cur, const int64_t cur_s, const float * msg, const int64_t msg_s,
        float * dst, const int64_t dst_s, const int E, const int R) {
    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int e    = blockIdx.x*LOOP_GATE_OUT_WARPS + warp;

    extern __shared__ float hs[]; // R

    ggml_cuda_pdl_sync();

    for (int k = tid; k < R; k += LOOP_GATE_OUT_WARPS*WARP_SIZE) {
        hs[k] = h[t*R + k];
    }
    __syncthreads();
    if (e >= E) {
        return;
    }

    const float * f = wf + e*wf_s;
    const float * g = wg + e*wg_s;
    float a = 0.0f;
    float b = 0.0f;
    for (int k = lane; k < R; k += WARP_SIZE) {
        a += f[k]*hs[k];
        b += g[k]*hs[k];
    }
    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
        // the unfused ops round every step: scale_bias, two muls, add
        const float keep = -1.0f*tanhf(a) + 1.0f;
        dst[t*dst_s + e] = __fadd_rn(__fmul_rn(keep, cur[t*cur_s + e]), __fmul_rn(tanhf(b), msg[t*msg_s + e]));
    }
}

// as loop_gate_out_f32 with R = 4*R4: R4 lanes per channel, one float4 of each row per lane
template <int R4>
static __global__ void __launch_bounds__(LOOP_GATE_OUT_WARPS*WARP_SIZE) loop_gate_out_f32_v(
        const float * __restrict__ h, const float * __restrict__ wf, const int64_t wf_s,
        const float * __restrict__ wg, const int64_t wg_s,
        const float * cur, const int64_t cur_s, const float * msg, const int64_t msg_s,
        float * dst, const int64_t dst_s, const int E) {
    static_assert(WARP_SIZE % R4 == 0, "R4 must divide the warp");
    constexpr int per_warp = WARP_SIZE / R4;

    const int t    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int k4   = lane % R4;
    const int e    = (blockIdx.x*LOOP_GATE_OUT_WARPS + warp)*per_warp + lane / R4;

    __shared__ float4 hs[R4];

    ggml_cuda_pdl_sync();

    if (tid < R4) {
        hs[tid] = ((const float4 *) (h + t*4*R4))[tid];
    }
    __syncthreads();

    float a = 0.0f;
    float b = 0.0f;
    if (e < E) {
        const float4 hv = hs[k4];
        const float4 f  = ((const float4 *) (wf + e*wf_s))[k4];
        const float4 g  = ((const float4 *) (wg + e*wg_s))[k4];
        a = f.x*hv.x + f.y*hv.y + f.z*hv.z + f.w*hv.w;
        b = g.x*hv.x + g.y*hv.y + g.z*hv.z + g.w*hv.w;
    }
#pragma unroll
    for (int off = R4/2; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, off, WARP_SIZE);
        b += __shfl_xor_sync(0xffffffff, b, off, WARP_SIZE);
    }
    if (e < E && k4 == 0) {
        const float keep = -1.0f*tanhf(a) + 1.0f;
        dst[t*dst_s + e] = __fadd_rn(__fmul_rn(keep, cur[t*cur_s + e]), __fmul_rn(tanhf(b), msg[t*msg_s + e]));
    }
}

int ggml_cuda_try_loop_gate_fusion(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = attn_res_env_flag("GGML_CUDA_LOOP_GATE_FUSION");
    static const int64_t max_tokens = [] {
        const char * v = getenv("GGML_CUDA_LOOP_GATE_MAX_TOKENS");
        return v != nullptr ? (int64_t) atoll(v) : (int64_t) 8;
    }();
    ggml_tensor * cat = cgraph->nodes[i];
    if (!enabled || cat->op != GGML_OP_CONCAT || cat->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cat, 0) != 1 ||
            cat->ne[1] != 2 || cat->ne[2] > max_tokens) {
        return 0;
    }

    ggml_tensor * r      = nullptr;
    ggml_tensor * rv     = nullptr;
    ggml_tensor * d      = nullptr;
    ggml_tensor * mm[2]  = { nullptr, nullptr };
    ggml_tensor * th[2]  = { nullptr, nullptr };
    ggml_tensor * sb     = nullptr;
    ggml_tensor * keep   = nullptr; // (1 - forget)*cur
    ggml_tensor * gained = nullptr; // gain*msg
    ggml_tensor * out    = nullptr;

    std::vector<int>     idx = { i };
    std::vector<ggml_op> ops = { GGML_OP_CONCAT };
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 20 && out == nullptr; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        bool ok = false;
        if (r == nullptr && t->op == GGML_OP_RMS_NORM && t->src[0] == cat) {
            r = t;
            ok = true;
        } else if (r != nullptr && rv == nullptr && t->op == GGML_OP_RESHAPE && t->src[0] == r) {
            rv = t;
            ok = true;
        } else if (rv != nullptr && d == nullptr && t->op == GGML_OP_MUL_MAT && t->src[1] == rv) {
            d = t;
            ok = true;
        } else if (d != nullptr && t->op == GGML_OP_MUL_MAT && t->src[1] == d && (mm[0] == nullptr || mm[1] == nullptr)) {
            mm[mm[0] == nullptr ? 0 : 1] = t;
            ok = true;
        } else if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_TANH) {
            for (int k = 0; k < 2; ++k) {
                if (mm[k] != nullptr && t->src[0] == mm[k] && th[k] == nullptr) {
                    th[k] = t;
                    ok = true;
                    break;
                }
            }
        } else if (sb == nullptr && t->op == GGML_OP_SCALE && (t->src[0] == th[0] || t->src[0] == th[1]) && t->src[0] != nullptr) {
            sb = t;
            ok = true;
        } else if (t->op == GGML_OP_MUL && sb != nullptr && keep == nullptr && t->src[0] == sb) {
            keep = t;
            ok = true;
        } else if (t->op == GGML_OP_MUL && gained == nullptr && t->src[0] != nullptr &&
                   (t->src[0] == th[0] || t->src[0] == th[1]) && t->src[0] != (sb ? sb->src[0] : nullptr)) {
            gained = t;
            ok = true;
        } else if (t->op == GGML_OP_ADD && keep != nullptr && gained != nullptr && t->src[0] == keep && t->src[1] == gained) {
            out = t;
            ok = true;
        }
        if (!ok) {
            if (r != nullptr) {
                attn_res_trace("loop-gate", cgraph, i, "unexpected node in the subgraph");
            }
            return 0;
        }
        idx.push_back(j);
        ops.push_back(t->op);
    }
    if (out == nullptr) {
        return 0;
    }

    // which branch is the forget gate: the one behind scale_bias
    const ggml_tensor * thf = sb->src[0];
    const ggml_tensor * mmf = thf->src[0];
    const ggml_tensor * mmg = mmf == mm[0] ? mm[1] : mm[0];
    float sb_scale, sb_bias;
    memcpy(&sb_scale, (const float *) sb->op_params + 0, sizeof(float));
    memcpy(&sb_bias,  (const float *) sb->op_params + 1, sizeof(float));

    const ggml_tensor * a   = cat->src[0]; // cur as [E, 1, T]
    const ggml_tensor * b   = cat->src[1]; // msg as [E, 1, T]
    const ggml_tensor * cur = keep->src[1];
    const ggml_tensor * msg = gained->src[1];
    const ggml_tensor * wd  = d->src[0];
    const ggml_tensor * wf  = mmf->src[0];
    const ggml_tensor * wg  = mmg->src[0];

    const int64_t E = a->ne[0];
    const int64_t T = a->ne[2];
    const int64_t R = wd->ne[1];
    float eps;
    memcpy(&eps, r->op_params, sizeof(float));

    const bool shape_ok =
        mmg != nullptr && sb_scale == -1.0f && sb_bias == 1.0f &&
        a->ne[1] == 1 && b->ne[0] == E && b->ne[1] == 1 && b->ne[2] == T && a->ne[3] == 1 && b->ne[3] == 1 &&
        a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) &&
        cur->data == a->data && msg->data == b->data && ggml_nelements(cur) == E*T && ggml_nelements(msg) == E*T &&
        cur->type == GGML_TYPE_F32 && msg->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) && ggml_is_contiguous(msg) &&
        rv->ne[0] == 2*E && rv->ne[1] == T &&
        wd->type == GGML_TYPE_F32 && wd->ne[0] == 2*E && wd->nb[0] == sizeof(float) && wd->ne[2] == 1 &&
        wf->type == GGML_TYPE_F32 && wf->ne[0] == R && wf->ne[1] == E && wf->nb[0] == sizeof(float) && wf->ne[2] == 1 &&
        wg->type == GGML_TYPE_F32 && wg->ne[0] == R && wg->ne[1] == E && wg->nb[0] == sizeof(float) && wg->ne[2] == 1 &&
        attn_res_is_weight_view(wd) && attn_res_is_weight_view(wf) && attn_res_is_weight_view(wg) &&
        out->type == GGML_TYPE_F32 && ggml_is_contiguous(out) && ggml_nelements(out) == E*T &&
        R <= 1024;
    if (!shape_ok) {
        attn_res_trace("loop-gate", cgraph, i, "shapes or types");
        return 0;
    }
    // out[e] reads cur[e] and msg[e] only: an exact alias is fine, a shifted overlap is not
    if ((out->data != cur->data && attn_res_overlap(out, cur)) || (out->data != msg->data && attn_res_overlap(out, msg))) {
        attn_res_trace("loop-gate", cgraph, i, "output overlaps an input");
        return 0;
    }
    const int out_idx = idx.back();
    if (attn_res_planned(ctx, idx)) {
        attn_res_trace("loop-gate", cgraph, i, "planned by the graph fusion");
        return 0;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idx.data(), (int) idx.size(), ops.data(), &out_idx, 1)) {
        attn_res_trace("loop-gate", cgraph, i, "uses outside the subgraph");
        return 0;
    }

    static std::atomic<bool> logged{false};
    if (attn_res_env_flag("GGML_CUDA_ATTN_RES_TRACE") && !logged.exchange(true)) {
        fprintf(stderr, "ggml_cuda: loop-gate fusion active: E=%lld rank=%lld tokens=%lld nodes=%d\n",
                (long long) E, (long long) R, (long long) T, (int) idx.size());
    }

    auto aligned = [](const void * ptr, int64_t stride_bytes) {
        return ((uintptr_t) ptr % 16) == 0 && stride_bytes % 16 == 0;
    };
    ggml_cuda_pool_alloc<float> h(ctx.pool(), R*T);
    const bool down4 = E % 4 == 0 && aligned(a->data, a->nb[2]) && aligned(b->data, b->nb[2]) &&
                       aligned(wd->data, wd->nb[1]) && aligned((const char *) wd->data + E*sizeof(float), wd->nb[1]);
    const dim3 grid_down((unsigned) R, (unsigned) T, 1);
    if (down4) {
        loop_gate_down_f32<true><<<grid_down, LOOP_GATE_DOWN_THREADS, 0, ctx.stream()>>>(
            (const float *) a->data, a->nb[2]/sizeof(float), (const float *) b->data, b->nb[2]/sizeof(float),
            (const float *) wd->data, wd->nb[1]/sizeof(float), h.get(), (int) E, (int) R, eps);
    } else {
        loop_gate_down_f32<false><<<grid_down, LOOP_GATE_DOWN_THREADS, 0, ctx.stream()>>>(
            (const float *) a->data, a->nb[2]/sizeof(float), (const float *) b->data, b->nb[2]/sizeof(float),
            (const float *) wd->data, wd->nb[1]/sizeof(float), h.get(), (int) E, (int) R, eps);
    }
    CUDA_CHECK(cudaGetLastError());

    const bool out4 = aligned(wf->data, wf->nb[1]) && aligned(wg->data, wg->nb[1]);
    auto launch_v = [&](auto r4) {
        constexpr int R4 = decltype(r4)::value;
        constexpr int per_block = LOOP_GATE_OUT_WARPS*(WARP_SIZE/R4);
        const dim3 grid((unsigned) ((E + per_block - 1)/per_block), (unsigned) T, 1);
        loop_gate_out_f32_v<R4><<<grid, LOOP_GATE_OUT_WARPS*WARP_SIZE, 0, ctx.stream()>>>(
            h.get(), (const float *) wf->data, wf->nb[1]/sizeof(float), (const float *) wg->data, wg->nb[1]/sizeof(float),
            (const float *) cur->data, E, (const float *) msg->data, E, (float *) out->data, E, (int) E);
    };
    if (out4 && R == 64) {
        launch_v(std::integral_constant<int, 16>());
    } else if (out4 && R == 128) {
        launch_v(std::integral_constant<int, 32>());
    } else if (out4 && R == 32) {
        launch_v(std::integral_constant<int, 8>());
    } else {
        const dim3 grid_out((unsigned) ((E + LOOP_GATE_OUT_WARPS - 1)/LOOP_GATE_OUT_WARPS), (unsigned) T, 1);
        loop_gate_out_f32<<<grid_out, LOOP_GATE_OUT_WARPS*WARP_SIZE, R*sizeof(float), ctx.stream()>>>(
            h.get(), (const float *) wf->data, wf->nb[1]/sizeof(float), (const float *) wg->data, wg->nb[1]/sizeof(float),
            (const float *) cur->data, E, (const float *) msg->data, E, (float *) out->data, E, (int) E, (int) R);
    }
    CUDA_CHECK(cudaGetLastError());

    return out_idx - i;
}
