#include "mhc.cuh"

// One block per row (token): the RMS sum and the eight router logits are
// reduced over both streams, then thread 0 runs the gates and the sinkhorn
// passes, and all threads write the gated branch input.
static constexpr int MHC_ROUTE_THREADS = 256;

static __global__ void mhc_route_f32(
        const char * __restrict__ a, const char * __restrict__ b,
        const float * __restrict__ w, const float * __restrict__ bias,
        float * __restrict__ dst,
        const int64_t d, const int64_t n_tok,
        const size_t nba1, const size_t nba2, const size_t nbb1, const size_t nbb2,
        const int n_iter, const float eps) {
    const int64_t r = blockIdx.x;
    const int64_t t = r % n_tok;
    const int64_t s = r / n_tok;

    const float * va = (const float *) (a + t*nba1 + s*nba2);
    const float * vb = (const float *) (b + t*nbb1 + s*nbb2);
    float       * po = dst + r*(d + 6);

    float acc[9] = { 0 }; // 8 logits, then the sum of squares
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float xa = va[i];
        const float xb = vb[i];
        acc[8] += xa*xa + xb*xb;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            acc[k] += w[k*2*d + i]*xa + w[k*2*d + d + i]*xb;
        }
    }

    __shared__ float part[9][MHC_ROUTE_THREADS/WARP_SIZE];
    __shared__ float gate[2];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        const float v = warp_reduce_sum(acc[k]);
        if (lane == 0) {
            part[k][warp] = v;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float lg[9];
        for (int k = 0; k < 9; ++k) {
            lg[k] = 0.0f;
            for (int j = 0; j < (int) (blockDim.x/WARP_SIZE); ++j) {
                lg[k] += part[k][j];
            }
        }
        const float rms = rsqrtf(lg[8]/(2.0f*d) + eps);
        for (int k = 0; k < 8; ++k) {
            lg[k] = lg[k]*rms + bias[k];
        }
        gate[0] = 1.0f/(1.0f + expf(-lg[0]));
        gate[1] = 1.0f/(1.0f + expf(-lg[1]));

        float m00 = lg[4], m01 = lg[5], m10 = lg[6], m11 = lg[7];
        for (int it = 0; it < n_iter; ++it) {
            float mx, lse;
            mx = fmaxf(m00, m01); lse = mx + logf(expf(m00 - mx) + expf(m01 - mx)); m00 -= lse; m01 -= lse;
            mx = fmaxf(m10, m11); lse = mx + logf(expf(m10 - mx) + expf(m11 - mx)); m10 -= lse; m11 -= lse;
            mx = fmaxf(m00, m10); lse = mx + logf(expf(m00 - mx) + expf(m10 - mx)); m00 -= lse; m10 -= lse;
            mx = fmaxf(m01, m11); lse = mx + logf(expf(m01 - mx) + expf(m11 - mx)); m01 -= lse; m11 -= lse;
        }
        po[d + 0] = 2.0f/(1.0f + expf(-lg[2]));
        po[d + 1] = 2.0f/(1.0f + expf(-lg[3]));
        po[d + 2] = expf(m00);
        po[d + 3] = expf(m01);
        po[d + 4] = expf(m10);
        po[d + 5] = expf(m11);
    }
    __syncthreads();

    const float pre0 = gate[0];
    const float pre1 = gate[1];
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        po[i] = pre0*va[i] + pre1*vb[i];
    }
}

static __global__ void mhc_merge_f32(
        const char * __restrict__ a, const char * __restrict__ b,
        const char * __restrict__ x, const char * __restrict__ p,
        float * __restrict__ dst,
        const int64_t d, const int64_t n_tok,
        const size_t nba1, const size_t nba2, const size_t nbb1, const size_t nbb2,
        const size_t nbx1, const size_t nbx2, const size_t nbp1, const size_t nbp2) {
    const int64_t r = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= d) {
        return;
    }
    const int64_t t = r % n_tok;
    const int64_t s = r / n_tok;

    const float * vp = (const float *) (p + t*nbp1 + s*nbp2);
    const float xa = ((const float *) (a + t*nba1 + s*nba2))[i];
    const float xb = ((const float *) (b + t*nbb1 + s*nbb2))[i];
    const float xx = ((const float *) (x + t*nbx1 + s*nbx2))[i];

    float * vo = dst + r*2*d;
    vo[i]     = vp[d + 2]*xa + vp[d + 3]*xb + vp[d + 0]*xx;
    vo[d + i] = vp[d + 4]*xa + vp[d + 5]*xb + vp[d + 1]*xx;
}

void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a    = dst->src[0];
    const ggml_tensor * b    = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * bias = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_row = a->ne[1]*a->ne[2];
    const int     n_iter = ggml_get_op_params_i32(dst, 0);
    const float   eps    = ggml_get_op_params_f32(dst, 1);

    mhc_route_f32<<<n_row, MHC_ROUTE_THREADS, 0, ctx.stream()>>>(
        (const char *) a->data, (const char *) b->data,
        (const float *) w->data, (const float *) bias->data, (float *) dst->data,
        a->ne[0], a->ne[1], a->nb[1], a->nb[2], b->nb[1], b->nb[2], n_iter, eps);
}

void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * x = dst->src[2];
    const ggml_tensor * p = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t d     = a->ne[0];
    const int64_t n_row = a->ne[1]*a->ne[2];
    const int     threads = 256;
    const dim3    grid((d + threads - 1)/threads, n_row, 1);

    mhc_merge_f32<<<grid, threads, 0, ctx.stream()>>>(
        (const char *) a->data, (const char *) b->data, (const char *) x->data, (const char *) p->data,
        (float *) dst->data, d, a->ne[1],
        a->nb[1], a->nb[2], b->nb[1], b->nb[2], x->nb[1], x->nb[2], p->nb[1], p->nb[2]);
}
