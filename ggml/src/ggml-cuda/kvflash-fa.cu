#include "fattn.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>

struct ggml_cuda_kvflash_seg {
    const uint16_t * k;
    const uint16_t * v;
    int n_tok;
    int pos0;
    int k_stride;
    int v_stride;
};

extern "C" int ggml_cuda_kvflash_fa(
        const float * q,
        const ggml_cuda_kvflash_seg * segs, int n_segs,
        float * dst,
        int n_q, int n_head, int n_head_kv, int head_dim,
        float scale, int q_pos0, int pos_span);

static void set_tensor(ggml_tensor * t, ggml_type type, const void * data,
        int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    std::memset(t, 0, sizeof(*t));
    t->type = type;
    t->ne[0] = n0;
    t->ne[1] = n1;
    t->ne[2] = n2;
    t->ne[3] = n3;
    const size_t es = ggml_type_size(type);
    t->nb[0] = es;
    t->nb[1] = es * (size_t) n0;
    t->nb[2] = t->nb[1] * (size_t) n1;
    t->nb[3] = t->nb[2] * (size_t) n2;
    t->data = const_cast<void *>(data);
}

__global__ void kvflash_pack_fa(
        const ggml_cuda_kvflash_seg * __restrict__ segs, int n_segs,
        uint16_t * __restrict__ kdst, uint16_t * __restrict__ vdst,
        int n_kv, int n_h, int D) {
    const int s = (int) blockIdx.y;
    if (s >= n_segs) {
        return;
    }
    const ggml_cuda_kvflash_seg seg = segs[s];
    const int idx = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int total = seg.n_tok * n_h * D;
    if (idx >= total) {
        return;
    }
    const int d = idx % D;
    const int h = (idx / D) % n_h;
    const int t = idx / (D * n_h);
    const size_t dst_i = (size_t) d + (size_t) D * ((size_t) (seg.pos0 + t) + (size_t) n_kv * h);
    kdst[dst_i] = seg.k[(size_t) t * seg.k_stride + (size_t) h * D + d];
    vdst[dst_i] = seg.v[(size_t) t * seg.v_stride + (size_t) h * D + d];
}

__global__ void kvflash_causal_mask(half * __restrict__ mask, int n_kv, int n_q, int q_pos0) {
    const int kv = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int q = (int) blockIdx.y;
    if (kv >= n_kv || q >= n_q) {
        return;
    }
    const float v = kv <= q_pos0 + q ? 0.f : -INFINITY;
    mask[(size_t) q * n_kv + kv] = __float2half(v);
}

static bool grow(void ** ptr, size_t & cap, size_t need) {
    if (cap >= need) {
        return true;
    }
    void * next = nullptr;
    if (cudaMalloc(&next, need) != cudaSuccess) {
        return false;
    }
    cudaFree(*ptr);
    *ptr = next;
    cap = need;
    return true;
}

extern "C" int ggml_cuda_kvflash_fa(
        const float * q,
        const ggml_cuda_kvflash_seg * segs, int n_segs,
        float * dst,
        int n_q, int n_head, int n_head_kv, int head_dim,
        float scale, int q_pos0, int pos_span) {
    if (!q || !dst || !segs || n_segs <= 0 || n_q <= 0 || pos_span <= 0 ||
        n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0) {
        return 1;
    }

    static ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(0);
    static uint16_t * k_fa = nullptr;
    static uint16_t * v_fa = nullptr;
    static half * mask = nullptr;
    static size_t k_cap = 0;
    static size_t v_cap = 0;
    static size_t m_cap = 0;

    // Pad to the flash-attention tile so the decode kernel can use the vector path.
    const int n_fa = (pos_span + 255) & ~255;
    const size_t kv_elems = (size_t) n_fa * n_head_kv * head_dim;
    const size_t mask_elems = (size_t) n_fa * n_q;
    if (!grow((void **) &k_fa, k_cap, kv_elems * sizeof(uint16_t)) ||
        !grow((void **) &v_fa, v_cap, kv_elems * sizeof(uint16_t)) ||
        !grow((void **) &mask, m_cap, mask_elems * sizeof(half))) {
        return 2;
    }

    cudaStream_t stream = ctx->stream();
    static ggml_cuda_kvflash_seg * seg_dev = nullptr;
    static size_t seg_cap = 0;
    if (!grow((void **) &seg_dev, seg_cap, (size_t) n_segs * sizeof(ggml_cuda_kvflash_seg))) {
        return 3;
    }
    if (cudaMemcpyAsync(seg_dev, segs, (size_t) n_segs * sizeof(ggml_cuda_kvflash_seg),
                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return 3;
    }
    const int widest = n_segs > 0 ? segs[0].n_tok : 0;
    int max_tok = widest;
    for (int s = 1; s < n_segs; ++s) {
        max_tok = std::max(max_tok, segs[s].n_tok);
    }
    const int total = max_tok * n_head_kv * head_dim;
    const dim3 pack_grid((total + 255) / 256, n_segs);
    kvflash_pack_fa<<<pack_grid, 256, 0, stream>>>(
            seg_dev, n_segs, k_fa, v_fa, n_fa, n_head_kv, head_dim);
    kvflash_causal_mask<<<dim3((n_fa + 255) / 256, n_q), 256, 0, stream>>>(
            mask, n_fa, n_q, q_pos0);
    if (cudaGetLastError() != cudaSuccess) {
        return 5;
    }

    ggml_tensor Q, K, V, M, O;
    set_tensor(&Q, GGML_TYPE_F32, q, head_dim, n_q, n_head, 1);
    set_tensor(&K, GGML_TYPE_F16, k_fa, head_dim, n_fa, n_head_kv, 1);
    set_tensor(&V, GGML_TYPE_F16, v_fa, head_dim, n_fa, n_head_kv, 1);
    set_tensor(&M, GGML_TYPE_F16, mask, n_fa, n_q, 1, 1);
    set_tensor(&O, GGML_TYPE_F32, dst, head_dim, n_head, n_q, 1);
    O.op = GGML_OP_FLASH_ATTN_EXT;
    O.src[0] = &Q;
    O.src[1] = &K;
    O.src[2] = &V;
    O.src[3] = &M;
    float params[3] = { scale, 0.f, 0.f };
    std::memcpy(O.op_params, params, sizeof(params));

    if (!ggml_cuda_flash_attn_ext_supported(ctx->device, &O)) {
        return 6;
    }
    ggml_cuda_flash_attn_ext(*ctx, &O);
    if (cudaStreamSynchronize(stream) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        return 7;
    }
    return 0;
}
