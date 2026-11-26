#include "fattn.cuh"
#include "fattn-common.cuh"
#include "kvflash-fa.h"

#include <cstring>

static void set_tensor(ggml_tensor * t, ggml_type type, const void * data,
        int64_t n0, int64_t n1, int64_t n2, size_t nb1, size_t nb2) {
    std::memset(t, 0, sizeof(*t));
    t->type = type;
    t->ne[0] = n0;
    t->ne[1] = n1;
    t->ne[2] = n2;
    t->ne[3] = 1;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = nb1;
    t->nb[2] = nb2;
    t->nb[3] = nb2 * (size_t) n2;
    t->data = const_cast<void *>(data);
}

// One context per device, used only from the thread that runs the stream op.
static ggml_backend_cuda_context * kvflash_ctx(int device) {
    static ggml_backend_cuda_context * ctx[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }
    if (!ctx[device]) {
        ctx[device] = new ggml_backend_cuda_context(device);
    }
    return ctx[device];
}

extern "C" void * ggml_cuda_kvflash_stream(int device) {
    ggml_backend_cuda_context * ctx = kvflash_ctx(device);
    if (!ctx) {
        return nullptr;
    }
    ggml_cuda_set_device(device);
    return (void *) ctx->stream();
}

extern "C" int ggml_cuda_kvflash_fa(int device, ggml_cuda_kvflash_fa_call * c) {
    ggml_backend_cuda_context * ctx = kvflash_ctx(device);
    if (!ctx || !c || !c->q || !c->k || !c->v || !c->mask || !c->dst || c->n_q <= 0 || c->n_kv <= 0 ||
        c->n_head <= 0 || c->n_head_kv <= 0 || c->n_head % c->n_head_kv != 0 || c->head_dim <= 0) {
        return 1;
    }
    c->blocks = 0;

    const size_t hd2 = (size_t) c->head_dim * sizeof(half);
    ggml_tensor Q, K, V, M, O;
    set_tensor(&Q, GGML_TYPE_F32, c->q, c->head_dim, c->n_q, c->n_head, c->q_nb1, c->q_nb2);
    set_tensor(&K, GGML_TYPE_F16, c->k, c->head_dim, c->n_kv, c->n_head_kv, c->k_nb1, hd2);
    set_tensor(&V, GGML_TYPE_F16, c->v, c->head_dim, c->n_kv, c->n_head_kv, c->v_nb1, hd2);
    set_tensor(&M, GGML_TYPE_F16, c->mask, c->n_kv, c->n_q, 1,
            (size_t) c->n_kv * sizeof(half), (size_t) c->n_kv * c->n_q * sizeof(half));
    set_tensor(&O, GGML_TYPE_F32, c->dst, c->head_dim, c->n_head, c->n_q,
            (size_t) c->head_dim * sizeof(float), (size_t) c->head_dim * c->n_head * sizeof(float));
    O.op = GGML_OP_FLASH_ATTN_EXT;
    O.src[0] = &Q;
    O.src[1] = &K;
    O.src[2] = &V;
    O.src[3] = &M;
    const float params[3] = { c->scale, 0.f, 0.f };
    std::memcpy(O.op_params, params, sizeof(params));

    if (!ggml_cuda_flash_attn_ext_supported(device, &O)) {
        return 2;
    }
    ggml_cuda_fattn_parts parts;
    parts.parts = c->parts;
    parts.meta = (float2 *) c->meta;
    parts.cap_parts = c->parts ? c->cap_parts : 0;
    parts.cap_meta = c->meta ? c->cap_meta : 0;
    parts.allow_single = c->allow_single != 0;
    if (!ggml_cuda_flash_attn_ext_parts(*ctx, &O, &parts)) {
        return 3;
    }
    c->blocks = parts.blocks;
    return 0;
}
