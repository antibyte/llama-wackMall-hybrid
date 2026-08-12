#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-layer-policy.h"
#include "llama-model.h"
#include "llama-context.h"

#include "kvflash_pager.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    if (const char * value = std::getenv("LLAMA_KV_Q4_SCALE")) {
        const bool weighted   = std::strcmp(value, "weighted") == 0;
        const bool weighted_k = std::strcmp(value, "weighted-k") == 0;
        const bool weighted_v = std::strcmp(value, "weighted-v") == 0;
        if (value[0] != '\0' && std::strcmp(value, "legacy") != 0 && !weighted && !weighted_k && !weighted_v) {
            throw std::invalid_argument(
                "LLAMA_KV_Q4_SCALE must be legacy, weighted, weighted-k, or weighted-v");
        }
        q4_weighted_scale_k = (weighted || weighted_k) && type_k == GGML_TYPE_Q4_0;
        q4_weighted_scale_v = (weighted || weighted_v) && type_v == GGML_TYPE_Q4_0;
    }
    if (q4_weighted_scale_k || q4_weighted_scale_v) {
        LLAMA_LOG_WARN("%s: experimental weighted Q4_0 scale enabled for %s%s\n", __func__,
            q4_weighted_scale_k ? "K" : "", q4_weighted_scale_v ? "V" : "");
    }

    std::vector<uint8_t> turbo4_q8_fallback(n_layer, 0);
    size_t n_turbo4_q8_fallback = 0;
    if (const char * value = std::getenv("LLAMA_TURBO4_Q8_FALLBACK_LAYERS")) {
        // A process can own a Turbo4 target context and a Q8/Q4 MTP draft
        // context. The process-wide fallback policy applies only to Turbo4 K;
        // the common CLI validates that the target configuration is Turbo4.
        if (value[0] != '\0' && type_k == GGML_TYPE_TURBO4_K) {
            if (other || reuse || share) {
                throw std::invalid_argument(
                    "Turbo4 per-layer Q8 fallback does not support cache sharing or layer reuse yet");
            }
            turbo4_q8_fallback = llama_kv_layer_policy::parse_layer_set(value, n_layer);
            n_turbo4_q8_fallback = std::count(turbo4_q8_fallback.begin(), turbo4_q8_fallback.end(), uint8_t(1));
            LLAMA_LOG_WARN(
                "%s: experimental Turbo4 Q8 fallback enabled for %zu layer(s): %s\n",
                __func__, n_turbo4_q8_fallback, value);
        }
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(3u*(1 + n_stream)*n_layer*ggml_tensor_overhead()), //Reserve tensor metadata for up to 3 tensors per layer (K, V, and optional K_idx), plus one view per tensor per stream.
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        const ggml_type type_k_layer = turbo4_q8_fallback[il] ? GGML_TYPE_Q8_0 : type_k;
        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k_layer, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        const uint32_t n_embd_k_idx = hparams.n_embd_k_idx(il);
        ggml_tensor * k_idx = n_embd_k_idx > 0
            ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_k_idx, kv_size, n_stream)
            : nullptr;
        if (k_idx) {
            ggml_format_name(k_idx, "cache_k_idx_l%d", il);
            msa_strict_slots = (n_stream == n_seq_max);
        }

        std::vector<ggml_tensor *> k_idx_stream;
        for (uint32_t s = 0; s < n_stream; ++s) {
            k_idx_stream.push_back(k_idx
                ? ggml_view_2d(ctx, k_idx, n_embd_k_idx, kv_size, k_idx->nb[1], s*k_idx->nb[2])
                : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_idx, k_stream, v_stream, k_idx_stream });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k     = size_k_bytes();
        const size_t memory_size_v     = size_v_bytes();
        const size_t memory_size_k_idx = size_k_idx_bytes();
        const size_t memory_size_total = memory_size_k + memory_size_v + memory_size_k_idx;

        constexpr float mib = 1024.0f * 1024.0f;

        const char * k_type_log = n_turbo4_q8_fallback ? "mixed turbo4_k/q8_0" : ggml_type_name(type_k);
        const std::string k_log = format(", K (%s): %7.2f MiB", k_type_log, (float) memory_size_k / mib);
        const std::string v_log = format(", V (%s): %7.2f MiB", ggml_type_name(type_v), (float) memory_size_v / mib);

        std::string k_idx_log;
        if (memory_size_k_idx > 0) {
            k_idx_log = format(", K_idx (%s): %7.2f MiB", ggml_type_name(GGML_TYPE_F32), (float) memory_size_k_idx / mib);
        }

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs)%s%s%s\n", __func__,
                (float) memory_size_total / mib, kv_size, (int) layers.size(), n_seq_max, n_stream,
                k_log.c_str(), v_log.c_str(), k_idx_log.c_str());
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 || model.arch == LLM_ARCH_GLM_DSA) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (kvflash) {
        kvflash->reset();
        kvflash_cells.clear();
        kvflash_pos_mins.clear();
        kvflash_pos_maxs.clear();
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // empty range - nothing to remove
    if (p0 >= p1) {
        return true;
    }

    // MSA anchors block selection to absolute cache slots (slot == position). Tail trim and full removal preserve this invariant, but removing a prefix
    // or middle range would free slots while later cells survive, desynchronizing the indexer cache. Reject such removals before modifying the cache.
    if (msa_strict_slots) {
        for (llama_seq_id sid = 0; sid < (llama_seq_id) seq_to_stream.size(); ++sid) {
            if (seq_id >= 0 && sid != seq_id) {
                continue;
            }

            const auto & cells = v_cells[seq_to_stream[sid]];

            const llama_pos pmin = cells.seq_pos_min(sid);
            const llama_pos pmax = cells.seq_pos_max(sid);

            if (pmin < 0) {
                continue;   // empty sequence
            }

            const bool overlaps    = p0 <= pmax && p1 > pmin;   // the range removes something
            const bool leaves_tail = p1 <= pmax;                // cells beyond the range survive

            if (overlaps && leaves_tail) {
                LLAMA_LOG_WARN("%s: MSA: partial (non-suffix) removal [%d, %d) for seq %d is not supported "
                        "(block selection is anchored to cache slots) - rejected\n", __func__, p0, p1, sid);
                return false;
            }
        }
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    if (kvflash) {
        for (auto & entry : kvflash_cells) {
            auto & page = entry.second;
            bool changed = false;
            if (seq_id == -1 || seq_id == 0) {
                for (size_t i = 0; i < page.pos.size(); ++i) {
                    if (page.pos[i] >= p0 && page.pos[i] < p1) {
                        if (!changed) {
                            kvflash_unindex_page(page);
                        }
                        page.pos[i] = -1;
                        page.ext[i].reset();
                        page.shift[i] = 0;
                        changed = true;
                    }
                }
            }
            if (changed) {
                kvflash_recompute_page_bounds(page);
                kvflash_index_page(page);
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    if (kvflash) {
        if (seq_id_src != 0 || seq_id_dst != 0) {
            LLAMA_LOG_WARN("%s: KVFlash supports only sequence 0\n", __func__);
        }
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT((hparams.n_pos_per_embd() == 1 || kvflash) &&
            "seq_add() is only supported for n_pos_per_embd() == 1 outside KVFlash");
    if (kvflash && seq_id != 0) {
        LLAMA_LOG_WARN("%s: KVFlash supports only sequence 0\n", __func__);
        return;
    }

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    // A scalar cache shift is valid for M-RoPE text because its temporal,
    // height and width coordinates are identical and Qwen's fourth section is
    // empty.  Multimodal coordinates are not translation-equivalent, so reject
    // them before changing any resident or paged metadata.
    const bool mrope_text_shift = hparams.n_pos_per_embd() > 1;
    if (mrope_text_shift) {
        if (!kvflash || hparams.n_pos_per_embd() != 4 || hparams.rope_sections[3] != 0) {
            LLAMA_LOG_ERROR("%s: unsupported multi-axis KV position shift\n", __func__);
            return;
        }
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1) || !cells.seq_has(i, seq_id)) {
                continue;
            }
            const llama_pos pos = cells.pos_get(i);
            const llama_kv_cell_ext ext = cells.ext_get(i);
            const int64_t shifted = (int64_t) pos + shift;
            if (ext.x != pos || ext.y != pos ||
                shifted > std::numeric_limits<llama_pos>::max()) {
                LLAMA_LOG_ERROR("%s: KVFlash M-RoPE cache shift is only valid for text positions\n", __func__);
                return;
            }
        }
        for (const auto & entry : kvflash_cells) {
            const auto & page = entry.second;
            if (page.pos.size() != page.ext.size() || page.pos.size() != page.shift.size()) {
                LLAMA_LOG_ERROR("%s: invalid KVFlash host metadata\n", __func__);
                return;
            }
            for (size_t i = 0; i < page.pos.size(); ++i) {
                const llama_pos pos = page.pos[i];
                if (pos < p0 || pos >= p1) {
                    continue;
                }
                const int64_t shifted = (int64_t) pos + shift;
                if (page.ext[i].x != pos || page.ext[i].y != pos ||
                    shifted > std::numeric_limits<llama_pos>::max()) {
                    LLAMA_LOG_ERROR("%s: KVFlash M-RoPE cache shift is only valid for text positions\n", __func__);
                    return;
                }
            }
        }
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            const llama_kv_cell_ext ext = cells.ext_get(i);
            const bool removed = cells.pos_add(i, shift);
            if (!removed && mrope_text_shift) {
                llama_kv_cell_ext shifted_ext = ext;
                shifted_ext.x += shift;
                shifted_ext.y += shift;
                cells.ext_set(i, shifted_ext);
            }
            if (removed) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    if (kvflash) {
        for (auto & entry : kvflash_cells) {
            auto & page = entry.second;
            if (page.pos.size() != page.shift.size()) {
                LLAMA_LOG_ERROR("%s: invalid KVFlash host metadata\n", __func__);
                continue;
            }
            bool changed = false;
            for (size_t i = 0; i < page.pos.size(); ++i) {
                if (page.pos[i] < p0 || page.pos[i] >= p1) {
                    continue;
                }
                if (!changed) {
                    kvflash_unindex_page(page);
                    changed = true;
                }
                const int64_t pos = (int64_t) page.pos[i] + shift;
                const int64_t pending = (int64_t) page.shift[i] + shift;
                if (pos < 0) {
                    page.pos[i] = -1;
                    page.ext[i].reset();
                    page.shift[i] = 0;
                } else if (pos > std::numeric_limits<llama_pos>::max() ||
                           pending < std::numeric_limits<llama_pos>::min() ||
                           pending > std::numeric_limits<llama_pos>::max()) {
                    LLAMA_LOG_ERROR("%s: KVFlash position shift overflow\n", __func__);
                    page.pos[i] = -1;
                    page.ext[i].reset();
                    page.shift[i] = 0;
                } else {
                    page.pos[i] = (llama_pos) pos;
                    page.shift[i] = (llama_pos) pending;
                    if (mrope_text_shift) {
                        page.ext[i].x += shift;
                        page.ext[i].y += shift;
                    }
                }
            }
            if (changed) {
                kvflash_recompute_page_bounds(page);
                kvflash_index_page(page);
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    if (kvflash) {
        if (d != 1) {
            LLAMA_LOG_WARN("%s: KVFlash does not support KV position division\n", __func__);
        }
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    llama_pos result = cells.seq_pos_min(seq_id);
    if (seq_id == 0 && !kvflash_pos_mins.empty()) {
        const llama_pos pos = *kvflash_pos_mins.begin();
        result = result < 0 ? pos : std::min(result, pos);
    }
    return result;
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    llama_pos result = cells.seq_pos_max(seq_id);
    if (seq_id == 0 && !kvflash_pos_maxs.empty()) {
        result = std::max(result, *kvflash_pos_maxs.rbegin());
    }
    return result;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    // KVFlash maps each micro-batch immediately before its graph runs. Mapping
    // all micro-batches here could evict slots that an earlier graph has not
    // written yet when a prompt is larger than the resident pool.
    if (kvflash) {
        res.reserve(ubatches.size());
        for (const auto & ubatch : ubatches) {
            std::vector<int64_t> positions;
            positions.reserve(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                positions.push_back(ubatch.pos[i]);
            }
            if (!kvflash->can_map_positions(positions)) {
                LLAMA_LOG_ERROR("%s: KVFlash micro-batch cannot fit safely in the resident pool\n", __func__);
                return {};
            }
            res.emplace_back();
        }
        return res;
    }

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
                if (layer.k_idx_stream[ssrc]) {
                    GGML_ASSERT(layer.k_idx_stream[sdst]);
                    ggml_backend_tensor_copy(layer.k_idx_stream[ssrc], layer.k_idx_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        auto apply_shift_graph = [&]() -> bool {
            if (hparams.rope_type == LLAMA_ROPE_TYPE_NONE) {
                return true;
            }
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return false;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return false;
            }

            updated = true;
            return true;
        };

        if (kvflash) {
            while (true) {
                auto & cells = v_cells[0];
                bool resident_shifted = false;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    resident_shifted |= !cells.is_empty(i) && cells.get_shift(i) != 0;
                }
                if (resident_shifted) {
                    if (!apply_shift_graph()) {
                        return updated;
                    }
                }
                if (cells.get_has_shift()) {
                    cells.reset_shift();
                }

                std::vector<int> cold_shifted;
                for (const auto & entry : kvflash_cells) {
                    const auto & shifts = entry.second.shift;
                    if (std::any_of(shifts.begin(), shifts.end(),
                            [](llama_pos value) { return value != 0; })) {
                        cold_shifted.push_back(entry.first);
                    }
                }
                if (cold_shifted.empty()) {
                    break;
                }
                std::sort(cold_shifted.begin(), cold_shifted.end());

                common_kvflash::KvFlashState state;
                if (!kvflash->state_export(state)) {
                    LLAMA_LOG_ERROR("%s: failed to synchronize KVFlash before K-shift\n", __func__);
                    return updated;
                }
                const int n_blocks = state.pool_tokens / state.chunk_tokens;
                const int batch_pages = std::max(1,
                        n_blocks - state.sink_chunks - state.tail_window_chunks - 1);
                if ((int) cold_shifted.size() > batch_pages) {
                    cold_shifted.resize((size_t) batch_pages);
                }
                if (!kvflash->page_in_chunks(cold_shifted)) {
                    LLAMA_LOG_ERROR("%s: failed to page in KVFlash chunks for K-shift\n", __func__);
                    return updated;
                }
            }

            if (!kvflash_repack_after_shift()) {
                LLAMA_LOG_ERROR("%s: failed to repack KVFlash after K-shift\n", __func__);
                return updated;
            }
            updated = true;
        } else {
            if (!apply_shift_graph()) {
                return updated;
            }

            for (uint32_t s = 0; s < n_stream; ++s) {
                auto & cells = v_cells[s];

                cells.reset_shift();
            }
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        // MSA block selection assumes slot == logical position (append-only streams).
        if (msa_strict_slots) {
            for (uint32_t ii = 0; ii < n_tokens; ++ii) {
                const llama_pos pos = ubatch.pos[s*n_tokens + ii];

                if (pos < 0 || (uint64_t) pos >= cells.size()) {
                    LLAMA_LOG_WARN("%s: MSA: position %d is outside the cache range [0, %u)\n",
                            __func__, pos, cells.size());
                    return { };
                }

                const uint32_t idx = (uint32_t) pos;

                if (!cells.is_empty(idx)) {
                    LLAMA_LOG_WARN("%s: MSA: required slot %u is already occupied (stream %u)\n",
                            __func__, idx, seq_to_stream[seq_id]);
                    return { };
                }

                // strictly increasing positions, rules out duplicates and, for contiguous requests, is tightened to exact adjacency
                if (!res.idxs[s].empty() && (cont ? idx != res.idxs[s].back() + 1
                                                  : idx <= res.idxs[s].back())) {
                    LLAMA_LOG_WARN("%s: MSA: token positions are not %s within the ubatch\n",
                            __func__, cont ? "contiguous" : "strictly increasing");
                    return { };
                }

                res.idxs[s].push_back(idx);
            }

            continue;
        }

        // KVFlash: map logical positions to physical pool slots (may page-out).
        // find_slot is const but pager mutates residency; use mutable cast.
        if (kvflash) {
            auto * pager = const_cast<common_kvflash::KvFlashPager *>(kvflash.get());
            std::vector<int64_t> positions;
            positions.reserve(n_tokens);
            for (uint32_t ii = 0; ii < n_tokens; ++ii) {
                const llama_pos pos = ubatch.pos[s * n_tokens + ii];
                positions.push_back(pos);
            }
            if (!pager->alloc_positions(positions)) {
                LLAMA_LOG_ERROR("%s: KVFlash: cannot allocate %u logical positions (pool=%d)\n",
                        __func__, n_tokens, pager->pool_tokens());
                return {};
            }
            for (int64_t pos : positions) {
                const int phys = pager->slot_of(pos);
                if (phys < 0 || (uint32_t) phys >= cells.size()) {
                    LLAMA_LOG_ERROR("%s: KVFlash: no pool slot for pos %lld (pool=%d)\n",
                            __func__, (long long) pos, pager->pool_tokens());
                    return {};
                }
                res.idxs[s].push_back((uint32_t) phys);
            }
            continue;
        }

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (msa_strict_slots && (llama_pos) idx != ubatch.pos[i]) {
                LLAMA_LOG_ERROR("%s: MSA slot/position invariant violated: "
                        "writing pos %d into cell %u (stream %u). The indexer cache "
                        "would desync and block selection would silently corrupt. "
                        "This is a bug, please report it with reproduction steps.\n",
                        __func__, ubatch.pos[i], idx, sinfo.strm[s]);
                GGML_ABORT("MSA: slot != pos");
            }

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                // Same-pos re-apply must not trigger the [pos_min, overwritten]
                // purge. That can happen after a dry-run restore or when a
                // paged chunk is recalled before overwriting the same position.
                if (pos != ubatch.pos[i]) {
                    seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);
                }

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            // under MSA strict slots this path should be unreachable, since strict MSA placement never selects occupied cells
            GGML_ASSERT(seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1));
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        // KVFlash validates at seq_add() time that every shifted position is a
        // text-like M-RoPE coordinate.  The scalar NEOX shift graph is then
        // equivalent for Qwen's three equal, non-empty position sections.
        if (!kvflash || hparams.n_pos_per_embd() != 4 || hparams.rope_sections[3] != 0) {
            return false;
        }
    }
    // shifting would leave k_idx stale
    for (const auto & layer : layers) {
        if (layer.k_idx) {
            return false;
        }
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.get_shift(i) != 0) {
                return true;
            }
        }
    }

    if (kvflash) {
        for (const auto & entry : kvflash_cells) {
            const auto & shift = entry.second.shift;
            if (std::any_of(shift.begin(), shift.end(), [](llama_pos value) { return value != 0; })) {
                return true;
            }
        }
        return false;
    }
    return false;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

bool llama_kv_cache::can_commit_tree(
        llama_seq_id seq_id,
        const std::vector<uint32_t> & tree_cells,
        const std::vector<int32_t> & path) const {
    if (other || seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() ||
            tree_cells.empty() || path.empty() || path[0] != 0) {
        return false;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    std::vector<uint8_t> seen_cells(cells.size(), 0);
    for (uint32_t idx : tree_cells) {
        if (idx >= cells.size() || seen_cells[idx] || cells.is_empty(idx) || !cells.seq_has(idx, seq_id)) {
            return false;
        }
        seen_cells[idx] = 1;
    }

    std::vector<uint8_t> kept(tree_cells.size(), 0);
    llama_pos prev = -1;
    for (int32_t flat : path) {
        if (flat < 0 || (size_t) flat >= tree_cells.size() || kept[(size_t) flat]) {
            return false;
        }
        kept[(size_t) flat] = 1;
        const llama_pos pos = cells.pos_get(tree_cells[(size_t) flat]);
        if (prev >= 0 && pos != prev + 1) {
            return false;
        }
        prev = pos;
    }

    return true;
}

bool llama_kv_cache::commit_tree(
        llama_seq_id seq_id,
        const std::vector<uint32_t> & tree_cells,
        const std::vector<int32_t> & path) {
    if (!can_commit_tree(seq_id, tree_cells, path)) {
        return false;
    }

    const uint32_t stream = seq_to_stream[seq_id];
    auto & cells = v_cells[stream];
    auto & head = v_heads[stream];
    std::vector<uint8_t> kept(tree_cells.size(), 0);
    for (int32_t flat : path) {
        kept[(size_t) flat] = 1;
    }

    uint32_t new_head = cells.size();
    for (size_t i = 0; i < tree_cells.size(); ++i) {
        if (kept[i]) {
            continue;
        }
        const uint32_t idx = tree_cells[i];
        if (cells.seq_rm(idx, seq_id)) {
            new_head = std::min(new_head, idx);
        }
    }
    if (new_head < cells.size() && new_head < head) {
        head = new_head;
    }

    return true;
}

ggml_tensor * llama_kv_cache::get_v_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].v;
}

bool llama_kv_cache::init_kvflash(uint32_t pool_tokens, uint32_t max_context_tokens, int chunk_tokens) {
    if (pool_tokens == 0 || get_size() != pool_tokens) {
        LLAMA_LOG_ERROR("%s: pool_tokens=%u must match kv size=%u\n",
                __func__, pool_tokens, get_size());
        return false;
    }
    const uint32_t n_pos = hparams.n_pos_per_embd();
    if (n_stream != 1 || n_seq_max != 1 || (n_pos != 1 && n_pos != 4)) {
        LLAMA_LOG_ERROR("%s: KVFlash requires exactly one sequence, one stream, and 1D or M-RoPE positions\n", __func__);
        return false;
    }

    common_kvflash::KvFlashConfig cfg;
    cfg.chunk_tokens       = chunk_tokens;
    cfg.pool_tokens        = (int) pool_tokens;
    cfg.max_context_tokens = max_context_tokens;
    cfg.sink_chunks        = 1;
    // ~128-token protected tail (2 chunks @ 64). Smaller tail leaves more
    // pool for pageable history (important on 6 GiB with pool 512–1024).
    cfg.tail_window_chunks = std::max(1, 128 / chunk_tokens);
    cfg.zero_freed_blocks  = false;

    // Collect per-layer K/V storages for bit-exact host paging.
    std::vector<ggml_tensor *> ks;
    std::vector<ggml_tensor *> vs;
    for (const auto & layer : layers) {
        if (layer.k && layer.v) {
            ks.push_back(layer.k);
            vs.push_back(layer.v);
        }
    }

    const bool have_data = ks.size() == layers.size() && !ks.empty() &&
            std::all_of(ks.begin(), ks.end(), [](const ggml_tensor * t) {
                return t && t->data && t->buffer;
            }) &&
            std::all_of(vs.begin(), vs.end(), [](const ggml_tensor * t) {
                return t && t->data && t->buffer;
            });
    if (!have_data && !hparams.no_alloc) {
        LLAMA_LOG_ERROR("%s: KVFlash requires allocated K/V storage for every attention layer\n", __func__);
        return false;
    }

    auto pager = std::make_unique<common_kvflash::KvFlashPager>();
    if (!pager->attach(cfg,
            have_data ? ks : std::vector<ggml_tensor *>{},
            have_data ? vs : std::vector<ggml_tensor *>{})) {
        LLAMA_LOG_ERROR("%s: KvFlashPager::attach failed (pool=%u chunk=%d)\n",
                __func__, pool_tokens, chunk_tokens);
        return false;
    }

    pager->on_block_evicted = [this, chunk_tokens](int block) {
        this->kvflash_clear_block(block, chunk_tokens);
    };
    pager->on_block_paged_out = [this, chunk_tokens](int chunk, int block) {
        return this->kvflash_page_out_cells(chunk, block, chunk_tokens);
    };
    pager->on_block_paged_in = [this, chunk_tokens](int chunk, int block) {
        return this->kvflash_page_in_cells(chunk, block, chunk_tokens);
    };

    kvflash = std::move(pager);

    // Zero the whole pool so free slots contribute ~0 if ever attended.
    clear(/*data=*/true);

    LLAMA_LOG_INFO("%s: KVFlash ON pool=%u chunk=%d layers=%zu tail_chunks=%d (%s)\n",
            __func__, pool_tokens, chunk_tokens, layers.size(),
            cfg.tail_window_chunks, have_data ? "tensor paging" : "allocation sizing");
    return true;
}

common_kvflash::KvFlashPager * llama_kv_cache::get_kvflash() const {
    return kvflash.get();
}

bool llama_kv_cache::has_kvflash() const {
    return kvflash != nullptr;
}

void llama_kv_cache::kvflash_clear_block(int block, int chunk_tokens) {
    if (n_stream < 1 || block < 0) {
        return;
    }
    auto & cells = v_cells[0];
    const uint32_t i0 = (uint32_t) block * (uint32_t) chunk_tokens;
    const uint32_t i1 = i0 + (uint32_t) chunk_tokens;
    for (uint32_t i = i0; i < i1 && i < cells.size(); ++i) {
        if (!cells.is_empty(i)) {
            cells.rm(i);
        }
    }
}

void llama_kv_cache::kvflash_index_page(const kvflash_cell_page & page) {
    if (page.pos_min >= 0) {
        kvflash_pos_mins.insert(page.pos_min);
    }
    if (page.pos_max >= 0) {
        kvflash_pos_maxs.insert(page.pos_max);
    }
}

void llama_kv_cache::kvflash_unindex_page(const kvflash_cell_page & page) {
    if (page.pos_min >= 0) {
        const auto it = kvflash_pos_mins.find(page.pos_min);
        GGML_ASSERT(it != kvflash_pos_mins.end());
        kvflash_pos_mins.erase(it);
    }
    if (page.pos_max >= 0) {
        const auto it = kvflash_pos_maxs.find(page.pos_max);
        GGML_ASSERT(it != kvflash_pos_maxs.end());
        kvflash_pos_maxs.erase(it);
    }
}

void llama_kv_cache::kvflash_recompute_page_bounds(kvflash_cell_page & page) const {
    page.pos_min = -1;
    page.pos_max = -1;
    for (llama_pos pos : page.pos) {
        if (pos < 0) {
            continue;
        }
        page.pos_min = page.pos_min < 0 ? pos : std::min(page.pos_min, pos);
        page.pos_max = std::max(page.pos_max, pos);
    }
}

bool llama_kv_cache::kvflash_capture_page(int block, int chunk_tokens, kvflash_cell_page & page) const {
    if (n_stream != 1 || block < 0 || chunk_tokens <= 0) {
        return false;
    }
    const auto & cells = v_cells[0];
    const uint32_t first = (uint32_t) block * (uint32_t) chunk_tokens;
    if (first > cells.size() || (uint32_t) chunk_tokens > cells.size() - first) {
        return false;
    }
    page = {};
    page.pos.assign((size_t) chunk_tokens, -1);
    page.ext.resize((size_t) chunk_tokens);
    page.shift.assign((size_t) chunk_tokens, 0);
    for (int i = 0; i < chunk_tokens; ++i) {
        const uint32_t cell = first + (uint32_t) i;
        if (cells.is_empty(cell)) {
            continue;
        }
        if (cells.seq_count(cell) != 1 || !cells.seq_has(cell, 0)) {
            return false;
        }
        const llama_pos pos = cells.pos_get(cell);
        page.pos[(size_t) i] = pos;
        page.ext[(size_t) i] = cells.ext_get(cell);
        page.shift[(size_t) i] = cells.get_shift(cell);
    }
    kvflash_recompute_page_bounds(page);
    return true;
}

bool llama_kv_cache::kvflash_restore_page(int block, int chunk_tokens, const kvflash_cell_page & page) {
    if (n_stream != 1 || block < 0 || chunk_tokens <= 0 ||
        page.pos.size() != (size_t) chunk_tokens ||
        page.ext.size() != (size_t) chunk_tokens ||
        page.shift.size() != (size_t) chunk_tokens) {
        return false;
    }
    auto & cells = v_cells[0];
    const uint32_t first = (uint32_t) block * (uint32_t) chunk_tokens;
    if (first > cells.size() || (uint32_t) chunk_tokens > cells.size() - first) {
        return false;
    }
    for (int i = 0; i < chunk_tokens; ++i) {
        if (!cells.is_empty(first + (uint32_t) i)) {
            return false;
        }
    }
    for (int i = 0; i < chunk_tokens; ++i) {
        const llama_pos pos = page.pos[(size_t) i];
        const llama_pos shift = page.shift[(size_t) i];
        if (pos < 0) {
            if (shift != 0) {
                return false;
            }
            continue;
        }
        const int64_t original_pos = (int64_t) pos - (int64_t) shift;
        if (original_pos < 0 || original_pos > std::numeric_limits<llama_pos>::max()) {
            return false;
        }
        const uint32_t cell = first + (uint32_t) i;
        cells.pos_set(cell, (llama_pos) original_pos);
        cells.ext_set(cell, page.ext[(size_t) i]);
        cells.seq_add(cell, 0);
        if (shift != 0 && cells.pos_add(cell, shift)) {
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::kvflash_page_out_cells(int chunk, int block, int chunk_tokens) {
    if (chunk < 0) {
        return false;
    }
    kvflash_cell_page page;
    if (!kvflash_capture_page(block, chunk_tokens, page)) {
        return false;
    }
    const auto old = kvflash_cells.find(chunk);
    if (old != kvflash_cells.end()) {
        kvflash_unindex_page(old->second);
    }
    auto result = kvflash_cells.insert_or_assign(chunk, std::move(page));
    kvflash_index_page(result.first->second);
    return true;
}

bool llama_kv_cache::kvflash_page_in_cells(int chunk, int block, int chunk_tokens) {
    if (n_stream != 1 || chunk < 0 || block < 0 || chunk_tokens <= 0) {
        return false;
    }
    auto it = kvflash_cells.find(chunk);
    if (it == kvflash_cells.end()) {
        return false;
    }
    const kvflash_cell_page & page = it->second;
    if (page.pos.size() != (size_t) chunk_tokens ||
        page.ext.size() != (size_t) chunk_tokens ||
        page.shift.size() != (size_t) chunk_tokens) {
        return false;
    }
    if (!kvflash_restore_page(block, chunk_tokens, page)) {
        return false;
    }
    kvflash_unindex_page(page);
    kvflash_cells.erase(it);
    return true;
}

bool llama_kv_cache::kvflash_repack_after_shift() {
    if (!kvflash || n_stream != 1) {
        return false;
    }

    common_kvflash::KvFlashState old_state;
    if (!kvflash->state_export(old_state) || old_state.chunk_tokens <= 0 ||
        old_state.pool_tokens <= 0 || old_state.chunk_bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }

    struct source_page {
        common_kvflash::KvFlashStateChunk state;
        kvflash_cell_page page;
    };
    std::vector<source_page> sources;
    sources.reserve(old_state.chunks.size());
    for (const auto & chunk : old_state.chunks) {
        source_page source;
        source.state = chunk;
        if (chunk.block >= 0) {
            if (!kvflash_capture_page(chunk.block, old_state.chunk_tokens, source.page)) {
                return false;
            }
        } else {
            const auto it = kvflash_cells.find(chunk.chunk);
            if (it == kvflash_cells.end()) {
                return false;
            }
            source.page = it->second;
        }
        if (source.page.pos.size() != (size_t) old_state.chunk_tokens ||
            source.page.ext.size() != (size_t) old_state.chunk_tokens ||
            source.page.shift.size() != (size_t) old_state.chunk_tokens ||
            std::any_of(source.page.shift.begin(), source.page.shift.end(),
                [](llama_pos value) { return value != 0; })) {
            return false;
        }
        sources.push_back(std::move(source));
    }

    struct target_page {
        kvflash_cell_page page;
        uint64_t last_use = 0;
        bool resident_source = false;
        int old_block = -1;
        size_t payload_offset = 0;
    };
    std::map<int, target_page> targets;

    for (const auto & source : sources) {
        for (int row = 0; row < old_state.chunk_tokens; ++row) {
            const llama_pos pos = source.page.pos[(size_t) row];
            if (pos < 0) {
                continue;
            }
            if ((old_state.max_context_tokens > 0 && pos >= old_state.max_context_tokens) ||
                pos > std::numeric_limits<int>::max()) {
                return false;
            }
            const int chunk = pos / old_state.chunk_tokens;
            const int target_row = pos % old_state.chunk_tokens;
            auto result = targets.try_emplace(chunk);
            target_page & target = result.first->second;
            if (result.second) {
                target.page.pos.assign((size_t) old_state.chunk_tokens, -1);
                target.page.ext.resize((size_t) old_state.chunk_tokens);
                target.page.shift.assign((size_t) old_state.chunk_tokens, 0);
            }
            if (target.page.pos[(size_t) target_row] >= 0) {
                LLAMA_LOG_ERROR("%s: duplicate logical KVFlash position %d after shift\n", __func__, pos);
                return false;
            }
            target.page.pos[(size_t) target_row] = pos;
            target.page.ext[(size_t) target_row] = source.page.ext[(size_t) row];
            target.last_use = std::max(target.last_use, source.state.last_use);
            target.resident_source |= source.state.block >= 0;
            if (source.state.chunk == chunk && source.state.block >= 0) {
                target.old_block = source.state.block;
            }
        }
    }

    if (targets.empty()) {
        clear(false);
        return true;
    }

    size_t segment_offset = 0;
    std::vector<size_t> row_bytes;
    std::vector<size_t> segment_offsets;
    row_bytes.reserve(old_state.tensors.size());
    segment_offsets.reserve(old_state.tensors.size());
    for (const auto & tensor : old_state.tensors) {
        if (tensor.segment_bytes % (uint64_t) old_state.chunk_tokens != 0 ||
            tensor.segment_bytes > (uint64_t) std::numeric_limits<size_t>::max() ||
            segment_offset > (size_t) old_state.chunk_bytes ||
            (size_t) tensor.segment_bytes > (size_t) old_state.chunk_bytes - segment_offset) {
            return false;
        }
        segment_offsets.push_back(segment_offset);
        row_bytes.push_back((size_t) tensor.segment_bytes / (size_t) old_state.chunk_tokens);
        segment_offset += (size_t) tensor.segment_bytes;
    }
    if (segment_offset != (size_t) old_state.chunk_bytes ||
        targets.size() > std::numeric_limits<size_t>::max() /
            std::max<size_t>(1, (size_t) old_state.chunk_bytes)) {
        return false;
    }

    const size_t payload_bytes = targets.size() * (size_t) old_state.chunk_bytes;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> scratch;
    try {
        payload.assign(payload_bytes, 0);
        scratch.resize((size_t) old_state.chunk_bytes);
    } catch (const std::bad_alloc &) {
        LLAMA_LOG_ERROR("%s: cannot allocate %zu bytes for KVFlash repack\n", __func__, payload_bytes);
        return false;
    }

    size_t target_offset = 0;
    for (auto & entry : targets) {
        entry.second.payload_offset = target_offset;
        kvflash_recompute_page_bounds(entry.second.page);
        target_offset += (size_t) old_state.chunk_bytes;
    }

    std::vector<common_kvflash::KvFlashStateSpan> spans;
    for (const auto & source : sources) {
        if (!kvflash->state_spans(source.state.chunk, spans)) {
            return false;
        }
        size_t copied = 0;
        for (const auto & span : spans) {
            if (span.payload_offset != copied || copied > scratch.size() ||
                span.size > scratch.size() - copied) {
                return false;
            }
            if (span.tensor) {
                ggml_backend_tensor_get(span.tensor, scratch.data() + copied,
                        span.tensor_offset, span.size);
            } else if (span.host) {
                std::memcpy(scratch.data() + copied, span.host, span.size);
            } else if (span.size != 0) {
                return false;
            }
            copied += span.size;
        }
        if (copied != scratch.size()) {
            return false;
        }

        for (int row = 0; row < old_state.chunk_tokens; ++row) {
            const llama_pos pos = source.page.pos[(size_t) row];
            if (pos < 0) {
                continue;
            }
            const int chunk = pos / old_state.chunk_tokens;
            const int target_row = pos % old_state.chunk_tokens;
            const auto target_it = targets.find(chunk);
            if (target_it == targets.end()) {
                return false;
            }
            uint8_t * dst_page = payload.data() + target_it->second.payload_offset;
            for (size_t tensor = 0; tensor < row_bytes.size(); ++tensor) {
                const size_t row_size = row_bytes[tensor];
                const size_t segment = segment_offsets[tensor];
                std::memcpy(
                    dst_page + segment + (size_t) target_row * row_size,
                    scratch.data() + segment + (size_t) row * row_size,
                    row_size);
            }
        }
    }

    common_kvflash::KvFlashState new_state = old_state;
    new_state.chunks.clear();
    new_state.cur_chunk = targets.rbegin()->first;

    const int n_blocks = old_state.pool_tokens / old_state.chunk_tokens;
    std::vector<int> candidates;
    candidates.reserve(targets.size());
    for (const auto & entry : targets) {
        candidates.push_back(entry.first);
    }
    const int tail_first = std::max(0, new_state.cur_chunk - old_state.tail_window_chunks);
    std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const target_page & pa = targets.at(a);
        const target_page & pb = targets.at(b);
        const bool protected_a = a < old_state.sink_chunks || a >= tail_first;
        const bool protected_b = b < old_state.sink_chunks || b >= tail_first;
        if (protected_a != protected_b) {
            return protected_a > protected_b;
        }
        if (pa.resident_source != pb.resident_source) {
            return pa.resident_source > pb.resident_source;
        }
        if (pa.last_use != pb.last_use) {
            return pa.last_use > pb.last_use;
        }
        return a < b;
    });
    if ((int) candidates.size() > n_blocks) {
        candidates.resize((size_t) n_blocks);
    }
    std::set<int> resident_chunks(candidates.begin(), candidates.end());
    std::vector<uint8_t> used_blocks((size_t) n_blocks, 0);
    std::map<int, int> block_for;

    for (int chunk : resident_chunks) {
        const int block = targets.at(chunk).old_block;
        if (block >= 0 && block < n_blocks && !used_blocks[(size_t) block]) {
            block_for[chunk] = block;
            used_blocks[(size_t) block] = 1;
        }
    }
    for (int chunk : resident_chunks) {
        if (block_for.count(chunk) == 0 && chunk >= 0 && chunk < n_blocks &&
            !used_blocks[(size_t) chunk]) {
            block_for[chunk] = chunk;
            used_blocks[(size_t) chunk] = 1;
        }
    }
    int next_block = 0;
    for (int chunk : resident_chunks) {
        if (block_for.count(chunk) != 0) {
            continue;
        }
        while (next_block < n_blocks && used_blocks[(size_t) next_block]) {
            ++next_block;
        }
        if (next_block >= n_blocks) {
            return false;
        }
        block_for[chunk] = next_block;
        used_blocks[(size_t) next_block] = 1;
    }

    for (const auto & entry : targets) {
        const int chunk = entry.first;
        const auto block_it = block_for.find(chunk);
        new_state.chunks.push_back({
            chunk,
            block_it == block_for.end() ? -1 : block_it->second,
            entry.second.last_use,
        });
    }

    if (!kvflash->state_import(new_state)) {
        clear(true);
        return false;
    }

    v_cells[0].reset();
    v_heads[0] = 0;
    kvflash_cells.clear();
    kvflash_pos_mins.clear();
    kvflash_pos_maxs.clear();

    for (const auto & chunk : new_state.chunks) {
        const auto target_it = targets.find(chunk.chunk);
        if (target_it == targets.end()) {
            clear(true);
            return false;
        }
        const target_page & target = target_it->second;
        if (chunk.block >= 0) {
            if (!kvflash_restore_page(chunk.block, new_state.chunk_tokens, target.page)) {
                clear(true);
                return false;
            }
        } else {
            auto result = kvflash_cells.emplace(chunk.chunk, target.page);
            if (!result.second) {
                clear(true);
                return false;
            }
            kvflash_index_page(result.first->second);
        }

        if (!kvflash->state_spans(chunk.chunk, spans)) {
            clear(true);
            return false;
        }
        size_t copied = 0;
        const uint8_t * src_page = payload.data() + target.payload_offset;
        for (const auto & span : spans) {
            if (span.payload_offset != copied || copied > (size_t) new_state.chunk_bytes ||
                span.size > (size_t) new_state.chunk_bytes - copied) {
                clear(true);
                return false;
            }
            if (span.tensor) {
                ggml_backend_tensor_set(span.tensor, src_page + copied,
                        span.tensor_offset, span.size);
            } else if (span.host) {
                std::memcpy(span.host, src_page + copied, span.size);
            } else if (span.size != 0) {
                clear(true);
                return false;
            }
            copied += span.size;
        }
        if (copied != new_state.chunk_bytes) {
            clear(true);
            return false;
        }
    }

    return true;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    //
    // KVFlash: used_max_p1 covers every occupied physical cell (including after
    // relocation). Empty holes below used_max are mask-dropped via is_empty.
    // After any page_out, also ensure we never shrink below the pool when the
    // pager reports non-identity layout (holes can sit below a high used_max).
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    if (kvflash && !kvflash->is_identity() && kvflash->stats().page_outs > 0) {
        result = std::max(result, std::max(n_pad_cur, GGML_PAD((uint32_t) kvflash->pool_tokens(), n_pad_cur)));
        result = std::min(result, (uint32_t) kvflash->pool_tokens());
        // pad again within pool
        result = std::max(n_pad_cur, GGML_PAD(result, n_pad_cur));
        result = std::min(result, (uint32_t) get_size());
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_k_idx(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    auto * k_idx = layers[ikv].k_idx;
    GGML_ASSERT(k_idx);

    const uint64_t kv_size = get_size();
    const int64_t  n_idx   = k_idx->ne[0];                 // 128
    const uint32_t ns      = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k_idx,
            n_idx, 1, n_kv, ns,
            ggml_row_size(k_idx->type, n_idx),             // nb1 (single head)
            ggml_row_size(k_idx->type, n_idx),             // nb2 (per cell)
            ggml_row_size(k_idx->type, n_idx*kv_size),     // nb3 (per stream)
            ggml_row_size(k_idx->type, n_idx*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    const uint32_t flags = q4_weighted_scale_k && k->type == GGML_TYPE_Q4_0
        ? GGML_SET_ROWS_FLAG_Q4_0_WEIGHTED_SCALE
        : GGML_SET_ROWS_FLAG_NONE;
    return ggml_set_rows_ext(ctx, k, k_cur, k_idxs, flags);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        const uint32_t flags = q4_weighted_scale_v && v->type == GGML_TYPE_Q4_0
            ? GGML_SET_ROWS_FLAG_Q4_0_WEIGHTED_SCALE
            : GGML_SET_ROWS_FLAG_NONE;
        return ggml_set_rows_ext(ctx, v, v_cur, v_idxs, flags);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);
    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * k_idx = layers[ikv].k_idx;
    GGML_ASSERT(k_idx && "cpy_k_idx on a layer with no indexer cache");

    const int64_t n_embd_head = k_idx_cur->ne[0];          // 128
    const int64_t n_head      = k_idx_cur->ne[1];          // 1
    const int64_t n_tokens    = k_idx_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;        // 128

    GGML_ASSERT(ggml_row_size(k_idx_cur->type, n_embd_head) == k_idx_cur->nb[1]);
    k_idx_cur = ggml_view_2d(ctx, k_idx_cur, n_embd_gqa, n_tokens, k_idx_cur->nb[2], 0);

    const int64_t n_stream = k_idx->ne[2];
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        k_idx = ggml_reshape_2d(ctx, k_idx, n_embd_gqa, kv_size*n_stream);
    }
    return ggml_set_rows(ctx, k_idx, k_idx_cur, k_idxs);   // same k_idxs as the K store
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

size_t llama_kv_cache::size_k_idx_bytes() const {
    size_t size_k_idx_bytes = 0;

    for (const auto & layer : layers) {
        if (layer.k_idx) {
            size_k_idx_bytes += ggml_nbytes(layer.k_idx);
        }
    }

    return size_k_idx_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * rows,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        // CUDA CPY supports requantization to the standard Q4/Q8 formats, but
        // Turbo4 is intentionally quantized by SET_ROWS. Reuse that optimized
        // kernel with identity row indices for in-place K-cache rotation.
        tmp = cur->type == GGML_TYPE_TURBO4_K ?
                ggml_set_rows(ctx, cur, tmp, rows) :
                ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]
    ggml_tensor * k_idxs = nullptr; // I32 [n_shift], KVFlash sparse path

    std::vector<int32_t> sparse_idxs;
    std::vector<int32_t> sparse_shifts;

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    // identity head-row indices used to requantize F32 -> Turbo4 via SET_ROWS
    std::vector<ggml_tensor *> k_rows;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift && !k_idxs) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_idxs) {
        GGML_ASSERT(k_shift && sparse_idxs.size() == sparse_shifts.size());
        GGML_ASSERT((int64_t) sparse_idxs.size() == k_idxs->ne[0]);
        GGML_ASSERT(ggml_backend_buffer_is_host(k_idxs->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(k_shift->buffer));
        std::memcpy(k_idxs->data, sparse_idxs.data(), sparse_idxs.size() * sizeof(sparse_idxs[0]));
        std::memcpy(k_shift->data, sparse_shifts.data(), sparse_shifts.size() * sizeof(sparse_shifts[0]));
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }

    for (ggml_tensor * rows : k_rows) {
        GGML_ASSERT(ggml_backend_buffer_is_host(rows->buffer));
        int32_t * data = (int32_t *) rows->data;
        for (int64_t i = 0; i < rows->ne[0]; ++i) {
            data[i] = (int32_t) i;
        }
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    const bool has_turbo4 = std::any_of(layers.begin(), layers.end(), [](const auto & layer) {
        return layer.k && layer.k->type == GGML_TYPE_TURBO4_K;
    });
    const bool sparse_shift = n_stream == 1 && (kvflash || has_turbo4);
    if (sparse_shift) {
        const auto & cells = v_cells[0];
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.get_shift(i) != 0) {
                inp->sparse_idxs.push_back((int32_t) i);
                inp->sparse_shifts.push_back((int32_t) cells.get_shift(i));
            }
        }
        GGML_ASSERT(!inp->sparse_idxs.empty());
        inp->k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) inp->sparse_idxs.size());
        inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) inp->sparse_shifts.size());
        ggml_set_input(inp->k_idxs);
        ggml_set_input(inp->k_shift);
    } else {
        inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
        ggml_set_input(inp->k_shift);
    }

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    if (sparse_shift) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_embd_head_k = hparams.n_embd_head_k(il);
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            GGML_ASSERT(n_embd_head_k * n_head_kv == n_embd_k_gqa);
            GGML_ASSERT(layer.k->ne[0] == n_embd_k_gqa);

            ggml_tensor * cur = ggml_get_rows(ctx, layer.k, inp->k_idxs);

            cur = ggml_reshape_3d(ctx, cur,
                    n_embd_head_k, n_head_kv, (int64_t) inp->sparse_idxs.size());
            if (inp->k_rot) {
                cur = llama_mul_mat_hadamard(ctx, cur, inp->k_rot);
            }

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            const float freq_base_l = model.get_rope_freq_base(cparams, il);
            const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
            cur = build_rope_shift(cparams, ctx, cur, inp->k_shift, nullptr, nullptr,
                    rope_factors, freq_base_l, freq_scale_l, il);

            if (inp->k_rot) {
                cur = llama_mul_mat_hadamard(ctx, cur, inp->k_rot);
            }
            cur = ggml_reshape_2d(ctx, cur, n_embd_k_gqa, (int64_t) inp->sparse_idxs.size());

            const uint32_t flags = q4_weighted_scale_k && layer.k->type == GGML_TYPE_Q4_0 ?
                    GGML_SET_ROWS_FLAG_Q4_0_WEIGHTED_SCALE : GGML_SET_ROWS_FLAG_NONE;
            ggml_build_forward_expand(gf,
                    ggml_set_rows_ext(ctx, layer.k, cur, inp->k_idxs, flags));
        }

        res->add_input(std::move(inp));
        return gf;
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * rows = nullptr;
        if (layer.k->type == GGML_TYPE_TURBO4_K) {
            rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_head_kv);
            ggml_set_input(rows);
            inp->k_rows.push_back(rows);
        }

        // Standard block quantizers can update only the RoPE prefix. Turbo4's
        // 128-value transform couples the entire quantization block, so its
        // shift must dequantize and requantize the complete key head while
        // ggml_rope_ext still rotates only the first n_rot values.
        const int64_t n_embd_shift = layer.k->type == GGML_TYPE_TURBO4_K ?
                n_embd_head_k : n_rot;

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_embd_shift, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rows,
                rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::kvflash_state_write(
        llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    if (!kvflash || n_stream != 1 || (seq_id != -1 && seq_id != 0)) {
        throw std::runtime_error("invalid KVFlash state serialization request");
    }

    common_kvflash::KvFlashState state;
    if (!kvflash->state_export(state)) {
        throw std::runtime_error("failed to synchronize KVFlash state");
    }

    std::vector<kvflash_cell_page> pages;
    pages.reserve(state.chunks.size());
    for (const auto & chunk : state.chunks) {
        kvflash_cell_page page;
        if (chunk.block >= 0) {
            if (!kvflash_capture_page(chunk.block, state.chunk_tokens, page)) {
                throw std::runtime_error("failed to capture resident KVFlash metadata");
            }
        } else {
            const auto it = kvflash_cells.find(chunk.chunk);
            if (it == kvflash_cells.end()) {
                throw std::runtime_error("missing host KVFlash metadata");
            }
            page = it->second;
        }
        if (page.pos.size() != (size_t) state.chunk_tokens ||
            page.ext.size() != (size_t) state.chunk_tokens ||
            page.shift.size() != (size_t) state.chunk_tokens) {
            throw std::runtime_error("invalid KVFlash metadata page");
        }
        pages.push_back(std::move(page));
    }

    constexpr uint32_t magic = 0x5346564b; // "KVFS", little-endian
    constexpr uint32_t version = 1;
    const uint32_t tensor_count = (uint32_t) state.tensors.size();
    const uint32_t chunk_count = (uint32_t) state.chunks.size();

    io.write(&magic,                    sizeof(magic));
    io.write(&version,                  sizeof(version));
    io.write(&state.chunk_tokens,       sizeof(state.chunk_tokens));
    io.write(&state.pool_tokens,        sizeof(state.pool_tokens));
    io.write(&state.max_context_tokens, sizeof(state.max_context_tokens));
    io.write(&state.sink_chunks,        sizeof(state.sink_chunks));
    io.write(&state.tail_window_chunks, sizeof(state.tail_window_chunks));
    io.write(&state.cur_chunk,          sizeof(state.cur_chunk));
    io.write(&state.clock,              sizeof(state.clock));
    io.write(&state.chunk_bytes,        sizeof(state.chunk_bytes));
    io.write(&tensor_count,             sizeof(tensor_count));
    io.write(&chunk_count,              sizeof(chunk_count));

    for (const auto & tensor : state.tensors) {
        io.write(&tensor.type,          sizeof(tensor.type));
        io.write(&tensor.segment_bytes, sizeof(tensor.segment_bytes));
    }
    for (const auto & chunk : state.chunks) {
        io.write(&chunk.chunk,    sizeof(chunk.chunk));
        io.write(&chunk.block,    sizeof(chunk.block));
        io.write(&chunk.last_use, sizeof(chunk.last_use));
    }
    for (const auto & page : pages) {
        io.write(page.pos.data(),   page.pos.size()   * sizeof(page.pos[0]));
        io.write(page.ext.data(),   page.ext.size()   * sizeof(page.ext[0]));
        io.write(page.shift.data(), page.shift.size() * sizeof(page.shift[0]));
    }

    std::vector<common_kvflash::KvFlashStateSpan> spans;
    for (const auto & chunk : state.chunks) {
        if (!kvflash->state_spans(chunk.chunk, spans)) {
            throw std::runtime_error("failed to locate KVFlash state payload");
        }
        size_t payload_size = 0;
        for (const auto & span : spans) {
            if (span.payload_offset != payload_size || payload_size > (size_t) state.chunk_bytes ||
                span.size > (size_t) state.chunk_bytes - payload_size) {
                throw std::runtime_error("invalid KVFlash state payload layout");
            }
            if (span.tensor) {
                io.write_tensor(span.tensor, span.tensor_offset, span.size);
            } else if (span.host) {
                io.write(span.host, span.size);
            } else if (span.size != 0) {
                throw std::runtime_error("missing KVFlash state payload storage");
            }
            payload_size += span.size;
        }
        if (payload_size != state.chunk_bytes) {
            throw std::runtime_error("incomplete KVFlash state payload");
        }
    }
}

void llama_kv_cache::kvflash_state_read(
        llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    if (!kvflash || n_stream != 1 || (seq_id != -1 && seq_id != 0)) {
        throw std::runtime_error("invalid KVFlash state restoration request");
    }

    common_kvflash::KvFlashState expected;
    if (!kvflash->state_export(expected)) {
        throw std::runtime_error("failed to synchronize KVFlash before restoration");
    }

    constexpr uint32_t expected_magic = 0x5346564b;
    constexpr uint32_t expected_version = 1;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t tensor_count = 0;
    uint32_t chunk_count = 0;
    common_kvflash::KvFlashState state;

    io.read(&magic,                    sizeof(magic));
    io.read(&version,                  sizeof(version));
    io.read(&state.chunk_tokens,       sizeof(state.chunk_tokens));
    io.read(&state.pool_tokens,        sizeof(state.pool_tokens));
    io.read(&state.max_context_tokens, sizeof(state.max_context_tokens));
    io.read(&state.sink_chunks,        sizeof(state.sink_chunks));
    io.read(&state.tail_window_chunks, sizeof(state.tail_window_chunks));
    io.read(&state.cur_chunk,          sizeof(state.cur_chunk));
    io.read(&state.clock,              sizeof(state.clock));
    io.read(&state.chunk_bytes,        sizeof(state.chunk_bytes));
    io.read(&tensor_count,             sizeof(tensor_count));
    io.read(&chunk_count,              sizeof(chunk_count));

    const uint64_t max_chunks = state.chunk_tokens > 0 && state.max_context_tokens > 0 ?
        ((uint64_t) state.max_context_tokens + (uint64_t) state.chunk_tokens - 1) /
            (uint64_t) state.chunk_tokens : 0;
    if (magic != expected_magic || version != expected_version ||
        state.chunk_tokens != expected.chunk_tokens ||
        state.pool_tokens != expected.pool_tokens ||
        state.max_context_tokens != expected.max_context_tokens ||
        state.sink_chunks != expected.sink_chunks ||
        state.tail_window_chunks != expected.tail_window_chunks ||
        state.chunk_bytes != expected.chunk_bytes ||
        tensor_count != expected.tensors.size() ||
        (max_chunks > 0 && chunk_count > max_chunks)) {
        throw std::runtime_error("incompatible KVFlash state");
    }

    state.tensors.resize(tensor_count);
    for (auto & tensor : state.tensors) {
        io.read(&tensor.type,          sizeof(tensor.type));
        io.read(&tensor.segment_bytes, sizeof(tensor.segment_bytes));
    }
    state.chunks.resize(chunk_count);
    for (auto & chunk : state.chunks) {
        io.read(&chunk.chunk,    sizeof(chunk.chunk));
        io.read(&chunk.block,    sizeof(chunk.block));
        io.read(&chunk.last_use, sizeof(chunk.last_use));
    }

    std::vector<kvflash_cell_page> pages(chunk_count);
    for (auto & page : pages) {
        page.pos.resize((size_t) state.chunk_tokens);
        page.ext.resize((size_t) state.chunk_tokens);
        page.shift.resize((size_t) state.chunk_tokens);
        io.read(page.pos.data(),   page.pos.size()   * sizeof(page.pos[0]));
        io.read(page.ext.data(),   page.ext.size()   * sizeof(page.ext[0]));
        io.read(page.shift.data(), page.shift.size() * sizeof(page.shift[0]));
        for (size_t i = 0; i < page.pos.size(); ++i) {
            const llama_pos pos = page.pos[i];
            const llama_pos shift = page.shift[i];
            const int64_t original_pos = (int64_t) pos - (int64_t) shift;
            if (pos < -1 || (pos == -1 && shift != 0) ||
                (state.max_context_tokens > 0 && pos >= state.max_context_tokens) ||
                (pos >= 0 && (original_pos < 0 ||
                    original_pos > std::numeric_limits<llama_pos>::max() ||
                    (state.max_context_tokens > 0 && original_pos >= state.max_context_tokens)))) {
                throw std::runtime_error("invalid KVFlash cell metadata");
            }
        }
        kvflash_recompute_page_bounds(page);
    }

    if (!kvflash->state_import(state)) {
        clear(true);
        throw std::runtime_error("failed to import KVFlash pager state");
    }

    try {
        v_cells[0].reset();
        v_heads[0] = 0;
        kvflash_cells.clear();
        kvflash_pos_mins.clear();
        kvflash_pos_maxs.clear();

        for (size_t i = 0; i < state.chunks.size(); ++i) {
            const auto & chunk = state.chunks[i];
            const auto & page = pages[i];
            if (chunk.block >= 0) {
                if (!kvflash_restore_page(chunk.block, state.chunk_tokens, page)) {
                    throw std::runtime_error("failed to restore resident KVFlash metadata");
                }
            } else {
                auto result = kvflash_cells.emplace(chunk.chunk, page);
                if (!result.second) {
                    throw std::runtime_error("duplicate KVFlash host metadata");
                }
                kvflash_index_page(result.first->second);
            }
        }

        std::vector<common_kvflash::KvFlashStateSpan> spans;
        for (const auto & chunk : state.chunks) {
            if (!kvflash->state_spans(chunk.chunk, spans)) {
                throw std::runtime_error("failed to locate restored KVFlash payload");
            }
            size_t payload_size = 0;
            for (const auto & span : spans) {
                if (span.payload_offset != payload_size || payload_size > (size_t) state.chunk_bytes ||
                    span.size > (size_t) state.chunk_bytes - payload_size) {
                    throw std::runtime_error("invalid restored KVFlash payload layout");
                }
                if (span.tensor) {
                    io.read_tensor(span.tensor, span.tensor_offset, span.size);
                } else if (span.host) {
                    io.read(span.host, span.size);
                } else if (span.size != 0) {
                    throw std::runtime_error("missing restored KVFlash payload storage");
                }
                payload_size += span.size;
            }
            if (payload_size != state.chunk_bytes) {
                throw std::runtime_error("incomplete restored KVFlash payload");
            }
        }
    } catch (...) {
        clear(true);
        throw;
    }
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    if (kvflash) {
        kvflash_state_write(io, seq_id, flags);
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }
    if (kvflash) {
        kvflash_state_read(io, seq_id, flags);
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (size_k_idx_bytes() > 0) {
        const uint32_t has_k_idx_u32 = 1;
        io.write(&has_k_idx_u32, sizeof(has_k_idx_u32));

        for (const auto & layer : layers) {
            const uint32_t layer_has_k_idx = layer.k_idx ? 1 : 0;
            io.write(&layer_has_k_idx, sizeof(layer_has_k_idx));

            if (!layer_has_k_idx) {
                continue;
            }

            GGML_ASSERT(layer.k_idx_stream[cr.strm]);

            const int32_t k_idx_type_i = (int32_t) layer.k_idx->type;
            io.write(&k_idx_type_i, sizeof(k_idx_type_i));

            const uint64_t k_idx_size_row = ggml_row_size(layer.k_idx->type, layer.k_idx->ne[0]);
            io.write(&k_idx_size_row, sizeof(k_idx_size_row));

            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size   = range_size * k_idx_size_row;
                const size_t offset     = range.first * k_idx_size_row;

                io.write_tensor(layer.k_idx_stream[cr.strm], offset, buf_size);
            }
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (size_k_idx_bytes() > 0) {
        uint32_t has_k_idx_u32 = 0;
        io.read(&has_k_idx_u32, sizeof(has_k_idx_u32));

        if (has_k_idx_u32 != 1) {
            LLAMA_LOG_ERROR("%s: missing k_idx data in KV cache state\n", __func__);
            return false;
        }

        for (const auto & layer : layers) {
            uint32_t layer_has_k_idx = 0;
            io.read(&layer_has_k_idx, sizeof(layer_has_k_idx));

            const uint32_t expected_layer_has_k_idx = layer.k_idx ? 1 : 0;

            if (layer_has_k_idx != expected_layer_has_k_idx) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx state for layer: got %u, expected %u\n",
                    __func__, layer_has_k_idx, expected_layer_has_k_idx);
                return false;
            }

            if (!layer_has_k_idx) {
                continue;
            }

            GGML_ASSERT(layer.k_idx_stream[strm]);

            int32_t k_idx_type_i = -1;
            io.read(&k_idx_type_i, sizeof(k_idx_type_i));

            if (k_idx_type_i != (int32_t) layer.k_idx->type) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx type: got %d, expected %d\n",
                    __func__, k_idx_type_i, (int32_t) layer.k_idx->type);
                return false;
            }

            uint64_t k_idx_size_row = 0;
            io.read(&k_idx_size_row, sizeof(k_idx_size_row));

            const uint64_t expected_k_idx_size_row = ggml_row_size(layer.k_idx->type, layer.k_idx->ne[0]);

            if (k_idx_size_row != expected_k_idx_size_row) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx row size: got %zu, expected %zu\n",
                    __func__, (size_t) k_idx_size_row, (size_t) expected_k_idx_size_row);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    io.read_tensor(layer.k_idx_stream[strm], sinfo.head() * k_idx_size_row, cell_count * k_idx_size_row);
                } else {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        io.read_tensor(layer.k_idx_stream[strm], sinfo.idxs[0][i] * k_idx_size_row, k_idx_size_row);
                    }
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    if (kv->has_kvflash()) {
        auto sinfo = kv->find_slot(ubatches[i_cur], false);
        if (sinfo.empty()) {
            return false;
        }
        sinfos[i_cur] = std::move(sinfo);
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

const llama_kv_cache::slot_info & llama_kv_cache_context::get_slot_info() const {
    GGML_ASSERT(i_cur < sinfos.size());
    return sinfos[i_cur];
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_idx(ggml_context * ctx, int32_t il) const {
    return kv->get_k_idx(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k_idx(ctx, k_idx_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
