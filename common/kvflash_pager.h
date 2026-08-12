// common_kvflash_pager — KVFlash core (Lucebox port).
//
// FlashMemory-style (arXiv 2606.09079) decode-time KV residency:
// attention tensors allocate at POOL size; logical positions map to physical
// slots at 64-token chunk granularity; cold chunks page to host bit-exact.
//
// Scope: full-attention layers only. GDN/SSM recurrent state is never paged.
// Policy: pure LRU by default; optional score_hook for reselect().
//
// P0: sync ggml_backend_tensor_get/set only (no CUDA async stream).
// See KVFLASH_PLAN.md for integration PRs.
//
// Ported from Luce-Org/lucebox optimizations/kvflash (Apache-2.0 / project license).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace common_kvflash {

struct KvFlashConfig {
    int chunk_tokens       = 64;  // logical tokens per page
    int pool_tokens        = 0;   // resident pool capacity (multiple of chunk_tokens)
    int sink_chunks        = 1;   // leading chunks never evicted (attention sinks)
    int tail_window_chunks = 4;   // trailing chunks never evicted (local window)
};

struct KvFlashStats {
    int64_t page_outs   = 0;
    int64_t page_ins    = 0;
    int64_t host_bytes  = 0;
    int64_t moved_bytes = 0;
};

class KvFlashPager {
public:
    // Minimum pool: sinks + tail + 2 chunks (1 victim + append head).
    static int min_pool_tokens(const KvFlashConfig & cfg) {
        return (cfg.sink_chunks + cfg.tail_window_chunks + 2) * cfg.chunk_tokens;
    }

    // attn_k / attn_v: per full-attn layer tensors [head_dim, pool_tokens, n_head_kv].
    // May be empty for map-only unit tests (no host byte copies).
    bool attach(const KvFlashConfig & cfg,
                const std::vector<ggml_tensor *> & attn_k,
                const std::vector<ggml_tensor *> & attn_v) {
        if (cfg.pool_tokens <= 0 || cfg.pool_tokens % cfg.chunk_tokens != 0) {
            return false;
        }
        if (cfg.pool_tokens < min_pool_tokens(cfg)) {
            std::fprintf(stderr,
                "[kvflash] pool %d < minimum %d (%d sink + %d tail must leave a victim)\n",
                cfg.pool_tokens, min_pool_tokens(cfg),
                cfg.sink_chunks, cfg.tail_window_chunks);
            return false;
        }
        if (attn_k.size() != attn_v.size()) {
            return false;
        }

        cfg_     = cfg;
        attn_k_  = attn_k;
        attn_v_  = attn_v;
        n_blocks_ = cfg.pool_tokens / cfg.chunk_tokens;

        if (!attn_k.empty()) {
            const ggml_tensor * K0 = attn_k[0];
            // llama layout: [n_embd_gqa, pool_tokens, n_stream] — heads packed in ne[0]
            if ((int) K0->ne[1] < cfg.pool_tokens) {
                return false;
            }
            n_stream_    = (int) K0->ne[2];
            // One contiguous segment = chunk_tokens full GQA rows
            k_seg_bytes_ = (size_t) cfg.chunk_tokens * K0->nb[1];
            v_seg_bytes_ = (size_t) cfg.chunk_tokens * attn_v[0]->nb[1];
            // Per layer (K+V); only stream 0 is paged (n_parallel=1)
            chunk_bytes_ = (k_seg_bytes_ + v_seg_bytes_) * attn_k.size();
            zero_buf_.assign(std::max(k_seg_bytes_, v_seg_bytes_), 0);
        } else {
            n_stream_    = 0;
            k_seg_bytes_ = 0;
            v_seg_bytes_ = 0;
            chunk_bytes_ = 0;
            zero_buf_.clear();
        }

        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= 0; b--) {
            free_blocks_.push_back(b);
        }
        chunks_.clear();
        stats_     = {};
        clock_     = 0;
        cur_chunk_ = 0;
        epoch_     = 0;
        return true;
    }

    void set_block_order(const std::vector<int> & order) {
        free_blocks_.assign(order.rbegin(), order.rend());
    }

    void reset() {
        chunks_.clear();
        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= 0; b--) {
            free_blocks_.push_back(b);
        }
        stats_.host_bytes = 0;
        cur_chunk_ = 0;
        epoch_++;
    }

    void zero_free_blocks() {
        for (int b : free_blocks_) {
            zero_block(b);
        }
    }

    bool attached() const { return n_blocks_ > 0; }
    int  pool_tokens() const { return cfg_.pool_tokens; }
    int  chunk_tokens() const { return cfg_.chunk_tokens; }

    // Higher score = keep. Empty → pure LRU.
    std::function<float(int /*chunk*/)> score_hook;

    // Called after a pool block is freed (page_out). Host may clear cell
    // metadata for physical slots [block*chunk, (block+1)*chunk).
    std::function<void(int /*block*/)> on_block_evicted;

    bool alloc_span(int kv_start, int n_tok) {
        for (int i = 0; i < n_tok; ++i) {
            if (slot_for(kv_start + i) < 0) {
                std::fprintf(stderr, "[kvflash] no pool slot at pos %d (pool %d exhausted)\n",
                             kv_start + i, cfg_.pool_tokens);
                return false;
            }
        }
        return true;
    }

    // Physical pool slot for logical pos. Allocates / evicts at chunk granularity.
    int slot_for(int64_t pos) {
        const int c = (int) (pos / cfg_.chunk_tokens);
        const int prev_cur_chunk = cur_chunk_;
        if (c > cur_chunk_) {
            cur_chunk_ = c;
        }
        if ((int) chunks_.size() <= c) {
            chunks_.resize(c + 1);
        }
        ChunkState & st = chunks_[c];
        if (st.block < 0) {
            if (!ensure_free_block()) {
                cur_chunk_ = prev_cur_chunk;
                return -1;
            }
            st.block = free_blocks_.back();
            free_blocks_.pop_back();
            epoch_++;
            if (st.on_host) {
                copy_chunk(c, st.block, /*to_host=*/false);
                stats_.page_ins++;
                stats_.moved_bytes += (int64_t) chunk_bytes_;
            }
        }
        st.last_use = ++clock_;
        return st.block * cfg_.chunk_tokens + (int) (pos % cfg_.chunk_tokens);
    }

    bool page_out(int c) {
        if (c >= (int) chunks_.size() || chunks_[c].block < 0) {
            return false;
        }
        ChunkState & st = chunks_[c];
        if (has_tensor_storage() && !st.on_host) {
            st.host_data.resize(chunk_bytes_);
            stats_.host_bytes += (int64_t) chunk_bytes_;
        }
        copy_chunk(c, st.block, /*to_host=*/true);
        zero_block(st.block);
        st.on_host = true;
        const int freed = st.block;
        free_blocks_.push_back(st.block);
        st.block = -1;
        epoch_++;
        stats_.page_outs++;
        stats_.moved_bytes += (int64_t) chunk_bytes_;
        if (on_block_evicted) {
            on_block_evicted(freed);
        }
        return true;
    }

    bool page_in(int c) {
        if (c >= (int) chunks_.size() || !chunks_[c].on_host || chunks_[c].block >= 0) {
            return false;
        }
        return slot_for((int64_t) c * cfg_.chunk_tokens) >= 0;
    }

    void synchronize_paging() { /* P0: sync path — no-op */ }

    bool is_resident(int c) const {
        return c < (int) chunks_.size() && chunks_[c].block >= 0;
    }

    bool is_identity() const {
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (chunks_[c].block >= 0 && chunks_[c].block != c) {
                return false;
            }
            if (chunks_[c].block < 0 && chunks_[c].on_host) {
                return false;
            }
        }
        return true;
    }

    bool identity_prefix_covers(int n_tok) const {
        if (n_tok <= 0) {
            return true;
        }
        const int nc = (n_tok + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
        if (nc > (int) chunks_.size()) {
            return false;
        }
        for (int c = 0; c < nc; c++) {
            if (chunks_[c].block != c) {
                return false;
            }
        }
        return true;
    }

    int block_of(int c) const {
        return c < (int) chunks_.size() ? chunks_[c].block : -1;
    }

    int slot_of(int64_t pos) const {
        const int c = (int) (pos / cfg_.chunk_tokens);
        if (c >= (int) chunks_.size() || chunks_[c].block < 0) {
            return -1;
        }
        return chunks_[c].block * cfg_.chunk_tokens + (int) (pos % cfg_.chunk_tokens);
    }

    void fill_slot_pos(int32_t * dst) const {
        for (int i = 0; i < cfg_.pool_tokens; i++) {
            dst[i] = -1;
        }
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (chunks_[c].block < 0) {
                continue;
            }
            int32_t * p = dst + (size_t) chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++) {
                p[i] = (int32_t) c * cfg_.chunk_tokens + i;
            }
        }
    }

    const KvFlashStats & stats() const { return stats_; }
    int resident_blocks() const { return n_blocks_ - (int) free_blocks_.size(); }
    int n_chunks() const { return (int) chunks_.size(); }
    uint64_t epoch() const { return epoch_; }

    // F16 validity: 0 resident, -inf free (IEEE f16: 0xFC00 = -inf).
    void fill_slot_mask(uint16_t * dst) const {
        constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
        for (int i = 0; i < cfg_.pool_tokens; i++) {
            dst[i] = F16_NEG_INF;
        }
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (chunks_[c].block < 0) {
                continue;
            }
            uint16_t * p = dst + (size_t) chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++) {
                p[i] = F16_ZERO;
            }
        }
    }

    // f32 validity for llama graph masks.
    void fill_slot_mask_f32(float * dst) const {
        for (int i = 0; i < cfg_.pool_tokens; i++) {
            dst[i] = -INFINITY;
        }
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (chunks_[c].block < 0) {
                continue;
            }
            float * p = dst + (size_t) chunks_[c].block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; i++) {
                p[i] = 0.0f;
            }
        }
    }

    // Rebuild resident set as top-n_blocks by score_hook (sinks/tail protected).
    int reselect() {
        struct Cand {
            int   c;
            float s;
        };
        std::vector<Cand> cands;
        for (int c = 0; c < (int) chunks_.size(); c++) {
            const ChunkState & st = chunks_[c];
            if (st.block < 0 && !st.on_host) {
                continue;
            }
            const bool prot = c < cfg_.sink_chunks ||
                              c > cur_chunk_ - 1 - cfg_.tail_window_chunks;
            float sc;
            if (prot) {
                sc = 3.4e38f;
            } else if (score_hook) {
                sc = score_hook(c);
            } else {
                // Default LRU: more recent last_use → keep resident
                sc = (float) st.last_use;
            }
            cands.push_back({c, sc});
        }
        if (cands.empty()) {
            return 0;
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand & a, const Cand & b) { return a.s > b.s; });
        std::vector<uint8_t> want(chunks_.size(), 0);
        for (int i = 0; i < (int) cands.size() && i < n_blocks_; i++) {
            want[(size_t) cands[i].c] = 1;
        }

        int events = 0;
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (!want[(size_t) c] && chunks_[c].block >= 0) {
                page_out(c);
                events++;
            }
        }
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (want[(size_t) c] && chunks_[c].block < 0 && chunks_[c].on_host) {
                if (page_in(c)) {
                    events++;
                }
            }
        }
        return events;
    }

private:
    struct ChunkState {
        int                  block    = -1;
        bool                 on_host  = false;
        uint64_t             last_use = 0;
        std::vector<uint8_t> host_data;
    };

    bool ensure_free_block() {
        if (!free_blocks_.empty()) {
            return true;
        }
        int      victim  = -1;
        float    v_score = 0.f;
        uint64_t v_use   = 0;
        for (int c = 0; c < (int) chunks_.size(); c++) {
            if (chunks_[c].block < 0) {
                continue;
            }
            if (c < cfg_.sink_chunks) {
                continue;
            }
            if (c > cur_chunk_ - 1 - cfg_.tail_window_chunks) {
                continue;
            }
            if (score_hook) {
                const float s = score_hook(c);
                if (victim < 0 || s < v_score) {
                    victim  = c;
                    v_score = s;
                }
            } else {
                if (victim < 0 || chunks_[c].last_use < v_use) {
                    victim = c;
                    v_use  = chunks_[c].last_use;
                }
            }
        }
        return victim >= 0 && page_out(victim);
    }

    void copy_chunk(int c, int block, bool to_host) {
        if (!has_tensor_storage()) {
            return;
        }
        ChunkState & st = chunks_[c];
        uint8_t *    p  = st.host_data.data();
        // llama KV: [n_embd_gqa, n_tokens, n_stream] — copy stream-0 rows only
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t   = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t  seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                const size_t  off = (size_t) block * cfg_.chunk_tokens * t->nb[1];
                if (to_host) {
                    ggml_backend_tensor_get(t, p, off, seg);
                } else {
                    ggml_backend_tensor_set(t, p, off, seg);
                }
                p += seg;
            }
        }
    }

    void zero_block(int block) {
        if (!has_tensor_storage()) {
            return;
        }
        for (size_t l = 0; l < attn_k_.size(); l++) {
            for (int kv = 0; kv < 2; kv++) {
                ggml_tensor * t   = kv == 0 ? attn_k_[l] : attn_v_[l];
                const size_t  seg = kv == 0 ? k_seg_bytes_ : v_seg_bytes_;
                const size_t  off = (size_t) block * cfg_.chunk_tokens * t->nb[1];
                ggml_backend_tensor_set(t, zero_buf_.data(), off, seg);
            }
        }
    }

    bool has_tensor_storage() const {
        return !attn_k_.empty() && chunk_bytes_ > 0;
    }

    KvFlashConfig              cfg_;
    std::vector<ggml_tensor *> attn_k_, attn_v_;
    std::vector<ChunkState>    chunks_;
    std::vector<int>           free_blocks_;
    std::vector<uint8_t>       zero_buf_;
    KvFlashStats               stats_;
    size_t                     k_seg_bytes_ = 0, v_seg_bytes_ = 0, chunk_bytes_ = 0;
    int                        n_blocks_ = 0, n_stream_ = 0, cur_chunk_ = 0;
    uint64_t                   clock_ = 0;
    uint64_t                   epoch_ = 0;
};

// --- env / sizing helpers -------------------------------------------------

struct KvFlashAutoBudget {
    int64_t free_bytes       = 0;
    int64_t reserve_bytes    = 0;
    int64_t bytes_per_token  = 0;
    int     speed_cap_tokens = 16384;
};

// LLAMA_KVFLASH: 0/unset=off; "auto"; or integer tokens.
inline int kvflash_pool_from_env(int max_ctx, const KvFlashConfig & cfg = {},
                                 bool scorer_expected = false,
                                 const KvFlashAutoBudget & budget = {}) {
    const char * env = std::getenv("LLAMA_KVFLASH");
    if (!env) {
        return 0;
    }
    int tokens = 0;
    if (std::strcmp(env, "auto") == 0) {
        int speed_cap = budget.speed_cap_tokens;
        if (const char * mp = std::getenv("LLAMA_KVFLASH_MAX_POOL")) {
            speed_cap = std::max(256, std::atoi(mp));
        }
        if (budget.bytes_per_token > 0 && budget.free_bytes > 0) {
            const int64_t usable =
                    std::max<int64_t>(0, budget.free_bytes - budget.reserve_bytes) / 2;
            const int64_t vram_tokens = usable / budget.bytes_per_token;
            tokens = (int) std::min<int64_t>(
                    vram_tokens, std::min<int64_t>(max_ctx, speed_cap));
        } else {
            tokens = max_ctx / (scorer_expected ? 4 : 2);
        }
    } else {
        tokens = std::atoi(env);
    }
    if (tokens <= 0) {
        return 0;
    }
    tokens = ((tokens + 255) / 256) * 256;
    const int floor_tokens = ((KvFlashPager::min_pool_tokens(cfg) + 255) / 256) * 256;
    if (tokens < floor_tokens) {
        tokens = floor_tokens;
    }
    if (tokens > max_ctx) {
        tokens = (max_ctx / 256) * 256;
    }
    return tokens;
}

inline bool kvflash_policy_is_lru() {
    const char * env = std::getenv("LLAMA_KVFLASH_POLICY");
    return env && std::strcmp(env, "lru") == 0;
}

inline bool kvflash_policy_is_qk() {
    const char * env = std::getenv("LLAMA_KVFLASH_POLICY");
    return env && std::strcmp(env, "qk") == 0;
}

// Build physical write rows + causal f32 masks in slot space.
inline bool kvflash_fill_rows_and_masks(
        const KvFlashPager & pager,
        int kv_start, int n_tok, int mk_w, int swa_window,
        std::vector<int32_t> & rows,
        std::vector<float> * mfull, std::vector<float> * mswa) {
    rows.resize((size_t) n_tok);
    for (int i = 0; i < n_tok; ++i) {
        const int s = pager.slot_of(kv_start + i);
        if (s < 0) {
            return false;
        }
        rows[(size_t) i] = s;
    }
    if (!mfull) {
        return true;
    }
    std::vector<int32_t> spos((size_t) pager.pool_tokens(), -1);
    pager.fill_slot_pos(spos.data());
    mfull->assign((size_t) mk_w * n_tok, -INFINITY);
    if (mswa) {
        mswa->assign((size_t) mk_w * n_tok, -INFINITY);
    }
    const int s_hi = std::min(mk_w, (int) spos.size());
    for (int q = 0; q < n_tok; ++q) {
        const int abs_q  = kv_start + q;
        const int win_lo = std::max(0, abs_q - swa_window + 1);
        for (int s = 0; s < s_hi; ++s) {
            const int p = spos[(size_t) s];
            if (p < 0 || p > abs_q) {
                continue;
            }
            (*mfull)[(size_t) q * mk_w + s] = 0.0f;
            if (mswa && p >= win_lo) {
                (*mswa)[(size_t) q * mk_w + s] = 0.0f;
            }
        }
    }
    return true;
}

} // namespace common_kvflash
