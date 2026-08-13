// KVFlash core for a bounded full-attention KV resident pool.
//
// Logical positions map to fixed-size physical chunks. Cold chunks are copied
// bit-exactly to host memory and their physical blocks are reused. The pager
// is policy agnostic: allocation uses LRU unless score_hook is installed.
//
// llama.cpp stores the hybrid full-attention cache as packed token rows:
// [n_embd_gqa, pool_tokens, n_stream]. Only a single stream is supported.

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
#include <limits>
#include <new>
#include <numeric>
#include <vector>

namespace common_kvflash {

struct KvFlashConfig {
    int  chunk_tokens       = 64;
    int  pool_tokens        = 0;
    int64_t max_context_tokens = 0;
    int  sink_chunks        = 1;
    int  tail_window_chunks = 4;
    bool zero_freed_blocks  = false;
};

struct KvFlashStats {
    int64_t page_outs            = 0;
    int64_t page_ins             = 0;
    int64_t host_bytes           = 0;
    int64_t host_allocated_bytes = 0;
    int64_t moved_bytes          = 0;
};

struct KvFlashStateTensor {
    int32_t type = -1;
    uint64_t segment_bytes = 0;
};

struct KvFlashStateChunk {
    int32_t chunk = -1;
    int32_t block = -1;
    uint64_t last_use = 0;
};

struct KvFlashState {
    int32_t chunk_tokens = 0;
    int32_t pool_tokens = 0;
    int64_t max_context_tokens = 0;
    int32_t sink_chunks = 0;
    int32_t tail_window_chunks = 0;
    int32_t cur_chunk = 0;
    uint64_t clock = 0;
    uint64_t chunk_bytes = 0;
    std::vector<KvFlashStateTensor> tensors;
    std::vector<KvFlashStateChunk> chunks;
};

struct KvFlashStateSpan {
    ggml_tensor * tensor = nullptr;
    uint8_t * host = nullptr;
    size_t tensor_offset = 0;
    size_t payload_offset = 0;
    size_t size = 0;
};

class KvFlashPager {
public:
    KvFlashPager() = default;
    KvFlashPager(const KvFlashPager &) = delete;
    KvFlashPager & operator=(const KvFlashPager &) = delete;
    KvFlashPager(KvFlashPager &&) = delete;
    KvFlashPager & operator=(KvFlashPager &&) = delete;

    ~KvFlashPager() {
        (void) synchronize_paging();
        release_resources();
    }

    static int min_pool_tokens(const KvFlashConfig & cfg) {
        if (cfg.chunk_tokens <= 0 || cfg.sink_chunks < 0 || cfg.tail_window_chunks < 0) {
            return 0;
        }
        const int64_t chunks = (int64_t) cfg.sink_chunks + cfg.tail_window_chunks + 2;
        if (chunks > std::numeric_limits<int>::max() / (int64_t) cfg.chunk_tokens) {
            return std::numeric_limits<int>::max();
        }
        return (int) (chunks * cfg.chunk_tokens);
    }

    // Tensors must contain packed token rows and one stream. An empty tensor
    // list is supported for map-only tests.
    bool attach(const KvFlashConfig & cfg,
                const std::vector<ggml_tensor *> & attn_k,
                const std::vector<ggml_tensor *> & attn_v) {
        if (cfg.chunk_tokens <= 0 || cfg.pool_tokens <= 0 ||
            cfg.pool_tokens % cfg.chunk_tokens != 0 ||
            cfg.sink_chunks < 0 || cfg.tail_window_chunks < 0 ||
            cfg.max_context_tokens < 0 ||
            (cfg.max_context_tokens > 0 && cfg.max_context_tokens < cfg.pool_tokens)) {
            return false;
        }

        const int minimum_pool = min_pool_tokens(cfg);
        if (minimum_pool <= 0 || cfg.pool_tokens < minimum_pool) {
            std::fprintf(stderr,
                    "[kvflash] pool %d < minimum %d (%d sink + %d tail chunks)\n",
                    cfg.pool_tokens, minimum_pool, cfg.sink_chunks, cfg.tail_window_chunks);
            return false;
        }
        if (attn_k.size() != attn_v.size()) {
            return false;
        }

        std::vector<TensorBinding> validated;
        size_t validated_chunk_bytes = 0;
        validated.reserve(attn_k.size() * 2);

        for (size_t l = 0; l < attn_k.size(); ++l) {
            for (int kv = 0; kv < 2; ++kv) {
                ggml_tensor * tensor = kv == 0 ? attn_k[l] : attn_v[l];
                if (!tensor || !tensor->data || !tensor_buffer(tensor) ||
                    tensor->ne[0] <= 0 || tensor->ne[1] < cfg.pool_tokens ||
                    tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
                    tensor->nb[1] == 0 || !ggml_is_contiguous(tensor)) {
                    return false;
                }
                if ((size_t) cfg.chunk_tokens >
                    std::numeric_limits<size_t>::max() / tensor->nb[1]) {
                    return false;
                }
                const size_t segment_bytes = (size_t) cfg.chunk_tokens * tensor->nb[1];
                if ((size_t) cfg.pool_tokens >
                    std::numeric_limits<size_t>::max() / tensor->nb[1] ||
                    (size_t) cfg.pool_tokens * tensor->nb[1] > ggml_nbytes(tensor) ||
                    segment_bytes > std::numeric_limits<size_t>::max() - validated_chunk_bytes) {
                    return false;
                }

                ggml_backend_buffer_t buffer = tensor_buffer(tensor);
                TensorBinding binding;
                binding.tensor        = tensor;
                binding.segment_bytes = segment_bytes;
                binding.buft          = ggml_backend_buffer_get_type(buffer);
                binding.device        = ggml_backend_buft_get_device(binding.buft);
                binding.host          = ggml_backend_buffer_is_host(buffer);
                validated.push_back(binding);
                validated_chunk_bytes += segment_bytes;
            }
        }

        if (validated_chunk_bytes > (size_t) std::numeric_limits<int64_t>::max()) {
            return false;
        }

        if (!synchronize_paging()) {
            return false;
        }
        release_resources();

        cfg_         = cfg;
        n_blocks_    = cfg.pool_tokens / cfg.chunk_tokens;
        bindings_    = std::move(validated);
        chunk_bytes_ = validated_chunk_bytes;

        free_blocks_.reserve((size_t) n_blocks_);
        for (int b = n_blocks_ - 1; b >= 0; --b) {
            free_blocks_.push_back(b);
        }
        block_to_chunk_.assign((size_t) n_blocks_, -1);
        allocation_pins_.reserve((size_t) n_blocks_);
        stats_ = {};
        clock_ = 0;
        cur_chunk_ = 0;
        epoch_ = 0;
        identity_ = true;
        score_hook = nullptr;
        paging_failed_ = false;
        return true;
    }

    // Bind the cache tensors to the context's real backends. Paging copies
    // then use those same streams, preserving ordering with graph compute and
    // batching all layer transfers behind one synchronization per page batch.
    bool bind_backends(const std::vector<ggml_backend_t> & backends) {
        if (!synchronize_paging()) {
            return false;
        }

        std::vector<ggml_backend_t> validated_backends;
        std::vector<int> validated_indices(bindings_.size(), -1);

        ggml_backend_buffer_type_t common_host_buft = nullptr;
        bool host_buft_consistent = true;

        for (size_t i = 0; i < bindings_.size(); ++i) {
            const TensorBinding & binding = bindings_[i];
            if (binding.host) {
                continue;
            }

            ggml_backend_t match = nullptr;
            for (ggml_backend_t backend : backends) {
                if (backend && ggml_backend_get_device(backend) == binding.device &&
                    ggml_backend_get_default_buffer_type(backend) == binding.buft) {
                    match = backend;
                    break;
                }
            }
            if (!match) {
                return false;
            }

            auto it = std::find(validated_backends.begin(), validated_backends.end(), match);
            if (it == validated_backends.end()) {
                validated_indices[i] = (int) validated_backends.size();
                validated_backends.push_back(match);
            } else {
                validated_indices[i] = (int) std::distance(validated_backends.begin(), it);
            }

            ggml_backend_buffer_type_t host_buft =
                    ggml_backend_dev_host_buffer_type(binding.device);
            if (!host_buft) {
                host_buft_consistent = false;
            } else if (!common_host_buft) {
                common_host_buft = host_buft;
            } else if (common_host_buft != host_buft) {
                host_buft_consistent = false;
            }
        }

        page_backends_ = std::move(validated_backends);
        backend_used_.assign(page_backends_.size(), 0);
        host_buft_ = host_buft_consistent ? common_host_buft : nullptr;
        for (size_t i = 0; i < bindings_.size(); ++i) {
            bindings_[i].backend = validated_indices[i];
        }
        return true;
    }

    bool has_async_paging() const {
        for (ggml_backend_t backend : page_backends_) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(ggml_backend_get_device(backend), &props);
            if (props.caps.async) {
                return true;
            }
        }
        return false;
    }

    bool set_block_order(const std::vector<int> & order) {
        if (resident_blocks() != 0 || (int) order.size() != n_blocks_) {
            return false;
        }
        std::vector<uint8_t> seen((size_t) n_blocks_, 0);
        for (int block : order) {
            if (block < 0 || block >= n_blocks_ || seen[(size_t) block]) {
                return false;
            }
            seen[(size_t) block] = 1;
        }
        free_blocks_.assign(order.rbegin(), order.rend());
        return true;
    }

    void reset() {
        synchronize_paging();
        release_host_slabs();
        chunks_.clear();
        free_blocks_.clear();
        for (int b = n_blocks_ - 1; b >= 0; --b) {
            free_blocks_.push_back(b);
        }
        block_to_chunk_.assign((size_t) n_blocks_, -1);
        allocation_pins_.clear();
        stats_.host_bytes = 0;
        stats_.host_allocated_bytes = 0;
        clock_ = 0;
        cur_chunk_ = 0;
        identity_ = true;
        paging_failed_ = false;
        ++epoch_;
    }

    bool zero_free_blocks() {
        if (paging_failed_ || !zero_blocks(free_blocks_)) {
            paging_failed_ = true;
            return false;
        }
        return synchronize_paging();
    }

    bool attached() const {
        return n_blocks_ > 0;
    }

    int pool_tokens() const {
        return cfg_.pool_tokens;
    }

    int chunk_tokens() const {
        return cfg_.chunk_tokens;
    }

    int max_batch_tokens() const {
        if (!attached()) {
            return 0;
        }
        const int usable_chunks = std::max(1, n_blocks_ - cfg_.sink_chunks - 1);
        return usable_chunks * cfg_.chunk_tokens;
    }

    std::function<float(int)> score_hook;
    std::function<bool(int, int)> on_block_paged_out;
    std::function<bool(int, int)> on_block_paged_in;
    std::function<void(int)> on_block_evicted;

    bool has_score_hook() const {
        return (bool) score_hook;
    }

    bool can_map_positions(const std::vector<int64_t> & positions) const {
        std::vector<int64_t> sorted_positions = positions;
        std::sort(sorted_positions.begin(), sorted_positions.end());
        int previous_chunk = -1;
        int n_chunks = 0;
        for (size_t i = 0; i < sorted_positions.size(); ++i) {
            const int64_t pos = sorted_positions[i];
            if (!position_valid(pos)) {
                return false;
            }
            if (i > 0 && pos == sorted_positions[i - 1]) {
                return false;
            }
            const int chunk = (int) (pos / cfg_.chunk_tokens);
            if (chunk != previous_chunk) {
                previous_chunk = chunk;
                ++n_chunks;
            }
        }
        return n_chunks <= std::max(1, n_blocks_ - cfg_.sink_chunks - 1);
    }

    bool alloc_span(int64_t kv_start, int n_tok) {
        if (paging_failed_ || kv_start < 0 || n_tok < 0 ||
            (int64_t) n_tok > std::numeric_limits<int64_t>::max() - kv_start) {
            return false;
        }
        if (n_tok == 0) {
            return true;
        }
        const int64_t end = kv_start + n_tok;
        if (!position_valid(end - 1)) {
            return false;
        }

        const int first = (int) (kv_start / cfg_.chunk_tokens);
        const int last  = (int) ((end - 1) / cfg_.chunk_tokens);
        allocation_pins_.clear();
        allocation_pins_.reserve((size_t) (last - first + 1));
        for (int chunk = first; chunk <= last; ++chunk) {
            allocation_pins_.push_back(chunk);
        }
        if ((int) allocation_pins_.size() >
            std::max(1, n_blocks_ - cfg_.sink_chunks - 1)) {
            allocation_pins_.clear();
            return false;
        }
        return allocate_pinned_chunks();
    }

    bool alloc_positions(const std::vector<int64_t> & positions) {
        std::vector<int64_t> sorted_positions = positions;
        std::sort(sorted_positions.begin(), sorted_positions.end());
        allocation_pins_.clear();
        allocation_pins_.reserve(sorted_positions.size());
        int previous_chunk = -1;
        for (size_t i = 0; i < sorted_positions.size(); ++i) {
            const int64_t pos = sorted_positions[i];
            if (!position_valid(pos)) {
                allocation_pins_.clear();
                return false;
            }
            if (i > 0 && pos == sorted_positions[i - 1]) {
                allocation_pins_.clear();
                return false;
            }
            const int chunk = (int) (pos / cfg_.chunk_tokens);
            if (chunk != previous_chunk) {
                allocation_pins_.push_back(chunk);
                previous_chunk = chunk;
            }
        }
        if ((int) allocation_pins_.size() >
            std::max(1, n_blocks_ - cfg_.sink_chunks - 1)) {
            allocation_pins_.clear();
            return false;
        }
        return allocate_pinned_chunks();
    }

    int slot_for(int64_t pos) {
        if (!position_valid(pos)) {
            return -1;
        }
        allocation_pins_.clear();
        allocation_pins_.push_back((int) (pos / cfg_.chunk_tokens));
        if (!allocate_pinned_chunks()) {
            return -1;
        }
        return slot_of(pos);
    }

    bool page_out(int chunk) {
        const std::vector<int> chunks = { chunk };
        return page_out_batch(chunks);
    }

    bool page_in(int chunk) {
        if (paging_failed_ || chunk < 0 || chunk >= (int) chunks_.size() ||
            !chunks_[(size_t) chunk].on_host || chunks_[(size_t) chunk].block >= 0) {
            return false;
        }
        allocation_pins_.clear();
        allocation_pins_.push_back(chunk);
        return allocate_pinned_chunks();
    }

    bool page_in_chunks(const std::vector<int> & chunks) {
        if (paging_failed_ || chunks.empty()) {
            return !paging_failed_;
        }
        allocation_pins_ = chunks;
        std::sort(allocation_pins_.begin(), allocation_pins_.end());
        allocation_pins_.erase(
                std::unique(allocation_pins_.begin(), allocation_pins_.end()),
                allocation_pins_.end());
        if ((int) allocation_pins_.size() >
            std::max(1, n_blocks_ - cfg_.sink_chunks - 1)) {
            allocation_pins_.clear();
            return false;
        }
        for (int chunk : allocation_pins_) {
            if (chunk < 0 || chunk >= (int) chunks_.size()) {
                allocation_pins_.clear();
                return false;
            }
            const ChunkState & state = chunks_[(size_t) chunk];
            if (state.block < 0 && !state.on_host) {
                allocation_pins_.clear();
                return false;
            }
        }
        return allocate_pinned_chunks();
    }

    bool synchronize_paging() {
        for (size_t i = 0; i < page_backends_.size(); ++i) {
            if (i < backend_used_.size() && backend_used_[i]) {
                ggml_backend_synchronize(page_backends_[i]);
                backend_used_[i] = 0;
            }
        }
        for (ChunkState & state : chunks_) {
            state.host_inflight = false;
            state.device_inflight = false;
        }
        return !paging_failed_;
    }

    bool ensure_host_ready(int chunk) {
        if (paging_failed_) {
            return false;
        }
        if (chunk < 0 || chunk >= (int) chunks_.size() ||
            !chunks_[(size_t) chunk].host_inflight) {
            return true;
        }
        return synchronize_paging();
    }

    void mark_written(const std::vector<int> & chunks) {
        for (int chunk : chunks) {
            if (chunk < 0 || chunk >= (int) chunks_.size()) {
                continue;
            }
            ChunkState & state = chunks_[(size_t) chunk];
            if (state.block < 0) {
                continue;
            }
            state.dirty = true;
            state.snapshot_valid = false;
        }
    }

    void mark_all_resident_dirty() {
        for (ChunkState & state : chunks_) {
            if (state.block < 0) {
                continue;
            }
            state.dirty = true;
            state.snapshot_valid = false;
        }
    }

    // Queue D2H for dirty resident chunks except the open tail. Does not evict.
    // Call after the graph that wrote those rows, not from find_slot.
    int snapshot_sealed() {
        if (paging_failed_) {
            return -1;
        }
        int snapped = 0;
        for (int chunk = 0; chunk < (int) chunks_.size(); ++chunk) {
            ChunkState & state = chunks_[(size_t) chunk];
            if (state.block < 0 || !state.dirty || chunk == cur_chunk_) {
                continue;
            }
            if (!snapshot_resident(chunk)) {
                return -1;
            }
            ++snapped;
        }
        return snapped;
    }

    bool paging_ok() const {
        return !paging_failed_;
    }

    bool is_resident(int chunk) const {
        return chunk >= 0 && chunk < (int) chunks_.size() &&
                chunks_[(size_t) chunk].block >= 0;
    }

    bool has_host_snapshot(int chunk) const {
        return chunk >= 0 && chunk < (int) chunks_.size() &&
                chunks_[(size_t) chunk].on_host &&
                chunks_[(size_t) chunk].snapshot_valid &&
                chunks_[(size_t) chunk].host_ptr != nullptr;
    }

    bool is_identity() const {
        return identity_;
    }

    bool identity_prefix_covers(int n_tok) const {
        if (n_tok <= 0) {
            return true;
        }
        if (cfg_.chunk_tokens <= 0) {
            return false;
        }
        const int64_t count = ((int64_t) n_tok + cfg_.chunk_tokens - 1) /
                cfg_.chunk_tokens;
        if (count > (int64_t) chunks_.size()) {
            return false;
        }
        for (int chunk = 0; chunk < (int) count; ++chunk) {
            if (chunks_[(size_t) chunk].block != chunk) {
                return false;
            }
        }
        return true;
    }

    int block_of(int chunk) const {
        return chunk >= 0 && chunk < (int) chunks_.size() ?
                chunks_[(size_t) chunk].block : -1;
    }

    int slot_of(int64_t pos) const {
        if (!position_valid(pos)) {
            return -1;
        }
        const int chunk = (int) (pos / cfg_.chunk_tokens);
        if (!is_resident(chunk)) {
            return -1;
        }
        return chunks_[(size_t) chunk].block * cfg_.chunk_tokens +
                (int) (pos % cfg_.chunk_tokens);
    }

    void fill_slot_pos(int32_t * dst) const {
        std::fill(dst, dst + cfg_.pool_tokens, -1);
        for (int block = 0; block < n_blocks_; ++block) {
            const int chunk = block_to_chunk_[(size_t) block];
            if (chunk < 0) {
                continue;
            }
            int32_t * out = dst + (size_t) block * cfg_.chunk_tokens;
            const int64_t base = (int64_t) chunk * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; ++i) {
                const int64_t pos = base + i;
                out[i] = pos <= std::numeric_limits<int32_t>::max() ?
                        (int32_t) pos : -1;
            }
        }
    }

    const KvFlashStats & stats() const {
        return stats_;
    }

    int resident_blocks() const {
        return n_blocks_ - (int) free_blocks_.size();
    }

    int n_chunks() const {
        return (int) chunks_.size();
    }

    size_t host_slab_count() const {
        return host_slabs_.size();
    }

    uint64_t epoch() const {
        return epoch_;
    }

    bool state_export(KvFlashState & out) {
        if (paging_failed_ || !synchronize_paging()) {
            return false;
        }

        out = {};
        out.chunk_tokens = cfg_.chunk_tokens;
        out.pool_tokens = cfg_.pool_tokens;
        out.max_context_tokens = cfg_.max_context_tokens;
        out.sink_chunks = cfg_.sink_chunks;
        out.tail_window_chunks = cfg_.tail_window_chunks;
        out.cur_chunk = cur_chunk_;
        out.clock = clock_;
        out.chunk_bytes = chunk_bytes_;
        out.tensors.reserve(bindings_.size());
        for (const TensorBinding & binding : bindings_) {
            out.tensors.push_back({
                (int32_t) binding.tensor->type,
                (uint64_t) binding.segment_bytes,
            });
        }
        for (int chunk = 0; chunk < (int) chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t) chunk];
            if (state.block < 0 && !state.on_host) {
                continue;
            }
            out.chunks.push_back({
                chunk,
                state.block,
                state.last_use,
            });
        }
        return true;
    }

    bool state_import(const KvFlashState & in) {
        if (paging_failed_ || !synchronize_paging() ||
            in.chunk_tokens != cfg_.chunk_tokens ||
            in.pool_tokens != cfg_.pool_tokens ||
            in.max_context_tokens != cfg_.max_context_tokens ||
            in.sink_chunks != cfg_.sink_chunks ||
            in.tail_window_chunks != cfg_.tail_window_chunks ||
            in.chunk_bytes != chunk_bytes_ ||
            in.tensors.size() != bindings_.size()) {
            return false;
        }
        for (size_t i = 0; i < bindings_.size(); ++i) {
            if (in.tensors[i].type != (int32_t) bindings_[i].tensor->type ||
                in.tensors[i].segment_bytes != bindings_[i].segment_bytes) {
                return false;
            }
        }

        int max_chunk = -1;
        uint64_t max_last_use = 0;
        std::vector<uint8_t> seen_blocks((size_t) n_blocks_, 0);
        int resident = 0;
        int previous_chunk = -1;
        for (const KvFlashStateChunk & chunk : in.chunks) {
            const int64_t start = (int64_t) chunk.chunk * cfg_.chunk_tokens;
            if (chunk.chunk < 0 || chunk.chunk <= previous_chunk ||
                !position_valid(start) || chunk.block < -1 ||
                chunk.block >= n_blocks_) {
                return false;
            }
            previous_chunk = chunk.chunk;
            max_chunk = chunk.chunk;
            max_last_use = std::max(max_last_use, chunk.last_use);
            if (chunk.block >= 0) {
                if (seen_blocks[(size_t) chunk.block]) {
                    return false;
                }
                seen_blocks[(size_t) chunk.block] = 1;
                ++resident;
            }
        }
        const int64_t cur_chunk_start = (int64_t) in.cur_chunk * cfg_.chunk_tokens;
        if (resident > n_blocks_ || in.cur_chunk < 0 ||
            !position_valid(cur_chunk_start) ||
            (max_chunk >= 0 && in.cur_chunk < max_chunk)) {
            return false;
        }

        auto saved_score_hook = std::move(score_hook);
        release_host_slabs();
        chunks_.assign(max_chunk >= 0 ? (size_t) max_chunk + 1 : 0, {});
        block_to_chunk_.assign((size_t) n_blocks_, -1);
        free_blocks_.clear();
        allocation_pins_.clear();
        stats_ = {};
        paging_failed_ = false;

        for (const KvFlashStateChunk & chunk : in.chunks) {
            ChunkState & state = chunks_[(size_t) chunk.chunk];
            state.block = chunk.block;
            state.last_use = chunk.last_use;
            if (chunk.block >= 0) {
                block_to_chunk_[(size_t) chunk.block] = chunk.chunk;
                state.dirty = false;
                state.snapshot_valid = false;
            } else {
                if (!allocate_host_backing(state)) {
                    paging_failed_ = true;
                    score_hook = std::move(saved_score_hook);
                    return false;
                }
                state.on_host = true;
                state.dirty = false;
                state.snapshot_valid = true;
            }
        }
        for (int block = n_blocks_ - 1; block >= 0; --block) {
            if (!seen_blocks[(size_t) block]) {
                free_blocks_.push_back(block);
            }
        }

        cur_chunk_ = in.cur_chunk;
        clock_ = std::max(in.clock, max_last_use);
        identity_ = true;
        for (int block = 0; block < n_blocks_; ++block) {
            const int chunk = block_to_chunk_[(size_t) block];
            if (chunk >= 0 && chunk != block) {
                identity_ = false;
                break;
            }
        }
        score_hook = std::move(saved_score_hook);
        ++epoch_;
        return true;
    }

    bool state_spans(int chunk, std::vector<KvFlashStateSpan> & spans) {
        spans.clear();
        if (chunk < 0 || chunk >= (int) chunks_.size()) {
            return false;
        }
        ChunkState & state = chunks_[(size_t) chunk];
        if (state.block < 0 && !state.on_host) {
            return false;
        }
        if (!has_tensor_storage()) {
            return chunk_bytes_ == 0;
        }
        if (state.device_inflight && !synchronize_paging()) {
            return false;
        }
        if (state.block < 0) {
            if (!ensure_host_ready(chunk) || !state.host_ptr) {
                return false;
            }
            spans.push_back({
                nullptr,
                state.host_ptr,
                0,
                0,
                chunk_bytes_,
            });
            return true;
        }

        size_t payload_offset = 0;
        spans.reserve(bindings_.size());
        for (const TensorBinding & binding : bindings_) {
            spans.push_back({
                binding.tensor,
                nullptr,
                (size_t) state.block * cfg_.chunk_tokens * binding.tensor->nb[1],
                payload_offset,
                binding.segment_bytes,
            });
            payload_offset += binding.segment_bytes;
        }
        return payload_offset == chunk_bytes_;
    }

    void fill_slot_mask(uint16_t * dst) const {
        fill_slot_mask(dst, std::numeric_limits<int64_t>::max());
    }

    void fill_slot_mask(uint16_t * dst, int64_t valid_through) const {
        constexpr uint16_t F16_ZERO = 0x0000;
        constexpr uint16_t F16_NEG_INF = 0xFC00;
        std::fill(dst, dst + cfg_.pool_tokens, F16_NEG_INF);
        for (int block = 0; block < n_blocks_; ++block) {
            const int chunk = block_to_chunk_[(size_t) block];
            if (chunk < 0) {
                continue;
            }
            const int64_t start = (int64_t) chunk * cfg_.chunk_tokens;
            if (start > valid_through) {
                continue;
            }
            int valid = cfg_.chunk_tokens;
            const int64_t last = start + cfg_.chunk_tokens - 1;
            if (valid_through < last) {
                valid = (int) (valid_through - start + 1);
            }
            std::fill(dst + (size_t) block * cfg_.chunk_tokens,
                      dst + (size_t) block * cfg_.chunk_tokens + valid,
                      F16_ZERO);
        }
    }

    void fill_slot_mask_f32(float * dst) const {
        fill_slot_mask_f32(dst, std::numeric_limits<int64_t>::max());
    }

    void fill_slot_mask_f32(float * dst, int64_t valid_through) const {
        std::fill(dst, dst + cfg_.pool_tokens, -INFINITY);
        for (int block = 0; block < n_blocks_; ++block) {
            const int chunk = block_to_chunk_[(size_t) block];
            if (chunk < 0) {
                continue;
            }
            const int64_t start = (int64_t) chunk * cfg_.chunk_tokens;
            if (start > valid_through) {
                continue;
            }
            int valid = cfg_.chunk_tokens;
            const int64_t last = start + cfg_.chunk_tokens - 1;
            if (valid_through < last) {
                valid = (int) (valid_through - start + 1);
            }
            std::fill(dst + (size_t) block * cfg_.chunk_tokens,
                      dst + (size_t) block * cfg_.chunk_tokens + valid,
                      0.0f);
        }
    }

    // Relevance-based tau reselection. Pure LRU already evicts on demand, so
    // without a scorer this intentionally does no work or device traffic.
    int reselect() {
        if (paging_failed_) {
            return -1;
        }
        if (!score_hook) {
            return 0;
        }

        struct Candidate {
            int chunk;
            float score;
        };

        std::vector<uint8_t> want(chunks_.size(), 0);
        std::vector<Candidate> candidates;
        int protected_count = 0;

        for (int chunk = 0; chunk < (int) chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t) chunk];
            if (state.block < 0 && !state.on_host) {
                continue;
            }
            if (is_window_protected(chunk)) {
                want[(size_t) chunk] = 1;
                ++protected_count;
                continue;
            }
            float score = score_hook(chunk);
            if (std::isnan(score)) {
                score = -std::numeric_limits<float>::infinity();
            }
            candidates.push_back({ chunk, score });
        }

        if (protected_count > n_blocks_) {
            return -1;
        }
        const int keep = std::min<int>((int) candidates.size(),
                n_blocks_ - protected_count);
        auto higher = [](const Candidate & a, const Candidate & b) {
            return a.score != b.score ? a.score > b.score : a.chunk < b.chunk;
        };
        if (keep > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + keep,
                    candidates.end(), higher);
            for (int i = 0; i < keep; ++i) {
                want[(size_t) candidates[(size_t) i].chunk] = 1;
            }
        }

        std::vector<int> outgoing;
        std::vector<int> incoming;
        outgoing.reserve((size_t) n_blocks_);
        incoming.reserve((size_t) n_blocks_);
        for (int chunk = 0; chunk < (int) chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t) chunk];
            if (!want[(size_t) chunk] && state.block >= 0) {
                outgoing.push_back(chunk);
            } else if (want[(size_t) chunk] && state.block < 0 && state.on_host) {
                incoming.push_back(chunk);
            }
        }

        if (!outgoing.empty() && !page_out_batch(outgoing)) {
            return -1;
        }
        if (!incoming.empty()) {
            allocation_pins_ = incoming;
            std::sort(allocation_pins_.begin(), allocation_pins_.end());
            if (!allocate_pinned_chunks()) {
                return -1;
            }
        }
        return (int) (outgoing.size() + incoming.size());
    }

private:
    struct TensorBinding {
        ggml_tensor * tensor = nullptr;
        size_t segment_bytes = 0;
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_backend_dev_t device = nullptr;
        int backend = -1;
        bool host = false;
    };

    struct ChunkState {
        int block = -1;
        bool on_host = false;
        bool dirty = false;
        bool snapshot_valid = false;
        bool host_inflight = false;
        bool device_inflight = false;
        uint64_t last_use = 0;
        uint8_t * host_ptr = nullptr;
    };

    struct HostSlab {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<uint8_t> storage;
        size_t used = 0;
        size_t capacity = 0;

        uint8_t * data() {
            return buffer ?
                    (uint8_t *) ggml_backend_buffer_get_base(buffer) :
                    storage.data();
        }
    };

    static ggml_backend_buffer_t tensor_buffer(const ggml_tensor * tensor) {
        if (tensor->buffer) {
            return tensor->buffer;
        }
        const ggml_tensor * root = tensor->view_src;
        while (root && root->view_src) {
            root = root->view_src;
        }
        return root ? root->buffer : nullptr;
    }

    bool position_valid(int64_t pos) const {
        if (!attached() || cfg_.chunk_tokens <= 0 || pos < 0 ||
            pos > std::numeric_limits<int>::max()) {
            return false;
        }
        return cfg_.max_context_tokens <= 0 || pos < cfg_.max_context_tokens;
    }

    bool is_window_protected(int chunk) const {
        if (chunk < cfg_.sink_chunks) {
            return true;
        }
        const int64_t threshold = (int64_t) cur_chunk_ - 1 - cfg_.tail_window_chunks;
        return chunk > threshold;
    }

    bool is_allocation_pinned(int chunk) const {
        return std::binary_search(
                allocation_pins_.begin(), allocation_pins_.end(), chunk);
    }

    bool allocate_pinned_chunks() {
        if (paging_failed_) {
            allocation_pins_.clear();
            return false;
        }
        if (allocation_pins_.empty()) {
            return true;
        }

        std::sort(allocation_pins_.begin(), allocation_pins_.end());
        allocation_pins_.erase(
                std::unique(allocation_pins_.begin(), allocation_pins_.end()),
                allocation_pins_.end());
        const int max_chunk = allocation_pins_.back();
        const int64_t max_chunk_start = (int64_t) max_chunk * cfg_.chunk_tokens;
        if (!position_valid(max_chunk_start)) {
            allocation_pins_.clear();
            return false;
        }

        const int previous_cur = cur_chunk_;
        cur_chunk_ = std::max(cur_chunk_, max_chunk);

        std::vector<int> missing;
        missing.reserve(allocation_pins_.size());
        for (int chunk : allocation_pins_) {
            if (chunk >= (int) chunks_.size() || chunks_[(size_t) chunk].block < 0) {
                missing.push_back(chunk);
            }
        }

        const int need_victims = std::max<int>(
                0, (int) missing.size() - (int) free_blocks_.size());
        std::vector<int> victims;
        if (!select_victims(need_victims, victims)) {
            cur_chunk_ = previous_cur;
            allocation_pins_.clear();
            return false;
        }
        if (!victims.empty() && !page_out_batch(victims)) {
            cur_chunk_ = previous_cur;
            allocation_pins_.clear();
            return false;
        }

        if (missing.size() > free_blocks_.size()) {
            cur_chunk_ = previous_cur;
            allocation_pins_.clear();
            return false;
        }

        if ((int) chunks_.size() <= max_chunk) {
            chunks_.resize((size_t) max_chunk + 1);
        }

        std::vector<int> assigned_blocks;
        assigned_blocks.reserve(missing.size());
        for (size_t i = 0; i < missing.size(); ++i) {
            const int block = free_blocks_[free_blocks_.size() - 1 - i];
            if (block < 0 || block >= n_blocks_ ||
                block_to_chunk_[(size_t) block] != -1) {
                paging_failed_ = true;
                cur_chunk_ = previous_cur;
                allocation_pins_.clear();
                return false;
            }
            assigned_blocks.push_back(block);
            ChunkState & state = chunks_[(size_t) missing[i]];
            if (state.block >= 0) {
                paging_failed_ = true;
                cur_chunk_ = previous_cur;
                allocation_pins_.clear();
                return false;
            }
            if (state.on_host) {
                if (!ensure_host_ready(missing[i])) {
                    cur_chunk_ = previous_cur;
                    allocation_pins_.clear();
                    return false;
                }
                if (!copy_chunk(state, block, false)) {
                    paging_failed_ = true;
                    synchronize_paging();
                    cur_chunk_ = previous_cur;
                    allocation_pins_.clear();
                    return false;
                }
                state.device_inflight = true;
            }
        }

        free_blocks_.resize(free_blocks_.size() - missing.size());
        for (size_t i = 0; i < missing.size(); ++i) {
            ChunkState & state = chunks_[(size_t) missing[i]];
            state.block = assigned_blocks[i];
            block_to_chunk_[(size_t) assigned_blocks[i]] = missing[i];
            if (assigned_blocks[i] != missing[i]) {
                identity_ = false;
            }
            if (state.on_host) {
                ++stats_.page_ins;
                stats_.moved_bytes += (int64_t) chunk_bytes_;
                state.dirty = false;
                state.snapshot_valid = true;
            }
            ++epoch_;
        }
        if (on_block_paged_in) {
            for (size_t i = 0; i < missing.size(); ++i) {
                const int chunk = missing[i];
                if (chunks_[(size_t) chunk].on_host &&
                    !on_block_paged_in(chunk, assigned_blocks[i])) {
                    paging_failed_ = true;
                    allocation_pins_.clear();
                    return false;
                }
            }
        }
        for (int chunk : allocation_pins_) {
            chunks_[(size_t) chunk].last_use = ++clock_;
        }
        allocation_pins_.clear();
        return true;
    }

    bool select_victims(int count, std::vector<int> & victims) const {
        victims.clear();
        if (count <= 0) {
            return true;
        }

        struct Victim {
            int chunk;
            float score;
            uint64_t use;
        };
        std::vector<Victim> candidates;
        candidates.reserve((size_t) n_blocks_);
        for (int block = 0; block < n_blocks_; ++block) {
            const int chunk = block_to_chunk_[(size_t) block];
            if (chunk < 0) {
                continue;
            }
            const ChunkState & state = chunks_[(size_t) chunk];
            if (state.block != block || is_window_protected(chunk) ||
                is_allocation_pinned(chunk)) {
                continue;
            }
            float score = 0.0f;
            if (score_hook) {
                score = score_hook(chunk);
                if (std::isnan(score)) {
                    score = -std::numeric_limits<float>::infinity();
                }
            }
            candidates.push_back({ chunk, score, state.last_use });
        }
        if ((int) candidates.size() < count) {
            return false;
        }

        auto colder = [this](const Victim & a, const Victim & b) {
            if (score_hook && a.score != b.score) {
                return a.score < b.score;
            }
            return a.use != b.use ? a.use < b.use : a.chunk < b.chunk;
        };
        std::partial_sort(candidates.begin(), candidates.begin() + count,
                candidates.end(), colder);
        victims.reserve((size_t) count);
        for (int i = 0; i < count; ++i) {
            victims.push_back(candidates[(size_t) i].chunk);
        }
        return true;
    }

    bool page_out_batch(const std::vector<int> & chunks) {
        if (paging_failed_ || chunks.empty()) {
            return !paging_failed_;
        }

        std::vector<int> unique = chunks;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        std::vector<int> blocks;
        blocks.reserve(unique.size());

        for (int chunk : unique) {
            if (chunk < 0 || chunk >= (int) chunks_.size() ||
                chunks_[(size_t) chunk].block < 0) {
                return false;
            }
            ChunkState & state = chunks_[(size_t) chunk];
            if (state.block >= n_blocks_ ||
                block_to_chunk_[(size_t) state.block] != chunk) {
                paging_failed_ = true;
                return false;
            }
            if (!allocate_host_backing(state)) {
                paging_failed_ = true;
                return false;
            }
            blocks.push_back(state.block);
        }

        for (int chunk : unique) {
            ChunkState & state = chunks_[(size_t) chunk];
            const bool have_snapshot = state.snapshot_valid && !state.dirty;
            if (have_snapshot) {
                continue;
            }
            if (!copy_chunk(state, state.block, true)) {
                paging_failed_ = true;
                synchronize_paging();
                return false;
            }
            if (has_tensor_storage()) {
                state.host_inflight = true;
                stats_.moved_bytes += (int64_t) chunk_bytes_;
            }
            state.snapshot_valid = true;
            state.dirty = false;
        }

        if (on_block_paged_out) {
            for (size_t i = 0; i < unique.size(); ++i) {
                if (!on_block_paged_out(unique[i], blocks[i])) {
                    paging_failed_ = true;
                    return false;
                }
            }
        }

        if (cfg_.zero_freed_blocks) {
            if (!zero_blocks(blocks) || !synchronize_paging()) {
                paging_failed_ = true;
                return false;
            }
        }

        for (size_t i = 0; i < unique.size(); ++i) {
            ChunkState & state = chunks_[(size_t) unique[i]];
            state.on_host = true;
            state.block = -1;
            block_to_chunk_[(size_t) blocks[i]] = -1;
            free_blocks_.push_back(blocks[i]);
            identity_ = false;
            ++stats_.page_outs;
            ++epoch_;
            if (on_block_evicted) {
                on_block_evicted(blocks[i]);
            }
        }
        return true;
    }

    bool snapshot_resident(int chunk) {
        if (chunk < 0 || chunk >= (int) chunks_.size()) {
            return false;
        }
        ChunkState & state = chunks_[(size_t) chunk];
        if (state.block < 0) {
            return false;
        }
        if (state.snapshot_valid && !state.dirty) {
            return true;
        }
        if (!allocate_host_backing(state)) {
            paging_failed_ = true;
            return false;
        }
        if (has_tensor_storage()) {
            if (!copy_chunk(state, state.block, true)) {
                paging_failed_ = true;
                synchronize_paging();
                return false;
            }
            state.host_inflight = true;
            stats_.moved_bytes += (int64_t) chunk_bytes_;
        }
        state.on_host = true;
        state.snapshot_valid = true;
        state.dirty = false;
        return true;
    }

    bool allocate_host_backing(ChunkState & state) {
        if (!has_tensor_storage() || state.host_ptr) {
            return true;
        }

        if (host_slabs_.empty() ||
            host_slabs_.back().used > host_slabs_.back().capacity ||
            host_slabs_.back().capacity - host_slabs_.back().used < chunk_bytes_) {
            if (!allocate_host_slab()) {
                return false;
            }
        }

        HostSlab & slab = host_slabs_.back();
        uint8_t * base = slab.data();
        if (!base || slab.used > slab.capacity ||
            chunk_bytes_ > slab.capacity - slab.used) {
            return false;
        }
        state.host_ptr = base + slab.used;
        slab.used += chunk_bytes_;
        stats_.host_bytes += (int64_t) chunk_bytes_;
        return true;
    }

    bool allocate_host_slab() {
        if (chunk_bytes_ == 0) {
            return false;
        }

        constexpr size_t target_bytes = 64u * 1024u * 1024u;
        constexpr size_t max_pages = 64;
        size_t pages = std::max<size_t>(1, target_bytes / chunk_bytes_);
        pages = std::min(pages, max_pages);
        if (cfg_.max_context_tokens > 0) {
            const size_t logical_pages =
                    ((size_t) cfg_.max_context_tokens + cfg_.chunk_tokens - 1) /
                    (size_t) cfg_.chunk_tokens;
            const size_t used_pages = (size_t) stats_.host_bytes / chunk_bytes_;
            const size_t remaining_pages =
                    logical_pages > used_pages ? logical_pages - used_pages : 1;
            pages = std::min(pages, remaining_pages);
        }
        if (pages > std::numeric_limits<size_t>::max() / chunk_bytes_) {
            pages = 1;
        }
        size_t capacity = pages * chunk_bytes_;
        const size_t pageable_capacity = capacity;

        HostSlab slab;
        if (host_buft_) {
            while (!slab.buffer) {
                slab.buffer = ggml_backend_buft_alloc_buffer(host_buft_, capacity);
                if (slab.buffer &&
                    ggml_backend_buffer_get_size(slab.buffer) < capacity) {
                    ggml_backend_buffer_free(slab.buffer);
                    slab.buffer = nullptr;
                }
                if (slab.buffer || capacity == chunk_bytes_) {
                    break;
                }
                const size_t smaller_pages =
                        std::max<size_t>(1, capacity / chunk_bytes_ / 2);
                capacity = smaller_pages * chunk_bytes_;
            }
        }
        if (!slab.buffer) {
            capacity = pageable_capacity;
            try {
                slab.storage.resize(capacity);
            } catch (const std::bad_alloc &) {
                return false;
            }
        }
        slab.capacity = capacity;
        if (!slab.data()) {
            if (slab.buffer) {
                ggml_backend_buffer_free(slab.buffer);
            }
            return false;
        }
        host_slabs_.push_back(std::move(slab));
        stats_.host_allocated_bytes += (int64_t) capacity;
        return true;
    }

    void release_host_slabs() {
        for (HostSlab & slab : host_slabs_) {
            if (slab.buffer) {
                ggml_backend_buffer_free(slab.buffer);
                slab.buffer = nullptr;
            }
        }
        host_slabs_.clear();
    }

    bool copy_chunk(ChunkState & state, int block, bool to_host) {
        if (!has_tensor_storage()) {
            return true;
        }
        uint8_t * host = state.host_ptr;
        if (!host) {
            return false;
        }

        size_t host_offset = 0;
        for (const TensorBinding & binding : bindings_) {
            const size_t tensor_offset =
                    (size_t) block * cfg_.chunk_tokens * binding.tensor->nb[1];
            void * host_ptr = host + host_offset;
            if (binding.host) {
                void * tensor_ptr = (uint8_t *) binding.tensor->data + tensor_offset;
                if (to_host) {
                    std::memcpy(host_ptr, tensor_ptr, binding.segment_bytes);
                } else {
                    std::memcpy(tensor_ptr, host_ptr, binding.segment_bytes);
                }
            } else {
                if (binding.backend < 0 ||
                    binding.backend >= (int) page_backends_.size()) {
                    return false;
                }
                ggml_backend_t backend = page_backends_[(size_t) binding.backend];
                if (to_host) {
                    ggml_backend_tensor_get_async(backend, binding.tensor,
                            host_ptr, tensor_offset, binding.segment_bytes);
                } else {
                    ggml_backend_tensor_set_async(backend, binding.tensor,
                            host_ptr, tensor_offset, binding.segment_bytes);
                }
                backend_used_[(size_t) binding.backend] = 1;
            }
            host_offset += binding.segment_bytes;
        }
        return host_offset == chunk_bytes_;
    }

    bool zero_blocks(const std::vector<int> & blocks) {
        if (!has_tensor_storage() || blocks.empty()) {
            return true;
        }
        std::vector<int> sorted = blocks;
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        if (sorted.front() < 0 || sorted.back() >= n_blocks_) {
            return false;
        }

        size_t max_bytes = 0;
        for (size_t i = 0; i < sorted.size();) {
            size_t j = i + 1;
            while (j < sorted.size() && sorted[j] == sorted[j - 1] + 1) {
                ++j;
            }
            const size_t block_count = j - i;
            for (const TensorBinding & binding : bindings_) {
                if (block_count >
                    std::numeric_limits<size_t>::max() / binding.segment_bytes) {
                    return false;
                }
                max_bytes = std::max(max_bytes, block_count * binding.segment_bytes);
            }
            i = j;
        }
        if (zero_buf_.size() < max_bytes) {
            zero_buf_.resize(max_bytes, 0);
        }

        for (size_t i = 0; i < sorted.size();) {
            size_t j = i + 1;
            while (j < sorted.size() && sorted[j] == sorted[j - 1] + 1) {
                ++j;
            }
            const int block_count = (int) (j - i);
            for (const TensorBinding & binding : bindings_) {
                if ((size_t) block_count >
                    std::numeric_limits<size_t>::max() / binding.segment_bytes) {
                    return false;
                }
                const size_t bytes = (size_t) block_count * binding.segment_bytes;
                const size_t offset =
                        (size_t) sorted[i] * cfg_.chunk_tokens * binding.tensor->nb[1];
                if (binding.host) {
                    std::memset((uint8_t *) binding.tensor->data + offset, 0, bytes);
                } else {
                    if (binding.backend < 0 ||
                        binding.backend >= (int) page_backends_.size()) {
                        return false;
                    }
                    ggml_backend_tensor_set_async(
                            page_backends_[(size_t) binding.backend], binding.tensor,
                            zero_buf_.data(), offset, bytes);
                    backend_used_[(size_t) binding.backend] = 1;
                }
            }
            i = j;
        }
        return true;
    }

    bool has_tensor_storage() const {
        return !bindings_.empty() && chunk_bytes_ > 0;
    }

    void release_resources() {
        release_host_slabs();
        chunks_.clear();
        bindings_.clear();
        free_blocks_.clear();
        block_to_chunk_.clear();
        allocation_pins_.clear();
        page_backends_.clear();
        backend_used_.clear();
        zero_buf_.clear();
        host_buft_ = nullptr;
        chunk_bytes_ = 0;
        n_blocks_ = 0;
    }

    KvFlashConfig cfg_;
    std::vector<TensorBinding> bindings_;
    std::vector<ChunkState> chunks_;
    std::vector<int> free_blocks_;
    std::vector<int> block_to_chunk_;
    std::vector<int> allocation_pins_;
    std::vector<HostSlab> host_slabs_;
    std::vector<ggml_backend_t> page_backends_;
    std::vector<uint8_t> backend_used_;
    std::vector<uint8_t> zero_buf_;
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    KvFlashStats stats_;
    size_t chunk_bytes_ = 0;
    int n_blocks_ = 0;
    int cur_chunk_ = 0;
    uint64_t clock_ = 0;
    uint64_t epoch_ = 0;
    bool identity_ = true;
    bool paging_failed_ = false;
};

struct KvFlashAutoBudget {
    int64_t free_bytes       = 0;
    int64_t reserve_bytes    = 0;
    int64_t bytes_per_token  = 0;
    int     speed_cap_tokens = 16384;
};

inline KvFlashConfig kvflash_config_from_env() {
    KvFlashConfig cfg;
    cfg.chunk_tokens = 64;
    if (const char * chunk = std::getenv("LLAMA_KVFLASH_CHUNK")) {
        const int64_t value = std::strtoll(chunk, nullptr, 10);
        if (value > 0 && value <= std::numeric_limits<int>::max()) {
            cfg.chunk_tokens = (int) value;
        }
    }
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = std::max(1, 128 / cfg.chunk_tokens);
    cfg.zero_freed_blocks = false;
    return cfg;
}

inline int64_t kvflash_reserve_bytes_from_env() {
    int64_t mib = 512;
    if (const char * value = std::getenv("LLAMA_KVFLASH_RESERVE_MIB")) {
        mib = std::max<int64_t>(0, std::strtoll(value, nullptr, 10));
    }
    if (mib > std::numeric_limits<int64_t>::max() / (1024 * 1024)) {
        return std::numeric_limits<int64_t>::max();
    }
    return mib * 1024 * 1024;
}

inline int kvflash_pool_from_env(int max_ctx, const KvFlashConfig & cfg = {},
                                 bool scorer_expected = false,
                                 const KvFlashAutoBudget & budget = {}) {
    const char * env = std::getenv("LLAMA_KVFLASH");
    if (!env || max_ctx <= 0 || cfg.chunk_tokens <= 0) {
        return 0;
    }

    int64_t tokens = 0;
    if (std::strcmp(env, "auto") == 0) {
        int64_t speed_cap = budget.speed_cap_tokens;
        if (const char * value = std::getenv("LLAMA_KVFLASH_MAX_POOL")) {
            speed_cap = std::max<int64_t>(256, std::strtoll(value, nullptr, 10));
        }
        if (budget.bytes_per_token > 0 && budget.free_bytes > 0) {
            const int64_t denom = scorer_expected ? 4 : 2;
            const int64_t usable =
                    std::max<int64_t>(0, budget.free_bytes - budget.reserve_bytes) / denom;
            tokens = std::min<int64_t>(usable / budget.bytes_per_token,
                    std::min<int64_t>(max_ctx, speed_cap));
        } else {
            std::fprintf(stderr,
                    "[kvflash] auto requires a post-weight VRAM budget; disabling\n");
            return 0;
        }
    } else {
        tokens = std::strtoll(env, nullptr, 10);
    }
    if (tokens <= 0 || cfg.sink_chunks < 0 || cfg.tail_window_chunks < 0) {
        return 0;
    }

    const int64_t alignment = std::lcm<int64_t>(256, cfg.chunk_tokens);
    const int minimum = KvFlashPager::min_pool_tokens(cfg);
    if (minimum <= 0 || minimum > max_ctx) {
        std::fprintf(stderr,
                "[kvflash] context %d cannot fit minimum pool %d; disabling\n",
                max_ctx, minimum);
        return 0;
    }
    auto align_up = [alignment](int64_t value) {
        return (value / alignment + (value % alignment != 0)) * alignment;
    };
    const int64_t floor_tokens = align_up(minimum);
    if (floor_tokens > max_ctx) {
        std::fprintf(stderr,
                "[kvflash] context %d cannot fit aligned pool %lld; disabling\n",
                max_ctx, (long long) floor_tokens);
        return 0;
    }

    if (tokens <= max_ctx) {
        tokens = align_up(tokens);
    }
    tokens = std::max(tokens, floor_tokens);
    if (tokens > max_ctx) {
        tokens = (max_ctx / alignment) * alignment;
    }
    return tokens >= floor_tokens ? (int) tokens : 0;
}

inline bool kvflash_fill_rows_and_masks(
        const KvFlashPager & pager,
        int kv_start, int n_tok, int mk_w, int swa_window,
        std::vector<int32_t> & rows,
        std::vector<float> * mfull, std::vector<float> * mswa) {
    if (kv_start < 0 || n_tok < 0 || mk_w < 0 ||
        n_tok > std::numeric_limits<int>::max() - kv_start ||
        (n_tok > 0 && (size_t) mk_w >
            std::numeric_limits<size_t>::max() / (size_t) n_tok)) {
        return false;
    }

    rows.resize((size_t) n_tok);
    for (int i = 0; i < n_tok; ++i) {
        const int slot = pager.slot_of((int64_t) kv_start + i);
        if (slot < 0) {
            return false;
        }
        rows[(size_t) i] = slot;
    }
    if (!mfull) {
        return true;
    }

    std::vector<int32_t> slot_pos((size_t) pager.pool_tokens(), -1);
    pager.fill_slot_pos(slot_pos.data());
    mfull->assign((size_t) mk_w * n_tok, -INFINITY);
    if (mswa) {
        mswa->assign((size_t) mk_w * n_tok, -INFINITY);
    }

    const int slot_count = std::min(mk_w, (int) slot_pos.size());
    for (int query = 0; query < n_tok; ++query) {
        const int abs_query = kv_start + query;
        const int window_low = swa_window > 0 ?
                std::max(0, abs_query - swa_window + 1) : 0;
        for (int slot = 0; slot < slot_count; ++slot) {
            const int pos = slot_pos[(size_t) slot];
            if (pos < 0 || pos > abs_query) {
                continue;
            }
            (*mfull)[(size_t) query * mk_w + slot] = 0.0f;
            if (mswa && pos >= window_low) {
                (*mswa)[(size_t) query * mk_w + slot] = 0.0f;
            }
        }
    }
    return true;
}

} // namespace common_kvflash
