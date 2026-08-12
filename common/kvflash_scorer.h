// common_kvflash_scorer — pluggable chunk-relevance policy for KvFlashPager.
//
// With no scorer the pager is pure LRU. Implementations (later):
//   - Drafter scorer (DFlash / small Qwen as Memory Indexer)
//   - Target-QK scorer (pooled post-RoPE K vs decode Q)
//
// Ported interface from Luce-Org/lucebox optimizations/kvflash.

#pragma once

#include <cstdint>
#include <vector>

namespace common_kvflash {

struct KvFlashScorer {
    virtual ~KvFlashScorer() = default;

    // Fill out[c] with relevance (higher = keep) for each chunk of `ids`
    // (prompt + generated). Return false to skip reselect that round.
    virtual bool score_chunks(const std::vector<int32_t> & ids,
                              int chunk_tokens,
                              std::vector<float> & out) = 0;
};

} // namespace common_kvflash
