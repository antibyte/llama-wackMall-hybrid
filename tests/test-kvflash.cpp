// Unit tests for common/kvflash_pager — map-only (no model, no CUDA).
// Port gates A/B/C from Lucebox test_kvflash (logical residency).

#include "kvflash_pager.h"
#include "kvflash_scorer.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace common_kvflash;

static KvFlashConfig cfg_pool(int pool, int chunk = 64, int sink = 1, int tail = 1) {
    KvFlashConfig c;
    c.chunk_tokens       = chunk;
    c.pool_tokens        = pool;
    c.sink_chunks        = sink;
    c.tail_window_chunks = tail;
    return c;
}

static void test_min_pool() {
    KvFlashConfig c = cfg_pool(512, 64, 1, 4);
    // (1+4+2)*64 = 448
    assert(KvFlashPager::min_pool_tokens(c) == 448);
    KvFlashPager p;
    assert(!p.attach(cfg_pool(256, 64, 1, 4), {}, {})); // too small
    assert(p.attach(cfg_pool(512, 64, 1, 4), {}, {}));
    assert(p.pool_tokens() == 512);
    assert(p.chunk_tokens() == 64);
    std::printf("  ok min_pool + attach\n");
}

static void test_identity_append() {
    // pool = 4 chunks = 256; sink=1 tail=1 → min = 4*64 = 256 exactly
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    for (int i = 0; i < 128; i++) {
        const int s = p.slot_for(i);
        assert(s == i); // identity while under pool and no eviction
    }
    assert(p.is_identity());
    assert(p.identity_prefix_covers(128));
    assert(p.resident_blocks() == 2);
    assert(p.stats().page_outs == 0);
    std::printf("  ok identity append\n");
}

static void test_lru_eviction() {
    // 4 blocks; sink=1 tail=1 → 2 victims available
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));

    // Fill entire pool: chunks 0..3
    assert(p.alloc_span(0, 256));
    assert(p.resident_blocks() == 4);
    assert(p.stats().page_outs == 0);

    // Append chunk 4 → must evict unprotected middle (not sink 0, not tail)
    // cur_chunk becomes 4; tail protects chunks > 4-1-1 = 2 → chunk 3 protected?
    // tail: c > cur_chunk_ - 1 - tail_window_chunks
    // After slot_for(256): cur_chunk=4, tail protects c > 4-1-1 = 2, i.e. c>=3
    // sink protects c < 1. Evictable: c=1,2
    const int s = p.slot_for(256);
    assert(s >= 0);
    assert(p.stats().page_outs >= 1);
    assert(!p.is_resident(1) || !p.is_resident(2)); // at least one middle gone
    assert(p.is_resident(0)); // sink
    // newest chunk resident
    assert(p.is_resident(4));
    assert(p.slot_of(256) >= 0);
    assert(p.slot_of(0) == 0); // sink still identity block if not moved
    std::printf("  ok LRU eviction page_outs=%lld\n", (long long) p.stats().page_outs);
}

static void test_page_in_out_map() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(p.alloc_span(0, 256));
    assert(p.page_out(1));
    assert(!p.is_resident(1));
    assert(p.stats().page_outs == 1);
    // without tensors, on_host still set; page_in reallocates a free block
    assert(p.page_in(1));
    assert(p.is_resident(1));
    assert(p.stats().page_ins == 1);
    std::printf("  ok page_out/page_in map-only\n");
}

static void test_reselect() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(p.alloc_span(0, 320)); // chunks 0..4, one eviction already
    // Force known residency: page out 2 if resident
    if (p.is_resident(2)) {
        p.page_out(2);
    }
    assert(!p.is_resident(2));

    // Prefer chunk 2 over everything non-protected
    p.score_hook = [](int c) -> float {
        return c == 2 ? 100.f : (float) c;
    };
    const int events = p.reselect();
    assert(events >= 1);
    assert(p.is_resident(2));
    assert(p.is_resident(0)); // sink protected
    std::printf("  ok reselect events=%d\n", events);
}

static void test_slot_mask() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(p.alloc_span(0, 128)); // blocks 0,1 resident
    std::vector<uint16_t> m16(256);
    p.fill_slot_mask(m16.data());
    // first 128 slots zero, rest -inf
    for (int i = 0; i < 128; i++) {
        assert(m16[i] == 0x0000);
    }
    for (int i = 128; i < 256; i++) {
        assert(m16[i] == 0xFC00);
    }

    std::vector<float> mf(256);
    p.fill_slot_mask_f32(mf.data());
    for (int i = 0; i < 128; i++) {
        assert(mf[i] == 0.f);
    }
    for (int i = 128; i < 256; i++) {
        assert(std::isinf(mf[i]) && mf[i] < 0);
    }

    std::vector<int32_t> pos(256);
    p.fill_slot_pos(pos.data());
    for (int i = 0; i < 128; i++) {
        assert(pos[i] == i);
    }
    for (int i = 128; i < 256; i++) {
        assert(pos[i] == -1);
    }
    std::printf("  ok slot mask + pos\n");
}

static void test_shuffled_placement() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    // Hand out blocks in reverse: first alloc gets block 3
    p.set_block_order({3, 2, 1, 0});
    const int s0 = p.slot_for(0);
    assert(s0 == 3 * 64 + 0);
    const int s64 = p.slot_for(64);
    assert(s64 == 2 * 64 + 0);
    assert(!p.is_identity());
    assert(p.block_of(0) == 3);
    assert(p.block_of(1) == 2);
    std::printf("  ok shuffled placement\n");
}

static void test_env_pool() {
    // unset
    unsetenv("LLAMA_KVFLASH");
    assert(kvflash_pool_from_env(8192) == 0);

    setenv("LLAMA_KVFLASH", "1000", 1);
    // rounds to 1024, floor for default cfg min
    KvFlashConfig c = cfg_pool(0, 64, 1, 4);
    const int t = kvflash_pool_from_env(8192, c);
    assert(t >= 1024);
    assert(t % 256 == 0);

    setenv("LLAMA_KVFLASH", "auto", 1);
    const int a = kvflash_pool_from_env(8192, c, /*scorer=*/false);
    assert(a > 0 && a <= 8192);

    unsetenv("LLAMA_KVFLASH");
    std::printf("  ok env pool sizing\n");
}

static void test_fill_rows_masks() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(p.alloc_span(0, 64));
    std::vector<int32_t> rows;
    std::vector<float>   mfull;
    assert(kvflash_fill_rows_and_masks(p, 0, 4, /*mk_w=*/256, /*swa=*/0, rows, &mfull, nullptr));
    assert(rows.size() == 4);
    assert(rows[0] == 0 && rows[3] == 3);
    // q=3 can see p<=3
    assert(mfull[3 * 256 + 0] == 0.f);
    assert(mfull[3 * 256 + 3] == 0.f);
    assert(std::isinf(mfull[3 * 256 + 5]) && mfull[3 * 256 + 5] < 0);
    std::printf("  ok fill_rows_and_masks\n");
}

static void test_live_eviction_past_pool() {
    // pool=256 (4 chunks), sink=1 tail=1 → fill beyond pool
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    // Materialize 6 chunks (0..5) → forces eviction
    for (int pos = 0; pos < 384; pos++) {
        assert(p.slot_for(pos) >= 0);
    }
    assert(p.stats().page_outs >= 1);
    assert(p.resident_blocks() <= 4);
    // Sink still resident
    assert(p.is_resident(0));
    // Newest chunk resident
    assert(p.is_resident(5) || p.is_resident(4));
    // Reselect with default LRU should keep recent + sink
    const int ev = p.reselect();
    (void) ev;
    assert(p.is_resident(0));
    assert(p.resident_blocks() <= 4);
    std::printf("  ok live eviction past pool (page_outs=%lld reselect_ev=%d)\n",
            (long long) p.stats().page_outs, ev);
}

int main() {
    std::printf("test-kvflash\n");
    test_min_pool();
    test_identity_append();
    test_lru_eviction();
    test_page_in_out_map();
    test_reselect();
    test_slot_mask();
    test_shuffled_placement();
    test_env_pool();
    test_fill_rows_masks();
    test_live_eviction_past_pool();
    std::printf("all kvflash unit tests passed\n");
    return 0;
}
