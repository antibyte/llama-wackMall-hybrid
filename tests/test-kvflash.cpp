// Unit tests for common/kvflash_pager.

#include "kvflash_pager.h"
#include "kvflash_scorer.h"
#include "kvflash_qk.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static bool fill_sequential(KvFlashPager & pager, int64_t start, int n_tok) {
    const int chunk = pager.chunk_tokens();
    if (chunk <= 0 || n_tok < 0) {
        return false;
    }
    for (int off = 0; off < n_tok; off += chunk) {
        const int n = std::min(chunk, n_tok - off);
        if (!pager.alloc_span(start + off, n)) {
            return false;
        }
    }
    return true;
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

    KvFlashConfig invalid = c;
    invalid.chunk_tokens = 0;
    assert(KvFlashPager::min_pool_tokens(invalid) == 0);
    invalid.chunk_tokens = std::numeric_limits<int>::max();
    invalid.sink_chunks = std::numeric_limits<int>::max();
    assert(KvFlashPager::min_pool_tokens(invalid) == std::numeric_limits<int>::max());
    std::printf("  ok min_pool + attach\n");
}

static void test_transactional_attach_and_bounds() {
    KvFlashPager p;
    KvFlashConfig good = cfg_pool(256, 64, 1, 1);
    good.max_context_tokens = 512;
    assert(p.attach(good, {}, {}));

    KvFlashConfig bad = good;
    bad.chunk_tokens = 0;
    assert(!p.attach(bad, {}, {}));
    bad = good;
    bad.max_context_tokens = -1;
    assert(!p.attach(bad, {}, {}));
    assert(p.attached());
    assert(p.pool_tokens() == 256);

    assert(p.slot_for(-1) == -1);
    assert(p.slot_for(512) == -1);
    assert(p.slot_for((int64_t) std::numeric_limits<int>::max() + 1) == -1);
    assert(p.slot_of(-1) == -1);
    assert(p.block_of(-1) == -1);
    assert(!p.is_resident(-1));
    assert(p.resident_blocks() == 0);

    assert(!p.set_block_order({0, 1, 2}));
    assert(!p.set_block_order({0, 1, 1, 3}));
    assert(!p.set_block_order({0, 1, 2, 4}));
    assert(p.set_block_order({3, 2, 1, 0}));
    assert(p.slot_for(0) == 3 * 64);
    assert(!p.set_block_order({0, 1, 2, 3}));
    std::printf("  ok transactional attach + bounds\n");
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
    assert(fill_sequential(p, 0, 256));
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
    assert(fill_sequential(p, 0, 256));
    assert(p.page_out(1));
    assert(!p.is_resident(1));
    assert(p.stats().page_outs == 1);
    // without tensors, on_host still set; page_in reallocates a free block
    assert(p.page_in(1));
    assert(p.is_resident(1));
    assert(p.stats().page_ins == 1);
    std::printf("  ok page_out/page_in map-only\n");
}

static void test_pager_state_roundtrip_map() {
    KvFlashPager pager;
    KvFlashConfig cfg = cfg_pool(256, 64, 1, 1);
    cfg.max_context_tokens = 1024;
    assert(pager.attach(cfg, {}, {}));
    assert(fill_sequential(pager, 0, 320));

    KvFlashState saved;
    assert(pager.state_export(saved));
    assert(saved.chunk_tokens == cfg.chunk_tokens);
    assert(saved.pool_tokens == cfg.pool_tokens);
    assert(saved.max_context_tokens == cfg.max_context_tokens);
    assert(saved.chunks.size() == 5);
    assert(saved.chunk_bytes == 0);

    std::vector<int> resident;
    int cold = -1;
    for (const auto & chunk : saved.chunks) {
        if (chunk.block >= 0) {
            resident.push_back(chunk.chunk);
        } else {
            cold = chunk.chunk;
        }
    }
    assert(resident.size() == 4);
    assert(cold >= 0);

    assert(!pager.page_in_chunks({-1}));
    assert(!pager.page_in_chunks({saved.cur_chunk + 100}));

    KvFlashState invalid = saved;
    const int first_block = invalid.chunks[(size_t) resident[0]].block;
    invalid.chunks[(size_t) resident[1]].block = first_block;
    assert(!pager.state_import(invalid));

    invalid = saved;
    invalid.cur_chunk = (int32_t) (cfg.max_context_tokens / cfg.chunk_tokens);
    assert(!pager.state_import(invalid));

    assert(pager.slot_for(384) >= 0);
    pager.score_hook = [](int chunk) { return (float) chunk; };
    assert(pager.state_import(saved));
    assert(pager.has_score_hook());

    KvFlashState restored;
    assert(pager.state_export(restored));
    assert(restored.cur_chunk == saved.cur_chunk);
    assert(restored.clock == saved.clock);
    assert(restored.chunks.size() == saved.chunks.size());
    for (size_t i = 0; i < saved.chunks.size(); ++i) {
        assert(restored.chunks[i].chunk == saved.chunks[i].chunk);
        assert(restored.chunks[i].block == saved.chunks[i].block);
        assert(restored.chunks[i].last_use == saved.chunks[i].last_use);
        std::vector<KvFlashStateSpan> spans;
        assert(pager.state_spans(restored.chunks[i].chunk, spans));
        assert(spans.empty());
    }

    assert(pager.page_in_chunks({cold, cold}));
    assert(pager.is_resident(cold));
    std::printf("  ok pager state roundtrip (map-only)\n");
}

static void test_paging_callbacks() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(fill_sequential(p, 0, 256));

    std::vector<int> events;
    int saved_chunk = -1;
    int saved_block = -1;
    p.on_block_paged_out = [&](int chunk, int block) {
        assert(p.is_resident(chunk));
        saved_chunk = chunk;
        saved_block = block;
        events.push_back(1);
        return true;
    };
    p.on_block_evicted = [&](int block) {
        assert(saved_block == block);
        assert(events == std::vector<int>({1}));
        events.push_back(2);
    };
    p.on_block_paged_in = [&](int chunk, int block) {
        assert(chunk == saved_chunk);
        assert(block == p.block_of(chunk));
        assert(events == std::vector<int>({1, 2}));
        events.push_back(3);
        return true;
    };

    assert(p.page_out(1));
    assert(!p.is_resident(1));
    assert(p.page_in(1));
    assert(events == std::vector<int>({1, 2, 3}));
    std::printf("  ok paging callback ordering\n");
}

static void test_reselect() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(fill_sequential(p, 0, 320)); // chunks 0..4, one eviction already
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

static void test_reselect_protection_and_lru_noop() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(fill_sequential(p, 0, 320));

    const KvFlashStats before = p.stats();
    assert(p.reselect() == 0);
    assert(p.stats().page_outs == before.page_outs);
    assert(p.stats().page_ins == before.page_ins);

    p.score_hook = [](int chunk) {
        if (chunk == 1) {
            return std::numeric_limits<float>::infinity();
        }
        if (chunk == 2) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return -std::numeric_limits<float>::infinity();
    };
    assert(p.reselect() >= 0);
    assert(p.is_resident(0));
    assert(p.is_resident(1));
    assert(p.is_resident(3));
    assert(p.is_resident(4));
    p.reset();
    assert(p.has_score_hook());
    std::printf("  ok structural protection + no-score reselect noop\n");
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

static void test_partial_slot_mask() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(p.alloc_span(0, 128));

    std::vector<uint16_t> m16(256);
    p.fill_slot_mask(m16.data(), 70);
    for (int i = 0; i <= 70; ++i) {
        assert(m16[i] == 0x0000);
    }
    for (int i = 71; i < 256; ++i) {
        assert(m16[i] == 0xFC00);
    }

    std::vector<float> mf(256);
    p.fill_slot_mask_f32(mf.data(), 70);
    for (int i = 0; i <= 70; ++i) {
        assert(mf[i] == 0.0f);
    }
    for (int i = 71; i < 256; ++i) {
        assert(std::isinf(mf[i]) && mf[i] < 0.0f);
    }
    std::printf("  ok partial slot mask\n");
}

static void test_shuffled_placement() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    // Hand out blocks in reverse: first alloc gets block 3
    assert(p.set_block_order({3, 2, 1, 0}));
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
    const int a0 = kvflash_pool_from_env(8192, c, /*scorer=*/false);
    assert(a0 == 0);

    KvFlashAutoBudget budget;
    budget.free_bytes = 4ll * 1024 * 1024 * 1024;
    budget.reserve_bytes = 512ll * 1024 * 1024;
    budget.bytes_per_token = 2048;
    const int a = kvflash_pool_from_env(8192, c, /*scorer=*/false, budget);
    assert(a > 0 && a <= 8192);

    setenv("LLAMA_KVFLASH", "256", 1);
    assert(kvflash_pool_from_env(400, c) == 0);
    assert(kvflash_pool_from_env(500, c) == 0);

    setenv("LLAMA_KVFLASH", "9223372036854775807", 1);
    assert(kvflash_pool_from_env(8192, c) == 8192);

    KvFlashConfig c96 = cfg_pool(0, 96, 1, 1);
    setenv("LLAMA_KVFLASH", "1000", 1);
    assert(kvflash_pool_from_env(1536, c96) == 1536);

    unsetenv("LLAMA_KVFLASH");

    unsetenv("LLAMA_KVFLASH_CHUNK");
    KvFlashConfig from_env = kvflash_config_from_env();
    assert(from_env.chunk_tokens == 64);
    assert(from_env.tail_window_chunks == 2);
    assert(KvFlashPager::min_pool_tokens(from_env) == 320);
    setenv("LLAMA_KVFLASH_CHUNK", "128", 1);
    from_env = kvflash_config_from_env();
    assert(from_env.chunk_tokens == 128);
    assert(from_env.tail_window_chunks == 1);
    unsetenv("LLAMA_KVFLASH_CHUNK");
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
    assert(!kvflash_fill_rows_and_masks(p, -1, 4, 256, 0, rows, &mfull, nullptr));
    assert(!kvflash_fill_rows_and_masks(p, 0, -1, 256, 0, rows, &mfull, nullptr));
    assert(!kvflash_fill_rows_and_masks(p, 0, 4, -1, 0, rows, &mfull, nullptr));
    std::printf("  ok fill_rows_and_masks\n");
}

static void test_qk_score_math() {
    KvFlashQkLayer layer;
    layer.model_il = 0;
    layer.n_head = 4;
    layer.n_head_kv = 2;
    layer.n_embd_head = 4;
    layer.type_k = GGML_TYPE_F32;

    KvFlashTargetQkScorer scorer;
    assert(scorer.configure({layer}));

    std::vector<float> k0(8, 0.0f);
    k0[0] = 1.0f;
    k0[4] = 1.0f;
    std::vector<float> k1(8, 0.0f);
    k1[1] = 1.0f;
    k1[5] = 1.0f;
    assert(scorer.set_k_chunk(0, k0));
    assert(scorer.set_k_chunk(1, k1));

    std::vector<float> q(16, 0.0f);
    q[0] = 1.0f;
    q[4] = 1.0f;
    q[8] = 1.0f;
    q[12] = 1.0f;
    assert(scorer.set_q_layer(0, q.data()));
    assert(scorer.score_of(0) > scorer.score_of(1));

    std::vector<float> row(8, 0.0f);
    row[0] = 2.0f;
    row[4] = 2.0f;
    std::vector<uint8_t> packed((size_t) 2 * 8 * sizeof(float));
    std::memcpy(packed.data(), row.data(), 8 * sizeof(float));
    std::memcpy(packed.data() + 8 * sizeof(float), row.data(), 8 * sizeof(float));
    std::vector<float> means;
    assert(kvflash_qk_pool_k_means(GGML_TYPE_F32, packed.data(), 2, 8, 2, 4, means));
    assert(means.size() == 8);
    float nrm = 0.0f;
    for (int i = 0; i < 4; ++i) {
        nrm += means[(size_t) i] * means[(size_t) i];
    }
    assert(std::fabs(nrm - 1.0f) < 1e-5f);

    assert(kvflash_qk_type_supported(GGML_TYPE_Q4_0));
    std::vector<uint8_t> q4((size_t) ggml_row_size(GGML_TYPE_Q4_0, 32), 0);
    const ggml_fp16_t one = ggml_fp32_to_fp16(1.0f);
    std::memcpy(q4.data(), &one, sizeof(one));
    q4[sizeof(one)] = 0x09; // nibble 9 -> +1, high nibble 0 -> -8 at index 16
    std::vector<float> q4_row(32, 0.0f);
    assert(kvflash_qk_dequant_row(GGML_TYPE_Q4_0, q4.data(), q4_row.data(), 32));
    assert(std::fabs(q4_row[0] - 1.0f) < 1e-5f);
    assert(std::fabs(q4_row[16] + 8.0f) < 1e-5f);
    std::printf("  ok qk score math\n");
}

static void test_alloc_span_transactional() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    // 5 chunks cannot be pinned together (max = n_blocks - sink - 1 = 2)
    assert(!p.alloc_span(0, 320));
    assert(p.resident_blocks() == 0);
    assert(p.alloc_span(0, 128));
    assert(p.resident_blocks() == 2);
    assert(p.is_resident(0));
    assert(p.is_resident(1));
    std::printf("  ok transactional alloc_span\n");
}

static void test_snapshot_and_dirty_skip() {
    KvFlashPager p;
    assert(p.attach(cfg_pool(256, 64, 1, 1), {}, {}));
    assert(fill_sequential(p, 0, 256));
    p.mark_written({0, 1, 2, 3});
    const int snapped = p.snapshot_sealed();
    assert(snapped == 3); // skips open tail cur_chunk=3
    const int64_t after_snap = p.stats().moved_bytes;
    assert(after_snap >= 0);

    assert(p.page_out(1));
    assert(!p.is_resident(1));
    assert(p.stats().page_outs == 1);
    assert(p.stats().moved_bytes == after_snap); // dirty-skip, already snapshotted

    assert(p.page_in(1));
    const int64_t after_in = p.stats().moved_bytes;
    assert(p.page_out(1));
    assert(p.stats().moved_bytes == after_in); // still clean

    p.mark_written({2});
    assert(p.is_resident(2));
    assert(p.page_out(2));
    assert(!p.is_resident(2));
    std::printf("  ok snapshot + dirty-skip\n");
}

static void test_atomic_microbatch_mapping() {
    KvFlashPager p;
    KvFlashConfig cfg = cfg_pool(256, 64, 1, 1);
    cfg.max_context_tokens = 1024;
    assert(p.attach(cfg, {}, {}));

    const std::vector<int64_t> too_many = {0, 64, 128};
    assert(!p.can_map_positions(too_many));
    assert(!p.alloc_positions(too_many));
    assert(p.resident_blocks() == 0);
    assert(!p.can_map_positions({0, 0}));

    assert(p.alloc_positions({0, 63, 64, 127}));
    assert(p.is_resident(0));
    assert(p.is_resident(1));
    assert(p.slot_for(128) >= 0);
    assert(p.slot_for(192) >= 0);
    assert(p.resident_blocks() == 4);

    assert(p.alloc_positions({256, 319, 320, 383}));
    assert(p.is_resident(4));
    assert(p.is_resident(5));
    assert(p.is_resident(0));
    assert(p.resident_blocks() == 4);
    std::printf("  ok atomic micro-batch mapping\n");
}

static void test_tensor_roundtrip(ggml_backend_dev_t device) {
    assert(device != nullptr);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    assert(backend);

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);

    constexpr int width = 4;
    constexpr int pool = 256;
    constexpr int chunk = 64;
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, pool, 1);
    ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, pool, 1);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    assert(buffer);

    std::vector<float> k_data((size_t) width * pool);
    std::vector<float> v_data((size_t) width * pool);
    for (int pos = 0; pos < pool; ++pos) {
        for (int i = 0; i < width; ++i) {
            k_data[(size_t) pos * width + i] = (float) (1000 + pos * width + i);
            v_data[(size_t) pos * width + i] = (float) (-1000 - pos * width - i);
        }
    }
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));

    KvFlashConfig cfg = cfg_pool(pool, chunk, 1, 1);
    cfg.max_context_tokens = 512;
    cfg.zero_freed_blocks = false;
    KvFlashPager pager;
    assert(pager.attach(cfg, {k}, {v}));
    assert(pager.bind_backends({backend.get()}));
    assert(fill_sequential(pager, 0, pool));
    assert(pager.page_out(1));

    KvFlashState saved;
    assert(pager.state_export(saved));
    assert(saved.tensors.size() == 2);
    assert(saved.tensors[0].type == GGML_TYPE_F32);
    assert(saved.tensors[1].type == GGML_TYPE_F32);
    assert(saved.chunks.size() == 4);

    std::vector<std::vector<uint8_t>> saved_payloads(saved.chunks.size());
    for (size_t i = 0; i < saved.chunks.size(); ++i) {
        std::vector<KvFlashStateSpan> spans;
        assert(pager.state_spans(saved.chunks[i].chunk, spans));
        saved_payloads[i].resize((size_t) saved.chunk_bytes);
        size_t copied = 0;
        for (const auto & span : spans) {
            assert(span.payload_offset == copied);
            if (span.tensor) {
                ggml_backend_tensor_get(span.tensor, saved_payloads[i].data() + copied,
                        span.tensor_offset, span.size);
            } else {
                assert(span.host || span.size == 0);
                std::memcpy(saved_payloads[i].data() + copied, span.host, span.size);
            }
            copied += span.size;
        }
        assert(copied == saved.chunk_bytes);
    }

    assert(pager.slot_for(256) == chunk);

    std::vector<float> marker((size_t) width * chunk, 42.0f);
    const size_t block_one_offset = (size_t) chunk * k->nb[1];
    const size_t segment_bytes = (size_t) chunk * k->nb[1];
    ggml_backend_tensor_set(k, marker.data(), block_one_offset, segment_bytes);
    ggml_backend_tensor_set(v, marker.data(), block_one_offset, segment_bytes);

    assert(pager.page_in(1));
    assert(pager.block_of(1) == 2);
    assert(pager.synchronize_paging());

    std::vector<float> got_k((size_t) width * chunk);
    std::vector<float> got_v((size_t) width * chunk);
    const size_t recalled_offset = (size_t) pager.block_of(1) * chunk * k->nb[1];
    ggml_backend_tensor_get(k, got_k.data(), recalled_offset, segment_bytes);
    ggml_backend_tensor_get(v, got_v.data(), recalled_offset, segment_bytes);

    const auto k_first = k_data.begin() + (size_t) chunk * width;
    const auto v_first = v_data.begin() + (size_t) chunk * width;
    assert(std::equal(got_k.begin(), got_k.end(), k_first));
    assert(std::equal(got_v.begin(), got_v.end(), v_first));
    assert(pager.stats().page_outs == 2);
    assert(pager.stats().page_ins == 1);
    assert(pager.stats().host_bytes == 4 * (int64_t) segment_bytes);
    assert(pager.stats().host_allocated_bytes >= pager.stats().host_bytes);
    assert(pager.host_slab_count() == 1);

    assert(pager.state_import(saved));
    for (size_t i = 0; i < saved.chunks.size(); ++i) {
        std::vector<KvFlashStateSpan> spans;
        assert(pager.state_spans(saved.chunks[i].chunk, spans));
        size_t copied = 0;
        for (const auto & span : spans) {
            assert(span.payload_offset == copied);
            if (span.tensor) {
                ggml_backend_tensor_set(span.tensor, saved_payloads[i].data() + copied,
                        span.tensor_offset, span.size);
            } else {
                assert(span.host || span.size == 0);
                std::memcpy(span.host, saved_payloads[i].data() + copied, span.size);
            }
            copied += span.size;
        }
        assert(copied == saved.chunk_bytes);
    }

    for (int logical_chunk = 0; logical_chunk < pool / chunk; ++logical_chunk) {
        assert(pager.slot_for(logical_chunk * chunk) >= 0);
        assert(pager.synchronize_paging());
        const size_t offset = (size_t) pager.block_of(logical_chunk) * chunk * k->nb[1];
        ggml_backend_tensor_get(k, got_k.data(), offset, segment_bytes);
        ggml_backend_tensor_get(v, got_v.data(), offset, segment_bytes);
        const auto expected_k = k_data.begin() + (size_t) logical_chunk * chunk * width;
        const auto expected_v = v_data.begin() + (size_t) logical_chunk * chunk * width;
        assert(std::equal(got_k.begin(), got_k.end(), expected_k));
        assert(std::equal(got_v.begin(), got_v.end(), expected_v));
    }

    pager.reset();
    assert(pager.host_slab_count() == 0);
    assert(pager.stats().host_bytes == 0);
    assert(pager.stats().host_allocated_bytes == 0);

    assert(fill_sequential(pager, 0, pool));
    ggml_backend_tensor_set(k, k_data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, v_data.data(), 0, ggml_nbytes(v));
    pager.mark_written({0, 1, 2, 3});
    assert(pager.snapshot_sealed() == 3);
    const int64_t snap_bytes = pager.stats().moved_bytes;
    assert(snap_bytes > 0);
    assert(pager.page_out(1));
    assert(pager.stats().moved_bytes == snap_bytes);
    assert(pager.page_in(1));
    const int64_t after_in = pager.stats().moved_bytes;
    assert(pager.page_out(1));
    assert(pager.stats().moved_bytes == after_in);
    pager.mark_written({2});
    assert(pager.page_out(2));
    assert(pager.stats().moved_bytes > after_in);
    assert(pager.synchronize_paging());

    std::printf("  ok tensor roundtrip on %s\n", ggml_backend_dev_name(device));
}

static std::vector<float> turbo4_test_source() {
    constexpr int width = 128;
    constexpr int rows = 3;
    std::vector<float> result((size_t) width * rows);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < width; ++col) {
            const float x = (float) (row * width + col);
            result[(size_t) row * width + col] =
                    std::sin(0.071f * x) + 0.37f * std::cos(0.193f * x) + 0.1f * row;
        }
    }
    return result;
}

static std::vector<float> run_turbo4_get_rows(ggml_backend_dev_t device) {
    assert(device);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    assert(backend);

    constexpr int width = 128;
    constexpr int rows = 3;
    constexpr int gather_rows = 4;
    constexpr int graph_nodes = 16;
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);

    ggml_tensor * source = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_TURBO4_K, width, rows);
    ggml_tensor * set_rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows);
    ggml_tensor * get_rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, gather_rows);
    ggml_tensor * quantized = ggml_set_rows(ctx.get(), storage, source, set_rows);
    ggml_tensor * output = ggml_get_rows(ctx.get(), quantized, get_rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
    ggml_build_forward_expand(graph, output);
    assert(ggml_backend_dev_supports_op(device, quantized));
    assert(ggml_backend_dev_supports_op(device, output));

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    assert(buffer);

    const std::vector<float> source_data = turbo4_test_source();
    const int32_t set_data[rows] = {0, 1, 2};
    const int32_t get_data[gather_rows] = {2, 0, 2, 1};
    ggml_backend_tensor_set(source, source_data.data(), 0, ggml_nbytes(source));
    ggml_backend_tensor_set(set_rows, set_data, 0, sizeof(set_data));
    ggml_backend_tensor_set(get_rows, get_data, 0, sizeof(get_data));

    assert(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result((size_t) width * gather_rows);
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    return result;
}

static void assert_turbo4_get_rows_semantics(const std::vector<float> & output) {
    constexpr int width = 128;
    constexpr int gather_rows = 4;
    const int source_rows[gather_rows] = {2, 0, 2, 1};
    const std::vector<float> source = turbo4_test_source();
    double error = 0.0;
    double signal = 0.0;
    for (int row = 0; row < gather_rows; ++row) {
        for (int col = 0; col < width; ++col) {
            const float expected = source[(size_t) source_rows[row] * width + col];
            const float actual = output[(size_t) row * width + col];
            const double delta = (double) actual - expected;
            error += delta * delta;
            signal += (double) expected * expected;
        }
    }
    assert(signal > 0.0);
    assert(error / signal < 0.05);
}

static void test_cpu_turbo4_get_rows() {
    const auto output = run_turbo4_get_rows(
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    assert_turbo4_get_rows_semantics(output);
    std::printf("  ok Turbo4 GET_ROWS semantics on CPU\n");
}

static void test_gpu_turbo4_get_rows_if_requested() {
    if (!std::getenv("KVFLASH_TEST_GPU")) {
        std::printf("  skip GPU Turbo4 GET_ROWS (set KVFLASH_TEST_GPU=1)\n");
        return;
    }
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) {
        std::printf("  skip GPU Turbo4 GET_ROWS (no GPU backend)\n");
        return;
    }

    const auto cpu = run_turbo4_get_rows(
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    const auto gpu = run_turbo4_get_rows(device);
    assert_turbo4_get_rows_semantics(gpu);
    assert(cpu.size() == gpu.size());
    double error = 0.0;
    double signal = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        const double delta = (double) cpu[i] - gpu[i];
        error += delta * delta;
        signal += (double) cpu[i] * cpu[i];
    }
    assert(signal > 0.0);
    assert(error / signal < 1e-4);
    std::printf("  ok Turbo4 GET_ROWS CPU/GPU roundtrip\n");
}

static void test_mla_k_only_attach() {
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(device);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    assert(backend);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    assert(ctx);

    constexpr int width = 4;
    constexpr int pool = 256;
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, pool, 1);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    assert(buffer);

    KvFlashConfig cfg = cfg_pool(pool, 64, 1, 1);
    cfg.max_context_tokens = 512;
    KvFlashPager pager;
    assert(!pager.attach(cfg, {k}, {k, nullptr}));
    assert(pager.attach(cfg, {k}, {nullptr}));
    assert(pager.bind_backends({backend.get()}));
    assert(fill_sequential(pager, 0, pool));
    assert(pager.page_out(1));
    assert(pager.stats().page_outs == 1);
    std::printf("  ok MLA K-only attach\n");
}

static void test_cpu_tensor_roundtrip() {
    test_tensor_roundtrip(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
}

static void test_gpu_tensor_roundtrip_if_requested() {
    if (!std::getenv("KVFLASH_TEST_GPU")) {
        std::printf("  skip GPU tensor roundtrip (set KVFLASH_TEST_GPU=1)\n");
        return;
    }
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) {
        std::printf("  skip GPU tensor roundtrip (no GPU backend)\n");
        return;
    }
    test_tensor_roundtrip(device);
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

static void test_long_context_lru() {
    KvFlashPager p;
    KvFlashConfig cfg = cfg_pool(512, 64, 1, 2);
    cfg.max_context_tokens = 1 << 20;
    assert(p.attach(cfg, {}, {}));

    for (int pos = 0; pos < (1 << 20); pos += cfg.chunk_tokens) {
        assert(p.slot_for(pos) >= 0);
    }
    assert(p.n_chunks() == (1 << 20) / cfg.chunk_tokens);
    assert(p.resident_blocks() == cfg.pool_tokens / cfg.chunk_tokens);
    assert(p.is_resident(0));
    assert(!p.is_identity());
    assert(p.stats().page_outs > 16000);

    p.reset();
    assert(p.is_identity());
    assert(p.resident_blocks() == 0);
    std::printf("  ok long-context pool-bounded LRU\n");
}

int main() {
    std::printf("test-kvflash\n");
    test_min_pool();
    test_transactional_attach_and_bounds();
    test_identity_append();
    test_lru_eviction();
    test_page_in_out_map();
    test_pager_state_roundtrip_map();
    test_paging_callbacks();
    test_reselect();
    test_reselect_protection_and_lru_noop();
    test_slot_mask();
    test_partial_slot_mask();
    test_shuffled_placement();
    test_env_pool();
    test_fill_rows_masks();
    test_qk_score_math();
    test_alloc_span_transactional();
    test_snapshot_and_dirty_skip();
    test_atomic_microbatch_mapping();
    test_live_eviction_past_pool();
    test_long_context_lru();
    test_mla_k_only_attach();
    test_cpu_tensor_roundtrip();
    test_cpu_turbo4_get_rows();
    test_gpu_tensor_roundtrip_if_requested();
    test_gpu_turbo4_get_rows_if_requested();
    std::printf("all kvflash unit tests passed\n");
    return 0;
}
