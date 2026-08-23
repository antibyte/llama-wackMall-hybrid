#include "common.h"
#include "llama-expert-bw-profile.h"
#include "ngram-map.h"

#include <cstdio>
#include <string>

static int g_fails = 0;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

int main() {
    // Ling / hybrid GDN: recurrent pos_min is the tail, not missing SWA data.
    expect(common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, true, 0),
           "hybrid FULL n_swa=0 keeps LCP");
    expect(common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_PART, true, 0),
           "hybrid PART n_swa=0 keeps LCP");
    expect(common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, true, 0),
           "RS keeps LCP");
    expect(common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_RS, false, 4096),
           "RS keeps LCP even with n_swa");

    // SWA still needs a checkpoint when the window has slid.
    expect(!common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, true, 4096),
           "hybrid SWA does not keep LCP without checkpoint");
    expect(!common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, false, 0),
           "non-hybrid FULL does not keep LCP");
    expect(!common_keep_lcp_without_checkpoint(COMMON_CONTEXT_SEQ_RM_TYPE_NO, true, 0),
           "no-memory context does not keep LCP");

    expect(!common_prompt_kv_shift_safe(COMMON_CONTEXT_SEQ_RM_TYPE_RS, true, 0),
           "RS/GDN must not seq_add-shift KV");
    expect(!common_prompt_kv_shift_safe(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, true, 0),
           "hybrid n_swa=0 must not seq_add-shift KV");
    expect(!common_prompt_kv_shift_safe(COMMON_CONTEXT_SEQ_RM_TYPE_PART, true, 0),
           "hybrid PART n_swa=0 must not seq_add-shift KV");
    expect(common_prompt_kv_shift_safe(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, true, 4096),
           "SWA hybrid may shift when attn can_shift");
    expect(common_prompt_kv_shift_safe(COMMON_CONTEXT_SEQ_RM_TYPE_PART, false, 0),
           "non-hybrid PART may shift");
    expect(!common_prompt_kv_shift_safe_ctx(nullptr, 0),
           "null context cannot shift");

    // Append: reuse point is past the cached tail.
    expect(!common_recurrent_rollback_exceeds(10, 11, 0),
           "p0 past pos_max needs no rollback");
    expect(!common_recurrent_rollback_exceeds(-1, 5, 0),
           "empty memory needs no rollback");

    // Same-prompt logits trim: 1 token, no snapshots.
    expect(common_recurrent_rollback_exceeds(10, 10, 0),
           "rollback 1 exceeds n_rs=0");
    expect(!common_recurrent_rollback_exceeds(10, 10, 4),
           "rollback 1 fits n_rs=4");
    expect(common_recurrent_rollback_exceeds(10, 5, 4),
           "rollback 6 exceeds n_rs=4");

    expect(common_reuse_prefix_tokens(100, false, 40) == 100,
           "LCP kept when rollback fits");
    expect(common_reuse_prefix_tokens(100, true, 40) == 40,
           "checkpoint kept when rollback exceeds");
    expect(common_reuse_prefix_tokens(100, true, 0) == 0,
           "no checkpoint rebuilds from 0");
    expect(common_reuse_prefix_tokens(100, true, 120) == 0,
           "checkpoint past LCP is ignored");
    expect(common_reuse_prefix_tokens(0, true, 40) == 0,
           "empty LCP keeps nothing");

    expect(common_prefix_checkpoint_tokens({10, 40, 80, 120}, 100) == 80,
           "deepest checkpoint on the LCP");
    expect(common_prefix_checkpoint_tokens({10, 40, 80}, 40) == 40,
           "checkpoint exactly at LCP");
    expect(common_prefix_checkpoint_tokens({80, 120}, 40) == 0,
           "all checkpoints past LCP");
    expect(common_prefix_checkpoint_tokens({}, 40) == 0,
           "no checkpoints");

    expect(common_estimate_reuse_prefix_tokens(100, 100, 80, 2, true) == 100,
           "append keeps full cached prompt");
    expect(common_estimate_reuse_prefix_tokens(2000, 8000, 1800, 2, true) == 1800,
           "GDN rollback uses checkpoint on the LCP");
    expect(common_estimate_reuse_prefix_tokens(2000, 8000, 0, 2, true) == 0,
           "GDN rollback without checkpoint rebuilds");
    expect(common_estimate_reuse_prefix_tokens(3, 5, 0, 4, true) == 3,
           "short tail fits n_rs");
    expect(common_estimate_reuse_prefix_tokens(2000, 8000, 0, 2, false) == 2000,
           "non-GDN keeps token LCP");
    expect(common_estimate_reuse_prefix_tokens(0, 100, 40, 2, true) == 0,
           "empty LCP reuses nothing");

    expect(common_prompt_reuse_skip_unrelated(0, 0, 0.20f),
           "unrelated empty-reuse cache is skipped");
    expect(!common_prompt_reuse_skip_unrelated(1800, 0, 0.22f),
           "better GDN reuse is not skipped for low f_keep");
    expect(common_prompt_reuse_skip_unrelated(100, 500, 0.20f),
           "worse reuse still skipped when unrelated");
    expect(!common_prompt_reuse_skip_unrelated(0, 0, 0.40f),
           "related empty-reuse cache is eligible");

    expect(common_prompt_reuse_better(1800, 80, 0.22f, 0.90f, 0.20f, 0.80f),
           "higher GDN reuse wins over longer token LCP");
    expect(!common_prompt_reuse_better(80, 1800, 0.90f, 0.22f, 0.80f, 0.20f),
           "lower GDN reuse loses");
    expect(common_prompt_reuse_better(80, 80, 0.50f, -1.0f, 0.40f, 0.0f),
           "empty slot takes the first eligible cache");
    expect(common_prompt_reuse_better(80, 80, 0.60f, 0.40f, 0.50f, 0.30f),
           "equal reuse falls back to f_keep and sim");
    expect(!common_prompt_reuse_better(80, 80, 0.60f, 0.40f, 0.20f, 0.30f),
           "equal reuse requires both f_keep and sim to improve");

    expect(common_checkpoint_evict_index({}) == -1, "empty list has no victim");
    expect(common_checkpoint_evict_index({10}) == 0, "single checkpoint is the victim");
    expect(common_checkpoint_evict_index({10, 80}) == 1,
           "with two snapshots drop the later");
    expect(common_checkpoint_evict_index({10, 40, 80, 81, 82}) == 3,
           "keep turn-end, peel the extra next to it");
    expect(common_checkpoint_evict_index({10, 40, 80, 81}) == 2,
           "keep turn-end while collapsing the cluster");
    expect(common_checkpoint_evict_index({10, 11, 80}) == 1,
           "drop the extra next to the prefix snapshot");
    expect(common_checkpoint_evict_index({10, 40, 80}) == 1,
           "keep earliest prefix when gaps are similar");
    expect(common_checkpoint_evict_index({10, 3500, 3510}) == 1,
           "keep prefix and turn-end when only three");

    expect(common_checkpoint_too_dense(3602, 3600, 32), "1-token decode snapshot is dense");
    expect(common_checkpoint_too_dense(3600, 3600, 32), "duplicate n_tokens is dense");
    expect(common_checkpoint_too_dense(3600, 3600, 0), "duplicate is dense even with min_gap 0");
    expect(!common_checkpoint_too_dense(3640, 3600, 32), "32-token gap is kept");
    expect(!common_checkpoint_too_dense(80, 40, 0), "min_gap 0 keeps a later snapshot");
    expect(!common_checkpoint_too_dense(2437, 2416, 0),
           "last-user 21 tokens after turn-end is kept");

    expect(common_checkpoint_should_break(true, false, true, 100, 200, 8192),
           "last user always splits");
    expect(common_checkpoint_should_break(false, true, false, -1, 40, 8192),
           "raw completion: first role/tool splits");
    expect(!common_checkpoint_should_break(false, true, true, -1, 40, 8192),
           "chat prompt: skip system/role before last user");
    expect(!common_checkpoint_should_break(false, true, true, 40, 200, 8192),
           "historical role inside min_step does not split");
    expect(common_checkpoint_should_break(false, true, true, 40, 9000, 8192),
           "role after min_step splits");
    expect(!common_checkpoint_should_break(false, false, true, -1, 40, 8192),
           "non-prefix does not split");

    expect(common_checkpoint_should_snapshot(true, true, false, true, true, true),
           "semantic break still snapshots");
    expect(!common_checkpoint_should_snapshot(false, true, false, true, true, true),
           "GDN skips prompt_done after last_user snapshot");
    expect(common_checkpoint_should_snapshot(false, true, false, true, true, false),
           "GDN prompt_done if last_user snapshot is missing");
    expect(common_checkpoint_should_snapshot(false, true, false, true, false, true),
           "GDN raw completion still snapshots at prompt end");
    expect(common_checkpoint_should_snapshot(false, true, false, true, false, false),
           "GDN empty cache snapshots at prompt end");
    expect(!common_checkpoint_should_snapshot(false, false, true, true, true, true),
           "GDN ignores near_prompt_end");
    expect(common_checkpoint_should_snapshot(false, true, false, false, true, true),
           "non-GDN still snapshots at prompt_done");
    expect(common_checkpoint_should_snapshot(false, false, true, false, true, true),
           "non-GDN still snapshots near prompt end");
    expect(!common_checkpoint_should_snapshot(false, false, false, false, false, false),
           "no snapshot without a reason");

    expect(common_checkpoint_should_split_ubatch(true, -1, 100, 5000, 32),
           "first last_user splits");
    expect(!common_checkpoint_should_split_ubatch(false, -1, 100, 5000, 32),
           "no break means no split");
    expect(!common_checkpoint_should_split_ubatch(true, 90, 100, 5000, 32),
           "dense snapshot does not split");
    expect(common_checkpoint_should_split_ubatch(true, 90, 200, 5000, 32),
           "far from last snapshot splits");
    expect(!common_checkpoint_should_split_ubatch(true, 100, 100, 5000, 32),
           "already snapshotted at pos does not split");
    expect(!common_checkpoint_should_split_ubatch(true, -1, 4980, 5000, 32),
           "last_user inside final 32 tokens does not split");
    expect(common_checkpoint_should_split_ubatch(true, -1, 4900, 5000, 32),
           "last_user 100 tokens before prompt end splits");
    expect(!common_checkpoint_should_split_ubatch(true, -1, 5000, 5000, 32),
           "break at prompt end does not split leftover 0");
    expect(!common_checkpoint_should_split_ubatch(true, -1, 100, 5000, 32, true),
           "GDN wide prefill does not split at last_user");
    expect(common_checkpoint_should_split_ubatch(true, -1, 100, 5000, 32, false),
           "non-GDN still splits at last_user");

    expect(common_cmoe_first_prefill_at(64, 2048) == 256,
           "first prompt prefills at 4*decode");
    expect(common_cmoe_first_prefill_at(64, 128) == 128,
           "first threshold capped by prefill ubatch");
    expect(common_cmoe_first_prefill_at(64, 0) == 64,
           "no prefill ubatch uses decode size");
    expect(common_cmoe_first_prefill_at(0, 2048) == 2048,
           "missing decode ubatch uses prefill size");

    expect(common_cmoe_prefer_prefill(4000, 0, 64, 2048, false),
           "first long prompt uses prefill");
    expect(!common_cmoe_prefer_prefill(200, 0, 64, 2048, false),
           "first prompt under 4*decode stays decode to capture generate graph");
    expect(common_cmoe_prefer_prefill(400, 0, 64, 2048, false),
           "first prompt over 4*decode uses prefill");
    expect(!common_cmoe_prefer_prefill(50, 0, 64, 2048, false),
           "first decode-sized prompt stays decode");
    expect(!common_cmoe_prefer_prefill(200, 3400, 64, 2048, false),
           "follow-up leftover keeps decode graphs");
    expect(common_cmoe_prefer_prefill(3000, 3400, 64, 2048, false),
           "follow-up longer than prefill ubatch uses prefill");
    expect(common_cmoe_prefer_prefill(200, 3400, 64, 2048, true),
           "already prefilling stays until leftover fits decode");
    expect(!common_cmoe_prefer_prefill(50, 3400, 64, 2048, true),
           "prefill remainder switches to decode");
    expect(!common_cmoe_prefer_prefill(0, 3400, 64, 2048, false),
           "no leftover is decode");
    expect(!common_cmoe_prefer_prefill(100, 3400, 64, 0, false),
           "missing prefill ubatch still keeps follow-up on decode");

    expect(common_cmoe_graph_batch(64, 4) == 5,
           "ngram verify size keeps decode graph");
    expect(common_cmoe_graph_batch(64, 0) == 8,
           "no spec caps at mmvq graph batch");
    expect(common_cmoe_graph_batch(4, 4) == 4,
           "decode ubatch already below spec verify");
    expect(common_cmoe_graph_batch(64, 16) == 8,
           "spec larger than mmvq still caps at 8");
    expect(common_cmoe_graph_batch(0, 4) == 5,
           "missing decode ubatch still uses spec verify size");
    expect(common_cmoe_graph_batch(64, -1) == 8,
           "negative spec n_max ignored");

    expect(common_cmoe_use_ngram_leftover_graph(true, false),
           "ngram leftover uses graph-sized ubatch");
    expect(!common_cmoe_use_ngram_leftover_graph(true, true),
           "DFlash leftover keeps decode ubatch");
    expect(!common_cmoe_use_ngram_leftover_graph(false, false),
           "prefill is not ngram leftover");

    expect(common_cmoe_leftover_match_spec_logits(true, 4),
           "decode leftover with spec writes logits on every token");
    expect(!common_cmoe_leftover_match_spec_logits(true, 0),
           "decode leftover without spec keeps prompt logits");
    expect(!common_cmoe_leftover_match_spec_logits(false, 4),
           "prefill leftover does not force spec logits");

    expect(common_spec_draft_noise_trim_p0(13, 0) == 13,
           "stale ckpt pos_max=0 must not trim from 1");
    expect(common_spec_draft_noise_trim_p0(13, 12) == 13,
           "noise block starts at committed pos_next");
    expect(common_spec_draft_noise_trim_p0(1, 0) == 1,
           "single committed token still trims noise at 1");
    expect(common_spec_draft_noise_trim_p0(0, 5) == 6,
           "missing pos_next falls back to ckpt.pos_max+1");
    expect(common_spec_draft_noise_trim_p0(0, -1) == -1,
           "no committed pos skips trim");

    expect(!common_dflash_pos_jump(-1, 13),
           "empty draft is not a jump");
    expect(!common_dflash_pos_jump(12, 13),
           "first verify token at X+1 is consecutive");
    expect(common_dflash_pos_jump(0, 13),
           "draft pos_max=0 vs Y=13 is a jump");
    expect(common_dflash_pos_jump(12, 14),
           "skip of 2 is a jump");
    expect(common_dflash_pos_jump(12, 12),
           "Y=X is a jump");

    expect(common_spec_draft_covers_keep(-1, 0),
           "empty keep needs no draft cover");
    expect(!common_spec_draft_covers_keep(-1, 13),
           "empty draft does not cover n_past=13");
    expect(common_spec_draft_covers_keep(12, 13),
           "dft_max = n_past-1 covers the keep");
    expect(common_spec_draft_covers_keep(20, 13),
           "draft ahead of keep still covers");
    expect(!common_spec_draft_covers_keep(0, 13),
           "pos 0 does not cover n_past=13");

    expect(common_spec_clear_draft_kv(false, false),
           "stale prefix always clears");
    expect(common_spec_clear_draft_kv(true, true),
           "wide prefill skip-inject clears a kept prefix");
    expect(!common_spec_clear_draft_kv(true, false),
           "decode leftover keeps a consecutive DFlash prefix");
    expect(common_spec_clear_draft_kv(false, true),
           "stale plus prefill still clears");
    expect(!common_cmoe_prefer_prefill(50, 3400, 64, 1856, false),
           "follow-up leftover does not prefill just to reseed draft");

    expect(!common_spec_dflash_keep_draft_token(0, 1, 0.50f, 0.75f),
           "first draft below p_min is dropped");
    expect(common_spec_dflash_keep_draft_token(0, 1, 0.80f, 0.75f),
           "first draft at p_min is kept");
    expect(common_spec_dflash_keep_draft_token(1, 1, 0.10f, 0.75f),
           "later draft fills n_max even below p_min");
    expect(common_spec_dflash_keep_draft_token(0, 1, 0.10f, 0.0f),
           "p_min 0 keeps the first draft");
    expect(!common_spec_dflash_keep_draft_token(1, 2, 0.20f, 0.75f),
           "n_min=2 still requires p_min on the second token");
    expect(common_spec_dflash_keep_draft_token(2, 2, 0.20f, 0.75f),
           "after n_min the graph block is filled");

    expect(common_ngram_simple_search_stop(100, 0) == 0,
           "window 0 searches full history");
    expect(common_ngram_simple_search_stop(100, 1000) == 0,
           "window larger than history");
    expect(common_ngram_simple_search_stop(4000, 1024) == 2976,
           "window 1024 on long history");

    {
        const common_ngram_simple_config near{ /*n*/ 2, /*m*/ 2, /*window*/ 1024 };
        const llama_tokens toks = { 0, 0, 6, 7, 8, 6 };
        const auto draft = common_ngram_simple_draft(near, toks, 7);
        expect(!draft.empty() && draft[0] == 8,
               "ngram finds a nearby match");
        const common_ngram_simple_config far{ /*n*/ 2, /*m*/ 2, /*window*/ 3 };
        const auto missed = common_ngram_simple_draft(far, toks, 7);
        expect(missed.empty(),
               "ngram search window skips a distant match");
    }

    expect(!common_checkpoint_is_radix_index(0, 0), "empty list has no radix index");
    expect(common_checkpoint_is_radix_index(0, 1), "single snapshot is kept");
    expect(common_checkpoint_is_radix_index(0, 2) && common_checkpoint_is_radix_index(1, 2),
           "both of two snapshots are radix");
    expect(common_checkpoint_is_radix_index(0, 4) && common_checkpoint_is_radix_index(3, 4),
           "prefix and turn-end are radix");
    expect(!common_checkpoint_is_radix_index(1, 4) && !common_checkpoint_is_radix_index(2, 4),
           "interior snapshots are not radix");
    expect(!common_checkpoint_is_radix_index(-1, 4) && !common_checkpoint_is_radix_index(4, 4),
           "out of range is not radix");

    {
        std::string rec;
        float q = -1.0f;
        const std::string json =
            "{\n  \"schema\": \"llama-wackmall-expert-bw-v1\",\n"
            "  \"recommended\": \"cpu-heavy\",\n  \"q_star\": 0.124\n}\n";
        expect(llama_expert_tier::parse_bw_profile(json, rec, q), "parses indented bw profile");
        expect(rec == "cpu-heavy", "recommended cpu-heavy");
        expect(q > 0.12f && q < 0.13f, "q_star ~0.124");
        expect(!llama_expert_tier::parse_bw_profile("{\"schema\":\"nope\"}", rec, q),
               "rejects unknown schema");
        expect(llama_expert_tier::warm_admit_budget(0, 4, 1.0f) == 0, "no misses");
        expect(llama_expert_tier::warm_admit_budget(6, 0, 1.0f) == 0, "W=0");
        expect(llama_expert_tier::warm_admit_budget(6, 4, 0.0f) == 0, "q*=0");
        expect(llama_expert_tier::warm_admit_budget(6, 4, 1.0f) == 4, "immediate caps at W");
        expect(llama_expert_tier::warm_admit_budget(1, 4, 0.1029f) == 0, "q* skips singleton miss");
        expect(llama_expert_tier::warm_admit_budget(5, 4, 0.1029f) == 1, "q* admits one of five");
        expect(llama_expert_tier::warm_admit_budget(8, 4, 0.5f) == 4, "q* still caps at W");
        expect(llama_expert_tier::warm_keep_on_cpu_heavy(0.1029f), "cpu-heavy keeps W with q* cap");
        expect(!llama_expert_tier::warm_keep_on_cpu_heavy(0.0f), "q*=0 disables W");
        expect(!llama_expert_tier::warm_keep_on_cpu_heavy(1.0f), "q*=1 disables W on cpu-heavy");
        expect(llama_expert_tier::warm_admit_from_counts(1, 32), "decode token may fill W");
        expect(llama_expert_tier::warm_admit_from_counts(2, 32), "DFlash verify may fill W");
        expect(!llama_expert_tier::warm_admit_from_counts(1856, 32), "prefill must not fill W");
        expect(!llama_expert_tier::warm_admit_from_counts(0, 32), "empty graph does not fill W");
        expect(!llama_expert_tier::warm_admit_from_counts(8, 0), "TMAX=0 does not fill W");
    }

    if (g_fails) {
        fprintf(stderr, "%d check(s) failed\n", g_fails);
        return 1;
    }
    printf("test-hybrid-lcp: ok\n");
    return 0;
}
