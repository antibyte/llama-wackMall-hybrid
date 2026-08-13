#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_sampling_outputs_per_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)

    // DDTree verify: host pointer to parent_ids[n_tokens] (or n_tokens*n_seqs),
    // i32. When non-null, linear-attn layers use tree GDN/SSM kernels.
    // Lifetime must cover graph build + compute of the current ubatch.
    const int32_t * tree_parent_ids = nullptr;
    int32_t         tree_n_tokens   = 0; // tokens in parent_ids per sequence

    // Tree attention visibility: n_nodes×n_nodes row-major uint8 (1 = visible).
    // Tree batch tokens use pos = tree_n_past + dfs_index (unique slots).
    // Queries with tree_n_past <= pos < tree_n_past+n_nodes use this mask among
    // themselves and full visibility to the committed prefix (pos < tree_n_past).
    const uint8_t * tree_visibility = nullptr;
    int32_t         tree_n_nodes    = 0; // incl. root (= tree_n_tokens when one seq)
    int32_t         tree_n_past     = 0; // first tree-token absolute position
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    // KVFlash: full-attention KV resident pool size in tokens (0 = off).
    // When > 0, hybrid models allocate attn KV at this size (not n_ctx_seq)
    // and map logical positions via common_kvflash::KvFlashPager.
    // Set from env LLAMA_KVFLASH (tokens|auto) at context init.
    uint32_t kvflash_pool = 0;
    uint32_t kvflash_chunk = 64;
    // Reselect interval floor (tokens). Effective interval grows with history.
    // Env: LLAMA_KVFLASH_TAU (default 64).
    uint32_t kvflash_tau = 64;
    bool kvflash_qk = false;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
