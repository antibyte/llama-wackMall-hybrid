#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

// backend_set_input() writes the input tensors the last backend_apply() created. A kept graph of another
// shape (llama_context::gf_res_small) gets its own back before it is replayed: get appends one entry per
// sampler with backend inputs, set consumes them in the same order. get returns false for a sampler with
// backend inputs it does not know.
bool llama_sampler_backend_get_inputs(const llama_sampler * smpl, std::vector<std::vector<ggml_tensor *>> & out);
void llama_sampler_backend_set_inputs(llama_sampler * smpl, const std::vector<std::vector<ggml_tensor *>> & in, size_t & pos);

struct llama_sampler * llama_sampler_init_dry_testing(
        int32_t context_size,
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
