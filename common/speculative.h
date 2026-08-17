#pragma once

#include "llama.h"
#include "common.h"
#include "ddtree.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// peek a local draft GGUF header when --spec-type is unset (first split only)
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

// return the output limits needed by a parallel block-draft decode
common_speculative_output_limits common_speculative_get_draft_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// Inject only the accepted rows of a previously processed flat tree batch.
bool common_speculative_process_tree_path(
        common_speculative * spec,
        const llama_batch & batch,
        const std::vector<int32_t> & rows);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target nextn embeddings to be extracted
bool common_speculative_need_embd_nextn(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// After draft() in DDTree mode: return the last built tree for seq_id.
// out_tree is a pointer into speculative-owned storage (valid until next draft).
// Returns false if no tree was built.
bool common_speculative_get_tree(
        common_speculative * spec,
        llama_seq_id seq_id,
        const common_ddtree ** out_tree);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// Aggregate wall-clock timings collected by speculative implementations.
// Times are microseconds (matching ggml_time_us). Zero when unused.
struct common_speculative_perf {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    // outer API call counts / totals
    size_t  n_call_draft   = 0;
    size_t  n_call_process = 0;
    int64_t t_draft_us     = 0; // common_speculative_draft() wall time
    int64_t t_process_us   = 0; // common_speculative_process() wall time (inject / combined take)

    // draft-dflash internal split (0 for other types)
    int64_t t_draft_decode_us = 0; // llama_decode of the noise block
    int64_t t_draft_sample_us = 0; // host/backend sampling of mask positions
    int64_t t_ddtree_us       = 0; // top-K extract + tree build (DDTree mode)
    size_t  n_draft_blocks    = 0; // number of noise-block decodes
    size_t  n_draft_block_tok = 0; // sum of noise-block token counts (id_last + masks)
    size_t  n_draft_out_tok   = 0; // sum of draft tokens actually returned
    size_t  n_ddtree_builds   = 0; // number of DDTree builds
    size_t  n_ddtree_nodes    = 0; // sum of tree non-root nodes

    // process path classification (draft-dflash)
    size_t n_process_combined   = 0; // fused inject already done in target graph
    size_t n_process_standalone = 0; // host feature gather + encode + inject decode
};

// Fill perf for the first implementation matching `type`, or the first impl if type is NONE.
// Returns false if no matching implementation exists.
bool common_speculative_get_perf(
        const common_speculative * spec,
        common_speculative_type type,
        common_speculative_perf & out);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
