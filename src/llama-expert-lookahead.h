#pragma once

#include "ggml-backend.h"

#include <cstdint>

class llm_graph_result;
struct llama_model;

namespace llama_expert_lookahead {

enum class prediction_point {
    post_attn,
    post_moe,
};

enum class norm_source {
    target,
    source,
};

// These configuration queries are used during graph construction. They parse
// environment controls without requiring the runtime trace collector to have
// been initialized yet.
bool graph_enabled(uint32_t n_tokens, uint32_t n_seqs, bool mtp_graph);
prediction_point point();
norm_source norm();
int distance();
int top_m(int n_expert);

void configure_mtp(int mtp_n);
void configure_request_scoped(bool enabled);
void init(const llama_model & model);

void request_begin();
void request_end();

// Submit small trace readbacks after graph execution. This function never
// synchronizes a backend. complete_graph() must only be called after the
// existing post-graph scheduler synchronization.
void enqueue_graph(ggml_backend_sched_t sched, const llm_graph_result & result, uint32_t n_ctx);
void complete_graph();

bool enabled();

} // namespace llama_expert_lookahead
