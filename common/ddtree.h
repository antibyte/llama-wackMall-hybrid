// DDTree - Diffusion Draft Tree for DFlash speculative decoding.
//
// Port of build_ddtree from Lucebox (liranringel/ddtree / dflash::common::ddtree).
// Builds a best-first tree from per-position top-K log-probability distributions
// of a block-diffusion drafter for tree-structured verification.
//
// Self-contained: standard library only.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

// A flat tree built from the draft's top-K softmax distributions.
// Slot 0 is the tree root (previous bonus / last accepted token);
// slots 1..n_nodes are the tree nodes in stable depth order.
struct common_ddtree {
    int                         n_nodes = 0; // excludes root
    std::vector<int32_t>        token_ids;   // size n_nodes
    std::vector<int>            depths;      // size n_nodes (1..L)
    std::vector<int32_t>        parents;     // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps; // size n_nodes + 1
    // (1 + n_nodes)^2 row-major: visibility[i*N+j] = 1 if j is ancestor of i (incl. self)
    std::vector<uint8_t>        visibility;
};

// Per-position top-K log-softmax extraction (CPU).
// logits:        [n_positions * vocab] f32
// out_log_probs: [n_positions * K] f32, descending (rank 0 = argmax)
// out_token_ids: [n_positions * K] i32
void common_ddtree_extract_topk(
        const float * logits,
        int n_positions,
        int vocab,
        int K,
        float * out_log_probs,
        int32_t * out_token_ids,
        float temperature = 1.0f);

// Build a DDTree from per-position top-K distributions.
// top_log_probs / top_token_ids: [L * K]
// L: max depth, K: branching, budget: max non-root nodes
// chain_seed: pre-seed full top-1 chain (defensive, matches Lucebox default)
common_ddtree common_ddtree_build(
        const float * top_log_probs,
        const int32_t * top_token_ids,
        int L,
        int K,
        int budget,
        bool chain_seed = true);

// Return positions for a flat verification batch containing the root followed
// by all tree nodes. Siblings share a position; descendants advance by depth.
std::vector<int32_t> common_ddtree_positions(
        const common_ddtree & tree,
        int32_t root_pos);

// Walk tree following target posterior argmax at each node.
// Returns accepted node indices (starting with root 0).
// out_next_token: first rejected / bonus token from posterior
// out_node_idx: last accepted node index
std::vector<int> common_ddtree_follow(
        const common_ddtree & tree,
        const int32_t * posterior,
        int & out_next_token,
        int * out_node_idx = nullptr);
