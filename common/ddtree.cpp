#include "ddtree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>

void common_ddtree_extract_topk(
        const float * logits,
        int n_positions,
        int vocab,
        int K,
        float * out_log_probs,
        int32_t * out_token_ids,
        float temperature) {
    struct Entry {
        float   logit;
        int32_t id;
    };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);
    K = std::max(1, std::min(K, vocab));

    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t) i * (size_t) vocab;
        std::vector<Entry> heap;
        heap.reserve((size_t) K);

        float running_max     = -std::numeric_limits<float>::infinity();
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            if (l > running_max) {
                if (running_max > -std::numeric_limits<float>::infinity()) {
                    running_sum_exp *= std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            if ((int) heap.size() < K) {
                heap.push_back({l, (int32_t) j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t) j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(std::max(running_sum_exp, 1e-30f));

        // sort_heap + cmp_greater => descending order (no reverse)
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t) i * (size_t) K + (size_t) k] = heap[(size_t) k].logit - log_z;
            out_token_ids[(size_t) i * (size_t) K + (size_t) k] = heap[(size_t) k].id;
        }
    }
}

common_ddtree common_ddtree_build(
        const float * top_log_probs,
        const int32_t * top_token_ids,
        int L,
        int K,
        int budget,
        bool chain_seed) {
    common_ddtree tree;
    if (budget <= 0 || L <= 0 || K <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    struct HeapEntry {
        float            neg_logw;
        std::vector<int> ranks;
        int              parent_index;
        int              depth;
        int              rank;
        float            logw;
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve((size_t) budget);
    tree.depths.reserve((size_t) budget);
    tree.parents.reserve((size_t) budget + 1);
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t) (d - 1) * (size_t) K + 0];
            cum_logw += top_log_probs[(size_t) (d - 1) * (size_t) K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[(size_t) prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t) (d - 1) * (size_t) K + 0]
                    + top_log_probs[(size_t) (d - 1) * (size_t) K + 1];
                heap.push({
                    -sibling_logw,
                    {1},
                    prev_idx,
                    d,
                    1,
                    sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0];
        heap.push({
            -root_logw,
            {0},
            0,
            1,
            0,
            root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int     depth_minus_1 = top.depth - 1;
        const int     rank          = top.rank;
        const int32_t token_id      = top_token_ids[(size_t) depth_minus_1 * (size_t) K + (size_t) rank];

        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[(size_t) top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t) depth_minus_1 * (size_t) K + (size_t) rank]
                + top_log_probs[(size_t) depth_minus_1 * (size_t) K + (size_t) rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                -sibling_logw,
                std::move(sibling_ranks),
                top.parent_index,
                top.depth,
                rank + 1,
                sibling_logw,
            });
        }

        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t) top.depth * (size_t) K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                -child_logw,
                std::move(child_ranks),
                current_index,
                top.depth + 1,
                0,
                child_logw,
            });
        }
    }

    // Ancestor-only visibility mask (tree attention).
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t) N * (size_t) N, 0);
    tree.visibility[0] = 1;
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[(size_t) i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t) i * (size_t) N + (size_t) j] =
                tree.visibility[(size_t) p * (size_t) N + (size_t) j];
        }
        tree.visibility[(size_t) i * (size_t) N + (size_t) i] = 1;
    }

    return tree;
}

std::vector<int> common_ddtree_follow(
        const common_ddtree & tree,
        const int32_t * posterior,
        int & out_next_token,
        int * out_node_idx) {
    std::vector<int> accepted;
    accepted.reserve((size_t) tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[(size_t) current_index];
        auto it = children.find(next_token);
        if (it == children.end()) {
            break;
        }
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) {
        *out_node_idx = current_index;
    }
    return accepted;
}

std::vector<int32_t> common_ddtree_top1_chain(
        const float * /*top_log_probs*/,
        const int32_t * top_token_ids,
        int L,
        int K,
        int max_depth) {
    std::vector<int32_t> chain;
    const int depth = std::min(L, max_depth);
    chain.reserve((size_t) depth);
    for (int d = 0; d < depth; d++) {
        chain.push_back(top_token_ids[(size_t) d * (size_t) K + 0]);
    }
    return chain;
}

std::vector<std::vector<int32_t>> common_ddtree_paths(
        const common_ddtree & tree,
        const float * top_log_probs,
        const int32_t * top_token_ids,
        int L,
        int K,
        int max_paths) {
    std::vector<std::vector<int32_t>> paths;
    if (tree.n_nodes <= 0 || max_paths <= 0) {
        return paths;
    }

    // Collect leaves (nodes with no children)
    std::vector<int> leaves;
    for (int i = 1; i <= tree.n_nodes; i++) {
        if (tree.child_maps[(size_t) i].empty()) {
            leaves.push_back(i);
        }
    }
    if (leaves.empty()) {
        // no leaves? take deepest nodes
        for (int i = 1; i <= tree.n_nodes; i++) {
            leaves.push_back(i);
        }
    }

    struct ScoredPath {
        float score;
        std::vector<int32_t> tokens;
    };
    std::vector<ScoredPath> scored;
    scored.reserve(leaves.size());

    for (int leaf : leaves) {
        // walk to root
        std::vector<int> nodes;
        int cur = leaf;
        while (cur > 0) {
            nodes.push_back(cur);
            cur = tree.parents[(size_t) cur];
        }
        std::reverse(nodes.begin(), nodes.end());

        std::vector<int32_t> toks;
        float score = 0.0f;
        for (int ni : nodes) {
            const int depth = tree.depths[(size_t) (ni - 1)]; // depths indexed as node-1 for token_ids
            toks.push_back(tree.token_ids[(size_t) (ni - 1)]);
            // approximate path score: sum top-1 logp at each depth (independence)
            // better: use actual rank logp if we stored ranks; use top1 as proxy
            if (depth >= 1 && depth <= L) {
                score += top_log_probs[(size_t) (depth - 1) * (size_t) K + 0];
            }
            (void) top_token_ids;
        }
        scored.push_back({score, std::move(toks)});
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredPath & a, const ScoredPath & b) {
        return a.score > b.score;
    });

    for (size_t i = 0; i < scored.size() && (int) paths.size() < max_paths; i++) {
        // dedupe identical token sequences
        bool dup = false;
        for (const auto & p : paths) {
            if (p == scored[i].tokens) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            paths.push_back(std::move(scored[i].tokens));
        }
    }
    return paths;
}
