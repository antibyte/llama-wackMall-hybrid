#include "ddtree.h"

#include <algorithm>
#include <cassert>
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
    if (logits == nullptr || out_log_probs == nullptr || out_token_ids == nullptr ||
            n_positions <= 0 || vocab <= 0 || K <= 0) {
        return;
    }

    struct Entry {
        float   logit;
        int32_t id;
    };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);
    K = std::min(K, vocab);
    std::vector<Entry> heap;
    heap.reserve((size_t) K);

    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t) i * (size_t) vocab;
        heap.clear();

        float max_logit = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < vocab; j++) {
            const float scaled = li[j] * inv_t;
            const float l = std::isfinite(scaled) ? scaled : -std::numeric_limits<float>::infinity();
            max_logit = std::max(max_logit, l);

            if ((int) heap.size() < K) {
                heap.push_back({l, (int32_t) j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t) j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }

        float sum_exp = 0.0f;
        if (std::isfinite(max_logit)) {
            for (int j = 0; j < vocab; ++j) {
                const float scaled = li[j] * inv_t;
                if (std::isfinite(scaled)) {
                    sum_exp += std::exp(scaled - max_logit);
                }
            }
        }
        const float log_z = std::isfinite(max_logit) ?
            max_logit + std::log(std::max(sum_exp, 1e-30f)) : 0.0f;

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
            heap.push({
                -sibling_logw,
                top.parent_index,
                top.depth,
                rank + 1,
                sibling_logw,
            });
        }

        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t) top.depth * (size_t) K + 0];
            heap.push({
                -child_logw,
                current_index,
                top.depth + 1,
                0,
                child_logw,
            });
        }
    }

    // llama_batch requires nondecreasing positions for each sequence. Store
    // nodes in stable depth order and remap all flat tree indices accordingly.
    std::vector<int32_t> order((size_t) tree.n_nodes);
    for (int32_t i = 0; i < tree.n_nodes; ++i) {
        order[(size_t) i] = i + 1;
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return tree.depths[(size_t) a - 1] < tree.depths[(size_t) b - 1];
    });

    std::vector<int32_t> old_to_new((size_t) tree.n_nodes + 1, -1);
    old_to_new[0] = 0;
    for (int32_t i = 0; i < tree.n_nodes; ++i) {
        old_to_new[(size_t) order[(size_t) i]] = i + 1;
    }

    std::vector<int32_t> token_ids;
    std::vector<int> depths;
    std::vector<int32_t> parents;
    token_ids.reserve((size_t) tree.n_nodes);
    depths.reserve((size_t) tree.n_nodes);
    parents.reserve((size_t) tree.n_nodes + 1);
    parents.push_back(-1);
    for (int32_t old_index : order) {
        const int32_t old_parent = tree.parents[(size_t) old_index];
        const int32_t new_parent = old_to_new[(size_t) old_parent];
        assert(new_parent >= 0);
        token_ids.push_back(tree.token_ids[(size_t) old_index - 1]);
        depths.push_back(tree.depths[(size_t) old_index - 1]);
        parents.push_back(new_parent);
    }
    tree.token_ids = std::move(token_ids);
    tree.depths = std::move(depths);
    tree.parents = std::move(parents);
    tree.child_maps.assign((size_t) tree.n_nodes + 1, {});
    for (int32_t i = 1; i <= tree.n_nodes; ++i) {
        tree.child_maps[(size_t) tree.parents[(size_t) i]][tree.token_ids[(size_t) i - 1]] = i;
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

std::vector<int32_t> common_ddtree_positions(
        const common_ddtree & tree,
        int32_t root_pos) {
    std::vector<int32_t> positions;
    positions.reserve((size_t) tree.n_nodes + 1);
    positions.push_back(root_pos);
    for (int i = 0; i < tree.n_nodes; ++i) {
        positions.push_back(root_pos + tree.depths[(size_t) i]);
    }
    return positions;
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
