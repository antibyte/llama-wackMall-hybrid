// Unit tests for common/ddtree — Diffusion Draft Tree (Lucebox port).
// Pure CPU; no model load required.

#include "ddtree.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static void test_extract_topk_argmax() {
    // 2 positions, vocab 5. Position 0 peaks at token 3; pos 1 at token 1.
    const int n_pos = 2, vocab = 5, K = 3;
    std::vector<float> logits = {
        // pos 0
        0.1f, 0.2f, 0.0f, 5.0f, 0.3f,
        // pos 1
        1.0f, 4.0f, 0.5f, 0.1f, 0.2f,
    };
    std::vector<float>   lp(n_pos * K);
    std::vector<int32_t> ids(n_pos * K);

    common_ddtree_extract_topk(logits.data(), n_pos, vocab, K, lp.data(), ids.data(), 1.0f);

    assert(ids[0] == 3);
    assert(ids[K] == 1);
    // log-probs descending within each position
    assert(lp[0] >= lp[1] && lp[1] >= lp[2]);
    assert(lp[K] >= lp[K + 1] && lp[K + 1] >= lp[K + 2]);
    // softmax sums to ~1
    float s0 = 0.f, s1 = 0.f;
    for (int k = 0; k < K; k++) {
        s0 += std::exp(lp[k]);
        s1 += std::exp(lp[K + k]);
    }
    // only top-K mass, so sum < 1; but top-1 should dominate
    assert(std::exp(lp[0]) > 0.5f);
    assert(std::exp(lp[K]) > 0.5f);
    (void) s0; (void) s1;
    std::printf("ok extract_topk_argmax\n");
}

static void test_build_chain_seed() {
    // L=4, K=2, budget=4 → full top-1 chain of length 4
    const int L = 4, K = 2, budget = 4;
    std::vector<float>   lp(L * K);
    std::vector<int32_t> ids(L * K);
    for (int d = 0; d < L; d++) {
        ids[d * K + 0] = 100 + d;      // top-1
        ids[d * K + 1] = 200 + d;      // top-2
        lp [d * K + 0] = -0.1f;        // high conf
        lp [d * K + 1] = -2.0f;
    }

    common_ddtree tree = common_ddtree_build(lp.data(), ids.data(), L, K, budget, /*chain_seed=*/true);
    assert(tree.n_nodes == budget);
    // chain-seed places top-1 path first
    for (int d = 0; d < L; d++) {
        assert(tree.token_ids[d] == 100 + d);
        assert(tree.depths[d] == d + 1);
    }
    // parents: root=0, then linear chain
    assert(tree.parents[0] == -1);
    assert(tree.parents[1] == 0);
    assert(tree.parents[2] == 1);
    assert(tree.parents[3] == 2);
    assert(tree.parents[4] == 3);

    // visibility: node 3 sees ancestors 0,1,2,3
    const int N = 1 + tree.n_nodes;
    assert(tree.visibility[(size_t) 3 * N + 0] == 1);
    assert(tree.visibility[(size_t) 3 * N + 3] == 1);
    assert(tree.visibility[(size_t) 3 * N + 4] == 0); // not yet / sibling

    auto chain = common_ddtree_top1_chain(lp.data(), ids.data(), L, K, L);
    assert(chain.size() == (size_t) L);
    assert(chain[0] == 100 && chain[3] == 103);
    std::printf("ok build_chain_seed\n");
}

static void test_build_expands_siblings() {
    // budget > L → siblings get inserted after the chain
    const int L = 3, K = 3, budget = 8;
    std::vector<float>   lp(L * K, -5.0f);
    std::vector<int32_t> ids(L * K, 0);
    for (int d = 0; d < L; d++) {
        for (int k = 0; k < K; k++) {
            ids[d * K + k] = 1000 + d * 10 + k;
            lp [d * K + k] = -0.2f * (k + 1); // rank 0 best
        }
    }
    // Make depth-0 rank-1 very attractive so it expands early
    lp[0 * K + 1] = -0.15f;

    common_ddtree tree = common_ddtree_build(lp.data(), ids.data(), L, K, budget, true);
    assert(tree.n_nodes == budget);
    // top-1 chain always present under chain_seed
    assert(tree.token_ids[0] == 1000);
    // at least one sibling of root's first child exists
    bool found_sib = false;
    for (int i = 0; i < tree.n_nodes; i++) {
        if (tree.token_ids[i] == 1001) { // depth1 rank1
            found_sib = true;
            break;
        }
    }
    assert(found_sib);
    std::printf("ok build_expands_siblings nodes=%d\n", tree.n_nodes);
}

static void test_follow_verified() {
    // Build a small tree, then follow a posterior that walks the top-1 chain
    // and diverges at the end.
    const int L = 3, K = 2, budget = 3;
    std::vector<float>   lp(L * K, -1.0f);
    std::vector<int32_t> ids(L * K);
    for (int d = 0; d < L; d++) {
        ids[d * K + 0] = 10 + d;
        ids[d * K + 1] = 50 + d;
        lp [d * K + 0] = -0.1f;
        lp [d * K + 1] = -3.0f;
    }
    common_ddtree tree = common_ddtree_build(lp.data(), ids.data(), L, K, budget, true);
    assert(tree.n_nodes == 3);

    // posterior: root wants 10, node1 wants 11, node2 wants 12, node3 wants 99 (reject)
    // flat indices: 0=root, 1=tok10, 2=tok11, 3=tok12
    std::vector<int32_t> posterior = { 10, 11, 12, 99 };
    int next = -1, node = -1;
    auto accepted = common_ddtree_follow(tree, posterior.data(), next, &node);
    assert(accepted.size() == 4); // root + 3
    assert(accepted[0] == 0);
    assert(next == 99);
    assert(node == 3);

    // Reject at first step: root posterior not in children
    posterior = { 777, 0, 0, 0 };
    accepted = common_ddtree_follow(tree, posterior.data(), next, &node);
    assert(accepted.size() == 1);
    assert(next == 777);
    assert(node == 0);
    std::printf("ok follow_verified\n");
}

static void test_paths() {
    const int L = 3, K = 2, budget = 6;
    std::vector<float>   lp(L * K, -1.0f);
    std::vector<int32_t> ids(L * K);
    for (int d = 0; d < L; d++) {
        ids[d * K + 0] = 10 + d;
        ids[d * K + 1] = 50 + d;
        lp [d * K + 0] = -0.1f;
        lp [d * K + 1] = -0.5f;
    }
    common_ddtree tree = common_ddtree_build(lp.data(), ids.data(), L, K, budget, true);
    auto paths = common_ddtree_paths(tree, lp.data(), ids.data(), L, K, /*max_paths=*/4);
    assert(!paths.empty());
    // first path should start with top-1 first token
    assert(paths[0][0] == 10);
    std::printf("ok paths n=%zu first_len=%zu\n", paths.size(), paths[0].size());
}

int main() {
    test_extract_topk_argmax();
    test_build_chain_seed();
    test_build_expands_siblings();
    test_follow_verified();
    test_paths();
    std::printf("all ddtree tests passed\n");
    return 0;
}
