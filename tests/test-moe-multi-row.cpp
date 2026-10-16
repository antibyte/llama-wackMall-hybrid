#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static void test_type(enum ggml_type type) {
    constexpr int n = 512;
    constexpr int nrc = 4;

    const auto * traits = ggml_get_type_traits_cpu(type);
    const auto * q8_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    GGML_ASSERT(traits->from_float != nullptr);
    GGML_ASSERT(traits->vec_dot != nullptr);
    GGML_ASSERT(q8_traits->from_float != nullptr);

    std::vector<float> weights(n);
    std::array<std::vector<float>, nrc> inputs;
    for (int i = 0; i < n; ++i) {
        weights[i] = 1.25f*std::sin(0.017f*i) - 0.75f*std::cos(0.031f*i);
        for (int r = 0; r < nrc; ++r) {
            inputs[r].resize(n);
            inputs[r][i] = std::sin((0.011f + 0.003f*r)*i) + 0.1f*r;
        }
    }

    const size_t weight_bytes = ggml_row_size(type, n);
    const size_t input_bytes = ggml_row_size(traits->vec_dot_type, n);
    std::vector<uint8_t> weight_q(weight_bytes);
    std::vector<uint8_t> input_q(nrc*input_bytes);
    traits->from_float(weights.data(), weight_q.data(), n);
    for (int r = 0; r < nrc; ++r) {
        q8_traits->from_float(inputs[r].data(), input_q.data() + r*input_bytes, n);
    }

    std::array<float, nrc> reference{};
    std::array<float, nrc> combined{};
    for (int r = 0; r < nrc; ++r) {
        traits->vec_dot(n, &reference[r], 0, weight_q.data(), 0,
                input_q.data() + r*input_bytes, 0, 1);
    }
    traits->vec_dot(n, combined.data(), sizeof(float), weight_q.data(), 0,
            input_q.data(), input_bytes, nrc);
    GGML_ASSERT(std::memcmp(reference.data(), combined.data(), sizeof(reference)) == 0);
}

static void test_q4_gate_up_pair() {
    constexpr int n = 2048;
    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    const auto * q8_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    std::vector<float> gate(n);
    std::vector<float> up(n);
    std::vector<float> input(n);
    for (int i = 0; i < n; ++i) {
        gate[i] = 1.25f*std::sin(0.017f*i) - 0.75f*std::cos(0.031f*i);
        up[i] = 0.85f*std::sin(0.023f*i) + 0.35f*std::cos(0.019f*i);
        input[i] = std::sin(0.011f*i) + 0.125f*std::cos(0.007f*i);
    }

    const size_t weight_bytes = ggml_row_size(GGML_TYPE_Q4_K, n);
    const size_t input_bytes = ggml_row_size(traits->vec_dot_type, n);
    std::vector<uint8_t> gate_q(weight_bytes);
    std::vector<uint8_t> up_q(weight_bytes);
    std::vector<uint8_t> input_q(input_bytes);
    traits->from_float(gate.data(), gate_q.data(), n);
    traits->from_float(up.data(), up_q.data(), n);
    q8_traits->from_float(input.data(), input_q.data(), n);

    float gate_reference = 0.0f;
    float up_reference = 0.0f;
    float gate_pair = 0.0f;
    float up_pair = 0.0f;
    traits->vec_dot(n, &gate_reference, 0, gate_q.data(), 0, input_q.data(), 0, 1);
    traits->vec_dot(n, &up_reference, 0, up_q.data(), 0, input_q.data(), 0, 1);
    ggml_vec_dot_q4_K_q8_K_pair(n, &gate_pair, &up_pair,
            gate_q.data(), up_q.data(), input_q.data());
    GGML_ASSERT(std::memcmp(&gate_reference, &gate_pair, sizeof(float)) == 0);
    GGML_ASSERT(std::memcmp(&up_reference, &up_pair, sizeof(float)) == 0);

    ggml_cpu_moe_set_fused_gate_up(true);
    GGML_ASSERT(ggml_cpu_moe_get_fused_gate_up());
    ggml_cpu_moe_set_fused_gate_up(false);
    GGML_ASSERT(!ggml_cpu_moe_get_fused_gate_up());
}

static void fill_moe_weight(ggml_tensor * tensor, int salt) {
    const auto * traits = ggml_get_type_traits_cpu(tensor->type);
    GGML_ASSERT(traits->from_float != nullptr);
    std::vector<float> row(tensor->ne[0]);
    for (int64_t e = 0; e < tensor->ne[2]; ++e) {
        for (int64_t r = 0; r < tensor->ne[1]; ++r) {
            for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                row[i] = float((i*17 + r*3 + e*7 + salt*11) % 101 - 50)*0.003f;
            }
            traits->from_float(row.data(), (char *) tensor->data + e*tensor->nb[2] + r*tensor->nb[1], tensor->ne[0]);
        }
    }
}

static void test_rows4(enum ggml_type type) {
    constexpr int n = 512;
    constexpr int rows = 8;
    const auto * traits = ggml_get_type_traits_cpu(type);
    const auto * q8_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    std::vector<float> weights(rows*n);
    std::vector<float> input(n);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < n; ++i) {
            weights[r*n + i] = 0.017f*(r + 1)*std::sin(0.013f*i + 0.2f*r) - 0.004f*i;
        }
    }
    for (int i = 0; i < n; ++i) {
        input[i] = std::sin(0.011f*i) + 0.125f*std::cos(0.007f*i);
    }

    const size_t row_bytes = ggml_row_size(type, n);
    const size_t input_bytes = ggml_row_size(traits->vec_dot_type, n);
    std::vector<uint8_t> weight_q(rows*row_bytes);
    std::vector<uint8_t> input_q(input_bytes);
    for (int r = 0; r < rows; ++r) {
        traits->from_float(weights.data() + r*n, weight_q.data() + r*row_bytes, n);
    }
    q8_traits->from_float(input.data(), input_q.data(), n);

    for (int base = 0; base + 4 <= rows; ++base) {
        std::array<float, 4> reference{};
        std::array<float, 4> combined{};
        for (int r = 0; r < 4; ++r) {
            traits->vec_dot(n, &reference[r], 0, weight_q.data() + (base + r)*row_bytes, 0, input_q.data(), 0, 1);
        }
        if (type == GGML_TYPE_Q4_K) {
            ggml_vec_dot_q4_K_q8_K_rows4(n, combined.data(), weight_q.data() + base*row_bytes, row_bytes, input_q.data());
        } else {
            ggml_vec_dot_q6_K_q8_K_rows4(n, combined.data(), weight_q.data() + base*row_bytes, row_bytes, input_q.data());
        }
        GGML_ASSERT(std::memcmp(reference.data(), combined.data(), sizeof(reference)) == 0);
    }
}

static void test_moe_cold(int n_tokens, int mask_mode, bool optimized, enum ggml_type down_type) {
    constexpr int n_embd = 512;
    constexpr int n_ff = 512;
    constexpr int n_out = 256;
    constexpr int n_expert = 4;
    constexpr int n_ids = 2;
    ggml_init_params params = {8*1024*1024, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    auto * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert);
    auto * up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert);
    auto * down = ggml_new_tensor_3d(ctx, down_type, n_ff, n_out, n_expert);
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, n_tokens);
    auto * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert);
    auto * counts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert + 1);
    fill_moe_weight(gate, 1);
    fill_moe_weight(up, 2);
    fill_moe_weight(down, 3);

    std::array<int32_t, n_expert + 1> expected_counts{};
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < n_embd; ++i) {
            ((float *) x->data)[t*n_embd + i] = float((i*13 + t*19) % 127 - 63)*0.01f;
        }
        for (int k = 0; k < n_ids; ++k) {
            const int e = (t/2 + k) % n_expert;
            ((int32_t *) ids->data)[t*n_ids + k] = e;
            ++expected_counts[e];
            ++expected_counts[n_expert];
        }
    }
    for (int e = 0; e < n_expert; ++e) {
        ((int32_t *) mask->data)[e] = mask_mode == 0 ? 1 : mask_mode == 1 ? e % 2 : 0;
    }

    auto * result = ggml_moe_cold(ctx, gate, up, down, x, ids, mask, counts, nullptr, nullptr, nullptr);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_cpu_moe_set_reuse_rows(optimized);
    ggml_cpu_moe_set_multi_row(optimized);
    ggml_cpu_moe_set_fused_gate_up(optimized);

    std::vector<uint8_t> reference(ggml_nbytes(result));
    for (int threads : {1, 2, 4, 8}) {
        auto plan = ggml_graph_plan(graph, threads, nullptr);
        GGML_ASSERT(plan.n_threads == threads);
        std::vector<uint8_t> work(plan.work_size + 64);
        plan.work_data = (uint8_t *) ((uintptr_t(work.data()) + 63) & ~uintptr_t(63));
        std::memset(counts->data, 0, ggml_nbytes(counts));
        std::memset(result->data, 0xff, ggml_nbytes(result));
        GGML_ASSERT(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(std::memcmp(counts->data, expected_counts.data(), ggml_nbytes(counts)) == 0);
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_ids; ++k) {
                const int e = ((int32_t *) ids->data)[t*n_ids + k];
                const auto * column = (const float *) ((const char *) result->data + t*result->nb[2] + k*result->nb[1]);
                bool nonzero = false;
                for (int i = 0; i < n_out; ++i) {
                    GGML_ASSERT(std::isfinite(column[i]));
                    GGML_ASSERT(((int32_t *) mask->data)[e] != 0 || column[i] == 0.0f);
                    nonzero |= column[i] != 0.0f;
                }
                GGML_ASSERT(((int32_t *) mask->data)[e] == 0 || nonzero);
            }
        }
        if (threads == 1) {
            std::memcpy(reference.data(), result->data, reference.size());
        } else if (std::memcmp(reference.data(), result->data, reference.size()) != 0) {
            GGML_ABORT("MOE_COLD output differs: tokens=%d mask=%d optimized=%d threads=%d down=%d", n_tokens, mask_mode, optimized, threads, (int) down_type);
        }
    }
    ggml_free(ctx);
}

int main() {
    ggml_cpu_init();
    if (ggml_cpu_has_avx2()) {
        test_type(GGML_TYPE_Q4_K);
        test_type(GGML_TYPE_Q5_K);
        test_q4_gate_up_pair();
        test_rows4(GGML_TYPE_Q4_K);
        test_rows4(GGML_TYPE_Q6_K);
    }
    for (bool optimized : {false, true}) {
        if (optimized && !ggml_cpu_has_avx2()) {
            continue;
        }
        for (int n_tokens : {1, 3, 9}) {
            for (int mask_mode : {0, 1, 2}) {
                test_moe_cold(n_tokens, mask_mode, optimized, GGML_TYPE_Q5_K);
            }
        }
        test_moe_cold(1, 0, optimized, GGML_TYPE_Q4_K);
        test_moe_cold(1, 0, optimized, GGML_TYPE_Q6_K);
    }
    return 0;
}
