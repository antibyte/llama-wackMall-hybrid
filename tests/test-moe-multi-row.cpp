#include "ggml-cpu.h"
#include "ggml.h"
#include "quants.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static void test_type(enum ggml_type type) {
    constexpr int n = 512;
    constexpr int nrc = 4;

    const auto * traits = ggml_get_type_traits_cpu(type);
    const auto * q8_traits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    assert(traits->from_float != nullptr);
    assert(traits->vec_dot != nullptr);
    assert(q8_traits->from_float != nullptr);

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
    assert(std::memcmp(reference.data(), combined.data(), sizeof(reference)) == 0);
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
    assert(std::memcmp(&gate_reference, &gate_pair, sizeof(float)) == 0);
    assert(std::memcmp(&up_reference, &up_pair, sizeof(float)) == 0);

    ggml_cpu_moe_set_fused_gate_up(true);
    assert(ggml_cpu_moe_get_fused_gate_up());
    ggml_cpu_moe_set_fused_gate_up(false);
    assert(!ggml_cpu_moe_get_fused_gate_up());
}

int main() {
    ggml_cpu_init();
    if (!ggml_cpu_has_avx2()) {
        return 0;
    }
    test_type(GGML_TYPE_Q4_K);
    test_type(GGML_TYPE_Q5_K);
    test_q4_gate_up_pair();
    return 0;
}
