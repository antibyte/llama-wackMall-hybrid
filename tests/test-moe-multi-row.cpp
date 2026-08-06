#include "ggml-cpu.h"
#include "ggml.h"

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

int main() {
    ggml_cpu_init();
    if (!ggml_cpu_has_avx2()) {
        return 0;
    }
    test_type(GGML_TYPE_Q4_K);
    test_type(GGML_TYPE_Q5_K);
    return 0;
}
