// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double weighted_error(const std::vector<float> & source, const std::vector<float> & reconstructed) {
    double result = 0.0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const double x = source[i];
        const double diff = reconstructed[i] - x;
        result += x*x*diff*diff;
    }
    return result;
}

std::vector<float> make_input() {
    std::vector<float> result(32*64);
    uint64_t state = 0x214c91e5729af873ULL;
    for (std::size_t i = 0; i < result.size(); ++i) {
        state = state*6364136223846793005ULL + 1442695040888963407ULL;
        const float uniform = static_cast<float>((state >> 40) / static_cast<double>(1ULL << 24));
        const float wave = 0.65f*std::sin(0.071f*static_cast<float>(i));
        result[i] = wave + 0.7f*(uniform - 0.5f);
        if (i % 97 == 0) {
            result[i] *= 8.0f;
        }
    }
    return result;
}

void test_weighted_scale() {
    const std::vector<float> source = make_input();
    std::vector<block_q4_0> legacy(source.size()/QK4_0);
    std::vector<block_q4_0> weighted(source.size()/QK4_0);
    std::vector<float> legacy_out(source.size());
    std::vector<float> weighted_out(source.size());

    quantize_row_q4_0_ref(source.data(), legacy.data(), source.size());
    quantize_row_q4_0_weighted_ref(source.data(), weighted.data(), source.size());
    dequantize_row_q4_0(legacy.data(), legacy_out.data(), source.size());
    dequantize_row_q4_0(weighted.data(), weighted_out.data(), source.size());

    bool scale_changed = false;
    for (std::size_t i = 0; i < legacy.size(); ++i) {
        require(std::memcmp(legacy[i].qs, weighted[i].qs, sizeof(legacy[i].qs)) == 0,
                "weighted Q4_0 changed quantized values");
        scale_changed |= legacy[i].d != weighted[i].d;
    }

    require(scale_changed, "weighted Q4_0 did not change any block scale");
    require(weighted_error(source, weighted_out) < weighted_error(source, legacy_out),
            "weighted Q4_0 did not reduce its target error");
}

void test_zero_block() {
    std::vector<float> source(QK4_0, 0.0f);
    block_q4_0 block = {};
    quantize_row_q4_0_weighted_ref(source.data(), &block, source.size());
    require(ggml_fp16_to_fp32(block.d) == 0.0f, "zero block has a nonzero scale");
    for (uint8_t value : block.qs) {
        require(value == 0x88, "zero block has unexpected quantized values");
    }
}

}

int main() {
    test_weighted_scale();
    test_zero_block();
    return 0;
}
