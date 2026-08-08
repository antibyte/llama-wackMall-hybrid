// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#include "turboquant-ref.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> make_row(std::size_t size, float phase) {
    std::vector<float> result(size);
    for (std::size_t i = 0; i < size; ++i) {
        const float x = static_cast<float>(i);
        result[i] = 0.7f * std::sin(0.071f * x + phase) + 0.2f * std::cos(0.173f * x - 0.5f * phase) + 0.01f * static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    return result;
}

std::uint64_t random_state = 0x8f3f73b5cf1d2a49ULL;

double random_uniform() {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((random_state >> 11) + 1) / static_cast<double>((1ULL << 53) + 1);
}

float random_normal() {
    const double radius = std::sqrt(-2.0 * std::log(random_uniform()));
    const double angle = 6.28318530717958647692 * random_uniform();
    return static_cast<float>(radius * std::cos(angle));
}

void test_sizes() {
    using namespace turboquant_ref;
    require(encoded_size(format::turbo3, 128) == 50, "Turbo3 block size mismatch");
    require(encoded_size(format::turbo4, 128) == 68, "Turbo4 block size mismatch");
    require(encoded_size(format::turbo3, 64) == 50, "Turbo3 padded row size mismatch");
    require(encoded_size(format::turbo4, 256) == 136, "Turbo4 two-block row size mismatch");
    require(std::abs(bits_per_value(format::turbo3, 128) - 3.125) < 1e-12, "Turbo3 bits/value mismatch");
    require(std::abs(bits_per_value(format::turbo4, 128) - 4.25) < 1e-12, "Turbo4 bits/value mismatch");
}

void test_rotation_roundtrip(std::size_t size) {
    const std::vector<float> row = make_row(size, 0.3f);
    const std::vector<float> rotated = turboquant_ref::rotate_forward(row.data(), row.size());
    const std::vector<float> restored = turboquant_ref::rotate_inverse(rotated.data(), rotated.size(), row.size());
    const turboquant_ref::error_metrics metrics = turboquant_ref::compare(row.data(), restored.data(), row.size());
    require(metrics.max_abs_error < 2e-6, "WHT rotation is not reversible");
    require(std::abs(metrics.norm_ratio - 1.0) < 2e-6, "WHT rotation did not preserve the norm");
}

void test_codec(turboquant_ref::format type, std::size_t size) {
    const std::vector<float> row = make_row(size, 0.7f);
    const std::vector<std::uint8_t> first = turboquant_ref::quantize(type, row.data(), row.size());
    const std::vector<std::uint8_t> second = turboquant_ref::quantize(type, row.data(), row.size());
    require(first == second, "TurboQuant encoding is not deterministic");
    require(first.size() == turboquant_ref::encoded_size(type, size), "TurboQuant encoded size mismatch");

    const std::vector<float> restored = turboquant_ref::reconstruct(type, first.data(), first.size(), row.size());
    const turboquant_ref::error_metrics metrics = turboquant_ref::compare(row.data(), restored.data(), row.size());
    require(std::isfinite(metrics.relative_l2), "TurboQuant relative L2 is not finite");
    require(std::abs(metrics.norm_ratio - 1.0) < 0.002, "TurboQuant corrected norm is inaccurate");
    require(metrics.cosine > (type == turboquant_ref::format::turbo3 ? 0.93 : 0.97), "TurboQuant cosine is below the reference threshold");

    const std::vector<float> query = make_row(size, -0.4f);
    const double dot_error = turboquant_ref::normalized_dot_error(query.data(), row.data(), size, type);
    require(dot_error < (type == turboquant_ref::format::turbo3 ? 0.08 : 0.05), "TurboQuant normalized dot error is above the reference threshold");
}

void test_random_quality(turboquant_ref::format type, std::size_t size) {
    constexpr std::size_t samples = 256;
    double cosine_sum = 0.0;
    double relative_l2_sum = 0.0;
    double dot_error_sum = 0.0;
    double dot_error_max = 0.0;
    for (std::size_t sample = 0; sample < samples; ++sample) {
        std::vector<float> query(size);
        std::vector<float> key(size);
        for (std::size_t i = 0; i < size; ++i) {
            query[i] = random_normal();
            key[i] = random_normal();
        }
        const std::vector<std::uint8_t> encoded = turboquant_ref::quantize(type, key.data(), key.size());
        const std::vector<float> restored = turboquant_ref::reconstruct(type, encoded.data(), encoded.size(), key.size());
        const turboquant_ref::error_metrics metrics = turboquant_ref::compare(key.data(), restored.data(), key.size());
        const double dot_error = turboquant_ref::normalized_dot_error(query.data(), key.data(), size, type);
        cosine_sum += metrics.cosine;
        relative_l2_sum += metrics.relative_l2;
        dot_error_sum += dot_error;
        dot_error_max = std::max(dot_error_max, dot_error);
    }

    const double mean_cosine = cosine_sum / samples;
    const double mean_relative_l2 = relative_l2_sum / samples;
    const double mean_dot_error = dot_error_sum / samples;
    const double cosine_limit = type == turboquant_ref::format::turbo3 ? 0.96 : 0.98;
    const double mean_dot_limit = type == turboquant_ref::format::turbo3 ? 0.025 : 0.015;
    const double max_dot_limit = type == turboquant_ref::format::turbo3 ? 0.09 : 0.06;
    require(mean_cosine > cosine_limit, "random TurboQuant cosine is below the reference threshold");
    require(mean_dot_error < mean_dot_limit, "random TurboQuant mean dot error is above the reference threshold");
    require(dot_error_max < max_dot_limit, "random TurboQuant maximum dot error is above the reference threshold");

    std::cout << turboquant_ref::format_name(type)
              << " row=" << size
              << " mean_cosine=" << mean_cosine
              << " mean_relative_l2=" << mean_relative_l2
              << " mean_dot_error=" << mean_dot_error
              << " max_dot_error=" << dot_error_max << '\n';
}

void test_zero(turboquant_ref::format type) {
    const std::vector<float> row(256, 0.0f);
    const std::vector<std::uint8_t> encoded = turboquant_ref::quantize(type, row.data(), row.size());
    const std::vector<float> restored = turboquant_ref::reconstruct(type, encoded.data(), encoded.size(), row.size());
    require(std::all_of(restored.begin(), restored.end(), [](float value) { return value == 0.0f; }), "zero row did not reconstruct as zero");
}

void test_validation() {
    bool rejected = false;
    try {
        turboquant_ref::encoded_size(turboquant_ref::format::turbo3, 0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "empty row was accepted");

    std::vector<float> row(128, 0.0f);
    row[3] = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try {
        turboquant_ref::quantize(turboquant_ref::format::turbo3, row.data(), row.size());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "non-finite row was accepted");

    row[3] = 0.0f;
    const std::vector<std::uint8_t> encoded = turboquant_ref::quantize(turboquant_ref::format::turbo3, row.data(), row.size());
    rejected = false;
    try {
        turboquant_ref::reconstruct(turboquant_ref::format::turbo3, encoded.data(), encoded.size() - 1, row.size());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "truncated TurboQuant row was accepted");
}

void test_runtime_turbo4_golden() {
    const std::vector<float> row = make_row(256, 0.83f);
    const std::vector<std::uint8_t> golden = turboquant_ref::quantize(
        turboquant_ref::format::turbo4, row.data(), row.size());
    std::vector<block_turbo4_k> runtime(row.size() / QK_TURBO4_K);
    quantize_row_turbo4_k_ref(row.data(), runtime.data(), static_cast<int64_t>(row.size()));
    require(golden.size() == runtime.size() * sizeof(block_turbo4_k), "runtime Turbo4 size differs from golden codec");
    require(std::memcmp(golden.data(), runtime.data(), golden.size()) == 0, "runtime Turbo4 encoding differs from golden codec");

    std::vector<float> runtime_reconstruction(row.size());
    dequantize_row_turbo4_k(runtime.data(), runtime_reconstruction.data(), static_cast<int64_t>(row.size()));
    const std::vector<float> golden_reconstruction = turboquant_ref::reconstruct(
        turboquant_ref::format::turbo4, golden.data(), golden.size(), row.size());
    const turboquant_ref::error_metrics metrics = turboquant_ref::compare(
        golden_reconstruction.data(), runtime_reconstruction.data(), row.size());
    require(metrics.max_abs_error < 1e-7, "runtime Turbo4 reconstruction differs from golden codec");

    require(ggml_blck_size(GGML_TYPE_TURBO4_K) == QK_TURBO4_K, "runtime Turbo4 block size is not registered");
    require(ggml_type_size(GGML_TYPE_TURBO4_K) == sizeof(block_turbo4_k), "runtime Turbo4 type size is not registered");
    require(std::string(ggml_type_name(GGML_TYPE_TURBO4_K)) == "turbo4_k", "runtime Turbo4 type name is not registered");
}

} // namespace

int main() {
    try {
        test_sizes();
        for (const std::size_t size : { 64u, 128u, 256u }) {
            test_rotation_roundtrip(size);
            test_codec(turboquant_ref::format::turbo3, size);
            test_codec(turboquant_ref::format::turbo4, size);
            test_random_quality(turboquant_ref::format::turbo3, size);
            test_random_quality(turboquant_ref::format::turbo4, size);
        }
        test_zero(turboquant_ref::format::turbo3);
        test_zero(turboquant_ref::format::turbo4);
        test_validation();
        test_runtime_turbo4_golden();
        std::cout << "TurboQuant reference codec tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
