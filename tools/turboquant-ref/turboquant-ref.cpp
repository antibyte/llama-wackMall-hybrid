// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#include "turboquant-ref.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace turboquant_ref {
namespace {

constexpr std::array<float, 8> centroids_3bit = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

constexpr std::array<float, 16> centroids_4bit = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

constexpr std::array<float, block_size> signs_first = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
};

constexpr std::array<float, block_size> signs_second = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
};

constexpr float inv_sqrt_128 = 0.08838834764831845f;

std::size_t block_count(std::size_t logical_size) {
    if (logical_size == 0) {
        throw std::invalid_argument("TurboQuant row size must be positive");
    }
    return (logical_size + block_size - 1) / block_size;
}

void check_finite(const float * input, std::size_t size) {
    if (input == nullptr) {
        throw std::invalid_argument("TurboQuant input is null");
    }
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(input[i])) {
            throw std::invalid_argument("TurboQuant input contains a non-finite value");
        }
    }
}

void transform_block(float * values, bool inverse) {
    const auto & first  = inverse ? signs_second : signs_first;
    const auto & second = inverse ? signs_first : signs_second;

    for (std::size_t i = 0; i < block_size; ++i) {
        values[i] *= first[i];
    }
    for (std::size_t width = 1; width < block_size; width *= 2) {
        for (std::size_t begin = 0; begin < block_size; begin += 2 * width) {
            for (std::size_t i = 0; i < width; ++i) {
                const float a = values[begin + i];
                const float b = values[begin + i + width];
                values[begin + i]         = a + b;
                values[begin + i + width] = a - b;
            }
        }
    }
    for (std::size_t i = 0; i < block_size; ++i) {
        values[i] *= inv_sqrt_128 * second[i];
    }
}

template<std::size_t N>
std::uint8_t nearest_centroid(float value, const std::array<float, N> & centroids) {
    std::size_t best = 0;
    float best_error = std::abs(value - centroids[0]);
    for (std::size_t i = 1; i < N; ++i) {
        const float error = std::abs(value - centroids[i]);
        if (error < best_error) {
            best = i;
            best_error = error;
        }
    }
    return static_cast<std::uint8_t>(best);
}

template<typename Block>
void append_block(std::vector<std::uint8_t> & output, const Block & block) {
    const std::size_t offset = output.size();
    output.resize(offset + sizeof(Block));
    std::memcpy(output.data() + offset, &block, sizeof(Block));
}

template<typename Block>
Block read_block(const std::uint8_t * input, std::size_t offset) {
    Block block;
    std::memcpy(&block, input + offset, sizeof(Block));
    return block;
}

float corrected_norm(const std::uint8_t * indices, std::size_t count, const float * centroids, float source_norm) {
    double reconstructed_norm_sq = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const float value = centroids[indices[i]];
        reconstructed_norm_sq += static_cast<double>(value) * value;
    }
    const float reconstructed_norm = static_cast<float>(std::sqrt(reconstructed_norm_sq));
    return reconstructed_norm > 1e-10f ? source_norm / reconstructed_norm : source_norm;
}

} // namespace

const char * format_name(format type) {
    switch (type) {
        case format::turbo3: return "turbo3";
        case format::turbo4: return "turbo4";
    }
    throw std::invalid_argument("unknown TurboQuant reference format");
}

format parse_format(const std::string & name) {
    if (name == "turbo3") {
        return format::turbo3;
    }
    if (name == "turbo4") {
        return format::turbo4;
    }
    throw std::invalid_argument("format must be turbo3 or turbo4");
}

std::size_t encoded_size(format type, std::size_t logical_size) {
    const std::size_t blocks = block_count(logical_size);
    switch (type) {
        case format::turbo3: return blocks * sizeof(block_turbo3);
        case format::turbo4: return blocks * sizeof(block_turbo4);
    }
    throw std::invalid_argument("unknown TurboQuant reference format");
}

double bits_per_value(format type, std::size_t logical_size) {
    return 8.0 * static_cast<double>(encoded_size(type, logical_size)) / static_cast<double>(logical_size);
}

std::vector<float> rotate_forward(const float * input, std::size_t logical_size) {
    check_finite(input, logical_size);
    std::vector<float> result(block_count(logical_size) * block_size, 0.0f);
    std::copy(input, input + logical_size, result.begin());
    for (std::size_t offset = 0; offset < result.size(); offset += block_size) {
        transform_block(result.data() + offset, false);
    }
    return result;
}

std::vector<float> rotate_inverse(const float * input, std::size_t padded_size, std::size_t logical_size) {
    if (padded_size == 0 || padded_size % block_size != 0 || logical_size == 0 || logical_size > padded_size) {
        throw std::invalid_argument("invalid TurboQuant rotated row dimensions");
    }
    check_finite(input, padded_size);
    std::vector<float> result(input, input + padded_size);
    for (std::size_t offset = 0; offset < result.size(); offset += block_size) {
        transform_block(result.data() + offset, true);
    }
    result.resize(logical_size);
    return result;
}

std::vector<std::uint8_t> quantize(format type, const float * input, std::size_t logical_size) {
    check_finite(input, logical_size);
    const std::vector<float> rotated = rotate_forward(input, logical_size);
    std::vector<std::uint8_t> output;
    output.reserve(encoded_size(type, logical_size));

    for (std::size_t offset = 0; offset < rotated.size(); offset += block_size) {
        double source_norm_sq = 0.0;
        for (std::size_t i = 0; i < block_size; ++i) {
            const std::size_t source_index = offset + i;
            const float value = source_index < logical_size ? input[source_index] : 0.0f;
            source_norm_sq += static_cast<double>(value) * value;
        }
        const float source_norm = static_cast<float>(std::sqrt(source_norm_sq));
        const float inv_norm = source_norm > 1e-10f ? 1.0f / source_norm : 0.0f;

        std::array<std::uint8_t, block_size> indices = {};
        if (type == format::turbo3) {
            block_turbo3 block = {};
            for (std::size_t i = 0; i < block_size; ++i) {
                const float normalized = rotated[offset + i] * inv_norm;
                const std::uint8_t index = nearest_centroid(normalized, centroids_3bit);
                indices[i] = index;
                block.indices_low[i / 4] |= static_cast<std::uint8_t>((index & 0x3u) << (2 * (i % 4)));
                block.indices_high[i / 8] |= static_cast<std::uint8_t>(((index >> 2) & 0x1u) << (i % 8));
            }
            block.norm = ggml_fp32_to_fp16(corrected_norm(indices.data(), block_size, centroids_3bit.data(), source_norm));
            append_block(output, block);
        } else if (type == format::turbo4) {
            block_turbo4 block = {};
            for (std::size_t i = 0; i < block_size; ++i) {
                const float normalized = rotated[offset + i] * inv_norm;
                const std::uint8_t index = nearest_centroid(normalized, centroids_4bit);
                indices[i] = index;
                block.indices[i / 2] |= static_cast<std::uint8_t>((index & 0xfu) << (4 * (i % 2)));
            }
            block.norm = ggml_fp32_to_fp16(corrected_norm(indices.data(), block_size, centroids_4bit.data(), source_norm));
            block.reserved = ggml_fp32_to_fp16(0.0f);
            append_block(output, block);
        } else {
            throw std::invalid_argument("unknown TurboQuant reference format");
        }
    }
    return output;
}

std::vector<float> dequantize_rotated(format type, const std::uint8_t * input, std::size_t input_size, std::size_t logical_size) {
    if (input == nullptr || input_size != encoded_size(type, logical_size)) {
        throw std::invalid_argument("invalid TurboQuant encoded row size");
    }
    std::vector<float> result(block_count(logical_size) * block_size);
    std::size_t input_offset = 0;
    for (std::size_t output_offset = 0; output_offset < result.size(); output_offset += block_size) {
        if (type == format::turbo3) {
            const block_turbo3 block = read_block<block_turbo3>(input, input_offset);
            const float norm = ggml_fp16_to_fp32(block.norm);
            for (std::size_t i = 0; i < block_size; ++i) {
                const std::uint8_t low = (block.indices_low[i / 4] >> (2 * (i % 4))) & 0x3u;
                const std::uint8_t high = (block.indices_high[i / 8] >> (i % 8)) & 0x1u;
                result[output_offset + i] = centroids_3bit[low | (high << 2)] * norm;
            }
            input_offset += sizeof(block_turbo3);
        } else if (type == format::turbo4) {
            const block_turbo4 block = read_block<block_turbo4>(input, input_offset);
            const float norm = ggml_fp16_to_fp32(block.norm);
            for (std::size_t i = 0; i < block_size; ++i) {
                const std::uint8_t index = (block.indices[i / 2] >> (4 * (i % 2))) & 0xfu;
                result[output_offset + i] = centroids_4bit[index] * norm;
            }
            input_offset += sizeof(block_turbo4);
        } else {
            throw std::invalid_argument("unknown TurboQuant reference format");
        }
    }
    return result;
}

std::vector<float> reconstruct(format type, const std::uint8_t * input, std::size_t input_size, std::size_t logical_size) {
    const std::vector<float> rotated = dequantize_rotated(type, input, input_size, logical_size);
    return rotate_inverse(rotated.data(), rotated.size(), logical_size);
}

error_metrics compare(const float * reference, const float * candidate, std::size_t size) {
    if (size == 0) {
        throw std::invalid_argument("cannot compare empty TurboQuant rows");
    }
    check_finite(reference, size);
    check_finite(candidate, size);

    double error_sq = 0.0;
    double reference_sq = 0.0;
    double candidate_sq = 0.0;
    double dot = 0.0;
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        const double error = static_cast<double>(candidate[i]) - reference[i];
        error_sq += error * error;
        reference_sq += static_cast<double>(reference[i]) * reference[i];
        candidate_sq += static_cast<double>(candidate[i]) * candidate[i];
        dot += static_cast<double>(reference[i]) * candidate[i];
        max_abs_error = std::max(max_abs_error, std::abs(error));
    }

    error_metrics metrics;
    metrics.mse = error_sq / static_cast<double>(size);
    metrics.rmse = std::sqrt(metrics.mse);
    metrics.max_abs_error = max_abs_error;
    metrics.relative_l2 = reference_sq > 0.0 ? std::sqrt(error_sq / reference_sq) : std::sqrt(error_sq);
    if (reference_sq > 0.0 && candidate_sq > 0.0) {
        metrics.cosine = dot / std::sqrt(reference_sq * candidate_sq);
        metrics.norm_ratio = std::sqrt(candidate_sq / reference_sq);
    } else if (reference_sq != candidate_sq) {
        metrics.cosine = 0.0;
        metrics.norm_ratio = 0.0;
    }
    return metrics;
}

double normalized_dot_error(const float * query, const float * key, std::size_t logical_size, format type) {
    check_finite(query, logical_size);
    check_finite(key, logical_size);
    const std::vector<std::uint8_t> encoded = quantize(type, key, logical_size);
    const std::vector<float> rotated_key = dequantize_rotated(type, encoded.data(), encoded.size(), logical_size);
    const std::vector<float> rotated_query = rotate_forward(query, logical_size);

    double exact = 0.0;
    double approximate = 0.0;
    double query_norm_sq = 0.0;
    double key_norm_sq = 0.0;
    for (std::size_t i = 0; i < logical_size; ++i) {
        exact += static_cast<double>(query[i]) * key[i];
        query_norm_sq += static_cast<double>(query[i]) * query[i];
        key_norm_sq += static_cast<double>(key[i]) * key[i];
    }
    for (std::size_t i = 0; i < rotated_query.size(); ++i) {
        approximate += static_cast<double>(rotated_query[i]) * rotated_key[i];
    }
    const double scale = std::sqrt(query_norm_sq * key_norm_sq);
    return scale > std::numeric_limits<double>::epsilon() ? std::abs(approximate - exact) / scale : 0.0;
}

} // namespace turboquant_ref
