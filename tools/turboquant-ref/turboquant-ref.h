// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace turboquant_ref {

constexpr std::size_t block_size = 128;

enum class format {
    turbo3,
    turbo4,
};

struct block_turbo3 {
    std::uint16_t norm;
    std::uint8_t  indices_low[block_size / 4];
    std::uint8_t  indices_high[block_size / 8];
};

struct block_turbo4 {
    std::uint16_t norm;
    std::uint16_t reserved;
    std::uint8_t  indices[block_size / 2];
};

static_assert(sizeof(block_turbo3) == 50, "unexpected Turbo3 reference block size");
static_assert(sizeof(block_turbo4) == 68, "unexpected Turbo4 reference block size");

struct error_metrics {
    double mse             = 0.0;
    double rmse            = 0.0;
    double max_abs_error   = 0.0;
    double relative_l2     = 0.0;
    double cosine          = 1.0;
    double norm_ratio      = 1.0;
};

const char * format_name(format type);
format parse_format(const std::string & name);

std::size_t encoded_size(format type, std::size_t logical_size);
double bits_per_value(format type, std::size_t logical_size);

std::vector<float> rotate_forward(const float * input, std::size_t logical_size);
std::vector<float> rotate_inverse(const float * input, std::size_t padded_size, std::size_t logical_size);

std::vector<std::uint8_t> quantize(format type, const float * input, std::size_t logical_size);
std::vector<float> dequantize_rotated(format type, const std::uint8_t * input, std::size_t input_size, std::size_t logical_size);
std::vector<float> reconstruct(format type, const std::uint8_t * input, std::size_t input_size, std::size_t logical_size);

error_metrics compare(const float * reference, const float * candidate, std::size_t size);
double normalized_dot_error(const float * query, const float * key, std::size_t logical_size, format type);

} // namespace turboquant_ref
