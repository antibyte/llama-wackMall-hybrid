// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT

#include "turboquant-ref.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::filesystem::path input;
    std::filesystem::path key_input;
    std::filesystem::path output;
    turboquant_ref::format type = turboquant_ref::format::turbo3;
    std::size_t row_size = 0;
    std::size_t tokens = 0;
    std::size_t q_heads = 0;
    std::size_t kv_heads = 0;
};

void print_usage(const char * program) {
    std::cout
        << "Usage: " << program << " --input rows.f32 --row-size N [--format turbo3|turbo4] [--output metrics.json]\n"
        << "       " << program << " --input qk-pairs.f32 --key-input keys.f32 --row-size N\n"
        << "           --tokens N --q-heads N --kv-heads N [--format turbo3|turbo4]\n"
        << "\n"
        << "The input is a raw little-endian float32 file containing complete rows.\n"
        << "Causal mode expects alternating Q/K rows in --input and unique K rows in --key-input.\n"
        << "The output path must differ from the input path. Without --output, JSON is written to stdout.\n";
}

std::size_t parse_size(const std::string & value, const char * option) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * option) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + option);
            }
            return argv[i];
        };
        if (arg == "--input") {
            result.input = require_value("--input");
        } else if (arg == "--key-input") {
            result.key_input = require_value("--key-input");
        } else if (arg == "--output") {
            result.output = require_value("--output");
        } else if (arg == "--format") {
            result.type = turboquant_ref::parse_format(require_value("--format"));
        } else if (arg == "--row-size") {
            result.row_size = parse_size(require_value("--row-size"), "--row-size");
        } else if (arg == "--tokens") {
            result.tokens = parse_size(require_value("--tokens"), "--tokens");
        } else if (arg == "--q-heads") {
            result.q_heads = parse_size(require_value("--q-heads"), "--q-heads");
        } else if (arg == "--kv-heads") {
            result.kv_heads = parse_size(require_value("--kv-heads"), "--kv-heads");
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (result.input.empty() || result.row_size == 0) {
        throw std::invalid_argument("--input and --row-size are required");
    }
    const bool any_causal = !result.key_input.empty() || result.tokens != 0 || result.q_heads != 0 || result.kv_heads != 0;
    const bool all_causal = !result.key_input.empty() && result.tokens != 0 && result.q_heads != 0 && result.kv_heads != 0;
    if (any_causal != all_causal) {
        throw std::invalid_argument("causal mode requires --key-input, --tokens, --q-heads, and --kv-heads together");
    }
    if (all_causal && (result.q_heads % result.kv_heads != 0)) {
        throw std::invalid_argument("--q-heads must be divisible by --kv-heads");
    }
    return result;
}

std::filesystem::path normalized_path(const std::filesystem::path & path) {
    return std::filesystem::absolute(path).lexically_normal();
}

std::vector<float> read_rows(const std::filesystem::path & path, std::size_t row_size) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("input is not a regular file: " + path.string());
    }

    const std::uintmax_t byte_size = std::filesystem::file_size(path);
    const std::size_t row_bytes = row_size * sizeof(float);
    if (byte_size == 0 || byte_size % row_bytes != 0) {
        throw std::runtime_error("input size is not a positive multiple of row-size * sizeof(float)");
    }
    if (byte_size / sizeof(float) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("input is too large for this build");
    }

    std::vector<float> data(static_cast<std::size_t>(byte_size / sizeof(float)));
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byte_size))) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return data;
}

std::string json_escape(const std::string & value) {
    std::ostringstream result;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (ch < 0x20) {
                    result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
                } else {
                    result << ch;
                }
        }
    }
    return result.str();
}

std::string analyze(const options & opts, const std::vector<float> & data) {
    const std::size_t rows = data.size() / opts.row_size;
    double error_sq = 0.0;
    double reference_sq = 0.0;
    double max_abs_error = 0.0;
    double cosine_sum = 0.0;
    double norm_ratio_sum = 0.0;
    double dot_error_sum = 0.0;
    double dot_error_max = 0.0;
    std::size_t dot_pairs = 0;

    for (std::size_t row = 0; row < rows; ++row) {
        const float * source = data.data() + row * opts.row_size;
        const std::vector<std::uint8_t> encoded = turboquant_ref::quantize(opts.type, source, opts.row_size);
        const std::vector<float> reconstructed = turboquant_ref::reconstruct(opts.type, encoded.data(), encoded.size(), opts.row_size);
        const turboquant_ref::error_metrics metrics = turboquant_ref::compare(source, reconstructed.data(), opts.row_size);
        error_sq += metrics.mse * static_cast<double>(opts.row_size);
        for (std::size_t i = 0; i < opts.row_size; ++i) {
            reference_sq += static_cast<double>(source[i]) * source[i];
        }
        max_abs_error = std::max(max_abs_error, metrics.max_abs_error);
        cosine_sum += metrics.cosine;
        norm_ratio_sum += metrics.norm_ratio;

        if (row % 2 == 1) {
            const float * query = data.data() + (row - 1) * opts.row_size;
            const double error = turboquant_ref::normalized_dot_error(query, source, opts.row_size, opts.type);
            dot_error_sum += error;
            dot_error_max = std::max(dot_error_max, error);
            ++dot_pairs;
        }
    }

    const double elements = static_cast<double>(data.size());
    const double mse = error_sq / elements;
    const double relative_l2 = reference_sq > 0.0 ? std::sqrt(error_sq / reference_sq) : std::sqrt(error_sq);
    const std::size_t encoded_bytes = rows * turboquant_ref::encoded_size(opts.type, opts.row_size);

    std::ostringstream json;
    json << std::setprecision(10)
         << "{\n"
         << "  \"input\": \"" << json_escape(normalized_path(opts.input).string()) << "\",\n"
         << "  \"format\": \"" << turboquant_ref::format_name(opts.type) << "\",\n"
         << "  \"rows\": " << rows << ",\n"
         << "  \"row_size\": " << opts.row_size << ",\n"
         << "  \"logical_values\": " << data.size() << ",\n"
         << "  \"encoded_bytes\": " << encoded_bytes << ",\n"
         << "  \"bits_per_value\": " << turboquant_ref::bits_per_value(opts.type, opts.row_size) << ",\n"
         << "  \"mse\": " << mse << ",\n"
         << "  \"rmse\": " << std::sqrt(mse) << ",\n"
         << "  \"relative_l2\": " << relative_l2 << ",\n"
         << "  \"max_abs_error\": " << max_abs_error << ",\n"
         << "  \"mean_cosine\": " << cosine_sum / static_cast<double>(rows) << ",\n"
         << "  \"mean_norm_ratio\": " << norm_ratio_sum / static_cast<double>(rows) << ",\n"
         << "  \"dot_pairs\": " << dot_pairs << ",\n"
         << "  \"mean_normalized_dot_error\": " << (dot_pairs > 0 ? dot_error_sum / static_cast<double>(dot_pairs) : 0.0) << ",\n"
         << "  \"max_normalized_dot_error\": " << dot_error_max << "\n"
         << "}\n";
    return json.str();
}

double dot_product(const float * left, const float * right, std::size_t size) {
    double result = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        result += static_cast<double>(left[i]) * right[i];
    }
    return result;
}

std::string analyze_causal(const options & opts, const std::vector<float> & pairs, const std::vector<float> & keys) {
    const std::size_t expected_pair_rows = opts.tokens * opts.q_heads * 2;
    const std::size_t expected_key_rows = opts.tokens * opts.kv_heads;
    if (pairs.size() != expected_pair_rows * opts.row_size) {
        throw std::runtime_error("Q/K pair input dimensions do not match --tokens and --q-heads");
    }
    if (keys.size() != expected_key_rows * opts.row_size) {
        throw std::runtime_error("key input dimensions do not match --tokens and --kv-heads");
    }

    const std::size_t q_heads_per_kv = opts.q_heads / opts.kv_heads;
    for (std::size_t token = 0; token < opts.tokens; ++token) {
        for (std::size_t query_head = 0; query_head < opts.q_heads; ++query_head) {
            const std::size_t pair_key_row = (token * opts.q_heads + query_head) * 2 + 1;
            const std::size_t key_head = query_head / q_heads_per_kv;
            const std::size_t key_row = token * opts.kv_heads + key_head;
            const float * paired_key = pairs.data() + pair_key_row * opts.row_size;
            const float * unique_key = keys.data() + key_row * opts.row_size;
            if (!std::equal(paired_key, paired_key + opts.row_size, unique_key)) {
                throw std::runtime_error("paired key rows do not match the unique key input");
            }
        }
    }

    std::vector<std::vector<float>> rotated_keys;
    rotated_keys.reserve(expected_key_rows);
    double error_sq = 0.0;
    double reference_sq = 0.0;
    double max_abs_error = 0.0;
    double cosine_sum = 0.0;
    double norm_ratio_sum = 0.0;
    for (std::size_t row = 0; row < expected_key_rows; ++row) {
        const float * source = keys.data() + row * opts.row_size;
        const std::vector<std::uint8_t> encoded = turboquant_ref::quantize(opts.type, source, opts.row_size);
        rotated_keys.push_back(turboquant_ref::dequantize_rotated(opts.type, encoded.data(), encoded.size(), opts.row_size));
        const std::vector<float> reconstructed = turboquant_ref::reconstruct(opts.type, encoded.data(), encoded.size(), opts.row_size);
        const turboquant_ref::error_metrics metrics = turboquant_ref::compare(source, reconstructed.data(), opts.row_size);
        error_sq += metrics.mse * static_cast<double>(opts.row_size);
        reference_sq += dot_product(source, source, opts.row_size);
        max_abs_error = std::max(max_abs_error, metrics.max_abs_error);
        cosine_sum += metrics.cosine;
        norm_ratio_sum += metrics.norm_ratio;
    }

    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(opts.row_size));
    double normalized_dot_error_sum = 0.0;
    double normalized_dot_error_max = 0.0;
    double scaled_logit_error_sum = 0.0;
    double scaled_logit_error_max = 0.0;
    double softmax_kl_sum = 0.0;
    double softmax_kl_max = 0.0;
    double probability_error_sum = 0.0;
    double probability_error_max = 0.0;
    std::size_t attention_pairs = 0;
    std::size_t attention_queries = 0;

    for (std::size_t query_token = 0; query_token < opts.tokens; ++query_token) {
        for (std::size_t query_head = 0; query_head < opts.q_heads; ++query_head) {
            const std::size_t pair_row = (query_token * opts.q_heads + query_head) * 2;
            const float * query = pairs.data() + pair_row * opts.row_size;
            const std::vector<float> rotated_query = turboquant_ref::rotate_forward(query, opts.row_size);
            const double query_norm_sq = dot_product(query, query, opts.row_size);
            const std::size_t key_head = query_head / q_heads_per_kv;
            std::vector<double> exact_logits(query_token + 1);
            std::vector<double> approximate_logits(query_token + 1);

            for (std::size_t key_token = 0; key_token <= query_token; ++key_token) {
                const std::size_t key_row = key_token * opts.kv_heads + key_head;
                const float * key = keys.data() + key_row * opts.row_size;
                const double exact = dot_product(query, key, opts.row_size);
                const double approximate = dot_product(rotated_query.data(), rotated_keys[key_row].data(), rotated_query.size());
                const double key_norm_sq = dot_product(key, key, opts.row_size);
                const double norm_scale = std::sqrt(query_norm_sq * key_norm_sq);
                const double normalized_error = norm_scale > std::numeric_limits<double>::epsilon()
                    ? std::abs(approximate - exact) / norm_scale
                    : 0.0;
                const double scaled_error = std::abs(approximate - exact) * attention_scale;
                normalized_dot_error_sum += normalized_error;
                normalized_dot_error_max = std::max(normalized_dot_error_max, normalized_error);
                scaled_logit_error_sum += scaled_error;
                scaled_logit_error_max = std::max(scaled_logit_error_max, scaled_error);
                exact_logits[key_token] = exact * attention_scale;
                approximate_logits[key_token] = approximate * attention_scale;
                ++attention_pairs;
            }

            const double exact_max = *std::max_element(exact_logits.begin(), exact_logits.end());
            const double approximate_max = *std::max_element(approximate_logits.begin(), approximate_logits.end());
            double exact_sum = 0.0;
            double approximate_sum = 0.0;
            for (std::size_t i = 0; i < exact_logits.size(); ++i) {
                exact_sum += std::exp(exact_logits[i] - exact_max);
                approximate_sum += std::exp(approximate_logits[i] - approximate_max);
            }
            const double exact_log_sum = exact_max + std::log(exact_sum);
            const double approximate_log_sum = approximate_max + std::log(approximate_sum);
            double query_kl = 0.0;
            double query_probability_error = 0.0;
            for (std::size_t i = 0; i < exact_logits.size(); ++i) {
                const double exact_probability = std::exp(exact_logits[i] - exact_log_sum);
                const double approximate_probability = std::exp(approximate_logits[i] - approximate_log_sum);
                query_kl += exact_probability * ((exact_logits[i] - exact_log_sum) - (approximate_logits[i] - approximate_log_sum));
                query_probability_error = std::max(query_probability_error, std::abs(approximate_probability - exact_probability));
            }
            query_kl = std::max(0.0, query_kl);
            softmax_kl_sum += query_kl;
            softmax_kl_max = std::max(softmax_kl_max, query_kl);
            probability_error_sum += query_probability_error;
            probability_error_max = std::max(probability_error_max, query_probability_error);
            ++attention_queries;
        }
    }

    const double elements = static_cast<double>(keys.size());
    const double mse = error_sq / elements;
    const double relative_l2 = reference_sq > 0.0 ? std::sqrt(error_sq / reference_sq) : std::sqrt(error_sq);
    const std::size_t encoded_bytes = expected_key_rows * turboquant_ref::encoded_size(opts.type, opts.row_size);

    std::ostringstream json;
    json << std::setprecision(10)
         << "{\n"
         << "  \"input\": \"" << json_escape(normalized_path(opts.input).string()) << "\",\n"
         << "  \"key_input\": \"" << json_escape(normalized_path(opts.key_input).string()) << "\",\n"
         << "  \"analysis_mode\": \"causal_attention\",\n"
         << "  \"format\": \"" << turboquant_ref::format_name(opts.type) << "\",\n"
         << "  \"tokens\": " << opts.tokens << ",\n"
         << "  \"q_heads\": " << opts.q_heads << ",\n"
         << "  \"kv_heads\": " << opts.kv_heads << ",\n"
         << "  \"rows\": " << expected_key_rows << ",\n"
         << "  \"row_size\": " << opts.row_size << ",\n"
         << "  \"logical_values\": " << keys.size() << ",\n"
         << "  \"encoded_bytes\": " << encoded_bytes << ",\n"
         << "  \"bits_per_value\": " << turboquant_ref::bits_per_value(opts.type, opts.row_size) << ",\n"
         << "  \"mse\": " << mse << ",\n"
         << "  \"rmse\": " << std::sqrt(mse) << ",\n"
         << "  \"relative_l2\": " << relative_l2 << ",\n"
         << "  \"max_abs_error\": " << max_abs_error << ",\n"
         << "  \"mean_cosine\": " << cosine_sum / static_cast<double>(expected_key_rows) << ",\n"
         << "  \"mean_norm_ratio\": " << norm_ratio_sum / static_cast<double>(expected_key_rows) << ",\n"
         << "  \"dot_pairs\": " << attention_pairs << ",\n"
         << "  \"mean_normalized_dot_error\": " << normalized_dot_error_sum / static_cast<double>(attention_pairs) << ",\n"
         << "  \"max_normalized_dot_error\": " << normalized_dot_error_max << ",\n"
         << "  \"attention_queries\": " << attention_queries << ",\n"
         << "  \"mean_abs_scaled_logit_error\": " << scaled_logit_error_sum / static_cast<double>(attention_pairs) << ",\n"
         << "  \"max_abs_scaled_logit_error\": " << scaled_logit_error_max << ",\n"
         << "  \"mean_softmax_kl\": " << softmax_kl_sum / static_cast<double>(attention_queries) << ",\n"
         << "  \"max_softmax_kl\": " << softmax_kl_max << ",\n"
         << "  \"mean_max_attention_probability_error\": " << probability_error_sum / static_cast<double>(attention_queries) << ",\n"
         << "  \"max_attention_probability_error\": " << probability_error_max << "\n"
         << "}\n";
    return json.str();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        if (!opts.output.empty()) {
            const std::filesystem::path normalized_output = normalized_path(opts.output);
            if (normalized_path(opts.input) == normalized_output ||
                (!opts.key_input.empty() && normalized_path(opts.key_input) == normalized_output)) {
                throw std::runtime_error("refusing to overwrite an input file");
            }
        }
        const std::vector<float> data = read_rows(opts.input, opts.row_size);
        const std::string json = opts.key_input.empty()
            ? analyze(opts, data)
            : analyze_causal(opts, data, read_rows(opts.key_input, opts.row_size));
        if (opts.output.empty()) {
            std::cout << json;
        } else {
            std::ofstream stream(opts.output, std::ios::binary | std::ios::trunc);
            if (!stream.write(json.data(), static_cast<std::streamsize>(json.size()))) {
                throw std::runtime_error("failed to write output file: " + opts.output.string());
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
