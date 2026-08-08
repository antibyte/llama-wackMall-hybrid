#pragma once

#include <charconv>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_kv_layer_policy {

inline std::string trim(const std::string & value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) {
        ++first;
    }

    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) {
        --last;
    }

    return value.substr(first, last - first);
}

inline uint32_t parse_layer_id(const std::string & text, uint32_t n_layer) {
    const std::string value = trim(text);
    if (value.empty()) {
        throw std::invalid_argument("empty layer id");
    }

    uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("invalid layer id: " + value);
    }
    if (result >= n_layer) {
        throw std::out_of_range("layer id out of range: " + value);
    }
    return result;
}

inline std::vector<uint8_t> parse_layer_set(const std::string & text, uint32_t n_layer) {
    if (n_layer == 0) {
        throw std::invalid_argument("layer count must be positive");
    }

    std::vector<uint8_t> selected(n_layer, 0);
    if (trim(text).empty()) {
        return selected;
    }

    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string::npos ? text.size() : comma;
        const std::string item = trim(text.substr(begin, end - begin));
        if (item.empty()) {
            throw std::invalid_argument("empty layer range");
        }

        const size_t dash = item.find('-');
        if (dash != std::string::npos && item.find('-', dash + 1) != std::string::npos) {
            throw std::invalid_argument("invalid layer range: " + item);
        }

        const uint32_t first = parse_layer_id(item.substr(0, dash), n_layer);
        const uint32_t last = dash == std::string::npos
            ? first
            : parse_layer_id(item.substr(dash + 1), n_layer);
        if (first > last) {
            throw std::invalid_argument("descending layer range: " + item);
        }

        for (uint64_t il = first; il <= last; ++il) {
            if (selected[il]) {
                throw std::invalid_argument("duplicate layer id: " + std::to_string(il));
            }
            selected[il] = 1;
        }

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }

    return selected;
}

} // namespace llama_kv_layer_policy
