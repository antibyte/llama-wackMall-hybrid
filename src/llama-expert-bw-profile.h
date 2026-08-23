#pragma once

#include <cstdlib>
#include <string>

namespace llama_expert_tier {

inline bool json_quoted_field(const std::string & text, const char * key, std::string & out) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        size_t i = pos + needle.size();
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) {
            i++;
        }
        if (i >= text.size() || text[i] != ':') {
            pos++;
            continue;
        }
        i++;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) {
            i++;
        }
        if (i >= text.size() || text[i] != '"') {
            return false;
        }
        i++;
        std::string value;
        while (i < text.size() && text[i] != '"') {
            if (text[i] == '\\' && i + 1 < text.size()) {
                i++;
            }
            value.push_back(text[i]);
            i++;
        }
        if (i >= text.size()) {
            return false;
        }
        out = std::move(value);
        return true;
    }
    return false;
}

inline bool parse_bw_profile(const std::string & text, std::string & recommended, float & q_star) {
    std::string schema;
    if (!json_quoted_field(text, "schema", schema) || schema != "llama-wackmall-expert-bw-v1") {
        return false;
    }
    if (!json_quoted_field(text, "recommended", recommended)) {
        return false;
    }
    if (recommended != "cpu-heavy" && recommended != "hybrid" && recommended != "pcie-heavy") {
        return false;
    }
    const std::string needle = "\"q_star\"";
    const size_t pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    size_t i = pos + needle.size();
    while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) {
        i++;
    }
    if (i >= text.size() || text[i] != ':') {
        return false;
    }
    i++;
    char * end = nullptr;
    q_star = strtof(text.c_str() + i, &end);
    if (end == text.c_str() + i || q_star < 0.0f || q_star > 1.0f) {
        return false;
    }
    return true;
}

// Copies allowed this graph: min(W, round(q* * unique cold misses)).
// q_star <= 0 admits nothing; q_star >= 1 fills up to W.
inline int warm_admit_budget(int n_miss, int n_warm, float q_star) {
    if (n_miss <= 0 || n_warm <= 0 || q_star <= 0.0f) {
        return 0;
    }
    if (q_star > 1.0f) {
        q_star = 1.0f;
    }
    int q = (int) (q_star * (float) n_miss + 0.5f);
    if (q < 0) {
        q = 0;
    }
    if (q > n_miss) {
        q = n_miss;
    }
    if (q > n_warm) {
        q = n_warm;
    }
    return q;
}

// cpu-heavy keeps W only when q* is a real overlap cap, not fill-all or never-copy.
inline bool warm_keep_on_cpu_heavy(float q_star) {
    return q_star > 0.0f && q_star < 1.0f;
}

// Peak per-expert selections in one graph. Prefill-sized batches exceed TMAX
// and use ggml_moe_count instead of the fused LUT path.
inline bool warm_admit_from_counts(int peak_count, int tmax) {
    return peak_count > 0 && tmax > 0 && peak_count <= tmax;
}

} // namespace llama_expert_tier
