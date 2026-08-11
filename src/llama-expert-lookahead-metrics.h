#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_expert_lookahead {

struct sample_metrics {
    uint64_t actual_count = 0;
    uint64_t predicted_count = 0;
    uint64_t intersection_count = 0;
    uint64_t union_count = 0;
    uint64_t cold_actual_count = 0;
    uint64_t cold_intersection_count = 0;
    uint64_t predicted_fixed_count = 0;
    uint64_t useful_predicted_cold_count = 0;
    uint64_t false_positive_cold_count = 0;
    uint64_t missed_cold_count = 0;
    double actual_weight = 0.0;
    double covered_weight = 0.0;
    bool top1_hit = false;
};

// Deduplicate while preserving rank order. Invalid expert IDs are ignored.
std::vector<int32_t> unique_valid_prefix(
        const std::vector<int32_t> & ids,
        size_t prefix,
        int n_expert);

// Evaluate one routing sample. actual_weights is parallel to actual_ids. A
// missing weight is treated as zero. fixed_mask[e] is nonzero only for fixed
// tier residency; warm or future transient residency must not be included.
sample_metrics evaluate_sample(
        const std::vector<int32_t> & actual_ids,
        const std::vector<float> & actual_weights,
        const std::vector<int32_t> & predicted_ids,
        const std::vector<uint8_t> & fixed_mask,
        size_t predicted_prefix,
        int n_expert);

} // namespace llama_expert_lookahead
