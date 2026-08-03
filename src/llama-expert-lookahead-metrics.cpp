#include "llama-expert-lookahead-metrics.h"

#include <algorithm>

namespace llama_expert_lookahead {

std::vector<int32_t> unique_valid_prefix(
        const std::vector<int32_t> & ids,
        size_t prefix,
        int n_expert) {
    std::vector<int32_t> result;
    if (n_expert <= 0 || prefix == 0) {
        return result;
    }

    std::vector<uint8_t> seen((size_t) n_expert, 0);
    const size_t limit = std::min(prefix, ids.size());
    result.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
        const int32_t expert = ids[i];
        if (expert < 0 || expert >= n_expert || seen[(size_t) expert]) {
            continue;
        }
        seen[(size_t) expert] = 1;
        result.push_back(expert);
    }
    return result;
}

sample_metrics evaluate_sample(
        const std::vector<int32_t> & actual_ids,
        const std::vector<float> & actual_weights,
        const std::vector<int32_t> & predicted_ids,
        const std::vector<uint8_t> & fixed_mask,
        size_t predicted_prefix,
        int n_expert) {
    sample_metrics result;
    if (n_expert <= 0) {
        return result;
    }

    const auto predicted = unique_valid_prefix(predicted_ids, predicted_prefix, n_expert);
    std::vector<uint8_t> predicted_set((size_t) n_expert, 0);
    for (int32_t expert : predicted) {
        predicted_set[(size_t) expert] = 1;
    }

    std::vector<uint8_t> actual_set((size_t) n_expert, 0);
    std::vector<double> actual_weight((size_t) n_expert, 0.0);
    for (size_t i = 0; i < actual_ids.size(); ++i) {
        const int32_t expert = actual_ids[i];
        if (expert < 0 || expert >= n_expert || actual_set[(size_t) expert]) {
            continue;
        }
        actual_set[(size_t) expert] = 1;
        if (i < actual_weights.size()) {
            actual_weight[(size_t) expert] = std::max(0.0, (double) actual_weights[i]);
        }
    }

    const auto is_fixed = [&](int expert) {
        return expert >= 0 && (size_t) expert < fixed_mask.size() && fixed_mask[(size_t) expert] != 0;
    };

    result.predicted_count = predicted.size();
    for (int expert = 0; expert < n_expert; ++expert) {
        if (actual_set[(size_t) expert]) {
            result.actual_count++;
            result.actual_weight += actual_weight[(size_t) expert];
            if (!is_fixed(expert)) {
                result.cold_actual_count++;
            }
            if (predicted_set[(size_t) expert]) {
                result.intersection_count++;
                result.covered_weight += actual_weight[(size_t) expert];
                if (!is_fixed(expert)) {
                    result.cold_intersection_count++;
                }
            } else if (!is_fixed(expert)) {
                result.missed_cold_count++;
            }
        }

        if (predicted_set[(size_t) expert]) {
            if (is_fixed(expert)) {
                result.predicted_fixed_count++;
            } else if (actual_set[(size_t) expert]) {
                result.useful_predicted_cold_count++;
            } else {
                result.false_positive_cold_count++;
            }
        }
    }

    result.union_count = result.actual_count + result.predicted_count - result.intersection_count;
    const auto actual_ranked = unique_valid_prefix(actual_ids, actual_ids.size(), n_expert);
    result.top1_hit = !actual_ranked.empty() && !predicted.empty() && actual_ranked.front() == predicted.front();
    return result;
}

} // namespace llama_expert_lookahead
