#include "llama-expert-lookahead-metrics.h"

#include <cmath>
#include <cstdlib>
#include <vector>

using llama_expert_lookahead::evaluate_sample;

static void check(bool condition) {
    if (!condition) {
        abort();
    }
}

static void check_close(double actual, double expected) {
    check(std::fabs(actual - expected) < 1e-6);
}

static void test_duplicate_predictions_are_set_members() {
    const std::vector<int32_t> actual = { 1, 2, 3 };
    const std::vector<float> weights = { 0.5f, 0.3f, 0.2f };
    const std::vector<int32_t> predicted = { 1, 1, 2, 7, 7 };
    const std::vector<uint8_t> fixed(8, 0);

    const auto m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(m.actual_count == 3);
    check(m.predicted_count == 3);
    check(m.intersection_count == 2);
    check(m.union_count == 4);
    check(m.useful_predicted_cold_count == 2);
    check(m.false_positive_cold_count == 1);
    check(m.missed_cold_count == 1);
    check(m.top1_hit);
    check_close(m.actual_weight, 1.0);
    check_close(m.covered_weight, 0.8);
}

static void test_fixed_and_cold_accounting() {
    const std::vector<int32_t> actual = { 0, 2, 4, 6 };
    const std::vector<float> weights = { 0.4f, 0.3f, 0.2f, 0.1f };
    const std::vector<int32_t> predicted = { 0, 1, 2, 5, 6 };
    std::vector<uint8_t> fixed(8, 0);
    fixed[0] = 1;
    fixed[1] = 1;

    const auto m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(m.predicted_fixed_count == 2);
    check(m.cold_actual_count == 3);
    check(m.cold_intersection_count == 2);
    check(m.useful_predicted_cold_count == 2);
    check(m.false_positive_cold_count == 1);
    check(m.missed_cold_count == 1);
    check_close(m.covered_weight, 0.8);
}

static void test_prefix_and_invalid_ids() {
    const std::vector<int32_t> actual = { 1, 2, 3 };
    const std::vector<float> weights = { 0.6f, 0.3f, 0.1f };
    const std::vector<int32_t> predicted = { -1, 9, 3, 2, 1 };
    const std::vector<uint8_t> fixed(8, 0);

    const auto short_m = evaluate_sample(actual, weights, predicted, fixed, 3, 8);
    check(short_m.predicted_count == 1);
    check(short_m.intersection_count == 1);
    check(!short_m.top1_hit);

    const auto full_m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(full_m.predicted_count == 3);
    check(full_m.intersection_count == 3);
    check(full_m.false_positive_cold_count == 0);
    check(full_m.missed_cold_count == 0);
}

static void test_all_actual_experts_fixed() {
    const std::vector<int32_t> actual = { 1, 2 };
    const std::vector<float> weights = { 0.5f, 0.5f };
    const std::vector<int32_t> predicted = { 4, 5 };
    std::vector<uint8_t> fixed(8, 0);
    fixed[1] = 1;
    fixed[2] = 1;

    const auto m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(m.cold_actual_count == 0);
    check(m.cold_intersection_count == 0);
    check(m.missed_cold_count == 0);
    check(m.false_positive_cold_count == 2);
}

static void test_prefix_smaller_equal_and_larger_than_actual() {
    const std::vector<int32_t> actual = { 1, 2, 3, 4 };
    const std::vector<float> weights = { 0.4f, 0.3f, 0.2f, 0.1f };
    const std::vector<int32_t> predicted = { 1, 2, 3, 4, 5, 6 };
    const std::vector<uint8_t> fixed(8, 0);

    const auto smaller = evaluate_sample(actual, weights, predicted, fixed, 2, 8);
    check(smaller.intersection_count == 2);
    check(smaller.missed_cold_count == 2);
    check(smaller.false_positive_cold_count == 0);

    const auto equal = evaluate_sample(actual, weights, predicted, fixed, 4, 8);
    check(equal.intersection_count == 4);
    check(equal.missed_cold_count == 0);
    check(equal.false_positive_cold_count == 0);

    const auto larger = evaluate_sample(actual, weights, predicted, fixed, 6, 8);
    check(larger.intersection_count == 4);
    check(larger.missed_cold_count == 0);
    check(larger.false_positive_cold_count == 2);
}

static void test_all_predictions_fixed() {
    const std::vector<int32_t> actual = { 1, 2 };
    const std::vector<float> weights = { 0.6f, 0.4f };
    const std::vector<int32_t> predicted = { 1, 3, 4 };
    std::vector<uint8_t> fixed(8, 0);
    fixed[1] = 1;
    fixed[3] = 1;
    fixed[4] = 1;

    const auto m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(m.predicted_fixed_count == 3);
    check(m.intersection_count == 1);
    check(m.cold_actual_count == 1);
    check(m.false_positive_cold_count == 0);
    check(m.missed_cold_count == 1);
}

static void test_no_prediction_hits() {
    const std::vector<int32_t> actual = { 1, 2, 3 };
    const std::vector<float> weights = { 0.5f, 0.3f, 0.2f };
    const std::vector<int32_t> predicted = { 4, 5, 6 };
    const std::vector<uint8_t> fixed(8, 0);

    const auto m = evaluate_sample(actual, weights, predicted, fixed, predicted.size(), 8);
    check(m.intersection_count == 0);
    check(m.cold_intersection_count == 0);
    check(m.missed_cold_count == 3);
    check(m.false_positive_cold_count == 3);
    check_close(m.covered_weight, 0.0);
}

int main() {
    test_duplicate_predictions_are_set_members();
    test_fixed_and_cold_accounting();
    test_prefix_and_invalid_ids();
    test_all_actual_experts_fixed();
    test_prefix_smaller_equal_and_larger_than_actual();
    test_all_predictions_fixed();
    test_no_prediction_hits();
    return 0;
}
