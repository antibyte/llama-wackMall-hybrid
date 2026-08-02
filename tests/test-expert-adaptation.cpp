#include "llama-expert-adaptation.h"

#include <cstdlib>
#include <stdexcept>
#include <vector>

using llama_expert_adaptation::select_fixed_repin;

static void check(bool condition) {
    if (!condition) {
        abort();
    }
}

static void test_hysteresis_and_sentinel_protection() {
    const std::vector<int32_t> slots = {0, 1, -1};
    const std::vector<int32_t> lut = {0, 1, 2, 2};
    const std::vector<float> score = {10.0f, 20.0f, 16.0f, 15.0f};

    check(!select_fixed_repin(slots, lut, score, {31, 64, 0}, 2));

    const auto selected = select_fixed_repin(slots, lut, score, {32, 64, 0}, 2);
    check((bool) selected);
    check(selected.slot == 0);
    check(selected.incumbent == 0);
    check(selected.replacement == 2);
    check(selected.slot != 2);
}

static void test_exact_threshold_does_not_repin() {
    const std::vector<int32_t> slots = {0, 1, -1};
    const std::vector<int32_t> lut = {0, 1, 2};
    const std::vector<float> score = {10.0f, 20.0f, 15.0f};
    check(!select_fixed_repin(slots, lut, score, {32, 32, 0}, 2));
}

static void test_empty_slot_and_deterministic_ties() {
    const std::vector<int32_t> slots = {-1, 1, -1};
    const std::vector<int32_t> lut = {2, 1, 2, 2};
    const std::vector<float> score = {7.0f, 20.0f, 7.0f, 6.0f};
    const auto selected = select_fixed_repin(slots, lut, score, {0, 0, 0}, 2);
    check((bool) selected);
    check(selected.slot == 0);
    check(selected.incumbent == -1);
    check(selected.replacement == 0);
}

static void test_invalid_state_is_rejected() {
    bool threw = false;
    try {
        (void) select_fixed_repin({0, -1}, {0, 1}, {1.0f, 2.0f}, {1}, 1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw);
}

int main() {
    test_hysteresis_and_sentinel_protection();
    test_exact_threshold_does_not_repin();
    test_empty_slot_and_deterministic_ties();
    test_invalid_state_is_rejected();
    return 0;
}
