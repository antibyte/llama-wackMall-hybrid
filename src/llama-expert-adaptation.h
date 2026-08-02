#pragma once

#include <cstdint>
#include <vector>

namespace llama_expert_adaptation {

struct decision {
    int slot        = -1;
    int incumbent   = -1;
    int replacement = -1;

    explicit operator bool() const {
        return slot >= 0 && replacement >= 0;
    }
};

// Select at most one fixed-slot replacement. The sentinel is never selected
// as a destination. Experts mapped to the sentinel are cold candidates.
// Ties are resolved by the lowest slot/expert ID for reproducibility.
decision select_fixed_repin(
        const std::vector<int32_t> & slot_expert,
        const std::vector<int32_t> & lut,
        const std::vector<float> & score,
        const std::vector<int32_t> & dwell,
        int sentinel,
        int min_dwell = 32,
        float min_gain = 1.5f);

} // namespace llama_expert_adaptation
