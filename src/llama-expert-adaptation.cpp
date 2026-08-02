#include "llama-expert-adaptation.h"

#include <cfloat>
#include <stdexcept>

namespace llama_expert_adaptation {

decision select_fixed_repin(
        const std::vector<int32_t> & slot_expert,
        const std::vector<int32_t> & lut,
        const std::vector<float> & score,
        const std::vector<int32_t> & dwell,
        int sentinel,
        int min_dwell,
        float min_gain) {
    if (slot_expert.size() != dwell.size()) {
        throw std::invalid_argument("slot_expert and dwell sizes differ");
    }
    if (lut.size() != score.size()) {
        throw std::invalid_argument("lut and score sizes differ");
    }
    if (sentinel < 0 || sentinel >= (int) slot_expert.size()) {
        throw std::invalid_argument("sentinel is outside slot range");
    }
    if (min_dwell < 0 || min_gain < 1.0f) {
        throw std::invalid_argument("invalid adaptation hysteresis");
    }

    int slot = -1;
    float incumbent_score = FLT_MAX;
    for (int s = 0; s < (int) slot_expert.size(); ++s) {
        if (s == sentinel) {
            continue;
        }
        const int expert = slot_expert[s];
        if (expert < -1 || expert >= (int) score.size()) {
            throw std::invalid_argument("slot contains invalid expert");
        }
        const float value = expert < 0 ? 0.0f : score[expert];
        if (value < incumbent_score) {
            incumbent_score = value;
            slot = s;
        }
    }

    int replacement = -1;
    float replacement_score = 0.0f;
    for (int expert = 0; expert < (int) score.size(); ++expert) {
        if (lut[expert] == sentinel && score[expert] > replacement_score) {
            replacement_score = score[expert];
            replacement = expert;
        }
    }

    if (slot < 0 || replacement < 0) {
        return {};
    }

    const int incumbent = slot_expert[slot];
    if (incumbent >= 0 && (dwell[slot] < min_dwell || replacement_score <= min_gain*incumbent_score)) {
        return {};
    }

    return {slot, incumbent, replacement};
}

} // namespace llama_expert_adaptation
