#include "llama-expert-cache.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace llama_expert_cache {

void second_hit_admission::reset(int n_expert, uint64_t window) {
    if (n_expert < 0 || window == 0) {
        throw std::invalid_argument("invalid second-hit admission dimensions");
    }
    epoch_ = 0;
    window_ = window;
    last_miss_epoch_.assign(n_expert, 0);
}

void second_hit_admission::next_epoch() {
    if (epoch_ == std::numeric_limits<uint64_t>::max()) {
        std::fill(last_miss_epoch_.begin(), last_miss_epoch_.end(), 0);
        epoch_ = 1;
    } else {
        ++epoch_;
    }
}

bool second_hit_admission::record_miss(int expert) {
    if (!valid_expert(expert) || epoch_ == 0) {
        return false;
    }
    const uint64_t previous = last_miss_epoch_[expert];
    const bool admitted = previous != 0 && epoch_ - previous <= window_;
    last_miss_epoch_[expert] = epoch_;
    return admitted;
}

void second_hit_admission::mark_resident(int expert) {
    if (valid_expert(expert)) {
        last_miss_epoch_[expert] = 0;
    }
}

uint64_t second_hit_admission::epoch() const {
    return epoch_;
}

uint64_t second_hit_admission::window() const {
    return window_;
}

bool second_hit_admission::valid_expert(int expert) const {
    return expert >= 0 && expert < (int) last_miss_epoch_.size();
}

state::state(int n_expert, int n_fixed, int n_warm, const std::vector<int32_t> & fixed_experts) {
    reset(n_expert, n_fixed, n_warm, fixed_experts);
}

void state::reset(int n_expert, int n_fixed, int n_warm, const std::vector<int32_t> & fixed_experts) {
    if (n_expert < 0 || n_fixed < 0 || n_warm < 0 || n_fixed > n_expert ||
        fixed_experts.size() != (size_t) n_fixed) {
        throw std::invalid_argument("invalid expert cache dimensions");
    }

    n_expert_ = n_expert;
    n_fixed_  = n_fixed;
    n_warm_   = n_warm;
    sentinel_ = n_fixed + n_warm;
    clock_    = 0;
    lut_.assign(n_expert, sentinel_);
    slot_expert_.assign(sentinel_ + 1, -1);
    warm_age_.assign(n_warm, 0);

    for (size_t slot = 0; slot < fixed_experts.size(); ++slot) {
        const int expert = fixed_experts[slot];
        if (!valid_expert(expert) || lut_[expert] != sentinel_) {
            throw std::invalid_argument("invalid or duplicate fixed expert");
        }
        slot_expert_[slot] = expert;
        lut_[expert] = (int32_t) slot;
    }
}

int state::n_expert() const {
    return n_expert_;
}

int state::n_fixed() const {
    return n_fixed_;
}

int state::n_warm() const {
    return n_warm_;
}

int state::sentinel() const {
    return sentinel_;
}

int state::n_slots() const {
    return sentinel_ + 1;
}

location state::locate(int expert) const {
    if (!valid_expert(expert)) {
        return location::cold;
    }
    const int slot = lut_[expert];
    if (slot >= 0 && slot < n_fixed_) {
        return location::fixed;
    }
    if (valid_warm_slot(slot)) {
        return location::warm;
    }
    return location::cold;
}

int state::slot_for(int expert) const {
    return valid_expert(expert) ? lut_[expert] : sentinel_;
}

int state::expert_in_slot(int slot) const {
    return slot >= 0 && slot < (int) slot_expert_.size() ? slot_expert_[slot] : -1;
}

bool state::touch_warm(int expert) {
    if (locate(expert) != location::warm) {
        return false;
    }
    const int slot = lut_[expert];
    warm_age_[slot - n_fixed_] = ++clock_;
    return true;
}

insertion state::insert_warm(int expert, const std::vector<bool> & protected_slots) {
    if (!valid_expert(expert) || n_warm_ == 0) {
        return {};
    }
    if (locate(expert) == location::fixed) {
        return {};
    }
    if (locate(expert) == location::warm) {
        touch_warm(expert);
        return {lut_[expert], -1};
    }

    const insertion target = insertion_target(protected_slots);
    if (!target) {
        return {};
    }

    if (target.evicted >= 0) {
        lut_[target.evicted] = sentinel_;
    }
    bind_warm(target.slot, expert);
    return target;
}

insertion state::insertion_target(const std::vector<bool> & protected_slots) const {
    if (n_warm_ == 0) {
        return {};
    }

    int victim_slot = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int slot = n_fixed_; slot < sentinel_; ++slot) {
        if (slot < (int) protected_slots.size() && protected_slots[slot]) {
            continue;
        }
        if (slot_expert_[slot] < 0) {
            victim_slot = slot;
            break;
        }
        const uint64_t age = warm_age_[slot - n_fixed_];
        if (age < oldest) {
            oldest = age;
            victim_slot = slot;
        }
    }
    if (victim_slot < 0) {
        return {};
    }
    return {victim_slot, slot_expert_[victim_slot]};
}

int state::evict_warm_slot(int slot) {
    if (!valid_warm_slot(slot)) {
        throw std::invalid_argument("invalid warm eviction slot");
    }
    const int evicted = slot_expert_[slot];
    if (evicted >= 0) {
        lut_[evicted] = sentinel_;
        slot_expert_[slot] = -1;
        warm_age_[slot - n_fixed_] = 0;
    }
    return evicted;
}

void state::bind_ready_warm_slot(int slot, int expert) {
    if (!valid_warm_slot(slot) || !valid_expert(expert) || slot_expert_[slot] >= 0 ||
            locate(expert) != location::cold) {
        throw std::invalid_argument("invalid ready warm binding");
    }
    bind_warm(slot, expert);
}

int state::replace_fixed(int fixed_slot, int expert) {
    if (fixed_slot < 0 || fixed_slot >= n_fixed_ || !valid_expert(expert)) {
        throw std::invalid_argument("invalid fixed replacement");
    }

    const int current_slot = lut_[expert];
    if (current_slot == fixed_slot) {
        return slot_expert_[fixed_slot];
    }
    if (current_slot >= 0 && current_slot < n_fixed_) {
        throw std::invalid_argument("expert already occupies another fixed slot");
    }
    if (valid_warm_slot(current_slot)) {
        slot_expert_[current_slot] = -1;
        warm_age_[current_slot - n_fixed_] = 0;
    }

    const int evicted = slot_expert_[fixed_slot];
    if (evicted >= 0) {
        lut_[evicted] = sentinel_;
    }
    slot_expert_[fixed_slot] = expert;
    lut_[expert] = fixed_slot;
    return evicted;
}

void state::reset_warm_ages() {
    clock_ = 0;
    for (int slot = n_fixed_; slot < sentinel_; ++slot) {
        if (slot_expert_[slot] >= 0) {
            warm_age_[slot - n_fixed_] = ++clock_;
        } else {
            warm_age_[slot - n_fixed_] = 0;
        }
    }
}

const std::vector<int32_t> & state::lut() const {
    return lut_;
}

const std::vector<int32_t> & state::slot_experts() const {
    return slot_expert_;
}

const std::vector<uint64_t> & state::warm_ages() const {
    return warm_age_;
}

bool state::validate(std::string * error) const {
    auto fail = [&](const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    };

    if (n_expert_ < 0 || n_fixed_ < 0 || n_warm_ < 0 || sentinel_ != n_fixed_ + n_warm_) {
        return fail("invalid dimensions");
    }
    if ((int) lut_.size() != n_expert_ || (int) slot_expert_.size() != sentinel_ + 1 ||
        (int) warm_age_.size() != n_warm_) {
        return fail("invalid vector sizes");
    }
    if (slot_expert_[sentinel_] != -1) {
        return fail("sentinel slot is occupied");
    }

    std::vector<int> occurrences(n_expert_, 0);
    for (int slot = 0; slot < sentinel_; ++slot) {
        const int expert = slot_expert_[slot];
        if (expert < 0) {
            continue;
        }
        if (!valid_expert(expert)) {
            std::ostringstream message;
            message << "slot " << slot << " has invalid expert " << expert;
            return fail(message.str());
        }
        if (++occurrences[expert] != 1 || lut_[expert] != slot) {
            std::ostringstream message;
            message << "slot/lut mismatch for expert " << expert << " at slot " << slot;
            return fail(message.str());
        }
    }
    for (int expert = 0; expert < n_expert_; ++expert) {
        const int slot = lut_[expert];
        if (slot == sentinel_) {
            if (occurrences[expert] != 0) {
                return fail("cold expert still occupies a slot");
            }
            continue;
        }
        if (slot < 0 || slot >= sentinel_ || occurrences[expert] != 1 || slot_expert_[slot] != expert) {
            std::ostringstream message;
            message << "invalid lut slot " << slot << " for expert " << expert;
            return fail(message.str());
        }
    }
    return true;
}

bool state::valid_expert(int expert) const {
    return expert >= 0 && expert < n_expert_;
}

bool state::valid_warm_slot(int slot) const {
    return slot >= n_fixed_ && slot < sentinel_;
}

void state::bind_warm(int slot, int expert) {
    slot_expert_[slot] = expert;
    lut_[expert] = slot;
    warm_age_[slot - n_fixed_] = ++clock_;
}

} // namespace llama_expert_cache
