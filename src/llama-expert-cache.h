#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_expert_cache {

enum class location {
    fixed,
    warm,
    cold,
};

struct insertion {
    int slot    = -1;
    int evicted = -1;

    explicit operator bool() const {
        return slot >= 0;
    }
};

// Optional admission probation for the warm LRU. A cold expert becomes
// eligible after a second miss within window epochs. Epochs are completed
// graph updates, not tokens, so MTP verify batches remain one transaction.
class second_hit_admission {
public:
    void reset(int n_expert, uint64_t window);
    void next_epoch();

    // Record one or more selections of a cold expert in the current epoch.
    // Returns true only when a previous miss is still inside the window.
    bool record_miss(int expert);

    // Clear stale probation after the expert becomes fixed or warm.
    void mark_resident(int expert);

    uint64_t epoch() const;
    uint64_t window() const;

private:
    bool valid_expert(int expert) const;

    uint64_t epoch_  = 0;
    uint64_t window_ = 1;
    std::vector<uint64_t> last_miss_epoch_;
};

// Host-side ownership state for one layer. The final slot is always the
// sentinel; fixed and warm slots are disjoint contiguous ranges before it.
class state {
public:
    state() = default;
    state(int n_expert, int n_fixed, int n_warm, const std::vector<int32_t> & fixed_experts);

    void reset(int n_expert, int n_fixed, int n_warm, const std::vector<int32_t> & fixed_experts);

    int n_expert() const;
    int n_fixed() const;
    int n_warm() const;
    int sentinel() const;
    int n_slots() const;

    location locate(int expert) const;
    int slot_for(int expert) const;
    int expert_in_slot(int slot) const;

    // Update recency for an existing warm hit. Returns false for non-warm
    // experts or invalid IDs.
    bool touch_warm(int expert);

    // Insert a cold expert. protected_slots is indexed by absolute slot and
    // prevents eviction of slots used by the current graph/update transaction.
    insertion insert_warm(int expert, const std::vector<bool> & protected_slots);

    // Return the empty or oldest LRU slot that insert_warm would use for a
    // cold expert, without mutating ownership.
    insertion insertion_target(const std::vector<bool> & protected_slots) const;

    // Reserve/release helpers for asynchronous fills. Eviction immediately
    // maps the old expert to the sentinel; binding is allowed only after the
    // copy into the empty warm slot has completed.
    int evict_warm_slot(int slot);
    void bind_ready_warm_slot(int slot, int expert);

    // Move expert into a fixed slot. If it was warm, its old warm slot becomes
    // empty. Returns the previous fixed expert, or -1 for an empty fixed slot.
    int replace_fixed(int fixed_slot, int expert);

    // Rebase warm ages without changing ownership. Used at a safe request
    // boundary when request-local LRU aging is selected.
    void reset_warm_ages();

    const std::vector<int32_t> & lut() const;
    const std::vector<int32_t> & slot_experts() const;
    const std::vector<uint64_t> & warm_ages() const;

    bool validate(std::string * error = nullptr) const;

private:
    bool valid_expert(int expert) const;
    bool valid_warm_slot(int slot) const;
    void bind_warm(int slot, int expert);

    int n_expert_ = 0;
    int n_fixed_  = 0;
    int n_warm_   = 0;
    int sentinel_ = 0;
    uint64_t clock_ = 0;
    std::vector<int32_t> lut_;
    std::vector<int32_t> slot_expert_;
    std::vector<uint64_t> warm_age_;
};

} // namespace llama_expert_cache
