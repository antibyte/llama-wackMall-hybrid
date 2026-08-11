#include "llama-expert-cache.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using llama_expert_cache::location;
using llama_expert_cache::second_hit_admission;
using llama_expert_cache::state;

static void check(bool condition) {
    if (!condition) {
        abort();
    }
}

static void require_valid(const state & cache) {
    std::string error;
    check(cache.validate(&error));
    check(error.empty());
}

static void test_layout_and_sentinel() {
    state cache(8, 2, 2, {4, 1});
    check(cache.sentinel() == 4);
    check(cache.n_slots() == 5);
    check(cache.slot_for(4) == 0);
    check(cache.slot_for(1) == 1);
    check(cache.expert_in_slot(cache.sentinel()) == -1);
    check(cache.locate(4) == location::fixed);
    check(cache.locate(3) == location::cold);
    require_valid(cache);
}

static void test_lru_and_fixed_protection() {
    state cache(8, 2, 2, {0, 1});
    std::vector<bool> protect(cache.n_slots(), false);

    const auto a = cache.insert_warm(2, protect);
    const auto b = cache.insert_warm(3, protect);
    check(a.slot == 2 && a.evicted == -1);
    check(b.slot == 3 && b.evicted == -1);
    check(cache.touch_warm(2));

    const auto target = cache.insertion_target(protect);
    check(target.slot == 3 && target.evicted == 3);
    check(cache.locate(3) == location::warm);

    const auto c = cache.insert_warm(4, protect);
    check(c.slot == 3 && c.evicted == 3);
    check(cache.locate(3) == location::cold);
    check(cache.locate(4) == location::warm);
    check(cache.expert_in_slot(0) == 0);
    check(cache.expert_in_slot(1) == 1);
    require_valid(cache);
}

static void test_protected_slot_and_full_transaction() {
    state cache(8, 1, 1, {0});
    std::vector<bool> protect(cache.n_slots(), false);
    const auto first = cache.insert_warm(1, protect);
    check(first.slot == 1);
    protect[first.slot] = true;
    check(!cache.insert_warm(2, protect));
    check(cache.slot_for(1) == 1);
    check(cache.locate(2) == location::cold);
    require_valid(cache);
}

static void test_warm_to_fixed_promotion() {
    state cache(8, 2, 2, {0, 1});
    std::vector<bool> protect(cache.n_slots(), false);
    const auto inserted = cache.insert_warm(5, protect);
    check((bool) inserted);
    const int warm_slot = inserted.slot;

    const int evicted = cache.replace_fixed(1, 5);
    check(evicted == 1);
    check(cache.locate(5) == location::fixed);
    check(cache.locate(1) == location::cold);
    check(cache.expert_in_slot(warm_slot) == -1);
    check(cache.slot_for(1) == cache.sentinel());
    require_valid(cache);
}

static void test_async_reserve_and_publish() {
    state cache(8, 2, 2, {0, 1});
    std::vector<bool> protect(cache.n_slots(), false);
    check((bool) cache.insert_warm(2, protect));
    check((bool) cache.insert_warm(3, protect));

    const auto target = cache.insertion_target(protect);
    check(target.slot == 2 && target.evicted == 2);
    check(cache.evict_warm_slot(target.slot) == 2);
    check(cache.locate(2) == location::cold);
    check(cache.expert_in_slot(target.slot) == -1);
    require_valid(cache);

    cache.bind_ready_warm_slot(target.slot, 4);
    check(cache.locate(4) == location::warm);
    check(cache.slot_for(4) == target.slot);
    require_valid(cache);
}

static void test_batch_lut_stability_and_request_age_reset() {
    state cache(8, 2, 2, {0, 1});
    std::vector<bool> protect(cache.n_slots(), false);
    check((bool) cache.insert_warm(2, protect));
    check((bool) cache.insert_warm(3, protect));

    // A submitted graph/verify batch only performs lookups. Warm touches may
    // update host-side recency, but cannot change the LUT visible to the graph.
    const auto before_batch = cache.lut();
    check(cache.locate(0) == location::fixed);
    check(cache.touch_warm(2));
    check(cache.touch_warm(3));
    check(cache.lut() == before_batch);

    // Request-local aging is a rebase, not a residency reset.
    cache.reset_warm_ages();
    check(cache.lut() == before_batch);
    check(cache.warm_ages()[0] == 1);
    check(cache.warm_ages()[1] == 2);
    require_valid(cache);
}

static void test_invalid_initialization() {
    bool threw = false;
    try {
        state cache(4, 2, 1, {1, 1});
        (void) cache;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw);
}

static void test_second_hit_admission() {
    second_hit_admission admission;
    admission.reset(8, 3);

    admission.next_epoch();
    check(!admission.record_miss(4));

    admission.next_epoch();
    check(admission.record_miss(4));
    admission.mark_resident(4);

    admission.next_epoch();
    check(!admission.record_miss(4));
    admission.next_epoch();
    admission.next_epoch();
    admission.next_epoch();
    admission.next_epoch();
    check(!admission.record_miss(4));

    check(!admission.record_miss(-1));
    check(!admission.record_miss(8));
    check(admission.epoch() == 7);
    check(admission.window() == 3);
}

static void test_invalid_second_hit_initialization() {
    second_hit_admission admission;
    bool threw = false;
    try {
        admission.reset(4, 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw);
}

int main() {
    test_layout_and_sentinel();
    test_lru_and_fixed_protection();
    test_protected_slot_and_full_transaction();
    test_warm_to_fixed_promotion();
    test_async_reserve_and_publish();
    test_batch_lut_stability_and_request_age_reset();
    test_invalid_initialization();
    test_second_hit_admission();
    test_invalid_second_hit_initialization();
    return 0;
}
