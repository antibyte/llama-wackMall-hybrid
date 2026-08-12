#include "llama-graph.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const int32_t parents[] = { -1, 0, 0, 1 };
    const std::vector<int32_t> expected = {
        1, 2, 3,
        2, 3, 4,
        2, 3, 5,
        3, 4, 6,
    };

    const auto result = llm_tree_conv_state_indices(parents, 4, 4);
    assert(result == expected);

    const int32_t invalid[] = { -1, 2 };
    assert(llm_tree_conv_state_indices(invalid, 2, 4).empty());
    assert(llm_tree_conv_state_indices(parents, 4, 1).empty());

    std::puts("all tree state tests passed");
    return 0;
}
