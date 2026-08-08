#include "llama-kv-layer-policy.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void expect_invalid(const std::string & text, uint32_t n_layer) {
    bool failed = false;
    try {
        (void) llama_kv_layer_policy::parse_layer_set(text, n_layer);
    } catch (const std::exception &) {
        failed = true;
    }
    require(failed, "invalid policy was accepted");
}

int main() {
    {
        const std::vector<uint8_t> selected = llama_kv_layer_policy::parse_layer_set("", 40);
        for (const uint8_t value : selected) {
            require(value == 0, "empty policy selected a layer");
        }
    }

    {
        const std::vector<uint8_t> selected = llama_kv_layer_policy::parse_layer_set("3, 7, 11-15,39", 40);
        for (uint32_t il = 0; il < selected.size(); ++il) {
            const bool expected = il == 3 || il == 7 || (il >= 11 && il <= 15) || il == 39;
            require((selected[il] != 0) == expected, "valid policy selected the wrong layers");
        }
    }

    expect_invalid(",", 40);
    expect_invalid("3,", 40);
    expect_invalid("3,,7", 40);
    expect_invalid("-1", 40);
    expect_invalid("7-3", 40);
    expect_invalid("3-7-9", 40);
    expect_invalid("40", 40);
    expect_invalid("3,3", 40);
    expect_invalid("3,2-4", 40);
    expect_invalid("3", 0);

    return 0;
}
