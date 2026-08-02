#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_expert_placement {

struct layer_config {
    int fixed_slots = 0;
    uint64_t slot_bytes = 0;
};

struct manifest {
    std::string model_architecture;
    std::string profile_sha256;
    int model_layers = 0;
    int model_experts = 0;
    int min_slots = 0;
    int max_slots = 0;
    uint64_t model_file_size = 0;
    uint64_t fixed_budget_bytes = 0;
    uint64_t fixed_bytes_used = 0;
    uint64_t sentinel_bytes = 0;
    std::vector<layer_config> layers;
};

bool parse_manifest(
        const std::string & path,
        int expected_layers,
        int expected_experts,
        manifest & result,
        std::string & error);

bool sha256_file(const std::string & path, std::string & digest, std::string & error);

} // namespace llama_expert_placement
