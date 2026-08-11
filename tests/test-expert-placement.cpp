#include "llama-expert-placement.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using llama_expert_placement::manifest;
using llama_expert_placement::parse_manifest;
using llama_expert_placement::sha256_file;

static void check(bool condition) {
    if (!condition) {
        abort();
    }
}

static std::string valid_manifest() {
    return
        "# llama-wackmall-expert-placement-v1\n"
        "# model_architecture=test\n"
        "# model_file_size=123\n"
        "# model_layers=2\n"
        "# model_experts=4\n"
        "# profile_sha256=0000000000000000000000000000000000000000000000000000000000000000\n"
        "# objective=counts-per-byte\n"
        "# fixed_budget_bytes=60\n"
        "# fixed_bytes_used=50\n"
        "# sentinel_bytes=30\n"
        "# min_slots=1\n"
        "# max_slots=3\n"
        "layer,fixed_slots,slot_bytes\n"
        "0,1,10\n"
        "1,2,20\n";
}

static std::filesystem::path write_temp(const std::string & content, const char * name) {
    const auto path = std::filesystem::temp_directory_path() /
            (std::string("llama-expert-placement-") + name + "-" + std::to_string(std::rand()) + ".csv");
    std::ofstream(path) << content;
    return path;
}

static void test_valid() {
    const auto path = write_temp(valid_manifest(), "valid");
    manifest parsed;
    std::string error;
    check(parse_manifest(path.string(), 2, 4, parsed, error));
    check(error.empty());
    check(parsed.layers.size() == 2);
    check(parsed.layers[0].fixed_slots == 1 && parsed.layers[0].slot_bytes == 10);
    check(parsed.layers[1].fixed_slots == 2 && parsed.layers[1].slot_bytes == 20);
    std::filesystem::remove(path);
}

static void test_dimension_mismatch() {
    const auto path = write_temp(valid_manifest(), "dims");
    manifest parsed;
    std::string error;
    check(!parse_manifest(path.string(), 3, 4, parsed, error));
    check(error.find("do not match") != std::string::npos);
    std::filesystem::remove(path);
}

static void test_duplicate_layer() {
    std::string content = valid_manifest();
    content.replace(content.rfind("1,2,20"), 6, "0,2,20");
    const auto path = write_temp(content, "duplicate");
    manifest parsed;
    std::string error;
    check(!parse_manifest(path.string(), 2, 4, parsed, error));
    check(error.find("duplicate") != std::string::npos);
    std::filesystem::remove(path);
}

static void test_byte_mismatch() {
    std::string content = valid_manifest();
    content.replace(content.find("fixed_bytes_used=50"), 19, "fixed_bytes_used=49");
    const auto path = write_temp(content, "bytes");
    manifest parsed;
    std::string error;
    check(!parse_manifest(path.string(), 2, 4, parsed, error));
    check(error.find("byte totals") != std::string::npos);
    std::filesystem::remove(path);
}

static void test_sha256() {
    const auto path = write_temp("abc", "sha256");
    std::string digest;
    std::string error;
    check(sha256_file(path.string(), digest, error));
    check(error.empty());
    check(digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::filesystem::remove(path);
}

int main() {
    test_valid();
    test_dimension_mismatch();
    test_duplicate_layer();
    test_byte_mismatch();
    test_sha256();
    return 0;
}
