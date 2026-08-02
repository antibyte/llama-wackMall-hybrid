#include "llama-expert-placement.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace llama_expert_placement {
namespace {

// FIPS 180-4 SHA-256, adapted from ggml-opencl's self-contained public-domain
// reference. Keeping it local avoids adding a mandatory crypto dependency to
// libllama merely to bind a placement manifest to its source profile.
struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t buffer[64];
    size_t buffer_size;
};

constexpr uint32_t sha256_constants[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

uint32_t rotate_right(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
        words[i] = ((uint32_t) block[4*i] << 24) | ((uint32_t) block[4*i + 1] << 16) |
                ((uint32_t) block[4*i + 2] << 8) | (uint32_t) block[4*i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + s1 + choose + sha256_constants[i] + words[i];
        const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void sha256_init(sha256_context & context) {
    const uint32_t initial[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    };
    std::memcpy(context.state, initial, sizeof(initial));
    context.bit_length = 0;
    context.buffer_size = 0;
}

void sha256_update(sha256_context & context, const uint8_t * data, size_t length) {
    context.bit_length += (uint64_t) length*8;
    while (length > 0) {
        const size_t count = std::min(length, sizeof(context.buffer) - context.buffer_size);
        std::memcpy(context.buffer + context.buffer_size, data, count);
        context.buffer_size += count;
        data += count;
        length -= count;
        if (context.buffer_size == sizeof(context.buffer)) {
            sha256_compress(context.state, context.buffer);
            context.buffer_size = 0;
        }
    }
}

std::string sha256_final(sha256_context & context) {
    const uint64_t bit_length = context.bit_length;
    context.buffer[context.buffer_size++] = 0x80;
    if (context.buffer_size > 56) {
        while (context.buffer_size < 64) {
            context.buffer[context.buffer_size++] = 0;
        }
        sha256_compress(context.state, context.buffer);
        context.buffer_size = 0;
    }
    while (context.buffer_size < 56) {
        context.buffer[context.buffer_size++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        context.buffer[context.buffer_size++] = (uint8_t) (bit_length >> (8*i));
    }
    sha256_compress(context.state, context.buffer);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (int i = 0; i < 8; ++i) {
        for (int byte = 0; byte < 4; ++byte) {
            const uint8_t value = (uint8_t) (context.state[i] >> (24 - 8*byte));
            const size_t offset = (size_t) (i*4 + byte)*2;
            result[offset] = hex[value >> 4];
            result[offset + 1] = hex[value & 0x0f];
        }
    }
    return result;
}

bool parse_u64(const std::string & value, uint64_t & result) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') {
        return false;
    }
    result = (uint64_t) parsed;
    return true;
}

bool parse_int(const std::string & value, int & result) {
    uint64_t parsed = 0;
    if (!parse_u64(value, parsed) || parsed > INT_MAX) {
        return false;
    }
    result = (int) parsed;
    return true;
}

std::vector<std::string> split_exact(const std::string & line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

bool is_sha256(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c) || (c >= 'a' && c <= 'f');
    });
}

} // namespace

bool sha256_file(const std::string & path, std::string & digest, std::string & error) {
    digest.clear();
    error.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open profile '" + path + "' for hashing";
        return false;
    }
    sha256_context context;
    sha256_init(context);
    uint8_t buffer[64*1024];
    while (input) {
        input.read((char *) buffer, sizeof(buffer));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            sha256_update(context, buffer, (size_t) count);
        }
    }
    if (!input.eof()) {
        error = "failed while hashing profile '" + path + "'";
        return false;
    }
    digest = sha256_final(context);
    return true;
}

bool parse_manifest(
        const std::string & path,
        int expected_layers,
        int expected_experts,
        manifest & result,
        std::string & error) {
    result = {};
    error.clear();
    std::ifstream input(path);
    if (!input) {
        error = "cannot open placement manifest '" + path + "'";
        return false;
    }

    std::string line;
    int line_number = 0;
    if (!std::getline(input, line)) {
        error = "placement manifest is empty";
        return false;
    }
    line_number++;
    if (line != "# llama-wackmall-expert-placement-v1") {
        error = "line 1 must be '# llama-wackmall-expert-placement-v1'";
        return false;
    }

    std::map<std::string, std::string> metadata;
    bool found_header = false;
    while (std::getline(input, line)) {
        line_number++;
        if (line == "layer,fixed_slots,slot_bytes") {
            found_header = true;
            break;
        }
        if (line.rfind("# ", 0) != 0) {
            error = "line " + std::to_string(line_number) + " must be metadata or the CSV header";
            return false;
        }
        const size_t equals = line.find('=', 2);
        if (equals == std::string::npos || equals == 2 || equals + 1 == line.size()) {
            error = "invalid metadata at line " + std::to_string(line_number);
            return false;
        }
        const std::string key = line.substr(2, equals - 2);
        const std::string value = line.substr(equals + 1);
        if (!metadata.emplace(key, value).second) {
            error = "duplicate metadata key '" + key + "'";
            return false;
        }
    }
    if (!found_header) {
        error = "placement manifest is missing its CSV header";
        return false;
    }

    const char * required[] = {
        "model_architecture", "model_file_size", "model_layers", "model_experts",
        "profile_sha256", "objective", "fixed_budget_bytes", "fixed_bytes_used",
        "sentinel_bytes", "min_slots", "max_slots",
    };
    for (const char * key : required) {
        if (!metadata.count(key)) {
            error = "placement manifest is missing metadata key '" + std::string(key) + "'";
            return false;
        }
    }
    result.model_architecture = metadata["model_architecture"];
    result.profile_sha256 = metadata["profile_sha256"];
    if (result.model_architecture.empty() || !is_sha256(result.profile_sha256) ||
            !parse_u64(metadata["model_file_size"], result.model_file_size) ||
            !parse_int(metadata["model_layers"], result.model_layers) ||
            !parse_int(metadata["model_experts"], result.model_experts) ||
            !parse_u64(metadata["fixed_budget_bytes"], result.fixed_budget_bytes) ||
            !parse_u64(metadata["fixed_bytes_used"], result.fixed_bytes_used) ||
            !parse_u64(metadata["sentinel_bytes"], result.sentinel_bytes) ||
            !parse_int(metadata["min_slots"], result.min_slots) ||
            !parse_int(metadata["max_slots"], result.max_slots)) {
        error = "placement manifest contains invalid metadata values";
        return false;
    }
    if (metadata["objective"] != "counts-per-byte" && metadata["objective"] != "counts") {
        error = "placement objective must be counts-per-byte or counts";
        return false;
    }
    if (result.model_layers != expected_layers || result.model_experts != expected_experts) {
        error = "placement dimensions " + std::to_string(result.model_layers) + "x" +
                std::to_string(result.model_experts) + " do not match model " +
                std::to_string(expected_layers) + "x" + std::to_string(expected_experts);
        return false;
    }
    if (result.min_slots < 1 || result.max_slots < result.min_slots || result.max_slots > expected_experts) {
        error = "placement min/max slot bounds are invalid";
        return false;
    }

    result.layers.assign(expected_layers, {});
    std::vector<bool> seen(expected_layers, false);
    uint64_t fixed_sum = 0;
    uint64_t sentinel_sum = 0;
    while (std::getline(input, line)) {
        line_number++;
        const auto fields = split_exact(line);
        int layer = -1;
        int slots = 0;
        uint64_t slot_bytes = 0;
        if (fields.size() != 3 || !parse_int(fields[0], layer) || !parse_int(fields[1], slots) ||
                !parse_u64(fields[2], slot_bytes)) {
            error = "invalid placement row at line " + std::to_string(line_number);
            return false;
        }
        if (layer < 0 || layer >= expected_layers || seen[layer] ||
                slots < result.min_slots || slots > result.max_slots || slot_bytes == 0) {
            error = "out-of-range or duplicate placement row at line " + std::to_string(line_number);
            return false;
        }
        if ((uint64_t) slots > UINT64_MAX / slot_bytes || fixed_sum > UINT64_MAX - (uint64_t) slots*slot_bytes ||
                sentinel_sum > UINT64_MAX - slot_bytes) {
            error = "placement byte total overflows uint64";
            return false;
        }
        seen[layer] = true;
        result.layers[layer] = {slots, slot_bytes};
        fixed_sum += (uint64_t) slots*slot_bytes;
        sentinel_sum += slot_bytes;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        error = "placement manifest does not contain every model layer exactly once";
        return false;
    }
    if (fixed_sum != result.fixed_bytes_used || sentinel_sum != result.sentinel_bytes ||
            result.fixed_bytes_used > result.fixed_budget_bytes) {
        error = "placement byte totals do not match manifest metadata";
        return false;
    }
    return true;
}

} // namespace llama_expert_placement
