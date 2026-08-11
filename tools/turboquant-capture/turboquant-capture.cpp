// Copyright (c) 2026 llama-wackMall contributors
// SPDX-License-Identifier: Apache-2.0

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct capture_options {
    std::filesystem::path output_dir;
    std::filesystem::path final_logits_path;
    std::filesystem::path logits_window_path;
    std::filesystem::path attention_replay_dir;
    std::filesystem::path attention_live_mask_dir;
    std::size_t max_tokens_per_layer = 256;
    std::size_t prompt_max_tokens = 0;
    std::size_t logits_window_start = 0;
    bool logits_window_start_set = false;
    bool attention_live_mask_continue = false;
    int layer_min = 0;
    int layer_max = INT_MAX;
    bool capture_values = false;
};

void print_usage(int, char ** argv) {
    std::printf("\nTurboQuant Q/K capture:\n\n");
    std::printf("  %s -m model.gguf -f prompt.txt -n 32 --capture-dir /tmp/tq-capture\n", argv[0]);
    std::printf("\nCapture-specific options:\n");
    std::printf("  --capture-dir DIR          enable capture into a new directory\n");
    std::printf("  --capture-max-tokens N     maximum captured tokens per layer (default: 256)\n");
    std::printf("  --capture-layer-min N      first layer to capture (default: 0)\n");
    std::printf("  --capture-layer-max N      last layer to capture (default: all)\n");
    std::printf("  --capture-values           also capture a materialized pre-attention V tensor\n\n");
    std::printf("  --prompt-max-tokens N      truncate the tokenized prompt to its first N tokens\n");
    std::printf("  --dump-final-logits FILE   write final prompt logits as a new raw F32 file\n\n");
    std::printf("  --dump-logits-window FILE  write raw F32 logits from a prompt position onward\n");
    std::printf("  --logits-window-start N    zero-based first prompt position for the logit window\n\n");
    std::printf("  --attention-replay DIR     apply an offline attention delta replay\n\n");
    std::printf("  --attention-live-mask DIR  apply a diagnostic layer-local exported keep set\n\n");
    std::printf("  --attention-live-mask-continue  keep applying the fixed mask during decode\n\n");
    std::printf("Without --capture-dir the tool is a deterministic no-capture control.\n\n");
}

long parse_long(const char * value, const char * option, long minimum, long maximum) {
    errno = 0;
    char * end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || result < minimum || result > maximum) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return result;
}

std::vector<char *> extract_capture_options(int argc, char ** argv, capture_options & options) {
    std::vector<char *> forwarded;
    forwarded.reserve(static_cast<std::size_t>(argc));
    forwarded.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * option) -> char * {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + option);
            }
            return argv[i];
        };
        if (arg == "--capture-dir") {
            options.output_dir = value("--capture-dir");
        } else if (arg == "--capture-max-tokens") {
            options.max_tokens_per_layer = static_cast<std::size_t>(parse_long(value("--capture-max-tokens"), "--capture-max-tokens", 1, INT_MAX));
        } else if (arg == "--capture-layer-min") {
            options.layer_min = static_cast<int>(parse_long(value("--capture-layer-min"), "--capture-layer-min", 0, INT_MAX));
        } else if (arg == "--capture-layer-max") {
            options.layer_max = static_cast<int>(parse_long(value("--capture-layer-max"), "--capture-layer-max", 0, INT_MAX));
        } else if (arg == "--capture-values") {
            options.capture_values = true;
        } else if (arg == "--prompt-max-tokens") {
            options.prompt_max_tokens = static_cast<std::size_t>(parse_long(value("--prompt-max-tokens"), "--prompt-max-tokens", 1, INT_MAX));
        } else if (arg == "--dump-final-logits") {
            options.final_logits_path = value("--dump-final-logits");
        } else if (arg == "--dump-logits-window") {
            options.logits_window_path = value("--dump-logits-window");
        } else if (arg == "--logits-window-start") {
            options.logits_window_start = static_cast<std::size_t>(parse_long(
                value("--logits-window-start"), "--logits-window-start", 0, INT_MAX));
            options.logits_window_start_set = true;
        } else if (arg == "--attention-replay") {
            options.attention_replay_dir = value("--attention-replay");
        } else if (arg == "--attention-live-mask") {
            options.attention_live_mask_dir = value("--attention-live-mask");
        } else if (arg == "--attention-live-mask-continue") {
            options.attention_live_mask_continue = true;
        } else {
            if (arg == "-h" || arg == "--help" || arg == "--usage") {
                print_usage(argc, argv);
            }
            forwarded.push_back(argv[i]);
        }
    }
    if (options.layer_min > options.layer_max) {
        throw std::invalid_argument("capture layer minimum exceeds maximum");
    }
    if (options.logits_window_path.empty() != !options.logits_window_start_set) {
        throw std::invalid_argument("--dump-logits-window and --logits-window-start must be used together");
    }
    if (!options.attention_replay_dir.empty() && !options.attention_live_mask_dir.empty()) {
        throw std::invalid_argument("--attention-replay and --attention-live-mask are mutually exclusive");
    }
    if (options.attention_live_mask_continue && options.attention_live_mask_dir.empty()) {
        throw std::invalid_argument("--attention-live-mask-continue requires --attention-live-mask");
    }
    return forwarded;
}

bool little_endian_host() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t *>(&value) == 1;
}

bool parse_layer_name(const char * name, const char * prefix, int & layer) {
    const std::size_t prefix_size = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_size) != 0 || name[prefix_size] != '-') {
        return false;
    }
    const char * value = name + prefix_size + 1;
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    layer = static_cast<int>(parsed);
    return true;
}

std::string json_escape(const std::string & value) {
    std::ostringstream result;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (ch < 0x20) {
                    result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
                } else {
                    result << ch;
                }
        }
    }
    return result.str();
}

std::string layer_filename(int layer, const char * suffix) {
    std::ostringstream name;
    name << "layer-" << std::setw(3) << std::setfill('0') << layer << suffix;
    return name.str();
}

struct tensor_sample {
    std::vector<float> values;
    std::size_t head_dim = 0;
    std::size_t heads = 0;
    std::size_t tokens = 0;
    std::uint64_t graph = 0;
};

struct pending_layer {
    tensor_sample query;
    tensor_sample key;
    tensor_sample value;
    bool has_query = false;
    bool has_key = false;
    bool has_value = false;
};

class capture_state {
public:
    explicit capture_state(capture_options options) : options_(std::move(options)) {
        if (!little_endian_host()) {
            throw std::runtime_error("raw float32 capture currently requires a little-endian host");
        }
        if (options_.output_dir.empty()) {
            return;
        }
        if (std::filesystem::exists(options_.output_dir)) {
            throw std::runtime_error("capture directory already exists: " + options_.output_dir.string());
        }
        if (!std::filesystem::create_directories(options_.output_dir)) {
            throw std::runtime_error("failed to create capture directory: " + options_.output_dir.string());
        }
        std::ofstream manifest(options_.output_dir / "manifest.csv", std::ios::binary);
        manifest << "event,graph,layer,tokens,head_dim,q_heads,kv_heads,key_rows,pair_rows\n";
        if (!manifest) {
            throw std::runtime_error("failed to initialize capture manifest");
        }
    }

    bool enabled() const {
        return !options_.output_dir.empty();
    }

    bool wants(const ggml_tensor * tensor) const {
        if (!enabled() || !error_.empty()) {
            return false;
        }
        if (std::strcmp(tensor->name, "model.input_embed") == 0) {
            return true;
        }
        int layer = -1;
        const bool is_query = parse_layer_name(tensor->name, "Qcur", layer);
        const bool is_key = !is_query && parse_layer_name(tensor->name, "Kcur", layer);
        const bool is_value = options_.capture_values && !is_query && !is_key &&
                parse_layer_name(tensor->name, "Vcapture", layer);
        if (!is_query && !is_key && !is_value) {
            return false;
        }
        const bool expected_op = is_value ? tensor->op == GGML_OP_DUP : tensor->op == GGML_OP_ROPE;
        return expected_op && layer >= options_.layer_min && layer <= options_.layer_max &&
                captured_tokens(layer) < options_.max_tokens_per_layer;
    }

    bool observe(ggml_tensor * tensor) noexcept {
        try {
            if (std::strcmp(tensor->name, "model.input_embed") == 0) {
                if (!pending_.empty()) {
                    throw std::runtime_error("incomplete Q/K/V capture at graph boundary");
                }
                ++graph_;
                return true;
            }

            int layer = -1;
            const bool is_query = parse_layer_name(tensor->name, "Qcur", layer);
            const bool is_key = !is_query && parse_layer_name(tensor->name, "Kcur", layer);
            const bool is_value = options_.capture_values && !is_query && !is_key &&
                    parse_layer_name(tensor->name, "Vcapture", layer);
            const bool expected_op = is_value ? tensor->op == GGML_OP_DUP : tensor->op == GGML_OP_ROPE;
            if ((!is_query && !is_key && !is_value) || !expected_op) {
                return true;
            }
            capture_tensor(tensor, layer, is_query, is_key);
            return true;
        } catch (const std::exception & exception) {
            error_ = exception.what();
            return false;
        }
    }

    const std::string & error() const {
        return error_;
    }

    std::size_t event_count() const {
        return event_;
    }

    std::uint64_t graph_count() const {
        return graph_;
    }

    std::size_t total_tokens() const {
        std::size_t result = 0;
        for (const auto & item : captured_tokens_) {
            result += item.second;
        }
        return result;
    }

    void finish(
            const std::string & model_path,
            const std::vector<llama_token> & generated,
            std::uint64_t token_hash,
            std::uint64_t prompt_token_hash,
            std::size_t prompt_tokens,
            std::size_t context_size,
            std::size_t batch_size,
            std::size_t ubatch_size,
            ggml_type cache_type_k,
            ggml_type cache_type_v) {
        if (!enabled()) {
            return;
        }
        if (!error_.empty()) {
            throw std::runtime_error(error_);
        }
        if (!pending_.empty()) {
            throw std::runtime_error("capture ended with an incomplete Q/K/V group");
        }

        const std::filesystem::path normalized_model = std::filesystem::absolute(model_path).lexically_normal();
        std::ofstream output(options_.output_dir / "summary.json", std::ios::binary);
        output << "{\n"
               << "  \"model\": \"" << json_escape(normalized_model.string()) << "\",\n"
               << "  \"model_bytes\": " << std::filesystem::file_size(normalized_model) << ",\n"
               << "  \"capture_stage\": \""
               << (options_.capture_values ? "post-rope-qk-pre-attention-v" : "post-rope")
               << "\",\n"
               << "  \"capture_values\": " << (options_.capture_values ? "true" : "false") << ",\n"
               << "  \"mtp_included\": false,\n"
               << "  \"prompt_tokens\": " << prompt_tokens << ",\n"
               << "  \"prompt_token_hash_fnv1a64\": \"" << std::hex << std::setw(16) << std::setfill('0') << prompt_token_hash << std::dec << "\",\n"
               << "  \"context\": " << context_size << ",\n"
               << "  \"batch\": " << batch_size << ",\n"
               << "  \"ubatch\": " << ubatch_size << ",\n"
               << "  \"cache_type_k\": \"" << ggml_type_name(cache_type_k) << "\",\n"
               << "  \"cache_type_v\": \"" << ggml_type_name(cache_type_v) << "\",\n"
               << "  \"graphs\": " << graph_ << ",\n"
               << "  \"events\": " << event_ << ",\n"
               << "  \"captured_layer_tokens\": " << total_tokens() << ",\n"
               << "  \"max_tokens_per_layer\": " << options_.max_tokens_per_layer << ",\n"
               << "  \"generated_token_hash_fnv1a64\": \"" << std::hex << std::setw(16) << std::setfill('0') << token_hash << std::dec << "\",\n"
               << "  \"generated_tokens\": [";
        for (std::size_t i = 0; i < generated.size(); ++i) {
            output << (i == 0 ? "" : ", ") << generated[i];
        }
        output << "]\n}\n";
        if (!output) {
            throw std::runtime_error("failed to write capture summary");
        }
    }

private:
    std::size_t captured_tokens(int layer) const {
        const auto found = captured_tokens_.find(layer);
        return found == captured_tokens_.end() ? 0 : found->second;
    }

    tensor_sample copy_tensor(ggml_tensor * tensor, std::size_t tokens) const {
        if (tensor->type != GGML_TYPE_F32) {
            throw std::runtime_error(std::string("capture requires F32 Q/K/V tensors, got ") + ggml_type_name(tensor->type));
        }
        if (!ggml_is_contiguous(tensor)) {
            throw std::runtime_error("capture requires contiguous Q/K/V tensors");
        }
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1) {
            throw std::runtime_error("unexpected Q/K/V tensor dimensions");
        }

        tensor_sample sample;
        sample.head_dim = static_cast<std::size_t>(tensor->ne[0]);
        sample.heads = static_cast<std::size_t>(tensor->ne[1]);
        sample.tokens = tokens;
        sample.graph = graph_;
        sample.values.resize(sample.head_dim * sample.heads * sample.tokens);
        ggml_backend_tensor_get(tensor, sample.values.data(), 0, sample.values.size() * sizeof(float));
        return sample;
    }

    void capture_tensor(ggml_tensor * tensor, int layer, bool is_query, bool is_key) {
        const std::size_t already = captured_tokens(layer);
        if (already >= options_.max_tokens_per_layer) {
            return;
        }
        const std::size_t available = static_cast<std::size_t>(tensor->ne[2]);
        const std::size_t tokens = std::min(available, options_.max_tokens_per_layer - already);
        pending_layer & pending = pending_[layer];
        if (is_query) {
            if (pending.has_query) {
                throw std::runtime_error("duplicate Q tensor for layer " + std::to_string(layer));
            }
            pending.query = copy_tensor(tensor, tokens);
            pending.has_query = true;
        } else if (is_key) {
            if (pending.has_key) {
                throw std::runtime_error("duplicate K tensor for layer " + std::to_string(layer));
            }
            pending.key = copy_tensor(tensor, tokens);
            pending.has_key = true;
        } else {
            if (pending.has_value) {
                throw std::runtime_error("duplicate V tensor for layer " + std::to_string(layer));
            }
            pending.value = copy_tensor(tensor, tokens);
            pending.has_value = true;
        }
        if (pending.has_query && pending.has_key && (!options_.capture_values || pending.has_value)) {
            write_pair(layer, pending);
            pending_.erase(layer);
        }
    }

    void write_pair(int layer, const pending_layer & pending) {
        const tensor_sample & query = pending.query;
        const tensor_sample & key = pending.key;
        if (query.graph != key.graph || query.head_dim != key.head_dim || query.tokens != key.tokens) {
            throw std::runtime_error("Q/K capture mismatch for layer " + std::to_string(layer));
        }
        if (key.heads == 0 || query.heads % key.heads != 0) {
            throw std::runtime_error("Q heads are not divisible by KV heads for layer " + std::to_string(layer));
        }
        if (options_.capture_values &&
                (pending.value.graph != query.graph || pending.value.head_dim != query.head_dim ||
                 pending.value.heads != key.heads || pending.value.tokens != query.tokens)) {
            throw std::runtime_error("Q/K/V capture mismatch for layer " + std::to_string(layer));
        }

        const std::filesystem::path key_path = options_.output_dir / layer_filename(layer, "-keys.f32");
        std::ofstream key_output(key_path, std::ios::binary | std::ios::app);
        key_output.write(reinterpret_cast<const char *>(key.values.data()), static_cast<std::streamsize>(key.values.size() * sizeof(float)));
        if (!key_output) {
            throw std::runtime_error("failed to append " + key_path.string());
        }

        const std::filesystem::path pair_path = options_.output_dir / layer_filename(layer, "-qk-pairs.f32");
        std::ofstream pair_output(pair_path, std::ios::binary | std::ios::app);
        const std::size_t query_heads_per_kv = query.heads / key.heads;
        for (std::size_t token = 0; token < query.tokens; ++token) {
            for (std::size_t query_head = 0; query_head < query.heads; ++query_head) {
                const std::size_t key_head = query_head / query_heads_per_kv;
                const float * query_row = query.values.data() + (token * query.heads + query_head) * query.head_dim;
                const float * key_row = key.values.data() + (token * key.heads + key_head) * key.head_dim;
                pair_output.write(reinterpret_cast<const char *>(query_row), static_cast<std::streamsize>(query.head_dim * sizeof(float)));
                pair_output.write(reinterpret_cast<const char *>(key_row), static_cast<std::streamsize>(key.head_dim * sizeof(float)));
            }
        }
        if (!pair_output) {
            throw std::runtime_error("failed to append " + pair_path.string());
        }

        if (options_.capture_values) {
            const std::filesystem::path value_path = options_.output_dir / layer_filename(layer, "-values.f32");
            std::ofstream value_output(value_path, std::ios::binary | std::ios::app);
            value_output.write(
                    reinterpret_cast<const char *>(pending.value.values.data()),
                    static_cast<std::streamsize>(pending.value.values.size() * sizeof(float)));
            if (!value_output) {
                throw std::runtime_error("failed to append " + value_path.string());
            }
        }


        std::ofstream manifest(options_.output_dir / "manifest.csv", std::ios::binary | std::ios::app);
        manifest << event_ << ',' << graph_ << ',' << layer << ',' << query.tokens << ',' << query.head_dim << ','
                 << query.heads << ',' << key.heads << ',' << query.tokens * key.heads << ','
                 << query.tokens * query.heads * 2 << '\n';
        if (!manifest) {
            throw std::runtime_error("failed to append capture manifest");
        }

        captured_tokens_[layer] += query.tokens;
        ++event_;
    }

    capture_options options_;
    std::map<int, pending_layer> pending_;
    std::map<int, std::size_t> captured_tokens_;
    std::string error_;
    std::uint64_t graph_ = 0;
    std::size_t event_ = 0;
};

struct replay_record {
    std::uint64_t graph = 0;
    int layer = -1;
    std::size_t tensor_token_offset = 0;
    std::size_t tokens = 0;
    std::size_t head_dim = 0;
    std::size_t heads = 0;
    std::size_t byte_offset = 0;
    bool applied = false;
};

std::vector<std::string> split_csv_row(const std::string & line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        fields.push_back(line.substr(start, comma == std::string::npos ? comma : comma - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (!fields.empty() && !fields.back().empty() && fields.back().back() == '\r') {
        fields.back().pop_back();
    }
    return fields;
}

std::size_t parse_size_field(const std::string & value, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("invalid replay ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t checked_replay_values(std::size_t tokens, std::size_t head_dim, std::size_t heads) {
    if (head_dim > std::numeric_limits<std::size_t>::max() / heads ||
        tokens > std::numeric_limits<std::size_t>::max() / (head_dim * heads)) {
        throw std::runtime_error("attention replay dimensions overflow size_t");
    }
    return tokens * head_dim * heads;
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream result;
    result << std::hex << std::setw(16) << std::setfill('0') << value;
    return result.str();
}

class attention_replay_state {
public:
    explicit attention_replay_state(const std::filesystem::path & directory) : directory_(directory) {
        if (directory_.empty()) {
            return;
        }
        const std::filesystem::path manifest_path = directory_ / "manifest.csv";
        const std::filesystem::path delta_path = directory_ / "attention-delta.f32";
        if (!std::filesystem::is_regular_file(manifest_path) || !std::filesystem::is_regular_file(delta_path)) {
            throw std::runtime_error("attention replay directory is incomplete: " + directory_.string());
        }

        std::ifstream manifest(manifest_path, std::ios::binary);
        std::string line;
        bool header_seen = false;
        while (std::getline(manifest, line)) {
            if (line.empty()) {
                continue;
            }
            if (line.rfind("# ", 0) == 0) {
                const std::size_t equal = line.find('=', 2);
                if (equal == std::string::npos || equal == 2) {
                    throw std::runtime_error("invalid attention replay metadata");
                }
                metadata_[line.substr(2, equal - 2)] = line.substr(equal + 1);
                continue;
            }
            if (!header_seen) {
                if (line != "graph,layer,tensor_token_offset,tokens,head_dim,heads,byte_offset\r" &&
                    line != "graph,layer,tensor_token_offset,tokens,head_dim,heads,byte_offset") {
                    throw std::runtime_error("invalid attention replay CSV header");
                }
                header_seen = true;
                continue;
            }
            const std::vector<std::string> fields = split_csv_row(line);
            if (fields.size() != 7) {
                throw std::runtime_error("invalid attention replay CSV row");
            }
            replay_record record;
            record.graph = parse_size_field(fields[0], "graph");
            const std::size_t layer = parse_size_field(fields[1], "layer");
            if (layer > INT_MAX) {
                throw std::runtime_error("attention replay layer exceeds INT_MAX");
            }
            record.layer = static_cast<int>(layer);
            record.tensor_token_offset = parse_size_field(fields[2], "tensor token offset");
            record.tokens = parse_size_field(fields[3], "token count");
            record.head_dim = parse_size_field(fields[4], "head dimension");
            record.heads = parse_size_field(fields[5], "head count");
            record.byte_offset = parse_size_field(fields[6], "byte offset");
            if (record.graph == 0 || record.tokens == 0 || record.head_dim == 0 || record.heads == 0 ||
                record.layer < 0) {
                throw std::runtime_error("attention replay CSV row contains zero or invalid dimensions");
            }
            records_.push_back(record);
        }
        if (manifest.bad() || !header_seen || records_.empty()) {
            throw std::runtime_error("attention replay manifest is empty or unreadable");
        }
        const char * required[] = {
            "schema", "model", "model_bytes", "prompt_tokens", "prompt_token_hash_fnv1a64",
            "context", "batch", "ubatch", "cache_type_k", "cache_type_v"
        };
        for (const char * name : required) {
            if (metadata_.find(name) == metadata_.end()) {
                throw std::runtime_error(std::string("attention replay is missing metadata: ") + name);
            }
        }
        if (metadata_["schema"] != "llama-wackmall-attention-delta-replay-v1") {
            throw std::runtime_error("unsupported attention replay schema");
        }

        const std::uintmax_t delta_bytes = std::filesystem::file_size(delta_path);
        if (delta_bytes == 0 || delta_bytes % sizeof(float) != 0 || delta_bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("invalid attention replay delta size");
        }
        delta_.resize(static_cast<std::size_t>(delta_bytes / sizeof(float)));
        std::ifstream delta_input(delta_path, std::ios::binary);
        delta_input.read(reinterpret_cast<char *>(delta_.data()), static_cast<std::streamsize>(delta_bytes));
        if (!delta_input) {
            throw std::runtime_error("failed to read attention replay delta");
        }
        for (std::size_t i = 0; i < records_.size(); ++i) {
            const replay_record & record = records_[i];
            const std::size_t values = checked_replay_values(record.tokens, record.head_dim, record.heads);
            if (record.byte_offset % sizeof(float) != 0 || record.byte_offset / sizeof(float) > delta_.size() ||
                values > delta_.size() - record.byte_offset / sizeof(float)) {
                throw std::runtime_error("attention replay record exceeds the delta file");
            }
            for (std::size_t j = 0; j < i; ++j) {
                if (records_[j].graph == record.graph && records_[j].layer == record.layer) {
                    throw std::runtime_error("attention replay duplicates a graph/layer record");
                }
            }
        }
    }

    bool enabled() const {
        return !directory_.empty();
    }

    void validate(
            const std::filesystem::path & model_path,
            std::uint64_t                 prompt_hash,
            std::size_t                   prompt_tokens,
            std::size_t                   context,
            std::size_t                   batch,
            std::size_t                   ubatch,
            ggml_type                     cache_type_k,
            ggml_type                     cache_type_v) const {
        if (!enabled()) {
            return;
        }
        const std::filesystem::path normalized_model = std::filesystem::absolute(model_path).lexically_normal();
        if (metadata_.at("model") != normalized_model.string() ||
            parse_size_field(metadata_.at("model_bytes"), "model bytes") != std::filesystem::file_size(normalized_model) ||
            parse_size_field(metadata_.at("prompt_tokens"), "prompt tokens") != prompt_tokens ||
            metadata_.at("prompt_token_hash_fnv1a64") != hex_u64(prompt_hash) ||
            parse_size_field(metadata_.at("context"), "context") != context ||
            parse_size_field(metadata_.at("batch"), "batch") != batch ||
            parse_size_field(metadata_.at("ubatch"), "ubatch") != ubatch ||
            metadata_.at("cache_type_k") != ggml_type_name(cache_type_k) ||
            metadata_.at("cache_type_v") != ggml_type_name(cache_type_v)) {
            throw std::runtime_error("attention replay does not match model, prompt, context, batch, or cache types");
        }
    }

    bool wants(const ggml_tensor * tensor) const {
        if (!enabled() || !error_.empty()) {
            return false;
        }
        if (std::strcmp(tensor->name, "model.input_embed") == 0) {
            return true;
        }
        int layer = -1;
        if (!parse_layer_name(tensor->name, "attn_pregate", layer)) {
            return false;
        }
        for (const replay_record & record : records_) {
            if (!record.applied && record.graph == graph_ && record.layer == layer) {
                return true;
            }
        }
        return false;
    }

    bool observe(ggml_tensor * tensor) noexcept {
        try {
            if (std::strcmp(tensor->name, "model.input_embed") == 0) {
                ++graph_;
                return true;
            }
            int layer = -1;
            if (!parse_layer_name(tensor->name, "attn_pregate", layer)) {
                return true;
            }
            for (replay_record & record : records_) {
                if (!record.applied && record.graph == graph_ && record.layer == layer) {
                    apply(tensor, record);
                    record.applied = true;
                    ++applied_;
                    return true;
                }
            }
            return true;
        } catch (const std::exception & exception) {
            error_ = exception.what();
            return false;
        }
    }

    void finish() const {
        if (!error_.empty()) {
            throw std::runtime_error(error_);
        }
        if (enabled() && applied_ != records_.size()) {
            throw std::runtime_error("attention replay ended before every delta record was applied");
        }
    }

    const std::string & error() const {
        return error_;
    }

    std::size_t applied() const {
        return applied_;
    }

private:
    void apply(ggml_tensor * tensor, const replay_record & record) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("attention replay requires a contiguous F32 2D attention tensor");
        }
        const std::size_t row_width = checked_replay_values(1, record.head_dim, record.heads);
        if (static_cast<std::size_t>(tensor->ne[0]) != row_width ||
            record.tensor_token_offset > static_cast<std::size_t>(tensor->ne[1]) ||
            record.tokens > static_cast<std::size_t>(tensor->ne[1]) - record.tensor_token_offset) {
            throw std::runtime_error("attention replay tensor dimensions do not match the record");
        }
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        const float * delta = delta_.data() + record.byte_offset / sizeof(float);
        float * destination = values.data() + record.tensor_token_offset * row_width;
        for (std::size_t i = 0; i < record.tokens * row_width; ++i) {
            destination[i] += delta[i];
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }

    std::filesystem::path directory_;
    std::map<std::string, std::string> metadata_;
    std::vector<replay_record> records_;
    std::vector<float> delta_;
    std::string error_;
    std::uint64_t graph_ = 0;
    std::size_t applied_ = 0;
};

class attention_live_mask_state {
public:
    attention_live_mask_state(const std::filesystem::path & directory, bool continuation) :
        directory_(directory), continuation_(continuation) {
        if (directory_.empty()) {
            return;
        }
        const std::filesystem::path manifest_path = directory_ / "manifest.csv";
        const std::filesystem::path keep_path = directory_ / "attention-keep.u8";
        if (!std::filesystem::is_regular_file(manifest_path) || !std::filesystem::is_regular_file(keep_path)) {
            throw std::runtime_error("attention live-mask directory is incomplete: " + directory_.string());
        }

        std::ifstream manifest(manifest_path, std::ios::binary);
        std::string line;
        bool header_seen = false;
        while (std::getline(manifest, line)) {
            if (line.empty()) {
                continue;
            }
            if (line.rfind("# ", 0) == 0) {
                const std::size_t equal = line.find('=', 2);
                if (equal == std::string::npos || equal == 2) {
                    throw std::runtime_error("invalid attention live-mask metadata");
                }
                metadata_[line.substr(2, equal - 2)] = line.substr(equal + 1);
                continue;
            }
            if (!header_seen) {
                if (line != "graph,layer,tensor_token_offset,tokens,head_dim,heads,byte_offset\r" &&
                    line != "graph,layer,tensor_token_offset,tokens,head_dim,heads,byte_offset") {
                    throw std::runtime_error("invalid attention live-mask CSV header");
                }
                header_seen = true;
                continue;
            }
            const std::vector<std::string> fields = split_csv_row(line);
            if (fields.size() != 7) {
                throw std::runtime_error("invalid attention live-mask CSV row");
            }
            replay_record record;
            record.graph = parse_size_field(fields[0], "graph");
            const std::size_t layer = parse_size_field(fields[1], "layer");
            if (layer > INT_MAX) {
                throw std::runtime_error("attention live-mask layer exceeds INT_MAX");
            }
            record.layer = static_cast<int>(layer);
            record.tensor_token_offset = parse_size_field(fields[2], "tensor token offset");
            record.tokens = parse_size_field(fields[3], "token count");
            record.head_dim = parse_size_field(fields[4], "head dimension");
            record.heads = parse_size_field(fields[5], "head count");
            record.byte_offset = parse_size_field(fields[6], "byte offset");
            if (record.graph == 0 || record.tokens == 0 || record.layer < 0) {
                throw std::runtime_error("attention live-mask CSV row contains invalid fields");
            }
            if (layer_ < 0) {
                layer_ = record.layer;
            } else if (layer_ != record.layer) {
                throw std::runtime_error("attention live-mask currently requires exactly one layer");
            }
            for (const replay_record & existing : records_) {
                if (existing.graph == record.graph && existing.layer == record.layer) {
                    throw std::runtime_error("attention live-mask duplicates a graph/layer record");
                }
            }
            records_.push_back(record);
            last_record_graph_ = std::max(last_record_graph_, record.graph);
        }
        if (manifest.bad() || !header_seen || records_.empty()) {
            throw std::runtime_error("attention live-mask manifest is empty or unreadable");
        }
        const char * required[] = {
            "schema", "model", "model_bytes", "prompt_tokens", "prompt_token_hash_fnv1a64",
            "context", "batch", "ubatch", "cache_type_k", "cache_type_v",
            "budget", "train_tokens", "eval_tokens"
        };
        for (const char * name : required) {
            if (metadata_.find(name) == metadata_.end()) {
                throw std::runtime_error(std::string("attention live-mask is missing metadata: ") + name);
            }
        }
        if (metadata_["schema"] != "llama-wackmall-attention-delta-replay-v1") {
            throw std::runtime_error("unsupported attention live-mask schema");
        }

        train_tokens_ = parse_size_field(metadata_["train_tokens"], "train tokens");
        const std::size_t budget = parse_size_field(metadata_["budget"], "budget");
        if (train_tokens_ == 0 || budget == 0 || budget > train_tokens_ ||
            parse_size_field(metadata_["eval_tokens"], "eval tokens") == 0) {
            throw std::runtime_error("attention live-mask has invalid policy dimensions");
        }
        if (std::filesystem::file_size(keep_path) != train_tokens_) {
            throw std::runtime_error("attention live-mask keep set has the wrong size");
        }
        keep_.resize(train_tokens_);
        std::ifstream keep_input(keep_path, std::ios::binary);
        keep_input.read(reinterpret_cast<char *>(keep_.data()), static_cast<std::streamsize>(keep_.size()));
        if (!keep_input) {
            throw std::runtime_error("failed to read attention live-mask keep set");
        }
        std::size_t retained = 0;
        for (std::uint8_t value : keep_) {
            if (value > 1) {
                throw std::runtime_error("attention live-mask keep set is not binary");
            }
            retained += value;
        }
        if (retained != budget) {
            throw std::runtime_error("attention live-mask keep set does not match its budget");
        }
    }

    bool enabled() const {
        return !directory_.empty();
    }

    int layer() const {
        return layer_;
    }

    void validate(
            const std::filesystem::path & model_path,
            std::uint64_t                 prompt_hash,
            std::size_t                   prompt_tokens,
            std::size_t                   context,
            std::size_t                   batch,
            std::size_t                   ubatch,
            ggml_type                     cache_type_k,
            ggml_type                     cache_type_v) const {
        if (!enabled()) {
            return;
        }
        const std::filesystem::path normalized_model = std::filesystem::absolute(model_path).lexically_normal();
        if (metadata_.at("model") != normalized_model.string() ||
            parse_size_field(metadata_.at("model_bytes"), "model bytes") != std::filesystem::file_size(normalized_model) ||
            parse_size_field(metadata_.at("prompt_tokens"), "prompt tokens") != prompt_tokens ||
            metadata_.at("prompt_token_hash_fnv1a64") != hex_u64(prompt_hash) ||
            parse_size_field(metadata_.at("context"), "context") != context ||
            parse_size_field(metadata_.at("batch"), "batch") != batch ||
            parse_size_field(metadata_.at("ubatch"), "ubatch") != ubatch ||
            metadata_.at("cache_type_k") != ggml_type_name(cache_type_k) ||
            metadata_.at("cache_type_v") != ggml_type_name(cache_type_v)) {
            throw std::runtime_error("attention live-mask does not match model, prompt, context, batch, or cache types");
        }
    }

    bool wants(const ggml_tensor * tensor) const {
        if (!enabled() || !error_.empty()) {
            return false;
        }
        if (std::strcmp(tensor->name, "model.input_embed") == 0) {
            return true;
        }
        int layer = -1;
        if (!parse_layer_name(tensor->name, "triattention_live_mask", layer)) {
            return false;
        }
        for (const replay_record & record : records_) {
            if (!record.applied && record.graph == graph_ && record.layer == layer) {
                return true;
            }
        }
        return continuation_ && graph_ > last_record_graph_ && layer == layer_;
    }

    bool observe(ggml_tensor * tensor) noexcept {
        try {
            if (std::strcmp(tensor->name, "model.input_embed") == 0) {
                ++graph_;
                return true;
            }
            int layer = -1;
            if (!parse_layer_name(tensor->name, "triattention_live_mask", layer)) {
                return true;
            }
            for (replay_record & record : records_) {
                if (!record.applied && record.graph == graph_ && record.layer == layer) {
                    apply(tensor, record);
                    record.applied = true;
                    ++applied_;
                    return true;
                }
            }
            if (continuation_ && graph_ > last_record_graph_ && layer == layer_) {
                replay_record continuation_record;
                continuation_record.graph = graph_;
                continuation_record.layer = layer;
                continuation_record.tensor_token_offset = 0;
                continuation_record.tokens = static_cast<std::size_t>(tensor->ne[1]);
                apply(tensor, continuation_record);
                ++continuation_applied_;
            }
            return true;
        } catch (const std::exception & exception) {
            error_ = exception.what();
            return false;
        }
    }

    void finish() const {
        if (!error_.empty()) {
            throw std::runtime_error(error_);
        }
        if (enabled() && applied_ != records_.size()) {
            throw std::runtime_error("attention live-mask ended before every record was applied");
        }
    }

    const std::string & error() const {
        return error_;
    }

    std::size_t applied() const {
        return applied_;
    }

    std::size_t continuation_applied() const {
        return continuation_applied_;
    }

    void finish_continuation(std::size_t expected) const {
        if (!error_.empty()) {
            throw std::runtime_error(error_);
        }
        if (continuation_ && continuation_applied_ != expected) {
            throw std::runtime_error("attention live-mask continuation record count is incomplete");
        }
    }

private:
    void apply(ggml_tensor * tensor, const replay_record & record) {
        if ((tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) ||
            !ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("attention live-mask requires a contiguous F16/F32 2D mask tensor");
        }
        const std::size_t key_count = static_cast<std::size_t>(tensor->ne[0]);
        const std::size_t query_count = static_cast<std::size_t>(tensor->ne[1]);
        if (train_tokens_ > key_count || record.tensor_token_offset > query_count ||
            record.tokens > query_count - record.tensor_token_offset) {
            throw std::runtime_error("attention live-mask tensor dimensions do not match the record");
        }

        if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> values(static_cast<std::size_t>(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(values[0]));
            const ggml_fp16_t negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            for (std::size_t query = record.tensor_token_offset;
                 query < record.tensor_token_offset + record.tokens; ++query) {
                for (std::size_t key = 0; key < train_tokens_; ++key) {
                    if (!keep_[key]) {
                        values[query*key_count + key] = negative_infinity;
                    }
                }
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(values[0]));
        } else {
            std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(values[0]));
            const float negative_infinity = -std::numeric_limits<float>::infinity();
            for (std::size_t query = record.tensor_token_offset;
                 query < record.tensor_token_offset + record.tokens; ++query) {
                for (std::size_t key = 0; key < train_tokens_; ++key) {
                    if (!keep_[key]) {
                        values[query*key_count + key] = negative_infinity;
                    }
                }
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(values[0]));
        }
    }

    std::filesystem::path directory_;
    std::map<std::string, std::string> metadata_;
    std::vector<replay_record> records_;
    std::vector<std::uint8_t> keep_;
    std::string error_;
    std::uint64_t graph_ = 0;
    std::uint64_t last_record_graph_ = 0;
    std::size_t train_tokens_ = 0;
    std::size_t applied_ = 0;
    std::size_t continuation_applied_ = 0;
    int layer_ = -1;
    bool continuation_ = false;
};

struct tool_callback_state {
    capture_state * capture = nullptr;
    attention_replay_state * replay = nullptr;
    attention_live_mask_state * live_mask = nullptr;
};

bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<tool_callback_state *>(user_data);
    if (ask) {
        return (state->capture && state->capture->wants(tensor)) ||
               (state->replay && state->replay->wants(tensor)) ||
               (state->live_mask && state->live_mask->wants(tensor));
    }
    return (!state->capture || state->capture->observe(tensor)) &&
           (!state->replay || state->replay->observe(tensor)) &&
           (!state->live_mask || state->live_mask->observe(tensor));
}

std::uint64_t hash_tokens(const std::vector<llama_token> & tokens) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const llama_token token : tokens) {
        const std::uint32_t value = static_cast<std::uint32_t>(token);
        for (unsigned int byte = 0; byte < sizeof(value); ++byte) {
            hash ^= (value >> (8 * byte)) & 0xffu;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

struct logits_window {
    std::size_t start = 0;
    std::size_t vocab_size = 0;
    std::vector<float> values;
};

bool decode_tokens(
        llama_context *                  context,
        const std::vector<llama_token> & tokens,
        logits_window *                  window = nullptr) {
    const std::size_t batch_size = std::max<std::size_t>(1, llama_n_batch(context));
    if (window != nullptr) {
        if (window->start >= tokens.size()) {
            throw std::invalid_argument("logits window start is outside the prompt");
        }
        const std::size_t rows = tokens.size() - window->start;
        if (window->vocab_size == 0 || rows > std::numeric_limits<std::size_t>::max()/window->vocab_size) {
            throw std::overflow_error("logits window dimensions overflow size_t");
        }
        window->values.clear();
        window->values.reserve(rows*window->vocab_size);
    }
    for (std::size_t offset = 0; offset < tokens.size(); offset += batch_size) {
        const std::size_t count = std::min(batch_size, tokens.size() - offset);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + offset), static_cast<int32_t>(count));
        std::vector<int8_t> output_flags;
        if (window != nullptr && offset + count > window->start) {
            output_flags.assign(count, 0);
            const std::size_t first = window->start > offset ? window->start - offset : 0;
            std::fill(output_flags.begin() + first, output_flags.end(), 1);
            batch.logits = output_flags.data();
        }
        if (llama_decode(context, batch) != 0) {
            return false;
        }
        if (!output_flags.empty()) {
            for (std::size_t i = 0; i < count; ++i) {
                if (!output_flags[i]) {
                    continue;
                }
                const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(i));
                if (logits == nullptr) {
                    throw std::runtime_error("requested prompt logits are unavailable");
                }
                window->values.insert(window->values.end(), logits, logits + window->vocab_size);
            }
        }
    }
    if (window != nullptr) {
        const std::size_t expected = (tokens.size() - window->start)*window->vocab_size;
        if (window->values.size() != expected) {
            throw std::runtime_error("logits window row count is incomplete");
        }
    }
    return true;
}

void write_logits_window(
        const std::filesystem::path & path,
        const logits_window &         window,
        std::size_t                   prompt_tokens,
        std::uint64_t                 prompt_token_hash) {
    if (path.empty()) {
        return;
    }
    const std::filesystem::path metadata_path = path.string() + ".json";
    if (std::filesystem::exists(path) || std::filesystem::exists(metadata_path)) {
        throw std::runtime_error("logits window output or metadata already exists: " + path.string());
    }
    std::ofstream output(path, std::ios::binary | std::ios::out);
    output.write(
        reinterpret_cast<const char *>(window.values.data()),
        static_cast<std::streamsize>(window.values.size()*sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write logits window: " + path.string());
    }
    output.close();

    std::ofstream metadata(metadata_path, std::ios::out);
    metadata << "{\n"
             << "  \"schema\": \"llama-wackmall-logits-window-v1\",\n"
             << "  \"data\": " << std::quoted(path.filename().string()) << ",\n"
             << "  \"dtype\": \"little-endian-f32\",\n"
             << "  \"prompt_token_count\": " << prompt_tokens << ",\n"
             << "  \"prompt_token_hash_fnv1a64\": \"" << std::hex << std::setw(16)
             << std::setfill('0') << prompt_token_hash << std::dec << "\",\n"
             << "  \"start_token\": " << window.start << ",\n"
             << "  \"rows\": " << prompt_tokens - window.start << ",\n"
             << "  \"vocab_size\": " << window.vocab_size << "\n"
             << "}\n";
    if (!metadata) {
        throw std::runtime_error("failed to write logits window metadata: " + metadata_path.string());
    }
}

void write_final_logits(
        const std::filesystem::path & path,
        const llama_vocab *          vocab,
        llama_context *              context) {
    if (path.empty()) {
        return;
    }
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("final logits output already exists: " + path.string());
    }
    const float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
        throw std::runtime_error("final prompt logits are unavailable");
    }
    const std::size_t count = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    std::ofstream output(path, std::ios::binary | std::ios::out);
    output.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(count * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write final logits: " + path.string());
    }
}

struct backend_guard {
    ~backend_guard() {
        llama_backend_free();
    }
};

} // namespace

int main(int argc, char ** argv) {
    try {
        capture_options capture_opts;
        std::vector<char *> forwarded = extract_capture_options(argc, argv, capture_opts);
        if (capture_opts.capture_values && capture_opts.output_dir.empty()) {
            throw std::invalid_argument("--capture-values requires --capture-dir");
        }
        if (!capture_opts.final_logits_path.empty() && std::filesystem::exists(capture_opts.final_logits_path)) {
            throw std::invalid_argument("final logits output already exists: " + capture_opts.final_logits_path.string());
        }
        if (!capture_opts.logits_window_path.empty()) {
            const std::filesystem::path metadata_path = capture_opts.logits_window_path.string() + ".json";
            if (std::filesystem::exists(capture_opts.logits_window_path) || std::filesystem::exists(metadata_path)) {
                throw std::invalid_argument("logits window output or metadata already exists: " + capture_opts.logits_window_path.string());
            }
        }
        if (capture_opts.capture_values) {
            if (setenv("LLAMA_TURBOQUANT_CAPTURE_VALUES", "1", 1) != 0) {
                throw std::runtime_error("failed to enable materialized V capture");
            }
        } else {
            unsetenv("LLAMA_TURBOQUANT_CAPTURE_VALUES");
        }

        common_params params;
        params.n_ctx = 4096;
        params.n_batch = 128;
        params.n_ubatch = 128;
        params.n_predict = 8;
        params.warmup = false;

        common_init();
        if (!common_params_parse(static_cast<int>(forwarded.size()), forwarded.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
            return 1;
        }
        if (params.prompt.empty()) {
            throw std::invalid_argument("a prompt from -p or -f is required");
        }
        if (params.n_predict < 0) {
            throw std::invalid_argument("capture requires a finite non-negative -n value");
        }
        const bool speculative_disabled =
            params.speculative.types == std::vector<common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE };
        if ((!capture_opts.attention_replay_dir.empty() || !capture_opts.attention_live_mask_dir.empty()) &&
            !speculative_disabled) {
            throw std::invalid_argument("attention replay and live-mask diagnostics require speculative decoding to be disabled");
        }

        const std::size_t prompt_max_tokens = capture_opts.prompt_max_tokens;
        const std::filesystem::path final_logits_path = capture_opts.final_logits_path;
        const std::filesystem::path logits_window_path = capture_opts.logits_window_path;
        const std::size_t logits_window_start = capture_opts.logits_window_start;
        const std::filesystem::path attention_replay_dir = capture_opts.attention_replay_dir;
        const std::filesystem::path attention_live_mask_dir = capture_opts.attention_live_mask_dir;
        const bool attention_live_mask_continue = capture_opts.attention_live_mask_continue;
        capture_state capture(std::move(capture_opts));
        attention_replay_state replay(attention_replay_dir);
        attention_live_mask_state live_mask(attention_live_mask_dir, attention_live_mask_continue);
        if (live_mask.enabled()) {
            const std::string layer = std::to_string(live_mask.layer());
            if (setenv("LLAMA_TURBOQUANT_LIVE_MASK_LAYER", layer.c_str(), 1) != 0) {
                throw std::runtime_error("failed to enable the diagnostic attention live mask");
            }
        } else {
            unsetenv("LLAMA_TURBOQUANT_LIVE_MASK_LAYER");
        }
        tool_callback_state callback_state = { &capture, &replay, &live_mask };
        if (capture.enabled() || replay.enabled() || live_mask.enabled()) {
            params.cb_eval = capture_callback;
            params.cb_eval_user_data = &callback_state;
        }

        llama_backend_init();
        const backend_guard backend;
        llama_numa_init(params.numa);

        std::vector<llama_token> generated;
        {
            common_init_result_ptr init = common_init_from_params(params);
            if (!init || !init->model() || !init->context()) {
                throw std::runtime_error("failed to initialize model and context");
            }

            llama_model * model = init->model();
            llama_context * context = init->context();
            const llama_vocab * vocab = llama_model_get_vocab(model);
            const bool add_special = llama_vocab_get_add_bos(vocab);
            std::vector<llama_token> prompt_tokens = common_tokenize(context, params.prompt, add_special, true);
            if (prompt_tokens.empty()) {
                throw std::runtime_error("prompt tokenized to an empty sequence");
            }
            if (prompt_max_tokens > 0 && prompt_tokens.size() > prompt_max_tokens) {
                prompt_tokens.resize(prompt_max_tokens);
            }
            const std::uint64_t prompt_token_hash = hash_tokens(prompt_tokens);
            replay.validate(
                params.model.path,
                prompt_token_hash,
                prompt_tokens.size(),
                llama_n_ctx(context),
                llama_n_batch(context),
                llama_n_ubatch(context),
                params.cache_type_k,
                params.cache_type_v);
            live_mask.validate(
                params.model.path,
                prompt_token_hash,
                prompt_tokens.size(),
                llama_n_ctx(context),
                llama_n_batch(context),
                llama_n_ubatch(context),
                params.cache_type_k,
                params.cache_type_v);
            if (prompt_tokens.size() + static_cast<std::size_t>(params.n_predict) > llama_n_ctx(context)) {
                throw std::runtime_error("prompt plus generated tokens exceeds the configured context");
            }
            logits_window prompt_logits;
            logits_window * prompt_logits_ptr = nullptr;
            if (!logits_window_path.empty()) {
                prompt_logits.start = logits_window_start;
                prompt_logits.vocab_size = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
                prompt_logits_ptr = &prompt_logits;
            }
            const auto prompt_begin = std::chrono::steady_clock::now();
            if (!decode_tokens(context, prompt_tokens, prompt_logits_ptr)) {
                const std::string error = !capture.error().empty() ? capture.error() :
                    (!replay.error().empty() ? replay.error() : live_mask.error());
                throw std::runtime_error(error.empty() ? "prompt decode failed" : error);
            }
            const auto prompt_end = std::chrono::steady_clock::now();
            replay.finish();
            live_mask.finish();
            write_logits_window(
                logits_window_path, prompt_logits, prompt_tokens.size(), prompt_token_hash);
            write_final_logits(final_logits_path, vocab, context);

            llama_sampler * sampler = llama_sampler_init_greedy();
            if (sampler == nullptr) {
                throw std::runtime_error("failed to create greedy sampler");
            }
            const auto decode_begin = std::chrono::steady_clock::now();
            std::size_t generation_decodes = 0;
            for (int i = 0; i < params.n_predict; ++i) {
                const llama_token token = llama_sampler_sample(sampler, context, -1);
                llama_sampler_accept(sampler, token);
                generated.push_back(token);
                if (llama_vocab_is_eog(vocab, token)) {
                    break;
                }
                const std::vector<llama_token> one = { token };
                if (!decode_tokens(context, one)) {
                    llama_sampler_free(sampler);
                    const std::string error = !capture.error().empty() ? capture.error() :
                        (!replay.error().empty() ? replay.error() : live_mask.error());
                    throw std::runtime_error(error.empty() ? "generation decode failed" : error);
                }
                ++generation_decodes;
            }
            const auto decode_end = std::chrono::steady_clock::now();
            llama_sampler_free(sampler);
            live_mask.finish_continuation(generation_decodes);

            const double prompt_ms = std::chrono::duration<double, std::milli>(prompt_end - prompt_begin).count();
            const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_begin).count();
            const double prompt_tps = prompt_ms > 0.0 ? 1000.0*prompt_tokens.size()/prompt_ms : 0.0;
            const double decode_tps = decode_ms > 0.0 ? 1000.0*generated.size()/decode_ms : 0.0;
            const std::string generated_text = common_detokenize(vocab, generated, false);

            const std::uint64_t token_hash = hash_tokens(generated);
            capture.finish(
                params.model.path,
                generated,
                token_hash,
                prompt_token_hash,
                prompt_tokens.size(),
                llama_n_ctx(context),
                llama_n_batch(context),
                llama_n_ubatch(context),
                params.cache_type_k,
                params.cache_type_v);
            std::cout << "{\"capture_enabled\":" << (capture.enabled() ? "true" : "false")
                      << ",\"attention_replay_enabled\":" << (replay.enabled() ? "true" : "false")
                      << ",\"attention_replay_records\":" << replay.applied()
                      << ",\"attention_live_mask_enabled\":" << (live_mask.enabled() ? "true" : "false")
                      << ",\"attention_live_mask_records\":" << live_mask.applied()
                      << ",\"attention_live_mask_continuation_records\":" << live_mask.continuation_applied()
                      << ",\"capture_events\":" << capture.event_count()
                      << ",\"capture_graphs\":" << capture.graph_count()
                      << ",\"prompt_tokens\":" << prompt_tokens.size()
                      << ",\"prompt_token_hash_fnv1a64\":\""
                      << std::hex << std::setw(16) << std::setfill('0') << prompt_token_hash << std::dec << "\""
                      << ",\"prompt_ms\":" << std::fixed << std::setprecision(3) << prompt_ms
                      << ",\"prompt_tps\":" << prompt_tps
                      << ",\"decode_tokens\":" << generated.size()
                      << ",\"decode_ms\":" << decode_ms
                      << ",\"decode_tps\":" << decode_tps
                      << ",\"generated_token_hash_fnv1a64\":\""
                      << std::hex << std::setw(16) << std::setfill('0') << token_hash << std::dec
                      << "\",\"generated_text\":\"" << json_escape(generated_text)
                      << "\",\"generated_tokens\":[";
            for (std::size_t i = 0; i < generated.size(); ++i) {
                std::cout << (i == 0 ? "" : ",") << generated[i];
            }
            std::cout << "]}\n";
        }
        return 0;
    } catch (const std::exception & exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
