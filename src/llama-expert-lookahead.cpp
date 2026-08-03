#include "llama-expert-lookahead.h"

#include "llama-expert-lookahead-metrics.h"
#include "llama-expert-tier.h"
#include "llama-graph.h"
#include "llama-model.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define LOOKAHEAD_LOG(...) fprintf(stderr, __VA_ARGS__)

namespace llama_expert_lookahead {

struct config {
    bool loaded = false;
    bool requested = false;
    bool valid = false;
    std::string json_path;
    prediction_point point = prediction_point::post_attn;
    norm_source norm = norm_source::target;
    int distance = 1;
    int requested_top_m = 16;
};

struct trace_record {
    uint64_t token_index = 0;
    int source_layer = -1;
    int target_layer = -1;
    std::vector<int32_t> actual;
    std::vector<float> weights;
    std::vector<uint8_t> actual_fixed;
    std::vector<int32_t> predicted;
    std::vector<uint8_t> predicted_fixed;
};

struct pending_item {
    int source_layer = -1;
    int target_layer = -1;
    size_t predicted_offset = 0;
    size_t predicted_count = 0;
    size_t actual_offset = 0;
    size_t actual_count = 0;
    size_t weights_offset = 0;
    size_t weights_count = 0;
    ggml_tensor * predicted_tensor = nullptr;
    ggml_tensor * actual_tensor = nullptr;
    ggml_tensor * weights_tensor = nullptr;
    ggml_backend_t predicted_backend = nullptr;
    ggml_backend_t actual_backend = nullptr;
    ggml_backend_t weights_backend = nullptr;
    std::vector<uint8_t> fixed_mask;
};

struct pending_graph {
    bool active = false;
    std::vector<pending_item> items;
    uint64_t submit_us = 0;
    uint64_t bytes = 0;
};

struct aggregate {
    uint64_t samples = 0;
    uint64_t top1_hits = 0;
    uint64_t actual_count = 0;
    uint64_t predicted_count = 0;
    uint64_t intersection_count = 0;
    uint64_t cold_actual_count = 0;
    uint64_t cold_intersection_count = 0;
    uint64_t predicted_fixed_count = 0;
    uint64_t useful_predicted_cold_count = 0;
    uint64_t false_positive_cold_count = 0;
    uint64_t missed_cold_count = 0;
    uint64_t estimated_h2d_bytes = 0;
    uint64_t estimated_useful_h2d_bytes = 0;
    double actual_weight = 0.0;
    double covered_weight = 0.0;
    double jaccard_sum = 0.0;
};

static std::mutex g_mutex;
static config g_config;
static bool g_initialized = false;
static bool g_runtime_enabled = false;
static bool g_request_scoped = false;
static bool g_request_active = false;
static bool g_shutdown_registered = false;
static int g_mtp_n = 0;
static int g_n_layer = 0;
static int g_n_expert = 0;
static int g_n_expert_used = 0;
static int g_effective_top_m = 0;
static std::vector<uint64_t> g_expert_bytes_by_layer;
static uint32_t g_context = 0;
static uint64_t g_request_id = 0;
static uint64_t g_token_index = 0;
static uint64_t g_d2h_bytes = 0;
static uint64_t g_d2h_submit_us = 0;
static bool g_host_pinned = false;
static std::string g_model_desc;
static std::string g_predictor_backend;
static std::string g_actual_ids_backend;
static std::string g_actual_weights_backend;
static std::vector<trace_record> g_records;
static pending_graph g_pending;
static ggml_backend_buffer_t g_host_buffer = nullptr;

static bool env_enabled(const char * name) {
    const char * value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

static bool parse_positive_int(const char * text, int & value) {
    if (!text || !text[0]) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return false;
    }
    value = (int) parsed;
    return true;
}

static void load_config_locked() {
    if (g_config.loaded) {
        return;
    }
    g_config.loaded = true;
    g_config.requested = env_enabled("LLAMA_EXPERT_LOOKAHEAD_TRACE");
    if (!g_config.requested) {
        return;
    }

    const char * json_path = getenv("LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON");
    if (!json_path || !json_path[0] || strcmp(json_path, "0") == 0) {
        LOOKAHEAD_LOG("expert_lookahead: trace requested without LLAMA_EXPERT_LOOKAHEAD_TRACE_JSON; trace disabled\n");
        return;
    }
    g_config.json_path = json_path;

    if (const char * value = getenv("LLAMA_EXPERT_LOOKAHEAD_DISTANCE")) {
        if (!parse_positive_int(value, g_config.distance)) {
            LOOKAHEAD_LOG("expert_lookahead: invalid LLAMA_EXPERT_LOOKAHEAD_DISTANCE='%s'; trace disabled\n", value);
            return;
        }
    }
    if (const char * value = getenv("LLAMA_EXPERT_LOOKAHEAD_TOP_M")) {
        if (!parse_positive_int(value, g_config.requested_top_m)) {
            LOOKAHEAD_LOG("expert_lookahead: invalid LLAMA_EXPERT_LOOKAHEAD_TOP_M='%s'; trace disabled\n", value);
            return;
        }
    }
    if (const char * value = getenv("LLAMA_EXPERT_LOOKAHEAD_POINT")) {
        if (strcmp(value, "post-attn") == 0) {
            g_config.point = prediction_point::post_attn;
        } else if (strcmp(value, "post-moe") == 0) {
            g_config.point = prediction_point::post_moe;
        } else {
            LOOKAHEAD_LOG("expert_lookahead: invalid LLAMA_EXPERT_LOOKAHEAD_POINT='%s'; expected post-attn or post-moe\n", value);
            return;
        }
    }
    if (const char * value = getenv("LLAMA_EXPERT_LOOKAHEAD_NORM")) {
        if (strcmp(value, "target") == 0) {
            g_config.norm = norm_source::target;
        } else if (strcmp(value, "source") == 0) {
            g_config.norm = norm_source::source;
        } else {
            LOOKAHEAD_LOG("expert_lookahead: invalid LLAMA_EXPERT_LOOKAHEAD_NORM='%s'; expected target or source\n", value);
            return;
        }
    }
    g_config.valid = true;
}

static bool config_graph_enabled_locked(uint32_t n_tokens, uint32_t n_seqs, bool mtp_graph) {
    load_config_locked();
    if (!g_config.valid || g_mtp_n > 0 || mtp_graph) {
        return false;
    }
    return n_tokens == 1 && n_seqs == 1;
}

bool graph_enabled(uint32_t n_tokens, uint32_t n_seqs, bool mtp_graph) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initialized && !g_runtime_enabled) {
        return false;
    }
    return config_graph_enabled_locked(n_tokens, n_seqs, mtp_graph);
}

prediction_point point() {
    std::lock_guard<std::mutex> lock(g_mutex);
    load_config_locked();
    return g_config.point;
}

norm_source norm() {
    std::lock_guard<std::mutex> lock(g_mutex);
    load_config_locked();
    return g_config.norm;
}

int distance() {
    std::lock_guard<std::mutex> lock(g_mutex);
    load_config_locked();
    return g_config.distance;
}

int top_m(int n_expert) {
    std::lock_guard<std::mutex> lock(g_mutex);
    load_config_locked();
    return std::max(0, std::min(g_config.requested_top_m, n_expert));
}

void configure_mtp(int mtp_n) {
    if (mtp_n <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mtp_n = std::max(g_mtp_n, mtp_n);
    if (g_runtime_enabled) {
        g_runtime_enabled = false;
        LOOKAHEAD_LOG("expert_lookahead: trace disabled for MTP-%d; Phase 1 is no-MTP only\n", g_mtp_n);
    }
}

void configure_request_scoped(bool enabled_value) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_request_scoped = enabled_value;
}

static void reset_request_locked() {
    g_token_index = 0;
    g_d2h_bytes = 0;
    g_d2h_submit_us = 0;
    g_context = 0;
    g_predictor_backend.clear();
    g_actual_ids_backend.clear();
    g_actual_weights_backend.clear();
    g_records.clear();
    g_pending = {};
}

static void write_json_string(FILE * file, const std::string & value) {
    fputc('"', file);
    for (unsigned char c : value) {
        switch (c) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\b': fputs("\\b", file); break;
            case '\f': fputs("\\f", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (c < 0x20) {
                    fprintf(file, "\\u%04x", c);
                } else {
                    fputc(c, file);
                }
                break;
        }
    }
    fputc('"', file);
}

template<typename T>
static void write_number_array(FILE * file, const std::vector<T> & values) {
    fputc('[', file);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            fputc(',', file);
        }
        fprintf(file, "%lld", (long long) values[i]);
    }
    fputc(']', file);
}

static void write_float_array(FILE * file, const std::vector<float> & values) {
    fputc('[', file);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            fputc(',', file);
        }
        fprintf(file, "%.9g", (double) values[i]);
    }
    fputc(']', file);
}

static void write_bool_array(FILE * file, const std::vector<uint8_t> & values) {
    fputc('[', file);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            fputc(',', file);
        }
        fputs(values[i] ? "true" : "false", file);
    }
    fputc(']', file);
}

static void add_metric(aggregate & total, const sample_metrics & sample, uint64_t expert_bytes) {
    total.samples++;
    total.top1_hits += sample.top1_hit ? 1 : 0;
    total.actual_count += sample.actual_count;
    total.predicted_count += sample.predicted_count;
    total.intersection_count += sample.intersection_count;
    total.cold_actual_count += sample.cold_actual_count;
    total.cold_intersection_count += sample.cold_intersection_count;
    total.predicted_fixed_count += sample.predicted_fixed_count;
    total.useful_predicted_cold_count += sample.useful_predicted_cold_count;
    total.false_positive_cold_count += sample.false_positive_cold_count;
    total.missed_cold_count += sample.missed_cold_count;
    total.estimated_h2d_bytes += (sample.useful_predicted_cold_count + sample.false_positive_cold_count)*expert_bytes;
    total.estimated_useful_h2d_bytes += sample.useful_predicted_cold_count*expert_bytes;
    total.actual_weight += sample.actual_weight;
    total.covered_weight += sample.covered_weight;
    if (sample.union_count > 0) {
        total.jaccard_sum += (double) sample.intersection_count / (double) sample.union_count;
    }
}

static void write_ratio(FILE * file, double numerator, double denominator) {
    if (denominator <= 0.0) {
        fputs("null", file);
    } else {
        fprintf(file, "%.9g", numerator / denominator);
    }
}

static void write_aggregate(FILE * file, const aggregate & value) {
    fprintf(file,
            "{\"samples\":%llu,\"top1_hits\":%llu,\"top1_rate\":",
            (unsigned long long) value.samples,
            (unsigned long long) value.top1_hits);
    write_ratio(file, (double) value.top1_hits, (double) value.samples);
    fprintf(file,
            ",\"actual_count\":%llu,\"predicted_count\":%llu,\"intersection_count\":%llu,\"recall\":",
            (unsigned long long) value.actual_count,
            (unsigned long long) value.predicted_count,
            (unsigned long long) value.intersection_count);
    write_ratio(file, (double) value.intersection_count, (double) value.actual_count);
    fputs(",\"jaccard_mean\":", file);
    write_ratio(file, value.jaccard_sum, (double) value.samples);
    fputs(",\"weighted_coverage\":", file);
    write_ratio(file, value.covered_weight, value.actual_weight);
    fprintf(file,
            ",\"cold_actual_count\":%llu,\"cold_intersection_count\":%llu,\"cold_recall\":",
            (unsigned long long) value.cold_actual_count,
            (unsigned long long) value.cold_intersection_count);
    write_ratio(file, (double) value.cold_intersection_count, (double) value.cold_actual_count);
    fprintf(file,
            ",\"predicted_fixed_count\":%llu,\"useful_predicted_cold_count\":%llu,"
            "\"false_positive_cold_count\":%llu,\"missed_cold_count\":%llu,"
            "\"estimated_h2d_bytes\":%llu,\"estimated_useful_h2d_bytes\":%llu,\"estimated_h2d_ms\":null,"
            "\"estimated_timely_fraction\":null}",
            (unsigned long long) value.predicted_fixed_count,
            (unsigned long long) value.useful_predicted_cold_count,
            (unsigned long long) value.false_positive_cold_count,
            (unsigned long long) value.missed_cold_count,
            (unsigned long long) value.estimated_h2d_bytes,
            (unsigned long long) value.estimated_useful_h2d_bytes);
}

static std::vector<uint8_t> record_fixed_mask(const trace_record & record) {
    std::vector<uint8_t> fixed((size_t) std::max(0, g_n_expert), 0);
    for (size_t i = 0; i < record.actual.size() && i < record.actual_fixed.size(); ++i) {
        const int expert = record.actual[i];
        if (expert >= 0 && expert < g_n_expert && record.actual_fixed[i]) {
            fixed[(size_t) expert] = 1;
        }
    }
    for (size_t i = 0; i < record.predicted.size() && i < record.predicted_fixed.size(); ++i) {
        const int expert = record.predicted[i];
        if (expert >= 0 && expert < g_n_expert && record.predicted_fixed[i]) {
            fixed[(size_t) expert] = 1;
        }
    }
    return fixed;
}

static std::vector<int> metric_prefixes_locked() {
    std::set<int> unique;
    for (int prefix : { 8, 12, 16, g_effective_top_m }) {
        if (prefix > 0 && prefix <= g_effective_top_m) {
            unique.insert(prefix);
        }
    }
    return std::vector<int>(unique.begin(), unique.end());
}

static std::string output_path_locked() {
    std::string path = g_config.json_path;
    const std::string marker = "%r";
    size_t pos = 0;
    while ((pos = path.find(marker, pos)) != std::string::npos) {
        const std::string replacement = std::to_string(g_request_id);
        path.replace(pos, marker.size(), replacement);
        pos += replacement.size();
    }
    return path;
}

static bool write_request_locked() {
    if (!g_runtime_enabled || g_config.json_path.empty()) {
        return false;
    }

    const std::string path = output_path_locked();
    const std::string temporary = path + ".tmp." + std::to_string(ggml_time_us());
    FILE * file = fopen(temporary.c_str(), "w");
    if (!file) {
        LOOKAHEAD_LOG("expert_lookahead: cannot open trace '%s': %s\n", temporary.c_str(), strerror(errno));
        return false;
    }

    const auto prefixes = metric_prefixes_locked();
    std::map<int, std::map<int, aggregate>> layer_metrics;
    std::map<int, aggregate> overall_metrics;
    for (const auto & record : g_records) {
        const auto fixed = record_fixed_mask(record);
        const uint64_t expert_bytes = record.target_layer >= 0 && (size_t) record.target_layer < g_expert_bytes_by_layer.size() ?
                g_expert_bytes_by_layer[(size_t) record.target_layer] : 0;
        for (int prefix : prefixes) {
            const auto metric = evaluate_sample(record.actual, record.weights, record.predicted,
                    fixed, (size_t) prefix, g_n_expert);
            add_metric(overall_metrics[prefix], metric, expert_bytes);
            add_metric(layer_metrics[record.target_layer][prefix], metric, expert_bytes);
        }
    }

    fputs("{\n  \"schema\":\"llama-wackmall-router-lookahead-v1\",\n  \"request_id\":", file);
    fprintf(file, "%llu,\n  \"model\":", (unsigned long long) g_request_id);
    write_json_string(file, g_model_desc);
    fprintf(file,
            ",\n  \"context\":%u,\n  \"mtp_n\":%d,\n"
            "  \"config\":{\"distance\":%d,\"top_m_requested\":%d,\"top_m_effective\":%d,"
            "\"point\":\"%s\",\"norm\":\"%s\",\"n_layer\":%d,\"n_expert\":%d,\"actual_top_k\":%d},\n",
            g_context, g_mtp_n, g_config.distance, g_config.requested_top_m, g_effective_top_m,
            g_config.point == prediction_point::post_attn ? "post-attn" : "post-moe",
            g_config.norm == norm_source::target ? "target" : "source",
            g_n_layer, g_n_expert, g_n_expert_used);
    fprintf(file,
            "  \"status\":{\"routing_authority\":\"actual-only\",\"productive_prefetch\":false,"
            "\"pinned_host_readback\":%s,\"predictor_gpu_ms\":null,\"d2h_copy_ms\":null,"
            "\"available_lead_ms\":null,\"predictor_ids_backend\":",
            g_host_pinned ? "true" : "false");
    write_json_string(file, g_predictor_backend);
    fputs(",\"actual_ids_backend\":", file);
    write_json_string(file, g_actual_ids_backend);
    fputs(",\"actual_weights_backend\":", file);
    write_json_string(file, g_actual_weights_backend);
    fputs(",\"timing_note\":", file);
    write_json_string(file, "Phase 1 defers readback until graph completion; mid-graph timing requires separate event instrumentation");
    fputs("},\n  \"readback\":{\"bytes\":", file);
    fprintf(file, "%llu,\"submit_us\":%llu},\n  \"overall\":{",
            (unsigned long long) g_d2h_bytes,
            (unsigned long long) g_d2h_submit_us);
    bool first = true;
    for (const auto & entry : overall_metrics) {
        if (!first) {
            fputc(',', file);
        }
        fprintf(file, "\"top_%d\":", entry.first);
        write_aggregate(file, entry.second);
        first = false;
    }
    fputs("},\n  \"layers\":[", file);
    first = true;
    for (const auto & layer : layer_metrics) {
        if (!first) {
            fputc(',', file);
        }
        const uint64_t expert_bytes = layer.first >= 0 && (size_t) layer.first < g_expert_bytes_by_layer.size() ?
                g_expert_bytes_by_layer[(size_t) layer.first] : 0;
        fprintf(file, "{\"layer\":%d,\"expert_bytes\":%llu,\"metrics\":{", layer.first,
                (unsigned long long) expert_bytes);
        bool first_metric = true;
        for (const auto & entry : layer.second) {
            if (!first_metric) {
                fputc(',', file);
            }
            fprintf(file, "\"top_%d\":", entry.first);
            write_aggregate(file, entry.second);
            first_metric = false;
        }
        fputs("}}", file);
        first = false;
    }
    fputs("],\n  \"records\":[", file);
    for (size_t i = 0; i < g_records.size(); ++i) {
        const auto & record = g_records[i];
        if (i > 0) {
            fputc(',', file);
        }
        fprintf(file,
                "{\"token_index\":%llu,\"source_layer\":%d,\"target_layer\":%d,\"actual\":",
                (unsigned long long) record.token_index, record.source_layer, record.target_layer);
        write_number_array(file, record.actual);
        fputs(",\"actual_weights\":", file);
        write_float_array(file, record.weights);
        fputs(",\"actual_fixed\":", file);
        write_bool_array(file, record.actual_fixed);
        fputs(",\"predicted\":", file);
        write_number_array(file, record.predicted);
        fputs(",\"predicted_fixed\":", file);
        write_bool_array(file, record.predicted_fixed);
        fputc('}', file);
    }
    fputs("]\n}\n", file);

    const bool write_ok = fflush(file) == 0 && !ferror(file);
    const int close_result = fclose(file);
    if (!write_ok || close_result != 0) {
        const int saved_errno = errno;
        std::remove(temporary.c_str());
        LOOKAHEAD_LOG("expert_lookahead: failed to finish trace '%s': %s\n", temporary.c_str(), strerror(saved_errno));
        return false;
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        const int saved_errno = errno;
        std::remove(temporary.c_str());
        LOOKAHEAD_LOG("expert_lookahead: cannot publish trace '%s': %s\n", path.c_str(), strerror(saved_errno));
        return false;
    }

    LOOKAHEAD_LOG("expert_lookahead: wrote request %llu trace with %zu layer-token records to %s\n",
            (unsigned long long) g_request_id, g_records.size(), path.c_str());
    return true;
}

static void shutdown_trace() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_request_active && !g_request_scoped) {
        write_request_locked();
        g_request_active = false;
    }
    if (g_host_buffer) {
        ggml_backend_buffer_free(g_host_buffer);
        g_host_buffer = nullptr;
    }
}

void init(const llama_model & model) {
    std::lock_guard<std::mutex> lock(g_mutex);
    load_config_locked();
    if (g_initialized) {
        return;
    }
    g_initialized = true;
    if (!g_config.valid) {
        return;
    }
    if (g_mtp_n > 0) {
        LOOKAHEAD_LOG("expert_lookahead: trace disabled for MTP-%d; Phase 1 is no-MTP only\n", g_mtp_n);
        return;
    }
    const char * adapt = getenv("LLAMA_EXPERT_ADAPT");
    if (!adapt || strcmp(adapt, "0") != 0) {
        LOOKAHEAD_LOG("expert_lookahead: trace requires LLAMA_EXPERT_ADAPT=0 for a stable fixed-tier snapshot\n");
        return;
    }
    const char * warm = getenv("LLAMA_EXPERT_WARM_SLOTS");
    if (warm && strcmp(warm, "0") != 0) {
        LOOKAHEAD_LOG("expert_lookahead: trace requires LLAMA_EXPERT_WARM_SLOTS=0\n");
        return;
    }
    if (env_enabled("LLAMA_EXPERT_STATIC_NO_SYNC")) {
        LOOKAHEAD_LOG("expert_lookahead: trace is incompatible with LLAMA_EXPERT_STATIC_NO_SYNC=1\n");
        return;
    }

    g_n_layer = (int) model.hparams.n_layer();
    g_n_expert = (int) model.hparams.n_expert;
    g_n_expert_used = (int) model.hparams.n_expert_used;
    if (g_n_layer <= 0 || g_n_expert <= 0 || g_n_expert_used <= 0 || g_config.distance >= g_n_layer) {
        LOOKAHEAD_LOG("expert_lookahead: model dimensions are incompatible with distance %d; trace disabled\n",
                g_config.distance);
        return;
    }
    g_effective_top_m = std::min(g_config.requested_top_m, g_n_expert);
    g_expert_bytes_by_layer.assign((size_t) g_n_layer, 0);
    for (int il = 0; il < g_n_layer; ++il) {
        const llama_layer & layer = model.layers[il];
        for (ggml_tensor * weight : { layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }) {
            if (weight) {
                g_expert_bytes_by_layer[(size_t) il] += ggml_nbytes(weight)/(uint64_t) g_n_expert;
            }
        }
    }
    g_model_desc = model.desc();
    g_runtime_enabled = true;
    if (!g_shutdown_registered) {
        atexit(shutdown_trace);
        g_shutdown_registered = true;
    }
    LOOKAHEAD_LOG("expert_lookahead: Phase 1 trace enabled: point=%s norm=%s distance=%d top_m=%d actual_top_k=%d; productive prefetch is off\n",
            g_config.point == prediction_point::post_attn ? "post-attn" : "post-moe",
            g_config.norm == norm_source::target ? "target" : "source",
            g_config.distance, g_effective_top_m, g_n_expert_used);
}

bool enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_runtime_enabled;
}

void request_begin() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_runtime_enabled) {
        return;
    }
    if (g_request_active) {
        LOOKAHEAD_LOG("expert_lookahead: overlapping requests are unsupported in Phase 1; trace collection suspended\n");
        g_runtime_enabled = false;
        return;
    }
    g_request_id++;
    reset_request_locked();
    g_request_active = true;
}

void request_end() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_runtime_enabled || !g_request_active) {
        return;
    }
    if (g_pending.active) {
        LOOKAHEAD_LOG("expert_lookahead: request ended with a pending readback; trace not published\n");
        g_pending = {};
    } else {
        write_request_locked();
    }
    g_request_active = false;
    g_records.clear();
}

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static void merge_backend_name(std::string & current, ggml_backend_t backend) {
    const std::string name = backend ? ggml_backend_name(backend) : "unknown";
    if (current.empty()) {
        current = name;
    } else if (current != name) {
        current = "mixed";
    }
}

static void consider_host_backend(ggml_backend_t candidate, ggml_backend_t & selected) {
    if (!candidate) {
        return;
    }
    if (!selected) {
        selected = candidate;
        return;
    }
    const auto selected_type = ggml_backend_dev_type(ggml_backend_get_device(selected));
    const auto candidate_type = ggml_backend_dev_type(ggml_backend_get_device(candidate));
    if (selected_type == GGML_BACKEND_DEVICE_TYPE_CPU && candidate_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
        selected = candidate;
    }
}

static bool ensure_host_buffer_locked(ggml_backend_t backend, size_t size) {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    if (backend) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft) {
            buft = host_buft;
        }
    }

    if (g_host_buffer && ggml_backend_buffer_get_type(g_host_buffer) == buft &&
            ggml_backend_buffer_get_size(g_host_buffer) >= size) {
        g_host_pinned = buft != ggml_backend_cpu_buffer_type();
        return true;
    }
    if (g_host_buffer) {
        ggml_backend_buffer_free(g_host_buffer);
        g_host_buffer = nullptr;
    }
    g_host_buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!g_host_buffer && buft != ggml_backend_cpu_buffer_type()) {
        buft = ggml_backend_cpu_buffer_type();
        g_host_buffer = ggml_backend_buft_alloc_buffer(buft, size);
    }
    g_host_pinned = g_host_buffer && buft != ggml_backend_cpu_buffer_type();
    return g_host_buffer != nullptr;
}

void enqueue_graph(ggml_backend_sched_t sched, const llm_graph_result & result, uint32_t n_ctx) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_runtime_enabled) {
        return;
    }
    const auto & traces = result.get_lookahead_traces();
    if (traces.empty()) {
        return;
    }
    if (g_request_scoped && !g_request_active) {
        return;
    }
    if (!g_request_active) {
        g_request_id++;
        reset_request_locked();
        g_request_active = true;
    }
    if (g_pending.active) {
        LOOKAHEAD_LOG("expert_lookahead: internal error: previous graph readback is still pending\n");
        return;
    }

    size_t bytes = 0;
    uint64_t payload_bytes = 0;
    std::vector<pending_item> items;
    items.reserve(traces.size());
    ggml_backend_t first_backend = nullptr;
    for (const auto & trace : traces) {
        if (!trace.predicted_ids || !trace.actual_ids || !trace.actual_weights) {
            continue;
        }
        if (trace.predicted_ids->type != GGML_TYPE_I32 || trace.actual_ids->type != GGML_TYPE_I32 ||
                trace.actual_weights->type != GGML_TYPE_F32) {
            LOOKAHEAD_LOG("expert_lookahead: unsupported trace tensor types for target layer %d\n", trace.target_layer);
            continue;
        }
        pending_item item;
        item.source_layer = trace.source_layer;
        item.target_layer = trace.target_layer;
        item.predicted_count = ggml_nelements(trace.predicted_ids);
        item.actual_count = ggml_nelements(trace.actual_ids);
        item.weights_count = ggml_nelements(trace.actual_weights);
        if (item.actual_count != item.weights_count || item.predicted_count == 0 || item.actual_count == 0) {
            LOOKAHEAD_LOG("expert_lookahead: invalid trace shapes for target layer %d\n", trace.target_layer);
            continue;
        }
        item.predicted_offset = align_up(bytes, 64);
        bytes = item.predicted_offset + item.predicted_count*sizeof(int32_t);
        item.actual_offset = align_up(bytes, 64);
        bytes = item.actual_offset + item.actual_count*sizeof(int32_t);
        item.weights_offset = align_up(bytes, 64);
        bytes = item.weights_offset + item.weights_count*sizeof(float);
        item.predicted_tensor = trace.predicted_ids;
        item.actual_tensor = trace.actual_ids;
        item.weights_tensor = trace.actual_weights;
        item.predicted_backend = ggml_backend_sched_get_tensor_backend(sched, trace.predicted_ids);
        item.actual_backend = ggml_backend_sched_get_tensor_backend(sched, trace.actual_ids);
        item.weights_backend = ggml_backend_sched_get_tensor_backend(sched, trace.actual_weights);
        if (!item.predicted_backend || !item.actual_backend || !item.weights_backend) {
            LOOKAHEAD_LOG("expert_lookahead: missing tensor backend for target layer %d\n", trace.target_layer);
            continue;
        }
        item.fixed_mask = llama_expert_tier::fixed_expert_mask(trace.target_layer, g_n_expert);
        consider_host_backend(item.predicted_backend, first_backend);
        consider_host_backend(item.actual_backend, first_backend);
        consider_host_backend(item.weights_backend, first_backend);
        merge_backend_name(g_predictor_backend, item.predicted_backend);
        merge_backend_name(g_actual_ids_backend, item.actual_backend);
        merge_backend_name(g_actual_weights_backend, item.weights_backend);
        payload_bytes += item.predicted_count*sizeof(int32_t) +
                item.actual_count*sizeof(int32_t) + item.weights_count*sizeof(float);
        items.push_back(std::move(item));
    }
    if (items.empty() || !ensure_host_buffer_locked(first_backend, bytes)) {
        LOOKAHEAD_LOG("expert_lookahead: cannot allocate %zu bytes for trace readback\n", bytes);
        return;
    }

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(g_host_buffer);
    const int64_t started = ggml_time_us();
    for (const auto & item : items) {
        ggml_backend_tensor_get_async(item.predicted_backend, item.predicted_tensor,
                base + item.predicted_offset, 0, item.predicted_count*sizeof(int32_t));
        ggml_backend_tensor_get_async(item.actual_backend, item.actual_tensor,
                base + item.actual_offset, 0, item.actual_count*sizeof(int32_t));
        ggml_backend_tensor_get_async(item.weights_backend, item.weights_tensor,
                base + item.weights_offset, 0, item.weights_count*sizeof(float));
    }
    g_pending.active = true;
    g_pending.items = std::move(items);
    g_pending.submit_us = (uint64_t) std::max<int64_t>(0, ggml_time_us() - started);
    g_pending.bytes = payload_bytes;
    g_context = n_ctx;
}

void complete_graph() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_runtime_enabled || !g_pending.active || !g_host_buffer) {
        return;
    }
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(g_host_buffer);
    for (const auto & item : g_pending.items) {
        trace_record record;
        record.token_index = g_token_index;
        record.source_layer = item.source_layer;
        record.target_layer = item.target_layer;
        const int32_t * predicted = (const int32_t *) (base + item.predicted_offset);
        const int32_t * actual = (const int32_t *) (base + item.actual_offset);
        const float * weights = (const float *) (base + item.weights_offset);
        record.predicted.assign(predicted, predicted + item.predicted_count);
        record.actual.assign(actual, actual + item.actual_count);
        record.weights.assign(weights, weights + item.weights_count);
        record.predicted_fixed.reserve(record.predicted.size());
        for (int expert : record.predicted) {
            record.predicted_fixed.push_back(expert >= 0 && (size_t) expert < item.fixed_mask.size() && item.fixed_mask[(size_t) expert]);
        }
        record.actual_fixed.reserve(record.actual.size());
        for (int expert : record.actual) {
            record.actual_fixed.push_back(expert >= 0 && (size_t) expert < item.fixed_mask.size() && item.fixed_mask[(size_t) expert]);
        }
        g_records.push_back(std::move(record));
    }
    g_token_index++;
    g_d2h_bytes += g_pending.bytes;
    g_d2h_submit_us += g_pending.submit_us;
    g_pending = {};
}

} // namespace llama_expert_lookahead
