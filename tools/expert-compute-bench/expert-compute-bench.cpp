#include "llama.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(GGML_USE_CUDA)
extern "C" bool expert_compute_q4_k_avx2(
        const void * weights, const void * input, int n_embd, int rows,
        int iterations, float * output, double * kernel_ms);
extern "C" bool expert_compute_q4_k_avx2_f32(
        const void * weights, const float * input, int n_embd, int rows,
        int iterations, float * output, void * quantized, double * kernel_ms);
#endif

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
    std::string model_path;
    std::string json_path;
    int device = 0;
    int layer = 0;
    int expert = 0;
    int repeats = 30;
    int warmups = 5;
    int queued_iterations = 100;
};

struct tensor_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        ggml_backend_buffer_free(buffer);
    }
};

struct backend_deleter {
    void operator()(ggml_backend_t backend) const {
        ggml_backend_free(backend);
    }
};

struct context_deleter {
    void operator()(ggml_context * context) const {
        ggml_free(context);
    }
};

struct model_deleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

using buffer_ptr = std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, tensor_buffer_deleter>;
using backend_ptr = std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, backend_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using model_ptr = std::unique_ptr<llama_model, model_deleter>;

struct summary {
    double min = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
};

struct mode_result {
    std::string mode;
    size_t weight_bytes = 0;
    summary latency_ms;
    summary queued_ms;
    double checksum = 0.0;
};

struct comparison_result {
    std::string name;
    size_t elements = 0;
    size_t exact = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double kernel_ms = 0.0;
};

static uint64_t parse_u64(const std::string & text, const char * name) {
    if (text.empty() || text[0] == '-') {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

static int parse_int(const std::string & text, const char * name, int minimum) {
    const uint64_t value = parse_u64(text, name);
    if (value < (uint64_t) minimum || value > (uint64_t) std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("out-of-range ") + name + ": " + text);
    }
    return (int) value;
}

static void print_usage(const char * program) {
    std::cerr
        << "usage: " << program << " --model PATH [options]\n"
        << "  --layer N               base MoE layer (default 0)\n"
        << "  --expert N              expert ID (default 0)\n"
        << "  --device N              CUDA backend ordinal (default 0)\n"
        << "  --repeats N             independent samples (default 30)\n"
        << "  --warmups N             warm-up computes (default 5)\n"
        << "  --queued-iterations N   graphs per queued sample (default 100)\n"
        << "  --json PATH             output without overwriting an existing file\n";
}

static options parse_options(int argc, char ** argv) {
    options parsed;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto take_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[index];
        };
        if (argument == "--model") {
            parsed.model_path = take_value();
        } else if (argument == "--json") {
            parsed.json_path = take_value();
        } else if (argument == "--layer") {
            parsed.layer = parse_int(take_value(), "layer", 0);
        } else if (argument == "--expert") {
            parsed.expert = parse_int(take_value(), "expert", 0);
        } else if (argument == "--device") {
            parsed.device = parse_int(take_value(), "device", 0);
        } else if (argument == "--repeats") {
            parsed.repeats = parse_int(take_value(), "repeats", 1);
        } else if (argument == "--warmups") {
            parsed.warmups = parse_int(take_value(), "warmups", 0);
        } else if (argument == "--queued-iterations") {
            parsed.queued_iterations = parse_int(take_value(), "queued-iterations", 1);
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (parsed.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    return parsed;
}

static std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << (int) c << std::dec;
                } else {
                    output << c;
                }
        }
    }
    return output.str();
}

static summary summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("cannot summarize empty samples");
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double fraction) {
        const double position = fraction * (values.size() - 1);
        const size_t lower = (size_t) std::floor(position);
        const size_t upper = (size_t) std::ceil(position);
        const double weight = position - lower;
        return values[lower] * (1.0 - weight) + values[upper] * weight;
    };
    return {
        values.front(),
        percentile(0.50),
        percentile(0.95),
        std::accumulate(values.begin(), values.end(), 0.0) / values.size(),
    };
}

static ggml_backend_dev_t select_gpu_device(int ordinal) {
    int found = 0;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (found++ == ordinal) {
            return device;
        }
    }
    return nullptr;
}

static ggml_tensor * make_weight(ggml_context * context, const ggml_tensor * source, const char * name) {
    if (!source || source->ne[2] <= 0 || source->ne[3] != 1) {
        throw std::runtime_error(std::string("invalid source tensor for ") + name);
    }
    ggml_tensor * result = ggml_new_tensor_2d(context, source->type, source->ne[0], source->ne[1]);
    ggml_set_name(result, name);
    return result;
}

static void upload_expert(ggml_tensor * destination, const ggml_tensor * source, int expert) {
    if (expert < 0 || expert >= source->ne[2]) {
        throw std::runtime_error("expert ID is outside tensor range");
    }
    const size_t bytes = ggml_nbytes(destination);
    if (source->nb[2] != bytes) {
        throw std::runtime_error(std::string("expert slice is not contiguous in ") + source->name);
    }
    ggml_backend_tensor_set(destination, (const char *) source->data + source->nb[2] * expert, 0, bytes);
}

static void compute_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend graph compute failed with status " + std::to_string(status));
    }
}

static summary measure_latency(ggml_backend_t backend, ggml_cgraph * graph, int repeats, int warmups) {
    for (int index = 0; index < warmups; ++index) {
        compute_graph(backend, graph);
        ggml_backend_synchronize(backend);
    }
    std::vector<double> values;
    values.reserve((size_t) repeats);
    for (int index = 0; index < repeats; ++index) {
        const auto begin = clock_type::now();
        compute_graph(backend, graph);
        ggml_backend_synchronize(backend);
        const auto end = clock_type::now();
        values.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return summarize(std::move(values));
}

static summary measure_queued(ggml_backend_t backend, ggml_cgraph * graph, int repeats, int iterations) {
    std::vector<double> values;
    values.reserve((size_t) repeats);
    for (int sample = 0; sample < repeats; ++sample) {
        const auto begin = clock_type::now();
        for (int index = 0; index < iterations; ++index) {
            compute_graph(backend, graph);
        }
        ggml_backend_synchronize(backend);
        const auto end = clock_type::now();
        values.push_back(std::chrono::duration<double, std::milli>(end - begin).count() / iterations);
    }
    return summarize(std::move(values));
}

static double output_checksum(ggml_tensor * tensor) {
    std::vector<float> values((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    double checksum = 0.0;
    for (size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("expert compute produced a non-finite output");
        }
        checksum += values[index] * (double) (index + 1);
    }
    return checksum;
}

static comparison_result compare_tensors(
        const std::string & name, ggml_tensor * first, ggml_tensor * second) {
    if (!ggml_are_same_shape(first, second) || first->type != GGML_TYPE_F32 || second->type != GGML_TYPE_F32) {
        throw std::runtime_error("cannot compare incompatible tensors: " + name);
    }
    std::vector<float> a((size_t) ggml_nelements(first));
    std::vector<float> b(a.size());
    ggml_backend_tensor_get(first, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_get(second, b.data(), 0, b.size()*sizeof(float));
    comparison_result result;
    result.name = name;
    result.elements = a.size();
    double total_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            throw std::runtime_error("comparison produced a non-finite value: " + name);
        }
        if (a[i] == b[i]) {
            result.exact++;
        }
        const double error = std::abs((double) a[i] - (double) b[i]);
        result.max_abs = std::max(result.max_abs, error);
        total_abs += error;
    }
    result.mean_abs = a.empty() ? 0.0 : total_abs/a.size();
    return result;
}

static comparison_result compare_values(
        const std::string & name, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("cannot compare incompatible values: " + name);
    }
    comparison_result result;
    result.name = name;
    result.elements = a.size();
    double total_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) result.exact++;
        const double error = std::abs((double) a[i] - (double) b[i]);
        result.max_abs = std::max(result.max_abs, error);
        total_abs += error;
    }
    result.mean_abs = a.empty() ? 0.0 : total_abs/a.size();
    return result;
}

static std::string make_json(const options & config, const llama_model & model,
        ggml_backend_dev_t device, const ggml_tensor * gate_weight,
        const std::vector<mode_result> & results, const std::vector<comparison_result> & comparisons) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(9);
    output << "{\n  \"schema\":\"llama-wackmall-expert-compute-v1\","
           << "\n  \"scope\":\"resident expert compute; no H2D transfer or router is measured\","
           << "\n  \"model\":\"" << json_escape(model.desc()) << "\""
           << ",\n  \"device\":\"" << json_escape(ggml_backend_dev_name(device)) << "\""
           << ",\n  \"config\":{\"device_ordinal\":" << config.device
           << ",\"layer\":" << config.layer << ",\"expert\":" << config.expert
           << ",\"repeats\":" << config.repeats << ",\"warmups\":" << config.warmups
           << ",\"queued_iterations\":" << config.queued_iterations
           << ",\"weight_type\":\"" << ggml_type_name(gate_weight->type) << "\""
           << ",\"n_embd\":" << gate_weight->ne[0] << ",\"n_ff\":" << gate_weight->ne[1] << "},"
           << "\n  \"results\":[";
    bool first = true;
    const auto write_summary = [&](const summary & value) {
        output << "{\"min\":" << value.min << ",\"median\":" << value.median
               << ",\"p95\":" << value.p95 << ",\"mean\":" << value.mean << '}';
    };
    for (const mode_result & result : results) {
        output << (first ? "" : ",") << "\n    {\"mode\":\"" << result.mode
               << "\",\"weight_bytes\":" << result.weight_bytes << ",\"latency_ms\":";
        write_summary(result.latency_ms);
        output << ",\"queued_ms\":";
        write_summary(result.queued_ms);
        output << ",\"checksum\":" << result.checksum << '}';
        first = false;
    }
    output << "\n  ],\n  \"cpu_gpu_comparison\":[";
    first = true;
    for (const comparison_result & comparison : comparisons) {
        output << (first ? "" : ",") << "\n    {\"name\":\"" << comparison.name
               << "\",\"elements\":" << comparison.elements
               << ",\"exact\":" << comparison.exact
               << ",\"exact_fraction\":" << (comparison.elements ? (double) comparison.exact/comparison.elements : 0.0)
               << ",\"max_abs\":" << comparison.max_abs
               << ",\"mean_abs\":" << comparison.mean_abs
               << ",\"kernel_ms\":" << comparison.kernel_ms << '}';
        first = false;
    }
    output << "\n  ]\n}\n";
    return output.str();
}

static void write_json(const options & config, const std::string & json) {
    if (config.json_path.empty()) {
        std::cout << json;
        return;
    }
    {
        std::ifstream existing(config.json_path, std::ios::binary);
        if (existing.good()) {
            throw std::runtime_error("refusing to overwrite existing JSON: " + config.json_path);
        }
    }
    std::ofstream output(config.json_path, std::ios::binary | std::ios::out);
    if (!output) {
        throw std::runtime_error("cannot create JSON: " + config.json_path);
    }
    output << json;
    if (!output) {
        throw std::runtime_error("failed while writing JSON: " + config.json_path);
    }
    std::cout << "wrote " << config.json_path << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options config = parse_options(argc, argv);
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_ptr model(llama_model_load_from_file(config.model_path.c_str(), model_params));
        if (!model) {
            throw std::runtime_error("failed to load model");
        }
        if (config.layer >= (int) model->hparams.n_layer()) {
            throw std::runtime_error("layer is outside the base model range");
        }
        const llama_layer & layer = model->layers[(size_t) config.layer];
        const ggml_tensor * source_gate = layer.ffn_gate_exps;
        const ggml_tensor * source_up = layer.ffn_up_exps;
        const ggml_tensor * source_down = layer.ffn_down_exps;
        if (!source_gate || !source_up || !source_down) {
            throw std::runtime_error("selected layer does not have separate gate, up, and down expert tensors");
        }
        if (config.expert >= source_gate->ne[2] || config.expert >= source_up->ne[2] ||
                config.expert >= source_down->ne[2]) {
            throw std::runtime_error("expert is outside the selected layer tensor range");
        }
        if (source_gate->ne[0] != source_up->ne[0] || source_gate->ne[1] != source_up->ne[1] ||
                source_gate->ne[1] != source_down->ne[0] || source_down->ne[1] != source_gate->ne[0]) {
            throw std::runtime_error("unsupported expert matrix dimensions");
        }

        ggml_backend_dev_t device = select_gpu_device(config.device);
        if (!device) {
            throw std::runtime_error("requested GPU backend ordinal is not available");
        }
        backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        if (!backend) {
            throw std::runtime_error("failed to initialize GPU backend");
        }

        ggml_init_params context_params = {};
        context_params.mem_size = 16 * 1024 * 1024;
        context_params.mem_buffer = nullptr;
        context_params.no_alloc = true;
        context_ptr context(ggml_init(context_params));
        if (!context) {
            throw std::runtime_error("failed to initialize graph context");
        }

        ggml_tensor * weight_gate = make_weight(context.get(), source_gate, "gate.weight");
        ggml_tensor * weight_up = make_weight(context.get(), source_up, "up.weight");
        ggml_tensor * weight_down = make_weight(context.get(), source_down, "down.weight");
        ggml_tensor * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, source_gate->ne[0], 1);
        ggml_tensor * down_input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, source_down->ne[0], 1);
        ggml_set_name(input, "hidden.input");
        ggml_set_name(down_input, "activated.input");

        ggml_tensor * gate = ggml_mul_mat(context.get(), weight_gate, input);
        ggml_tensor * up = ggml_mul_mat(context.get(), weight_up, input);
        ggml_tensor * activated = ggml_mul(context.get(), ggml_silu(context.get(), gate), up);
        ggml_tensor * full_output = ggml_mul_mat(context.get(), weight_down, activated);
        ggml_tensor * down_output = ggml_mul_mat(context.get(), weight_down, down_input);
        ggml_set_name(activated, "gate-up.output");
        ggml_set_name(full_output, "full.output");
        ggml_set_name(down_output, "down.output");

        ggml_cgraph * graph_gate_up = ggml_new_graph_custom(context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_cgraph * graph_gate = ggml_new_graph_custom(context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_cgraph * graph_up = ggml_new_graph_custom(context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_cgraph * graph_down = ggml_new_graph_custom(context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_cgraph * graph_full = ggml_new_graph_custom(context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(graph_gate_up, activated);
        ggml_build_forward_expand(graph_gate, gate);
        ggml_build_forward_expand(graph_up, up);
        ggml_build_forward_expand(graph_down, down_output);
        ggml_build_forward_expand(graph_full, full_output);

        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (!buffer) {
            throw std::runtime_error("failed to allocate GPU graph tensors");
        }
        upload_expert(weight_gate, source_gate, config.expert);
        upload_expert(weight_up, source_up, config.expert);
        upload_expert(weight_down, source_down, config.expert);
        std::vector<float> input_values((size_t) source_gate->ne[0]);
        std::vector<float> down_values((size_t) source_down->ne[0]);
        for (size_t index = 0; index < input_values.size(); ++index) {
            input_values[index] = (float) std::sin((double) index * 0.013);
        }
        for (size_t index = 0; index < down_values.size(); ++index) {
            down_values[index] = (float) std::cos((double) index * 0.017);
        }
        ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
        ggml_backend_tensor_set(down_input, down_values.data(), 0, down_values.size() * sizeof(float));

        std::vector<mode_result> results;
        for (const auto & entry : std::vector<std::tuple<std::string, ggml_cgraph *, ggml_tensor *, size_t>>{
                     {"gate-up", graph_gate_up, activated, ggml_nbytes(weight_gate) + ggml_nbytes(weight_up)},
                     {"down", graph_down, down_output, ggml_nbytes(weight_down)},
                     {"full", graph_full, full_output, ggml_nbytes(weight_gate) + ggml_nbytes(weight_up) + ggml_nbytes(weight_down)},
             }) {
            mode_result result;
            result.mode = std::get<0>(entry);
            result.weight_bytes = std::get<3>(entry);
            result.latency_ms = measure_latency(backend.get(), std::get<1>(entry), config.repeats, config.warmups);
            result.queued_ms = measure_queued(backend.get(), std::get<1>(entry), config.repeats, config.queued_iterations);
            result.checksum = output_checksum(std::get<2>(entry));
            results.push_back(result);
        }

        ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_device) {
            throw std::runtime_error("CPU backend is not available");
        }
        backend_ptr cpu_backend(ggml_backend_dev_init(cpu_device, nullptr));
        if (!cpu_backend) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        context_ptr cpu_context(ggml_init(context_params));
        if (!cpu_context) {
            throw std::runtime_error("failed to initialize CPU comparison context");
        }
        ggml_tensor * cpu_weight_gate = make_weight(cpu_context.get(), source_gate, "cpu.gate.weight");
        ggml_tensor * cpu_weight_up = make_weight(cpu_context.get(), source_up, "cpu.up.weight");
        ggml_tensor * cpu_input = ggml_new_tensor_2d(cpu_context.get(), GGML_TYPE_F32, source_gate->ne[0], 1);
        ggml_tensor * cpu_gate = ggml_mul_mat(cpu_context.get(), cpu_weight_gate, cpu_input);
        ggml_tensor * cpu_up = ggml_mul_mat(cpu_context.get(), cpu_weight_up, cpu_input);
        ggml_tensor * cpu_activated = ggml_mul(cpu_context.get(), ggml_silu(cpu_context.get(), cpu_gate), cpu_up);
        ggml_cgraph * cpu_graph = ggml_new_graph_custom(cpu_context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(cpu_graph, cpu_activated);
        ggml_cgraph * cpu_gate_graph = ggml_new_graph_custom(cpu_context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_cgraph * cpu_up_graph = ggml_new_graph_custom(cpu_context.get(), GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(cpu_gate_graph, cpu_gate);
        ggml_build_forward_expand(cpu_up_graph, cpu_up);
        buffer_ptr cpu_buffer(ggml_backend_alloc_ctx_tensors(cpu_context.get(), cpu_backend.get()));
        if (!cpu_buffer) {
            throw std::runtime_error("failed to allocate CPU comparison tensors");
        }
        upload_expert(cpu_weight_gate, source_gate, config.expert);
        upload_expert(cpu_weight_up, source_up, config.expert);
        ggml_backend_tensor_set(cpu_input, input_values.data(), 0, input_values.size()*sizeof(float));
        compute_graph(cpu_backend.get(), cpu_graph);
        ggml_backend_synchronize(cpu_backend.get());
        compute_graph(backend.get(), graph_gate);
        compute_graph(backend.get(), graph_up);
        ggml_backend_synchronize(backend.get());
        compute_graph(cpu_backend.get(), cpu_gate_graph);
        compute_graph(cpu_backend.get(), cpu_up_graph);
        ggml_backend_synchronize(cpu_backend.get());

        std::vector<comparison_result> comparisons;
        comparisons.push_back(compare_tensors("gate", gate, cpu_gate));
        comparisons.push_back(compare_tensors("up", up, cpu_up));
        comparisons.push_back(compare_tensors("activated", activated, cpu_activated));
#if defined(GGML_USE_CUDA)
        if (source_gate->type == GGML_TYPE_Q4_K) {
            std::vector<uint8_t> quantized_input(
                    (size_t) (source_gate->ne[0]/QK_K)*sizeof(block_q8_K));
            quantize_row_q8_K_ref(input_values.data(),
                    (block_q8_K *) quantized_input.data(), source_gate->ne[0]);
            std::vector<uint8_t> combined_weights(
                    ggml_nbytes(source_gate)/(size_t) source_gate->ne[2] +
                    ggml_nbytes(source_up)/(size_t) source_up->ne[2]);
            const size_t gate_bytes = ggml_nbytes(source_gate)/(size_t) source_gate->ne[2];
            const char * gate_source = (const char *) source_gate->data + source_gate->nb[2]*config.expert;
            const char * up_source = (const char *) source_up->data + source_up->nb[2]*config.expert;
            memcpy(combined_weights.data(), gate_source, gate_bytes);
            memcpy(combined_weights.data() + gate_bytes, up_source, combined_weights.size() - gate_bytes);
            std::vector<float> deterministic_combined(
                    (size_t) source_gate->ne[1] + (size_t) source_up->ne[1]);
            double deterministic_combined_ms = 0.0;
            if (!expert_compute_q4_k_avx2(combined_weights.data(), quantized_input.data(),
                        (int) source_gate->ne[0], (int) deterministic_combined.size(), 100,
                        deterministic_combined.data(), &deterministic_combined_ms)) {
                throw std::runtime_error("deterministic Q4_K CUDA comparison failed");
            }
            std::vector<float> deterministic_gate(
                    deterministic_combined.begin(), deterministic_combined.begin() + source_gate->ne[1]);
            std::vector<float> deterministic_up(
                    deterministic_combined.begin() + source_gate->ne[1], deterministic_combined.end());
            std::vector<float> cpu_gate_values(deterministic_gate.size());
            std::vector<float> cpu_up_values(deterministic_up.size());
            ggml_backend_tensor_get(cpu_gate, cpu_gate_values.data(), 0, cpu_gate_values.size()*sizeof(float));
            ggml_backend_tensor_get(cpu_up, cpu_up_values.data(), 0, cpu_up_values.size()*sizeof(float));
            comparison_result deterministic_gate_comparison =
                compare_values("deterministic_gate", deterministic_gate, cpu_gate_values);
            comparison_result deterministic_up_comparison =
                compare_values("deterministic_up", deterministic_up, cpu_up_values);
            deterministic_gate_comparison.kernel_ms = deterministic_combined_ms;
            deterministic_up_comparison.kernel_ms = deterministic_combined_ms;
            comparisons.push_back(deterministic_gate_comparison);
            comparisons.push_back(deterministic_up_comparison);

            std::vector<uint8_t> device_quantized_input(quantized_input.size());
            std::vector<float> device_quantized_combined(deterministic_combined.size());
            double device_quantized_ms = 0.0;
            if (!expert_compute_q4_k_avx2_f32(combined_weights.data(), input_values.data(),
                        (int) source_gate->ne[0], (int) device_quantized_combined.size(), 100,
                        device_quantized_combined.data(), device_quantized_input.data(), &device_quantized_ms)) {
                throw std::runtime_error("device-quantized deterministic Q4_K CUDA comparison failed");
            }
            if (device_quantized_input != quantized_input) {
                throw std::runtime_error("device Q8_K quantization differs from CPU reference");
            }
            std::vector<float> device_quantized_gate(
                    device_quantized_combined.begin(), device_quantized_combined.begin() + source_gate->ne[1]);
            std::vector<float> device_quantized_up(
                    device_quantized_combined.begin() + source_gate->ne[1], device_quantized_combined.end());
            comparison_result device_quantized_gate_comparison =
                compare_values("device_quantized_gate", device_quantized_gate, cpu_gate_values);
            comparison_result device_quantized_up_comparison =
                compare_values("device_quantized_up", device_quantized_up, cpu_up_values);
            device_quantized_gate_comparison.kernel_ms = device_quantized_ms;
            device_quantized_up_comparison.kernel_ms = device_quantized_ms;
            comparisons.push_back(device_quantized_gate_comparison);
            comparisons.push_back(device_quantized_up_comparison);
        }
#endif

        write_json(config, make_json(config, *model, device, source_gate, results, comparisons));
        cpu_buffer.reset();
        cpu_context.reset();
        cpu_backend.reset();
        buffer.reset();
        context.reset();
        backend.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    }
}
