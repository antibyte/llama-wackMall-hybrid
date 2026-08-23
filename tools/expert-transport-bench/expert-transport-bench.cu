#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
    int device = 0;
    int repeats = 100;
    int warmups = 10;
    int overlap_us = 0;
    bool overlap_cpu = false;
    size_t working_set_mib = 16;
    std::string json_path;
    std::vector<std::pair<std::string, size_t>> segments;
};

struct sample {
    double device_ms = 0.0;
    double wall_ms = 0.0;
    double stage_ms = 0.0;
    double compute_ms = 0.0;
    double exposed_copy_ms = 0.0;
    double hidden_copy_ratio = 0.0;
};

struct stats {
    double min = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double mean = 0.0;
};

struct result {
    std::string mode;
    std::string caveat;
    size_t bytes = 0;
    std::vector<sample> samples;
};

static void check_cuda(cudaError_t status, const char * expression, const char * file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << file << ':' << line << ": " << expression << " failed: "
                << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

class cuda_event {
public:
    explicit cuda_event(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&value_, flags));
    }

    ~cuda_event() {
        if (value_) {
            cudaEventDestroy(value_);
        }
    }

    cuda_event(const cuda_event &) = delete;
    cuda_event & operator=(const cuda_event &) = delete;

    operator cudaEvent_t() const {
        return value_;
    }

private:
    cudaEvent_t value_ = nullptr;
};

class cuda_stream {
public:
    cuda_stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking));
    }

    ~cuda_stream() {
        if (value_) {
            cudaStreamDestroy(value_);
        }
    }

    cuda_stream(const cuda_stream &) = delete;
    cuda_stream & operator=(const cuda_stream &) = delete;

    operator cudaStream_t() const {
        return value_;
    }

private:
    cudaStream_t value_ = nullptr;
};

class device_allocation {
public:
    explicit device_allocation(size_t bytes) {
        CUDA_CHECK(cudaMalloc(&value_, bytes));
    }

    ~device_allocation() {
        if (value_) {
            cudaFree(value_);
        }
    }

    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;

    void * get() const {
        return value_;
    }

private:
    void * value_ = nullptr;
};

class host_allocation {
public:
    host_allocation(size_t bytes, unsigned int flags) {
        CUDA_CHECK(cudaHostAlloc(&value_, bytes, flags));
    }

    ~host_allocation() {
        if (value_) {
            cudaFreeHost(value_);
        }
    }

    host_allocation(const host_allocation &) = delete;
    host_allocation & operator=(const host_allocation &) = delete;

    void * get() const {
        return value_;
    }

private:
    void * value_ = nullptr;
};

__global__ void checksum_kernel(const uint4 * source, size_t words, unsigned long long * checksum) {
    unsigned long long sum = 0;
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < words; index += stride) {
        const uint4 value = source[index];
        sum += value.x;
        sum += value.y;
        sum += value.z;
        sum += value.w;
    }
    atomicAdd(checksum, sum);
}

__global__ void spin_kernel(unsigned long long cycles, unsigned long long * sink) {
    const unsigned long long begin = clock64();
    unsigned long long value = begin;
    while (clock64() - begin < cycles) {
        value = value * 2862933555777941757ULL + 3037000493ULL;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *sink ^= value;
    }
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
    if (value > (uint64_t) std::numeric_limits<int>::max() || value < (uint64_t) minimum) {
        throw std::runtime_error(std::string("out-of-range ") + name + ": " + text);
    }
    return (int) value;
}

static void print_usage(const char * program) {
    std::cerr
        << "usage: " << program << " --segment NAME=BYTES [options]\n"
        << "  --segment NAME=BYTES   repeatable model-derived transfer size\n"
        << "  --device N             CUDA device (default 0)\n"
        << "  --repeats N            measured repetitions (default 100)\n"
        << "  --warmups N            warm-up repetitions (default 10)\n"
        << "  --working-set-mib N    rotating source set (default 16 MiB)\n"
        << "  --overlap-us N         synthetic GPU window; 0 disables (default 0)\n"
        << "  --overlap-cpu          pin H2D against a concurrent host DRAM read\n"
        << "  --json PATH            write JSON without overwriting an existing file\n";
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
        if (argument == "--segment") {
            const std::string value = take_value();
            const size_t separator = value.find('=');
            if (separator == std::string::npos || separator == 0 || separator + 1 == value.size()) {
                throw std::runtime_error("segment must use NAME=BYTES: " + value);
            }
            const std::string name = value.substr(0, separator);
            if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") != std::string::npos) {
                throw std::runtime_error("segment name contains unsupported characters: " + name);
            }
            const uint64_t bytes = parse_u64(value.substr(separator + 1), "segment bytes");
            if (bytes == 0 || bytes % sizeof(uint4) != 0 || bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("segment bytes must be a positive multiple of 16");
            }
            parsed.segments.emplace_back(name, (size_t) bytes);
        } else if (argument == "--device") {
            parsed.device = parse_int(take_value(), "device", 0);
        } else if (argument == "--repeats") {
            parsed.repeats = parse_int(take_value(), "repeats", 1);
        } else if (argument == "--warmups") {
            parsed.warmups = parse_int(take_value(), "warmups", 0);
        } else if (argument == "--working-set-mib") {
            parsed.working_set_mib = (size_t) parse_u64(take_value(), "working-set-mib");
            if (parsed.working_set_mib == 0 || parsed.working_set_mib > 4096) {
                throw std::runtime_error("working-set-mib must be in [1, 4096]");
            }
        } else if (argument == "--overlap-us") {
            parsed.overlap_us = parse_int(take_value(), "overlap-us", 0);
        } else if (argument == "--overlap-cpu") {
            parsed.overlap_cpu = true;
        } else if (argument == "--json") {
            parsed.json_path = take_value();
            if (parsed.json_path.empty()) {
                throw std::runtime_error("json path must not be empty");
            }
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (parsed.segments.empty()) {
        throw std::runtime_error("at least one --segment NAME=BYTES is required");
    }
    return parsed;
}

static double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static stats summarize(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("cannot summarize an empty sample set");
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

static std::vector<double> collect(const result & value, double sample::* member) {
    std::vector<double> output;
    output.reserve(value.samples.size());
    for (const sample & item : value.samples) {
        output.push_back(item.*member);
    }
    return output;
}

static void fill_sources(unsigned char * pageable, unsigned char * pinned, unsigned char * mapped, size_t bytes) {
    for (size_t index = 0; index < bytes; ++index) {
        const unsigned char value = (unsigned char) ((index * 1315423911ULL + 0x5aU) & 0xffU);
        pageable[index] = value;
        pinned[index] = value;
        mapped[index] = value;
    }
}

static sample measure_copy(void * destination, const void * source, size_t bytes, cudaStream_t stream) {
    cuda_event begin;
    cuda_event end;
    const auto wall_begin = clock_type::now();
    CUDA_CHECK(cudaEventRecord(begin, stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    const auto wall_end = clock_type::now();
    float device_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&device_ms, begin, end));
    return {device_ms, elapsed_ms(wall_begin, wall_end)};
}

static sample measure_staged_copy(void * destination, void * staging, const void * source,
        size_t bytes, cudaStream_t stream) {
    const auto wall_begin = clock_type::now();
    const auto stage_begin = wall_begin;
    std::memcpy(staging, source, bytes);
    const auto stage_end = clock_type::now();
    cuda_event begin;
    cuda_event end;
    CUDA_CHECK(cudaEventRecord(begin, stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, staging, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    const auto wall_end = clock_type::now();
    float device_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&device_ms, begin, end));
    sample output;
    output.device_ms = device_ms;
    output.wall_ms = elapsed_ms(wall_begin, wall_end);
    output.stage_ms = elapsed_ms(stage_begin, stage_end);
    return output;
}

static sample measure_read(const void * source, size_t bytes, void * checksum, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(checksum, 0, sizeof(unsigned long long), stream));
    cuda_event begin;
    cuda_event end;
    const auto wall_begin = clock_type::now();
    CUDA_CHECK(cudaEventRecord(begin, stream));
    const int blocks = std::min<int>(4096, std::max<int>(1, (int) ((bytes / sizeof(uint4) + 255) / 256)));
    checksum_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<const uint4 *>(source), bytes / sizeof(uint4),
            static_cast<unsigned long long *>(checksum));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    const auto wall_end = clock_type::now();
    float device_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&device_ms, begin, end));
    return {device_ms, elapsed_ms(wall_begin, wall_end)};
}

static sample measure_overlap(void * destination, const void * source, size_t bytes,
        int overlap_us, int clock_rate_khz, void * sink, cudaStream_t copy_stream,
        cudaStream_t compute_stream) {
    cuda_event copy_begin;
    cuda_event copy_end;
    cuda_event compute_begin;
    cuda_event compute_end;
    const unsigned long long cycles = (unsigned long long) clock_rate_khz * overlap_us / 1000ULL;
    const auto wall_begin = clock_type::now();
    CUDA_CHECK(cudaEventRecord(compute_begin, compute_stream));
    spin_kernel<<<1, 1, 0, compute_stream>>>(cycles, static_cast<unsigned long long *>(sink));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(compute_end, compute_stream));
    CUDA_CHECK(cudaEventRecord(copy_begin, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaEventRecord(copy_end, copy_stream));
    CUDA_CHECK(cudaEventSynchronize(compute_end));
    CUDA_CHECK(cudaEventSynchronize(copy_end));
    const auto wall_end = clock_type::now();
    float copy_ms = 0.0f;
    float compute_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, copy_begin, copy_end));
    CUDA_CHECK(cudaEventElapsedTime(&compute_ms, compute_begin, compute_end));
    sample output;
    output.device_ms = copy_ms;
    output.compute_ms = compute_ms;
    output.wall_ms = elapsed_ms(wall_begin, wall_end);
    output.exposed_copy_ms = std::max(0.0, output.wall_ms - output.compute_ms);
    output.hidden_copy_ratio = copy_ms > 0.0f
            ? std::clamp(1.0 - output.exposed_copy_ms / copy_ms, 0.0, 1.0)
            : 0.0;
    return output;
}

static uint64_t cpu_checksum(const unsigned char * source, size_t bytes) {
    uint64_t acc = 0;
    const size_t n = bytes / sizeof(uint64_t);
    const uint64_t * words = reinterpret_cast<const uint64_t *>(source);
    for (size_t i = 0; i < n; ++i) {
        acc += words[i];
    }
    return acc;
}

static sample measure_overlap_cpu(void * destination, const unsigned char * dma_source,
        const unsigned char * cpu_source, size_t bytes, cudaStream_t copy_stream) {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> copies{0};
    std::atomic<uint64_t> sink{0};
    cuda_event copy_begin;
    cuda_event copy_end;
    std::thread worker([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            sink.fetch_add(cpu_checksum(cpu_source, bytes), std::memory_order_relaxed);
            copies.fetch_add(1, std::memory_order_relaxed);
        }
    });
    const auto wall_begin = clock_type::now();
    CUDA_CHECK(cudaEventRecord(copy_begin, copy_stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, dma_source, bytes, cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaEventRecord(copy_end, copy_stream));
    CUDA_CHECK(cudaEventSynchronize(copy_end));
    const auto wall_end = clock_type::now();
    stop.store(true, std::memory_order_relaxed);
    worker.join();
    float copy_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, copy_begin, copy_end));
    sample output;
    output.device_ms = copy_ms;
    output.wall_ms = elapsed_ms(wall_begin, wall_end);
    const uint64_t ncopy = copies.load(std::memory_order_relaxed);
    output.compute_ms = ncopy > 0 ? output.wall_ms / (double) ncopy : output.wall_ms;
    return output;
}

static void append_result_json(std::ostream & output, const result & value, bool first) {
    const stats device = summarize(collect(value, &sample::device_ms));
    const stats wall = summarize(collect(value, &sample::wall_ms));
    const stats stage = summarize(collect(value, &sample::stage_ms));
    const stats compute = summarize(collect(value, &sample::compute_ms));
    const stats exposed = summarize(collect(value, &sample::exposed_copy_ms));
    const stats hidden = summarize(collect(value, &sample::hidden_copy_ratio));
    const double gib_per_s = device.median > 0.0
            ? (double) value.bytes / (device.median / 1000.0) / (1024.0 * 1024.0 * 1024.0)
            : 0.0;
    output << (first ? "" : ",") << "\n    {\"mode\":\"" << json_escape(value.mode)
           << "\",\"bytes\":" << value.bytes
           << ",\"caveat\":\"" << json_escape(value.caveat) << "\""
           << ",\"device_ms\":{\"min\":" << device.min << ",\"median\":" << device.median
           << ",\"p95\":" << device.p95 << ",\"mean\":" << device.mean << "}"
           << ",\"wall_ms\":{\"min\":" << wall.min << ",\"median\":" << wall.median
           << ",\"p95\":" << wall.p95 << ",\"mean\":" << wall.mean << "}"
           << ",\"stage_ms\":{\"median\":" << stage.median << "}"
           << ",\"compute_ms\":{\"median\":" << compute.median << "}"
           << ",\"exposed_copy_ms\":{\"median\":" << exposed.median << "}"
           << ",\"hidden_copy_ratio\":{\"median\":" << hidden.median << "}"
           << ",\"device_gib_per_s_median\":" << gib_per_s << '}';
}

static std::string make_json(const options & config, const cudaDeviceProp & properties,
        size_t source_slots, size_t working_bytes,
        const std::vector<std::pair<std::string, std::vector<result>>> & all_results) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    output << "{\n  \"schema\":\"llama-wackmall-expert-transport-v1\","
           << "\n  \"scope\":\"transport-only; no quantized expert matmul is measured\","
           << "\n  \"device\":{\"ordinal\":" << config.device
           << ",\"name\":\"" << json_escape(properties.name) << "\""
           << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor << "\""
           << ",\"total_vram_bytes\":" << properties.totalGlobalMem
           << ",\"can_map_host_memory\":" << (properties.canMapHostMemory ? "true" : "false")
           << ",\"unified_addressing\":" << (properties.unifiedAddressing ? "true" : "false")
           << ",\"async_engine_count\":" << properties.asyncEngineCount
           << ",\"concurrent_kernels\":" << (properties.concurrentKernels ? "true" : "false")
           << ",\"clock_rate_khz\":" << properties.clockRate << "},"
           << "\n  \"config\":{\"repeats\":" << config.repeats
           << ",\"warmups\":" << config.warmups
           << ",\"working_set_mib\":" << config.working_set_mib
           << ",\"working_set_bytes\":" << working_bytes
           << ",\"source_slots\":" << source_slots
           << ",\"overlap_us\":" << config.overlap_us
           << ",\"overlap_cpu\":" << (config.overlap_cpu ? "true" : "false") << "},"
           << "\n  \"segments\":[";
    bool first_segment = true;
    for (const auto & group : all_results) {
        if (!first_segment) {
            output << ',';
        }
        first_segment = false;
        output << "\n    {\"name\":\"" << json_escape(group.first) << "\",\"bytes\":"
               << group.second.front().bytes << ",\"results\":[";
        bool first_result = true;
        for (const result & value : group.second) {
            append_result_json(output, value, first_result);
            first_result = false;
        }
        output << "\n    ]}";
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
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (config.device >= device_count) {
            throw std::runtime_error("CUDA device ordinal is not available");
        }
        CUDA_CHECK(cudaSetDevice(config.device));
        cudaDeviceProp properties = {};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, config.device));

        size_t max_bytes = 0;
        for (const auto & segment : config.segments) {
            max_bytes = std::max(max_bytes, segment.second);
        }
        const size_t requested_working_bytes = config.working_set_mib * 1024ULL * 1024ULL;
        const size_t source_slots = std::max<size_t>(1, requested_working_bytes / max_bytes);
        if (source_slots > std::numeric_limits<size_t>::max() / max_bytes) {
            throw std::runtime_error("working set size overflows size_t");
        }
        const size_t working_bytes = source_slots * max_bytes;

        std::vector<unsigned char> pageable(working_bytes);
        host_allocation pinned(working_bytes, cudaHostAllocDefault);
        host_allocation mapped(working_bytes, cudaHostAllocMapped);
        host_allocation staging(max_bytes, cudaHostAllocDefault);
        host_allocation cpu_scratch(max_bytes, cudaHostAllocDefault);
        device_allocation target(max_bytes);
        device_allocation device_source(working_bytes);
        device_allocation checksum(sizeof(unsigned long long));
        fill_sources(pageable.data(), static_cast<unsigned char *>(pinned.get()),
                static_cast<unsigned char *>(mapped.get()), working_bytes);
        std::memcpy(cpu_scratch.get(), pinned.get(), max_bytes);
        CUDA_CHECK(cudaMemcpy(device_source.get(), pinned.get(), working_bytes, cudaMemcpyHostToDevice));

        void * mapped_device = nullptr;
        if (properties.canMapHostMemory) {
            CUDA_CHECK(cudaHostGetDevicePointer(&mapped_device, mapped.get(), 0));
        }

        cuda_stream copy_stream;
        cuda_stream compute_stream;
        std::vector<std::pair<std::string, std::vector<result>>> all_results;
        for (const auto & segment : config.segments) {
            std::vector<result> results;
            const auto run_copy_mode = [&](const std::string & mode, const std::string & caveat,
                    auto measure) {
                result value;
                value.mode = mode;
                value.caveat = caveat;
                value.bytes = segment.second;
                for (int repetition = -config.warmups; repetition < config.repeats; ++repetition) {
                    const size_t slot = (size_t) (repetition + config.warmups) % source_slots;
                    const size_t offset = slot * max_bytes;
                    const sample measured = measure(offset);
                    if (repetition >= 0) {
                        value.samples.push_back(measured);
                    }
                }
                results.push_back(std::move(value));
            };

            run_copy_mode("pageable_h2d", "wall time includes any runtime pageable staging", [&](size_t offset) {
                return measure_copy(target.get(), pageable.data() + offset, segment.second, copy_stream);
            });
            run_copy_mode("pageable_to_pinned_to_h2d", "wall time includes CPU memcpy into one pinned staging buffer", [&](size_t offset) {
                return measure_staged_copy(target.get(), staging.get(), pageable.data() + offset,
                        segment.second, copy_stream);
            });
            run_copy_mode("pinned_h2d", "source is already page-locked", [&](size_t offset) {
                return measure_copy(target.get(), static_cast<unsigned char *>(pinned.get()) + offset,
                        segment.second, copy_stream);
            });
            run_copy_mode("device_read", "sequential checksum kernel; comparator for mapped_read only", [&](size_t offset) {
                return measure_read(static_cast<unsigned char *>(device_source.get()) + offset,
                        segment.second, checksum.get(), compute_stream);
            });
            if (mapped_device) {
                run_copy_mode("mapped_read", "sequential checksum kernel over mapped host memory; not a quantized matmul", [&](size_t offset) {
                    return measure_read(static_cast<unsigned char *>(mapped_device) + offset,
                            segment.second, checksum.get(), compute_stream);
                });
            }
            if (config.overlap_us > 0) {
                run_copy_mode("pinned_h2d_overlap", "synthetic one-block clock window; not model compute", [&](size_t offset) {
                    return measure_overlap(target.get(), static_cast<unsigned char *>(pinned.get()) + offset,
                            segment.second, config.overlap_us, properties.clockRate, checksum.get(),
                            copy_stream, compute_stream);
                });
            }
            if (config.overlap_cpu) {
                run_copy_mode("pinned_h2d_cpu_overlap",
                        "PCIe fill overlapped with a host DRAM checksum of a disjoint buffer",
                        [&](size_t offset) {
                    return measure_overlap_cpu(
                            target.get(),
                            static_cast<unsigned char *>(pinned.get()) + offset,
                            static_cast<unsigned char *>(cpu_scratch.get()),
                            segment.second, copy_stream);
                });
            }
            all_results.emplace_back(segment.first, std::move(results));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        write_json(config, make_json(config, properties, source_slots, working_bytes, all_results));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 2;
    }
}
