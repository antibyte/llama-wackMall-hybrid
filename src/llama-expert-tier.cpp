#include "llama-expert-tier.h"

#include "llama-expert-cache.h"
#include "llama-expert-placement.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// tier status must print at default verbosity; libllama INFO requires -v
#define TIER_LOG(...) fprintf(stderr, __VA_ARGS__)

namespace llama_expert_tier {

enum class warm_admission_mode {
    immediate,
    second_hit,
    frequency,
};

enum class expert_usage_mode {
    cumulative,
    session,
};

struct store {
    ggml_tensor * w_hot  = nullptr; // [ne0, ne1, n_slots] on GPU, sentinel slot zeroed
    ggml_tensor * lut    = nullptr; // i32 [n_expert] on GPU: expert -> hot slot | sentinel
    ggml_tensor * mask   = nullptr; // i32 [n_expert] on CPU: 1 = cold
    ggml_tensor * counts = nullptr; // i32 [n_expert+1] on CPU: selection stats
    int  il      = -1;
    bool is_down = false;
};

// per-layer grouping for stats and online repin
struct layer_tier {
    int il       = -1;
    int n_expert = 0;
    int n_slots  = 0; // incl. sentinel slot
    int n_fixed  = 0;
    int sentinel = 0;
    store * sd = nullptr; // store holding the down weight (counts source)
    std::vector<std::pair<const ggml_tensor *, store *>> ws;
    std::vector<int32_t> slot_expert; // [n_slots], -1 = empty
    std::vector<int32_t> lut_host;    // [n_expert]
    std::vector<float>   score;       // [n_expert], cumulative (LLAMA_EXPERT_DECAY)
    std::vector<uint64_t> cum;        // [n_expert], seed + observations (legacy persistence)
    std::vector<uint64_t> observed;   // [n_expert], observations from this process only
    std::vector<int32_t> dwell;       // [n_slots]
    llama_expert_cache::state warm_cache;
    llama_expert_cache::second_hit_admission warm_admission;
    std::vector<float> warm_frequency;
    std::vector<int32_t> warm_pending;
    bool warm_enabled = false;
    uint64_t cum_fixed = 0, cum_warm = 0, cum_cold = 0, cum_total = 0, cum_graphs = 0;
    uint64_t warm_promotions = 0, warm_evictions = 0;
    uint64_t warm_to_fixed = 0;
    uint64_t warm_admission_deferrals = 0;
};

static int  g_S        = 16;
static int  g_W        = 0;
static int  g_tmax     = 16;
static bool g_adapt    = false;
static bool g_hot_only = false; // set by build_moe_cold per layer
static bool g_warm_auto = false;
static int  g_warm_auto_max = 4;
static int  g_vram_reserve_mib = 512;
static bool g_warm_reset_request = true;
static warm_admission_mode g_warm_admission = warm_admission_mode::immediate;
static expert_usage_mode g_usage_mode = expert_usage_mode::cumulative;
static int  g_warm_admission_window = 8;
static bool g_prefetch_requested = false;
static bool g_prefetch_ready = false;
static bool g_collect_counts = true;
static bool g_timing_enabled = false;
static int  g_cpu_single_row_chunk = 64;
static bool g_cpu_parallel_activation = false;
static bool g_cpu_async = false;
static int  g_cpu_down_prefetch = 0;
static bool g_cpu_reuse_rows = false;
static bool g_static_no_sync_requested = false;
static bool g_static_no_sync_active = false;
static int  g_prefetch_streams = 1;
static int  g_prefetch_max_inflight = 2;
static int  g_mtp_n = 0;
static bool g_warm_mtp_guarded = false;
static bool g_variable_placement = false;
static bool g_configured_enabled = false;
static llama_expert_placement::manifest g_placement;

static ggml_context * g_ctx_gpu = nullptr; // owned for process lifetime
static ggml_context * g_ctx_cpu = nullptr;

static std::unordered_map<const ggml_tensor *, store> g_stores;
static std::vector<layer_tier> g_layers;
static uint64_t g_repins = 0; // hot-set changes since init
static uint64_t g_warm_promotions = 0;
static uint64_t g_warm_evictions  = 0;
static uint64_t g_warm_to_fixed   = 0;
static uint64_t g_warm_admission_deferrals = 0;
static uint64_t g_h2d_copies      = 0;
static uint64_t g_h2d_bytes       = 0;
static int64_t  g_h2d_copy_us     = 0;
static std::string g_model_desc;
static uint64_t g_request_id = 0;
static size_t g_async_layer_cursor = 0;

struct async_copy_job {
    ggml_backend_buffer_t host_buffer = nullptr;
    ggml_backend_event_t event = nullptr;
    void * host_ptr = nullptr;
    size_t capacity = 0;
    bool active = false;
    bool completed = false;
    int layer = -1;
    int slot = -1;
    int expert = -1;
    uint64_t bytes = 0;
    int64_t started_us = 0;
    int64_t completed_us = 0;
};

static ggml_backend_t g_prefetch_backend = nullptr;
static std::vector<async_copy_job> g_async_jobs;
static std::mutex g_async_mutex;
static std::condition_variable g_async_cv;
static std::deque<size_t> g_async_queue;
static std::thread g_async_worker;
static bool g_async_stop = false;

static bool parse_nonnegative_int(const char * value, int & result) {
    if (!value || !value[0]) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    result = (int) parsed;
    return true;
}

static bool output_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value && value[0] && strcmp(value, "0");
}

static bool warm_mtp_experimental() {
    const char * value = getenv("LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL");
    return value && !strcmp(value, "1");
}

static void warm_invariant_abort(const layer_tier & L, const char * where) {
    std::string error;
    if (L.warm_cache.validate(&error)) {
        if ((int) L.warm_pending.size() != L.warm_cache.n_warm()) {
            error = "pending vector size mismatch";
        }
    }
    if (error.empty()) {
        std::vector<int> pending_occurrences(L.n_expert, 0);
        for (size_t warm_index = 0; warm_index < L.warm_pending.size(); ++warm_index) {
            const int expert = L.warm_pending[warm_index];
            if (expert < 0) {
                continue;
            }
            const int slot = L.warm_cache.n_fixed() + (int) warm_index;
            if (expert >= L.n_expert) {
                error = "pending slot " + std::to_string(slot) + " has invalid expert " + std::to_string(expert);
                break;
            }
            if (++pending_occurrences[expert] != 1) {
                error = "pending expert " + std::to_string(expert) + " is duplicated at slot " + std::to_string(slot);
                break;
            }
            if (L.warm_cache.locate(expert) != llama_expert_cache::location::cold ||
                    L.warm_cache.expert_in_slot(slot) != -1) {
                error = "pending expert " + std::to_string(expert) + " owns non-empty slot " + std::to_string(slot);
                break;
            }
        }
    }
    if (error.empty()) {
        const auto & lut = L.warm_cache.lut();
        if (L.sd && L.sd->mask && L.sd->mask->data) {
            const int32_t * mask = (const int32_t *) L.sd->mask->data;
            for (int expert = 0; expert < L.n_expert; ++expert) {
                const int want = lut[expert] == L.warm_cache.sentinel() ? 1 : 0;
                if (mask[expert] != want) {
                    error = "mask/lut mismatch for expert " + std::to_string(expert);
                    break;
                }
            }
        }
    }
    if (!error.empty()) {
        TIER_LOG("expert_tier: FATAL warm-cache invariant at %s: layer=%d state=%s request=%llu mtp_step=-1\n",
                where, L.il, error.c_str(), (unsigned long long) g_request_id);
        abort();
    }
}

static void upload_warm_maps(layer_tier & L) {
    const auto & lut = L.warm_cache.lut();
    const int sentinel = L.warm_cache.sentinel();
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        int32_t * mask = (int32_t *) st.mask->data;
        for (int expert = 0; expert < L.n_expert; ++expert) {
            mask[expert] = lut[expert] == sentinel ? 1 : 0;
        }
        ggml_backend_tensor_set(st.lut, lut.data(), 0, L.n_expert*sizeof(int32_t));
    }
}

static void async_worker_main() {
    while (true) {
        size_t job_index = 0;
        {
            std::unique_lock<std::mutex> lock(g_async_mutex);
            g_async_cv.wait(lock, [] { return g_async_stop || !g_async_queue.empty(); });
            if (g_async_stop && g_async_queue.empty()) {
                return;
            }
            job_index = g_async_queue.front();
            g_async_queue.pop_front();
        }

        // Measure the actual blocking interval per stream. Timestamping when a
        // later job is merely enqueued would double-count time spent behind an
        // earlier copy on the same stream.
        const int64_t wait_started_us = ggml_time_us();
        ggml_backend_event_synchronize(g_async_jobs[job_index].event);

        {
            std::lock_guard<std::mutex> lock(g_async_mutex);
            g_async_jobs[job_index].started_us = wait_started_us;
            g_async_jobs[job_index].completed_us = ggml_time_us();
            g_async_jobs[job_index].completed = true;
        }
    }
}

static void shutdown_async_resources() {
    {
        std::lock_guard<std::mutex> lock(g_async_mutex);
        g_async_stop = true;
    }
    g_async_cv.notify_all();
    if (g_async_worker.joinable()) {
        g_async_worker.join();
    }

    for (auto & job : g_async_jobs) {
        if (job.event) {
            ggml_backend_event_free(job.event);
            job.event = nullptr;
        }
        if (job.host_buffer) {
            ggml_backend_buffer_free(job.host_buffer);
            job.host_buffer = nullptr;
        }
    }
    g_async_jobs.clear();
    g_async_queue.clear();
    if (g_prefetch_backend) {
        ggml_backend_free(g_prefetch_backend);
        g_prefetch_backend = nullptr;
    }
    g_prefetch_ready = false;
}

static bool init_async_resources(ggml_backend_dev_t dev, size_t staging_size) {
    if (!dev || staging_size == 0 || g_prefetch_streams != 1 || g_prefetch_max_inflight <= 0) {
        return false;
    }

    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
        TIER_LOG("expert_tier: async prefetch requires backend async, pinned-host, and event support\n");
        return false;
    }
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) {
        TIER_LOG("expert_tier: async prefetch has no device host-buffer type\n");
        return false;
    }

    g_prefetch_backend = ggml_backend_dev_init(dev, nullptr);
    if (!g_prefetch_backend) {
        TIER_LOG("expert_tier: cannot create the async prefetch backend stream\n");
        return false;
    }

    g_async_jobs.resize((size_t) g_prefetch_max_inflight);
    for (auto & job : g_async_jobs) {
        job.host_buffer = ggml_backend_buft_alloc_buffer(host_buft, staging_size);
        job.event = ggml_backend_event_new(dev);
        if (!job.host_buffer || !job.event) {
            TIER_LOG("expert_tier: cannot allocate async prefetch staging/event resources\n");
            shutdown_async_resources();
            return false;
        }
        job.host_ptr = ggml_backend_buffer_get_base(job.host_buffer);
        job.capacity = ggml_backend_buffer_get_size(job.host_buffer);
    }

    g_async_stop = false;
    g_async_worker = std::thread(async_worker_main);
    g_prefetch_ready = true;
    atexit(shutdown_async_resources);
    TIER_LOG("expert_tier: async prefetch ready: streams=1 max_inflight=%d staging=%.2f MiB\n",
            g_prefetch_max_inflight, (double) staging_size/(1024.0*1024.0));
    return true;
}

static int reserve_async_job() {
    std::lock_guard<std::mutex> lock(g_async_mutex);
    for (size_t i = 0; i < g_async_jobs.size(); ++i) {
        if (!g_async_jobs[i].active) {
            g_async_jobs[i].active = true;
            g_async_jobs[i].completed = false;
            g_async_jobs[i].completed_us = 0;
            return (int) i;
        }
    }
    return -1;
}

static bool warm_expert_pending(const layer_tier & L, int expert) {
    return std::find(L.warm_pending.begin(), L.warm_pending.end(), expert) != L.warm_pending.end();
}

static void poll_async_completions() {
    struct completion {
        int layer;
        int slot;
        int expert;
        uint64_t bytes;
        int64_t started_us;
        int64_t completed_us;
    };
    std::vector<completion> completed;

    {
        std::lock_guard<std::mutex> lock(g_async_mutex);
        for (auto & job : g_async_jobs) {
            if (!job.active || !job.completed) {
                continue;
            }
            completed.push_back({job.layer, job.slot, job.expert, job.bytes, job.started_us, job.completed_us});
            job.active = false;
            job.completed = false;
            job.layer = -1;
            job.slot = -1;
            job.expert = -1;
            job.bytes = 0;
        }
    }

    std::vector<bool> changed(g_layers.size(), false);
    for (const auto & done : completed) {
        if (done.layer < 0 || done.layer >= (int) g_layers.size()) {
            TIER_LOG("expert_tier: FATAL async completion has invalid layer=%d\n", done.layer);
            abort();
        }
        layer_tier & L = g_layers[done.layer];
        const int warm_index = done.slot - L.warm_cache.n_fixed();
        if (warm_index < 0 || warm_index >= (int) L.warm_pending.size() ||
                L.warm_pending[warm_index] != done.expert) {
            TIER_LOG("expert_tier: FATAL async completion mismatch layer=%d expert=%d slot=%d request=%llu\n",
                    done.layer, done.expert, done.slot, (unsigned long long) g_request_id);
            abort();
        }
        try {
            L.warm_cache.bind_ready_warm_slot(done.slot, done.expert);
        } catch (const std::exception & error) {
            TIER_LOG("expert_tier: FATAL async publish layer=%d expert=%d slot=%d: %s\n",
                    done.layer, done.expert, done.slot, error.what());
            abort();
        }
        if (g_warm_admission != warm_admission_mode::immediate) {
            L.warm_admission.mark_resident(done.expert);
        }
        L.warm_pending[warm_index] = -1;
        L.lut_host = L.warm_cache.lut();
        changed[done.layer] = true;
        g_h2d_copy_us += std::max<int64_t>(0, done.completed_us - done.started_us);
    }

    for (size_t layer = 0; layer < changed.size(); ++layer) {
        if (changed[layer]) {
            upload_warm_maps(g_layers[layer]);
            warm_invariant_abort(g_layers[layer], "async_publish");
        }
    }
}

static bool schedule_async_copy(layer_tier & L, llama_expert_cache::state & next,
        int expert, const llama_expert_cache::insertion & target) {
    const int job_index = reserve_async_job();
    if (job_index < 0) {
        return false;
    }

    async_copy_job & job = g_async_jobs[(size_t) job_index];
    size_t staging_offset = 0;
    for (const auto & kv : L.ws) {
        const ggml_tensor * weight = kv.first;
        const size_t slice = ggml_nbytes(weight)/L.n_expert;
        if (staging_offset + slice > job.capacity) {
            TIER_LOG("expert_tier: FATAL async staging overflow layer=%d expert=%d need=%zu capacity=%zu\n",
                    L.il, expert, staging_offset + slice, job.capacity);
            abort();
        }
        memcpy((char *) job.host_ptr + staging_offset, (const char *) weight->data + slice*expert, slice);
        staging_offset += slice;
    }

    next.evict_warm_slot(target.slot);
    const int warm_index = target.slot - next.n_fixed();
    if (warm_index < 0 || warm_index >= (int) L.warm_pending.size() || L.warm_pending[warm_index] >= 0) {
        TIER_LOG("expert_tier: FATAL async reservation layer=%d expert=%d slot=%d\n", L.il, expert, target.slot);
        abort();
    }
    L.warm_pending[warm_index] = expert;

    staging_offset = 0;
    for (const auto & kv : L.ws) {
        const ggml_tensor * weight = kv.first;
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(weight)/L.n_expert;
        ggml_backend_tensor_set_async(g_prefetch_backend, st.w_hot,
                (const char *) job.host_ptr + staging_offset, slice*target.slot, slice);
        staging_offset += slice;
    }
    ggml_backend_event_record(job.event, g_prefetch_backend);

    {
        std::lock_guard<std::mutex> lock(g_async_mutex);
        job.layer = L.il;
        job.slot = target.slot;
        job.expert = expert;
        job.bytes = staging_offset;
        job.started_us = 0;
        g_async_queue.push_back((size_t) job_index);
    }
    g_async_cv.notify_one();

    g_h2d_copies++;
    g_h2d_bytes += staging_offset;
    return true;
}

void configure_mtp(int mtp_n) {
    if (mtp_n <= 0) {
        return;
    }
    const bool already_configured = g_mtp_n >= mtp_n;
    g_mtp_n = std::max(g_mtp_n, mtp_n);
    if (already_configured) {
        return;
    }
    if (warm_mtp_experimental()) {
        TIER_LOG("expert_tier: WARNING experimental warm cache with MTP-%d explicitly enabled; deterministic equivalence is not established\n",
                g_mtp_n);
        return;
    }

    g_warm_mtp_guarded = true;
    if (g_layers.empty()) {
        return;
    }

    // Context creation is a graph-free boundary. Drain dedicated transfers,
    // remove all warm ownership, and restore CPU fallback before an MTP
    // context can share the process-wide tier state.
    if (g_prefetch_ready) {
        shutdown_async_resources();
    }
    for (auto & L : g_layers) {
        if (!L.warm_enabled) {
            continue;
        }
        for (int slot = L.warm_cache.n_fixed(); slot < L.warm_cache.sentinel(); ++slot) {
            L.warm_cache.evict_warm_slot(slot);
        }
        std::fill(L.warm_pending.begin(), L.warm_pending.end(), -1);
        L.lut_host = L.warm_cache.lut();
        upload_warm_maps(L);
        warm_invariant_abort(L, "mtp_guard");
    }
    g_prefetch_requested = false;
    TIER_LOG("expert_tier: warm cache disabled for MTP-%d; set LLAMA_EXPERT_WARM_MTP_EXPERIMENTAL=1 only for controlled diagnostics\n",
            g_mtp_n);
}

static bool parse_heat_csv(const std::string & path, int n_layer, int n_expert,
        std::vector<std::vector<std::pair<int64_t, int32_t>>> & heat) {
    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_ERROR("%s: cannot open heat csv '%s'\n", __func__, path.c_str());
        return false;
    }
    heat.assign(n_layer, {});
    std::vector<std::vector<bool>> seen(n_layer, std::vector<bool>(n_expert, false));
    std::string line;
    if (!std::getline(in, line)) {
        LLAMA_LOG_ERROR("%s: heat csv '%s' is empty\n", __func__, path.c_str());
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line != "layer,expert,count") {
        LLAMA_LOG_ERROR("%s: heat csv '%s' has invalid header '%s'\n", __func__, path.c_str(), line.c_str());
        return false;
    }
    int line_no = 1;
    while (std::getline(in, line)) {
        line_no++;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' has an empty row at line %d\n", __func__, path.c_str(), line_no);
            return false;
        }
        if (line.back() == ',') {
            LLAMA_LOG_ERROR("%s: heat csv '%s' has an empty trailing column at line %d\n", __func__, path.c_str(), line_no);
            return false;
        }
        std::stringstream ss(line);
        std::string tok;
        std::vector<int64_t> vals;
        try {
            while (std::getline(ss, tok, ',')) {
                size_t used = 0;
                const int64_t value = std::stoll(tok, &used);
                if (used != tok.size()) {
                    throw std::invalid_argument("trailing characters");
                }
                vals.push_back(value);
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' has an invalid integer at line %d: %s\n", __func__, path.c_str(), line_no, e.what());
            return false;
        }
        if (vals.size() != 3) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' line %d has %zu columns, expected 3\n", __func__, path.c_str(), line_no, vals.size());
            return false;
        }
        if (vals[0] < 0 || vals[0] >= n_layer) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' line %d has layer %lld outside [0, %d)\n", __func__, path.c_str(), line_no, (long long) vals[0], n_layer);
            return false;
        }
        const int il = (int) vals[0];
        if (vals[1] < 0 || vals[1] >= n_expert) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' line %d has expert %lld outside [0, %d)\n", __func__, path.c_str(), line_no, (long long) vals[1], n_expert);
            return false;
        }
        const int expert = (int) vals[1];
        if (vals[2] < 0) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' line %d has negative count %lld\n", __func__, path.c_str(), line_no, (long long) vals[2]);
            return false;
        }
        if (seen[il][expert]) {
            LLAMA_LOG_ERROR("%s: heat csv '%s' line %d duplicates layer %d expert %d\n", __func__, path.c_str(), line_no, il, expert);
            return false;
        }
        seen[il][expert] = true;
        heat[il].emplace_back(vals[2], (int32_t) expert);
    }
    return true;
}

static void copy_expert_to_slot(const layer_tier & L, int expert, int slot) {
    const int64_t started = ggml_time_us();
    uint64_t copied = 0;
    for (const auto & kv : L.ws) {
        const ggml_tensor * w = kv.first;
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(w)/L.n_expert;
        ggml_backend_tensor_set(st.w_hot, (const char *) w->data + slice*expert, slice*slot, slice);
        copied += slice;
    }
    g_h2d_copies++;
    g_h2d_bytes += copied;
    g_h2d_copy_us += ggml_time_us() - started;
}

// Warm-cache updates are graph-granular: all counts from one completed graph
// are classified against one stable LUT. Up to W highest-frequency cold
// experts from that graph are admitted, and a newly written slot is protected
// from being replaced again in the same transaction.
static void maybe_update_warm(layer_tier & L) {
    if (!L.sd || !L.sd->counts->data) {
        return;
    }
    int32_t * cnt = (int32_t *) L.sd->counts->data;
    const int n = L.n_expert;
    if (cnt[n] == 0) {
        return;
    }

    static const float decay = [] {
        const char * e = getenv("LLAMA_EXPERT_DECAY");
        return e ? (float) atof(e) : 1.0f;
    }();

    const bool use_probation = g_warm_admission != warm_admission_mode::immediate;
    const bool use_frequency = g_warm_admission == warm_admission_mode::frequency;
    if (use_probation) {
        L.warm_admission.next_epoch();
    }
    const float frequency_decay = use_frequency ? powf(0.5f, 1.0f/(float) g_warm_admission_window) : 0.0f;

    std::vector<std::pair<int32_t, int>> cold_selected;
    for (int expert = 0; expert < n; ++expert) {
        const int32_t selected = cnt[expert];
        if (use_frequency) {
            L.warm_frequency[expert] = L.warm_frequency[expert]*frequency_decay + (float) selected;
        }
        L.score[expert] = L.score[expert]*decay + (float) selected;
        L.cum[expert] += (uint64_t) selected;
        L.observed[expert] += (uint64_t) selected;
        if (selected > 0) {
            switch (L.warm_cache.locate(expert)) {
                case llama_expert_cache::location::fixed:
                    L.cum_fixed += (uint64_t) selected;
                    if (use_probation) {
                        L.warm_admission.mark_resident(expert);
                    }
                    break;
                case llama_expert_cache::location::warm:
                    L.cum_warm += (uint64_t) selected;
                    L.warm_cache.touch_warm(expert);
                    if (use_probation) {
                        L.warm_admission.mark_resident(expert);
                    }
                    break;
                case llama_expert_cache::location::cold:
                    L.cum_cold += (uint64_t) selected;
                    if ((!use_probation || L.warm_admission.record_miss(expert)) &&
                            !warm_expert_pending(L, expert)) {
                        cold_selected.emplace_back(selected, expert);
                    } else {
                        L.warm_admission_deferrals++;
                        g_warm_admission_deferrals++;
                    }
                    break;
            }
        }
        cnt[expert] = 0;
    }
    L.cum_total += (uint64_t) cnt[n];
    L.cum_graphs++;
    cnt[n] = 0;

    llama_expert_cache::state next = L.warm_cache;
    bool maps_changed = false;

    if (g_adapt) {
        for (int slot = 0; slot < next.n_fixed(); ++slot) {
            L.dwell[slot]++;
        }

        int fixed_slot = -1;
        float fixed_score = FLT_MAX;
        for (int slot = 0; slot < next.n_fixed(); ++slot) {
            const int expert = next.expert_in_slot(slot);
            const float score = expert < 0 ? 0.0f : L.score[expert];
            if (score < fixed_score) {
                fixed_score = score;
                fixed_slot = slot;
            }
        }

        int candidate = -1;
        float candidate_score = 0.0f;
        for (int expert = 0; expert < n; ++expert) {
            if (next.locate(expert) != llama_expert_cache::location::fixed &&
                    !warm_expert_pending(L, expert) && L.score[expert] > candidate_score) {
                candidate_score = L.score[expert];
                candidate = expert;
            }
        }

        if (fixed_slot >= 0 && candidate >= 0) {
            const int incumbent = next.expert_in_slot(fixed_slot);
            if (incumbent < 0 || (L.dwell[fixed_slot] >= 32 && candidate_score > 1.5f*fixed_score)) {
                const bool from_warm = next.locate(candidate) == llama_expert_cache::location::warm;
                copy_expert_to_slot(L, candidate, fixed_slot);
                next.replace_fixed(fixed_slot, candidate);
                if (use_probation) {
                    L.warm_admission.mark_resident(candidate);
                }
                L.dwell[fixed_slot] = 0;
                g_repins++;
                if (from_warm) {
                    L.warm_to_fixed++;
                    g_warm_to_fixed++;
                }
                maps_changed = true;
            }
        }
    }

    if (g_warm_mtp_guarded) {
        if (maps_changed) {
            L.warm_cache = std::move(next);
            L.lut_host = L.warm_cache.lut();
            upload_warm_maps(L);
        }
        warm_invariant_abort(L, "mtp_guarded_update");
        return;
    }

    std::sort(cold_selected.begin(), cold_selected.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });

    std::vector<bool> protected_slots(next.n_slots(), false);
    for (size_t warm_index = 0; warm_index < L.warm_pending.size(); ++warm_index) {
        if (L.warm_pending[warm_index] >= 0) {
            protected_slots[(size_t) next.n_fixed() + warm_index] = true;
        }
    }
    int admitted = 0;
    for (const auto & selected : cold_selected) {
        if (admitted >= next.n_warm() || next.locate(selected.second) != llama_expert_cache::location::cold) {
            continue;
        }
        const auto target = next.insertion_target(protected_slots);
        if (!target) {
            continue;
        }
        if (use_frequency) {
            if (target.evicted >= 0 &&
                    L.warm_frequency[selected.second] <= L.warm_frequency[target.evicted]) {
                L.warm_admission_deferrals++;
                g_warm_admission_deferrals++;
                continue;
            }
        }
        if (g_prefetch_ready) {
            if (!schedule_async_copy(L, next, selected.second, target)) {
                break;
            }
            protected_slots[target.slot] = true;
            L.warm_promotions++;
            g_warm_promotions++;
            if (target.evicted >= 0) {
                L.warm_evictions++;
                g_warm_evictions++;
            }
            admitted++;
            maps_changed = true;
            continue;
        }
        const auto inserted = next.insert_warm(selected.second, protected_slots);
        if (!inserted) {
            continue;
        }
        copy_expert_to_slot(L, selected.second, inserted.slot);
        if (use_probation) {
            L.warm_admission.mark_resident(selected.second);
        }
        protected_slots[inserted.slot] = true;
        L.warm_promotions++;
        g_warm_promotions++;
        if (inserted.evicted >= 0) {
            L.warm_evictions++;
            g_warm_evictions++;
        }
        admitted++;
        maps_changed = true;
    }

    if (maps_changed) {
        L.warm_cache = std::move(next);
        L.lut_host = L.warm_cache.lut();
        upload_warm_maps(L);
    }
    warm_invariant_abort(L, "update");
}

// Legacy fixed-tier update. This path is intentionally unchanged when W=0.
// Counts are written by the fused cold op and consumed between compute steps.
static void maybe_update_fixed(layer_tier & L) {
    if (!L.sd || !L.sd->counts->data) {
        return;
    }
    int32_t * cnt = (int32_t *) L.sd->counts->data;
    const int n = L.n_expert;
    if (cnt[n] == 0) {
        return; // no compute since last visit
    }
    const int32_t * mask = (const int32_t *) L.sd->mask->data;

    static const float decay = [] {
        const char * e = getenv("LLAMA_EXPERT_DECAY");
        return e ? (float) atof(e) : 1.0f;
    }();

    for (int e = 0; e < n; e++) {
        L.score[e] = L.score[e]*decay + (float) cnt[e];
        L.cum[e]  += (uint64_t) cnt[e];
        L.observed[e] += (uint64_t) cnt[e];
        if (mask[e]) {
            L.cum_cold += (uint64_t) cnt[e];
        }
        cnt[e] = 0;
    }
    L.cum_total += (uint64_t) cnt[n];
    L.cum_graphs++;
    cnt[n] = 0;

    if (!g_adapt) {
        return;
    }

    for (int s = 0; s < L.n_slots; s++) {
        L.dwell[s]++;
    }

    // coldest hot slot (empty slots count as score 0)
    int si = -1;
    float smin = FLT_MAX;
    for (int s = 0; s < L.n_slots; s++) {
        if (s == L.sentinel) {
            continue;
        }
        const float v = L.slot_expert[s] < 0 ? 0.0f : L.score[L.slot_expert[s]];
        if (v < smin) {
            smin = v;
            si = s;
        }
    }
    // hottest cold expert
    int ec = -1;
    float sc = 0.0f;
    for (int e = 0; e < n; e++) {
        if (mask[e] && L.score[e] > sc) {
            sc = L.score[e];
            ec = e;
        }
    }
    if (si < 0 || ec < 0) {
        return;
    }
    const int eold = L.slot_expert[si];
    if (eold >= 0 && (L.dwell[si] < 32 || sc <= 1.5f*smin)) {
        return; // hysteresis: keep the incumbent
    }

    L.lut_host[ec] = si;
    if (eold >= 0) {
        L.lut_host[eold] = L.sentinel;
    }
    for (auto & kv : L.ws) {
        const ggml_tensor * w = kv.first;
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(w)/n;
        ggml_backend_tensor_set(st.w_hot, (const char *) w->data + slice*ec, slice*si, slice);
        ggml_backend_tensor_set(st.lut, L.lut_host.data(), 0, n*sizeof(int32_t));
        int32_t * m = (int32_t *) st.mask->data;
        m[ec] = 0;
        if (eold >= 0) {
            m[eold] = 1;
        }
    }
    L.slot_expert[si] = ec;
    L.dwell[si] = 0;
    g_repins++;
}

static void write_json_string(FILE * f, const std::string & value) {
    fputc('"', f);
    for (unsigned char c : value) {
        switch (c) {
            case '"': fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\b': fputs("\\b", f); break;
            case '\f': fputs("\\f", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (c < 0x20) {
                    fprintf(f, "\\u%04x", c);
                } else {
                    fputc(c, f);
                }
                break;
        }
    }
    fputc('"', f);
}

static void dump_stats() {
    if (g_cpu_async) {
        uint64_t jobs = 0;
        uint64_t wait_us = 0;
        ggml_cpu_moe_async_stats_get(&jobs, &wait_us);
        TIER_LOG("expert_tier: CPU async cold jobs %llu join_wait_ms %.3f\n",
                (unsigned long long) jobs, (double) wait_us/1000.0);
    }
    if (output_env_enabled("LLAMA_EXPERT_STATS")) {
        const char * p = getenv("LLAMA_EXPERT_STATS");
        FILE * f = strcmp(p, "1") ? fopen(p, "w") : stderr;
        if (f) {
            fprintf(f, "expert_tier: repins %llu\n", (unsigned long long) g_repins);
            fprintf(f, "expert_tier: warm_admissions %llu warm_evictions %llu warm_to_fixed %llu h2d_copies %llu h2d_bytes %llu h2d_ms %.3f\n",
                    (unsigned long long) g_warm_promotions, (unsigned long long) g_warm_evictions,
                    (unsigned long long) g_warm_to_fixed, (unsigned long long) g_h2d_copies,
                    (unsigned long long) g_h2d_bytes, (double) g_h2d_copy_us/1000.0);
            fprintf(f, "expert_tier: warm_admission_deferrals %llu\n",
                    (unsigned long long) g_warm_admission_deferrals);
            for (const auto & L : g_layers) {
                if (L.ws.empty() || L.cum_total == 0) {
                    continue;
                }
                if (L.warm_enabled) {
                    fprintf(f, "layer %2d: fixed %llu warm %llu cold %llu / %llu (%.1f%%) graphs %llu warm_admissions %llu evictions %llu warm_to_fixed %llu\n",
                            L.il, (unsigned long long) L.cum_fixed, (unsigned long long) L.cum_warm,
                            (unsigned long long) L.cum_cold, (unsigned long long) L.cum_total,
                            100.0*(double) L.cum_cold/(double) L.cum_total,
                            (unsigned long long) L.cum_graphs, (unsigned long long) L.warm_promotions,
                            (unsigned long long) L.warm_evictions, (unsigned long long) L.warm_to_fixed);
                } else {
                    fprintf(f, "layer %2d: cold %llu / %llu (%.1f%%) graphs %llu\n", L.il,
                            (unsigned long long) L.cum_cold, (unsigned long long) L.cum_total,
                            100.0*(double) L.cum_cold/(double) L.cum_total,
                            (unsigned long long) L.cum_graphs);
                }
                if (g_timing_enabled) {
                    uint64_t runs = 0;
                    uint64_t wall_us = 0;
                    uint64_t gate_up_us = 0;
                    uint64_t activation_us = 0;
                    uint64_t down_us = 0;
                    ggml_cpu_moe_profile_get(L.il, &runs, &wall_us);
                    ggml_cpu_moe_profile_get_phases(L.il, &gate_up_us, &activation_us, &down_us);
                    fprintf(f, "layer %2d: cpu_cold_ms %.3f cpu_gate_up_ms %.3f cpu_activation_ms %.3f cpu_down_ms %.3f cpu_cold_runs %llu\n", L.il,
                            (double) wall_us/1000.0, (double) gate_up_us/1000.0,
                            (double) activation_us/1000.0, (double) down_us/1000.0,
                            (unsigned long long) runs);
                }
            }
            // invariant: mask agrees with lut, slot<->expert bijection holds
            for (const auto & L : g_layers) {
                if (!L.sd) {
                    continue;
                }
                if (L.warm_enabled) {
                    warm_invariant_abort(L, "stats");
                    continue;
                }
                const int32_t * m = (const int32_t *) L.sd->mask->data;
                int bad = 0;
                for (int e = 0; e < L.n_expert; e++) {
                    const int want = (L.lut_host[e] == L.sentinel) ? 1 : 0;
                    if (m[e] != want) {
                        bad++;
                    }
                }
                for (int s = 0; s < L.n_slots; s++) {
                    if (s == L.sentinel || L.slot_expert[s] < 0) {
                        continue;
                    }
                    if (L.lut_host[L.slot_expert[s]] != s) {
                        bad++;
                    }
                }
                if (bad) {
                    fprintf(f, "layer %2d: INVARIANT VIOLATIONS %d\n", L.il, bad);
                }
            }
            if (f != stderr) {
                fclose(f);
            }
        }
    }
    if (output_env_enabled("LLAMA_EXPERT_USAGE")) {
        const char * p = getenv("LLAMA_EXPERT_USAGE");
        FILE * f = fopen(p, "w");
        if (f) {
            fprintf(f, "layer,expert,count\n");
            for (const auto & L : g_layers) {
                const auto & usage = g_usage_mode == expert_usage_mode::session ? L.observed : L.cum;
                if (usage.empty()) {
                    continue;
                }
                for (int e = 0; e < L.n_expert; e++) {
                    if (usage[e] > 0) {
                        fprintf(f, "%d,%d,%llu\n", L.il, e, (unsigned long long) usage[e]);
                    }
                }
            }
            fclose(f);
        }
    }
    if (output_env_enabled("LLAMA_EXPERT_STATS_JSON")) {
        const char * p = getenv("LLAMA_EXPERT_STATS_JSON");
        FILE * f = fopen(p, "w");
        if (!f) {
            TIER_LOG("expert_tier: cannot write JSON stats '%s': %s\n", p, strerror(errno));
            return;
        }
        uint64_t selected_total = 0;
        uint64_t fixed_hits = 0;
        uint64_t warm_hits = 0;
        uint64_t cold_hits = 0;
        uint64_t cpu_cold_us = 0;
        uint64_t cpu_gate_up_us = 0;
        uint64_t cpu_activation_us = 0;
        uint64_t cpu_down_us = 0;
        uint64_t cpu_async_jobs = 0;
        uint64_t cpu_async_wait_us = 0;
        ggml_cpu_moe_async_stats_get(&cpu_async_jobs, &cpu_async_wait_us);
        int hot_slots_total = 0;
        int hot_slots_min = INT32_MAX;
        int hot_slots_max = 0;
        for (const auto & L : g_layers) {
            if (L.ws.empty()) {
                continue;
            }
            selected_total += L.cum_total;
            fixed_hits += L.warm_enabled ? L.cum_fixed : L.cum_total - L.cum_cold;
            warm_hits += L.cum_warm;
            cold_hits += L.cum_cold;
            uint64_t runs = 0;
            uint64_t wall_us = 0;
            ggml_cpu_moe_profile_get(L.il, &runs, &wall_us);
            cpu_cold_us += wall_us;
            uint64_t gate_up_us = 0;
            uint64_t activation_us = 0;
            uint64_t down_us = 0;
            ggml_cpu_moe_profile_get_phases(L.il, &gate_up_us, &activation_us, &down_us);
            cpu_gate_up_us += gate_up_us;
            cpu_activation_us += activation_us;
            cpu_down_us += down_us;
            hot_slots_total += L.n_fixed;
            hot_slots_min = std::min(hot_slots_min, L.n_fixed);
            hot_slots_max = std::max(hot_slots_max, L.n_fixed);
        }
        if (hot_slots_min == INT32_MAX) {
            hot_slots_min = 0;
        }
        fputs("{\n  \"model\": ", f);
        write_json_string(f, g_model_desc);
        fprintf(f,
                ",\n  \"context\": 0,\n  \"mtp_n\": %d,\n  \"timing_enabled\": %s,\n  \"layers\": [",
                g_mtp_n, g_timing_enabled ? "true" : "false");
        bool first = true;
        for (const auto & L : g_layers) {
            if (L.ws.empty()) {
                continue;
            }
            uint64_t cpu_cold_runs = 0;
            uint64_t layer_cpu_cold_us = 0;
            uint64_t layer_gate_up_us = 0;
            uint64_t layer_activation_us = 0;
            uint64_t layer_down_us = 0;
            ggml_cpu_moe_profile_get(L.il, &cpu_cold_runs, &layer_cpu_cold_us);
            ggml_cpu_moe_profile_get_phases(L.il, &layer_gate_up_us, &layer_activation_us, &layer_down_us);
            fprintf(f, "%s\n    {\"layer\": %d, \"fixed_slots\": %d, \"selected_total\": %llu, \"hot_hits\": %llu, \"warm_hits\": %llu, \"cold_hits\": %llu, \"warm_promotions\": %llu, \"warm_evictions\": %llu, \"cpu_cold_runs\": %llu, \"cpu_cold_ms\": %.3f, \"cpu_gate_up_ms\": %.3f, \"cpu_activation_ms\": %.3f, \"cpu_down_ms\": %.3f}",
                    first ? "" : ",", L.il, L.n_fixed, (unsigned long long) L.cum_total,
                    (unsigned long long) (L.warm_enabled ? L.cum_fixed : L.cum_total - L.cum_cold),
                    (unsigned long long) L.cum_warm, (unsigned long long) L.cum_cold,
                    (unsigned long long) L.warm_promotions, (unsigned long long) L.warm_evictions,
                    (unsigned long long) cpu_cold_runs, (double) layer_cpu_cold_us/1000.0,
                    (double) layer_gate_up_us/1000.0, (double) layer_activation_us/1000.0,
                    (double) layer_down_us/1000.0);
            first = false;
        }
        fprintf(f,
                "\n  ],\n"
                "  \"hot_slots_per_layer\": %d,\n"
                "  \"hot_slots_total\": %d,\n"
                "  \"hot_slots_min\": %d,\n"
                "  \"hot_slots_max\": %d,\n"
                "  \"warm_slots_per_layer\": %d,\n"
                "  \"cpu_single_row_chunk\": %d,\n"
                "  \"cpu_parallel_activation\": %s,\n"
                "  \"cpu_down_prefetch\": %d,\n"
                "  \"cpu_reuse_rows\": %s,\n"
                "  \"cpu_async\": %s,\n"
                "  \"cpu_async_jobs\": %llu,\n"
                "  \"cpu_async_wait_ms\": %.3f,\n"
                "  \"selected_total\": %llu,\n"
                "  \"hot_hits\": %llu,\n"
                "  \"warm_hits\": %llu,\n"
                "  \"cold_hits\": %llu,\n"
                "  \"repins\": %llu,\n"
                "  \"warm_promotions\": %llu,\n"
                "  \"warm_evictions\": %llu,\n"
                "  \"warm_admission_deferrals\": %llu,\n"
                "  \"h2d_copies\": %llu,\n"
                "  \"h2d_bytes\": %llu,\n"
                "  \"h2d_copy_ms\": %.3f,\n"
                "  \"cpu_expert_ms\": %.3f,\n"
                "  \"cpu_gate_up_ms\": %.3f,\n"
                "  \"cpu_activation_ms\": %.3f,\n"
                "  \"cpu_down_ms\": %.3f,\n"
                "  \"gpu_expert_ms\": 0.0,\n"
                "  \"sync_wait_ms\": 0.0,\n"
                "  \"decode_tokens\": 0,\n"
                "  \"decode_tokens_per_second\": 0.0\n"
                "}\n",
                g_variable_placement ? -1 : g_S, hot_slots_total, hot_slots_min, hot_slots_max,
                g_W, g_cpu_single_row_chunk, g_cpu_parallel_activation ? "true" : "false",
                g_cpu_down_prefetch, g_cpu_reuse_rows ? "true" : "false",
                g_cpu_async ? "true" : "false", (unsigned long long) cpu_async_jobs,
                (double) cpu_async_wait_us/1000.0,
                (unsigned long long) selected_total, (unsigned long long) fixed_hits,
                (unsigned long long) warm_hits, (unsigned long long) cold_hits,
                (unsigned long long) g_repins, (unsigned long long) g_warm_promotions,
                (unsigned long long) g_warm_evictions, (unsigned long long) g_warm_admission_deferrals,
                (unsigned long long) g_h2d_copies,
                (unsigned long long) g_h2d_bytes, (double) g_h2d_copy_us/1000.0,
                (double) cpu_cold_us/1000.0, (double) cpu_gate_up_us/1000.0,
                (double) cpu_activation_us/1000.0, (double) cpu_down_us/1000.0);
        fclose(f);
    }
}

void update() {
    if (g_static_no_sync_active) {
        return;
    }
    if (g_prefetch_ready) {
        // update() is called only after the just-submitted graph has completed.
        // Publishing a finished slot here keeps one LUT stable for the entire
        // graph, including a multi-token MTP verification graph.
        poll_async_completions();
    }
    if (g_layers.empty()) {
        return;
    }

    const size_t begin = g_prefetch_ready ? g_async_layer_cursor % g_layers.size() : 0;
    for (size_t offset = 0; offset < g_layers.size(); ++offset) {
        auto & L = g_layers[(begin + offset) % g_layers.size()];
        if (L.warm_enabled) {
            maybe_update_warm(L);
        } else {
            maybe_update_fixed(L);
        }
    }
    if (g_prefetch_ready) {
        // A global in-flight cap would otherwise always favor the first model
        // layers. Rotate admission priority without skipping count harvesting.
        g_async_layer_cursor = (begin + 1) % g_layers.size();
    }
}

bool requires_post_graph_sync() {
    return !g_static_no_sync_active;
}

void request_begin() {
    g_request_id++;
    if (!g_warm_reset_request) {
        return;
    }
    for (auto & L : g_layers) {
        if (L.warm_enabled) {
            L.warm_cache.reset_warm_ages();
            if (g_warm_admission != warm_admission_mode::immediate) {
                L.warm_admission.reset(L.n_expert, (uint64_t) g_warm_admission_window);
            }
            if (g_warm_admission == warm_admission_mode::frequency) {
                std::fill(L.warm_frequency.begin(), L.warm_frequency.end(), 0.0f);
            }
            warm_invariant_abort(L, "request_begin");
        }
    }
}

size_t expert_weight_bytes(const llama_model & model) {
    size_t b = 0;
    for (int il = 0; il < (int) model.hparams.n_layer(); il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w) {
                b += ggml_nbytes(w);
            }
        }
    }
    return b;
}

void configure_enabled(bool enabled) {
    g_configured_enabled = enabled;
}

static bool explicitly_requested_by_env() {
    static const char * controls[] = {
        "LLAMA_EXPERT_HOT",
        "LLAMA_EXPERT_S",
        "LLAMA_EXPERT_PLACEMENT",
        "LLAMA_EXPERT_ADAPT",
        "LLAMA_EXPERT_STATS",
        "LLAMA_EXPERT_USAGE",
        "LLAMA_EXPERT_STATS_JSON",
        "LLAMA_EXPERT_TIMING",
        "LLAMA_EXPERT_CPU_CHUNK",
        "LLAMA_EXPERT_CPU_ACT_PARALLEL",
        "LLAMA_EXPERT_CPU_ASYNC",
        "LLAMA_EXPERT_CPU_DOWN_PREFETCH",
        "LLAMA_EXPERT_CPU_REUSE_ROWS",
        "LLAMA_EXPERT_WARM_SLOTS",
        "LLAMA_EXPERT_STATIC_NO_SYNC",
    };
    for (const char * name : controls) {
        if (getenv(name) != nullptr) {
            return true;
        }
    }
    return false;
}

void init(const llama_model & model) {
    // libllama's architecture and state tests create several unrelated model
    // contexts in one process. The tier owns process-wide caches, so it must
    // not attach implicitly to those direct API contexts. CLI/common users
    // keep the zero-config -cmoe default via configure_enabled(true), while an
    // explicit LLAMA_EXPERT_* control remains a supported direct opt-in.
    if (!g_configured_enabled && !explicitly_requested_by_env()) {
        return;
    }
    if (!g_layers.empty()) {
        return; // already initialized
    }
    char model_desc[256] = {};
    llama_model_desc(&model, model_desc, sizeof(model_desc));
    g_model_desc = model_desc;

    const char * env_s   = getenv("LLAMA_EXPERT_S");
    const char * env_hot = getenv("LLAMA_EXPERT_HOT");
    const char * env_placement = getenv("LLAMA_EXPERT_PLACEMENT");
    if (const char * e = getenv("LLAMA_EXPERT_ADAPT")) {
        g_adapt = atoi(e) != 0;
    } else {
        g_adapt = true; // Auto-fit and online adaptation ON by default
    }
    if (const char * e = getenv("LLAMA_EXPERT_STATIC_NO_SYNC")) {
        int value = 0;
        if (!parse_nonnegative_int(e, value) || value > 1) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_STATIC_NO_SYNC='%s'; static no-sync disabled\n", __func__, e);
        } else {
            g_static_no_sync_requested = value == 1;
        }
    }

    bool manual_S = false;
    if (env_s) {
        g_S = std::max(0, atoi(env_s));
        manual_S = (env_s != nullptr);
    }

    if (const char * e = getenv("LLAMA_EXPERT_TMAX")) {
        g_tmax = std::max(0, atoi(e));
    }
    if (const char * e = getenv("LLAMA_EXPERT_VRAM_RESERVE_MIB")) {
        if (!parse_nonnegative_int(e, g_vram_reserve_mib) || g_vram_reserve_mib > 65536) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_VRAM_RESERVE_MIB='%s'; expected 0..65536\n", __func__, e);
            return;
        }
    }

    bool warm_requested = false;
    if (const char * e = getenv("LLAMA_EXPERT_WARM_SLOTS")) {
        if (!strcmp(e, "auto")) {
            g_warm_auto = true;
            warm_requested = true;
        } else if (!parse_nonnegative_int(e, g_W)) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_WARM_SLOTS='%s'; warm cache disabled\n", __func__, e);
            g_W = 0;
        } else {
            warm_requested = g_W > 0;
        }
    }
    if (warm_requested) {
        if (const char * e = getenv("LLAMA_EXPERT_WARM_AUTO_MAX")) {
            if (!parse_nonnegative_int(e, g_warm_auto_max) || g_warm_auto_max < 1 || g_warm_auto_max > 4096) {
                TIER_LOG("%s: invalid LLAMA_EXPERT_WARM_AUTO_MAX='%s'; expected 1..4096\n", __func__, e);
                return;
            }
        }
        const char * policy = getenv("LLAMA_EXPERT_WARM_POLICY");
        if (policy && strcmp(policy, "lru")) {
            TIER_LOG("%s: unsupported LLAMA_EXPERT_WARM_POLICY='%s'; warm cache disabled\n", __func__, policy);
            warm_requested = false;
            g_warm_auto = false;
            g_W = 0;
        }
        const char * reset = getenv("LLAMA_EXPERT_WARM_RESET");
        if (reset && strcmp(reset, "request") && strcmp(reset, "persistent")) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_WARM_RESET='%s'; warm cache disabled\n", __func__, reset);
            warm_requested = false;
            g_warm_auto = false;
            g_W = 0;
        } else {
            g_warm_reset_request = !reset || !strcmp(reset, "request");
        }
        const char * admission = getenv("LLAMA_EXPERT_WARM_ADMISSION");
        if (!admission || !strcmp(admission, "immediate")) {
            g_warm_admission = warm_admission_mode::immediate;
        } else if (!strcmp(admission, "second-hit")) {
            g_warm_admission = warm_admission_mode::second_hit;
        } else if (!strcmp(admission, "frequency")) {
            g_warm_admission = warm_admission_mode::frequency;
        } else {
            TIER_LOG("%s: unsupported LLAMA_EXPERT_WARM_ADMISSION='%s'; warm cache disabled\n", __func__, admission);
            warm_requested = false;
            g_warm_auto = false;
            g_W = 0;
        }
        if (const char * e = getenv("LLAMA_EXPERT_WARM_ADMISSION_WINDOW")) {
            if (!parse_nonnegative_int(e, g_warm_admission_window) || g_warm_admission_window == 0) {
                TIER_LOG("%s: invalid LLAMA_EXPERT_WARM_ADMISSION_WINDOW='%s'; warm cache disabled\n", __func__, e);
                warm_requested = false;
                g_warm_auto = false;
                g_W = 0;
            }
        }
        int prefetch = 0;
        if (const char * e = getenv("LLAMA_EXPERT_WARM_PREFETCH")) {
            if (!parse_nonnegative_int(e, prefetch) || prefetch > 1) {
                TIER_LOG("%s: invalid LLAMA_EXPERT_WARM_PREFETCH='%s'; warm cache disabled\n", __func__, e);
                warm_requested = false;
                g_warm_auto = false;
                g_W = 0;
            }
        }
        g_prefetch_requested = warm_requested && prefetch == 1;
        if (g_prefetch_requested) {
            if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_STREAMS")) {
                if (!parse_nonnegative_int(e, g_prefetch_streams) || g_prefetch_streams != 1) {
                    TIER_LOG("%s: LLAMA_EXPERT_PREFETCH_STREAMS='%s' is unsupported; exactly one stream is required; warm cache disabled\n",
                            __func__, e);
                    warm_requested = false;
                    g_warm_auto = false;
                    g_W = 0;
                }
            }
            if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT")) {
                if (!parse_nonnegative_int(e, g_prefetch_max_inflight) ||
                        g_prefetch_max_inflight < 1 || g_prefetch_max_inflight > 64) {
                    TIER_LOG("%s: invalid LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT='%s'; expected 1..64; warm cache disabled\n",
                            __func__, e);
                    warm_requested = false;
                    g_warm_auto = false;
                    g_W = 0;
                }
            }
            if (!warm_requested) {
                g_prefetch_requested = false;
            }
        } else if (getenv("LLAMA_EXPERT_PREFETCH_STREAMS") || getenv("LLAMA_EXPERT_PREFETCH_MAX_INFLIGHT")) {
            TIER_LOG("%s: prefetch stream/inflight settings ignored because LLAMA_EXPERT_WARM_PREFETCH is not 1\n", __func__);
        }
        if (!warm_requested) {
            g_prefetch_requested = false;
        }
    }
    const bool warm_config_requested = warm_requested;
    if (warm_requested && g_mtp_n > 0 && g_warm_mtp_guarded) {
        TIER_LOG("%s: warm cache guarded off for MTP-%d after deterministic mismatch; fixed hot tier remains enabled\n",
                __func__, g_mtp_n);
        warm_requested = false;
        g_warm_auto = false;
        g_W = 0;
        g_prefetch_requested = false;
    }
    const bool stats_enabled = output_env_enabled("LLAMA_EXPERT_STATS");
    const bool usage_enabled = output_env_enabled("LLAMA_EXPERT_USAGE");
    const bool json_stats_enabled = output_env_enabled("LLAMA_EXPERT_STATS_JSON");
    if (const char * timing = getenv("LLAMA_EXPERT_TIMING")) {
        int value = 0;
        if (!parse_nonnegative_int(timing, value) || value > 1) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_TIMING='%s'; expected 0 or 1\n", __func__, timing);
            return;
        }
        g_timing_enabled = value == 1;
    }
    if (const char * chunk = getenv("LLAMA_EXPERT_CPU_CHUNK")) {
        if (!parse_nonnegative_int(chunk, g_cpu_single_row_chunk) ||
                g_cpu_single_row_chunk < 16 || g_cpu_single_row_chunk > 256 ||
                (g_cpu_single_row_chunk & (g_cpu_single_row_chunk - 1)) != 0) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_CPU_CHUNK='%s'; expected a power of two in 16..256\n",
                    __func__, chunk);
            return;
        }
    }
    if (const char * parallel = getenv("LLAMA_EXPERT_CPU_ACT_PARALLEL")) {
        int value = 0;
        if (!parse_nonnegative_int(parallel, value) || value > 1) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_CPU_ACT_PARALLEL='%s'; expected 0 or 1\n",
                    __func__, parallel);
            return;
        }
        g_cpu_parallel_activation = value == 1;
    }
    if (const char * async = getenv("LLAMA_EXPERT_CPU_ASYNC")) {
        int value = 0;
        if (!parse_nonnegative_int(async, value) || value > 1) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_CPU_ASYNC='%s'; expected 0 or 1\n",
                    __func__, async);
            return;
        }
        g_cpu_async = value == 1;
    }
    if (const char * prefetch = getenv("LLAMA_EXPERT_CPU_DOWN_PREFETCH")) {
        if (!parse_nonnegative_int(prefetch, g_cpu_down_prefetch) || g_cpu_down_prefetch > 8) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_CPU_DOWN_PREFETCH='%s'; expected 0..8\n",
                    __func__, prefetch);
            return;
        }
    }
    if (const char * reuse = getenv("LLAMA_EXPERT_CPU_REUSE_ROWS")) {
        int value = 0;
        if (!parse_nonnegative_int(reuse, value) || value > 1) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_CPU_REUSE_ROWS='%s'; expected 0 or 1\n",
                    __func__, reuse);
            return;
        }
        g_cpu_reuse_rows = value == 1;
    }
    ggml_cpu_moe_set_single_row_chunk(g_cpu_single_row_chunk);
    ggml_cpu_moe_set_parallel_activation(g_cpu_parallel_activation);
    ggml_cpu_moe_set_down_prefetch(g_cpu_down_prefetch);
    ggml_cpu_moe_set_reuse_rows(g_cpu_reuse_rows);
    ggml_cpu_moe_set_async(g_cpu_async);
    ggml_cpu_moe_async_stats_reset();
    ggml_cpu_moe_profile_reset();
    ggml_cpu_moe_profile_enable(g_timing_enabled);
    if (g_timing_enabled && !stats_enabled && !json_stats_enabled) {
        TIER_LOG("%s: LLAMA_EXPERT_TIMING is enabled without a stats output path\n", __func__);
    }
    if (const char * mode = getenv("LLAMA_EXPERT_USAGE_MODE")) {
        if (!strcmp(mode, "cumulative")) {
            g_usage_mode = expert_usage_mode::cumulative;
        } else if (!strcmp(mode, "session")) {
            g_usage_mode = expert_usage_mode::session;
        } else {
            TIER_LOG("%s: invalid LLAMA_EXPERT_USAGE_MODE='%s'; expected cumulative or session\n", __func__, mode);
            return;
        }
        if (!usage_enabled) {
            TIER_LOG("%s: LLAMA_EXPERT_USAGE_MODE is ignored because LLAMA_EXPERT_USAGE is disabled\n", __func__);
        }
    }

    const int n_layer  = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;

    std::vector<std::vector<std::pair<int64_t, int32_t>>> heat;
    if (env_hot && env_hot[0] && !parse_heat_csv(env_hot, n_layer, n_expert, heat)) {
        return;
    }
    if (env_placement && env_placement[0]) {
        if (!env_hot || !env_hot[0]) {
            TIER_LOG("%s: LLAMA_EXPERT_PLACEMENT requires LLAMA_EXPERT_HOT for expert rankings\n", __func__);
            return;
        }
        if (manual_S) {
            TIER_LOG("%s: LLAMA_EXPERT_PLACEMENT and LLAMA_EXPERT_S are mutually exclusive\n", __func__);
            return;
        }
        if (g_adapt || warm_config_requested) {
            TIER_LOG("%s: variable placement currently requires LLAMA_EXPERT_ADAPT=0 and no warm slots\n", __func__);
            return;
        }
        std::string placement_error;
        if (!llama_expert_placement::parse_manifest(
                    env_placement, n_layer, n_expert, g_placement, placement_error)) {
            TIER_LOG("%s: invalid LLAMA_EXPERT_PLACEMENT: %s\n", __func__, placement_error.c_str());
            return;
        }
        if (g_placement.model_architecture != model.arch_name()) {
            TIER_LOG("%s: placement architecture '%s' does not match loaded model '%s'\n",
                    __func__, g_placement.model_architecture.c_str(), model.arch_name().c_str());
            return;
        }
        std::string profile_digest;
        if (!llama_expert_placement::sha256_file(env_hot, profile_digest, placement_error)) {
            TIER_LOG("%s: cannot validate placement profile: %s\n", __func__, placement_error.c_str());
            return;
        }
        if (profile_digest != g_placement.profile_sha256) {
            TIER_LOG("%s: placement profile SHA-256 %s does not match LLAMA_EXPERT_HOT %s\n",
                    __func__, g_placement.profile_sha256.c_str(), profile_digest.c_str());
            return;
        }
        g_variable_placement = true;
    }
    if (stats_enabled || usage_enabled || json_stats_enabled || g_cpu_async) {
        atexit(dump_stats);
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        TIER_LOG("%s: expert tiering disabled (no GPU device found)\n", __func__);
        return;
    }

    // Calculate per-expert slot memory size across all layers for Auto-Fit
    size_t bytes_per_slot_all_layers = 0;
    std::vector<size_t> bytes_per_slot_by_layer(n_layer, 0);
    size_t max_layer_staging_bytes = 0;
    int n_tensors = 0;
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        size_t layer_staging_bytes = 0;
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w && ggml_backend_buffer_is_host(w->buffer)) {
                n_tensors++;
                if (n_expert > 0) {
                    const size_t slice = ggml_nbytes(w) / n_expert;
                    bytes_per_slot_all_layers += slice;
                    bytes_per_slot_by_layer[il] += slice;
                    if (w->ne[3] == 1 && (int) w->ne[2] == n_expert) {
                        layer_staging_bytes += slice;
                    }
                }
            }
        }
        max_layer_staging_bytes = std::max(max_layer_staging_bytes, layer_staging_bytes);
    }

    if (n_tensors == 0) {
        TIER_LOG("%s: expert tiering disabled (no host expert weights found)\n", __func__);
        return;
    }

    // Reserve memory for the CUDA runtime, display, alignment and graph-capture
    // buffers that allocate after this sizing. The default is 512 MiB because
    // 300 MiB caused capture OOM on the original 6 GiB test card.
    const size_t safety_buffer = (size_t) g_vram_reserve_mib*1024*1024;
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    const size_t usable_vram = (free_vram > safety_buffer) ? (free_vram - safety_buffer) : 0;

    if (g_variable_placement) {
        uint64_t fixed_bytes = 0;
        uint64_t sentinel_bytes = 0;
        int min_slots = n_expert;
        int max_slots = 0;
        for (int il = 0; il < n_layer; ++il) {
            const auto & config = g_placement.layers[il];
            if (config.slot_bytes != bytes_per_slot_by_layer[il]) {
                TIER_LOG("%s: placement layer %d slot bytes %llu do not match loaded tensor bytes %zu\n",
                        __func__, il, (unsigned long long) config.slot_bytes, bytes_per_slot_by_layer[il]);
                return;
            }
            fixed_bytes += (uint64_t) config.fixed_slots*config.slot_bytes;
            sentinel_bytes += config.slot_bytes;
            min_slots = std::min(min_slots, config.fixed_slots);
            max_slots = std::max(max_slots, config.fixed_slots);
        }
        if (fixed_bytes != g_placement.fixed_bytes_used || sentinel_bytes != g_placement.sentinel_bytes) {
            TIER_LOG("%s: placement byte totals changed after model load\n", __func__);
            return;
        }
        const uint64_t required = fixed_bytes + sentinel_bytes;
        // Match the established fixed-S autofit contract: the configured hot
        // budget must fit after the safety reserve, while the one sentinel per
        // layer must at least fit in actual free memory. This preserves exact
        // byte parity with a proven uniform S configuration. Report when the
        // sentinels consume part of the conservative reserve.
        if (fixed_bytes > usable_vram || required > free_vram) {
            TIER_LOG("%s: variable placement requires %.2f MiB fixed + %.2f MiB sentinels; %.2f MiB fixed-budget and %.2f MiB physical free VRAM are available\n",
                    __func__, (double) fixed_bytes/(1024.0*1024.0),
                    (double) sentinel_bytes/(1024.0*1024.0),
                    (double) usable_vram/(1024.0*1024.0),
                    (double) free_vram/(1024.0*1024.0));
            return;
        }
        if (required > usable_vram) {
            TIER_LOG("%s: variable placement sentinels consume %.2f MiB of the conservative VRAM reserve\n",
                    __func__, (double) (required - usable_vram)/(1024.0*1024.0));
        }
        g_S = max_slots;
        TIER_LOG("%s: variable placement validated: %d total fixed slots, range %d..%d, %.2f MiB fixed + %.2f MiB sentinels\n",
                __func__, std::accumulate(g_placement.layers.begin(), g_placement.layers.end(), 0,
                    [](int total, const auto & layer) { return total + layer.fixed_slots; }),
                min_slots, max_slots, (double) fixed_bytes/(1024.0*1024.0),
                (double) sentinel_bytes/(1024.0*1024.0));
    }

    if (!g_variable_placement && !manual_S && bytes_per_slot_all_layers > 0 && !warm_requested) {
        if (usable_vram >= bytes_per_slot_all_layers) {
            int autofit_s = (int) (usable_vram / bytes_per_slot_all_layers);
            g_S = std::clamp(autofit_s, 1, n_expert);
            TIER_LOG("%s: AUTO-FIT ENGINE -> set S = %d (Free VRAM: %.2f / %.2f GB, per-slot size: %.2f MB, total experts: %d)\n",
                           __func__, g_S, (double)free_vram / (1024.0*1024.0*1024.0),
                           (double)total_vram / (1024.0*1024.0*1024.0),
                           (double)bytes_per_slot_all_layers / (1024.0*1024.0), n_expert);
        } else {
            g_S = 0;
            TIER_LOG("%s: AUTO-FIT ENGINE -> Insufficient free VRAM (%.2f MB free vs %.2f MB required per slot), expert tiering on GPU disabled\n",
                           __func__, (double)free_vram / (1024.0*1024.0), (double)bytes_per_slot_all_layers / (1024.0*1024.0));
        }
    }

    // clamp a forced S to what fits; the unclamped path OOMs at graph capture
    if (!g_variable_placement && manual_S && bytes_per_slot_all_layers > 0 && !warm_requested) {
        const int afford = (int) (usable_vram / bytes_per_slot_all_layers);
        if (g_S > afford) {
            TIER_LOG("%s: manual S = %d exceeds free VRAM, clamping to %d\n", __func__, g_S, afford);
            g_S = std::max(afford, 0);
        }
    }

    // With an explicitly requested warm tier, keep the existing/default fixed
    // tier first, reserve its sentinel, then consume only remaining slot-sized
    // VRAM. Auto remains conservatively capped at four slots by default, but
    // the cap is explicit so larger GPUs can use a measured higher limit.
    if (warm_requested && bytes_per_slot_all_layers > 0) {
        const int slot_budget = (int) (usable_vram / bytes_per_slot_all_layers);
        const int fixed_afford = std::max(slot_budget - 1, 0);
        g_S = std::min(g_S, n_expert);
        if (g_S > fixed_afford) {
            TIER_LOG("%s: fixed S = %d leaves no safe sentinel budget, clamping to %d\n", __func__, g_S, fixed_afford);
            g_S = fixed_afford;
        }
        const int warm_afford = std::max(slot_budget - g_S - 1, 0);
        const int requested = g_warm_auto ? std::min(warm_afford, std::min(g_warm_auto_max, n_expert)) : g_W;
        g_W = std::min(requested, warm_afford);
        if (!g_warm_auto && g_W < requested) {
            TIER_LOG("%s: warm W = %d exceeds remaining VRAM, clamping to %d\n", __func__, requested, g_W);
        }
        TIER_LOG("%s: HYBRID AUTO-FIT -> fixed S = %d, warm W = %d (slot budget %d, per-slot %.2f MiB)\n",
                __func__, g_S, g_W, slot_budget, (double) bytes_per_slot_all_layers/(1024.0*1024.0));
    }

    if (g_S <= 0) {
        TIER_LOG("%s: expert tiering disabled (S=0)\n", __func__);
        return;
    }
    ggml_backend_buffer_type_t buft_gpu = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip_gpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_init_params ip_cpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    g_ctx_gpu = ggml_init(ip_gpu);
    g_ctx_cpu = ggml_init(ip_cpu);

    std::vector<int32_t> lut_host(n_expert);
    std::vector<int32_t> mask_host(n_expert);
    std::vector<char>    zeros;

    size_t total_bytes = 0;
    int64_t hits = 0, total = 0;

    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        const int fixed_s = g_variable_placement ? g_placement.layers[il].fixed_slots : g_S;

        // top-S experts of this layer by heat
        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            int64_t layer_total = 0;
            for (const auto & p : h) {
                layer_total += p.first;
            }
            for (int i = 0; i < (int) h.size() && i < fixed_s; i++) {
                top.push_back(h[i].second);
                hits += h[i].first;
            }
            total += layer_total;
        }
        if (top.empty()) {
            if (!g_adapt && g_W == 0) {
                continue;
            }
            for (int i = 0; i < std::min(fixed_s, n_expert); i++) {
                top.push_back(i);
            }
        }
        if (g_W > 0 && (int) top.size() < fixed_s) {
            for (int expert = 0; expert < n_expert && (int) top.size() < fixed_s; ++expert) {
                if (std::find(top.begin(), top.end(), expert) == top.end()) {
                    top.push_back(expert);
                }
            }
        }

        const int sentinel = g_W > 0 ? fixed_s + g_W : (int) top.size();
        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) sentinel);
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (!w || !ggml_backend_buffer_is_host(w->buffer)) {
                continue;
            }
            if (w->ne[3] != 1 || (int) w->ne[2] != n_expert) {
                continue;
            }
            const size_t slice = ggml_nbytes(w)/n_expert;

            store s;
            const int n_slots = g_W > 0 ? fixed_s + g_W + 1 : (g_adapt ? fixed_s + 1 : (int) top.size() + 1);
            s.w_hot  = ggml_new_tensor_3d(g_ctx_gpu, w->type, w->ne[0], w->ne[1], n_slots);
            s.lut    = ggml_new_tensor_2d(g_ctx_gpu, GGML_TYPE_I32, 1, n_expert);
            s.mask   = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert);
            s.counts = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert + 1);
            s.il      = il;
            s.is_down = (w == l.ffn_down_exps);

            ggml_set_name(s.w_hot, (std::string(w->name) + ".hot").c_str());

            // tensors are allocated lazily below per-context; collect first
            // (allocation happens after all tensors are created)
            g_stores[w] = s;
            total_bytes += ggml_nbytes(s.w_hot);
            (void) slice;
        }

        // per-layer tensors are ready; fill them after global allocation
    }

    // allocate all tensors
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_gpu, buft_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_cpu, buft_cpu);
    if (!buf_gpu) {
        TIER_LOG("%s: expert tiering GPU allocation failed (VRAM full), falling back to CPU host execution\n", __func__);
        return;
    }
    ggml_backend_buffer_set_usage(buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (buf_cpu) {
        ggml_backend_buffer_set_usage(buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // fill lut/mask/hot weights
    g_layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        const int fixed_s = g_variable_placement ? g_placement.layers[il].fixed_slots : g_S;

        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            for (int i = 0; i < (int) h.size() && i < fixed_s; i++) {
                top.push_back(h[i].second);
            }
        }
        if (top.empty()) {
            if (!g_adapt && g_W == 0) {
                continue;
            }
            for (int i = 0; i < std::min(fixed_s, n_expert); i++) {
                top.push_back(i);
            }
        }
        if (g_W > 0 && (int) top.size() < fixed_s) {
            for (int expert = 0; expert < n_expert && (int) top.size() < fixed_s; ++expert) {
                if (std::find(top.begin(), top.end(), expert) == top.end()) {
                    top.push_back(expert);
                }
            }
        }

        const int sentinel = g_W > 0 ? fixed_s + g_W : (int) top.size();
        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) sentinel);
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            store & s = it->second;
            const size_t slice = ggml_nbytes(w)/n_expert;

            ggml_backend_tensor_set(s.lut,  lut_host.data(),  0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(s.mask, mask_host.data(), 0, n_expert*sizeof(int32_t));

            if (zeros.size() < slice) {
                zeros.assign(slice, 0);
            }
            if (s.w_hot && s.w_hot->buffer) {
                for (int slot = (int) top.size(); slot <= sentinel; ++slot) {
                    ggml_backend_tensor_set(s.w_hot, zeros.data(), slice*slot, slice);
                }
                for (int sl = 0; sl < (int) top.size(); sl++) {
                    const void * src_ptr = w->data ? (const char *) w->data + slice*top[sl] : zeros.data();
                    ggml_backend_tensor_set(s.w_hot, src_ptr, slice*sl, slice);
                }
            }
        }

        // group the stores of this layer for stats/repin
        layer_tier L;
        L.il       = il;
        L.n_expert = n_expert;
        L.sentinel = sentinel;
        L.n_fixed  = fixed_s;
        L.n_slots  = g_W > 0 ? fixed_s + g_W + 1 : (g_adapt ? fixed_s + 1 : (int) top.size() + 1);
        L.slot_expert.assign(L.n_slots, -1);
        L.dwell.assign(g_W > 0 ? fixed_s : L.n_slots, 0);
        L.lut_host = lut_host;
        L.score.assign(n_expert, 0.0f);
        L.cum.assign(n_expert, 0);
        L.observed.assign(n_expert, 0);
        if (il < (int) heat.size()) {
            for (const auto & p : heat[il]) {
                L.cum[p.second]   = (uint64_t) p.first;
                L.score[p.second] = (float) p.first; // warm start from seed
            }
        }
        for (int s = 0; s < (int) top.size(); s++) {
            L.slot_expert[s] = top[s];
        }
        if (g_W > 0) {
            L.warm_cache.reset(n_expert, fixed_s, g_W, top);
            L.warm_admission.reset(n_expert, (uint64_t) g_warm_admission_window);
            L.warm_frequency.assign(n_expert, 0.0f);
            L.warm_pending.assign(g_W, -1);
            L.warm_enabled = true;
            L.slot_expert = L.warm_cache.slot_experts();
            L.lut_host = L.warm_cache.lut();
        }
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            memset(it->second.counts->data, 0, (n_expert + 1)*sizeof(int32_t));
            L.ws.push_back({w, &it->second});
            if (it->second.is_down) {
                L.sd = &it->second;
            }
        }
        if (!L.ws.empty()) {
            if (L.warm_enabled) {
                warm_invariant_abort(L, "init");
            }
            g_layers[il] = std::move(L);
        }
    }

    if (g_prefetch_requested) {
        if (g_W <= 0) {
            TIER_LOG("%s: async prefetch disabled because no warm slots fit\n", __func__);
            g_prefetch_requested = false;
        } else if (!init_async_resources(dev, max_layer_staging_bytes)) {
            TIER_LOG("%s: async prefetch initialization failed; using the synchronous warm-cache path\n", __func__);
            g_prefetch_requested = false;
        }
    }

    g_collect_counts = g_adapt || g_W > 0 || stats_enabled || usage_enabled || json_stats_enabled || g_timing_enabled;
    if (g_static_no_sync_requested) {
        std::vector<const char *> blockers;
        if (g_adapt) {
            blockers.push_back("LLAMA_EXPERT_ADAPT is enabled");
        }
        if (g_W > 0 || warm_config_requested) {
            blockers.push_back("warm slots are enabled or requested");
        }
        if (stats_enabled || usage_enabled || json_stats_enabled || g_timing_enabled) {
            blockers.push_back("expert stats or usage output is enabled");
        }
        if (blockers.empty()) {
            g_static_no_sync_active = true;
            g_collect_counts = false;
            TIER_LOG("expert_tier: static no-sync enabled; count collection and post-graph tier updates are disabled\n");
        } else {
            TIER_LOG("expert_tier: static no-sync rejected:");
            for (const char * blocker : blockers) {
                TIER_LOG(" %s;", blocker);
            }
            TIER_LOG(" using synchronized tier updates\n");
        }
    }

    if (g_variable_placement) {
        TIER_LOG("%s: expert tiering on: variable fixed slots (max %d) + %d warm slots/layer, %zu tensors, %.2f GiB pinned, seed coverage %.1f%%\n",
                __func__, g_S, g_W, g_stores.size(), (double) total_bytes/(1 << 30),
                total > 0 ? 100.0*hits/total : 0.0);
    } else {
        TIER_LOG("%s: expert tiering on: %d fixed + %d warm slots/layer, %zu tensors, %.2f GiB pinned, seed coverage %.1f%%\n",
                __func__, g_S, g_W, g_stores.size(), (double) total_bytes/(1 << 30),
                total > 0 ? 100.0*hits/total : 0.0);
    }
    if (usage_enabled) {
        TIER_LOG("%s: expert usage export mode: %s\n", __func__,
                g_usage_mode == expert_usage_mode::session ? "session" : "cumulative");
    }
    TIER_LOG("%s: CPU cold singleton chunk size: %d rows; block-parallel activation: %s; down prefetch: %d; row reuse: %s; async overlap: %s\n",
            __func__, g_cpu_single_row_chunk, g_cpu_parallel_activation ? "on" : "off",
            g_cpu_down_prefetch, g_cpu_reuse_rows ? "on" : "off", g_cpu_async ? "on" : "off");
    if (g_W > 0) {
        if (g_warm_admission == warm_admission_mode::frequency) {
            TIER_LOG("%s: warm admission frequency window=%d graphs\n", __func__, g_warm_admission_window);
        } else if (g_warm_admission == warm_admission_mode::second_hit) {
            TIER_LOG("%s: warm admission second-hit window=%d graphs\n", __func__, g_warm_admission_window);
        } else {
            TIER_LOG("%s: warm admission immediate\n", __func__);
        }
    }
}

ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) {
    auto it = g_stores.find(w);
    if (it == g_stores.end() || ids->ne[1] > (int64_t) g_tmax) {
        return ggml_mul_mat_id(ctx, w, x, ids);
    }

    const store & s = it->second;

    // get_rows wants matching trailing dims, so flatten ids to 1d first
    ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ids), ids->ne[0]*ids->ne[1]);
    ggml_tensor * hot_flat = ggml_get_rows(ctx, s.lut, ids_flat); // [1, n_used*n_tokens]
    ggml_tensor * ids_hot  = ggml_reshape_2d(ctx, hot_flat, ids->ne[0], ids->ne[1]);
    ggml_tensor * hot      = ggml_mul_mat_id(ctx, s.w_hot, x, ids_hot);    // GPU

    if (g_hot_only) {
        return hot;
    }

    ggml_tensor * cold     = ggml_mul_mat_id_cold(ctx, w, x, ids, s.mask); // CPU

    return ggml_add(ctx, hot, cold);
}

bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids) {
    g_hot_only = false;
    if (!eligible || g_stores.empty()) {
        return false;
    }
    auto ig = g_stores.find(gate_w);
    auto iu = g_stores.find(up_w);
    auto id = g_stores.find(down_w);
    if (ig == g_stores.end() || iu == g_stores.end() || id == g_stores.end() ||
        ids->ne[1] > (int64_t) g_tmax) {
        return false;
    }
    g_hot_only = true;
    return true;
}

bool cpu_async_enabled() {
    return g_cpu_async;
}

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids) {
    if (!g_hot_only) {
        return nullptr;
    }
    store & sd = g_stores[down_w];
    ggml_tensor * result = ggml_moe_cold(ctx, gate_w, up_w, down_w, x, ids, sd.mask,
            g_collect_counts ? sd.counts : nullptr);
    result->op_params[0] = sd.il;
    return result;
}

ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids) {
    // prompt-sized batches skip the fused path; harvest router decisions anyway
    if (!g_collect_counts || g_hot_only || g_stores.empty()) {
        return nullptr;
    }
    auto it = g_stores.find(down_w);
    if (it == g_stores.end() || ids->ne[1] <= (int64_t) g_tmax) {
        return nullptr;
    }
    return ggml_moe_count(ctx, ids, it->second.counts);
}

}
