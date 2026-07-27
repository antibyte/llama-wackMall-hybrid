#include "llama-expert-tier.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// tier status must print at default verbosity; libllama INFO requires -v
#define TIER_LOG(...) fprintf(stderr, __VA_ARGS__)

namespace llama_expert_tier {

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
    int sentinel = 0;
    store * sd = nullptr; // store holding the down weight (counts source)
    std::vector<std::pair<const ggml_tensor *, store *>> ws;
    std::vector<int32_t> slot_expert; // [n_slots], -1 = empty
    std::vector<int32_t> lut_host;    // [n_expert]
    std::vector<float>   score;       // [n_expert], cumulative (LLAMA_EXPERT_DECAY)
    std::vector<uint64_t> cum;        // [n_expert], exact counts for persistence
    std::vector<int32_t> dwell;       // [n_slots]
    uint64_t cum_cold = 0, cum_total = 0, cum_graphs = 0;
};

static int  g_S        = 16;
static int  g_tmax     = 16;
static bool g_adapt    = false;
static bool g_hot_only = false; // set by build_moe_cold per layer

static ggml_context * g_ctx_gpu = nullptr; // owned for process lifetime
static ggml_context * g_ctx_cpu = nullptr;

static std::unordered_map<const ggml_tensor *, store> g_stores;
static std::vector<layer_tier> g_layers;
static uint64_t g_repins = 0; // hot-set changes since init

static bool parse_heat_csv(const std::string & path, int n_layer,
        std::vector<std::vector<std::pair<int64_t, int32_t>>> & heat) {
    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_ERROR("%s: cannot open heat csv '%s'\n", __func__, path.c_str());
        return false;
    }
    heat.resize(n_layer);
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<int64_t> vals;
        while (std::getline(ss, tok, ',')) {
            vals.push_back(std::stoll(tok));
        }
        if (vals.size() < 3) {
            continue;
        }
        const int il = (int) vals[0];
        if (il < 0 || il >= n_layer) {
            continue;
        }
        heat[il].emplace_back(vals[2], (int32_t) vals[1]);
    }
    return true;
}

// consume selection counts, update decayed scores + cumulative stats, and
// (LLAMA_EXPERT_ADAPT) repin at most one slot. called at graph build time,
// i.e. between compute steps; counts are written by the fused cold op.
static void maybe_update(layer_tier & L) {
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

static void dump_stats() {
    if (const char * p = getenv("LLAMA_EXPERT_STATS")) {
        FILE * f = strcmp(p, "1") ? fopen(p, "w") : stderr;
        if (f) {
            fprintf(f, "expert_tier: repins %llu\n", (unsigned long long) g_repins);
            for (const auto & L : g_layers) {
                if (L.ws.empty() || L.cum_total == 0) {
                    continue;
                }
                fprintf(f, "layer %2d: cold %llu / %llu (%.1f%%) graphs %llu\n", L.il,
                        (unsigned long long) L.cum_cold, (unsigned long long) L.cum_total,
                        100.0*(double) L.cum_cold/(double) L.cum_total,
                        (unsigned long long) L.cum_graphs);
            }
            // invariant: mask agrees with lut, slot<->expert bijection holds
            for (const auto & L : g_layers) {
                if (!L.sd) {
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
    if (const char * p = getenv("LLAMA_EXPERT_USAGE")) {
        FILE * f = fopen(p, "w");
        if (f) {
            fprintf(f, "layer,expert,count\n");
            for (const auto & L : g_layers) {
                if (L.cum.empty()) {
                    continue;
                }
                for (int e = 0; e < L.n_expert; e++) {
                    if (L.cum[e] > 0) {
                        fprintf(f, "%d,%d,%llu\n", L.il, e, (unsigned long long) L.cum[e]);
                    }
                }
            }
            fclose(f);
        }
    }
}

void update() {
    for (auto & L : g_layers) {
        maybe_update(L);
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

void init(const llama_model & model) {
    if (!g_layers.empty()) {
        return; // already initialized
    }
    const char * env_s   = getenv("LLAMA_EXPERT_S");
    const char * env_hot = getenv("LLAMA_EXPERT_HOT");
    if (const char * e = getenv("LLAMA_EXPERT_ADAPT")) {
        g_adapt = atoi(e) != 0;
    } else {
        g_adapt = true; // Auto-fit and online adaptation ON by default
    }

    bool manual_S = false;
    if (env_s) {
        g_S = std::max(0, atoi(env_s));
        manual_S = (env_s != nullptr);
    }

    if (const char * e = getenv("LLAMA_EXPERT_TMAX")) {
        g_tmax = std::max(0, atoi(e));
    }

    const int n_layer  = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;

    std::vector<std::vector<std::pair<int64_t, int32_t>>> heat;
    if (env_hot && env_hot[0] && !parse_heat_csv(env_hot, n_layer, heat)) {
        return;
    }
    if (getenv("LLAMA_EXPERT_STATS") || getenv("LLAMA_EXPERT_USAGE")) {
        atexit(dump_stats);
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        TIER_LOG("%s: expert tiering disabled (no GPU device found)\n", __func__);
        return;
    }

    // Calculate per-expert slot memory size across all layers for Auto-Fit
    size_t bytes_per_slot_all_layers = 0;
    int n_tensors = 0;
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w && ggml_backend_buffer_is_host(w->buffer)) {
                n_tensors++;
                if (n_expert > 0) {
                    bytes_per_slot_all_layers += ggml_nbytes(w) / n_expert;
                }
            }
        }
    }

    if (n_tensors == 0) {
        TIER_LOG("%s: expert tiering disabled (no host expert weights found)\n", __func__);
        return;
    }

    // Reserve 512 MB for the CUDA runtime, display, alignment and the graph
    // capture buffers that allocate after this sizing (300 MB let capture OOM)
    const size_t safety_buffer = 512ULL * 1024 * 1024;
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    const size_t usable_vram = (free_vram > safety_buffer) ? (free_vram - safety_buffer) : 0;

    if (!manual_S && bytes_per_slot_all_layers > 0) {
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
    if (manual_S && bytes_per_slot_all_layers > 0) {
        const int afford = (int) (usable_vram / bytes_per_slot_all_layers);
        if (g_S > afford) {
            TIER_LOG("%s: manual S = %d exceeds free VRAM, clamping to %d\n", __func__, g_S, afford);
            g_S = std::max(afford, 0);
        }
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

        // top-S experts of this layer by heat
        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            int64_t layer_total = 0;
            for (const auto & p : h) {
                layer_total += p.first;
            }
            for (int i = 0; i < (int) h.size() && i < g_S; i++) {
                top.push_back(h[i].second);
                hits += h[i].first;
            }
            total += layer_total;
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(g_S, n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
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
            s.w_hot  = ggml_new_tensor_3d(g_ctx_gpu, w->type, w->ne[0], w->ne[1], g_adapt ? g_S + 1 : (int) top.size() + 1);
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

        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int i = 0; i < (int) h.size() && i < g_S; i++) {
                top.push_back(h[i].second);
            }
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(g_S, n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
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
                ggml_backend_tensor_set(s.w_hot, zeros.data(), slice*top.size(), slice);
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
        L.sentinel = (int) top.size();
        L.n_slots  = g_adapt ? g_S + 1 : (int) top.size() + 1;
        L.slot_expert.assign(L.n_slots, -1);
        L.dwell.assign(L.n_slots, 0);
        L.lut_host = lut_host;
        L.score.assign(n_expert, 0.0f);
        L.cum.assign(n_expert, 0);
        if (il < (int) heat.size()) {
            for (const auto & p : heat[il]) {
                L.cum[p.second]   = (uint64_t) p.first;
                L.score[p.second] = (float) p.first; // warm start from seed
            }
        }
        for (int s = 0; s < (int) top.size(); s++) {
            L.slot_expert[s] = top[s];
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
            g_layers[il] = std::move(L);
        }
    }

                TIER_LOG("%s: expert tiering on: %d slots/layer, %zu tensors, %.2f GiB pinned, seed coverage %.1f%%\n",
            __func__, g_S, g_stores.size(), (double) total_bytes/(1 << 30),
            total > 0 ? 100.0*hits/total : 0.0);
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

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids) {
    if (!g_hot_only) {
        return nullptr;
    }
    store & sd = g_stores[down_w];
    return ggml_moe_cold(ctx, gate_w, up_w, down_w, x, ids, sd.mask, sd.counts);
}

ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids) {
    // prompt-sized batches skip the fused path; harvest router decisions anyway
    if (g_hot_only || g_stores.empty()) {
        return nullptr;
    }
    auto it = g_stores.find(down_w);
    if (it == g_stores.end() || ids->ne[1] <= (int64_t) g_tmax) {
        return nullptr;
    }
    return ggml_moe_count(ctx, ids, it->second.counts);
}

}
