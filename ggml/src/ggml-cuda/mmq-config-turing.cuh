// MMQ configuration for Turing (sm_75) with the DP4A data layout.
//
// The Pascal table is the baseline for the DP4A layout; the knobs below are the
// compile-time overrides of the Turing sweep. The defaults reproduce the
// Pascal table exactly, so a build without the knobs behaves like before.
//
// I         rows of the weight matrix per tile (more rows: less activation traffic)
// J_MAX     largest column tile the type may use (more columns: less weight traffic)
// STREAM_K  split the K range over all SMs instead of one tile per block
//
// Measured on the GTX 1660 Ti (sm_75) with the DP4A layout: J_MAX 128 gives
// +7.1 % (Q4_K) and +10.1 % (Q8_0) prefill against the Pascal table's 64, I 128
// costs 22 % (Q4_K), and stream-k helps Q8_0 by 3.6 % but costs Q4_K 0.5 %.
#ifndef GGML_CUDA_TURING_MMQ_I
#define GGML_CUDA_TURING_MMQ_I 64
#endif

#ifndef GGML_CUDA_TURING_MMQ_J_MAX
#define GGML_CUDA_TURING_MMQ_J_MAX 128
#endif

#ifndef GGML_CUDA_TURING_MMQ_STREAM_K
#define GGML_CUDA_TURING_MMQ_STREAM_K false
#endif

// The Pascal table knows 8, 16, 24, 32, 40, 48 and 64 columns without the
// fallback and 8, 16, 32 and 64 with it. Only the plain case may grow.
static constexpr __host__ __device__ bool ggml_cuda_mmq_turing_j_known(int J, bool fallback) {
    if (fallback) {
        return J == 8 || J == 16 || J == 32 || J == 64;
    }
    if (J == 8 || J == 16 || J == 24 || J == 32 || J == 40 || J == 48 || J == 64) {
        return true;
    }
    return J <= GGML_CUDA_TURING_MMQ_J_MAX && J % 8 == 0;
}

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_turing(ggml_type type, int J, bool fallback) {
    ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config_pascal(type, J, fallback);
    if (cfg.type == GGML_TYPE_COUNT) {
        if (!ggml_cuda_mmq_turing_j_known(J, fallback)) {
            return cfg;
        }
        // J above the Pascal table: reuse the widest entry of the same type
        cfg = ggml_cuda_mmq_get_config_pascal(type, 64, fallback);
        if (cfg.type == GGML_TYPE_COUNT) {
            return cfg;
        }
        cfg = ggml_cuda_mmq_config(cfg.type, cfg.nthreads, cfg.occupancy, cfg.I, J,
                cfg.sram_layout, cfg.K_vram, cfg.stream_k, cfg.fallback);
    }
    if (cfg.I != GGML_CUDA_TURING_MMQ_I || cfg.stream_k != GGML_CUDA_TURING_MMQ_STREAM_K) {
        cfg = ggml_cuda_mmq_config(cfg.type, cfg.nthreads, cfg.occupancy, GGML_CUDA_TURING_MMQ_I,
                cfg.J, cfg.sram_layout, cfg.K_vram, GGML_CUDA_TURING_MMQ_STREAM_K, cfg.fallback);
    }
    return cfg;
}
