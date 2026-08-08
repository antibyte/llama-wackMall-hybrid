// Copyright 2026 llama-wackMall-hybrid contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Build this small, model-independent benchmark against either this fork or
// ik_llama.cpp.  Define BENCH_IK_LLAMA for the latter so that the native IQK
// fused up/gate operation is used.  The ordinary build deliberately expresses
// the same calculation as two matrix multiplications plus SiLU and multiply.

#include "ggml.h"
#include "ggml-impl.h"
#if !defined(BENCH_IK_LLAMA)
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void compute_graph(std::vector<uint8_t> & work, ggml_cgraph * graph, int threads) {
#if defined(BENCH_IK_LLAMA)
    ggml_cplan plan = ggml_graph_plan(graph, threads);
#else
    ggml_cplan plan = ggml_graph_plan(graph, threads, nullptr);
#endif
    work.resize(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const int status = ggml_graph_compute(graph, &plan);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %d\n", status);
        std::exit(2);
    }
}

static ggml_tensor * make_result(
        ggml_context * ctx,
        ggml_tensor * up,
        ggml_tensor * gate,
        ggml_tensor * input) {
#if defined(BENCH_IK_LLAMA)
    return ggml_fused_up_gate(ctx, up, gate, input, GGML_UNARY_OP_SILU);
#else
    ggml_tensor * up_out   = ggml_mul_mat(ctx, up, input);
    ggml_tensor * gate_out = ggml_mul_mat(ctx, gate, input);
    return ggml_mul(ctx, up_out, ggml_silu(ctx, gate_out));
#endif
}

static void fill_weights(float * data, size_t count, float phase) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0.11f*std::sin(0.0017f*float(i) + phase)
                + 0.07f*std::cos(0.0031f*float(i) - phase);
    }
}

int main(int argc, char ** argv) {
    const int columns = argc > 1 ? std::atoi(argv[1]) : 1;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 50;
    if (columns < 1 || columns > 4 || threads < 1 || iterations < 1) {
        std::fprintf(stderr, "usage: %s [columns 1..4] [threads] [iterations]\n", argv[0]);
        return 1;
    }

    constexpr int64_t n_embd = 2048;
    constexpr int64_t n_ff = 512;
    constexpr ggml_type weight_type = GGML_TYPE_Q4_K;
    const size_t weight_elements = size_t(n_embd)*size_t(n_ff);
    const size_t context_bytes = 64u*1024u*1024u;
    ggml_init_params params = { context_bytes, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 2;
    }

    ggml_tensor * f_gate[2];
    ggml_tensor * f_up[2];
    ggml_tensor * q_gate[2];
    ggml_tensor * q_up[2];
    ggml_cgraph * graphs[2];
    ggml_tensor * results[2];

    for (int pair = 0; pair < 2; ++pair) {
        f_gate[pair] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        f_up[pair] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        fill_weights(static_cast<float *>(f_gate[pair]->data), weight_elements, 0.3f + pair);
        fill_weights(static_cast<float *>(f_up[pair]->data), weight_elements, 1.1f + pair);

        q_gate[pair] = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff);
        q_up[pair] = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff);
        ggml_quantize_chunk(weight_type, static_cast<const float *>(f_gate[pair]->data),
                q_gate[pair]->data, 0, n_ff, n_embd, nullptr
#if defined(BENCH_IK_LLAMA)
                , nullptr
#endif
                );
        ggml_quantize_chunk(weight_type, static_cast<const float *>(f_up[pair]->data),
                q_up[pair]->data, 0, n_ff, n_embd, nullptr
#if defined(BENCH_IK_LLAMA)
                , nullptr
#endif
                );
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, columns);
    float * input_data = static_cast<float *>(input->data);
    for (int64_t i = 0; i < n_embd*columns; ++i) {
        input_data[i] = 0.2f*std::sin(0.013f*float(i)) + 0.03f*std::cos(0.021f*float(i));
    }

    for (int pair = 0; pair < 2; ++pair) {
        results[pair] = make_result(ctx, q_up[pair], q_gate[pair], input);
        graphs[pair] = ggml_new_graph(ctx);
        ggml_build_forward_expand(graphs[pair], results[pair]);
    }

    std::vector<uint8_t> work[2];
    compute_graph(work[0], graphs[0], threads);
    compute_graph(work[1], graphs[1], threads);

    int64_t total_us = 0;
    double checksum = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const int pair = i & 1;
        const int64_t begin = ggml_time_us();
        compute_graph(work[pair], graphs[pair], threads);
        total_us += ggml_time_us() - begin;
        const float * out = static_cast<const float *>(results[pair]->data);
        checksum += out[(i*37) % (n_ff*columns)];
    }

    const double average_us = double(total_us)/iterations;
    const double matmul_flops = 4.0*double(n_embd)*double(n_ff)*columns;
    std::printf("backend=%s columns=%d threads=%d iterations=%d avg_us=%.3f projected_gflops=%.3f checksum=%.9g\n",
#if defined(BENCH_IK_LLAMA)
            "ik-fused",
#else
            "wack-graph",
#endif
            columns, threads, iterations, average_us, matmul_flops/average_us/1000.0, checksum);

    ggml_free(ctx);
    return 0;
}
