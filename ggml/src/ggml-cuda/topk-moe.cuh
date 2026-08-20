#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

struct ggml_cuda_topk_moe_grouped_args {
    bool sigmoid = true;
    bool prob_bias = true;
    bool norm = true;
    bool scale = true;
    int n_groups = 8;
    int n_exp_per_group = 16;
    int n_group_used = 4;
    int group_top = 2;
    float weight_eps = 0.0f;
};

struct ggml_cuda_topk_moe_grouped_match {
    const ggml_tensor * logits  = nullptr;
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;
    const ggml_tensor * bias    = nullptr;
    int                 n_nodes = 0;
    int                 ids_idx = -1;
    int                 weights_idx = -1;
};

float ggml_cuda_moe_weight_eps();

bool ggml_cuda_topk_moe_grouped_fusion(const struct ggml_cgraph * cgraph, int node_idx,
                                       ggml_cuda_topk_moe_grouped_args & args);

bool ggml_cuda_topk_moe_grouped_match_nodes(const struct ggml_cgraph * cgraph, int node_idx,
                                            ggml_cuda_topk_moe_grouped_args & args,
                                            ggml_cuda_topk_moe_grouped_match & match);

void ggml_cuda_op_topk_moe_grouped(ggml_backend_cuda_context & ctx,
                                   const ggml_tensor * logits,
                                   ggml_tensor * weights,
                                   ggml_tensor * ids,
                                   const ggml_tensor * clamp,
                                   const ggml_tensor * scale,
                                   const ggml_tensor * bias,
                                   const ggml_cuda_topk_moe_grouped_args & args);
