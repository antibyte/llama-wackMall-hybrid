#include "models.h"
#include "llama-expert-lookahead.h"
#include "llama-expert-tier.h"

// Aleph Alpha Kolibri 1
// Qwen3-MoE GQA with per-head q/k RMSNorm.
// Sliding-window layers use RoPE; full-attention layers use no positional encoding.
// Sandwich norms around attention and the MoE block.
// Every layer is MoE plus one ungated shared expert.
// Router: top-k on (logits + expert_bias), weights are unbiased sigmoid(logits).

void llama_model_kolibri1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa);

    hparams.n_expert_shared = 1;
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    if (hparams.n_ff_shexp == 0) {
        // test-llama-archs omits this key; real GGUFs store the shared expert width
        hparams.n_ff_shexp = hparams.n_ff_exp * hparams.n_expert_shared;
    }

    if (hparams.n_swa == 0) {
        throw std::runtime_error("kolibri1: sliding_window must be > 0");
    }
    if (hparams.n_expert_shared != 1) {
        throw std::runtime_error(format("kolibri1: expected exactly 1 shared expert, got %u", hparams.n_expert_shared));
    }
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID_LOGIT_ADD) {
        throw std::runtime_error(format("kolibri1: expected expert_gating_func %d (sigmoid_logit_add), got %u",
                    (int) LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID_LOGIT_ADD, hparams.expert_gating_func));
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    uint32_t swa_period = 5;
    if (ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false)) {
        hparams.set_swa_pattern(swa_period);
    } else if (!ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer())) {
        hparams.set_swa_pattern(5);
    }

    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_kolibri1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("kolibri1: n_expert and n_expert_used must be > 0");
    }

    const int64_t n_ff_exp   = hparams.n_ff_exp;
    const int64_t n_ff_shexp = hparams.n_ff_shexp;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_kolibri1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_kolibri1::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    const bool lookahead_active = llama_expert_lookahead::graph_enabled(params.ubatch.n_tokens, params.ubatch.n_seqs, false);
    const int  lookahead_distance = lookahead_active ? llama_expert_lookahead::distance() : 0;
    const int  lookahead_top_m = lookahead_active ? llama_expert_lookahead::top_m((int) n_expert) : 0;
    const bool lookahead_trace = lookahead_active && llama_expert_lookahead::enabled();

    const int prefetch_top_m = llama_expert_tier::page_prefetch_top_m(n_tokens);

    // Predicts the routing of a later layer from this layer's residual stream:
    // the target layer's ffn_norm weight and router on the current hidden state.
    auto build_predicted_experts = [&](ggml_tensor * source, int target_layer, int top_m) {
        const auto & target = model.layers[target_layer];
        ggml_tensor * predictor_input = build_norm(source, target.ffn_norm, NULL, LLM_NORM_RMS, target_layer);
        ggml_tensor * logits = build_lora_mm(target.ffn_gate_inp, predictor_input);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        logits = ggml_add(ctx0, logits, target.ffn_exp_probs_b);
        cb(logits, "lookahead_logits", target_layer);
        return ggml_argsort_top_k(ctx0, logits, top_m);
    };

    auto build_lookahead_prediction = [&](ggml_tensor * source, int source_layer) {
        if (!lookahead_active || lookahead_top_m <= 0) {
            return;
        }
        const int target_layer = source_layer + lookahead_distance;
        if (target_layer >= n_layer || !llama_expert_lookahead::predictor_enabled(target_layer)) {
            return;
        }
        ggml_tensor * predicted_ids = build_predicted_experts(source, target_layer, lookahead_top_m);
        cb(predicted_ids, "lookahead_topk", target_layer);
        ggml_build_forward_expand(gf, predicted_ids);
        if (lookahead_trace) {
            ggml_set_output(predicted_ids);
            res->add_lookahead_prediction({ source_layer, target_layer, lookahead_top_m, predicted_ids, nullptr, nullptr });
        }
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // sliding layers: RoPE; full-attention layers: no positional encoding
        const bool use_rope = hparams.is_swa(il);

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);

            if (use_rope) {
                const float freq_base_l  = model.get_rope_freq_base (cparams, il);
                const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);
            }

            cur = build_attn(inp_attn,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (llama_expert_lookahead::point() == llama_expert_lookahead::prediction_point::post_attn) {
            build_lookahead_prediction(ffn_inp, il);
        }

        // read the next layer's predicted cold experts from disk while this layer computes
        ggml_tensor * page_prefetch = nullptr;
        if (prefetch_top_m > 0 && il + 1 < n_layer) {
            const auto & next = model.layers[il + 1];
            ggml_tensor * predicted = build_predicted_experts(ffn_inp, il + 1, prefetch_top_m);
            cb(predicted, "page_prefetch_ids", il + 1);
            page_prefetch = llama_expert_tier::build_page_prefetch(ctx0,
                    next.ffn_gate_exps, next.ffn_up_exps, next.ffn_down_exps, predicted);
        }

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * actual_ids = nullptr;
        ggml_tensor * actual_weights = nullptr;

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                0.0f,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                lookahead_trace ? &actual_ids : nullptr,
                lookahead_trace ? &actual_weights : nullptr,
                lookahead_trace && llama_expert_lookahead::layer_enabled(il),
                page_prefetch);
        cb(moe_out, "ffn_moe_out", il);
        if (lookahead_trace && llama_expert_lookahead::layer_enabled(il)) {
            // the allocator reuses the router buffers before the trace reads them
            ggml_tensor * ids_snapshot = ggml_dup(ctx0, actual_ids);
            ggml_tensor * weights_snapshot = ggml_dup(ctx0, actual_weights);
            ggml_set_output(ids_snapshot);
            ggml_set_output(weights_snapshot);
            ggml_build_forward_expand(gf, ids_snapshot);
            ggml_build_forward_expand(gf, weights_snapshot);
            res->set_lookahead_actual(il, ids_snapshot, weights_snapshot);
        }

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   NULL, NULL,
                layer.ffn_gate_shexp, NULL, NULL,
                layer.ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        if (llama_expert_lookahead::point() == llama_expert_lookahead::prediction_point::post_moe) {
            build_lookahead_prediction(cur, il);
        }

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
