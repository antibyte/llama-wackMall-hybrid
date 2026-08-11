#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    LLAMA_LOG_INFO("%s: DFlash extract_layers = [", __func__);
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        LLAMA_LOG_INFO("%d%s", target_layer_ids[i], i + 1 < target_layer_ids.size() ? ", " : "");
    }
    LLAMA_LOG_INFO("]\n");

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

bool llama_model_dflash::build_dflash_injection(
        llm_graph_context              & graph,
        ggml_tensor                    * features,
        const llama_cparams            & cparams_draft,
        const llama_memory_context_i * mctx_draft) const {
    if (features == nullptr || mctx_draft == nullptr) {
        return false;
    }

    const int64_t n_tokens    = graph.ubatch.n_tokens;
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_head_kv   = hparams.n_head_kv();

    GGML_ASSERT(features->ne[0] == hparams.n_embd_inp_enc());
    GGML_ASSERT(features->ne[1] == n_tokens);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    auto inp_pos = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());
    inp_pos->pos = ggml_new_tensor_1d(
            graph.ctx0, GGML_TYPE_I32, n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(inp_pos->pos);
    ggml_tensor * pos = inp_pos->pos;
    graph.res->add_input(std::move(inp_pos));

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;

    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    if (use_iswa) {
        const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(mctx_draft);
        auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams_draft, mctx, true);

        inp->self_k_idxs     = mctx->get_base()->build_input_k_idxs(graph.ctx0, graph.ubatch);
        inp->self_v_idxs     = mctx->get_base()->build_input_v_idxs(graph.ctx0, graph.ubatch);
        inp->self_k_idxs_swa = mctx->get_swa()->build_input_k_idxs(graph.ctx0, graph.ubatch);
        inp->self_v_idxs_swa = mctx->get_swa()->build_input_v_idxs(graph.ctx0, graph.ubatch);

        inp->self_k_rot     = mctx->get_base()->build_input_k_rot(graph.ctx0);
        inp->self_v_rot     = mctx->get_base()->build_input_v_rot(graph.ctx0);
        inp->self_k_rot_swa = mctx->get_swa()->build_input_k_rot(graph.ctx0);
        inp->self_v_rot_swa = mctx->get_swa()->build_input_v_rot(graph.ctx0);

        inp_attn_iswa = static_cast<llm_graph_input_attn_kv_iswa *>(graph.res->add_input(std::move(inp)));
    } else {
        const auto * mctx = static_cast<const llama_kv_cache_context *>(mctx_draft);
        auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams_draft, mctx, true);

        inp->self_k_idxs = mctx->build_input_k_idxs(graph.ctx0, graph.ubatch);
        inp->self_v_idxs = mctx->build_input_v_idxs(graph.ctx0, graph.ubatch);
        inp->self_k_rot  = mctx->build_input_k_rot(graph.ctx0);
        inp->self_v_rot  = mctx->build_input_v_rot(graph.ctx0);

        inp_attn = static_cast<llm_graph_input_attn_kv *>(graph.res->add_input(std::move(inp)));
    }

    ggml_tensor * cur = ggml_mul_mat(graph.ctx0, fc, features);
    graph.cb(cur, "dflash_fc", -1);

    cur = ggml_rms_norm(graph.ctx0, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(graph.ctx0, cur, output_norm_enc);
    graph.cb(cur, "dflash_enc_norm", -1);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto & layer = layers[il];

        ggml_tensor * Kcur = ggml_mul_mat(graph.ctx0, layer.wk, cur);
        ggml_tensor * Vcur = ggml_mul_mat(graph.ctx0, layer.wv, cur);

        Kcur = ggml_reshape_3d(graph.ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(graph.ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = ggml_rms_norm(graph.ctx0, Kcur, hparams.f_norm_rms_eps);
        Kcur = ggml_mul(graph.ctx0, Kcur, layer.attn_k_norm);
        Kcur = ggml_rope_ext(
                graph.ctx0, Kcur, pos, nullptr,
                hparams.n_rot(), hparams.rope_type, cparams_draft.n_ctx_orig_yarn,
                cparams_draft.rope_freq_base, cparams_draft.rope_freq_scale,
                cparams_draft.yarn_ext_factor, cparams_draft.yarn_attn_factor,
                cparams_draft.yarn_beta_fast, cparams_draft.yarn_beta_slow);

        graph.cb(Kcur, "dflash_k_inject", il);
        graph.cb(Vcur, "dflash_v_inject", il);

        if (use_iswa) {
            const bool is_swa = hparams.is_swa(il);
            const auto * kv = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
            ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
            ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
            ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
            ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;

            if (k_rot != nullptr) {
                Kcur = llama_mul_mat_hadamard(graph.ctx0, Kcur, k_rot);
            }
            if (v_rot != nullptr) {
                Vcur = llama_mul_mat_hadamard(graph.ctx0, Vcur, v_rot);
            }

            ggml_build_forward_expand(graph.gf, kv->cpy_k(graph.ctx0, Kcur, k_idxs, il));
            ggml_build_forward_expand(graph.gf, kv->cpy_v(graph.ctx0, Vcur, v_idxs, il));
        } else {
            if (inp_attn->self_k_rot != nullptr) {
                Kcur = llama_mul_mat_hadamard(graph.ctx0, Kcur, inp_attn->self_k_rot);
            }
            if (inp_attn->self_v_rot != nullptr) {
                Vcur = llama_mul_mat_hadamard(graph.ctx0, Vcur, inp_attn->self_v_rot);
            }

            ggml_build_forward_expand(graph.gf, inp_attn->mctx->cpy_k(graph.ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(graph.gf, inp_attn->mctx->cpy_v(graph.ctx0, Vcur, inp_attn->get_v_idxs(), il));
        }
    }

    return true;
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // KV cache injection
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);

        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                // rotate K/V into the cache's rotated space
                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;
                if (k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                // rotate K/V into the cache's rotated space
                if (inp_attn->self_k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        ggml_tensor * Kcur = build_lora_mm(layer.wk, noise_norm);
        ggml_tensor * Vcur = build_lora_mm(layer.wv, noise_norm);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = ggml_get_rows(ctx0, inpL, inp_out_ids);
    cb(cur, "result_embd", -1);

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model (shared via ctx_other)
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
