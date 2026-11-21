// Inkling 2 (Thinking Machines): looped MoE transformer with relative-position attention, short convolutions and
// cross-layer attention residuals. Ported from the Unsloth llama.cpp b12588-mix (root "looped" GGUF layout).
#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// max elements of the dense relative bias per attention call (LLAMA_INKLING2_ATTN_CHUNK_ELEMS), max chunks per layer
static int64_t inkling2_kq_bias_max() {
    static const int64_t v = [] {
        const char * e = getenv("LLAMA_INKLING2_ATTN_CHUNK_ELEMS");
        const long long n = e ? atoll(e) : 0;
        return n > 0 ? (int64_t) n : int64_t(1) << 26;
    }();
    return v;
}
static constexpr int64_t n_kq_chunk_max = 64;

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    if (!hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded || !hparams.n_ff_exp ||
        !hparams.n_expert_used || hparams.n_expert_used > hparams.n_expert) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or experts");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    if (hparams.loop_n_pass > 1) {
        ml.get_key(LLM_KV_LOOP_GATE_RANK, hparams.loop_gate_rank);
        if (!hparams.loop_core_start || !hparams.loop_gate_rank || hparams.loop_core_start % hparams.attn_res_block_size ||
            (hparams.loop_core_end - hparams.loop_core_start) % hparams.attn_res_block_size) {
            throw std::runtime_error("inkling2: the looped core must start after the first block and span whole blocks");
        }
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(0);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il)) {
            throw std::runtime_error("inkling2: invalid attention dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;
    const int64_t n_ff_exp       = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    const bool loop = hparams.loop_n_pass > 1;

    // physical layers; llama_model_base::load_tensors() shares them with the later visits
    for (int i = 0; i < n_layer - (int) hparams.loop_n_extra(); ++i) {
        auto & layer = layers[i];

        const int iv = hparams.loop_visit(i);

        const int64_t n_head      = hparams.n_head(iv);
        const int64_t n_head_kv   = hparams.n_head_kv(iv);
        const int64_t n_embd_head = hparams.n_embd_head_k(iv);
        const int64_t n_extent    = hparams.is_swa(iv) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(iv)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(iv) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        // message controllers of the looped core: (attn, k, v, mlp) x (down, gain, forget)
        if (loop && i >= (int) hparams.loop_core_start && i < (int) hparams.loop_core_end) {
            static const llm_tensor down[4]   = {LLM_TENSOR_LOOP_ATTN_DOWN,   LLM_TENSOR_LOOP_K_DOWN,   LLM_TENSOR_LOOP_V_DOWN,   LLM_TENSOR_LOOP_MLP_DOWN};
            static const llm_tensor gain[4]   = {LLM_TENSOR_LOOP_ATTN_GAIN,   LLM_TENSOR_LOOP_K_GAIN,   LLM_TENSOR_LOOP_V_GAIN,   LLM_TENSOR_LOOP_MLP_GAIN};
            static const llm_tensor forget[4] = {LLM_TENSOR_LOOP_ATTN_FORGET, LLM_TENSOR_LOOP_K_FORGET, LLM_TENSOR_LOOP_V_FORGET, LLM_TENSOR_LOOP_MLP_FORGET};

            const int64_t n_rank = hparams.loop_gate_rank;

            for (int m = 0; m < 4; ++m) {
                layer.loop_down[m]   = create_tensor(tn(down[m],   "weight", i), {2*n_embd, n_rank}, 0);
                layer.loop_gain[m]   = create_tensor(tn(gain[m],   "weight", i), {n_rank, n_embd}, 0);
                layer.loop_forget[m] = create_tensor(tn(forget[m], "weight", i), {n_rank, n_embd}, 0);
            }
        }
    }

    if (loop) {
        const int64_t n_query = 2*(hparams.loop_core_end - hparams.loop_core_start);

        loop_res_query    = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, n_query}, 0);
        loop_kv_res_query = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, n_query}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale:
// x*min(1, t/|x|) = t*x/max(|x|, t)
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    return ggml_scale(ctx0, ggml_l2_norm(ctx0, cur, threshold), threshold);
}

// appends a block output to the RMS-normalized earlier blocks [n_embd, n_blocks, n_tokens] (null: none yet)
ggml_tensor * llama_model_inkling2::graph::push_block(ggml_tensor * blocks_norm, ggml_tensor * block) {
    ggml_tensor * b = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), 1e-6f);
    return blocks_norm ? ggml_concat(ctx0, blocks_norm, b, 1) : b;
}

// attention over the depth: mix the earlier block outputs (normalized, see push_block) and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(ggml_tensor * blocks_norm, ggml_tensor * cur, ggml_tensor * query) {
    ggml_tensor * states = push_block(blocks_norm, cur);

    ggml_tensor * queries = n_tokens == 1 ? ggml_reshape_3d(ctx0, query, n_embd, query->ne[1], 1) :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// loop message controller: (1 - tanh(F h))*cur + tanh(G h)*message, h = D [rms(cur); rms(message)]; module: q, k, v, mlp
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const llama_layer & layer, int module) {
    ggml_tensor * h = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    h = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, 1e-6f), 2*n_embd, n_tokens);
    h = build_lora_mm(layer.loop_down[module], h);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(layer.loop_gain[module],   h));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(layer.loop_forget[module], h));

    return ggml_add(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, forget, -1.0f, 1.0f), cur), ggml_mul(ctx0, gain, message));
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t d_conv       = n_conv - 1;

    ggml_tensor * past = ggml_view_3d(ctx0, history, d_conv, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    // store the last d_conv inputs: slot 0 is the final state, slot s the state s tokens back (rollback)
    {
        const auto *  mctx_cur  = inp->mctx;
        const int64_t K         = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
        const size_t  row_size  = ggml_row_size(state->type, hparams.n_embd_r()); // state is one row of n_embd_r per cell
        const size_t  offs      = offset*d_conv*ggml_element_size(state);

        for (int64_t slot = 0; slot < n_written; ++slot) {
            ggml_tensor * last = ggml_view_3d(ctx0, x, d_conv, n_channels, n_seqs, x->nb[1], x->nb[2],
                    (x->ne[0] - d_conv - slot)*ggml_element_size(x));
            ggml_tensor * cell = ggml_view_2d(ctx0, state, d_conv*n_channels, n_seqs, row_size,
                    ((size_t) slot*mctx_cur->get_size() + mctx_cur->get_head())*row_size + offs);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, cell));
        }
    }

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    auto * inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    auto * attn = inp->get_attn();
    auto * recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    ggml_tensor * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // do not normalize raw embeddings input (encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    std::vector<ggml_tensor *> blocks;
    ggml_tensor *              blocks_norm = nullptr; // blocks, normalized and concatenated for mix()

    // each pass of the looped core restarts from the prelude state and reads the previous pass's blocks through messages
    const int64_t n_loop_core = hparams.loop_n_pass > 1 ? hparams.loop_core_end - hparams.loop_core_start : 0;

    std::vector<ggml_tensor *> prelude_blocks;
    ggml_tensor *              prelude_norm = nullptr;
    ggml_tensor *              prelude_cur  = nullptr;
    ggml_tensor *              bank_norm    = nullptr;
    ggml_tensor *              bank_last    = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const int    ip    = hparams.loop_layer(il);

        ggml_tensor * messages[4] = {nullptr, nullptr, nullptr, nullptr};

        if (n_loop_core > 0 && ip >= (int) hparams.loop_core_start && ip < (int) hparams.loop_core_end) {
            const int pass = (il - (int) hparams.loop_core_start)/n_loop_core;

            if (ip == (int) hparams.loop_core_start) {
                if (pass == 0) {
                    prelude_blocks = blocks;
                    prelude_norm   = blocks_norm;
                    prelude_cur    = cur;
                } else {
                    bank_norm = nullptr;
                    for (size_t ib = prelude_blocks.size() + 1; ib < blocks.size(); ++ib) {
                        bank_norm = push_block(bank_norm, blocks[ib]);
                    }
                    bank_last   = cur;
                    blocks      = prelude_blocks;
                    blocks_norm = prelude_norm;
                    cur         = prelude_cur;
                }
            }

            if (pass > 0) {
                const int64_t row = 2*(ip - (int) hparams.loop_core_start);

                ggml_tensor * q_attn = ggml_view_2d(ctx0, model.loop_res_query,    n_embd, 2, model.loop_res_query->nb[1],    row*model.loop_res_query->nb[1]);
                ggml_tensor * q_kv   = ggml_view_2d(ctx0, model.loop_kv_res_query, n_embd, 2, model.loop_kv_res_query->nb[1], row*model.loop_kv_res_query->nb[1]);

                // rows: (attn, mlp) and (k, v)
                ggml_tensor * m_attn = mix(bank_norm, bank_last, q_attn);
                ggml_tensor * m_kv   = mix(bank_norm, bank_last, q_kv);

                for (int j = 0; j < 2; ++j) {
                    messages[j == 0 ? 0 : 3] = ggml_cont(ctx0, ggml_view_2d(ctx0, m_attn, n_embd, n_tokens, m_attn->nb[2], j*m_attn->nb[1]));
                    messages[j == 0 ? 1 : 2] = ggml_cont(ctx0, ggml_view_2d(ctx0, m_kv,   n_embd, n_tokens, m_kv->nb[2],   j*m_kv->nb[1]));
                }
            }
        }

        const bool    is_swa       = hparams.is_swa(il);
        const int64_t n_head       = hparams.n_head(il);
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_head  = hparams.n_embd_head_k(il);
        const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

        ggml_tensor * state   = recr->mctx->get_r_l(il);
        ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

        res->t_layer_inp[il] = cur;

        ggml_tensor * inpSA = cur;

        // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
        ggml_tensor * inputs[3] = {cur, cur, cur};
        if (layer.attn_res_query) {
            ggml_tensor * mixed = mix(blocks_norm, cur, layer.attn_res_query);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], (is_swa ? 0 : j)*mixed->nb[1]));
            }
        }

        if (messages[0]) {
            // the messages enter before the input norms, so SWA layers normalize three inputs
            for (int j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], messages[j], layer, j);
            }
            cb(inputs[0], "loop_q_inp", il);
        }

        if (is_swa && messages[0]) {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            }
        } else if (is_swa) {
            inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
        }

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
            ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
            ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
            ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

            Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip);
            Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip);
            Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
            Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);

            if (!is_swa) {
                Qcur = ggml_mul(ctx0, Qcur, tau);
                Rcur = ggml_mul(ctx0, Rcur, tau);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            cb(Rcur, "Rcur", il);

            ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;
            ggml_tensor * mask    = is_swa ? attn->get_kq_mask_swa() : attn->get_kq_mask();

            const int64_t n_stream = mask->ne[3];
            const int64_t n_kv     = rel_pos->ne[0];

            // per-query sink logits [n_head, n_seq_tokens, n_stream]
            ggml_tensor * sinks = ggml_reshape_3d(ctx0, build_lora_mm(layer.attn_sink_proj, inputs[0]), n_head, n_tokens/n_stream, n_stream);

            // project before gathering: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            // the dense bias (and kq) is n_kv x n_tokens x n_head: chunk query rows past the budget,
            // sized for a full cache so the chunking does not change while it fills
            const int64_t n_seq_tokens = n_tokens/n_stream;
            const int64_t n_kv_max = std::max<int64_t>(n_kv, is_swa ? (int64_t) hparams.n_swa + cparams.n_ubatch : (int64_t) cparams.n_ctx_seq);
            int64_t n_rows_max = std::max({ int64_t(1), inkling2_kq_bias_max()/(n_kv_max*n_head*n_stream),
                    (n_seq_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max });
            if (n_rows_max < n_seq_tokens) {
                const int64_t n_chunks = (n_seq_tokens + n_rows_max - 1)/n_rows_max;
                n_rows_max = std::min(n_seq_tokens, GGML_PAD((n_seq_tokens + n_chunks - 1)/n_chunks, 16));
            }
            auto rows = [&](ggml_tensor * t, int d, int64_t r0, int64_t nr) {
                int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                ne[d] = nr;
                return ggml_cont(ctx0, ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], r0*t->nb[d]));
            };
            std::vector<ggml_tensor *> outs;
            for (int64_t r0 = 0; r0 < n_seq_tokens; r0 += n_rows_max) {
                const int64_t nr  = std::min(n_rows_max, n_seq_tokens - r0);
                const bool    all = nr == n_seq_tokens;
                ggml_tensor * q_c  = Qcur;
                ggml_tensor * kq_b = nullptr;
                ggml_tensor * t_c  = all ? sinks : ggml_view_3d(ctx0, sinks, n_head, nr, n_stream, sinks->nb[1], sinks->nb[2], r0*sinks->nb[1]);
                if (all) {
                    kq_b = ggml_get_rows(ctx0, table, rel_pos);
                    kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, nr, n_stream);
                } else {
                    q_c = ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                            n_embd_head, n_head, nr*n_stream);
                    ggml_tensor * table_c = ggml_reshape_4d(ctx0, table, table->ne[0], table->ne[1], n_seq_tokens, n_stream);
                    table_c = ggml_view_4d(ctx0, table_c, table_c->ne[0], table_c->ne[1], nr, n_stream,
                            table_c->nb[1], table_c->nb[2], table_c->nb[3], r0*table_c->nb[2]);
                    ggml_tensor *& rel_pos_dev = inp->self_rel_pos_dev[is_swa];
                    if (!rel_pos_dev) {
                        rel_pos_dev = ggml_cont(ctx0, rel_pos);
                    }
                    ggml_tensor * rel_pos_c = ggml_view_3d(ctx0, rel_pos_dev, n_kv, nr, n_stream,
                            rel_pos_dev->nb[1], n_seq_tokens*rel_pos_dev->nb[1], r0*rel_pos_dev->nb[1]);
                    kq_b = ggml_get_rows(ctx0, table_c, rel_pos_c);
                }
                cb(kq_b, "kq_b", il);

                // kq_b: the gathered bias [n_head, n_kv, nr, n_stream]; only the first chunk stores K and V
                ggml_tensor * out = build_attn_rel(inp, q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr,
                        kq_b, t_c, 1.0f/sqrtf(n_embd_head), il, r0);
                outs.push_back(all ? out : ggml_reshape_3d(ctx0, out, out->ne[0], nr, n_stream));
            }
            while (outs.size() > 1) {
                std::vector<ggml_tensor *> next;
                for (size_t c = 0; c + 1 < outs.size(); c += 2) {
                    next.push_back(ggml_concat(ctx0, outs[c], outs[c + 1], 1));
                }
                if (outs.size() % 2) {
                    next.push_back(outs.back());
                }
                outs = std::move(next);
            }
            cur = n_rows_max >= n_seq_tokens ? outs[0] : ggml_reshape_2d(ctx0, outs[0], outs[0]->ne[0], n_tokens);

            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));
            cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
            cb(cur, "attn_out", il);
        }

        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        if (ip % hparams.attn_res_block_size == 0) {
            blocks.push_back(inpSA);
            blocks_norm = push_block(blocks_norm, inpSA);
        } else {
            cur = ggml_add(ctx0, inpSA, cur);
        }

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // MoE
        cur = ggml_reshape_2d(ctx0, mix(blocks_norm, cur, layer.ffn_res_query), n_embd, n_tokens);
        if (messages[3]) {
            cur = control(cur, messages[3], layer, 3);
            cb(cur, "loop_ffn_inp", il);
        }
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        cur = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                nullptr,
                nullptr,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SITU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                nullptr,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);

        cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_inp, cur);
        cb(cur, "l_out", il);
    }

    cur = ggml_reshape_2d(ctx0, mix(blocks_norm, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
