// Target-QK chunk scorer for KvFlashPager.
// Pooled post-RoPE K (per chunk, kv-head) vs last-token post-RoPE Q.
// Higher score = keep resident.

#pragma once

#include "ggml.h"
#include "kvflash_scorer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace common_kvflash {

struct KvFlashQkLayer {
    int32_t model_il = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_embd_head = 0;
    ggml_type type_k = GGML_TYPE_F32;
    ggml_type type_v = GGML_TYPE_F32;
};

inline bool kvflash_qk_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
            type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
}

inline void kvflash_qk_l2_normalize(float * x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        std::memset(x, 0, (size_t) n * sizeof(float));
        return;
    }
    const float inv = 1.0f / std::sqrt(sum);
    for (int i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

inline float kvflash_qk_dot(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Dequantize one K row of n_embd_gqa values into dst.
inline bool kvflash_qk_dequant_row(ggml_type type, const uint8_t * src, float * dst, int n) {
    if (n <= 0 || !src || !dst) {
        return false;
    }
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, (size_t) n * sizeof(float));
        return true;
    }
    if (type == GGML_TYPE_F16) {
        const ggml_fp16_t * in = (const ggml_fp16_t *) src;
        for (int i = 0; i < n; ++i) {
            dst[i] = ggml_fp16_to_fp32(in[i]);
        }
        return true;
    }
    if (type == GGML_TYPE_Q8_0) {
        const int block = 32;
        if (n % block != 0) {
            return false;
        }
        const int n_blocks = n / block;
        const size_t nb = ggml_row_size(type, n);
        if (nb != (size_t) n_blocks * (sizeof(ggml_fp16_t) + (size_t) block)) {
            return false;
        }
        const uint8_t * p = src;
        for (int b = 0; b < n_blocks; ++b) {
            ggml_fp16_t d16;
            std::memcpy(&d16, p, sizeof(d16));
            p += sizeof(d16);
            const float d = ggml_fp16_to_fp32(d16);
            for (int i = 0; i < block; ++i) {
                dst[b * block + i] = d * (float) (int8_t) p[i];
            }
            p += block;
        }
        return true;
    }
    if (type == GGML_TYPE_Q4_0) {
        const int block = 32;
        if (n % block != 0) {
            return false;
        }
        const int n_blocks = n / block;
        const size_t nb = ggml_row_size(type, n);
        if (nb != (size_t) n_blocks * (sizeof(ggml_fp16_t) + (size_t) block / 2)) {
            return false;
        }
        const uint8_t * p = src;
        for (int b = 0; b < n_blocks; ++b) {
            ggml_fp16_t d16;
            std::memcpy(&d16, p, sizeof(d16));
            p += sizeof(d16);
            const float d = ggml_fp16_to_fp32(d16);
            for (int i = 0; i < block / 2; ++i) {
                const int q = p[i];
                dst[b * block + i] = (float) ((q & 0x0f) - 8) * d;
                dst[b * block + i + block / 2] = (float) ((q >> 4) - 8) * d;
            }
            p += block / 2;
        }
        return true;
    }
    return false;
}

// Mean-pool chunk_tokens rows, then L2-normalize each kv-head.
inline bool kvflash_qk_pool_k_means(
        ggml_type type,
        const uint8_t * packed,
        int chunk_tokens,
        int n_embd_gqa,
        int n_head_kv,
        int n_embd_head,
        std::vector<float> & out_heads) {
    if (chunk_tokens <= 0 || n_head_kv <= 0 || n_embd_head <= 0 ||
        n_embd_gqa != n_head_kv * n_embd_head || !packed) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(type, n_embd_gqa);
    std::vector<float> row((size_t) n_embd_gqa);
    std::vector<float> acc((size_t) n_embd_gqa, 0.0f);
    for (int t = 0; t < chunk_tokens; ++t) {
        if (!kvflash_qk_dequant_row(type, packed + (size_t) t * row_bytes, row.data(), n_embd_gqa)) {
            return false;
        }
        for (int i = 0; i < n_embd_gqa; ++i) {
            acc[(size_t) i] += row[(size_t) i];
        }
    }
    const float inv = 1.0f / (float) chunk_tokens;
    out_heads.assign((size_t) n_embd_gqa, 0.0f);
    for (int i = 0; i < n_embd_gqa; ++i) {
        out_heads[(size_t) i] = acc[(size_t) i] * inv;
    }
    for (int h = 0; h < n_head_kv; ++h) {
        kvflash_qk_l2_normalize(out_heads.data() + (size_t) h * n_embd_head, n_embd_head);
    }
    return true;
}

// Score one chunk: max cosine over GQA group, mean over layers.
inline float kvflash_qk_score_chunk(
        const std::vector<KvFlashQkLayer> & layers,
        const std::vector<float> & k_means, // [n_layers * n_embd_gqa_l0] ragged via offsets
        const std::vector<int> & k_off,
        const std::vector<float> & q,       // [n_layers * n_head * n_embd_head] via q_off
        const std::vector<int> & q_off) {
    if (layers.empty()) {
        return -INFINITY;
    }
    float sum = 0.0f;
    int n = 0;
    for (size_t li = 0; li < layers.size(); ++li) {
        const KvFlashQkLayer & layer = layers[li];
        if (layer.n_head <= 0 || layer.n_head_kv <= 0 || layer.n_embd_head <= 0 ||
            layer.n_head % layer.n_head_kv != 0) {
            continue;
        }
        if (li >= k_off.size() || li >= q_off.size()) {
            continue;
        }
        const float * k = k_means.data() + k_off[li];
        const float * qq = q.data() + q_off[li];
        const int group = layer.n_head / layer.n_head_kv;
        float best_sum = 0.0f;
        int n_kv = 0;
        for (int kv = 0; kv < layer.n_head_kv; ++kv) {
            float best = -INFINITY;
            for (int g = 0; g < group; ++g) {
                const int qh = kv * group + g;
                const float cos = kvflash_qk_dot(
                        k + kv * layer.n_embd_head,
                        qq + qh * layer.n_embd_head,
                        layer.n_embd_head);
                best = std::max(best, cos);
            }
            if (std::isfinite(best)) {
                best_sum += best;
                ++n_kv;
            }
        }
        if (n_kv > 0) {
            sum += best_sum / (float) n_kv;
            ++n;
        }
    }
    return n > 0 ? sum / (float) n : -INFINITY;
}

class KvFlashTargetQkScorer : public KvFlashScorer {
public:
    bool configure(const std::vector<KvFlashQkLayer> & layers_in) {
        layers = layers_in;
        k_off.clear();
        q_off.clear();
        int k_cursor = 0;
        int q_cursor = 0;
        have_q = false;
        for (const KvFlashQkLayer & layer : layers) {
            if (layer.n_head <= 0 || layer.n_head_kv <= 0 || layer.n_embd_head <= 0 ||
                !kvflash_qk_type_supported(layer.type_k)) {
                return false;
            }
            k_off.push_back(k_cursor);
            q_off.push_back(q_cursor);
            k_cursor += layer.n_head_kv * layer.n_embd_head;
            q_cursor += layer.n_head * layer.n_embd_head;
        }
        k_stride = k_cursor;
        q_vals.assign((size_t) q_cursor, 0.0f);
        chunk_k.clear();
        chunk_valid.clear();
        return k_stride > 0;
    }

    bool set_k_chunk(int chunk, const std::vector<float> & packed_heads) {
        if (chunk < 0 || (int) packed_heads.size() != k_stride) {
            return false;
        }
        if (chunk >= (int) chunk_k.size()) {
            chunk_k.resize((size_t) chunk + 1);
            chunk_valid.resize((size_t) chunk + 1, 0);
        }
        chunk_k[(size_t) chunk] = packed_heads;
        chunk_valid[(size_t) chunk] = 1;
        return true;
    }

    void invalidate_k(int chunk) {
        if (chunk >= 0 && chunk < (int) chunk_valid.size()) {
            chunk_valid[(size_t) chunk] = 0;
        }
    }

    bool set_q_layer(size_t layer_idx, const float * q_heads) {
        if (layer_idx >= layers.size() || !q_heads) {
            return false;
        }
        const KvFlashQkLayer & layer = layers[layer_idx];
        float * dst = q_vals.data() + q_off[layer_idx];
        const int n = layer.n_head * layer.n_embd_head;
        std::memcpy(dst, q_heads, (size_t) n * sizeof(float));
        for (int h = 0; h < layer.n_head; ++h) {
            kvflash_qk_l2_normalize(dst + (size_t) h * layer.n_embd_head, layer.n_embd_head);
        }
        have_q = true;
        return true;
    }

    bool has_query() const {
        return have_q;
    }

    bool has_k(int chunk) const {
        return chunk >= 0 && chunk < (int) chunk_valid.size() && chunk_valid[(size_t) chunk];
    }

    float score_of(int chunk) const {
        if (!have_q || !has_k(chunk)) {
            return -std::numeric_limits<float>::infinity();
        }
        return kvflash_qk_score_chunk(layers, chunk_k[(size_t) chunk], k_off, q_vals, q_off);
    }

    bool score_chunks(const std::vector<int32_t> &,
                      int,
                      std::vector<float> & out) override {
        if (!have_q) {
            return false;
        }
        out.assign(chunk_valid.size(), -INFINITY);
        for (int chunk = 0; chunk < (int) chunk_valid.size(); ++chunk) {
            if (chunk_valid[(size_t) chunk]) {
                out[(size_t) chunk] = score_of(chunk);
            }
        }
        return true;
    }

    const std::vector<KvFlashQkLayer> & layer_specs() const {
        return layers;
    }

    int k_dim() const {
        return k_stride;
    }

private:
    std::vector<KvFlashQkLayer> layers;
    std::vector<int> k_off;
    std::vector<int> q_off;
    std::vector<std::vector<float>> chunk_k;
    std::vector<uint8_t> chunk_valid;
    std::vector<float> q_vals;
    int k_stride = 0;
    bool have_q = false;
};

} // namespace common_kvflash
