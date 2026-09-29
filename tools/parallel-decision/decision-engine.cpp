#include "decision-engine.h"

#include "chat.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <stdexcept>

namespace llama_decision {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active;  // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix;   // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // Greedy: follow single-path edges for free; return the options at the next divergence.
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    int tree_rows() const {
        int n = 0;
        for (const auto & p : node_prefix) {
            n += (int) (suffix.size() + p.size());
        }
        return n;
    }

    // Exact constrained distribution: log-softmax at each node over its allowed tokens,
    // summed along every candidate path, normalised over candidates.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<float>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp(x - mx);
            }
            const float lz = mx + (float) std::log(z);
            std::vector<float> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<float> path_lp(paths.size(), 0.0f);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it == opts.end()) {
                    throw std::runtime_error("candidate token missing from its trie node");
                }
                path_lp[i] += node_logp[n][it - opts.begin()];
            }
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        double z = 0;
        for (float x : path_lp) {
            z += std::exp(x - path_lp[best]);
        }
        probs.clear();
        for (float x : path_lp) {
            probs.push_back((float) (std::exp(x - path_lp[best]) / z));
        }
        winner       = best;
        path_score   = probs[best];
        scored_nodes = (int) node_prefix.size();
    }
};

} // namespace

// ---------------------------------------------------------------- engine

engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1),
      pad_branches(llama_model_is_recurrent(llama_get_model(ctx)) || llama_model_is_hybrid(llama_get_model(ctx))) {
    if (n_seqs < 3) {
        throw std::invalid_argument("a decision engine needs at least 3 sequences");
    }
}

tokens_t engine::tokenize(const std::string & text, bool add_special) const {
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
void engine::decode_parts(const std::vector<prompt_part> & parts) {
    // Stay inside the reserved micro-batch. llama_decode splits anything larger,
    // and a hybrid model only scores equal-length branches in one micro-batch.
    const int n_batch = (int) std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    auto flush = [&]() {
        const int rc = batch.n_tokens > 0 ? llama_decode(ctx, batch) : 0;
        common_batch_clear(batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                             : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (batch.n_tokens == n_batch) {
                flush();
            }
            common_batch_add(batch, (*p.toks)[i], p.pos0 + (llama_pos) i, { p.seq }, false);
        }
    }
    flush();
    llama_batch_free(batch);
}

// Restore (or build) the cached static prefix on seq_snap. Only this engine's own sequences are
// touched, so it can share a context with other users (e.g. server slots).
bool engine::prepare_prefix(const tokens_t & shared, bool allow_cache) {
    if (allow_cache && !shared.empty() && shared == cached &&
        llama_memory_seq_pos_max(mem, seq_snap) == (llama_pos) cached.size() - 1) {
        return true;
    }
    for (llama_seq_id s = seq_snap; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
    cached.clear();
    if (!shared.empty()) {
        decode_parts({ { &shared, 0, seq_snap } });
        cached = shared;
    }
    return false;
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
//
// Recurrent layers take a ubatch only when every sequence in it holds the same number of tokens
// (split_equal), so on recurrent/hybrid models uneven branches would split one decode into many
// passes. There each group is right-padded to its longest branch (longest first, so padding stays
// small). Every layer is causal, so the logits read at a branch's last real token don't see the
// padding, and the padded cells are removed with the branch.
std::vector<std::vector<float>> engine::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free) {
    std::vector<std::vector<float>> result(branches.size());
    std::vector<size_t> order(branches.size());
    for (size_t b = 0; b < order.size(); ++b) {
        order[b] = b;
    }
    if (pad_branches) {
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return branches[a].toks.size() > branches[b].toks.size();
        });
    }
    const int max_rows = (int) std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    size_t start = 0;
    while (start < order.size()) {
        const int width = (int) branches[order[start]].toks.size(); // padded length of this group
        size_t end  = start;
        int    rows = 0;
        while (end < order.size() && (int) (end - start) < n_free) {
            const int need = pad_branches ? width : (int) branches[order[end]].toks.size();
            if (rows + need > max_rows) {
                break;
            }
            rows += need;
            ++end;
        }
        if (end == start) {
            throw std::runtime_error("a decision suffix exceeds the batch size");
        }
        llama_batch batch = llama_batch_init(rows, 0, 1);
        std::vector<int> out_idx;
        for (size_t k = start; k < end; ++k) {
            const auto &       br  = branches[order[k]];
            const llama_seq_id seq = first + (llama_seq_id) (k - start);
            llama_memory_seq_rm(mem, seq, -1, -1);
            llama_memory_seq_cp(mem, br.trunk, seq, -1, -1);
            const int n = pad_branches ? width : (int) br.toks.size();
            for (int i = 0; i < n; ++i) {
                const bool last = i + 1 == (int) br.toks.size();
                if (last) {
                    out_idx.push_back(batch.n_tokens);
                }
                common_batch_add(batch, br.toks[std::min(i, (int) br.toks.size() - 1)], br.pos0 + (llama_pos) i, { seq }, last);
            }
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision branches"
                                             : "llama_decode failed on the decision branches (" + std::to_string(rc) + ")");
        }
        for (size_t k = start; k < end; ++k) {
            const float * logits = llama_get_logits_ith(ctx, out_idx[k - start]);
            for (llama_token t : branches[order[k]].cands) {
                result[order[k]].push_back(logits[t]);
            }
            llama_memory_seq_rm(mem, first + (llama_seq_id) (k - start), -1, -1);
        }
        start = end;
    }
    return result;
}

result engine::decide(const std::string & shared_text, const std::string & context_text,
                      const std::vector<field_input> & inputs, const options & opt) {
    batch_result b = decide_batch(shared_text, { context_text }, inputs, opt);
    result r = std::move(b.items[0]);
    r.cache_hit     = b.cache_hit;
    r.shared_tokens = b.shared_tokens;
    r.rounds        = b.rounds;
    r.prefill_ms    = b.prefill_ms;
    r.scoring_ms    = b.scoring_ms;
    return r;
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (contexts.empty()) {
        throw std::invalid_argument("a decision needs at least one context");
    }
    const tokens_t shared = tokenize(shared_text, true);
    std::vector<tokens_t> prefixes;
    for (const auto & text : contexts) {
        prefixes.push_back(tokenize(text, shared.empty()));
        if (prefixes.back().empty()) {
            throw std::invalid_argument("the decision context must not be empty");
        }
    }

    std::vector<decision_field> fields;
    int total    = 0;
    int branches = 0; // round-1 branches of one context
    for (const auto & in : inputs) {
        const size_t n = in.candidates.size();
        if (n < 1 || n > 255) {
            throw std::invalid_argument("each field needs 1-255 allowed values");
        }
        tokens_t suffix;
        std::vector<tokens_t> paths;
        if (opt.split_boundary) {
            for (const auto & c : in.candidates) {
                paths.push_back(tokenize(c + "\n", false));
            }
            suffix = tokenize(in.suffix, false);
        } else {
            // Tokenise each complete "suffix + value + terminator" and split at the longest token
            // prefix shared by every candidate: the value's first token is exactly what the model
            // would write itself (e.g. `":` then ` true`, not `": ` then `true`).
            std::vector<tokens_t> seqs;
            for (const auto & c : in.candidates) {
                seqs.push_back(tokenize(in.suffix + c + "\n", false));
            }
            size_t common = seqs[0].size();
            for (const auto & s : seqs) {
                size_t c = 0;
                while (c < std::min(common, s.size()) && s[c] == seqs[0][c]) {
                    ++c;
                }
                common = std::min({ common, c, s.size() - 1 });
            }
            suffix.assign(seqs[0].begin(), seqs[0].begin() + common);
            for (const auto & s : seqs) {
                paths.emplace_back(s.begin() + common, s.end());
            }
        }
        if (suffix.empty()) {
            throw std::invalid_argument("a field suffix must not be empty");
        }
        size_t max_path = 0;
        for (size_t a = 0; a < paths.size(); ++a) {
            max_path = std::max(max_path, paths[a].size());
            for (size_t b = 0; b < a; ++b) {
                const size_t m = std::min(paths[a].size(), paths[b].size());
                if (std::equal(paths[a].begin(), paths[a].begin() + m, paths[b].begin())) {
                    throw std::invalid_argument("two allowed values tokenise to colliding paths");
                }
            }
        }
        decision_field field(suffix, paths);
        field.build_nodes();
        field.use_tree = opt.mode == "tree" ? true : opt.mode == "greedy" ? false : n <= opt.tree_max;
        total    += field.use_tree ? field.tree_rows() : (int) (suffix.size() + max_path);
        branches += field.use_tree ? (int) field.node_prefix.size() : 1;
        fields.push_back(std::move(field));
    }

    batch_result out;
    out.shared_tokens = shared.size();
    out.rows          = total * (int) contexts.size();
    out.items.resize(contexts.size());

    const auto t0 = std::chrono::steady_clock::now();
    out.cache_hit = prepare_prefix(shared, opt.allow_cache);
    out.prefill_ms += ms_since(t0);

    // each context in a group holds one trunk sequence; the rest of the pool scores branches
    const size_t per_group = std::clamp<size_t>(n_pool / (1 + branches), 1, contexts.size());
    for (size_t g0 = 0; g0 < contexts.size(); g0 += per_group) {
        const size_t n_group = std::min(per_group, contexts.size() - g0);

        const auto tp = std::chrono::steady_clock::now();
        std::vector<prompt_part> parts;
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
            llama_memory_seq_rm(mem, trunk, -1, -1);
            if (!shared.empty()) {
                llama_memory_seq_cp(mem, seq_snap, trunk, -1, -1);
            }
            parts.push_back({ &prefixes[g0 + i], (llama_pos) shared.size(), trunk });
        }
        decode_parts(parts);
        llama_synchronize(ctx); // llama_decode is asynchronous: wait for the prefill so its time isn't billed to scoring
        out.prefill_ms += ms_since(tp);

        // round 1 carries every tree node and each greedy field's first step; later rounds only
        // continue greedy fields that are still open
        const auto ts = std::chrono::steady_clock::now();
        std::vector<std::vector<decision_field>> state(n_group, fields);
        bool first = true;
        while (true) {
            std::vector<branch> todo;
            std::vector<std::pair<size_t, size_t>> owner; // (context in group, field)
            for (size_t i = 0; i < n_group; ++i) {
                const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                const llama_pos    pos0  = (llama_pos) (shared.size() + prefixes[g0 + i].size());
                for (size_t f = 0; f < state[i].size(); ++f) {
                    auto & fd = state[i][f];
                    if (fd.use_tree) {
                        if (first) {
                            for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                                tokens_t ids = fd.suffix;
                                ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                                todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                                owner.push_back({ i, f });
                            }
                        }
                    } else {
                        tokens_t opts = fd.options();
                        if (!opts.empty()) {
                            tokens_t ids = fd.suffix;
                            ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                            todo.push_back({ trunk, pos0, ids, opts });
                            owner.push_back({ i, f });
                        }
                    }
                }
            }
            if (todo.empty()) {
                break;
            }
            const auto scores = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group);
            out.rounds += 1;
            std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group, std::vector<std::vector<std::vector<float>>>(fields.size()));
            for (size_t row = 0; row < owner.size(); ++row) {
                const auto [i, f] = owner[row];
                auto & fd = state[i][f];
                if (fd.use_tree) {
                    tree_scores[i][f].push_back(scores[row]);
                } else {
                    const auto & s    = scores[row];
                    const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                    double denom = 0;
                    for (float x : s) {
                        denom += std::exp(x - s[best]);
                    }
                    fd.select(todo[row].cands[best], (float) (1.0 / denom));
                }
            }
            if (first) {
                for (size_t i = 0; i < n_group; ++i) {
                    for (size_t f = 0; f < fields.size(); ++f) {
                        if (state[i][f].use_tree) {
                            state[i][f].finish_tree(tree_scores[i][f]);
                        }
                    }
                }
            }
            first = false;
        }
        for (size_t i = 0; i < n_group; ++i) {
            llama_memory_seq_rm(mem, seq_pool + (llama_seq_id) i, -1, -1);
            result & r = out.items[g0 + i];
            r.context_tokens = prefixes[g0 + i].size();
            r.rows           = total;
            for (auto & fd : state[i]) {
                if (fd.use_tree && fd.probs.empty()) {
                    fd.finish_tree({});
                }
                r.fields.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
            }
        }
        out.scoring_ms += ms_since(ts);
    }
    return out;
}

// ---------------------------------------------------------------- schema compiler

namespace {

int decimal_places(double x) {
    for (int k = 0; k <= 9; ++k) {
        const double v = x * std::pow(10.0, k);
        if (std::fabs(v - std::llround(v)) < 1e-9 * std::max(1.0, std::fabs(v))) {
            return k;
        }
    }
    return 9;
}

std::string json_text(const std::string & s) {
    return json(s).dump();
}

field_spec make_field(const std::string & name, const std::string & type, const std::string & description,
                      const json & spec, bool json_schema) {
    field_spec f;
    f.name        = name;
    f.description = description;
    const std::string kind = type;
    if (kind == "boolean") {
        f.type    = "boolean";
        f.values  = { json(true), json(false) };
        f.encoded = { "true", "false" };
    } else if (kind == "enum" || kind == "choice" || kind == "selection") {
        const char * key = spec.contains("enum") ? "enum" : "choices";
        if (!spec.contains(key) || !spec.at(key).is_array()) {
            throw std::invalid_argument("field \"" + name + "\": enum fields need a list of choices");
        }
        f.type = "enum";
        for (const auto & c : spec.at(key)) {
            if (!c.is_string()) {
                throw std::invalid_argument("field \"" + name + "\": enum choices must be strings");
            }
            const std::string v = c.get<std::string>();
            f.values.push_back(json(v));
            f.encoded.push_back(json_text(v));
        }
    } else if (kind == "integer") {
        if (!spec.contains("minimum") || !spec.contains("maximum") ||
            !spec.at("minimum").is_number_integer() || !spec.at("maximum").is_number_integer()) {
            throw std::invalid_argument("field \"" + name + "\": integer fields need integer minimum and maximum");
        }
        const long long lo = spec.at("minimum").get<long long>(), hi = spec.at("maximum").get<long long>();
        if (hi < lo || hi - lo + 1 > 255) {
            throw std::invalid_argument("field \"" + name + "\": integer bounds must define 1-255 values");
        }
        f.type = "integer";
        for (long long v = lo; v <= hi; ++v) {
            f.values.push_back(json(v));
            f.encoded.push_back(std::to_string(v));
            f.numbers.push_back((double) v);
        }
    } else if (kind == "number") {
        const char * step_key = json_schema ? "multipleOf" : "step";
        if (!spec.contains("minimum") || !spec.contains("maximum") || !spec.contains(step_key)) {
            throw std::invalid_argument("field \"" + name + "\": number fields need minimum, maximum and " + step_key);
        }
        const double lo = spec.at("minimum").get<double>(), hi = spec.at("maximum").get<double>(),
                     step = spec.at(step_key).get<double>();
        if (!(step > 0) || !(hi >= lo)) {
            throw std::invalid_argument("field \"" + name + "\": number needs ordered bounds and a positive step");
        }
        const double    count = (hi - lo) / step;
        const long long n     = std::llround(count);
        if (std::fabs(count - (double) n) > 1e-7 || n < 0 || n > 254) {
            throw std::invalid_argument("field \"" + name + "\": the number grid must include both ends and hold 1-255 values");
        }
        const int places = std::max({ decimal_places(lo), decimal_places(hi), decimal_places(step) });
        f.type = "number";
        for (long long i = 0; i <= n; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.*f", places, lo + (double) i * step);
            const double v = std::strtod(buf, nullptr);
            f.values.push_back(json(v));
            f.encoded.push_back(buf); // fixed width: every value has the same shape
            f.numbers.push_back(v);
        }
    } else {
        throw std::invalid_argument("field \"" + name + "\": supported types are boolean, enum, integer and number");
    }
    if (f.values.empty() || f.values.size() > 255) {
        throw std::invalid_argument("field \"" + name + "\" needs 1-255 allowed values");
    }
    for (size_t a = 0; a < f.encoded.size(); ++a) {
        for (size_t b = 0; b < a; ++b) {
            if (f.encoded[a] == f.encoded[b]) {
                throw std::invalid_argument("field \"" + name + "\" has duplicate allowed values");
            }
        }
    }
    const std::string agg = spec.value("aggregate", spec.value("x-aggregate", std::string("mode")));
    const bool numeric = f.type == "integer" || f.type == "number";
    if (agg != "mode" && !(numeric && (agg == "median" || agg == "mean"))) {
        throw std::invalid_argument("field \"" + name + "\": aggregate must be mode, or median/mean for numeric fields");
    }
    f.aggregate = agg;
    return f;
}

} // namespace

compiled_schema compile_schema(const json & schema, const std::string & instructions) {
    if (!schema.is_object()) {
        throw std::invalid_argument("\"schema\" must be an object");
    }
    compiled_schema cs;
    const bool json_schema = schema.contains("properties");
    const json & props = json_schema ? schema.at("properties") : schema;
    if (!props.is_object() || props.size() < 1 || props.size() > 32) {
        throw std::invalid_argument("the schema must define 1-32 fields");
    }
    for (const auto & e : props.items()) {
        const json & spec = e.value();
        if (!spec.is_object()) {
            throw std::invalid_argument("field \"" + e.key() + "\" must be an object");
        }
        std::string type = spec.value("type", std::string());
        if (spec.contains("enum")) {
            type = "enum";
        }
        std::string description = spec.value("description", std::string());
        if (!json_schema && description.empty()) {
            throw std::invalid_argument("field \"" + e.key() + "\" needs a description");
        }
        cs.specs.push_back(make_field(e.key(), type, description, spec, json_schema));
    }

    std::string catalog;
    for (const auto & f : cs.specs) {
        // the value's common leading characters are fixed in the suffix; only the rest is scored
        std::string common = f.encoded[0];
        for (const auto & v : f.encoded) {
            size_t c = 0;
            while (c < std::min(common.size(), v.size()) && common[c] == v[c]) {
                ++c;
            }
            common.resize(c);
        }
        field_input in;
        in.suffix = "  " + json_text(f.name) + ": " + common;
        for (const auto & v : f.encoded) {
            in.candidates.push_back(v.substr(common.size()));
        }
        cs.inputs.push_back(in);

        std::string allowed;
        for (size_t i = 0; i < f.encoded.size(); ++i) {
            allowed += (i ? ", " : "") + f.encoded[i];
        }
        catalog += (catalog.empty() ? "" : "\n") + json_text(f.name) + (f.description.empty() ? "" : ": " + f.description) +
                   "\nAllowed values: " + allowed;
    }
    cs.system_text = "Select the requested field value from its allowed values, based on the context. "
                     "Respond with the JSON value only.\n\nFields:\n" + catalog + "\n" + instructions;
    return cs;
}

std::pair<std::string, std::string> render_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                  const std::string & system_text, const std::string & context) {
    if (tmpls == nullptr) {
        return { system_text + "\nContext:\n", context + "\nOutput:\n{\n" };
    }
    static const std::string sentinel = "\x1f<<decision-context>>\x1f";
    common_chat_templates_inputs in;
    in.use_jinja             = use_jinja;
    in.add_generation_prompt = true;
    in.enable_thinking       = false;
    common_chat_msg sys;
    sys.role    = "system";
    sys.content = system_text;
    common_chat_msg usr;
    usr.role    = "user";
    usr.content = sentinel;
    in.messages = { sys, usr };
    const std::string prompt = common_chat_templates_apply(tmpls, in).prompt;
    const size_t at = prompt.find(sentinel);
    if (at == std::string::npos) {
        throw std::runtime_error("the chat template did not keep the user message");
    }
    return { prompt.substr(0, at), context + prompt.substr(at + sentinel.size()) + "{\n" };
}

json assemble(const compiled_schema & cs, const result & r) {
    json decision = json::object();
    json fields   = json::object();
    for (size_t i = 0; i < cs.specs.size(); ++i) {
        const auto & sp = cs.specs[i];
        const auto & fr = r.fields[i];
        int idx = fr.winner;
        json f = json::object();
        const bool numeric = !sp.numbers.empty();
        if (numeric && fr.probs.size() == sp.values.size()) {
            std::vector<int> order(sp.values.size());
            for (size_t k = 0; k < order.size(); ++k) {
                order[k] = (int) k;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return sp.numbers[a] < sp.numbers[b]; });
            auto quantile = [&](double q) {
                double acc = 0;
                for (int k : order) {
                    acc += fr.probs[k];
                    if (acc >= q) {
                        return k;
                    }
                }
                return order.back();
            };
            double mean = 0;
            for (size_t k = 0; k < sp.numbers.size(); ++k) {
                mean += sp.numbers[k] * fr.probs[k];
            }
            if (sp.aggregate == "median") {
                idx = quantile(0.5);
            } else if (sp.aggregate == "mean") {
                idx = 0;
                for (size_t k = 1; k < sp.numbers.size(); ++k) {
                    if (std::fabs(sp.numbers[k] - mean) < std::fabs(sp.numbers[idx] - mean)) {
                        idx = (int) k;
                    }
                }
            }
            json interval = json::array();
            interval.push_back(sp.values[quantile(0.1)]);
            interval.push_back(sp.values[quantile(0.9)]);
            f["interval_p10_p90"] = interval;
        }
        if (idx < 0 || idx >= (int) sp.values.size()) {
            throw std::runtime_error("field \"" + sp.name + "\" has no selected value");
        }
        decision[sp.name] = sp.values[idx];
        f["value"]        = sp.values[idx];
        f["probability"]  = (double) (fr.probs.size() == sp.values.size() ? fr.probs[idx] : fr.path_score);
        f["scored_nodes"] = fr.scored_nodes;
        f["tree"]         = fr.tree;
        fields[sp.name]   = f;
    }
    json out = json::object();
    out["decision"] = decision;
    out["fields"]   = fields;
    return out;
}

namespace {

constexpr const char * k_intern_system =
    "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. "
    "For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object "
    "mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.";

constexpr const char * k_answer_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
constexpr double       k_intern_temperature = 1.99241824;
constexpr int          k_answer_symbol_count = 62;

struct intern_option {
    std::string label;
    std::string description;
    int         value_index;
};

std::string field_instructions(const std::string & global, const field_spec & sp) {
    if (global.empty()) {
        return sp.description;
    }
    if (sp.description.empty()) {
        return global;
    }
    return global + " " + sp.description;
}

std::vector<intern_option> intern_options(const field_spec & sp) {
    std::vector<intern_option> out;
    if (sp.type == "boolean") {
        if (sp.values.size() != 2) {
            throw std::runtime_error("field \"" + sp.name + "\" is not a boolean pair");
        }
        out.push_back({ "no", "The answer is no (negative, or disagree with the claim).", 1 });
        out.push_back({ "yes", "The answer is yes (affirmative, or align with the claim).", 0 });
        return out;
    }
    if (sp.type == "enum") {
        for (int i = 0; i < (int) sp.values.size(); ++i) {
            const std::string label = sp.values[i].get<std::string>();
            out.push_back({ label, label, i });
        }
        return out;
    }
    for (int i = 0; i < (int) sp.encoded.size(); ++i) {
        out.push_back({ sp.encoded[i], sp.encoded[i], i });
    }
    return out;
}

std::vector<double> softmax_prefix(const std::vector<float> & logits, int n) {
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        max_logit = std::max(max_logit, (double) logits[i]);
    }
    std::vector<double> probs(n);
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp((double) logits[i] - max_logit);
        sum += probs[i];
    }
    for (double & p : probs) {
        p /= sum;
    }
    return probs;
}

// Published calibration. The chosen symbol stays the softmax argmax.
void scale_temperature(std::vector<double> & probs) {
    double max_log = -std::numeric_limits<double>::infinity();
    std::vector<double> logs(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        logs[i] = probs[i] > 0.0 ? std::log(probs[i]) : -std::numeric_limits<double>::infinity();
        max_log = std::max(max_log, logs[i]);
    }
    double sum = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = std::exp((logs[i] - max_log) / k_intern_temperature);
        sum += probs[i];
    }
    for (double & p : probs) {
        p /= sum;
    }
}

int best_option(const std::vector<double> & probs, const std::vector<intern_option> & opts) {
    int best = 0;
    for (int i = 1; i < (int) probs.size(); ++i) {
        if (probs[i] > probs[best] || (probs[i] == probs[best] && opts[i].label < opts[best].label)) {
            best = i;
        }
    }
    return best;
}

llama_token one_token(const llama_vocab * vocab, const std::string & text) {
    const tokens_t toks = common_tokenize(vocab, text, /*add_special=*/ false, /*parse_special=*/ true);
    if (toks.size() != 1) {
        throw std::runtime_error("\"" + text + "\" is not a single token");
    }
    return toks[0];
}

struct intern_batch {
    llama_batch batch;
    explicit intern_batch(int n) : batch(llama_batch_init(n, 0, 1)) {}
    ~intern_batch() { llama_batch_free(batch); }
    intern_batch(const intern_batch &) = delete;
    intern_batch & operator=(const intern_batch &) = delete;
};

// One causal forward. Logits are taken at the token before each <decision> marker.
std::vector<std::vector<float>> intern_forward(llama_context * ctx, const tokens_t & toks,
                                               const std::vector<int> & score_at, int n_fields,
                                               const std::vector<llama_token> & symbols) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        throw std::runtime_error("decision context has no memory");
    }
    llama_memory_clear(mem, true);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const int n_chunk = std::max(1, (int) std::min(llama_n_batch(ctx), llama_n_ubatch(ctx)));
    intern_batch held(n_chunk);
    llama_batch & batch = held.batch;

    std::vector<int> slot_field;
    int logits_in_batch = 0;
    std::vector<std::vector<float>> rows(n_fields);

    auto flush = [&]() {
        if (batch.n_tokens == 0) {
            return;
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                             : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (slot_field[i] < 0) {
                continue;
            }
            const float * row = llama_get_logits_ith(ctx, i);
            if (row == nullptr) {
                throw std::runtime_error("missing logits at a decision marker");
            }
            std::vector<float> got(symbols.size());
            for (size_t s = 0; s < symbols.size(); ++s) {
                if (symbols[s] < 0 || symbols[s] >= n_vocab) {
                    throw std::runtime_error("answer symbol token is outside the vocab");
                }
                got[s] = row[symbols[s]];
            }
            rows[slot_field[i]] = std::move(got);
        }
        common_batch_clear(batch);
        slot_field.clear();
        logits_in_batch = 0;
    };

    std::vector<int> field_of(toks.size(), -1);
    for (int field = 0; field < n_fields; ++field) {
        const int at = score_at[field];
        if (at < 0 || at >= (int) toks.size()) {
            throw std::runtime_error("decision marker is outside the prompt");
        }
        field_of[at] = field;
    }

    for (int i = 0; i < (int) toks.size(); ++i) {
        const bool want = field_of[i] >= 0;
        // The decision context reserves one output row per sequence, so each
        // micro-batch carries at most one marker logit.
        if (want && logits_in_batch > 0) {
            flush();
        }
        if (batch.n_tokens == n_chunk) {
            flush();
        }
        common_batch_add(batch, toks[i], (llama_pos) i, { 0 }, want);
        slot_field.push_back(want ? field_of[i] : -1);
        if (want) {
            ++logits_in_batch;
        }
    }
    flush();

    for (int field = 0; field < n_fields; ++field) {
        if (rows[field].empty()) {
            throw std::runtime_error("a decision field was not scored");
        }
    }
    return rows;
}

} // namespace

bool is_intern_decision(const llama_vocab * vocab) {
    if (vocab == nullptr) {
        return false;
    }
    const tokens_t toks = common_tokenize(vocab, "<decision>", /*add_special=*/ false, /*parse_special=*/ true);
    if (toks.size() != 1) {
        return false;
    }
    return common_token_to_piece(vocab, toks[0], true) == "<decision>";
}

json decide_intern(llama_context * ctx, const json & body, const std::string & model_name) {
    if (ctx == nullptr) {
        throw std::runtime_error("decision context is not loaded");
    }
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const llama_token marker = one_token(vocab, "<decision>");
    std::vector<llama_token> symbols;
    symbols.reserve(k_answer_symbol_count);
    for (int i = 0; i < k_answer_symbol_count; ++i) {
        symbols.push_back(one_token(vocab, std::string(1, k_answer_symbols[i])));
    }

    const std::string instructions = body.value("instructions", std::string());
    const compiled_schema cs = compile_schema(body.at("schema"), instructions);
    std::vector<std::vector<intern_option>> options;
    options.reserve(cs.specs.size());
    for (const auto & sp : cs.specs) {
        auto opts = intern_options(sp);
        if (opts.empty() || (int) opts.size() > k_answer_symbol_count) {
            throw std::invalid_argument("field \"" + sp.name + "\" needs 1-62 options");
        }
        options.push_back(std::move(opts));
    }

    json skeleton = json::object();
    for (const auto & sp : cs.specs) {
        skeleton[sp.name] = "<decision>";
    }
    const std::string skeleton_text = skeleton.dump(4, ' ', false);

    if (!body.contains("contexts") || !body.at("contexts").is_array() || body.at("contexts").empty()) {
        throw std::invalid_argument("\"contexts\" must be an array of 1-256 strings");
    }

    json results = json::array();
    double prefill_ms = 0;
    long long prompt_tokens = 0;
    long long scored_rows = 0;
    for (const auto & context_value : body.at("contexts")) {
        if (!context_value.is_string() || context_value.get<std::string>().empty()) {
            throw std::invalid_argument("every entry of \"contexts\" must be a non-empty string");
        }
        const std::string context = context_value.get<std::string>();
        std::string schema_text;
        for (size_t f = 0; f < cs.specs.size(); ++f) {
            if (f != 0) {
                schema_text += '\n';
            }
            schema_text += cs.specs[f].name + ": " + field_instructions(instructions, cs.specs[f]);
            for (size_t i = 0; i < options[f].size(); ++i) {
                schema_text += "\n    ";
                schema_text += k_answer_symbols[i];
                schema_text += " = ";
                schema_text += options[f][i].label;
                schema_text += ": ";
                schema_text += options[f][i].description;
            }
        }
        const std::string state = json(context).dump(-1, ' ', false);
        const std::string user = "Return one answer for every field using the supplied answer symbols.\n\n## State\n" +
                                 state + "\n## Decision schema\n" + schema_text;
        if (user.find("<decision>") != std::string::npos) {
            throw std::invalid_argument("Reserved decision marker appears in input evidence");
        }
        const std::string prompt = std::string("<|im_start|>system\n") + k_intern_system +
                                   "<|im_end|>\n<|im_start|>user\n" + user +
                                   "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + skeleton_text +
                                   "<|im_end|>\n";
        const tokens_t toks = common_tokenize(vocab, prompt, /*add_special=*/ false, /*parse_special=*/ true);
        if (toks.empty() || (int) toks.size() > (int) llama_n_ctx(ctx)) {
            throw std::invalid_argument("decision prompt exceeds the context");
        }
        std::vector<int> score_at;
        for (int i = 0; i < (int) toks.size(); ++i) {
            if (toks[i] == marker) {
                if (i < 1) {
                    throw std::runtime_error("decision marker has no preceding token");
                }
                score_at.push_back(i - 1);
            }
        }
        if (score_at.size() != cs.specs.size()) {
            throw std::runtime_error("decision marker count does not match the schema (" + std::to_string(score_at.size()) +
                                     " markers, " + std::to_string(cs.specs.size()) + " fields, " + std::to_string(toks.size()) +
                                     " tokens)");
        }

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::vector<float>> rows = intern_forward(ctx, toks, score_at, (int) cs.specs.size(), symbols);
        prefill_ms += ms_since(t0);

        result scored;
        scored.context_tokens = toks.size();
        scored.rows           = (int) cs.specs.size();
        for (size_t f = 0; f < cs.specs.size(); ++f) {
            const int n = (int) options[f].size();
            std::vector<double> raw = softmax_prefix(rows[f], n);
            const int raw_best = best_option(raw, options[f]);
            std::vector<double> scaled = raw;
            scale_temperature(scaled);
            if (best_option(scaled, options[f]) != raw_best) {
                throw std::runtime_error("temperature scaling changed the decision for \"" + cs.specs[f].name + "\"");
            }
            std::vector<float> aligned(cs.specs[f].values.size(), 0.0f);
            for (int i = 0; i < n; ++i) {
                aligned[options[f][i].value_index] = (float) scaled[i];
            }
            const int winner = options[f][raw_best].value_index;
            scored.fields.push_back({ winner, aligned[winner], n, false, std::move(aligned) });
        }
        json item = assemble(cs, scored);
        item["usage"] = { { "context_tokens", (long long) scored.context_tokens }, { "scored_rows", scored.rows } };
        results.push_back(std::move(item));
        prompt_tokens += (long long) toks.size();
        scored_rows += scored.rows;
    }

    json usage = json::object();
    usage["prompt_tokens"]  = prompt_tokens;
    usage["cached_tokens"]  = 0;
    usage["context_tokens"] = prompt_tokens;
    usage["scored_rows"]    = scored_rows;
    json timings = json::object();
    timings["prefill_ms"]      = prefill_ms;
    timings["scoring_ms"]      = 0.0;
    timings["total_ms"]        = prefill_ms;
    timings["rounds"]          = 1;
    timings["per_decision_ms"] = results.empty() ? 0.0 : prefill_ms / (double) results.size();

    json out = json::object();
    out["object"]  = "decision";
    out["results"] = std::move(results);
    out["model"]   = model_name;
    out["created"] = (long long) std::time(nullptr);
    out["usage"]   = std::move(usage);
    out["timings"] = std::move(timings);
    return out;
}

} // namespace llama_decision
