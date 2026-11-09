// Replay through the upstream llama.cpp graph; only observe named tensors.
//
// This is the harness the Dory oracle uses (`tools/oracle/dory/oracle.cpp`)
// extended with the mixture-of-experts checkpoints that
// `nemotron_h_moe` adds:
//
//   ffn_moe_logits   the router output
//   ffn_moe_probs    sigmoid(logits)
//   ffn_moe_topk     the selected expert ids (I32, not F32)
//   ffn_moe_weights  the routing weights of the selected experts
//   ffn_moe_out      the routed term, before the shared expert
//   ffn_shexp        the shared expert output
//   ffn_out          moe_out + ffn_shexp
//
// Every F32 checkpoint is written as raw little-endian floats and
// indexed by the manifest; the I32 top-k ids are written as raw
// little-endian int32.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct trace_state {
    std::string path;
    std::ofstream manifest;
    int step = 0;
    int first = 0;
    int rows = 1;

    explicit trace_state(const char * p) : path(p ? p : ""), manifest(path) {
        if (path.empty() || !manifest) throw std::runtime_error("set RMI_PARITY_TRACE");
        if (const char * s = std::getenv("RMI_NEMOTRON_TRACE_FROM")) first = std::stoi(s);
    }

    void tokens(const char * name, const std::vector<llama_token> & ids) {
        manifest << "{\"name\":\"" << name << "\",\"token_ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) manifest << (i ? "," : "") << ids[i];
        manifest << "]}\n";
        manifest.flush();
    }
};

// The checkpoints this harness observes. `embd` / `l_out` / `result_norm` /
// `result_output` are the per-layer and final ones the Dory oracle already
// uses; the rest are the MoE ones.
static bool wanted(const std::string & name) {
    // `nemotron_h_block_out` is the layer output after the residual add,
    // emitted for **every** layer (`src/models/nemotron-h.cpp:244`). The MoE
    // `l_out` is the same node, but only reached through `build_ffn_layer`, so
    // without this the trace has no checkpoint at all for the 23 Mamba-2
    // layers and the 6 attention layers -- exactly the layers where a
    // divergence would have to be localised.
    //
    // The `mamba2_*` nodes come from `src/models/mamba-base.cpp` and are the
    // only checkpoints *inside* the SSM branch. `mamba2_xbc` (the conv1d
    // output after the SiLU) and `mamba2_ygated` (the scan output after the
    // z gate) are both contiguous, so they can be compared element by
    // element. `mamba2_y_add_d` cannot: it is a `ggml_view_4d` over the scan
    // output that inherits the *input* projection's strides
    // (`src/models/mamba-base.cpp:284`), and the CPU backend copies a
    // strided region verbatim, so the buffer would not hold the logical
    // order the Rust side produces.
    //
    // `mamba_out` is the branch output before the residual add, which is
    // the split point that separates "the SSM branch is wrong" from "the
    // residual add is wrong".
    return name == "embd" || name == "l_out" || name == "nemotron_h_block_out" ||
           name == "result_norm" || name == "result_output" ||
           name == "attn_norm" ||
           name == "mamba2_xbc" || name == "mamba2_dt" || name == "mamba2_z" ||
           name == "mamba2_y_add_d" || name == "mamba2_ygated" || name == "mamba2_normed" ||
           name == "mamba_out" ||
           name == "ffn_moe_logits" || name == "ffn_moe_probs" ||
           name == "ffn_moe_topk" || name == "ffn_moe_weights" ||
           name == "ffn_moe_out" || name == "ffn_shexp" || name == "ffn_out";
}

static bool capture(ggml_tensor * t, bool ask, void * opaque) {
    auto & trace = *static_cast<trace_state *>(opaque);
    std::string name = t->name;
    int layer = -1;
    const auto dash = name.rfind('-');
    if (dash != std::string::npos && name.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        layer = std::stoi(name.substr(dash + 1));
        name.resize(dash);
    }
    const bool selected = trace.step + trace.rows > trace.first && wanted(name);
    if (ask) return selected;
    if (!selected) return true;

    // The top-k ids are I32; everything else is F32.
    const bool is_ids = t->type == GGML_TYPE_I32;
    if (!is_ids && t->type != GGML_TYPE_F32) {
        throw std::runtime_error("unexpected checkpoint type for " + name);
    }
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());

    const bool last = name == "result_norm" || name == "result_output";
    // `embd`, `l_out`, `ffn_moe_*` and `ffn_out` are `[n, n_tokens]`:
    // `ne[0]` is the feature axis and `ne[1]` the token axis. Only the two
    // final checkpoints are narrowed to the last row before they are emitted.
    const int rows = last ? 1 : trace.rows;
    // The MoE checkpoints are per-layer and per-token. Most of them are
    // `[n, n_tokens]` -- one row per token. `ffn_moe_weights` is the
    // exception: `ggml_get_rows` produces `[1, n_expert_used, n_tokens]`,
    // so `ne[1]` is the top-k slot axis and the token axis is `ne[2]`.
    // Treating that as a batched checkpoint threw here.
    const bool slot_axis = name == "ffn_moe_weights";
    // `mamba2_z`, `mamba2_y_add_d` and `mamba2_ygated` are
    // `[head_dim, n_head, n_tokens, n_seqs]`: the comparable unit is a
    // whole `(head_dim, n_head)` block, because that is how the Rust side
    // lays the SSM branch out (`y_buf[h * head_dim + k]`). `mamba2_z` and
    // `mamba2_y_add_d` are strided views, so the writer below walks their
    // strides rather than assuming a contiguous row.
    const bool wide = name == "mamba2_ygated" || name == "mamba2_z" ||
                      name == "mamba2_y_add_d";
    if (t->ne[3] != 1) throw std::runtime_error("unexpected checkpoint shape");
    if (slot_axis) {
        if (t->ne[0] != 1 || t->ne[2] != rows) {
            throw std::runtime_error("unexpected ffn_moe_weights shape: [" +
                                     std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) +
                                     ", " + std::to_string(t->ne[2]) + "]");
        }
    } else if (wide) {
        if (t->ne[2] != 1) {
            throw std::runtime_error("unexpected mamba2_ygated shape: [" +
                                     std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) +
                                     ", " + std::to_string(t->ne[2]) + "]");
        }
    } else if (t->ne[2] != 1 || t->ne[1] != rows) {
        throw std::runtime_error("unexpected batched checkpoint shape for " + name +
                                 ": ne1=" + std::to_string(t->ne[1]) + " rows=" + std::to_string(rows));
    }
    for (int row = 0; row < rows; ++row) {
        const int step = trace.step + (last ? trace.rows - 1 : row);
        if (step < trace.first) continue;
        const std::string file = trace.path + "." + std::to_string(step) + "." + t->name + ".f32";
        std::ofstream binary(file, std::ios::binary);
        // The feature axis is `ne[0]` for the per-token checkpoints and
        // `ne[1]` for `ffn_moe_weights`; the token axis is `ne[1]` and `ne[2]`
        // respectively. Writing `ne[0]` values from offset `row*nb[1]` is what
        // makes the record comparable to the Rust side's row.
        const int64_t width = slot_axis ? t->ne[1] : (wide ? t->ne[0] * t->ne[1] : t->ne[0]);
        const size_t base = slot_axis ? row * t->nb[2] : (wide ? row * t->nb[2] : row * t->nb[1]);
        for (int64_t i = 0; i < width; ++i) {
            // Walk the tensor's own strides instead of assuming a
            // contiguous row. The SSM nodes are `ggml_view_4d`s whose
            // second axis is strided (the state-space duality layout),
            // and a raw memcpy of the region would hand back the memory
            // order rather than the logical one -- which for `y` means
            // head-major instead of the `k + h * head_dim` order the Rust
            // side produces.
            const int64_t i0 = wide ? (i % t->ne[0]) : i;
            const int64_t i1 = wide ? (i / t->ne[0]) : 0;
            uint32_t bits;
            std::memcpy(&bits, data.data() + base + i0 * t->nb[0] + i1 * t->nb[1], 4);
            const char bytes[] = {char(bits), char(bits >> 8), char(bits >> 16), char(bits >> 24)};
            binary.write(bytes, 4);
        }
        if (!binary) throw std::runtime_error("write checkpoint failed");
        trace.manifest << "{\"name\":\"" << name << "\",\"layer\":";
        if (layer < 0) trace.manifest << "null"; else trace.manifest << layer;
        trace.manifest << ",\"step\":" << step << ",\"shape\":[" << width << "],\"binary_path\":\"";
        for (char c : file) { if (c == '\\' || c == '"') trace.manifest << '\\'; trace.manifest << c; }
        trace.manifest << "\"}\n";
    }
    trace.manifest.flush();
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: nemotron-oracle MODEL RAW_PROMPT_FILE N_PREDICT");
        const int count = std::stoi(argv[3]);
        if (count < 0) throw std::runtime_error("negative N_PREDICT");
        std::ifstream prompt_file(argv[2]);
        if (!prompt_file) throw std::runtime_error("read prompt failed");
        const std::string text((std::istreambuf_iterator<char>(prompt_file)), {});
        trace_state trace(std::getenv("RMI_PARITY_TRACE"));
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("load model failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt(text.size() + 16);
        // `add_special = true` so the reference adds the same BOS the
        // repository does; `parse_special = true` keeps the ChatML
        // markers intact.
        const int n = llama_tokenize(vocab, text.data(), text.size(), prompt.data(), prompt.size(), true, true);
        if (n <= 0) throw std::runtime_error("tokenize failed or empty prompt");
        prompt.resize(n);
        trace.tokens("prompt_ids", prompt);
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(128, n + count + 1);
        const int batch = std::getenv("RMI_NEMOTRON_BATCH") ? std::stoi(std::getenv("RMI_NEMOTRON_BATCH")) : 1;
        if (batch < 1 || batch > 32) throw std::runtime_error("batch must be 1..32");
        cp.n_batch = cp.n_ubatch = batch;
        cp.n_threads = cp.n_threads_batch = std::getenv("RMI_NEMOTRON_THREADS") ? std::stoi(std::getenv("RMI_NEMOTRON_THREADS")) : 1;
        const std::string kv = std::getenv("RMI_NEMOTRON_KV") ? std::getenv("RMI_NEMOTRON_KV") : "f32";
        if (kv != "f32" && kv != "f16") throw std::runtime_error("unknown KV format");
        cp.type_k = cp.type_v = kv == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = false;
        cp.cb_eval = capture;
        cp.cb_eval_user_data = &trace;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("create context failed");
        std::vector<llama_token> batches;
        for (int first = 0; first < n; first += batch) batches.push_back(std::min(batch, n - first));
        batches.insert(batches.end(), count, 1);
        trace.tokens("prefill_batches", batches);
        const auto decode = [&](llama_token * tokens, int size) {
            trace.rows = size;
            if (llama_decode(ctx.get(), llama_batch_get_one(tokens, size))) throw std::runtime_error("decode failed");
            trace.step += size;
            std::cerr << "evaluated " << trace.step << " tokens\n";
        };
        for (int first = 0; first < n; first += batch) decode(prompt.data() + first, std::min(batch, n - first));
        std::vector<llama_token> generated;
        for (int i = 0; i < count; ++i) {
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            const auto best = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
            generated.push_back(best);
            auto token = llama_token(best);
            decode(&token, 1);
        }
        trace.tokens("generated_ids", generated);
        std::cerr << "Nemotron oracle: " << n << " prompt + " << count << " decode steps\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
