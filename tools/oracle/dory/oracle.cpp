// Replay through the pinned upstream graph; only observe named tensors.
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
        if (const char * s = std::getenv("RMI_DORY_TRACE_FROM")) first = std::stoi(s);
    }

    void tokens(const char * name, const std::vector<llama_token> & ids) {
        manifest << "{\"name\":\"" << name << "\",\"token_ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) manifest << (i ? "," : "") << ids[i];
        manifest << "]}\n";
        manifest.flush();
    }
};

static bool capture(ggml_tensor * t, bool ask, void * opaque) {
    auto & trace = *static_cast<trace_state *>(opaque);
    std::string name = t->name;
    int layer = -1;
    const auto dash = name.rfind('-');
    if (dash != std::string::npos && name.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        layer = std::stoi(name.substr(dash + 1));
        name.resize(dash);
    }
    const bool selected = trace.step + trace.rows > trace.first &&
        (name == "embd" || name == "l_out" || name == "result_norm" || name == "result_output");
    if (ask) return selected;
    if (!selected || t->type != GGML_TYPE_F32) throw std::runtime_error("unexpected checkpoint");
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, data.size());
    const bool last = name == "result_norm" || name == "result_output";
    const int rows = last ? 1 : trace.rows;
    if (t->ne[1] != rows || t->ne[2] != 1 || t->ne[3] != 1) throw std::runtime_error("unexpected batched checkpoint shape");
    for (int row = 0; row < rows; ++row) {
        const int step = trace.step + (last ? trace.rows - 1 : row);
        if (step < trace.first) continue;
        const std::string file = trace.path + "." + std::to_string(step) + "." + t->name + ".f32";
        std::ofstream binary(file, std::ios::binary);
        for (int64_t i = 0; i < t->ne[0]; ++i) {
            uint32_t bits;
            std::memcpy(&bits, data.data() + row * t->nb[1] + i * t->nb[0], 4);
            const char bytes[] = {char(bits), char(bits >> 8), char(bits >> 16), char(bits >> 24)};
            binary.write(bytes, 4);
        }
        if (!binary) throw std::runtime_error("write checkpoint failed");
        trace.manifest << "{\"name\":\"" << name << "\",\"layer\":";
        if (layer < 0) trace.manifest << "null"; else trace.manifest << layer;
        trace.manifest << ",\"step\":" << step << ",\"shape\":[" << t->ne[0] << "],\"binary_path\":\"";
        for (char c : file) { if (c == '\\' || c == '\"') trace.manifest << '\\'; trace.manifest << c; }
        trace.manifest << "\"}\n";
    }
    trace.manifest.flush();
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: dory-oracle MODEL RAW_PROMPT_FILE N_PREDICT");
        const int count = std::stoi(argv[3]);
        if (count < 0) throw std::runtime_error("negative N_PREDICT");
        std::ifstream prompt_file(argv[2]);
        if (!prompt_file) throw std::runtime_error("read prompt failed");
        const std::string text((std::istreambuf_iterator<char>(prompt_file)), {});
        trace_state trace(std::getenv("RMI_PARITY_TRACE"));
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model_kv_override overrides[3] = {};
        int override_count = 0;
        if (const char * s = std::getenv("RMI_DORY_TEST_SWA")) {
            auto & o = overrides[override_count++];
            o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            std::strcpy(o.key, "dory.attention.sliding_window");
            o.val_i64 = std::stoi(s);
            if (o.val_i64 <= 0) throw std::runtime_error("invalid test SWA window");
        }
        if (const char * s = std::getenv("RMI_DORY_CACHE_MODE")) {
            if (std::string(s) != "per_loop" && std::string(s) != "last_loop") throw std::runtime_error("invalid cache mode");
            auto & o = overrides[override_count++];
            o.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            std::strcpy(o.key, "dory.recurrent_kv_cache_mode");
            std::strcpy(o.val_str, s);
        }
        if (override_count) mp.kv_overrides = overrides;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("load model failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt(text.size() + 16);
        const int n = llama_tokenize(vocab, text.data(), text.size(), prompt.data(), prompt.size(), true, true);
        if (n <= 0) throw std::runtime_error("tokenize failed or empty prompt");
        prompt.resize(n);
        trace.tokens("prompt_ids", prompt);
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(128, n + count + 1);
        const int batch = std::getenv("RMI_DORY_BATCH") ? std::stoi(std::getenv("RMI_DORY_BATCH")) : 1;
        if (batch < 1 || batch > 32) throw std::runtime_error("batch must be 1..32");
        cp.n_batch = cp.n_ubatch = batch;
        cp.n_threads = cp.n_threads_batch = std::getenv("RMI_DORY_THREADS") ? std::stoi(std::getenv("RMI_DORY_THREADS")) : 1;
        const std::string kv = std::getenv("RMI_DORY_KV") ? std::getenv("RMI_DORY_KV") : "f32";
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
        std::cerr << "Dory oracle: " << n << " prompt + " << count << " decode steps\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
