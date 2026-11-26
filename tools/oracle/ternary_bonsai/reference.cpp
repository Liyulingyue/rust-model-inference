// Fixed-prompt CPU reference using the model publisher's public llama API.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 7) return 2; // MODEL PROMPT PREFIX [CTX BATCH POSITIONS]
    const bool tokens_only = std::string(argv[1]) == "--tokens";
    const int context = argc == 7 ? std::stoi(argv[4]) : 256;
    const int batch = argc == 7 ? std::stoi(argv[5]) : 128;
    const int positions = argc == 7 ? std::stoi(argv[6]) : 4;
    if (context <= 0 || batch <= 0 || positions <= 0 || (tokens_only && argc != 4)) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = tokens_only;
    auto *model = llama_model_load_from_file(argv[tokens_only ? 2 : 1], mp);
    if (!model) return 1;
    const auto *vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = batch;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    const std::string text = argv[tokens_only ? 3 : 2];
    std::vector<llama_token> ids(text.size() + 4);
    const int count = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), false, true);
    if (count < 0 || (count == 0 && !tokens_only)) return 1;
    ids.resize(count);
    std::cout << "input";
    for (auto id : ids) std::cout << ' ' << id;
    std::cout << '\n';
    if (tokens_only) {
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    if (count > context || positions - 1 > context - count) return 2;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    std::ofstream out(std::string(argv[3]) + ".f32", std::ios::binary);
    if (!out) return 1;
    for (int i = 0; i < positions; ++i) {
        auto start = std::chrono::steady_clock::now();
        for (size_t offset = 0; offset < ids.size(); offset += batch) {
            const auto n = std::min<size_t>(batch, ids.size() - offset);
            if (llama_decode(ctx, llama_batch_get_one(ids.data() + offset, n)) != 0) return 1;
        }
        const float *logits = llama_get_logits_ith(ctx, -1);
        int n = llama_vocab_n_tokens(vocab);
        int top = std::max_element(logits, logits + n) - logits;
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        out.write(reinterpret_cast<const char *>(logits), n * sizeof(float));
        std::cout << "step " << i << ' ' << top << ' ' << ms << '\n';
        ids.assign(1, top);
    }
    out.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 1;
}
