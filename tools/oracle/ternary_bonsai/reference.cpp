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
    if (argc != 4) return 2; // MODEL PROMPT PREFIX or --tokens MODEL PROMPT
    const bool tokens_only = std::string(argv[1]) == "--tokens";
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = tokens_only;
    auto *model = llama_model_load_from_file(argv[tokens_only ? 2 : 1], mp);
    if (!model) return 1;
    const auto *vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 128;
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
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    std::ofstream out(std::string(argv[3]) + ".f32", std::ios::binary);
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, llama_batch_get_one(ids.data(), ids.size())) != 0) return 1;
        const float *logits = llama_get_logits_ith(ctx, -1);
        int n = llama_vocab_n_tokens(vocab);
        int top = std::max_element(logits, logits + n) - logits;
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        out.write(reinterpret_cast<const char *>(logits), n * sizeof(float));
        std::cout << "step " << i << ' ' << top << ' ' << ms << '\n';
        ids.assign(1, top);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out ? 0 : 1;
}
