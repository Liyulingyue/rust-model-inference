// CPU reference runner for the pinned llama.cpp revision in README.md.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct trace_state { std::string dir; int step = 0; };

static bool capture(ggml_tensor * t, bool ask, void * data) {
    auto & s = *static_cast<trace_state *>(data);
    std::string name(t->name);
    bool selected = name.find("model.") == 0 || name.find("l_out-") == 0 ||
                    name.find("lfm3_") == 0 || name == "result_norm" || name == "result_output" ||
                    // The attention output before the head gate and the SwiGLU
                    // output are the inputs two quantized projections take.
                    // Their callbacks name them, but the filter above would
                    // otherwise drop them.
                    name.find("attn_out") == 0 || name.find("ffn_swiglu") == 0;
    selected = selected && name.find(" (") == std::string::npos;
    if (ask) return selected;
    if (!selected || s.dir.empty() || t->type != GGML_TYPE_F32) return true;
    std::vector<char> storage(ggml_nbytes(t));
    ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
    std::ofstream f(s.dir + "/" + std::to_string(s.step) + "-" + name + ".f32", std::ios::binary);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    f.write(storage.data() + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], 4);
    std::ofstream manifest(s.dir + "/checkpoints.tsv", std::ios::app);
    manifest << s.step << '\t' << name;
    for (auto dim : t->ne) manifest << '\t' << dim;
    manifest << '\n';
    return bool(f) && bool(manifest);
}

int main(int argc, char ** argv) {
    if (argc != 5) { std::cerr << "oracle MODEL PROMPT_FILE OUTPUT_DIR N_GENERATE\n"; return 1; }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) return 1;
    std::string prompt((std::istreambuf_iterator<char>(input)), {});
    std::filesystem::create_directories(argv[3]);
    trace_state trace{argv[3]};
    std::ofstream(trace.dir + "/checkpoints.tsv", std::ios::trunc).close();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> ids(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.data(), prompt.size(), ids.data(), ids.size(), false, true);
    if (n < 0) return 3;
    ids.resize(n);
    std::ofstream tokens(trace.dir + "/tokens.txt");
    for (auto id : ids) tokens << id << ' ';
    tokens << '\n';
    int generate = std::stoi(argv[4]);
    if (generate == 0) { llama_model_free(model); llama_backend_free(); return 0; }
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(512, n + std::stoi(argv[4]));
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = false;
    if (!std::getenv("LFM3_NO_TRACE")) { cp.cb_eval = capture; cp.cb_eval_user_data = &trace; }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    for (int i = 0; i < n + generate - 1; ++i) {
        trace.step = i;
        auto batch = llama_batch_get_one(&ids[i], 1);
        if (llama_decode(ctx, batch)) return 5;
        if (i < n - 1) continue;
        float * logits = llama_get_logits_ith(ctx, -1);
        int size = llama_vocab_n_tokens(vocab);
        std::ofstream out(trace.dir + "/logits-" + std::to_string(i - n + 1) + ".f32", std::ios::binary);
        out.write(reinterpret_cast<char *>(logits), size * sizeof(float));
        auto next = std::max_element(logits, logits + size) - logits;
        tokens << next << ' ';
        ids.push_back(next);
    }
    tokens << '\n';
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
