// MiniCPM5 parity oracle: replay explicit token ids through llama.cpp and
// dump full logits per step as binary.
//   usage: dump_tokens_oracle <model.gguf> <out.bin> id0 id1 ...
// binary layout per step: [int32 step][int32 token][int32 n_vocab][n_vocab f32 logits]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <out.bin> id0 id1 ...\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path = argv[2];
    std::vector<llama_token> tokens;
    for (int i = 3; i < argc; i++) tokens.push_back(atoi(argv[i]));

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = true;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::ofstream ofs(out_path, std::ios::binary);

    for (size_t step = 0; step < tokens.size(); step++) {
        llama_batch batch = llama_batch_get_one(&tokens[step], 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        // dump every step

        const float * logits = llama_get_logits(ctx);
        int32_t hdr[3] = { (int32_t)step, tokens[step], n_vocab };
        ofs.write((const char *)hdr, sizeof(hdr));
        ofs.write((const char *)logits, n_vocab * sizeof(float));
        fprintf(stderr, "step %zu token %d logits dumped\n", step, tokens[step]);
    }

    // greedy continue 8 tokens for generation parity
    for (int g = 0; g < 8; g++) {
        const float * logits = llama_get_logits(ctx);
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        llama_batch batch = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, batch) != 0) break;
        int32_t hdr[3] = { (int32_t)(tokens.size() + g), best, n_vocab };
        ofs.write((const char *)hdr, sizeof(hdr));
        ofs.write((const char *)logits, n_vocab * sizeof(float));
        fprintf(stderr, "gen step %d: argmax=%d logit=%.4f\n", g, best, best_val);
    }
    return 0;
}
