// oracle.cpp - llama.cpp reference oracle for Hy-MT3 GGUF models.
// Emits: GGUF metadata, exact prompt token IDs, greedy-generated token IDs,
// full float32 logits per step, and decoded text. No sampling beyond greedy argmax.
//
// Build (from llama.cpp root):
//   g++ -O2 -std=c++17 -Iinclude oracle.cpp -Lbuild/bin -lllama -lggml -lggml-base -o oracle
// Run:
//   LD_LIBRARY_PATH=build/bin ./oracle -m model.gguf --prompt-file p.txt -n 32 -t 8 -c 2048 \
//       --add-bos 0 --outdir out/

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::string get_arg(int argc, char ** argv, const char * name, const char * def) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) return argv[i + 1];
    }
    return def ? std::string(def) : std::string();
}

static bool has_flag(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

// returns 0 on success
static int32_t decode_tokens(llama_context * ctx, const llama_token * toks, int n, llama_pos pos0, bool want_logits) {
    struct llama_batch_ext * b = llama_batch_ext_init(ctx);
    if (!b) { std::fprintf(stderr, "oracle: llama_batch_ext_init failed\n"); return -1; }
    for (int i = 0; i < n; ++i) {
        const int32_t idx = llama_batch_ext_add_token(b, /*seq_id*/ 0, toks[i]);
        if (idx < 0) {
            std::fprintf(stderr, "oracle: add_token failed rc=%d at i=%d\n", idx, i);
            llama_batch_ext_free(b);
            return -1;
        }
        const llama_pos p = pos0 + i;
        if (!llama_batch_ext_set_pos(b, idx, &p)) {
            std::fprintf(stderr, "oracle: set_pos failed at i=%d\n", i);
            llama_batch_ext_free(b);
            return -1;
        }
        if (want_logits && i == n - 1) {
            if (!llama_batch_ext_set_output_logits(b, idx, true)) {
                std::fprintf(stderr, "oracle: set_output_logits failed\n");
                llama_batch_ext_free(b);
                return -1;
            }
        }
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    llama_batch_ext_free(b);
    return rc;
}

static const float * last_logits(llama_context * ctx, int n_batch_tokens) {
    const float * l = llama_get_logits_ith(ctx, -1);
    if (!l) l = llama_get_logits_ith(ctx, n_batch_tokens - 1);
    if (!l) l = llama_get_logits_ith(ctx, 0);
    return l;
}

static std::string piece(const llama_vocab * vocab, llama_token t, bool special) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, special);
    if (n < 0) return std::string();
    return std::string(buf, n);
}

static void write_f32(const std::string & path, const float * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data), (std::streamsize)(n * sizeof(float)));
}

int main(int argc, char ** argv) {
    const std::string mpath  = get_arg(argc, argv, "-m", "");
    const std::string outdir = get_arg(argc, argv, "--outdir", ".");
    const int ngen     = std::atoi(get_arg(argc, argv, "-n", "32").c_str());
    const int nthreads = std::atoi(get_arg(argc, argv, "-t", "8").c_str());
    const int nctx     = std::atoi(get_arg(argc, argv, "-c", "2048").c_str());
    const int add_bos  = std::atoi(get_arg(argc, argv, "--add-bos", "0").c_str());
    const int topk     = std::atoi(get_arg(argc, argv, "--topk", "10").c_str());
    const int nbatch   = std::atoi(get_arg(argc, argv, "--nb", "256").c_str());

    if (mpath.empty()) { std::fprintf(stderr, "oracle: -m required\n"); return 2; }

    std::string prompt;
    const std::string pfile = get_arg(argc, argv, "--prompt-file", "");
    if (!pfile.empty()) {
        std::ifstream f(pfile, std::ios::binary);
        if (!f) { std::fprintf(stderr, "oracle: cannot open --prompt-file %s\n", pfile.c_str()); return 2; }
        prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    } else {
        prompt = get_arg(argc, argv, "-p", "");
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(mpath.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "oracle: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- GGUF metadata dump ----
    {
        std::ofstream mf(outdir + "/meta.txt");
        const int nm = llama_model_meta_count(model);
        for (int i = 0; i < nm; ++i) {
            char key[512] = {0};
            llama_model_meta_key_by_index(model, i, key, sizeof(key));
            std::vector<char> val(65536, 0);
            llama_model_meta_val_str_by_index(model, i, val.data(), val.size());
            mf << key << " = " << val.data() << "\n";
        }
        mf << "oracle.n_vocab = " << n_vocab << "\n";
        mf << "oracle.n_ctx_train = " << llama_model_n_ctx_train(model) << "\n";
        mf << "oracle.add_bos_arg = " << add_bos << "\n";
        mf << "oracle.vocab_add_bos = " << (int) llama_vocab_get_add_bos(vocab) << "\n";
        mf << "oracle.vocab_add_eos = " << (int) llama_vocab_get_add_eos(vocab) << "\n";
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = (uint32_t) nctx;
    // FIXED SETTINGS agreed with the Rust side (must match exactly):
    cparams.n_batch       = (uint32_t) nbatch;
    cparams.n_ubatch      = (uint32_t) nbatch;
    cparams.n_threads     = (uint32_t) nthreads;
    cparams.n_threads_batch = (uint32_t) nthreads;
    cparams.type_k        = GGML_TYPE_F32;   // KV cache F32
    cparams.type_v        = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.offload_kqv   = false;
    cparams.no_perf       = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "oracle: failed to create context\n"); return 1; }

    // ---- tokenize ----
    std::vector<llama_token> ptoks(prompt.size() + 16);
    int n_ptok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                               ptoks.data(), (int32_t) ptoks.size(), add_bos != 0, true);
    if (n_ptok < 0) {
        ptoks.resize((size_t) (-n_ptok) + 16);
        n_ptok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                               ptoks.data(), (int32_t) ptoks.size(), add_bos != 0, true);
    }
    if (n_ptok < 0) { std::fprintf(stderr, "oracle: tokenize failed\n"); return 1; }
    ptoks.resize(n_ptok);

    {
        std::ofstream f(outdir + "/prompt_tokens.json");
        f << "{\"n_prompt_tokens\": " << n_ptok << ", \"add_bos\": " << add_bos
          << ", \"prompt_tokens\": [";
        for (int i = 0; i < n_ptok; ++i) f << (i ? ", " : "") << ptoks[i];
        f << "]}\n";
    }

    std::printf("oracle: n_vocab=%d n_prompt_tokens=%d add_bos=%d\n", n_vocab, n_ptok, add_bos);

    // ---- prefill ----
    if (decode_tokens(ctx, ptoks.data(), n_ptok, 0, true) != 0) {
        std::fprintf(stderr, "oracle: prefill decode failed\n"); return 1;
    }
    const float * logits = last_logits(ctx, n_ptok);
    if (!logits) { std::fprintf(stderr, "oracle: null logits after prefill\n"); return 1; }

    // ---- greedy generation, capturing logits at each step ----
    std::vector<llama_token> gen;
    std::vector<float> all_logits;           // (ngen) x n_vocab, row-major per step
    std::ofstream topkf(outdir + "/topk.txt");

    llama_pos cur_pos = n_ptok;
    for (int step = 0; step < ngen; ++step) {
        // record logits for this step
        all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        if (step == 0) write_f32(outdir + "/logits_prompt.f32", logits, (size_t) n_vocab);

        // top-k for reporting
        {
            std::vector<int> idx(n_vocab);
            for (int i = 0; i < n_vocab; ++i) idx[i] = i;
            const int k = std::min(topk, n_vocab);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            topkf << "step " << step << ":";
            for (int i = 0; i < k; ++i) {
                topkf << " [" << idx[i] << " " << logits[idx[i]] << " " << piece(vocab, idx[i], false) << "]";
            }
            topkf << "\n";
        }

        // greedy argmax
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        gen.push_back(best);

        if (step + 1 == ngen) break;
        if (decode_tokens(ctx, &gen.back(), 1, cur_pos, true) != 0) {
            std::fprintf(stderr, "oracle: decode failed at step %d\n", step); return 1;
        }
        ++cur_pos;
        logits = last_logits(ctx, 1);
        if (!logits) { std::fprintf(stderr, "oracle: null logits at step %d\n", step); return 1; }
    }

    // ---- write logits blob ----
    {
        std::ofstream f(outdir + "/logits_all.bin", std::ios::binary);
        const char magic[4] = {'L', 'L', 'O', 'G'};
        const uint32_t ver = 1;
        f.write(magic, 4);
        f.write(reinterpret_cast<const char *>(&ver), 4);
        f.write(reinterpret_cast<const char *>(&n_vocab), 4);
        f.write(reinterpret_cast<const char *>(&ngen), 4);
        f.write(reinterpret_cast<const char *>(all_logits.data()),
                (std::streamsize)(all_logits.size() * sizeof(float)));
    }

    // ---- generated tokens + text ----
    {
        std::ofstream f(outdir + "/generated_tokens.json");
        f << "{\"n_generated\": " << (int) gen.size() << ", \"generated_tokens\": [";
        for (size_t i = 0; i < gen.size(); ++i) f << (i ? ", " : "") << gen[i];
        f << "]}\n";
    }
    std::string text;
    for (llama_token t : gen) text += piece(vocab, t, false);
    {
        std::ofstream f(outdir + "/text.txt", std::ios::binary);
        f << text;
    }
    {
        std::ofstream f(outdir + "/summary.json");
        f << "{\n  \"model\": \"" << mpath << "\",\n";
        f << "  \"n_vocab\": " << n_vocab << ",\n";
        f << "  \"n_prompt_tokens\": " << n_ptok << ",\n";
        f << "  \"add_bos\": " << add_bos << ",\n";
        f << "  \"n_generated\": " << (int) gen.size() << ",\n";
        f << "  \"n_ctx\": " << nctx << ",\n";
        f << "  \"threads\": " << nthreads << ",\n";
        f << "  \"settings\": {\n";
        f << "    \"sampling\": \"greedy_argmax\",\n";
        f << "    \"n_predict\": " << ngen << ",\n";
        f << "    \"n_ctx\": " << nctx << ",\n";
        f << "    \"n_batch\": " << nbatch << ",\n";
        f << "    \"n_ubatch\": " << nbatch << ",\n";
        f << "    \"n_threads\": " << nthreads << ",\n";
        f << "    \"n_gpu_layers\": 0,\n";
        f << "    \"kv_cache_type\": \"f32\",\n";
        f << "    \"flash_attn\": \"disabled\",\n";
        f << "    \"add_bos\": " << add_bos << ",\n";
        f << "    \"parse_special\": true,\n";
        f << "    \"llama_cpp_commit\": \"9c003a97ab34a69cc49243f9b02014038205ffae\"\n";
        f << "  },\n";
        f << "  \"prompt_tokens\": [";
        for (int i = 0; i < n_ptok; ++i) f << (i ? ", " : "") << ptoks[i];
        f << "],\n  \"generated_tokens\": [";
        for (size_t i = 0; i < gen.size(); ++i) f << (i ? ", " : "") << gen[i];
        f << "]\n}\n";
    }

    std::printf("oracle: generated %zu tokens\n", gen.size());
    std::printf("oracle: generated_ids:");
    for (llama_token t : gen) std::printf(" %d", t);
    std::printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

