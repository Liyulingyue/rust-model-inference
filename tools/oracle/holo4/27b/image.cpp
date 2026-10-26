// Fixed CPU mtmd probe: 64x64 red image, non-thinking ChatML, first-token logits.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: image MODEL MMPROJ LOGITS_F32");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 64;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context load failed");
        auto vp = mtmd_context_params_default();
        vp.use_gpu = false;
        vp.n_threads = 8;
        vp.warmup = false;
        vp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto * vision = mtmd_init_from_file(argv[2], model, vp);
        if (!vision) throw std::runtime_error("projector load failed");
        std::vector<unsigned char> pixels(64 * 64 * 3, 0);
        for (size_t i = 0; i < pixels.size(); i += 3) pixels[i] = 255;
        auto * owned_bitmap = mtmd_bitmap_init(64, 64, pixels.data());
        const auto * bitmap = owned_bitmap;
        const std::string prompt = std::string("<|im_start|>user\n") + mtmd_default_marker()
            + "What color is the image? Answer with one word.<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
        auto * chunks = mtmd_input_chunks_init();
        if (mtmd_tokenize(vision, chunks, &text, &bitmap, 1)) throw std::runtime_error("tokenize failed");
        std::cout << "{\"oracle_pin\":\"b96806d96061049a5b574269b049bf6241d63d46\",\"chunks\":[";
        llama_pos position = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks, i);
            if (i) std::cout << ',';
            std::cout << "{\"count\":" << mtmd_input_chunk_get_n_tokens(chunk)
                      << ",\"n_pos\":" << mtmd_input_chunk_get_n_pos(chunk) << ",\"text\":[";
            size_t n = 0;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &n);
                for (size_t j = 0; j < n; ++j) std::cout << (j ? "," : "") << ids[j];
            }
            std::cout << "],\"image_positions\":[";
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const auto * image = mtmd_input_chunk_get_tokens_image(chunk);
                for (size_t j = 0; j < mtmd_input_chunk_get_n_tokens(chunk); ++j) {
                    const auto pos = mtmd_image_tokens_get_decoder_pos(image, position, j);
                    std::cout << (j ? "," : "") << '[' << pos.t << ',' << pos.x << ',' << pos.y << ',' << pos.z << ']';
                }
            }
            std::cout << "]}";
            position += mtmd_input_chunk_get_n_pos(chunk);
        }
        llama_pos end = 0;
        if (mtmd_helper_eval_chunks(vision, ctx, chunks, 0, 0, 64, true, &end))
            throw std::runtime_error("multimodal decode failed");
        const auto n = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits_ith(ctx, -1);
        int top = 0;
        if (!logits || n != 248320) throw std::runtime_error("invalid logits");
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
            if (logits[i] > logits[top]) top = i;
        }
        std::ofstream out(argv[3], std::ios::binary);
        out.write(reinterpret_cast<const char *>(logits), n * sizeof(float));
        if (!out) throw std::runtime_error("cannot save logits");
        std::cout << "],\"end_pos\":" << end << ",\"top1\":" << top << "}\n";
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(owned_bitmap);
        mtmd_free(vision);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
