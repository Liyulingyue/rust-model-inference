// Read-only clip API probe; run against the Qwen3.5 Oracle's pinned CPU build.
#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include "llama.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct trace {
    std::filesystem::path directory;
    std::ofstream manifest;

    explicit trace(const char * path) : directory(path), manifest(directory / "vision.jsonl") {
        if (!manifest) throw std::runtime_error("cannot create vision manifest");
    }

    void write(const std::string & name, const std::vector<float> & values, int64_t dim) {
        const auto path = std::filesystem::absolute(directory / (name + ".f32"));
        std::ofstream binary(path, std::ios::binary);
        binary.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (!binary || values.empty() || values.size() % dim) throw std::runtime_error("invalid vision checkpoint");
        manifest << "{\"name\":\"qwen.vision." << name << "\",\"shape\":[" << dim << ','
                 << values.size() / dim << "],\"binary_path\":\"" << path.generic_string() << "\"}\n";
        manifest.flush();
    }
};

static bool callback(ggml_tensor * tensor, bool ask, void * data) {
    const std::string name = tensor->name;
    const bool embedding = name == "inp_pos_emb";
    const bool block = name.rfind("layer_out-", 0) == 0;
    if (ask) return embedding || block;
    if (!embedding && !block) return true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        throw std::runtime_error("expected contiguous F32 vision checkpoint");
    }
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    static_cast<trace *>(data)->write(embedding ? "embedding" : "block." + name.substr(10), values, tensor->ne[0]);
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: vision MMPROJ PIXELS_F32 WIDTH HEIGHT OUTPUT_DIR");
        const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
        if (width <= 0 || height <= 0 || width % 32 || height % 32) throw std::runtime_error("invalid image grid");
        trace output(argv[5]);
        std::vector<float> pixels(size_t(width) * height * 3);
        std::ifstream input(argv[2], std::ios::binary);
        input.read(reinterpret_cast<char *>(pixels.data()), pixels.size() * sizeof(float));
        if (!input || input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid pixel file size");
        llama_backend_init();
        clip_context_params params{};
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        params.cb_eval = callback;
        params.cb_eval_user_data = &output;
        auto context = clip_init(argv[1], params);
        if (!context.ctx_v || context.ctx_a || context.ctx_gen_a) throw std::runtime_error("expected vision-only projector");
        clip_image_f32 image;
        image.set_size({width, height}, false, false);
        image.cpy_buf(pixels);
        std::vector<float> projected(size_t(clip_n_mmproj_embd(context.ctx_v)) * clip_n_output_tokens(context.ctx_v, &image));
        if (!clip_image_encode(context.ctx_v, 1, &image, projected)) throw std::runtime_error("vision encode failed");
        output.write("output", projected, clip_n_mmproj_embd(context.ctx_v));
        clip_free(context.ctx_v);
        llama_backend_free();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
