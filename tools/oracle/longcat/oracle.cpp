// Checkpoint-only harness for pinned stable-diffusion.cpp; model math is upstream.
#include <fstream>
#include <stdexcept>
#include "model/diffusion/flux.hpp"
#include "ggml-cpu.h"

struct Oracle : Flux::FluxRunner {
    using FluxRunner::FluxRunner;
    ggml_backend_buffer_t weights = nullptr;
    ~Oracle() { if (weights) ggml_backend_buffer_free(weights); }
    void load(ModelLoader& loader) {
        weights = ggml_backend_alloc_ctx_tensors(params_ctx, runtime_backend);
        if (!weights) throw std::runtime_error("weight allocation failed");
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors, "model.diffusion_model");
        if (!loader.load_tensors(tensors)) throw std::runtime_error("weight loading failed");
    }
    void run(const std::string& input_dir, const std::string& output_dir, int ni, int nt, float t) {
        auto read = [&](const char* name, size_t n) {
            std::vector<float> values(n);
            std::ifstream f(input_dir + "/" + name, std::ios::binary);
            f.read(reinterpret_cast<char*>(values.data()), n * sizeof(float));
            if (!f || f.peek() != EOF) throw std::runtime_error(std::string("invalid input: ") + name);
            return values;
        };
        sd::Tensor<float> img({64, ni, 1}, read("img.f32", ni * 64));
        sd::Tensor<float> txt({3584, nt, 1}, read("txt.f32", nt * 3584));
        sd::Tensor<float> time({1}, {t});
        auto flat_ids = read("positions.f32", (ni + nt) * 3);
        std::vector<std::vector<float>> ids(ni + nt, std::vector<float>(3));
        for (int i=0; i<ni+nt; ++i) for(int j=0; j<3; ++j) ids[i][j]=flat_ids[i*3+j];
        sd::Tensor<float> pe({2, 2, 64, ni+nt}, Rope::embed_nd(ids, 1, 10000.0f, {16,56,56}));
        std::ofstream manifest(output_dir + "/trace.jsonl");
        auto graph = [&]() {
            auto gf = ggml_new_graph_custom(compute_ctx, FLUX_GRAPH_SIZE, false);
            auto ctx = get_context(gf);
            auto out = flux.forward_orig(&ctx, make_input(img), make_input(txt), make_input(time), nullptr, nullptr, make_input(pe));
            ctx.capture_tensor("longcat.output", out);
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        auto dump = [&]() {
            for (auto& [tensor, name] : debug_tensors) {
                std::vector<float> values(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*4);
                std::string filename = name + ".f32";
                std::ofstream file(output_dir+"/"+filename, std::ios::binary);
                file.write(reinterpret_cast<char*>(values.data()), values.size()*4);
                manifest << "{\"name\":\"" << name << "\",\"shape\":[" << tensor->ne[1] << "," << tensor->ne[0] << "],\"file\":\"" << filename << "\"}\n";
                if (!file || !manifest) return false;
            }
            return true;
        };
        // Allocate/execute upstream's graph directly: no weight cache, repack or scheduler backend.
        alloc_compute_ctx();
        auto gf = graph();
        for (auto& [tensor, name] : debug_tensors) ggml_build_forward_expand(gf, tensor);
        auto buffer = ggml_backend_alloc_ctx_tensors(compute_ctx, runtime_backend);
        if (!buffer) throw std::runtime_error("graph allocation failed");
        for (auto& [tensor, data] : backend_tensor_data_map) ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
        ggml_backend_cpu_set_n_threads(runtime_backend, 1);
        auto status = ggml_backend_graph_compute(runtime_backend, gf);
        bool success = status == GGML_STATUS_SUCCESS && dump();
        ggml_backend_buffer_free(buffer);
        if (!success) throw std::runtime_error("forward failed");
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: longcat-oracle MODEL INPUT_DIR OUTPUT_DIR IMAGE_TOKENS TEXT_TOKENS TIMESTEP MODEL_KIND");
        sd_set_log_callback([](sd_log_level_t, const char* message, void*) { fprintf(stderr, "%s", message); }, nullptr);
        ModelLoader loader;
        if (std::string(argv[7]) != "edit" && std::string(argv[7]) != "turbo") throw std::runtime_error("kind must be edit or turbo");
        if (!loader.init_from_file_and_convert_name(argv[1], "model.diffusion_model.", VERSION_LONGCAT)) throw std::runtime_error("invalid model");
        auto backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("CPU backend unavailable");
        { Oracle runner(backend, loader.get_tensor_storage_map(), "model.diffusion_model", VERSION_LONGCAT);
          runner.load(loader);
          runner.run(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stof(argv[6])); }
        ggml_backend_free(backend);
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
