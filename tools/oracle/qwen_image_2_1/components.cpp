// Uses the same fixed sd.cpp checkout and GGUF reader as the DiT oracle.
#define main qi21_legacy_dit_main
#include "main.cpp"
#undef main
#include "conditioning/conditioner.hpp"
#include "model/vae/wan_vae.hpp"
#include "runtime/denoiser.hpp"
#include "core/rng_mt19937.hpp"
#include <stdexcept>

std::vector<float> read_floats(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % 4 != 0) throw std::runtime_error("invalid F32 input");
    std::vector<float> values(static_cast<size_t>(file.tellg()) / 4);
    file.seekg(0); file.read(reinterpret_cast<char*>(values.data()), values.size() * 4);
    if (!file) throw std::runtime_error("truncated F32 input");
    return values;
}

ggml_backend_buffer_t load_parameters(ModelLoader& loader, ggml_backend_t backend,
                                      std::map<std::string, ggml_tensor*>& params) {
    const size_t alignment = ggml_backend_get_alignment(backend);
    size_t total = 0;
    for (auto [name, tensor] : params) total += (ggml_nbytes(tensor) + alignment - 1) / alignment * alignment;
    auto buffer = ggml_backend_alloc_buffer(backend, total);
    if (!buffer) throw std::runtime_error("cannot allocate Oracle weights");
    size_t offset = 0;
    for (auto [name, tensor] : params) {
        ggml_set_name(tensor, name.c_str());
        ggml_backend_tensor_alloc(buffer, tensor, static_cast<char*>(ggml_backend_buffer_get_base(buffer)) + offset);
        offset += (ggml_nbytes(tensor) + alignment - 1) / alignment * alignment;
    }
    loader.set_n_threads(8);
    if (!loader.load_tensors(params)) throw std::runtime_error("Oracle weight loading failed");
    return buffer;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: components MODE MODEL [INPUT WIDTH HEIGHT THREADS]");
        sd_set_log_callback(harness_log, nullptr);
        std::string mode = argv[1];
        if (mode == "noise") {
            if (argc != 5) throw std::runtime_error("noise SEED W H");
            const int w=std::atoi(argv[3]),h=std::atoi(argv[4]);
            MT19937RNG rng(std::strtoull(argv[2],nullptr,10));
            auto noise=rng.randn(w*h*64);
            qwen_image_2_1_trace_f32("qwen.sample.noise",noise,{w,h,64});
            return 0;
        }
        if (mode == "schedule") {
            FluxScheduler scheduler(std::atoi(argv[3]), VERSION_QWEN_IMAGE_2_1);
            auto sigmas = scheduler.get_sigmas(std::atoi(argv[2]), 0, 0, nullptr);
            qwen_image_2_1_trace_f32("qwen.sample.sigmas", sigmas, {static_cast<int64_t>(sigmas.size())});
            return 0;
        }
        auto backend = ggml_backend_cpu_init();
        ModelLoader loader;
        std::map<std::string, ggml_tensor*> params;
        if (mode == "vae" || mode == "vae_encode") {
            if (argc != 7) throw std::runtime_error("vae MODEL LATENT W H THREADS");
            if (!loader.init_from_file_and_convert_name(argv[2], "vae.", VERSION_QWEN_IMAGE_2_1)) throw std::runtime_error("VAE load failed");
            WAN::WanVAERunner runner(backend, loader.get_tensor_storage_map(), "first_stage_model", mode == "vae", VERSION_QWEN_IMAGE_2_1);
            runner.get_param_tensors(params);
            auto buffer = load_parameters(loader, backend, params);
            const int w = std::atoi(argv[4]), h = std::atoi(argv[5]);
            if (mode == "vae") {
                sd::Tensor<float> latent({w,h,64,1}, read_floats(argv[3]));
                auto mapped = runner.diffusion_to_vae_latents(latent);
                qwen_image_2_1_trace_f32("qwen.vae.denormalized", mapped.values(), {w,h,1,64});
                auto output = runner._compute(std::atoi(argv[6]), mapped, true);
                if (output.empty()) throw std::runtime_error("VAE compute failed");
                qwen_image_2_1_trace_f32("qwen.vae.output", output.values(), {w*16,h*16,1,4});
            } else {
                sd::Tensor<float> image({w,h,1,4}, read_floats(argv[3]));
                auto output = runner._compute(std::atoi(argv[6]), image, false);
                if (output.empty()) throw std::runtime_error("VAE encoding failed");
                qwen_image_2_1_trace_f32("qwen.vae.encoded_mean", output.values(), {w/16,h/16,1,64});
                auto mapped = runner.vae_to_diffusion_latents(output);
                qwen_image_2_1_trace_f32("qwen.vae.encoded", mapped.values(), {w/16,h/16,1,64});
            }
            runner.runner_end();
            ggml_backend_buffer_free(buffer);
        } else if (mode == "text") {
            if (argc != 5 && (argc < 9 || (argc-6)%3)) throw std::runtime_error("text MODEL PROMPT THREADS [VISION RGBA W H ...]");
            if (!loader.init_from_file_and_convert_name(argv[2], "text_encoders.llm.", VERSION_QWEN_IMAGE_2_1)) throw std::runtime_error("LLM load failed");
            if (argc > 5 && !loader.init_from_file(argv[5], "text_encoders.llm.visual.")) throw std::runtime_error("Vision load failed");
            LLMEmbedder conditioner(backend, loader.get_tensor_storage_map(), VERSION_QWEN_IMAGE_2_1, "", argc > 5);
            conditioner.set_flash_attention_enabled(false);
            conditioner.get_param_tensors(params);
            auto buffer = load_parameters(loader, backend, params);
            ConditionerParams request; request.text = argv[3];
            std::vector<sd::Tensor<float>> references;
            if (argc > 5) {
                for(int i=6;i<argc;i+=3) references.emplace_back(std::vector<int64_t>{std::atoi(argv[i+1]),std::atoi(argv[i+2]),4,1},read_floats(argv[i]));
                request.ref_images = &references;
            }
            auto condition = conditioner.get_learned_condition(std::atoi(argv[4]), request);
            if (condition.empty()) throw std::runtime_error("text encoding failed");
            qwen_image_2_1_trace_f32("qwen.text.context", condition.c_crossattn.values(), condition.c_crossattn.shape());
            if (argc > 5) { std::vector<float> slots; for(auto v:condition.c_token_types.values()) slots.push_back(v); qwen_image_2_1_trace_f32("qwen.text.image_slots",slots,{static_cast<int64_t>(slots.size())}); }
            conditioner.runner_end();
            ggml_backend_buffer_free(buffer);
        } else if (mode == "sample") {
            if (argc != 11 && (argc < 16 || (argc-13)%3)) throw std::runtime_error("sample DIT NOISE POSITIVE NEGATIVE W H STEPS CFG THREADS [POS_SLOTS NEG_SLOTS REF W H ...]");
            if (!loader.init_from_file(argv[2])) throw std::runtime_error("DiT load failed");
            Qwen::QwenImage21Runner runner(backend, loader.get_tensor_storage_map(), "model.diffusion_model", nullptr, "qwen_image_2_1_prefix_cache=false");
            RawGguf raw; if (!load_raw_gguf(argv[2], raw) || !fill_params_from_gguf(runner, raw)) throw std::runtime_error("DiT weights failed");
            const int w=std::atoi(argv[6]), h=std::atoi(argv[7]), steps=std::atoi(argv[8]), threads=std::atoi(argv[10]);
            const float cfg=std::atof(argv[9]);
            auto positive = read_floats(argv[4]), negative = read_floats(argv[5]);
            sd::Tensor<float> x({w,h,64,1}, read_floats(argv[3]));
            sd::Tensor<float> pos({4096,static_cast<int64_t>(positive.size()/4096)}, positive);
            sd::Tensor<float> neg({4096,static_cast<int64_t>(negative.size()/4096)}, negative);
            sd::Tensor<int32_t> pos_slots, neg_slots;
            std::vector<sd::Tensor<float>> refs;
            if (argc > 11) {
                auto ps=read_floats(argv[11]), ns=read_floats(argv[12]);
                pos_slots=sd::Tensor<int32_t>({static_cast<int64_t>(ps.size())},std::vector<int32_t>(ps.begin(),ps.end()));
                neg_slots=sd::Tensor<int32_t>({static_cast<int64_t>(ns.size())},std::vector<int32_t>(ns.begin(),ns.end()));
                for(int i=13;i<argc;i+=3) refs.emplace_back(std::vector<int64_t>{std::atoi(argv[i+1]),std::atoi(argv[i+2]),64,1},read_floats(argv[i]));
            }
            FluxScheduler scheduler(w*h,VERSION_QWEN_IMAGE_2_1);
            auto sigmas=scheduler.get_sigmas(steps,0,0,nullptr);
            qwen_image_2_1_trace_f32("qwen.sample.sigmas",sigmas,{steps+1});
            qwen_image_2_1_trace_f32("qwen.sample.noise",x.values(),{w,h,64});
            DiscreteFlowDenoiser denoiser;
            sd::guidance::ClassifierFreeGuidance guider(cfg,1.f);
            int evaluations=0;
            auto output=sample_euler([&](const sd::Tensor<float>& current,float sigma,int step) {
                if (step>1) qwen_image_2_1_trace_f32("qwen.sample.latent",current.values(),{w,h,64});
                sd::Tensor<float> t({1},{denoiser.sigma_to_t(sigma)});
                DiffusionParams request; request.x=&current; request.context=&pos; request.timesteps=&t; request.extra=QwenImage21DiffusionExtra{pos_slots.empty()?nullptr:&pos_slots,0}; request.ref_latents=&refs; request.ref_image_params.pass_to_dit=true;
                auto velocity=runner.compute(threads,request);
                qwen_image_2_1_trace_f32("qwen.output",velocity.values(),velocity.shape());
                sd::Tensor<float> uncond;
                if (cfg!=1.f) { request.context=&neg; request.extra=QwenImage21DiffusionExtra{neg_slots.empty()?nullptr:&neg_slots,0}; uncond=runner.compute(threads,request); qwen_image_2_1_trace_f32("qwen.output",uncond.values(),uncond.shape()); }
                sd::guidance::GuidanceInput input;input.pred_cond=&velocity;input.pred_uncond=cfg==1.f?nullptr:&uncond;
                auto result=guider.forward(input,{});
                qwen_image_2_1_trace_f32("qwen.sample.velocity",result.pred.values(),{w,h,64});
                auto scaling=denoiser.get_scalings(sigma);
                result.pred=result.pred*scaling[1]+current*scaling[0];
                ++evaluations;
                return result;
            },x,sigmas);
            if (output.empty() || evaluations!=steps) throw std::runtime_error("Euler sampling failed");
            qwen_image_2_1_trace_f32("qwen.sample.latent",output.values(),{w,h,64});

        } else throw std::runtime_error("unknown component mode");
        ggml_backend_free(backend);
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
