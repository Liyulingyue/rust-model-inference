// Trace the pinned llama.cpp graph; no replacement arithmetic kernels.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct snapshot {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

static std::map<std::string, snapshot> tensors;

static bool selected(const std::string & name) {
    for (const char * prefix : {"rmi_embedding", "rmi_embedding_norm-", "rmi_q-", "rmi_k-", "rmi_v-",
                               "rmi_attention-", "rmi_ffn_input-", "rmi_layer_output-", "result_embd"}) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

static bool capture(ggml_tensor * t, bool ask, void *) {
    if (ask) return selected(t->name);
    if (!selected(t->name)) return true;
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("expected F32 checkpoint");
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    snapshot result;
    const std::string name = t->name;
    const int rank = name == "result_embd_pooled" ? 1 :
        (name.rfind("rmi_q-", 0) == 0 || name.rfind("rmi_k-", 0) == 0 ||
         name.rfind("rmi_v-", 0) == 0) ? 3 : 2;
    for (int axis = rank - 1; axis >= 0; --axis) result.shape.push_back(t->ne[axis]);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    float value;
                    std::memcpy(&value, bytes.data() + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0], 4);
                    result.values.push_back(value);
                }
    if (!tensors.emplace(name, std::move(result)).second)
        throw std::runtime_error("duplicate checkpoint: " + name);
    return true;
}

int main(int argc, char ** argv) {
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    params.embedding = true;
    params.warmup = false;
    params.cb_eval = capture;
    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) return 1;
    const auto tokens = common_tokenize(llama_model_get_vocab(model), params.prompt, true, true);
    auto batch_tokens = tokens;
    if (llama_encode(ctx, llama_batch_get_one(batch_tokens.data(), batch_tokens.size()))) return 1;

    const char * path = std::getenv("RMI_PARITY_TRACE");
    if (!path) throw std::runtime_error("RMI_PARITY_TRACE is required");
    std::ofstream manifest(path);
    manifest << "{\"name\":\"embedding.tokens\",\"token_ids\":[";
    for (size_t i = 0; i < tokens.size(); ++i) manifest << (i ? "," : "") << tokens[i];
    manifest << "]}\n";
    std::map<std::string, int> occurrences;
    auto write = [&](const std::string & name, int layer, const snapshot & value) {
        const int occurrence = occurrences[name]++;
        const std::string binary_path = std::string(path) + "." + name + "." + std::to_string(occurrence) + ".f32";
        std::ofstream binary(binary_path, std::ios::binary);
        for (float f : value.values) {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            const char bytes[] = {char(bits), char(bits >> 8), char(bits >> 16), char(bits >> 24)};
            binary.write(bytes, 4);
        }
        manifest << "{\"name\":\"" << name << "\",\"layer\":";
        if (layer < 0) manifest << "null"; else manifest << layer;
        manifest << ",\"shape\":[";
        for (size_t i = 0; i < value.shape.size(); ++i) manifest << (i ? "," : "") << value.shape[i];
        manifest << "],\"occurrence\":" << occurrence << ",\"binary_path\":\"" << binary_path << "\"}\n";
        if (!binary || !manifest) throw std::runtime_error("trace write failed");
    };
    // Canonical graph order, independent of the backend scheduler's node order.
    write("bert.embedding", -1, tensors.at("rmi_embedding"));
    write("bert.embedding_norm", -1, tensors.at("rmi_embedding_norm-0"));
    const int layers = llama_model_n_layer(model);
    for (int layer = 0; layer < layers; ++layer) {
        const std::string suffix = "-" + std::to_string(layer);
        for (const auto & pair : std::vector<std::pair<std::string, std::string>>{
                 {"q", "rmi_q"}, {"k", "rmi_k"}, {"v", "rmi_v"},
                 {"attention", "rmi_attention"}, {"ffn_input", "rmi_ffn_input"}})
            write("bert." + pair.first, layer, tensors.at(pair.second + suffix));
        write("bert.layer_output", layer,
              tensors.at(layer == layers - 1 ? "result_embd" : "rmi_layer_output" + suffix));
    }
    write("embedding.pooled", -1, tensors.at("result_embd_pooled"));
    const int width = llama_model_n_embd(model);
    snapshot normalized{{width}, std::vector<float>(width)};
    common_embd_normalize(llama_get_embeddings_seq(ctx, 0), normalized.values.data(), width, 2);
    write("embedding.final", -1, normalized);
    return 0;
}
