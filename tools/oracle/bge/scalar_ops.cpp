// Execute actual pinned GGML F32 kernels on the same fixture as Rust.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    uint32_t n = 0, rows = 0;
    input.read(reinterpret_cast<char *>(&n), 4);
    input.read(reinterpret_cast<char *>(&rows), 4);
    if (!input || n == 0 || n > 4096 || rows != 5) return 2;
    ggml_init_params params = {64 * 1024 * 1024, nullptr, false};
    ggml_context *ctx = ggml_init(params);
    auto read = [&](int64_t width, int64_t count) {
        auto *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, count);
        input.read(static_cast<char *>(tensor->data), ggml_nbytes(tensor));
        if (!input) throw std::runtime_error("truncated fixture");
        return tensor;
    };
    auto *x = read(n, 1);
    auto *matrix = read(n, rows);
    auto *weight = read(n, 1);
    auto *bias = read(n, 1);
    auto *other = read(n, 1);
    const std::vector<const char *> names = {"matmul", "layer_norm", "rms_norm", "softmax", "silu", "add", "multiply"};
    const std::vector<ggml_tensor *> outputs = {
        ggml_mul_mat(ctx, matrix, x),
        ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), weight), bias),
        ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-5f), weight),
        ggml_soft_max(ctx, x), ggml_silu(ctx, x),
        ggml_add(ctx, x, other), ggml_mul(ctx, x, other)
    };
    auto *graph = ggml_new_graph(ctx);
    for (auto *tensor : outputs) ggml_build_forward_expand(graph, tensor);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) return 3;
    std::printf("{");
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::printf("%s\"%s\":[", i ? "," : "", names[i]);
        for (int64_t j = 0; j < ggml_nelements(outputs[i]); ++j) {
            uint32_t bits;
            std::memcpy(&bits, static_cast<char *>(outputs[i]->data) + j * 4, 4);
            std::printf("%s%u", j ? "," : "", bits);
        }
        std::printf("]");
    }
    std::printf("}\n");
    ggml_free(ctx);
}
