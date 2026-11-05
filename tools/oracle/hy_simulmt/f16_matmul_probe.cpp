#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
int main() {
    auto *ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    if (!ctx) return 1;
    constexpr int n = 37;
    auto *v16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, 1);
    auto *v32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    auto *p16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, 1);
    auto *p32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    for (int i = 0; i < n; ++i) {
        const auto v = ggml_fp32_to_fp16(std::sin(float(i) * 0.17f) * 3.7f);
        const float p = (float(i) + 0.37f) / 703.1f;
        ((ggml_fp16_t*)v16->data)[i] = v;
        ((float*)v32->data)[i] = ggml_fp16_to_fp32(v);
        ((ggml_fp16_t*)p16->data)[i] = ggml_fp32_to_fp16(p);
        ((float*)p32->data)[i] = p;
    }
    auto *mixed = ggml_mul_mat(ctx, v16, p32);
    auto *rounded = ggml_mul_mat(ctx, v16, p16);
    auto *wide = ggml_mul_mat(ctx, v32, p32);
    auto *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mixed);
    ggml_build_forward_expand(graph, rounded);
    ggml_build_forward_expand(graph, wide);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        return 1;
    }
    const float a = *(float*)mixed->data, b = *(float*)rounded->data, c = *(float*)wide->data;
    const bool exact = std::memcmp(&a, &b, sizeof(float)) == 0;
    std::printf("F16xF32=%.9g F16xF16=%.9g F32xF32=%.9g explicit_rounding_exact=%s\n", a, b, c, exact ? "true" : "false");
    ggml_free(ctx);
    return exact && a != c ? 0 : 1;
}
