// Standalone reference for ggml's ARM SiLU.
//
// The repository's SiLU used libm's `expf`, while ggml's vectorized SiLU
// uses the polynomial approximation in `ggml_v_expf`. The two differ by up
// to one ULP, and in the Nemotron-3-Nano 30B SSM branch that one ULP is
// enough to flip an 8-bit Q8_K activation level in the Q4_K projection
// that follows, which turns a 1e-7 relative difference into a 4e-4 one.
//
// This program reproduces ggml's ARM `ggml_vec_silu_f32` exactly, so the
// repository's implementation can be compared against it bit for bit.
//
// usage: silu_oracle <in.f32> <out.f32>
//
//   in.f32    N little-endian f32 values
//   out.f32   N little-endian f32 values, ggml's SiLU of the input

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- verbatim from ggml/src/ggml-cpu/vec.h (the aarch64 NEON block) ---

inline static float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), b, vdupq_n_f32(0x1.0e4020p-7f)), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

inline static float32x4_t ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1);
    const float32x4_t zero = vdupq_n_f32(0);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

// --- verbatim from ggml/src/ggml-cpu/vec.cpp (the aarch64 NEON branch) ---

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    (void) i;
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vdupq_n_f32(1.0f)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_v_silu(vdupq_n_f32(x[i]))[0];
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in.f32> <out.f32>\n", argv[0]);
        return 2;
    }
    FILE * f_in = fopen(argv[1], "rb");
    FILE * f_out = fopen(argv[2], "wb");
    if (!f_in || !f_out) { perror("open"); return 1; }

    fseek(f_in, 0, SEEK_END);
    const long bytes = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);
    const int n = (int) (bytes / 4);
    float * x = malloc((size_t) bytes);
    float * y = malloc((size_t) bytes);
    if (!x || !y) { fprintf(stderr, "oom\n"); return 1; }
    if (fread(x, 1, (size_t) bytes, f_in) != (size_t) bytes) {
        fprintf(stderr, "short read\n");
        return 1;
    }
    fclose(f_in);

    ggml_vec_silu_f32(n, y, x);

    fwrite(y, 1, (size_t) bytes, f_out);
    fclose(f_out);
    fprintf(stderr, "silu %d values\n", n);
    free(x);
    free(y);
    return 0;
}
