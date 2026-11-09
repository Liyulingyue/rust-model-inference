// Standalone Q5_0 x Q8_0 dot-product oracle.
//
// The 30B's `ssm_in.weight` is Q5_0, and the 4B's is Q8_0. The 4B's SSM
// branch matches the upstream oracle to 1 ULP while the 30B's diverges by
// ~1e-4, so the divergence is in the quantized-weight path rather than in
// the Mamba-2 arithmetic.
//
// This tool reproduces ggml's `ggml_vec_dot_q5_0_q8_0` for the ARM NEON
// target (the same target the repository and the oracle build for) so the
// Rust kernel can be compared byte for byte.
//
// The reference below is transcribed from
// `ggml/src/ggml-cpu/arch/arm/quants.c::ggml_vec_dot_q5_0_q8_0`:
//
//   * two blocks are handled per iteration, into two separate 4-lane
//     accumulators `sumv0` / `sumv1`;
//   * the high bit is applied through the `table_b2b_1` lookup, which
//     yields `(!bit) << 4` so that the `-16` bias is folded into the
//     subtraction;
//   * the per-lane products are integer dot products widened to F32 and
//     accumulated with `vmlaq_n_f32` (an FMA);
//   * the two accumulators are reduced with `vaddvq_f32` at the end.
//
// The per-lane grouping of the integer dot has to match the repository's
// kernel, which is why both are compiled with `-march=armv8.2-a+dotprod`
// and use the native `vdotq_s32`.
//
// usage: q5_0_oracle <weights.bin> <activations.bin> <out.bin> <n_out> <n_in>
//
//   weights.bin      n_out rows of n_in elements, Q5_0 (22 bytes / 32)
//   activations.bin  n_in Q8_0-quantized values (32 bytes / 32) plus the
//                    per-block F16 scales appended after the payload
//   out.bin          n_out F32 dot products

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK5_0 32
#define QK8_0 32

// ggml stores `d` as an IEEE-754 binary16 value. The conversion below is a
// verbatim copy of ggml's `ggml_compute_fp16_to_fp32`
// (`src/ggml-impl.h`), including the magic-number trick that handles
// half-precision subnormals.
static inline float fp32_from_bits(uint32_t w) {
    float f;
    memcpy(&f, &w, sizeof(f));
    return f;
}

static inline uint32_t fp32_to_bits(float f) {
    uint32_t w;
    memcpy(&w, &f, sizeof(w));
    return w;
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

typedef struct {
    uint16_t d;      // delta (F16)
    uint8_t  qh[4];  // 5-th bit of quants
    uint8_t  qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    uint16_t d;      // scale (F16)
    int8_t   qs[QK8_0];
} block_q8_0;

// ggml's `table_b2b_1`: expands 8 bits into 8 bytes, one per bit, where a
// *set* bit becomes `0x00` and a clear bit `0x10`.
//
// ggml writes the table as `static const uint64_t table_b2b_1[1 << 8] =
// { B8(10, 00) };` -- a macro that expands one entry per possible byte.
// Transcribing 256 magic 64-bit constants by hand is exactly the kind of
// transcription that silently produces a plausible-looking wrong table,
// so the table is generated from the rule instead. The rule is ggml's own
// comment on the table: "(!b) << 4", i.e. the byte is `0x10` when the bit
// is clear and `0x00` when it is set. That is what the ARM kernel
// subtracts from the nibble to get the `q - 16` bias.
static uint64_t table_b2b_1_entry(uint8_t bits) {
    uint64_t value = 0;
    for (int bit = 0; bit < 8; ++bit) {
        const uint8_t byte = (bits & (1u << bit)) ? 0x00 : 0x10;
        value |= (uint64_t) byte << (8 * bit);
    }
    return value;
}

// `ggml_vdotq_s32`.
//
// This mirrors ggml's own conditional exactly (`ggml-cpu-impl.h`): with the
// `dotprod` extension the native instruction is used, without it ggml
// falls back to a pairwise multiply-accumulate that produces the same
// total sum but with a different per-lane grouping. ggml's comment on the
// fallback says "do not use when individual lane values matter" -- and for
// Q5_0 they do, because each lane is scaled and accumulated separately in
// F32. So the reference has to be built with the same extension the
// repository's kernel uses, and both sides are compiled with
// `-march=armv8.2-a+dotprod`.
//
// The build is on aarch64, so the native path is the one that is taken.
#if defined(__ARM_FEATURE_DOTPROD)

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#else

static inline int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#endif

// Verbatim transcription of ggml's ARM `ggml_vec_dot_q5_0_q8_0`.
//
// The generic C version in `quants.c` accumulates `sumi0` and `sumi1`
// separately and adds them at the end; the ARM version instead keeps two
// 4-lane F32 accumulators, widens each block's integer dot product to F32
// and accumulates with an FMA. The two differ in rounding, so the ARM
// version is the one to compare against.
static float vec_dot_q5_0_q8_0_arm(const block_q5_0 * x, const block_q8_0 * y, int n) {
    const int qk = QK8_0;
    const int nb = n / qk;

    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const block_q5_0 * x0 = &x[ib];
        const block_q5_0 * x1 = &x[ib + 1];
        const block_q8_0 * y0 = &y[ib];
        const block_q8_0 * y1 = &y[ib + 1];

        uint32_t qh0;
        uint32_t qh1;
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        uint64_t tmp0[4];
        uint64_t tmp1[4];

        tmp0[0] = table_b2b_1_entry((qh0 >>  0) & 0xFF);
        tmp0[1] = table_b2b_1_entry((qh0 >>  8) & 0xFF);
        tmp0[2] = table_b2b_1_entry((qh0 >> 16) & 0xFF);
        tmp0[3] = table_b2b_1_entry((qh0 >> 24)       );

        tmp1[0] = table_b2b_1_entry((qh1 >>  0) & 0xFF);
        tmp1[1] = table_b2b_1_entry((qh1 >>  8) & 0xFF);
        tmp1[2] = table_b2b_1_entry((qh1 >> 16) & 0xFF);
        tmp1[3] = table_b2b_1_entry((qh1 >> 24)       );

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))),
                       fp16_to_fp32(x0->d)*fp16_to_fp32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))),
                       fp16_to_fp32(x1->d)*fp16_to_fp32(y1->d));
    }

    float sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        const int sumi = sumi0 + sumi1;
        sumf += (fp16_to_fp32(x[ib].d)*fp16_to_fp32(y[ib].d)) * sumi;
    }

    return sumf;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <weights.bin> <activations.bin> <out.bin> <n_out> <n_in>\n", argv[0]);
        return 2;
    }

    const int64_t n_out = atoll(argv[4]);
    const int64_t n_in  = atoll(argv[5]);
    if (n_in % QK5_0 != 0) {
        fprintf(stderr, "n_in %lld is not a multiple of %d\n", (long long) n_in, QK5_0);
        return 1;
    }
    const int64_t n_blocks = n_in / QK5_0;
    const int64_t row_bytes = n_blocks * (int64_t) sizeof(block_q5_0);

    block_q5_0 * weights = (block_q5_0 *) malloc((size_t) (n_out * row_bytes));
    block_q8_0 * acts    = (block_q8_0 *) malloc((size_t) (n_blocks * sizeof(block_q8_0)));
    float      * out     = (float *)        malloc((size_t) (n_out * sizeof(float)));
    if (!weights || !acts || !out) { fprintf(stderr, "oom\n"); return 1; }

    FILE * f_w = fopen(argv[1], "rb");
    FILE * f_a = fopen(argv[2], "rb");
    FILE * f_o = fopen(argv[3], "wb");
    if (!f_w || !f_a || !f_o) { perror("open"); return 1; }

    if (fread(weights, 1, (size_t) (n_out * row_bytes), f_w) != (size_t) (n_out * row_bytes)) {
        fprintf(stderr, "short read on weights\n");
        return 1;
    }
    if (fread(acts, 1, (size_t) (n_blocks * sizeof(block_q8_0)), f_a) != (size_t) (n_blocks * sizeof(block_q8_0))) {
        fprintf(stderr, "short read on activations\n");
        return 1;
    }
    fclose(f_w);
    fclose(f_a);

    for (int64_t row = 0; row < n_out; ++row) {
        out[row] = vec_dot_q5_0_q8_0_arm(&weights[row * n_blocks], acts, (int) n_in);
    }

    fwrite(out, 1, (size_t) (n_out * sizeof(float)), f_o);
    fclose(f_o);

    fprintf(stderr, "dot %lld rows of %lld elements\n", (long long) n_out, (long long) n_in);
    free(weights);
    free(acts);
    free(out);
    return 0;
}
