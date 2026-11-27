// Standalone Q5_1 dequantization oracle.
//
// The block layout and the arithmetic are transcribed verbatim from
// ggml's `src/ggml-quants.c` (`dequantize_row_q5_1`) and the F16
// conversion follows the IEEE-754 half-to-single rule that ggml's
// `GGML_FP16_TO_FP32` implements.
//
// The tool reads a file of raw Q5_1 blocks and writes the dequantized
// F32 values as raw little-endian floats, so the Rust kernel can be
// compared against it byte for byte.
//
// usage: q5_1_oracle <in.bin> <out.bin> [n_elements]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK5_1 32

// ggml stores `d` and `m` as two IEEE-754 binary16 values.
//
// The conversion below is a verbatim copy of ggml's
// `ggml_compute_fp16_to_fp32` (`src/ggml-impl.h`), including the
// magic-number trick that handles half-precision subnormals. A
// hand-written bit-twiddling conversion gets the subnormal exponent
// wrong, so the reference implementation is copied instead of
// re-derived.
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
    uint16_t d;      // delta
    uint16_t m;      // min
    uint8_t  qh[4];  // 5-th bit of quants
    uint8_t  qs[QK5_1 / 2];
} block_q5_1;

// Verbatim transcription of ggml's `dequantize_row_q5_1`.
static void dequantize_row_q5_1(const block_q5_1 * x, float * y, int64_t k) {
    static const int qk = QK5_1;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const float m = fp16_to_fp32(x[i].m);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.bin> <out.bin> [n_elements]\n", argv[0]);
        return 2;
    }

    FILE * f_in = fopen(argv[1], "rb");
    if (!f_in) { perror("open in"); return 1; }
    FILE * f_out = fopen(argv[2], "wb");
    if (!f_out) { perror("open out"); return 1; }

    fseek(f_in, 0, SEEK_END);
    const long in_bytes = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    const int64_t n_elements = (argc >= 4) ? atoll(argv[3]) : (int64_t)(in_bytes / 24 * QK5_1);
    if (in_bytes % 24 != 0) {
        fprintf(stderr, "input size %ld is not a multiple of the 24-byte block\n", in_bytes);
        return 1;
    }
    const int64_t n_blocks = n_elements / QK5_1;

    block_q5_1 * blocks = (block_q5_1 *) malloc((size_t) n_blocks * sizeof(block_q5_1));
    float       * values = (float *)        malloc((size_t) n_elements * sizeof(float));
    if (!blocks || !values) { fprintf(stderr, "oom\n"); return 1; }

    if (fread(blocks, 1, (size_t) n_blocks * sizeof(block_q5_1), f_in) !=
        (size_t) n_blocks * sizeof(block_q5_1)) {
        fprintf(stderr, "short read\n");
        return 1;
    }
    fclose(f_in);

    dequantize_row_q5_1(blocks, values, n_elements);

    fwrite(values, 1, (size_t) n_elements * sizeof(float), f_out);
    fclose(f_out);

    fprintf(stderr, "dequantized %lld elements from %lld blocks\n",
            (long long) n_elements, (long long) n_blocks);
    free(blocks);
    free(values);
    return 0;
}
