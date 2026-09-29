// Edge0 affine/LoRA reference. Build with vectorization and FP contraction off.
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float bf16_at(const unsigned char *data, size_t index) {
    uint16_t bits;
    memcpy(&bits, data + 2 * index, sizeof(bits));
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static float f16_at(const unsigned char *data, size_t index) {
    _Float16 value;
    memcpy(&value, data + 2 * index, sizeof(value));
    return (float)value;
}

int edge0_affine(const unsigned char *packed, const unsigned char *scales,
                 const unsigned char *biases, const unsigned char *lora_a,
                 const unsigned char *lora_b, const float *input, float *output,
                 size_t n_in, size_t n_out, unsigned bits, size_t rank,
                 float lora_scale) {
    if (!packed || !scales || !biases || !input || !output || !n_in || !n_out ||
        n_in % 64 || (bits != 4 && bits != 8) ||
        (rank == 0 && (lora_a || lora_b)) ||
        (rank != 0 && (!lora_a || !lora_b))) {
        return -1;
    }
    float *low = rank ? malloc(rank * sizeof(float)) : NULL;
    if (rank && !low) return -2;
    for (size_t r = 0; r < rank; ++r) {
        float total = 0.0f;
        for (size_t col = 0; col < n_in; ++col)
            total += input[col] * f16_at(lora_a, r * n_in + col);
        low[r] = total;
    }
    const size_t per_word = 32 / bits;
    const size_t words_per_row = n_in / per_word;
    const size_t groups_per_row = n_in / 64;
    const uint32_t mask = (1u << bits) - 1u;
    for (size_t row = 0; row < n_out; ++row) {
        float total = 0.0f;
        for (size_t col = 0; col < n_in; ++col) {
            uint32_t word;
            memcpy(&word, packed + 4 * (row * words_per_row + col / per_word),
                   sizeof(word));
            const uint32_t q = (word >> (bits * (col % per_word))) & mask;
            const size_t group = row * groups_per_row + col / 64;
            const float weight = bf16_at(scales, group) * (float)q +
                                 bf16_at(biases, group);
            total += input[col] * weight;
        }
        if (rank) {
            float delta = 0.0f;
            for (size_t r = 0; r < rank; ++r)
                delta += low[r] * f16_at(lora_b, row * rank + r);
            total += lora_scale * delta;
        }
        output[row] = total;
    }
    free(low);
    return 0;
}
