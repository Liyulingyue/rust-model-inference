// Reference for the official Voxtral causal Conv1d + exact GELU graph.
// Build with contraction and vectorization disabled; weights arrive as F32
// decoded exactly from the original BF16 safetensors.
#include <math.h>
#include <stddef.h>

// Keep clang from replacing separate scalar calls with sincosf.
__attribute__((noinline)) static float scalar_cos(float value) { return cosf(value); }
__attribute__((noinline)) static float scalar_sin(float value) { return sinf(value); }

void causal_conv_gelu(const float *input, const float *weight, const float *bias,
                      float *output, size_t frames, size_t in_channels,
                      size_t out_channels, size_t stride) {
    const size_t left_pad = 3 - stride;
    const size_t out_frames = (frames + left_pad - 3) / stride + 1;
    for (size_t t = 0; t < out_frames; ++t)
        for (size_t o = 0; o < out_channels; ++o) {
            float sum = bias[o];
            for (size_t i = 0; i < in_channels; ++i)
                for (size_t k = 0; k < 3; ++k) {
                    const size_t source = t * stride + k;
                    if (source >= left_pad && source - left_pad < frames)
                        sum += input[(source - left_pad) * in_channels + i]
                               * weight[(o * in_channels + i) * 3 + k];
                }
            output[t * out_channels + o] =
                0.5f * sum * (1.0f + erff(sum * 0.7071067811865475f));
        }
}

void rms_norm_f32(const float *input, const float *weight, float *output,
                  size_t width, float epsilon) {
    float sum = 0.0f;
    for (size_t i = 0; i < width; ++i)
        sum += input[i] * input[i];
    const float scale = 1.0f / sqrtf(sum / (float)width + epsilon);
    for (size_t i = 0; i < width; ++i)
        output[i] = weight[i] * (input[i] * scale);
}

void linear_f32(const float *input, const float *weight, const float *bias,
                float *output, size_t in_channels, size_t out_channels) {
    for (size_t o = 0; o < out_channels; ++o) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_channels; ++i)
            sum += input[i] * weight[o * in_channels + i];
        output[o] = bias ? sum + bias[o] : sum;
    }
}

void add_f32(float *left, const float *right, size_t length) {
    for (size_t i = 0; i < length; ++i)
        left[i] += right[i];
}

void add_scalar_f32(float *values, float scalar, size_t length) {
    for (size_t i = 0; i < length; ++i)
        values[i] += scalar;
}

void mul_f32(float *left, const float *right, size_t length) {
    for (size_t i = 0; i < length; ++i)
        left[i] *= right[i];
}

void silu_mul_f32(float *gate, const float *up, size_t length) {
    for (size_t i = 0; i < length; ++i)
        gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

void gelu_f32(float *values, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        const float value = values[i];
        values[i] = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
    }
}

void time_condition_f32(const float *frame_embedding, float *output) {
    const float log_theta = (float)log(10000.0);
    for (size_t i = 0; i < 1024; ++i) {
        const float frequency = expf(-log_theta * (float)i / 1024.0f);
        const float phase = 6.0f * frequency;
        output[i] = scalar_cos(phase) + frame_embedding[i];
        output[1024 + i] = scalar_sin(phase) + frame_embedding[1024 + i];
    }
}

void rope_f32(float *values, size_t position) {
    for (size_t i = 0; i < 32; ++i) {
        const float frequency = 1.0f / powf(1000000.0f, (float)(2 * i) / 64.0f);
        const float theta = (float)position * frequency;
        const float cosine = scalar_cos(theta), sine = scalar_sin(theta);
        for (size_t head = 0; head < 32; ++head) {
            const size_t low = head * 64 + i, high = low + 32;
            const float a = values[low], b = values[high];
            values[low] = a * cosine + (-b) * sine;
            values[high] = b * cosine + a * sine;
        }
    }
}

void attention_f32(const float *query, const float *keys, const float *values,
                   float *output, size_t frames) {
    if (frames < 1 || frames > 16) __builtin_trap();
    for (size_t head = 0; head < 32; ++head) {
        const size_t offset = head * 64;
        float scores[16], maximum = -INFINITY, sum = 0.0f;
        for (size_t past = 0; past < frames; ++past) {
            float score = 0.0f;
            for (size_t dimension = 0; dimension < 64; ++dimension)
                score += query[offset + dimension]
                       * keys[past * 2048 + offset + dimension];
            scores[past] = score * 0.125f;
            maximum = fmaxf(maximum, scores[past]);
        }
        for (size_t past = 0; past < frames; ++past) {
            scores[past] = expf(scores[past] - maximum);
            sum += scores[past];
        }
        for (size_t dimension = 0; dimension < 64; ++dimension)
            output[offset + dimension] = 0.0f;
        for (size_t past = 0; past < frames; ++past)
            for (size_t dimension = 0; dimension < 64; ++dimension)
                output[offset + dimension] +=
                    (scores[past] / sum) * values[past * 2048 + offset + dimension];
    }
}

void rope_text_f32(float *values, size_t heads, size_t position) {
    for (size_t i = 0; i < 64; ++i) {
        const float frequency = 1.0f / powf(1000000.0f, (float)(2 * i) / 128.0f);
        const float theta = (float)position * frequency;
        const float cosine = scalar_cos(theta), sine = scalar_sin(theta);
        for (size_t head = 0; head < heads; ++head) {
            const size_t low = head * 128 + i, high = low + 64;
            const float a = values[low], b = values[high];
            values[low] = a * cosine + (-b) * sine;
            values[high] = b * cosine + a * sine;
        }
    }
}

void attention_text_f32(const float *query, const float *keys, const float *values,
                        float *output, size_t tokens) {
    if (tokens < 1 || tokens > 4) __builtin_trap();
    const float scale = 1.0f / sqrtf(128.0f);
    for (size_t head = 0; head < 16; ++head) {
        const size_t kv = head / 8;
        float scores[4], maximum = -INFINITY, sum = 0.0f;
        for (size_t past = 0; past < tokens; ++past) {
            float score = 0.0f;
            for (size_t i = 0; i < 128; ++i)
                score += query[head * 128 + i] * keys[past * 256 + kv * 128 + i];
            scores[past] = score * scale;
            maximum = fmaxf(maximum, scores[past]);
        }
        for (size_t past = 0; past < tokens; ++past) {
            scores[past] = expf(scores[past] - maximum);
            sum += scores[past];
        }
        for (size_t i = 0; i < 128; ++i)
            output[head * 128 + i] = 0.0f;
        for (size_t past = 0; past < tokens; ++past)
            for (size_t i = 0; i < 128; ++i)
                output[head * 128 + i] +=
                    (scores[past] / sum) * values[past * 256 + kv * 128 + i];
    }
}
