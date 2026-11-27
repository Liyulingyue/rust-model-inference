/* Scalar CPU kernels for the pinned GLiNER2/Transformers graph. */
#include <math.h>
#include <stddef.h>

void linear_f32(const float *input, const float *weight, const float *bias,
                float *output, size_t rows, size_t in, size_t out) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < out; ++column) {
            double sum = 0.0;
            for (size_t index = 0; index < in; ++index) {
                float product = input[row * in + index] * weight[column * in + index];
                sum += (double)product;
            }
            output[row * out + column] = (float)sum + (bias ? bias[column] : 0.0f);
        }
    }
}

void bmm_f32(const float *left, const float *right, float *output,
             size_t batch, size_t rows, size_t inner, size_t cols) {
    for (size_t b = 0; b < batch; ++b) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < cols; ++column) {
                double sum = 0.0;
                for (size_t index = 0; index < inner; ++index) {
                    float product = left[(b * rows + row) * inner + index]
                        * right[(b * inner + index) * cols + column];
                    sum += (double)product;
                }
                output[(b * rows + row) * cols + column] = (float)sum;
            }
        }
    }
}

void norm_f32(const float *input, const float *weight, const float *bias,
              float *output, size_t rows, size_t width, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        const float *x = input + row * width;
        float *y = output + row * width;
        double sum = 0.0;
        for (size_t i = 0; i < width; ++i) sum += (double)x[i];
        double mean = sum / (double)width;
        double variance = 0.0;
        for (size_t i = 0; i < width; ++i) {
            double centered = (double)x[i] - mean;
            variance += centered * centered;
        }
        variance /= (double)width;
        float scale = 1.0f / sqrtf((float)variance + eps);
        for (size_t i = 0; i < width; ++i)
            y[i] = ((x[i] - (float)mean) * scale) * weight[i] + bias[i];
    }
}

void softmax_f32(const float *input, float *output, size_t rows, size_t width) {
    for (size_t row = 0; row < rows; ++row) {
        const float *x = input + row * width;
        float *y = output + row * width;
        float maximum = -INFINITY;
        for (size_t i = 0; i < width; ++i) if (x[i] > maximum) maximum = x[i];
        double sum = 0.0;
        for (size_t i = 0; i < width; ++i) {
            y[i] = expf(x[i] - maximum);
            sum += (double)y[i];
        }
        float scale = (float)(1.0 / sum);
        for (size_t i = 0; i < width; ++i) y[i] *= scale;
    }
}

void gelu_f32(const float *input, float *output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float x = input[i];
        output[i] = (0.5f * x) * (1.0f + erff(x * 0.7071067811865476f));
    }
}
