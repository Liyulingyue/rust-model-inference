// Independent IEEE F32 primitives for the pinned official Python graph.
// Compile without vectorization, contraction, reassociation or external BLAS.
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

void linear(const float *x, const float *w, const float *b, float *y,
            size_t rows, size_t in, size_t out) {
    for (size_t r = 0; r < rows; ++r)
        for (size_t o = 0; o < out; ++o) {
            float v = 0;
            for (size_t k = 0; k < in; ++k) v += x[r*in+k] * w[o*in+k];
            y[r*out+o] = b ? v + b[o] : v;
        }
}

void unary(const float *x, float *y, size_t n, int op, float a) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        switch (op) {
            case 0: y[i] = -v; break;
            case 1: y[i] = expf(v); break;
            case 2: y[i] = logf(v); break;
            case 3: y[i] = sinf(v); break;
            case 4: y[i] = cosf(v); break;
            case 5: y[i] = 0.5f*v*(1.0f+erff(v*0.7071067811865475f)); break;
            case 6: y[i] = fmaxf(v, a); break;
            case 7: y[i] = 1.0f/v; break;
            case 8: y[i] = powf(a, v); break;
            default: abort();
        }
    }
}

void binary(const float *x, const float *z, float *y, size_t n, int op) {
    for (size_t i = 0; i < n; ++i)
        switch (op) {
            case 0: y[i] = x[i]+z[i]; break;
            case 1: y[i] = x[i]-z[i]; break;
            case 2: y[i] = x[i]*z[i]; break;
            case 3: y[i] = x[i]/z[i]; break;
            default: abort();
        }
}

void norm(const float *x, const float *w, const float *b, float *y,
          size_t rows, size_t d, float eps) {
    for (size_t r = 0; r < rows; ++r) {
        const float *row = x+r*d;
        float mean = 0, var = 0;
        for (size_t i = 0; i < d; ++i) mean += row[i];
        mean /= (float)d;
        for (size_t i = 0; i < d; ++i) {
            float delta = row[i]-mean;
            var += delta*delta;
        }
        var /= (float)d;
        float scale = 1.0f/sqrtf(var+eps);
        for (size_t i = 0; i < d; ++i)
            y[r*d+i] = (row[i]-mean)*scale*w[i]+(b ? b[i] : 0.0f);
    }
}

void softmax(const float *x, float *y, size_t rows, size_t d) {
    for (size_t r = 0; r < rows; ++r) {
        float max = -INFINITY, sum = 0;
        for (size_t i = 0; i < d; ++i) max = fmaxf(max, x[r*d+i]);
        for (size_t i = 0; i < d; ++i) {
            y[r*d+i] = expf(x[r*d+i]-max);
            sum += y[r*d+i];
        }
        for (size_t i = 0; i < d; ++i) y[r*d+i] /= fmaxf(sum, 1e-12f);
    }
}

void sum_rows(const float *x, float *y, size_t rows, size_t d) {
    for (size_t r = 0; r < rows; ++r) {
        float sum = 0;
        for (size_t i = 0; i < d; ++i) sum += x[r*d+i];
        y[r] = sum;
    }
}

void attention(const float *q, const float *k, const float *v, const float *mask,
               float *y, size_t batches, size_t nq, size_t nk, size_t d, float scale) {
    float *scores = malloc(nk*sizeof(float));
    if (!scores) abort();
    for (size_t b = 0; b < batches; ++b)
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < nk; ++j) {
                float dot = 0;
                for (size_t a = 0; a < d; ++a)
                    dot += q[(b*nq+i)*d+a]*k[(b*nk+j)*d+a];
                scores[j] = dot*scale+(mask ? mask[(b*nq+i)*nk+j] : 0.0f);
            }
            softmax(scores, scores, 1, nk);
            for (size_t a = 0; a < d; ++a) {
                float dot = 0;
                for (size_t j = 0; j < nk; ++j) dot += scores[j]*v[(b*nk+j)*d+a];
                y[(b*nq+i)*d+a] = dot;
            }
        }
    free(scores);
}
