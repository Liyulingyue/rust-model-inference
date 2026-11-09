#ifndef RMI_EXP_GGML_GLSL
#define RMI_EXP_GGML_GLSL

// ggml's `ggml_v_expf`, the exponential behind every Mamba-2 SiLU.
//
// The CPU path (`silu_ggml_inplace` in `src/ops/activation/silu.rs`) runs
// ggml's own vector exponential, transcribed from `ggml/src/ggml-cpu/vec.h`.
// The repository already ships two other exponentials and neither of them
// matches it:
//
//   * `exp_approx_neon.glsl` carries ggml's range-reduction constants but
//     evaluates the degree-5 polynomial in a different order, so it differs
//     in the last mantissa bit;
//   * `exp_accurate.glsl` (in `host_libm.glsl`) uses the *same* range
//     reduction but a much longer double-double polynomial, so it is more
//     accurate and therefore also different.
//
// The Mamba-2 recurrence multiplies its state by `exp(dt * A)` on every
// token and every layer, and the SiLU gates the scan output, so a
// one-ULP difference in the exponential survives into the logits.
//
// Two details of ggml's ARM path are load-bearing and are reproduced here:
//
//   * the range reduction uses the "add a magic constant" trick:
//     `z = R + x * log2(e)` with `R = 0x1.8p23`, and the low nine bits of
//     `z`'s mantissa carry the integer part of `x * log2(e)`, which is
//     turned into `2^n` by a bit shift;
//   * the fast path is chosen **per group of four lanes**, not per lane.
//     ggml computes `vaddvq_u32(c) == 0` over the whole `float32x4_t`, so a
//     single lane with `|n| > 126` pushes all four lanes of that vector
//     through the overflow path, where the non-overflowing lanes use
//     `k * k + j` instead of `k + k * j`. The values differ, so the group
//     decision has to be reproduced exactly.
//
// The CPU calls the NEON path four lanes at a time, so this file exposes a
// four-lane entry point and the callers process four elements per
// invocation.

#include "f32_divide.glsl"

// The four results are written through an `out` array: a GLSL array
// parameter is an `in` by default, and a plain parameter's writes are
// discarded when the function returns.
void exp_ggml4(float x[4], out float result[4]) {
    const float r = uintBitsToFloat(0x4b400000u);      // 0x1.8p23
    const float log2e = uintBitsToFloat(0x3fb8aa3bu);  // 0x1.715476p+0
    const float c1 = uintBitsToFloat(0x3f317200u);     // 0x1.62e4p-1
    const float c2 = uintBitsToFloat(0x35bfbe8eu);     // 0x1.7f7d1cp-20
    const float p0 = uintBitsToFloat(0x3f7ffff6u);     // 0x1.ffffecp-1
    const float p1 = uintBitsToFloat(0x3efffedbu);     // 0x1.fffdb6p-2
    const float p2 = uintBitsToFloat(0x3e2aaf33u);     // 0x1.555e66p-3
    const float p3 = uintBitsToFloat(0x3d2b9f17u);     // 0x1.573e2ep-5
    const float p4 = uintBitsToFloat(0x3c072010u);     // 0x1.0e4020p-7

    float z[4];
    float n[4];
    float j[4];
    uint exponent[4];
    float scale[4];
    bool overflow[4];
    bool group_overflow = false;

    for (int lane = 0; lane < 4; lane++) {
        precise float value = x[lane];
        // ggml: `z = fma(r, x, LOG2E)`, where the ACLE `fma(a, b, c)` is
        // `a + b * c`, so this is `r + x * log2(e)`.
        precise float reduced = fma(value, log2e, r);
        z[lane] = reduced;
        precise float offset = reduced - r;
        n[lane] = offset;
        // ggml: `b = fms(fms(x, n, C1), n, C2)`, i.e. `(x - n*C1) - n*C2`
        // with one rounding each.
        precise float first = fma(-offset, c1, value);
        precise float second = fma(-offset, c2, first);
        precise float square = second * second;
        // ggml: `j = fma(P0*b, fma(fma(P1,P2,b), fma(P3,b,P4), u), u)`.
        // The ACLE `fma(a, b, c)` is `a + b*c`, so each term maps onto
        // GLSL's `fma(b, c, a)`.
        precise float t1 = fma(p2, second, p1);
        precise float t2 = fma(second, p4, p3);
        precise float t3 = fma(t2, square, t1);
        precise float t4 = fma(t3, square, p0 * second);
        j[lane] = t4;
        uint bits = uint(floatBitsToUint(reduced)) << 23u;
        exponent[lane] = bits;
        scale[lane] = uintBitsToFloat(bits + 0x3f800000u);
        bool over = abs(offset) > 126.0;
        overflow[lane] = over;
        group_overflow = group_overflow || over;
    }

    for (int lane = 0; lane < 4; lane++) {
        if (!group_overflow) {
            // ggml: `return fma(k, j, k)` -- the ACLE form is `k + j*k`.
            precise float value = fma(scale[lane], j[lane], scale[lane]);
            result[lane] = value;
            continue;
        }
        // ggml's overflow path.
        uint adjust = n[lane] <= 0.0 ? 0x82000000u : 0u;
        precise float s1 = uintBitsToFloat(adjust + 0x7f000000u);
        precise float s2 = uintBitsToFloat(exponent[lane] - adjust);
        bool big = abs(n[lane]) > 192.0;
        precise float inner;
        if (overflow[lane]) {
            // ggml: `vmulq_f32(fma(s2, s2, j), s1)` -- `fma(s2, s2, j)` is
            // `s2 + s2*j` under the ACLE convention.
            precise float scaled = fma(s2, j[lane], s2);
            inner = scaled * s1;
        } else {
            inner = fma(scale[lane], j[lane], scale[lane]);
        }
        result[lane] = big ? (s1 * s1) : inner;
    }
}

// ggml's SiLU: `x / (1 + exp(-x))`.
//
// The CPU divides with `vdivq_f32`, a true IEEE division, so the device
// uses the repository's correctly rounded helper rather than the driver's
// `/`.
void silu_ggml4(float x[4], out float result[4]) {
    float negated[4];
    float exponential[4];
    for (int lane = 0; lane < 4; lane++) {
        negated[lane] = 0.0 - x[lane];
    }
    exp_ggml4(negated, exponential);
    for (int lane = 0; lane < 4; lane++) {
        precise float denominator = 1.0 + exponential[lane];
        result[lane] = divide_finite_normal(x[lane], denominator);
    }
}

#endif
