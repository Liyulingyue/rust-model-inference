#ifndef RMI_NEMOTRON_H_SSM_LIBM_GLSL
#define RMI_NEMOTRON_H_SSM_LIBM_GLSL

// glibc's `logf`, transcribed.
//
// The CPU path computes the Mamba-2 softplus with the host's libm:
// `log(1 + exp(z))`. `f32::ln` calls glibc's `logf`, so the device has to
// reproduce *glibc's* `logf` and not merely a correct logarithm.
//
// `host_libm.glsl` already ships a `libm_log`, but it is the x87 80-bit
// logarithm: a different table and a different polynomial. It is more
// accurate than glibc's `logf` -- glibc's `logf` has a 0.818 ulp error
// because it rounds the F64 polynomial to F32 in two steps -- and the more
// accurate answer is not the same answer. On the softplus the two disagree
// by one ulp, and the Mamba-2 recurrence multiplies its state by
// `exp(dt * A)` on every token and every layer, so one ulp here survives
// into the logits.
//
// The algorithm is glibc's `sysdeps/ieee754/flt-32/e_logf.c` (which is
// ARM's optimized-routines `logf`):
//
//   * `x = 2^k * z` with `z` in `[OFF, 2*OFF]` and exact, where
//     `OFF = 0x3f330000`;
//   * the interval is split into sixteen subintervals, and the subinterval
//     `i` carries `1/c` and `log(c)`;
//   * `log(x) = log1p(z/c - 1) + log(c) + k*ln2`, with `log1p` evaluated by
//     a degree-four polynomial.
//
// The F64 arithmetic is what makes the result match: the table and the
// polynomial are evaluated in F64 and rounded to F32 once, at the end.
// `precise` keeps the compiler from contracting the polynomial into fused
// multiply-adds, which would change the rounding.

#extension GL_ARB_gpu_shader_fp64 : enable

// glibc's `expf` lives in `glibc_expf.glsl` so that the softmax kernels
// can share the same transcription.
#include "glibc_expf.glsl"

// `1 / c` for the sixteen subintervals.
const double LOGF_INVC[16] = double[16](
    1.3989071621465281lf,
    1.3403141896637998lf,
    1.286432210124115lf,
    1.2367150214269895lf,
    1.1906977166711752lf,
    1.1479821020556429lf,
    1.1082251448272158lf,
    1.0711297413057381lf,
    1.0364372789772831lf,
    1.0lf,
    0.9492859795739057lf,
    0.89510494286090037lf,
    0.84768216203511026lf,
    0.80503148516920009lf,
    0.7664671008843108lf,
    0.73142860331632797
);

// `log(c)` for the sixteen subintervals.
const double LOGF_LOGC[16] = double[16](
    -0.33569133332882284lf,
    -0.29290405637740741lf,
    -0.25187265809373688lf,
    -0.21245868807117255lf,
    -0.17453945183745634lf,
    -0.1380057072319758lf,
    -0.10275976698545139lf,
    -0.068713924470205245lf,
    -0.035789138739822798lf,
    0.0lf,
    0.052045177429294959lf,
    0.11081431298787942lf,
    0.16524952236951429lf,
    0.21687389031699977lf,
    0.26596350281213971lf,
    0.31275566640735569
);

// `ln(2)`.
const double LOGF_LN2 = 0.69314718055994529lf;

// The `log1p` polynomial: `A0 * r^4 + A1 * r^3 + A2 * r^2 + r`.
const double LOGF_A0 = -0.25089342214237154lf;
const double LOGF_A1 = 0.33345676574406602lf;
const double LOGF_A2 = -0.49999974858021029lf;

float glibc_logf(float x) {
    uint ix = floatBitsToUint(x);
    // `log(1) == 0` exactly, and the sign of zero must not matter.
    if (ix == 0x3f800000u) {
        return 0.0;
    }
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        // `x < 2^-126`, infinite or NaN.
        if (ix * 2u == 0u) {
            return -1.0 / 0.0;
        }
        if (ix == 0x7f800000u) {
            return x;
        }
        if ((ix & 0x80000000u) != 0u || ix * 2u >= 0xff000000u) {
            return 0.0 / 0.0;
        }
        // Subnormal: normalise it.
        ix = floatBitsToUint(uintBitsToFloat(ix) * 8388608.0);
        ix -= 23u << 23u;
    }
    // `x = 2^k * z`, with `z` in `[OFF, 2*OFF]` and exact.
    uint tmp = ix - 0x3f330000u;
    uint i = (tmp >> 19u) % 16u;
    int k = int(tmp) >> 23;
    uint iz = ix - (tmp & 0xff800000u);
    double invc = LOGF_INVC[i];
    double logc = LOGF_LOGC[i];
    double z = double(uintBitsToFloat(iz));
    // `log(x) = log1p(z/c - 1) + log(c) + k*ln2`.
    precise double r = z * invc - 1.0;
    precise double y0 = logc + double(k) * LOGF_LN2;
    precise double r2 = r * r;
    precise double y = LOGF_A1 * r + LOGF_A2;
    y = LOGF_A0 * r2 + y;
    y = y * r2 + (y0 + r);
    return float(y);
}

#endif
