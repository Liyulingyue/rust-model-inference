// glibc's `expf`, transcribed.
//
// The algorithm is glibc's `sysdeps/ieee754/flt-32/e_expf.c`, which is
// ARM's optimized-routines `expf`: the argument is scaled by
// `32 * log2(e)`, rounded to an integer `k`, and the result is
//
//   exp(x) = 2^(k/32) * (C0*r^3 + C1*r^2 + C2*r + 1)
//
// with `r` the residual in `[-1/2, 1/2]`. The `2^(k/32)` factor comes from
// a 32-entry table of `2^(i/32)` and the exponent `k/32`.
//
// glibc's `expf` has a 0.502 ulp error, so it is *not* the correctly
// rounded exponential: the repository's `libm_exp` computes the
// correctly rounded value in double-double arithmetic and disagrees with
// glibc on about one value in a hundred. The Mamba-2 decay
// `exp(dt * A_log)` multiplies the scan state on every token of every
// layer, so the disagreement is not harmless.
//
// glibc builds `2^(k/32)` by adding `k` into the exponent field of the
// table entry, which needs 64-bit bit manipulation. The same value is
// obtained here by scaling the table entry with the exact power of two
// `2^(k/32 - i/32)`, which is what the bit manipulation computes.
//
// The transcription needs `shaderFloat64`: the intermediates are genuine
// doubles, and a two-float emulation rounds them to ~49 bits, which
// disagrees with glibc on roughly one value in a hundred.

#ifndef RMI_GLIBC_EXPF_GLSL
#define RMI_GLIBC_EXPF_GLSL

#extension GL_ARB_gpu_shader_fp64 : enable

// `32 * log2(e)`.
const double EXPF_INVLN2N = 46.166241308446828lf;
// `0x1.8p+52`: the magic constant whose ulp is one.
const double EXPF_SHIFT = 6755399441055744.0lf;
// The `log1p` polynomial, scaled by `1/32` three times.
const double EXPF_C0 = 1.6938359250920212e-06lf;
const double EXPF_C1 = 0.00023459809789509004lf;
const double EXPF_C2 = 0.021660849396613134lf;

// `2^(i/32)` for the sixteen ... thirty-two subintervals.
const double EXPF_TAB[32] = double[32](
    1.0lf,
    1.0218971486541166lf,
    1.0442737824274138lf,
    1.0671404006768237lf,
    1.0905077326652577lf,
    1.1143867425958924lf,
    1.1387886347566916lf,
    1.1637248587775775lf,
    1.189207115002721lf,
    1.215247359980469lf,
    1.241857812073484lf,
    1.2690509571917332lf,
    1.2968395546510096lf,
    1.3252366431597413lf,
    1.3542555469368927lf,
    1.383909881963832lf,
    1.4142135623730951lf,
    1.4451808069770467lf,
    1.4768261459394993lf,
    1.5091644275934228lf,
    1.5422108254079407lf,
    1.5759808451078865lf,
    1.6104903319492543lf,
    1.6457554781539649lf,
    1.681792830507429lf,
    1.7186192981224779lf,
    1.7562521603732995lf,
    1.7947090750031072lf,
    1.8340080864093424lf,
    1.8741676341103lf,
    1.9152065613971474lf,
    1.9571441241754002
);

float glibc_expf(float x) {
    uint bits = floatBitsToUint(x);
    // `top12(x)`, the top twelve bits of the float.
    uint abstop = (bits >> 20) & 0x7ffu;
    if (abstop >= ((floatBitsToUint(88.0f) >> 20) & 0x7ffu)) {
        // `|x| >= 88` or NaN.
        if (bits == 0xff800000u) {
            return 0.0;
        }
        if (abstop >= ((0x7f800000u >> 20) & 0x7ffu)) {
            // NaN or `+inf`.
            return x + x;
        }
        if (x > 88.722831726074219f) {
            // Above `log(0x1p128)`: overflow.
            return uintBitsToFloat(0x7f800000u);
        }
        if (x < -103.97207641601562f) {
            // Below `log(0x1p-150)`: underflow.
            return 0.0;
        }
    }
    // `x * 32 * log2(e) = k + r` with `r` in `[-1/2, 1/2]` and `k` an
    // integer. The magic constant has an ulp of one, so the sum rounds
    // `z` to the nearest integer.
    precise double z = EXPF_INVLN2N * double(x);
    precise double kd = z + EXPF_SHIFT;
    precise double k = kd - EXPF_SHIFT;
    int ki = int(k);
    // `i` is `k` modulo the table size and `j` the rest; glibc takes the
    // table index from the low bits of the rounded value, which is the
    // same thing for the range the algorithm covers.
    int i = ((ki % 32) + 32) % 32;
    int j = (ki - i) / 32;
    // `2^(k/32)`: the table entry scaled by the exact power of two
    // `2^j`. The exponent is split in half so that both halves fit in an
    // F32 power of two.
    precise double s = EXPF_TAB[i];
    int exponent_half = int(floor(double(j) / 2.0));
    s *= double(uintBitsToFloat(uint(exponent_half + 127) << 23u));
    s *= double(uintBitsToFloat(uint(j - exponent_half + 127) << 23u));
    precise double r = z - k;
    precise double poly = EXPF_C0 * r + EXPF_C1;
    precise double r2 = r * r;
    precise double y = EXPF_C2 * r + 1.0;
    y = poly * r2 + y;
    y = y * s;
    return float(y);
}
#endif
