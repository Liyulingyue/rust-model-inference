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
