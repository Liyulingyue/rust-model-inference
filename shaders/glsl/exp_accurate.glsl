vec2 exp_horner(vec2 polynomial, float reduced, uint high_bits, uint low_bits) {
    precise float product = polynomial.x * reduced;
    precise float error = fma(polynomial.x, reduced, -product) + polynomial.y * reduced;
    precise float coefficient = uintBitsToFloat(high_bits);
    precise float sum = product + coefficient;
    precise float virtual_coefficient = sum - product;
    precise float sum_error =
        (product - (sum - virtual_coefficient)) + (coefficient - virtual_coefficient);
    sum_error += error + uintBitsToFloat(low_bits);
    precise float high = sum + sum_error;
    precise float low = sum_error - (high - sum);
    return vec2(high, low);
}

float exp_accurate(float x) {
    if (isnan(x) || x == 1.0 / 0.0) return x;
    if (x == -1.0 / 0.0) return 0.0;
    precise float z = fma(x, uintBitsToFloat(0x3fb8aa3bu), uintBitsToFloat(0x4b400000u));
    precise float n = z - uintBitsToFloat(0x4b400000u);
    if (n > 128.0) return 1.0 / 0.0;
    if (n < -150.0) return 0.0;
    precise float first = fma(-n, uintBitsToFloat(0x3f317200u), x);
    precise float reduced = fma(-n, uintBitsToFloat(0x35bfbe8eu), first);
    precise float reduced_error = fma(-n, uintBitsToFloat(0x35bfbe8eu), first - reduced);
    reduced_error = fma(-n, uintBitsToFloat(0x29779abdu), reduced_error);
    uint exponent = floatBitsToUint(z) << 23u;
    precise float scale = uintBitsToFloat(exponent + floatBitsToUint(1.0));
    // Keep the low terms until F32 rounding: a one-ULP exp error can cross an F16 probability tie.
    vec2 polynomial = vec2(uintBitsToFloat(0x32d7322bu), uintBitsToFloat(0x25fea89cu));
    polynomial = exp_horner(polynomial, reduced, 0x3493f27eu, 0xa808760au);
    polynomial = exp_horner(polynomial, reduced, 0x3638ef1du, 0x292ad8e6u);
    polynomial = exp_horner(polynomial, reduced, 0x37d00d01u, 0xaabfcbfdu);
    polynomial = exp_horner(polynomial, reduced, 0x39500d01u, 0xac3fcbfdu);
    polynomial = exp_horner(polynomial, reduced, 0x3ab60b61u, 0xae13e93fu);
    polynomial = exp_horner(polynomial, reduced, 0x3c088889u, 0xafeeeeefu);
    polynomial = exp_horner(polynomial, reduced, 0x3d2aaaabu, 0xb0aaaaabu);
    polynomial = exp_horner(polynomial, reduced, 0x3e2aaaabu, 0xb1aaaaabu);
    polynomial = exp_horner(polynomial, reduced, 0x3f000000u, 0u);
    polynomial = exp_horner(polynomial, reduced, 0x3f800000u, 0u);
    polynomial = exp_horner(polynomial, reduced, 0x3f800000u, 0u);
    precise float value = polynomial.x + (polynomial.x * reduced_error + polynomial.y);
    if (abs(n) <= 126.0) return scale * value;

    uint adjust = n <= 0.0 ? 0x82000000u : 0u;
    precise float scale1 = uintBitsToFloat(adjust + 0x7f000000u);
    precise float scale2 = uintBitsToFloat(exponent - adjust);
    return scale2 * value * scale1;
}
