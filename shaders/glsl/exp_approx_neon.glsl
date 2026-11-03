float exp_approx_neon(float x) {
    precise float z = fma(x, uintBitsToFloat(0x3fb8aa3bu), uintBitsToFloat(0x4b400000u));
    precise float n = z - uintBitsToFloat(0x4b400000u);
    precise float reduced = fma(-n, uintBitsToFloat(0x3f317200u), x);
    reduced = fma(-n, uintBitsToFloat(0x35bfbe8eu), reduced);
    uint exponent = floatBitsToUint(z) << 23u;
    precise float scale = uintBitsToFloat(exponent + floatBitsToUint(1.0));
    precise float square = reduced * reduced;
    precise float low = fma(uintBitsToFloat(0x3e2aaf33u), reduced, uintBitsToFloat(0x3efffedbu));
    precise float high = fma(uintBitsToFloat(0x3c072010u), reduced, uintBitsToFloat(0x3d2b9f17u));
    precise float polynomial = fma(high, square, low);
    polynomial = fma(polynomial, square, uintBitsToFloat(0x3f7ffff6u) * reduced);
    if (abs(n) <= 126.0) return fma(scale, polynomial, scale);
    uint adjust = n <= 0.0 ? 0x82000000u : 0u;
    precise float scale1 = uintBitsToFloat(adjust + 0x7f000000u);
    precise float scale2 = uintBitsToFloat(exponent - adjust);
    if (abs(n) > 192.0) return scale1 * scale1;
    return fma(scale2, polynomial, scale2) * scale1;
}
