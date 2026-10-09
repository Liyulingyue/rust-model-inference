float divide_finite_normal(float numerator, float denominator) {
    uint numerator_bits = floatBitsToUint(numerator);
    uint denominator_bits = floatBitsToUint(denominator);
    uint sign = (numerator_bits ^ denominator_bits) & 0x80000000u;
    uint numerator_exponent = (numerator_bits >> 23u) & 0xffu;
    uint denominator_exponent = (denominator_bits >> 23u) & 0xffu;
    if (numerator_exponent == 0u || numerator_exponent == 0xffu
        || denominator_exponent == 0u || denominator_exponent == 0xffu) {
        return numerator / denominator;
    }

    uint numerator_mantissa = (numerator_bits & 0x007fffffu) | 0x00800000u;
    uint denominator_mantissa = (denominator_bits & 0x007fffffu) | 0x00800000u;
    int exponent = int(numerator_exponent) - int(denominator_exponent);
    if (numerator_mantissa < denominator_mantissa) {
        numerator_mantissa <<= 1u;
        exponent -= 1;
    }

    uint quotient = 1u;
    uint remainder = numerator_mantissa - denominator_mantissa;
    for (uint bit = 0u; bit < 23u; ++bit) {
        quotient <<= 1u;
        remainder <<= 1u;
        if (remainder >= denominator_mantissa) {
            remainder -= denominator_mantissa;
            quotient |= 1u;
        }
    }

    remainder <<= 1u;
    bool guard = remainder >= denominator_mantissa;
    if (guard) remainder -= denominator_mantissa;
    bool sticky = remainder != 0u;
    if (guard && (sticky || (quotient & 1u) != 0u)) {
        quotient += 1u;
        if (quotient == 0x01000000u) {
            quotient >>= 1u;
            exponent += 1;
        }
    }

    int result_exponent = exponent + 127;
    if (result_exponent <= 0 || result_exponent >= 255) {
        return numerator / denominator;
    }
    return uintBitsToFloat(
        sign | (uint(result_exponent) << 23u) | (quotient & 0x007fffffu)
    );
}
