uint f32_to_f16_rte(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint exponent = (bits >> 23u) & 0xffu;
    uint mantissa = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        uint payload = mantissa >> 13u;
        return sign | 0x7c00u | (mantissa == 0u ? 0u : max(payload, 1u));
    }

    int half_exponent = int(exponent) - 112;
    if (half_exponent >= 31) return sign | 0x7c00u;
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x00800000u;
        uint shift = uint(14 - half_exponent);
        uint result = mantissa >> shift;
        uint remainder = mantissa & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (result & 1u) != 0u)) {
            result += 1u;
        }
        return sign | result;
    }

    uint result = mantissa >> 13u;
    uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0u)) {
        result += 1u;
        if (result == 0x400u) {
            result = 0u;
            half_exponent += 1;
            if (half_exponent == 31) return sign | 0x7c00u;
        }
    }
    return sign | (uint(half_exponent) << 10u) | result;
}

float round_f16_rte(float value) {
    return unpackHalf2x16(f32_to_f16_rte(value)).x;
}
