// Half products are exact in float; TwoSum prevents double rounding at half ties.
float half_fma(float a, float b, float c) {
    precise float p = a * b;
    precise float s = p + c;
    precise float v = s - p;
    precise float error = (p - (s - v)) + (c - v);
    uint bits = floatBitsToUint(s);
    uint exponent = (bits >> 23u) & 255u;
    if (error != 0.0 && exponent >= 102u && exponent <= 142u) {
        uint shift = exponent <= 112u ? 126u - exponent : 13u;
        uint mantissa = (bits & 0x7fffffu) | 0x800000u;
        if ((mantissa & ((1u << shift) - 1u)) == (1u << (shift - 1u))) {
            bits += ((error < 0.0) == (s < 0.0)) ? 1u : 0xffffffffu;
        }
    }
    return round_f16_rte(uintBitsToFloat(bits));
}

// Match the four eight-lane FP16 accumulators of the CPU dot.
float reduce_half32(float lanes[32]) {
    precise float pairs[4];
    for (uint i = 0u; i < 4u; i++) {
        precise float a = round_f16_rte(lanes[i] + lanes[i + 16u]);
        precise float b = round_f16_rte(lanes[i + 8u] + lanes[i + 24u]);
        precise float c = round_f16_rte(lanes[i + 4u] + lanes[i + 20u]);
        precise float d = round_f16_rte(lanes[i + 12u] + lanes[i + 28u]);
        pairs[i] = round_f16_rte(a + b) + round_f16_rte(c + d);
    }
    precise float result = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3]);
    return result;
}

float reduce_float8(float lanes[32]) {
    precise float result = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    return result;
}
