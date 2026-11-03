// Arm expf range reduction and coefficients (musl src/math/expf.c and exp2f_data.c).
// Copyright (c) 2017-2018, Arm Limited. SPDX-License-Identifier: MIT
// Keep the FMA evaluation used by the Linux CPU Oracle: a 1 ULP expf change
// can cross a recurrent activation quantization boundary.
float exp_f64(float x) {
    if (isnan(x)) return x;
    if (x > 88.72283935546875) return 1.0 / 0.0;
    if (x < -103.97208404541016) return 0.0;
    const uvec2 table[32] = uvec2[32](
        uvec2(0x00000000u, 0x3ff00000u),
        uvec2(0xd3158574u, 0x3fefd9b0u),
        uvec2(0x6cf9890fu, 0x3fefb558u),
        uvec2(0xd0125b51u, 0x3fef9301u),
        uvec2(0x3c7d517bu, 0x3fef72b8u),
        uvec2(0x3168b9aau, 0x3fef5487u),
        uvec2(0x6e756238u, 0x3fef387au),
        uvec2(0xf51fdee1u, 0x3fef1e9du),
        uvec2(0x0a31b715u, 0x3fef06feu),
        uvec2(0x373aa9cbu, 0x3feef1a7u),
        uvec2(0x4c123422u, 0x3feedea6u),
        uvec2(0x6061892du, 0x3feece08u),
        uvec2(0xd5362a27u, 0x3feebfdau),
        uvec2(0x569d4f82u, 0x3feeb42bu),
        uvec2(0xdd485429u, 0x3feeab07u),
        uvec2(0xb03a5585u, 0x3feea47eu),
        uvec2(0x667f3bcdu, 0x3feea09eu),
        uvec2(0xe8ec5f74u, 0x3fee9f75u),
        uvec2(0x73eb0187u, 0x3feea114u),
        uvec2(0x994cce13u, 0x3feea589u),
        uvec2(0x422aa0dbu, 0x3feeace5u),
        uvec2(0xb0cdc5e5u, 0x3feeb737u),
        uvec2(0x82a3f090u, 0x3feec491u),
        uvec2(0xb23e255du, 0x3feed503u),
        uvec2(0x995ad3adu, 0x3feee89fu),
        uvec2(0xf2fb5e47u, 0x3feeff76u),
        uvec2(0xdd85529cu, 0x3fef199bu),
        uvec2(0xdcef9069u, 0x3fef3720u),
        uvec2(0xdcfba487u, 0x3fef5818u),
        uvec2(0x337b9b5fu, 0x3fef7c97u),
        uvec2(0xa2a490dau, 0x3fefa4afu),
        uvec2(0x5b6e4540u, 0x3fefd076u)
    );
    precise double z = 46.166241308446828lf * double(x);
    double kd = roundEven(z);
    int k = int(kd);
    precise double r = z - kd;
    uvec2 bits = table[uint(k) & 31u];
    bits.y += uint(k) << 15u;
    double scale = packDouble2x32(bits);
    precise double polynomial = fma(1.6938359250920212e-06lf, r, 0.00023459809789509004lf);
    precise double square = r * r;
    precise double y = fma(0.021660849396613134lf, r, 1.0lf);
    y = fma(polynomial, square, y);
    return float(y * scale);
}
