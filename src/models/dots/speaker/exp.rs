// Scalar translation of pinned SLEEF 5a1d179d `xexpf`. Torch 2.8's ARM
// sigmoid kernel evaluates four independent F32 lanes with this polynomial;
// explicit `mul_add` calls preserve the AdvSIMD CONFIG=1 FMA association.

const R_LN2: f32 = 1.442_695_040_888_963_4;
const L2_UPPER: f32 = 0.693_145_751_953_125;
const L2_LOWER: f32 = 1.428_606_765_330_187e-6;

#[derive(Clone, Copy)]
struct DoubleF32 {
    high: f32,
    low: f32,
}

#[inline(always)]
fn df_add2_float(value: DoubleF32, addend: f32) -> DoubleF32 {
    let high = value.high + addend;
    let virtual_addend = high - value.high;
    let error = (value.high - (high - virtual_addend)) + (addend - virtual_addend);
    DoubleF32 {
        high,
        low: error + value.low,
    }
}

#[inline(always)]
fn df_add2(left: DoubleF32, right: DoubleF32) -> DoubleF32 {
    let high = left.high + right.high;
    let virtual_right = high - left.high;
    let error = (left.high - (high - virtual_right)) + (right.high - virtual_right);
    DoubleF32 {
        high,
        low: error + (left.low + right.low),
    }
}

#[inline(always)]
fn df_add(left: DoubleF32, right: DoubleF32) -> DoubleF32 {
    let high = left.high + right.high;
    let low = (left.high - high) + right.high;
    let low = low + left.low;
    DoubleF32 {
        high,
        low: low + right.low,
    }
}

#[inline(always)]
fn df_add_float_double(left: f32, right: DoubleF32) -> DoubleF32 {
    let high = left + right.high;
    let low = (left - high) + right.high;
    DoubleF32 {
        high,
        low: low + right.low,
    }
}

#[inline(always)]
fn df_neg(value: DoubleF32) -> DoubleF32 {
    DoubleF32 {
        high: -value.high,
        low: -value.low,
    }
}

#[inline(always)]
fn df_mul_float(value: DoubleF32, multiplier: f32) -> DoubleF32 {
    let high = value.high * multiplier;
    let high_error = value.high.mul_add(multiplier, -high);
    DoubleF32 {
        high,
        low: value.low.mul_add(multiplier, high_error),
    }
}

#[inline(always)]
fn df_mul(left: DoubleF32, right: DoubleF32) -> DoubleF32 {
    let high = left.high * right.high;
    let high_error = left.high.mul_add(right.high, -high);
    let low = left.low.mul_add(right.high, high_error);
    DoubleF32 {
        high,
        low: left.high.mul_add(right.low, low),
    }
}

#[inline(always)]
fn df_square(value: DoubleF32) -> DoubleF32 {
    let high = value.high * value.high;
    let high_error = value.high.mul_add(value.high, -high);
    DoubleF32 {
        high,
        low: (value.high + value.high).mul_add(value.low, high_error),
    }
}

#[inline(always)]
fn df_reciprocal(value: DoubleF32) -> DoubleF32 {
    let high = 1.0 / value.high;
    let correction = value.low.mul_add(-high, value.high.mul_add(-high, 1.0));
    DoubleF32 {
        high,
        low: high * correction,
    }
}

#[inline(always)]
fn df_divide(numerator: DoubleF32, denominator: DoubleF32) -> DoubleF32 {
    let reciprocal = 1.0 / denominator.high;
    let high = numerator.high * reciprocal;
    let numerator_error = reciprocal.mul_add(numerator.high, -high);
    let denominator_error = denominator
        .low
        .mul_add(-reciprocal, denominator.high.mul_add(-reciprocal, 1.0));
    let low = numerator.low.mul_add(reciprocal, numerator_error);
    DoubleF32 {
        high,
        low: high.mul_add(denominator_error, low),
    }
}

#[inline(always)]
fn pow2i(exponent: i32) -> f32 {
    f32::from_bits(exponent.wrapping_add(0x7f).wrapping_shl(23) as u32)
}

#[inline(always)]
fn ldexp2(value: f32, exponent: i32) -> f32 {
    let half = exponent >> 1;
    (value * pow2i(half)) * pow2i(exponent - half)
}

// Scalar translation of pinned SLEEF 5a1d179d `Sleef_tanhf4_u10`'s
// CONFIG=1 AdvSIMD path. Explicit `mul_add` calls preserve its FMA grouping.
#[inline(always)]
fn torch28_exp_double(value: DoubleF32) -> DoubleF32 {
    let exponent = ((value.high + value.low) * R_LN2).round_ties_even() as i32;
    let exponent_f32 = exponent as f32;
    let mut reduced = df_add2_float(value, exponent_f32 * -L2_UPPER);
    reduced = df_add2_float(reduced, exponent_f32 * -L2_LOWER);

    let mut polynomial = 0.000_198_096_02_f32;
    polynomial = polynomial.mul_add(reduced.high, 0.001_394_256_5);
    polynomial = polynomial.mul_add(reduced.high, 0.008_333_457);
    polynomial = polynomial.mul_add(reduced.high, 0.041_666_374);

    let mut result = df_add2_float(df_mul_float(reduced, polynomial), 0.166_666_66);
    result = df_add2_float(df_mul(reduced, result), 0.5);
    result = df_add2(reduced, df_mul(df_square(reduced), result));
    result = df_add_float_double(1.0, result);
    result.high = ldexp2(result.high, exponent);
    result.low = ldexp2(result.low, exponent);
    if value.high < -104.0 {
        DoubleF32 {
            high: 0.0,
            low: 0.0,
        }
    } else {
        result
    }
}

#[inline(always)]
pub(crate) fn torch28_exp(value: f32) -> f32 {
    let exponent = (value * R_LN2).round_ties_even() as i32;
    let exponent_f32 = exponent as f32;
    let mut reduced = exponent_f32.mul_add(-L2_UPPER, value);
    reduced = exponent_f32.mul_add(-L2_LOWER, reduced);

    let mut polynomial = 0.000_198_527_62_f32;
    polynomial = polynomial.mul_add(reduced, 0.001_393_043_6);
    polynomial = polynomial.mul_add(reduced, 0.008_333_361);
    polynomial = polynomial.mul_add(reduced, 0.041_666_485);
    polynomial = polynomial.mul_add(reduced, 0.166_666_67);
    polynomial = polynomial.mul_add(reduced, 0.5);

    let reduced_squared = reduced * reduced;
    let result = ldexp2(1.0 + reduced_squared.mul_add(polynomial, reduced), exponent);
    if value < -104.0 {
        0.0
    } else if value > 100.0 {
        f32::INFINITY
    } else {
        result
    }
}

#[inline(always)]
pub(in crate::models::dots) fn torch28_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + torch28_exp(-value))
}

#[inline(always)]
pub(crate) fn torch28_tanh(value: f32) -> f32 {
    let magnitude = value.abs();
    let exponential = torch28_exp_double(DoubleF32 {
        high: magnitude,
        low: 0.0,
    });
    let reciprocal = df_reciprocal(exponential);
    let numerator = df_add(exponential, df_neg(reciprocal));
    let denominator = df_add(exponential, reciprocal);
    let quotient = df_divide(numerator, denominator);
    let mut result = quotient.high + quotient.low;
    if magnitude > 8.664_34 || result.is_nan() {
        result = 1.0;
    }
    result = f32::from_bits(result.to_bits() ^ (value.to_bits() & 0x8000_0000));
    if value.is_nan() {
        f32::from_bits(0xffff_ffff)
    } else {
        result
    }
}

/// SLEEF `expm1f_u10`, using the same double-float exponential as tanh.
#[inline(always)]
pub(crate) fn torch28_expm1(value: f32) -> f32 {
    if value.to_bits() == 0x8000_0000 {
        return value;
    }
    if value > 88.722_83 {
        return f32::INFINITY;
    }
    if value < -16.635_532 {
        return -1.0;
    }
    let result = df_add2_float(
        torch28_exp_double(DoubleF32 {
            high: value,
            low: 0.0,
        }),
        -1.0,
    );
    result.high + result.low
}

/// Pinned Torch ARM `Vectorized<float>::erf` polynomial (vec128_float_neon.h).
#[inline(always)]
pub(crate) fn torch28_erf(value: f32) -> f32 {
    let t = 1.0 / 0.3275911_f32.mul_add(value.abs(), 1.0);
    let mut r = 1.061405429_f32.mul_add(t, -1.453152027);
    r = r.mul_add(t, 1.421413741);
    r = r.mul_add(t, -0.284496736);
    r = r.mul_add(t, 0.254829592);
    let negative_exp = -(-(value * value)).exp();
    let result = (t * negative_exp).mul_add(r, 1.0);
    f32::from_bits(result.to_bits() ^ (value.to_bits() & 0x8000_0000))
}

#[cfg(test)]
mod tests {
    use super::{torch28_erf, torch28_expm1};

    #[test]
    fn torch_arm_expm1_and_erf_match_reference_bits() {
        // Torch 2.9.1 CPU ARM, float32 tensors, including polynomial boundaries.
        for (input, expm1, erf) in [
            (0xc2b40000, 0xbf800000, 0xbf800000),
            (0xc1800000, 0xbf7ffffe, 0xbf800000),
            (0xc0c00000, 0xbf7f5d8d, 0xbf800000),
            (0xc0400000, 0xbf734128, 0xbf7ffe8d),
            (0xc0200000, 0xbf6afc7a, 0xbf7fe553),
            (0xbf800000, 0xbf21d2a7, 0xbf57bb3b),
            (0xbdcccccd, 0xbdc2e49a, 0xbde652fd),
            (0xb727c5ac, 0xb727c575, 0xb739ffe4),
            (0x80000000, 0x80000000, 0x80000000),
            (0x00000000, 0x00000000, 0x00000000),
            (0x3727c5ac, 0x3727c5e3, 0x3739ffe4),
            (0x3dcccccd, 0x3dd763da, 0x3de652fd),
            (0x3f800000, 0x3fdbf0a9, 0x3f57bb3b),
            (0x40200000, 0x4132eb7f, 0x3f7fe553),
            (0x40400000, 0x4198af2e, 0x3f7ffe8d),
            (0x40c00000, 0x43c936e3, 0x3f800000),
            (0x41800000, 0x4b07975e, 0x3f800000),
            (0x42b00000, 0x7ef882b7, 0x3f800000),
            (0x42b20000, 0x7f800000, 0x3f800000),
        ] {
            let value = f32::from_bits(input);
            assert_eq!(torch28_expm1(value).to_bits(), expm1, "expm1({value})");
            assert_eq!(torch28_erf(value).to_bits(), erf, "erf({value})");
        }
    }
}
