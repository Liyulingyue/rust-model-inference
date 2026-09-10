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
pub(in crate::models::dots) fn torch28_exp(value: f32) -> f32 {
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
pub(in crate::models::dots) fn torch28_tanh(value: f32) -> f32 {
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
