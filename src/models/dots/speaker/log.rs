// Scalar translation of pinned SLEEF 5a1d179d `xlogf_u1`. Torch 2.8's ARM
// unary kernel runs four independent F32 lanes; this keeps the same
// double-float primitives and FMA order without runtime FFI or a dependency.

#[derive(Clone, Copy)]
struct DoubleFloat {
    high: f32,
    low: f32,
}

#[inline(always)]
fn add2(left: f32, right: f32) -> DoubleFloat {
    let high = left + right;
    let carry = high - left;
    DoubleFloat {
        high,
        low: (left - (high - carry)) + (right - carry),
    }
}

#[inline(always)]
fn add(left: DoubleFloat, right: DoubleFloat) -> DoubleFloat {
    let high = left.high + right.high;
    DoubleFloat {
        high,
        low: (((left.high - high) + right.high) + left.low) + right.low,
    }
}

#[inline(always)]
fn add_scalar(left: DoubleFloat, right: f32) -> DoubleFloat {
    let high = left.high + right;
    DoubleFloat {
        high,
        low: ((left.high - high) + right) + left.low,
    }
}

#[inline(always)]
fn scale(value: DoubleFloat, factor: f32) -> DoubleFloat {
    DoubleFloat {
        high: value.high * factor,
        low: value.low * factor,
    }
}

#[inline(always)]
fn multiply_scalar(value: DoubleFloat, factor: f32) -> DoubleFloat {
    let high = value.high * factor;
    DoubleFloat {
        high,
        low: value.low.mul_add(factor, value.high.mul_add(factor, -high)),
    }
}

#[inline(always)]
fn divide(numerator: DoubleFloat, denominator: DoubleFloat) -> DoubleFloat {
    let reciprocal = 1.0 / denominator.high;
    let high = numerator.high * reciprocal;
    let numerator_error = reciprocal.mul_add(numerator.high, -high);
    let reciprocal_error =
        (-denominator.low).mul_add(reciprocal, (-denominator.high).mul_add(reciprocal, 1.0));
    DoubleFloat {
        high,
        low: high.mul_add(
            reciprocal_error,
            numerator.low.mul_add(reciprocal, numerator_error),
        ),
    }
}

pub(super) fn torch28_log(mut value: f32) -> f32 {
    let original = value;
    let subnormal = value < f32::MIN_POSITIVE;
    if subnormal {
        value *= f32::from_bits(0x5f80_0000);
    }

    let mut exponent =
        (((value * f32::from_bits(0x3faa_aaab)).to_bits() >> 23) & 0xff) as i32 - 127;
    if subnormal {
        exponent -= 64;
    }
    let mantissa = f32::from_bits((value.to_bits() as i32).wrapping_add((-exponent) << 23) as u32);

    let mut result = multiply_scalar(
        DoubleFloat {
            high: f32::from_bits(0x3f31_7218),
            low: f32::from_bits(0xb102_e308),
        },
        exponent as f32,
    );
    let ratio = divide(add2(-1.0, mantissa), add2(1.0, mantissa));
    let ratio_squared = ratio.high * ratio.high;
    let mut polynomial = f32::from_bits(0x3e9a_ff5c);
    polynomial = polynomial.mul_add(ratio_squared, f32::from_bits(0x3ecc_99ca));
    polynomial = polynomial.mul_add(ratio_squared, f32::from_bits(0x3f2a_aada));
    result = add(result, scale(ratio, 2.0));
    result = add_scalar(result, (ratio_squared * ratio.high) * polynomial);
    let logarithm = result.high + result.low;

    if value.is_infinite() {
        f32::INFINITY
    } else if value < 0.0 || value.is_nan() {
        f32::NAN
    } else if original == 0.0 {
        f32::NEG_INFINITY
    } else {
        logarithm
    }
}
