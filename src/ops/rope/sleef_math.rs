//! SLEEF-style double-float (Df2) arithmetic + reduced-argument sin/cos.
//!
//! Used by [`super::sleef_rope`] to match libsystem_sleef.dylib bit-for-bit
//! on macOS (dots.tts model). The Df2 representation pairs an `f32` high part
//! with the rounding error from the `f32` op as the low part, giving ~48-bit
//! effective precision.

#[derive(Clone, Copy)]
struct Df2 {
    hi: f32,
    lo: f32,
}

#[inline(always)]
fn df_add2(left: f32, right: f32) -> Df2 {
    let hi = left + right;
    let virtual_right = hi - left;
    Df2 {
        hi,
        lo: (left - (hi - virtual_right)) + (right - virtual_right),
    }
}

#[inline(always)]
fn df_add_plain(left: f32, right: f32) -> Df2 {
    let hi = left + right;
    Df2 {
        hi,
        lo: (left - hi) + right,
    }
}

#[inline(always)]
fn df_add(left: Df2, right: f32) -> Df2 {
    let hi = left.hi + right;
    Df2 {
        hi,
        lo: ((left.hi - hi) + right) + left.lo,
    }
}

#[inline(always)]
fn df_add2_scalar(left: Df2, right: f32) -> Df2 {
    let hi = left.hi + right;
    let virtual_right = hi - left.hi;
    Df2 {
        hi,
        lo: ((left.hi - (hi - virtual_right)) + (right - virtual_right)) + left.lo,
    }
}

#[inline(always)]
fn df_add_df(left: Df2, right: Df2) -> Df2 {
    let hi = left.hi + right.hi;
    let virtual_right = hi - left.hi;
    let error = (left.hi - (hi - virtual_right)) + (right.hi - virtual_right);
    Df2 {
        hi,
        lo: error + (left.lo + right.lo),
    }
}

#[inline(always)]
fn df_add_scalar(left: f32, right: Df2) -> Df2 {
    let hi = left + right.hi;
    Df2 {
        hi,
        lo: ((left - hi) + right.hi) + right.lo,
    }
}

#[inline(always)]
fn df_normalize(value: Df2) -> Df2 {
    let hi = value.hi + value.lo;
    Df2 {
        hi,
        lo: (value.hi - hi) + value.lo,
    }
}

#[inline(always)]
fn df_mul_scalar(left: f32, right: f32) -> Df2 {
    let hi = left * right;
    Df2 {
        hi,
        lo: left.mul_add(right, -hi),
    }
}

#[inline(always)]
fn df_mul_df_scalar(left: Df2, right: f32) -> Df2 {
    let hi = left.hi * right;
    Df2 {
        hi,
        lo: left.lo.mul_add(right, left.hi.mul_add(right, -hi)),
    }
}

#[inline(always)]
fn df_mul(left: Df2, right: Df2) -> Df2 {
    let hi = left.hi * right.hi;
    let lo = left.hi.mul_add(
        right.lo,
        left.lo.mul_add(right.hi, left.hi.mul_add(right.hi, -hi)),
    );
    Df2 { hi, lo }
}

#[inline(always)]
fn df_square(value: Df2) -> Df2 {
    let hi = value.hi * value.hi;
    let lo = (value.hi + value.hi).mul_add(value.lo, value.hi.mul_add(value.hi, -hi));
    Df2 { hi, lo }
}

#[inline(always)]
fn df_mul_to_scalar(left: Df2, right: Df2) -> f32 {
    left.hi
        .mul_add(right.hi, right.lo.mul_add(left.hi, left.lo * right.hi))
}

#[inline(always)]
fn mulsign(value: f32, sign_source: f32) -> f32 {
    f32::from_bits(value.to_bits() ^ (sign_source.to_bits() & 0x8000_0000))
}

#[inline(always)]
fn rempisubf(value: f32) -> (f32, i32) {
    let rounded4 = (value * 4.0).round_ties_even();
    let quadrant = (rounded4 - value.round_ties_even() * 4.0) as i32;
    (value - rounded4 * 0.25, quadrant)
}

#[inline(always)]
fn rempif(value: f32) -> (Df2, i32) {
    // Dots sessions are capped at 2048 positions, so the first rempi table
    // quartet is the only range needed here (ilogb(value) - 25 <= 0).
    const REMPI: [f32; 4] = [
        0.159_154_892,
        5.112_411_827e-8,
        3.626_141_271e-15,
        -2.036_222_915e-22,
    ];
    let mut x = df_mul_scalar(value, REMPI[0]);
    let (fraction, mut quadrant) = rempisubf(x.hi);
    x.hi = fraction;
    x = df_normalize(x);
    let y = df_mul_scalar(value, REMPI[1]);
    x = df_add_df(x, y);
    let (fraction, second_quadrant) = rempisubf(x.hi);
    quadrant += second_quadrant;
    x.hi = fraction;
    x = df_normalize(x);
    let y = df_mul_df_scalar(
        Df2 {
            hi: REMPI[2],
            lo: REMPI[3],
        },
        value,
    );
    x = df_add_df(x, y);
    x = df_normalize(x);
    x = df_mul(
        x,
        Df2 {
            hi: 3.141_592_741_012_573_2_f32 * 2.0,
            lo: -8.742_277_657_347_586e-8_f32 * 2.0,
        },
    );
    if value.abs() < 0.7 {
        (Df2 { hi: value, lo: 0.0 }, 0)
    } else {
        (x, quadrant)
    }
}

#[inline(always)]
pub(crate) fn sleef_sin_mode(theta: f32, force_large_range: bool) -> f32 {
    let (mut reduced, quadrant) = if !force_large_range && theta.abs() < 125.0 {
        let quadrant = (theta * 0.318_309_873_342_392_6_f32).round_ties_even() as i32;
        let q = quadrant as f32;
        let v = q.mul_add(-3.141_479_492_187_5, theta);
        let s = df_add2(v, q * -0.000_113_159_418_106_079_1_f32);
        (df_add(s, q * -1.984_187_258_941_005_9e-9_f32), quadrant)
    } else {
        let (mut value, base_quadrant) = rempif(theta);
        let quadrant = ((base_quadrant & 3) * 2 + if value.hi > 0.0 { 2 } else { 1 }) >> 2;
        if base_quadrant & 1 != 0 {
            value = df_add_df(
                value,
                Df2 {
                    hi: mulsign(3.141_592_741_012_573_2_f32 * -0.5, value.hi),
                    lo: mulsign(-8.742_277_657_347_586e-8_f32 * -0.5, value.hi),
                },
            );
        }
        value = df_normalize(value);
        (value, quadrant)
    };
    let square = df_square(reduced);
    let mut u = 2.608_315_980_978_659_4e-6_f32;
    u = u.mul_add(square.hi, -0.000_198_106_907_191_686_33);
    u = u.mul_add(square.hi, 0.008_333_078_585_565_09);
    let inner = df_add_plain(-0.166_666_597_127_914_43, u * square.hi);
    let polynomial = df_add_scalar(1.0, df_mul(inner, square));
    let mut result = df_mul_to_scalar(reduced, polynomial);
    if quadrant & 1 != 0 {
        result = f32::from_bits(result.to_bits() ^ 0x8000_0000);
    }
    if theta == 0.0 && theta.is_sign_negative() {
        -0.0
    } else {
        result
    }
}

#[inline(always)]
pub(crate) fn sleef_cos_mode(theta: f32, force_large_range: bool) -> f32 {
    let (reduced, quadrant) = if !force_large_range && theta.abs() < 125.0 {
        let rounded = theta
            .mul_add(0.318_309_873_342_392_6_f32, -0.5)
            .round_ties_even() as i32;
        let quadrant = 1 + 2 * rounded;
        let q = quadrant as f32;
        let mut reduced = df_add2(theta, q * (-3.141_479_492_187_5_f32 * 0.5));
        reduced = df_add2_scalar(reduced, q * (-0.000_113_159_418_106_079_1_f32 * 0.5));
        reduced = df_add2_scalar(reduced, q * (-1.984_187_258_941_005_9e-9_f32 * 0.5));
        (reduced, quadrant)
    } else {
        let (mut value, base_quadrant) = rempif(theta);
        let quadrant = ((base_quadrant & 3) * 2 + if value.hi > 0.0 { 8 } else { 7 }) >> 1;
        if base_quadrant & 1 == 0 {
            let sign = if value.hi > 0.0 { 1.0 } else { -1.0 };
            value = df_add_df(
                value,
                Df2 {
                    hi: mulsign(3.141_592_741_012_573_2_f32 * -0.5, sign),
                    lo: mulsign(-8.742_277_657_347_586e-8_f32 * -0.5, sign),
                },
            );
        }
        value = df_normalize(value);
        (value, quadrant)
    };
    let square = df_square(reduced);
    let mut u = 2.608_315_980_978_659_4e-6_f32;
    u = u.mul_add(square.hi, -0.000_198_106_907_191_686_33);
    u = u.mul_add(square.hi, 0.008_333_078_585_565_09);
    let inner = df_add_plain(-0.166_666_597_127_914_43, u * square.hi);
    let polynomial = df_add_scalar(1.0, df_mul(inner, square));
    let mut result = df_mul_to_scalar(reduced, polynomial);
    if quadrant & 2 == 0 {
        result = f32::from_bits(result.to_bits() ^ 0x8000_0000);
    }
    result
}

#[inline]
pub(crate) fn rope_sin_cos_sleef(theta: f32) -> (f32, f32) {
    (sleef_cos_mode(theta, false), sleef_sin_mode(theta, false))
}
