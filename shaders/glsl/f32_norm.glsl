float sqrt_f32(float value) {
    if (value == 0.0 || isinf(value)) return sqrt(value);
    precise float root = sqrt(value);
    precise float residual = fma(-root, root, value);
    return fma(residual, 0.5 / root, root);
}

float inverse_sqrt_f32(float value) {
    return divide_finite_normal(1.0, sqrt_f32(value));
}
