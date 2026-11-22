float q4_0_accumulate(float sum, int dot, float weight_scale, float activation_scale, uint mode) {
    if (mode == 8u) {
        precise float scale = weight_scale * activation_scale;
        return fma(scale, float(dot), sum);
    }
    precise float product = float(dot) * weight_scale;
    product = product * activation_scale;
    precise float result = sum + product;
    return result;
}
