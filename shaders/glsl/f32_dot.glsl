float reduce_f32_dot(float values[16], uint lanes) {
    if (lanes == 16u) {
        precise float a = (values[0] + values[8]) + (values[4] + values[12]);
        precise float b = (values[1] + values[9]) + (values[5] + values[13]);
        precise float c = (values[2] + values[10]) + (values[6] + values[14]);
        precise float d = (values[3] + values[11]) + (values[7] + values[15]);
        precise float result = (a + b) + (c + d);
        return result;
    }
    precise float result = ((values[0] + values[4]) + (values[1] + values[5]))
        + ((values[2] + values[6]) + (values[3] + values[7]));
    return result;
}
