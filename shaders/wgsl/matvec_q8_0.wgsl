@group(0) @binding(0) var<storage, read>        weight: array<u32>;
@group(0) @binding(1) var<storage, read>        input_q8: array<u32>;
@group(0) @binding(2) var<storage, read>        input_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>;

fn decode_f16(f16_bits: u32) -> f32 {
    let sign = (f16_bits >> 15u) & 1u;
    var exponent = i32((f16_bits >> 10u) & 0x1Fu);
    let mantissa = f16_bits & 0x3FFu;

    var significand: f32;
    var is_zero = false;
    var is_inf = false;

    if (exponent == 0) {
        if (mantissa == 0u) {
            is_zero = true;
        } else {
            significand = f32(mantissa) / 1024.0;
            significand = significand * exp2(-14.0);
        }
    } else {
        if (exponent == 31) {
            if (mantissa == 0u) {
                is_inf = true;
            } else {
                return 0.0;
            }
        } else {
            significand = 1.0 + f32(mantissa) / 1024.0;
            exponent = exponent - 15;
        }
    }

    if (is_zero) {
        if (sign != 0u) { return -0.0; }
        return 0.0;
    }
    if (is_inf) {
        if (sign != 0u) { return -1.0; }
        return 1.0;
    }

    var result = significand * exp2(f32(exponent));
    if (sign != 0u) {
        result = -result;
    }
    return result;
}

fn i8_to_i32(byte: u32) -> i32 {
    let bits = byte & 0xFFu;
    if (bits >= 128u) {
        return i32(bits) - 256;
    }
    return i32(bits);
}

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let out_idx = gid.x;
    let N = dims.y;

    if (out_idx >= N) { return; }

    let blocks_per_row = dims.z;
    let row_stride = dims.w;
    let row_off = out_idx * row_stride;

    var sum = 0.0;

    for (var block = 0u; block < blocks_per_row; block = block + 1u) {
        let weight_off = row_off + block * 34u;
        let input_off = block * 32u;

        let scale_word = weight[weight_off >> 2u];
        let scale_byte_off = weight_off & 3u;
        var scale_bits: u32;
        if (scale_byte_off == 0u) {
            scale_bits = scale_word & 0xFFFFu;
        } else {
            scale_bits = (scale_word >> 16u) & 0xFFFFu;
        }
        let weight_scale = decode_f16(scale_bits);

        let in_scale = input_scales[block];

        var dot = 0;
        for (var i = 0u; i < 32u; i = i + 1u) {
            let w_idx = weight_off + 2u + i;
            let w_word = weight[w_idx >> 2u];
            let w_byte_off = w_idx & 3u;
            let w_byte = (w_word >> (w_byte_off * 8u)) & 0xFFu;
            let w_val = i8_to_i32(w_byte);

            let in_idx = input_off + i;
            let in_word = input_q8[in_idx >> 2u];
            let in_byte_off = in_idx & 3u;
            let in_byte = (in_word >> (in_byte_off * 8u)) & 0xFFu;
            let in_val = i8_to_i32(in_byte);

            dot = dot + w_val * in_val;
        }

        sum = sum + weight_scale * in_scale * f32(dot);
    }

    output[out_idx] = sum;
}
