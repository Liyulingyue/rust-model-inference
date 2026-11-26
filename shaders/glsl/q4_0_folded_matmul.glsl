// A folded projection retains eight independent F32 sums across its K dimension.
// Summing each block to one float first loses precision on long Bonsai continuations.
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Arena { uint data[]; } arena;
layout(std430, binding = 1) readonly buffer W0 { uint data[]; } w0;
layout(std430, binding = 2) readonly buffer W1 { uint data[]; } w1;
layout(std430, binding = 3) readonly buffer W2 { uint data[]; } w2;
layout(push_constant) uniform PC {
    uint q8_word; uint scales_word; uint n_in; uint blocks_per_row;
    uint output0_word; uint rows0; uint output0_stride;
    uint output1_word; uint rows1; uint output1_stride;
    uint output2_word; uint rows2; uint output2_stride;
    uint groups; uint input_sum_word; uint input_stride; uint q8_stride;
    uint scales_stride; uint sums_stride; uint q8k_stride; uint q8k_scales_stride;
    uint token_rows;
} pc;

const uint token_tile = 32u;
shared float partials[64][33];

uint weight_word(uint slot, uint word) {
    if (slot == 0u) return w0.data[word];
    if (slot == 1u) return w1.data[word];
    return w2.data[word];
}

int dot4(uint weights, uint inputs) {
#ifdef PACKED_DOT
    return dotPacked4x8EXT(int(weights), int(inputs), 0u);
#else
    int sum = 0;
    for (int byte = 0; byte < 4; ++byte) {
        sum += bitfieldExtract(int(weights), 8 * byte, 8) * bitfieldExtract(int(inputs), 8 * byte, 8);
    }
    return sum;
#endif
}

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint stream = lane % 8u;
    uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uint row = group * 8u + lane / 8u;
    uint slot = gl_WorkGroupID.z % pc.groups;
    uint token = gl_WorkGroupID.z / pc.groups * token_tile;
    uint rows = slot == 0u ? pc.rows0 : (slot == 1u ? pc.rows1 : pc.rows2);
    uint live = min(token_tile, pc.token_rows - token);
    precise float acc[token_tile];
    for (uint t = 0u; t < live; ++t) acc[t] = 0.0;
    if (row < rows) {
        for (uint block = 0u; block < pc.blocks_per_row; ++block) {
            // Q4_0 packs a half scale and sixteen bytes into nine halfwords.
            uint halfword = (row * pc.blocks_per_row + block) * 9u;
            uint scale_word = weight_word(slot, halfword / 2u);
            uint scale_bits = (scale_word >> (16u * (halfword & 1u))) & 0xffffu;
            float w = unpackHalf2x16(scale_bits).x;
            uint body = halfword + 1u;
            uint word = body / 2u + stream % 4u;
            uint packed = weight_word(slot, word);
            if ((body & 1u) != 0u) packed = (packed >> 16u) | (weight_word(slot, word + 1u) << 16u);
            uint nibbles = (packed >> (4u * (stream / 4u))) & 0x0f0f0f0fu;
            uint weights = (nibbles + 0x78787878u) ^ 0x80808080u;
            for (uint t = 0u; t < live; ++t) {
                uint input_word = pc.q8_word + (token + t) * pc.q8_stride + block * 8u + stream;
                uint scale_index = pc.scales_word + (token + t) * pc.scales_stride + block;
                precise float scale = w * uintBitsToFloat(arena.data[scale_index]);
                acc[t] = fma(float(dot4(weights, arena.data[input_word])), scale, acc[t]);
            }
        }
    }
    for (uint t = 0u; t < live; ++t) partials[lane][t] = acc[t];
    barrier();
    if (row >= rows) return;
    uint out_word = slot == 0u ? pc.output0_word : (slot == 1u ? pc.output1_word : pc.output2_word);
    uint stride = slot == 0u ? pc.output0_stride : (slot == 1u ? pc.output1_stride : pc.output2_stride);
    uint k = lane - stream;
    for (uint t = stream; t < live; t += 8u) {
        precise float lo = (partials[k][t] + partials[k+4u][t]) + (partials[k+1u][t] + partials[k+5u][t]);
        precise float hi = (partials[k+2u][t] + partials[k+6u][t]) + (partials[k+3u][t] + partials[k+7u][t]);
        arena.data[out_word + (token + t) * stride + row] = floatBitsToUint(lo + hi);
    }
}
