
// Eight queries reuse 32 keys/values. Workspace is independent of clip length.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Arena { uint data[]; } arena;
layout(push_constant) uniform PC {
    uint q_word; uint k_word; uint v_word; uint output_word;
    uint queries; uint keys; uint q_heads; uint kv_heads; uint dim;
    float scale; uint causal;
} pc;
shared float queries[8][128];
// The extra column prevents every key lane from hitting the same shared bank.
shared float stage[32][129];
shared float scores[8][32];
shared float maximum[8];
shared float normalizer[8];
shared float rescale[8];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint query_base = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * 8u;
    uint head = gl_WorkGroupID.z;
    uint q = tid / 32u;
    uint lane = tid % 32u;
    uint kv_head = head / (pc.q_heads / pc.kv_heads);
    uint query = query_base + q;
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        uint row = query_base + i / 128u;
        uint d = i % 128u;
        queries[i / 128u][d] = row < pc.queries && d < pc.dim
            ? uintBitsToFloat(arena.data[pc.q_word + (row * pc.q_heads + head) * pc.dim + d]) : 0.0;
    }
    if (tid < 8u) { maximum[tid] = -3.402823466e38; normalizer[tid] = 0.0; }
    float out_value[MAX_DIM / 32u];
    for (uint d = 0u; d < MAX_DIM / 32u; d++) out_value[d] = 0.0;
    barrier();
    for (uint start = 0u; start < pc.keys; start += 32u) {
        uint live = min(32u, pc.keys - start);
        bool visible = lane < live && query < pc.queries
            && (pc.causal == 0u || start + lane <= query + pc.keys - pc.queries);
        float score = 0.0;
        for (uint chunk = 0u; chunk < pc.dim; chunk += 128u) {
            if (pc.dim > 128u) {
                for (uint i = tid; i < 8u * 128u; i += 256u) {
                    uint row = query_base + i / 128u;
                    uint d = chunk + i % 128u;
                    queries[i / 128u][i % 128u] = row < pc.queries && d < pc.dim
                        ? uintBitsToFloat(arena.data[pc.q_word + (row * pc.q_heads + head) * pc.dim + d]) : 0.0;
                }
            }
            for (uint i = tid; i < 32u * 128u; i += 256u) {
                uint key = start + i / 128u;
                uint d = chunk + i % 128u;
                stage[i / 128u][i % 128u] = key < pc.keys && d < pc.dim
                    ? uintBitsToFloat(arena.data[pc.k_word + (key * pc.kv_heads + kv_head) * pc.dim + d]) : 0.0;
            }
            barrier();
            if (visible) {
                for (uint d = 0u; d < min(128u, pc.dim - chunk); d++) score = fma(queries[q][d], stage[lane][d], score);
            }
            barrier();
        }
        score = visible ? score * pc.scale : -3.402823466e38;
        scores[q][lane] = score;
        barrier();
        if (lane == 0u) {
            float next_max = maximum[q];
            for (uint k = 0u; k < live; k++) next_max = max(next_max, scores[q][k]);
            float old_scale = exp(maximum[q] - next_max);
            float sum = normalizer[q] * old_scale;
            for (uint k = 0u; k < live; k++) {
                float p = scores[q][k] == -3.402823466e38 ? 0.0 : exp(scores[q][k] - next_max);
                scores[q][k] = p;
                sum += p;
            }
            maximum[q] = next_max;
            normalizer[q] = sum;
            rescale[q] = old_scale;
        }
        barrier();
        for (uint chunk = 0u; chunk < pc.dim; chunk += 128u) {
            for (uint i = tid; i < 32u * 128u; i += 256u) {
                uint key = start + i / 128u;
                uint d = chunk + i % 128u;
                stage[i / 128u][i % 128u] = key < pc.keys && d < pc.dim
                    ? uintBitsToFloat(arena.data[pc.v_word + (key * pc.kv_heads + kv_head) * pc.dim + d]) : 0.0;
            }
            barrier();
            for (uint d = 0u; d < 4u; d++) {
                uint index = chunk / 32u + d;
                out_value[index] *= rescale[q];
                for (uint k = 0u; k < live; k++) out_value[index] = fma(scores[q][k], stage[k][lane + 32u * d], out_value[index]);
            }
            barrier();
        }
    }
    if (query < pc.queries) {
        for (uint d = 0u; d < MAX_DIM / 32u; d++) {
            uint dimension = lane + 32u * d;
            if (dimension < pc.dim) arena.data[pc.output_word + (query * pc.q_heads + head) * pc.dim + dimension]
                = floatBitsToUint(out_value[d] / normalizer[q]);
        }
    }
}
