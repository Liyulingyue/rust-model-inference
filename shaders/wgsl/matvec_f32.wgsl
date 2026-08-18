@group(0) @binding(0) var<storage, read>       weight: array<u32>;
@group(0) @binding(1) var<storage, read>       input_q8: array<u32>;
@group(0) @binding(2) var<storage, read>        input_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             dims: vec4<u32>;

@compute @workgroup_size(1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let out_idx = gid.x;
    if (out_idx >= dims.y) { return; }
    output[out_idx] = f32(out_idx);
}
