// E:\Codes\rust-model-inference\references\wgpu-llm\crates\core\shaders\matvec.wgsl
// Matrix-vector multiply: y[N] = x[K] · W[N×K]^T  (decode-mode M=1)
//
// Each workgroup computes ROWS_PER_WG consecutive output elements via
// parallel dot-product reduction over K.  x-vector loads are amortized
// across all rows within the workgroup.  All 4 rows are reduced in
// parallel using vec4<f32> shared memory (8 barriers total, same as
// single-output).
//
// Dispatched as ceil(N / ROWS_PER_WG) workgroups of WG_SIZE threads.
//
// Binding 1 (weight matrix W) and the `read_b()` accessor are injected
// by the Rust shader loader, reusing the same injection as GEMM.
//
// The dims uniform reuses the GEMM layout: [M, N, K, trans_b].
// This shader reads N from dims.y and K from dims.z.

const WG_SIZE: u32 = 256u;
const ROWS_PER_WG: u32 = 4u;

@group(0) @binding(0) var<storage, read>       x: array<f32>;    // input [K]
@group(0) @binding(1) var<storage, read>       w: array<f32>;    // weight [N*K]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;    // output [N]
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;     // (M, N, K, trans_b)

var<workgroup> sdata: array<vec4<f32>, 256>;   // 4 KB — parallel 4-row reduction

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let N = dims.y;
    let K = dims.z;
    let base_row = wid.x * ROWS_PER_WG;
    let tid = lid.x;

    // Weight row bases (pre-computed to avoid repeated multiply in loop)
    let base0 = base_row * K;
    let base1 = (base_row + 1u) * K;
    let base2 = (base_row + 2u) * K;
    let base3 = (base_row + 3u) * K;

    // Accumulate 4 partial dot-products in a vec4 register.
    // x-vector is loaded once per iteration, reused for all 4 rows.
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (base_row + 3u < N) {
        // Fast path: all 4 rows valid — no per-row bounds checks
        var i = tid * 4u;
        loop {
            if (i >= K) { break; }
            let x_v = vec4<f32>(x[i], x[i+1u], x[i+2u], x[i+3u]);
            acc.x += dot(x_v, w[base0 + i..base0 + i + 4]);
            acc.y += dot(x_v, w[base1 + i..base1 + i + 4]);
            acc.z += dot(x_v, w[base2 + i..base2 + i + 4]);
            acc.w += dot(x_v, w[base3 + i..base3 + i + 4]);
            i += WG_SIZE * 4u;
        }
    } else {
        // Tail path: last workgroup may have fewer than 4 valid rows
        var i = tid * 4u;
        loop {
            if (i >= K) { break; }
            let x_v = vec4<f32>(x[i], x[i+1u], x[i+2u], x[i+3u]);
            if (base_row < N)      { acc.x += dot(x_v, w[base0 + i..base0 + i + 4]); }
            if (base_row + 1u < N) { acc.y += dot(x_v, w[base1 + i..base1 + i + 4]); }
            if (base_row + 2u < N) { acc.z += dot(x_v, w[base2 + i..base2 + i + 4]); }
            if (base_row + 3u < N) { acc.w += dot(x_v, w[base3 + i..base3 + i + 4]); }
            i += WG_SIZE * 4u;
        }
    }

    // Parallel tree reduction of all 4 rows simultaneously (vec4 add)
    sdata[tid] = acc;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
    }

    // Store results — only thread 0, only valid rows
    if (tid == 0u) {
        if (base_row < N)      { y[base_row]      = sdata[0].x; }
        if (base_row + 1u < N) { y[base_row + 1u] = sdata[0].y; }
        if (base_row + 2u < N) { y[base_row + 2u] = sdata[0].z; }
        if (base_row + 3u < N) { y[base_row + 3u] = sdata[0].w; }
    }
}
