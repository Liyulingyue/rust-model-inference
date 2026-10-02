//! The GPU twin of `run_block`.
//!
//! `run_block` in `dit.rs` is written for the CPU: it keeps one row of
//! activations live, normalises it, projects it, and moves on, so the weight
//! matrix streams past exactly once per step. On a GPU that becomes 1536 x 30
//! x 4 = 184,320 dispatches at a measured 128 us of submit-and-fence each, and
//! examples/gpu_ceiling puts the projection cost at 88 s per step against 81 s
//! for the same arithmetic on 20 CPU cores.
//!
//! So when the backend is the GPU the same block is evaluated in a different
//! order: normalise and modulate every row into a staging buffer first, then
//! issue one dispatch per projection covering the whole sequence. 120 dispatches
//! per step, and each weight matrix streams once.
//!
//! Element-wise work -- rms_norm, the AdaLN scale, silu, the residual add --
//! stays on the CPU either way. It is a small fraction of the arithmetic and
//! keeping it there means the two paths share it verbatim, so they cannot drift
//! apart numerically except where the projection itself rounds differently.
//!
//! The two paths are selected by `ops::gpu_matmul_active()` and never both run,
//! so CPU behaviour is bit-for-bit what it was before this existed.
use std::time::Instant;

use crate::vulkan::ops::ArenaRegion;
use crate::vulkan::{VulkanContext, VulkanError};

use super::dit::{
    add_modulated_residual, attention_into, rotate_interleaved_inplace, scale_modulated_branch,
    split_adaln_modulation, AdaLnModulation, BlockWeights, FFN_WIDTH, HEADS, HIDDEN,
    QK_RMS_EPSILON, QKV_WIDTH, RMS_EPSILON, ROPE_HEAD_WIDTH, TIME_WIDTH,
};
use crate::ops::{rms_norm, rms_norm_inplace};
use super::dit_gpu::{DitGpuSession, Projection};
use super::linear_into_ggml;
use super::Q8Scratch;
use crate::core::tensor::TensorSource;
use crate::core::thread_pool::ComputePool;

/// A `Result` that can be either a CPU or a GPU failure, so the caller can fall
/// back to the CPU block when the device rejects a shape or breaks mid-render.
type BlockOutcome = Result<(), String>;

pub(crate) struct Timings {
    pub modulation: std::time::Duration,
    pub rms_norm: std::time::Duration,
    pub scale_mod: std::time::Duration,
    pub linear_qkv: std::time::Duration,
    pub rope: std::time::Duration,
    pub attention: std::time::Duration,
    pub linear_out: std::time::Duration,
    pub linear_ffn: std::time::Duration,
    pub residual: std::time::Duration,
    pub block: std::time::Duration,
}

impl Default for Timings {
    fn default() -> Self {
        Self {
            modulation: std::time::Duration::ZERO,
            rms_norm: std::time::Duration::ZERO,
            scale_mod: std::time::Duration::ZERO,
            linear_qkv: std::time::Duration::ZERO,
            rope: std::time::Duration::ZERO,
            attention: std::time::Duration::ZERO,
            linear_out: std::time::Duration::ZERO,
            linear_ffn: std::time::Duration::ZERO,
            residual: std::time::Duration::ZERO,
            block: std::time::Duration::ZERO,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_block_gpu(
    context: &'static VulkanContext,
    session: &mut DitGpuSession,
    source: &dyn TensorSource,
    block: &BlockWeights,
    layer: usize,
    rows: usize,
    tokens: &mut [f32],
    rope: &[f32],
    time: Option<&[f32]>,
    qkv: &mut [f32],
    attention: &mut [f32],
    ffn: &mut [f32],
    scores: &mut [f32],
    modulation: &mut [f32],
    q8: &mut Q8Scratch,
    pool: &ComputePool,
    timings: &mut Timings,
) -> BlockOutcome {
    let hidden_len = rows * HIDDEN;
    let qkv_len = rows * QKV_WIDTH;
    let ffn_len = rows * FFN_WIDTH;
    let block_start = Instant::now();

    // --- AdaLN, same code as the CPU path -----------------------------------
    let t = Instant::now();
    let modulations: Option<AdaLnModulation<'_>> = if let Some(weights) = &block.modulation {
        let time = time.ok_or("Missing Z-Image AdaLN input")?;
        linear_into_ggml(
            source,
            &weights.matrix,
            TIME_WIDTH,
            HIDDEN * 4,
            time,
            &mut modulation[..HIDDEN * 4],
            q8,
            pool,
        )?;
        for (value, bias) in modulation[..HIDDEN * 4].iter_mut().zip(&weights.bias) {
            *value += *bias;
        }
        Some(split_adaln_modulation(&modulation[..HIDDEN * 4], HIDDEN)?)
    } else {
        if time.is_some() {
            return Err("Unexpected Z-Image AdaLN input".into());
        }
        None
    };
    timings.modulation += t.elapsed();

    // --- attention QKV projection, all rows in one dispatch -------------------
    // Stage the normalised, modulated activations for every row first: the
    // projection cannot start until the last row is ready, which is precisely
    // what the row-at-a-time loop was paying a dispatch for each time.
    let t = Instant::now();
    for row in 0..rows {
        let token = &tokens[row * HIDDEN..(row + 1) * HIDDEN];
        let normalized = &mut attention[row * HIDDEN..(row + 1) * HIDDEN];
        rms_norm(token, &block.attention_norm1, normalized, RMS_EPSILON);
        if let Some(values) = modulations {
            scale_modulated_branch(normalized, Some(values.scale_msa))?;
        }
    }
    timings.rms_norm += t.elapsed();

    let layout = *session.layout();
    let t = Instant::now();
    session.project(layer, Projection::Qkv, layout.x, &attention[..hidden_len], layout.out).map_err(|e| e.to_string())?;
    qkv[..qkv_len].copy_from_slice(session.readback(qkv_len));
    timings.linear_qkv += t.elapsed();

    // --- RoPE on the Q and K halves ------------------------------------------
    let t = Instant::now();
    for row in 0..rows {
        let rotation = &rope[row * ROPE_HEAD_WIDTH..(row + 1) * ROPE_HEAD_WIDTH];
        let row_qkv = &mut qkv[row * QKV_WIDTH..(row + 1) * QKV_WIDTH];
        for head in 0..HEADS {
            let start = head * ROPE_HEAD_WIDTH;
            let query = &mut row_qkv[start..start + ROPE_HEAD_WIDTH];
            rms_norm_inplace(query, &block.q_norm, QK_RMS_EPSILON);
            rotate_interleaved_inplace(query, rotation)?;
            let key_start = HIDDEN + start;
            let key = &mut row_qkv[key_start..key_start + ROPE_HEAD_WIDTH];
            rms_norm_inplace(key, &block.k_norm, QK_RMS_EPSILON);
            rotate_interleaved_inplace(key, rotation)?;
        }
    }
    timings.rope += t.elapsed();

    // --- attention ----------------------------------------------------------
    let t = Instant::now();
    attention_into(
        &qkv[..qkv_len],
        rows,
        HEADS,
        ROPE_HEAD_WIDTH,
        scores,
        &mut attention[..hidden_len],
    )?;
    timings.attention += t.elapsed();

    // --- output projection and residual --------------------------------------
    let t = Instant::now();
    session.project(
        layer,
        Projection::Out,
        layout.x,
        &attention[..hidden_len],
        layout.normed,
    ).map_err(|e| e.to_string())?;
    let projected = session.readback(hidden_len);
    // The residual is element-wise, so it stays on the CPU over the same buffer
    // the CPU path would use.
    for row in 0..rows {
        let out = &mut qkv[row * HIDDEN..(row + 1) * HIDDEN];
        out.copy_from_slice(&projected[row * HIDDEN..(row + 1) * HIDDEN]);
        rms_norm_inplace(out, &block.attention_norm2, RMS_EPSILON);
        add_modulated_residual(
            &mut tokens[row * HIDDEN..(row + 1) * HIDDEN],
            out,
            modulations.map(|values| values.gate_msa),
        )?;
    }
    timings.linear_out += t.elapsed();
    timings.residual += t.elapsed();

    // --- FFN -----------------------------------------------------------------
    // w1 and w3 share their input, so they are one fused dispatch writing both
    // halves; the shader's multi-output path takes them as two regions.
    let t = Instant::now();
    for row in 0..rows {
        let token = &tokens[row * HIDDEN..(row + 1) * HIDDEN];
        let normalized = &mut attention[row * HIDDEN..(row + 1) * HIDDEN];
        rms_norm(token, &block.ffn_norm1, normalized, RMS_EPSILON);
        if let Some(values) = modulations {
            scale_modulated_branch(normalized, Some(values.scale_mlp))?;
        }
    }
    timings.rms_norm += t.elapsed();

    let t = Instant::now();
    // gate and up share their input, so the normalised activations are uploaded
    // once and both projections read it. The activation is fused on the host so
    // the product never round-trips through the arena.
    session.project(layer, Projection::W1, layout.x, &attention[..hidden_len], layout.gate)
        .map_err(|e| e.to_string())?;
    //
    // `project` overwrites the readback buffer, so the gate half has to be
    // parked in `scratch` before w3 runs; `std::mem::take` drops the borrow so
    // the next call can take `&mut session` again.
    if session.scratch.len() < ffn_len {
        session.scratch.resize(ffn_len, 0.0);
    }
    let gate = session.readback(ffn_len).to_vec();
    session.scratch[..ffn_len].copy_from_slice(&gate);
    session.project(layer, Projection::W3, layout.x, &attention[..hidden_len], layout.up)
        .map_err(|e| e.to_string())?;
    let mut activated = std::mem::take(&mut session.scratch);
    for index in 0..ffn_len {
        let gate = activated[index];
        activated[index] = (gate / (1.0 + (-gate).exp())) * session.readback(ffn_len)[index];
    }
    timings.linear_ffn += t.elapsed();

    // w2 maps FFN_WIDTH back down to HIDDEN.
    let t = Instant::now();
    let activated = std::mem::take(&mut session.scratch);
    let result = session.project(
        layer,
        Projection::W2,
        layout.gate,
        &activated[..ffn_len],
        layout.out,
    );
    session.scratch = activated;
    result.map_err(|e| e.to_string())?;
    let down = session.readback(hidden_len);
    for row in 0..rows {
        let out = &mut qkv[row * HIDDEN..(row + 1) * HIDDEN];
        out.copy_from_slice(&down[row * HIDDEN..(row + 1) * HIDDEN]);
        rms_norm_inplace(out, &block.ffn_norm2, RMS_EPSILON);
        add_modulated_residual(
            &mut tokens[row * HIDDEN..(row + 1) * HIDDEN],
            out,
            modulations.map(|values| values.gate_mlp),
        )?;
    }
    timings.linear_ffn += t.elapsed();
    timings.residual += t.elapsed();

    timings.block += block_start.elapsed();
    Ok(())
}
