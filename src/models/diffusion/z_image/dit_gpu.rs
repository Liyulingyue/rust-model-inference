//! A Vulkan session for the Z-Image DiT's transformer blocks.
//!
//! `run_block` in `dit.rs` walks the sequence one row at a time and calls
//! `linear_into` per row, which is what the CPU wants -- it keeps only one row
//! of activations live and lets the weight matrix stream past. On the GPU that
//! shape is ruinous: a step issues 1536 x 30 x 4 = 184,320 dispatches, and
//! measurement puts a fixed 128 us submit-and-fence cost on every one of them
//! (`examples/gpu_ceiling`), so the projections alone would take 88 s, worse
//! than the 81 s the same arithmetic costs on 20 CPU cores.
//!
//! So the GPU gets its own path. Activations for all rows live in one arena
//! region, each weight matrix is uploaded once and bound once, and a whole
//! block's projection becomes a single dispatch: 120 per step instead of
//! 184,320, and the 5.64 GB of weights stream once per step rather than once
//! per row.
//!
//! Nothing here is reached unless `ops::gpu_matmul_active()` is true, so the
//! CPU path is untouched: same `run_block`, same per-row `linear_into`, same
//! numerics.
use std::collections::HashMap;

use crate::vulkan::ops::{ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TokenCommands};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};

use super::dit::{FFN_WIDTH, HEADS, HIDDEN, QKV_WIDTH, ROPE_HEAD_WIDTH};

/// A projection the DiT performs, named the way `BlockWeights` names it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Projection {
    Qkv,
    Out,
    W1,
    W2,
    W3,
}

impl Projection {
    pub(crate) fn n_out(self) -> usize {
        match self {
            Projection::Qkv => QKV_WIDTH,
            Projection::Out => HIDDEN,
            Projection::W1 | Projection::W3 => FFN_WIDTH,
            Projection::W2 => HIDDEN,
        }
    }

    /// Width of the activation this projection consumes.
    ///
    /// The down projection is the odd one out: it reads the FFN's 10240-wide
    /// activation, not a 3840-wide one. Hardcoding HIDDEN here made it read a
    /// truncated row, which is why `project_scaled` looked correct and still
    /// returned a result 128x off.
    pub(crate) fn n_in(self) -> usize {
        match self {
            Projection::Qkv | Projection::Out | Projection::W1 | Projection::W3 => HIDDEN,
            Projection::W2 => FFN_WIDTH,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Projection::Qkv => "qkv",
            Projection::Out => "out",
            Projection::W1 => "w1",
            Projection::W2 => "w2",
            Projection::W3 => "w3",
        }
    }
}

/// One bound (uploaded) weight matrix.
///
/// Both members outlive the session: the buffer holds the bytes the shader
/// reads and the descriptor set points at it. `OperatorBindings` has no
/// `Drop` and is `Copy`, so holding it by value is enough to keep the set
/// allocated for the render.
struct BoundWeight {
    buffer: GpuBuffer,
    bindings: OperatorBindings,
}

/// Whether to use the register-tiled Q8_0 kernel, which measured 11.8x faster
/// on Z-Image's W2 shape (64.4 ms against 757.6 ms, streaming eight distinct
/// weight matrices) while agreeing with the old kernel to 9.5e-7 relative, and
/// which brings a 1-step render from 148 s of denoise to 84 s.
///
/// `RUST_GPU_TILED=0` falls back to the one-token-per-weight kernel, which is
/// the escape hatch for a device without integer dot product and the other half
/// of the measurement in
/// `zimage_tiled_matmul_beats_the_one_token_per_weight_kernel`.
fn tiled_matmul_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RUST_GPU_TILED")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

/// Which gamma a norm binding holds. Each needs its own descriptor set: the
/// operator layout takes at most three buffers, and a block has six gammas.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum NormKind {
    AttentionNorm1,
    AttentionNorm2,
    FfnNorm1,
    FfnNorm2,
    QNorm,
    KNorm,
}

/// Arena regions, laid out once for a fixed maximum row count and grown by
/// rebuilding if a larger render shows up.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) x: ArenaRegion,
    pub(crate) normed: ArenaRegion,
    pub(crate) out: ArenaRegion,
    /// QKV is 3x HIDDEN, so it cannot share the `out` region: a single
    /// dispatch writes all of it and the two overlap in time.
    pub(crate) qkv: ArenaRegion,
    pub(crate) gate: ArenaRegion,
    pub(crate) up: ArenaRegion,
    q8: ArenaRegion,
    q8_scales: ArenaRegion,
    q4_1_input_sums: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
    /// DiT has no KV cache -- every token sees every token in its own block --
    /// so these hold one block's K and V in the [position][head][dim] shape the
    /// attention operators index, written from the interleaved QKV projection.
    kv_cache_k: ArenaRegion,
    kv_cache_v: ArenaRegion,
    /// rows x HEADS x rows scores, and rows x HIDDEN for the attention result.
    pub(crate) attention_scores: ArenaRegion,
    pub(crate) attention_out: ArenaRegion,
}

impl Layout {
    /// Sizes come from `quantize_rows_push` / `matmul_rows_push` in ops.rs:
    /// activations are `rows * width` f32, the Q8_0 staging is `rows * width`
    /// i8 with one f32 scale per 32 values, and `q4_1_input_sums` mirrors the
    /// scales.
    /// `with_attention` reserves the score and K/V regions. They are 134 MB at
    /// the 1056-token shape for one block's scores, so a session that does not
    /// run attention must not pay for them: the wider arena measurably slows the
    /// main stack down.
    fn build(rows: usize, with_attention: bool) -> Result<Self, VulkanError> {
        let mut cursor = 0usize;
        let mut take = |elements: usize| -> Result<ArenaRegion, VulkanError> {
            cursor = cursor.next_multiple_of(4);
            if cursor.checked_add(elements).is_none() {
                return Err(VulkanError::OutOfMemory);
            }
            let region = ArenaRegion {
                offset: cursor,
                size: elements,
            };
            cursor += elements;
            Ok(region)
        };
        // Two row buffers: the block alternates between "input to the current
        // projection" and "its output", and neither aliases the other.
        let rows_hidden = rows
            .checked_mul(HIDDEN * 4)
            .ok_or(VulkanError::OutOfMemory)?;
        let rows_ffn = rows
            .checked_mul(FFN_WIDTH * 4)
            .ok_or(VulkanError::OutOfMemory)?;
        let rows_qkv = rows
            .checked_mul(QKV_WIDTH * 4)
            .ok_or(VulkanError::OutOfMemory)?;
        let rows = rows as f64;
        Ok(Self {
            x: take(rows_hidden as usize)?,
            normed: take(rows_hidden as usize)?,
            out: take(rows_hidden as usize)?,
            qkv: take(rows_qkv as usize)?,
            gate: take(rows_ffn as usize)?,
            up: take(rows_ffn as usize)?,
            // The Q8_0 staging and its scales are sized for the widest input
            // any projection consumes, which is the FFN activation at
            // FFN_WIDTH rather than the 3840-wide one the QKV projection reads.
            q8: take((rows * FFN_WIDTH as f64) as usize)?,
            q8_scales: take((rows * (FFN_WIDTH / 32) as f64) as usize * 4)?,
            q4_1_input_sums: take((rows * (FFN_WIDTH / 32) as f64) as usize * 4)?,
            q8k: take((rows * FFN_WIDTH as f64) as usize)?,
            q8k_scales: take((rows * (FFN_WIDTH / 32) as f64) as usize * 4)?,
            kv_cache_k: take(if with_attention {
                (rows * HIDDEN as f64) as usize * 4
            } else {
                0
            })?,
            kv_cache_v: take(if with_attention {
                (rows * HIDDEN as f64) as usize * 4
            } else {
                0
            })?,
            attention_scores: take(if with_attention {
                (rows * HEADS as f64 * rows) as usize * 4
            } else {
                0
            })?,
            attention_out: take(if with_attention {
                (rows * HIDDEN as f64) as usize * 4
            } else {
                0
            })?,
        })
    }

    fn bytes(&self) -> usize {
        let regions = [
            self.x,
            self.normed,
            self.out,
            self.qkv,
            self.gate,
            self.up,
            self.q8,
            self.q8_scales,
            self.q4_1_input_sums,
            self.q8k,
            self.q8k_scales,
            self.kv_cache_k,
            self.kv_cache_v,
            self.attention_scores,
            self.attention_out,
        ];
        regions.iter().map(|r| r.end()).max().unwrap_or(0)
    }
}

/// Holds the uploaded weights and the arena for one model.
pub(crate) struct DitGpuSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    layout: Layout,
    rows: usize,
    weights: HashMap<(usize, Projection), BoundWeight>,
    /// Per-(layer, kind) descriptor sets for the element-wise norm gammas.
    norms: HashMap<(usize, NormKind), BoundWeight>,
    /// Scratch the caller reads back after each dispatch.
    readback: Vec<f32>,
    /// Host-side working buffer for element-wise work between dispatches, so
    /// the fused silu(gate) * up never has to round-trip through the arena.
    pub(crate) scratch: Vec<f32>,
    /// Staging for a scaled upload; see `project_scaled`.
    scaled: Vec<f32>,
    /// The current block's AdaLN scale. One buffer reused for every block: the
    /// values change per block but the handle does not, so binding it once also
    /// keeps the descriptor pool from growing with the layer count.
    modulation: Option<BoundWeight>,
}

impl DitGpuSession {
    /// Build for `rows` sequence rows. Returns `None` if the shape does not fit,
    /// so the caller can fall back rather than fail.
    pub(crate) fn new(context: &'static VulkanContext, rows: usize) -> Result<Self, VulkanError> {
        Self::new_with_attention(context, rows, false)
    }

    /// A session with the K/V and score regions reserved, for the attention
    /// operators. They are large enough to change the main stack's speed, so
    /// only a session that runs attention should ask for them.
    pub(crate) fn new_with_attention(
        context: &'static VulkanContext,
        rows: usize,
        with_attention: bool,
    ) -> Result<Self, VulkanError> {
        if rows == 0 || rows % 32 != 0 {
            // The shader stages the input row in 4096 shared words and the
            // sequence is padded to a multiple of 32 by the caller, so this is
            // a caller bug rather than a shape we should reject.
            return Err(VulkanError::UnsupportedShape(format!(
                "Z-Image DiT row count {rows} must be a nonzero multiple of 32"
            )));
        }
        if with_attention {
            crate::vulkan::ops::require_tiled_attention(context)?;
        }
        let layout = Layout::build(rows, with_attention)?;
        // 1.5x headroom over the computed regions, rounded up, so the driver's
        // own alignment does not push the last region past the arena.
        let arena_bytes = layout.bytes().next_multiple_of(1 << 20) + (1 << 20);
        // One descriptor set per (layer, projection) that ever gets bound: 34
        // blocks x 5 projections for Z-Image Turbo, and `bind_weight_buffers`
        // allocates a fresh set each time, so the pool has to cover all of them
        // for the life of the render.
        let ops = Qwen3Ops::new_device_local_with_size(context, arena_bytes, 256)?;
        Ok(Self {
            context,
            ops,
            layout,
            rows,
            weights: HashMap::new(),
            norms: HashMap::new(),
            // QKV is the widest projection; FFN_WIDTH (10240) is narrower, so
            // one buffer sized for QKV serves all five.
            readback: vec![0f32; rows * QKV_WIDTH],
            scratch: Vec::new(),
            scaled: Vec::new(),
            modulation: None,
        })
    }

    /// Whether this projection already has an uploaded matrix, so the caller
    /// can skip the tensor lookup as well as the upload.
    pub(crate) fn has_weight(&self, layer: usize, projection: Projection) -> bool {
        self.weights.contains_key(&(layer, projection))
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Upload and bind one block's projection. Called once per block the first
    /// time it is needed, so a repeated render pays the upload only once.
    pub(crate) fn bind_weight(
        &mut self,
        layer: usize,
        projection: Projection,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        self.bind_weight_as(layer, projection, bytes, GpuWeightFormat::Q8_0)
    }

    /// Bind a weight matrix in a named format. The main stack is Q8_0; the
    /// refiner stacks are F16 in the GGUF, and each has a tiled kernel, so both
    /// can stay on the device instead of falling through to the CPU row path.
    pub(crate) fn bind_weight_as(
        &mut self,
        layer: usize,
        projection: Projection,
        bytes: &[u8],
        format: GpuWeightFormat,
    ) -> Result<(), VulkanError> {
        if self.weights.contains_key(&(layer, projection)) {
            return Ok(());
        }
        let buffer = unsafe { self.context.upload_device_static(bytes) }?;
        let bindings = match self
            .ops
            .bind_weight_buffers(std::slice::from_ref(&buffer), &[format])
        {
            Ok(bindings) => bindings,
            Err(error) => {
                unsafe { self.context.destroy_buffer(&buffer) };
                return Err(error);
            }
        };
        self.weights
            .insert((layer, projection), BoundWeight { buffer, bindings });
        Ok(())
    }

    fn binding_for(
        &self,
        layer: usize,
        projection: Projection,
    ) -> Result<OperatorBindings, VulkanError> {
        self.weights
            .get(&(layer, projection))
            .map(|w| w.bindings)
            .ok_or_else(|| {
                VulkanError::UnsupportedShape(format!(
                    "Z-Image DiT projection {projection:?} for layer {layer} was not bound"
                ))
            })
    }

    /// Run one projection over every row: read `input` from `input_region`,
    /// write into `output_region`, and leave the result in `readback`.
    #[allow(clippy::too_many_arguments)]
    /// Run one projection over every row, unscaled.
    pub(crate) fn project(
        &mut self,
        layer: usize,
        projection: Projection,
        input_region: ArenaRegion,
        input: &[f32],
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.project_scaled(layer, projection, input_region, input, output_region, 1.0)
    }

    /// Run one projection over every row.
    ///
    /// `scale` multiplies the activations before they are uploaded, which is
    /// the only place a factor can go: `record_weight_matmul_rows` quantizes on
    /// the device and exposes no scale argument. The matmul is linear in its
    /// input, so `scale * (W . x) == W . (scale * x)`.
    ///
    /// Z-Image's FFN down projection passes 1.0, not the 1/128 `run_block`
    /// writes. That is not an oversight: `linear_into_scaled_impl` applies its
    /// scale twice, once to the activations before quantizing and once to the
    /// result afterwards, so the two cancel and the CPU path's `1/128` is a
    /// no-op. Applying it once here would have made the GPU disagree with the
    /// CPU by exactly that factor -- which is what the first version did, and
    /// is most of the 64/255 it produced.
    ///
    /// The cost is one pass over the activations, 0.7 MB for a 512x512 step,
    /// against 5.64 GB of weights.
    pub(crate) fn project_scaled(
        &mut self,
        layer: usize,
        projection: Projection,
        input_region: ArenaRegion,
        input: &[f32],
        output_region: ArenaRegion,
        scale: f32,
    ) -> Result<(), VulkanError> {
        let bindings = self.binding_for(layer, projection)?;
        if scale == 1.0 {
            self.ops.write_f32(input_region, input)?;
        } else {
            if self.scaled.len() < input.len() {
                self.scaled.resize(input.len(), 0.0);
            }
            for (destination, value) in self.scaled[..input.len()].iter_mut().zip(input) {
                *destination = value * scale;
            }
            self.ops
                .write_f32(input_region, &self.scaled[..input.len()])?;
        }
        let mut commands = TokenCommands::begin(self.context)?;
        self.record_projection(
            &mut commands,
            bindings,
            projection,
            input_region,
            output_region,
        )?;
        commands.submit_and_wait()?;
        self.read_into(projection, output_region)
    }

    /// Record one projection onto an in-flight command buffer.
    ///
    /// Nothing is uploaded and nothing is read back: the input has to be
    /// resident already. That is what makes fusing worth doing -- a block
    /// otherwise ships 167.8 MB of activations over PCIe, and at the ~10 GB/s
    /// that costs it is the entire step, against 58 ms for the dispatches
    /// themselves.
    pub(crate) fn record_projection(
        &self,
        commands: &mut TokenCommands<'_>,
        bindings: OperatorBindings,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        // The register-tiled kernel: measured 108x faster than the one-token-per
        // weight kernel on Z-Image's W2 shape while agreeing with it to 7.6e-7
        // relative. It needs integer dot product, so the older kernel stays as
        // the fallback for devices without it.
        if self.context.supports_integer_dot_product() && tiled_matmul_enabled() {
            return self.record_projection_tiled(
                commands,
                bindings,
                projection,
                input_region,
                output_region,
            );
        }
        self.record_projection_grouped(commands, bindings, projection, input_region, output_region)
    }

    /// The one-token-per-weight grouped kernel, kept for the benchmark that
    /// compares it against `record_projection_tiled` and as the fallback for
    /// devices without integer dot product.
    pub(crate) fn record_projection_grouped(
        &self,
        commands: &mut TokenCommands<'_>,
        bindings: OperatorBindings,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.ops.record_weight_matmul_rows(
            commands,
            bindings,
            input_region,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(output_region, projection.n_out(), projection.n_out() * 4)],
            projection.n_in(),
            self.rows,
            projection.n_in(),
        )
    }

    /// Time one projection with either the grouped or the tiled kernel and
    /// return the output, for `examples/zimage_tiled_bench` to compare.
    ///
    /// The input has to be resident already, which is why this takes a region
    /// rather than a slice.
    pub(crate) fn bench_projection(
        &self,
        layer: usize,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
        tiled: bool,
        iterations: usize,
    ) -> Result<(f64, Vec<f32>), VulkanError> {
        let bindings = self.binding_for(layer, projection)?;
        let record = |commands: &mut TokenCommands<'_>| {
            if tiled {
                self.record_projection_tiled(
                    commands,
                    bindings,
                    projection,
                    input_region,
                    output_region,
                )
            } else {
                self.record_projection_grouped(
                    commands,
                    bindings,
                    projection,
                    input_region,
                    output_region,
                )
            }
        };
        // Warm first: the first dispatch of a pipeline pays driver JIT.
        let mut commands = TokenCommands::begin(self.context)?;
        record(&mut commands)?;
        commands.submit_and_wait()?;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut commands = TokenCommands::begin(self.context)?;
            record(&mut commands)?;
            commands.submit_and_wait()?;
        }
        let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
        let output_len = self.rows * projection.n_out();
        let mut values = vec![0.0; output_len];
        self.ops.read_f32_into(output_region, &mut values)?;
        Ok((elapsed, values))
    }

    /// Time one projection per layer over a set of distinct weight matrices,
    /// all in a single command buffer.
    ///
    /// A single matrix read repeatedly measures an L2-resident kernel, and on
    /// GB10 that reported 110,000 GOP/s where the render with 150 distinct
    /// matrices per step reported 108. Streaming distinct matrices is what the
    /// render actually does, so it is the number worth comparing.
    pub(crate) fn bench_projection_streaming(
        &self,
        layers: &[usize],
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
        tiled: bool,
        iterations: usize,
    ) -> Result<f64, VulkanError> {
        let bindings: Vec<OperatorBindings> = layers
            .iter()
            .map(|layer| self.binding_for(*layer, projection))
            .collect::<Result<_, _>>()?;
        let start = std::time::Instant::now();
        let mut commands = TokenCommands::begin(self.context)?;
        for _ in 0..iterations {
            for (binding, _) in bindings.iter().zip(layers.iter()) {
                if tiled {
                    self.record_projection_tiled(
                        &mut commands,
                        *binding,
                        projection,
                        input_region,
                        output_region,
                    )?;
                } else {
                    self.record_projection_grouped(
                        &mut commands,
                        *binding,
                        projection,
                        input_region,
                        output_region,
                    )?;
                }
            }
        }
        commands.submit_and_wait()?;
        let elapsed = start.elapsed().as_secs_f64();
        Ok(elapsed / (iterations * layers.len()) as f64)
    }

    /// `record_projection` with the register-tiled grouped kernel, for
    /// measuring what the tiling is worth. Not wired into `run_block_gpu`.
    pub(crate) fn record_projection_tiled(
        &self,
        commands: &mut TokenCommands<'_>,
        bindings: OperatorBindings,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.ops.record_weight_matmul_tiled_rows(
            commands,
            bindings,
            input_region,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(output_region, projection.n_out(), projection.n_out() * 4)],
            projection.n_in(),
            self.rows,
            projection.n_in(),
        )
    }

    fn read_into(
        &mut self,
        projection: Projection,
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        let output_len = self.rows * projection.n_out();
        if self.readback.len() < output_len {
            self.readback.resize(output_len, 0.0);
        }
        self.ops
            .read_f32_into(output_region, &mut self.readback[..output_len])?;
        Ok(())
    }

    /// Bind the current block's AdaLN scale, reusing the buffer after the first
    /// block. A fresh `bind_buffers` per block would burn one descriptor set per
    /// layer per step out of a fixed pool.
    pub(crate) fn bind_modulation(&mut self, values: &[f32]) -> Result<(), VulkanError> {
        match &self.modulation {
            Some(bound) if bound.buffer.size as usize >= values.len() * 4 => {
                // SAFETY: a persistently mapped host-visible allocation, sized
                // once when it was created and only ever written with at most
                // this many floats.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        values.as_ptr(),
                        bound.buffer.mapped.cast::<f32>(),
                        values.len(),
                    );
                }
            }
            _ => {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|value| f32::to_le_bytes(*value))
                    .collect();
                let buffer = unsafe { self.context.upload_static(&bytes)? };
                let bindings = self.ops.bind_buffers(std::slice::from_ref(&buffer))?;
                self.modulation = Some(BoundWeight { buffer, bindings });
            }
        }
        Ok(())
    }

    pub(crate) fn modulation_bindings(&self) -> Result<OperatorBindings, VulkanError> {
        self.modulation
            .as_ref()
            .map(|bound| bound.bindings)
            .ok_or_else(|| VulkanError::UnsupportedShape("AdaLN scale was never bound".into()))
    }

    /// Fused rms_norm -> AdaLN -> QKV for one block.
    ///
    /// The normed and modulated activations are consumed by the QKV matmul and
    /// by nothing else, so the GPU keeps them instead of shipping them to the
    /// host and back.
    /// `modulation` is `None` for the context refiner, which carries no AdaLN
    /// weights; the norm still runs and the AdaLN dispatch is skipped, matching
    /// `scale_modulated_branch`'s no-op on a missing scale.
    pub(crate) fn record_attention_qkv(
        &mut self,
        layer: usize,
        rms_gamma: &[f32],
        modulation: Option<&[f32]>,
    ) -> Result<(), VulkanError> {
        self.bind_norm(layer, NormKind::AttentionNorm1, rms_gamma)?;
        let norm = self.norm_bindings(layer, NormKind::AttentionNorm1)?;
        let scale = match modulation {
            Some(values) => {
                self.bind_modulation(values)?;
                Some(self.modulation_bindings()?)
            }
            None => None,
        };
        let weights = self.binding_for(layer, Projection::Qkv)?;
        let layout = self.layout;
        let rows = self.rows;
        let hidden = crate::models::diffusion::z_image::dit::HIDDEN;
        let eps = crate::models::diffusion::z_image::dit::RMS_EPSILON;
        // `TokenCommands::begin(self.context)` rather than `self.begin()`: the
        // latter borrows all of `self`, which `record_projection` still needs.
        let t0 = std::time::Instant::now();
        let mut commands = TokenCommands::begin(self.context)?;
        self.ops.record_rms_norm_rows(
            &commands,
            norm,
            layout.x,
            layout.normed,
            hidden,
            eps,
            rows,
            hidden,
            hidden,
        )?;
        let t_norm = t0.elapsed();
        if let Some(scale) = scale {
            self.ops.record_adaln_modulate_rows(
                &commands,
                scale,
                layout.normed,
                hidden,
                rows,
                hidden,
            )?;
        }
        let t_adaln = t0.elapsed();
        self.record_projection(
            &mut commands,
            weights,
            Projection::Qkv,
            layout.normed,
            layout.qkv,
        )?;
        let t_mat = t0.elapsed();
        commands.submit_and_wait()?;
        let t_fence = t0.elapsed();
        self.read_into(Projection::Qkv, layout.qkv)?;
        {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                eprintln!(
                    "[qkv-probe] rms_norm={t_norm:?} +adaln={:?} +qkv_matmul={:?} +fence={:?} +readback={:?}",
                    t_adaln - t_norm, t_mat - t_adaln, t_fence - t_mat, t0.elapsed() - t_fence
                );
            }
        }
        Ok(())
    }

    /// Scores for the DiT's attention, from the interleaved QKV projection.
    ///
    /// `layout.qkv` must already hold the projected QKV with the qk_norm and
    /// RoPE already applied to its q and k halves, which is what `run_block`
    /// produces before it calls `attention_into`.
    pub(crate) fn record_diy_attention(
        &mut self,
        rows: usize,
        scores: ArenaRegion,
        out: ArenaRegion,
    ) -> Result<(), VulkanError> {
        let layout = self.layout;
        let mut commands = TokenCommands::begin(self.context)?;
        self.ops.record_diy_attention_full(
            &commands,
            layout.qkv,
            layout.kv_cache_k,
            layout.kv_cache_v,
            layout.gate,
            scores,
            out,
            QKV_WIDTH,
            HEADS,
            HEADS,
            ROPE_HEAD_WIDTH,
            rows,
        )?;
        commands.submit_and_wait()?;
        Ok(())
    }

    /// Open a command buffer for a fused chain of dispatches.
    pub(crate) fn begin(&self) -> Result<TokenCommands<'_>, VulkanError> {
        TokenCommands::begin(self.context)
    }

    /// Read a projection's result out of the arena.
    pub(crate) fn read_projection(
        &mut self,
        projection: Projection,
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.read_into(projection, output_region)
    }

    pub(crate) fn ops(&self) -> &Qwen3Ops<'static> {
        &self.ops
    }

    /// Mutable access, for the operations that allocate from the descriptor
    /// pool (`bind_buffers` and friends).
    pub(crate) fn ops_mut(&mut self) -> &mut Qwen3Ops<'static> {
        &mut self.ops
    }

    /// Bind one norm gamma for the arena-only pipeline.
    ///
    /// The descriptor set takes at most three buffers, so each norm kind gets
    /// its own set rather than sharing one six-way binding. They are created
    /// once per (layer, kind) and live as long as the render, because
    /// `bind_buffers` allocates from a pool that is never freed.
    pub(crate) fn bind_norm(
        &mut self,
        layer: usize,
        kind: NormKind,
        values: &[f32],
    ) -> Result<(), VulkanError> {
        if self.norms.contains_key(&(layer, kind)) {
            return Ok(());
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| f32::to_le_bytes(*value))
            .collect();
        let buffer = unsafe { self.context.upload_static(&bytes) }?;
        let bindings = self.ops.bind_buffers(std::slice::from_ref(&buffer))?;
        self.norms
            .insert((layer, kind), BoundWeight { buffer, bindings });
        Ok(())
    }

    pub(crate) fn norm_bindings(
        &self,
        layer: usize,
        kind: NormKind,
    ) -> Result<OperatorBindings, VulkanError> {
        self.norms
            .get(&(layer, kind))
            .map(|w| w.bindings)
            .ok_or_else(|| {
                VulkanError::UnsupportedShape(format!(
                    "Z-Image DiT norm {kind:?} for layer {layer} was not bound"
                ))
            })
    }

    /// The last projection's result, valid until the next `project` call.
    pub(crate) fn readback(&self, len: usize) -> &[f32] {
        &self.readback[..len]
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl Drop for DitGpuSession {
    fn drop(&mut self) {
        // Every uploaded buffer belongs to this render. On an uncertain fence,
        // confirm idle before freeing it; retain the handles if recovery fails.
        unsafe {
            let _ = self.context.destroy_completed_buffers(
                self.weights
                    .values()
                    .map(|bound| &bound.buffer)
                    .chain(self.norms.values().map(|bound| &bound.buffer))
                    .chain(self.modulation.iter().map(|bound| &bound.buffer)),
            );
        }
    }
}
