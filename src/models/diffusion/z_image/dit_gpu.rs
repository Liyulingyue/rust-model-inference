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

use crate::vulkan::ops::{
    tiled_matmul_enabled, ArenaRegion, GpuMatmulMode, GpuWeightFormat, OperatorBindings, Qwen3Ops,
    TokenCommands,
};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};

use super::dit::{FFN_WIDTH, HEADS, HIDDEN, QKV_WIDTH, QK_RMS_EPSILON, ROPE_HEAD_WIDTH};

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
    /// A second buffer, for the descriptor sets that carry a pair of weights
    /// (the fused Q/K norms). Single-buffer callers leave it `None`.
    extra: Option<GpuBuffer>,
    bindings: OperatorBindings,
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
    /// A second Q8_0 staging region, the same width as `q8`.
    ///
    /// w1 and w3 both read `x` and would otherwise both quantize into `q8`. In
    /// the fused chain that makes them share one buffer with only a barrier
    /// between them, and a barrier orders memory, not reads: w3's quantize can
    /// land in `q8` before w1's matmul has drained it, so w1 multiplies its
    /// weights against w3's activations. That is the `-inf` the fused FFN
    /// produces -- w1's output came back with an amax of 6314 against 103 for
    /// the same layer submitted one dispatch at a time. Giving w3 its own
    /// staging removes the write-after-read hazard entirely.
    q8_up: ArenaRegion,
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
    /// The (cos, sin) table `record_qk_norm_rope_interleaved_rows` reads: one
    /// `ROPE_HEAD_WIDTH` row per token, shared by every head of that token.
    /// Uploaded once per session -- it does not depend on the layer or the step.
    pub(crate) rope: ArenaRegion,
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
            // Paired with `q8`; see the field docs for why w3 cannot share it.
            q8_up: take((rows * FFN_WIDTH as f64) as usize)?,
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
            rope: take((rows * ROPE_HEAD_WIDTH as f64) as usize * 4)?,
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
            self.q8_up,
            self.rope,
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
    /// Per-layer Q/K norm gamma pairs for the fused interleaved kernel.
    qk_norms: HashMap<usize, BoundWeight>,
    /// Row count whose RoPE table is already resident in the arena.
    rope_rows: Option<usize>,
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
            qk_norms: HashMap::new(),
            rope_rows: None,
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
        // Refiner projections retain their F16 input rounding and F32 reduction.
        let mode = if format == GpuWeightFormat::F16 {
            GpuMatmulMode::RoundedInputF32
        } else {
            GpuMatmulMode::Prepared
        };
        let bindings =
            match self
                .ops
                .bind_weight_buffers_mode(std::slice::from_ref(&buffer), &[format], mode)
            {
                Ok(bindings) => bindings,
                Err(error) => {
                    unsafe { self.context.destroy_buffer(&buffer) };
                    return Err(error);
                }
            };
        self.weights.insert(
            (layer, projection),
            BoundWeight {
                buffer,
                extra: None,
                bindings,
            },
        );
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
        self.record_projection_grouped_q8(
            commands,
            bindings,
            projection,
            input_region,
            output_region,
            self.layout.q8,
        )
    }

    /// As [`Self::record_projection_grouped`], but quantizing into a caller
    /// chosen staging region. See `Layout::q8_up` for why the fused FFN needs
    /// w1 and w3 to quantize into different buffers.
    pub(crate) fn record_projection_grouped_q8(
        &self,
        commands: &mut TokenCommands<'_>,
        bindings: OperatorBindings,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
        q8: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.ops.record_weight_matmul_ordinary_rows(
            commands,
            bindings,
            input_region,
            q8,
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
        self.record_projection_tiled_q8(
            commands,
            bindings,
            projection,
            input_region,
            output_region,
            self.layout.q8,
        )
    }

    /// As [`Self::record_projection_tiled`], with a caller chosen Q8_0 staging
    /// region. See `Layout::q8_up`.
    pub(crate) fn record_projection_tiled_q8(
        &self,
        commands: &mut TokenCommands<'_>,
        bindings: OperatorBindings,
        projection: Projection,
        input_region: ArenaRegion,
        output_region: ArenaRegion,
        q8: ArenaRegion,
    ) -> Result<(), VulkanError> {
        self.ops.record_weight_matmul_tiled_rows(
            commands,
            bindings,
            input_region,
            q8,
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
                self.modulation = Some(BoundWeight {
                    buffer,
                    extra: None,
                    bindings,
                });
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
        q_norm: &[f32],
        k_norm: &[f32],
        fuse_qk_norm_rope: bool,
    ) -> Result<(), VulkanError> {
        self.bind_norm(layer, NormKind::AttentionNorm1, rms_gamma)?;
        // Bound here rather than at the RoPE dispatch so a block that never
        // reaches attention -- the context refiner does not -- never pays for the
        // descriptor set.
        self.bind_qk_norms(layer, q_norm, k_norm)?;
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
        // Q/K norm and RoPE fold into this same command buffer, so the QKV span
        // never leaves the device: that is the point. Without this the caller has
        // to download the whole `rows * QKV_WIDTH` projection just to rotate it,
        // which at 512x512 is 177 MB each way per block.
        if fuse_qk_norm_rope {
            self.record_interleaved_qk_norm_rope(&commands, layer, rows)?;
        }
        let t_rope = t0.elapsed();
        commands.submit_and_wait()?;
        let t_fence = t0.elapsed();
        // Only the CPU attention path needs the projection back on the host; when
        // attention also runs on the device this is pure waste.
        if !fuse_qk_norm_rope {
            self.read_into(Projection::Qkv, layout.qkv)?;
        }
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

    /// Record the whole FFN -- w1, w3, the SiLU product, then w2 -- into one
    /// command buffer, so the caller pays one submission instead of four and
    /// the `ffn_len`-wide gate/up buffers never cross the bus.
    ///
    /// The four dispatches read and write disjoint arena regions in sequence
    /// (`x` -> `gate`, `x` -> `up`, `gate`*`up` -> `gate`, `gate` -> `out`),
    /// and every recording ends in a barrier, so each stage sees the previous
    /// one's result. `record_projection` picks the same kernel per projection
    /// that `project`/`project_scaled` would have used, so the arithmetic is
    /// unchanged.
    ///
    /// All three projections run at scale 1.0, which is why the caller can
    /// leave `x` on the device instead of staging it through the host: see
    /// `project_scaled` for why the down projection must not carry the CPU
    /// path's `1/128`.
    pub(crate) fn record_ffn<'a>(
        &'a self,
        mut commands: TokenCommands<'a>,
        layer: usize,
        ffn_len: usize,
    ) -> Result<TokenCommands<'a>, VulkanError> {
        let layout = self.layout;
        // `SPLIT_FFN` gives every projection its own submission. The default
        // records all four into one buffer.
        let split = std::env::var("SPLIT_FFN").is_ok();
        for projection in [Projection::W1, Projection::W3, Projection::W2] {
            let bindings = self.binding_for(layer, projection)?;
            let (input, output) = match projection {
                Projection::W1 => (layout.x, layout.gate),
                Projection::W3 => (layout.x, layout.up),
                _ => (layout.gate, layout.out),
            };
            // w3 quantizes into its own staging: sharing `q8` with w1 lets it
            // overwrite activations w1's matmul has not drained yet.
            if matches!(projection, Projection::W3) {
                self.record_projection_tiled_q8(
                    &mut commands,
                    bindings,
                    projection,
                    input,
                    output,
                    layout.q8_up,
                )?;
            } else {
                self.record_projection(&mut commands, bindings, projection, input, output)?;
            }
            if matches!(projection, Projection::W3) {
                // The SiLU dispatch is the one that corrupts the chain, so it
                // gets its own submission instead of sharing w3's.
                self.ops
                    .record_silu_mul(&commands, layout.gate, layout.up, ffn_len)?;
            }
            if split && !matches!(projection, Projection::W2) {
                commands = commands.flush()?;
            }
        }
        Ok(commands)
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
        self.norms.insert(
            (layer, kind),
            BoundWeight {
                buffer,
                extra: None,
                bindings,
            },
        );
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

    /// Upload the per-layer Q/K norm gammas as one arena-plus-two descriptor
    /// set, which is what `record_qk_norm_rope_interleaved_rows` binds.
    ///
    /// Separate from `bind_norm` because that one puts the gamma alone in slot
    /// 0, while the fused kernel needs the arena at slot 0 and the two gammas
    /// after it. Created once per layer and kept for the render, like the rest.
    pub(crate) fn bind_qk_norms(
        &mut self,
        layer: usize,
        q_norm: &[f32],
        k_norm: &[f32],
    ) -> Result<(), VulkanError> {
        if self.qk_norms.contains_key(&layer) {
            return Ok(());
        }
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| f32::to_le_bytes(*value))
                .collect::<Vec<u8>>()
        };
        let q_buffer = unsafe { self.context.upload_static(&bytes(q_norm)) }?;
        let k_buffer = unsafe { self.context.upload_static(&bytes(k_norm)) }?;
        let bindings = self
            .ops
            .bind_buffers(&[q_buffer, k_buffer])
            .map_err(|error| error)?;
        self.qk_norms.insert(
            layer,
            BoundWeight {
                buffer: q_buffer,
                extra: Some(k_buffer),
                bindings,
            },
        );
        Ok(())
    }

    pub(crate) fn qk_norm_bindings(&self, layer: usize) -> Result<OperatorBindings, VulkanError> {
        self.qk_norms
            .get(&layer)
            .map(|w| w.bindings)
            .ok_or_else(|| {
                VulkanError::UnsupportedShape(format!(
                    "Z-Image DiT Q/K norms for layer {layer} were not bound"
                ))
            })
    }

    /// Upload the RoPE table into the arena once per session.
    ///
    /// The table depends only on the token geometry, not on the layer or the
    /// denoise step, so it is written once and reused by every block of every
    /// step. Redoing it per block would cost 1.9 MB x 34 x 8 for no reason; a
    /// length change means a different geometry and does re-upload.
    pub(crate) fn ensure_rope(&mut self, table: &[f32], rows: usize) -> Result<(), VulkanError> {
        if rows == 0 || table.len() != rows * ROPE_HEAD_WIDTH {
            return Err(VulkanError::UnsupportedShape(format!(
                "Z-Image RoPE table has {} entries, expected {} rows of {ROPE_HEAD_WIDTH}",
                table.len(),
                rows
            )));
        }
        if self.rope_rows == Some(rows) {
            return Ok(());
        }
        let rope = self.layout.rope;
        self.ops.write_f32(rope, table)?;
        self.rope_rows = Some(rows);
        Ok(())
    }

    /// Norm Q and K per head, then apply the interleaved RoPE, all on device.
    ///
    /// This is the fused replacement for the host loop that walked every
    /// `rows * HEADS` head on the CPU and had to bring the whole QKV buffer back
    /// across the bus to do it.
    pub(crate) fn record_interleaved_qk_norm_rope(
        &self,
        commands: &TokenCommands<'_>,
        layer: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let layout = self.layout;
        // One row of the projection output is `Q | K | V`, so Q and K are spans
        // inside the same buffer, `QKV_WIDTH` apart, not two separate blocks.
        self.ops.record_qk_norm_rope_interleaved_rows(
            commands,
            self.qk_norm_bindings(layer)?,
            layout.qkv,
            layout.qkv,
            HEADS,
            HEADS,
            ROPE_HEAD_WIDTH,
            layout.rope,
            QKV_WIDTH,
            QKV_WIDTH,
            HIDDEN,
            QK_RMS_EPSILON,
            true,
            true,
            rows,
        )
    }

    /// The last projection's result, valid until the next `project` call.
    pub(crate) fn readback(&self, len: usize) -> &[f32] {
        &self.readback[..len]
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}

#[cfg(test)]
mod f16_contract_tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn f16_refiner_projection_keeps_f32_accumulation() {
        let context = Box::leak(Box::new(VulkanContext::new().unwrap()));
        let rows = 32;
        let mut session = DitGpuSession::new(context, rows).unwrap();
        let weight: Vec<u8> = (0..FFN_WIDTH * HIDDEN)
            .flat_map(|_| crate::ops::f32_to_f16(1.0).to_le_bytes())
            .collect();
        session
            .bind_weight_as(0, Projection::W2, &weight, GpuWeightFormat::F16)
            .unwrap();
        let layout = *session.layout();
        // Finite F16 operands can overflow a half reduction. The historical
        // refiner rounds its input to F16 but keeps the accumulation in F32.
        let input = vec![-256.0625; rows * FFN_WIDTH];
        session.ops.write_f32(layout.gate, &input).unwrap();
        let bindings = session.binding_for(0, Projection::W2).unwrap();
        let mut output = vec![0.0; rows * HIDDEN];
        for tiled in [false, true] {
            let mut commands = TokenCommands::begin(context).unwrap();
            if tiled {
                session.record_projection_tiled(
                    &mut commands,
                    bindings,
                    Projection::W2,
                    layout.gate,
                    layout.out,
                )
            } else {
                session.record_projection_grouped(
                    &mut commands,
                    bindings,
                    Projection::W2,
                    layout.gate,
                    layout.out,
                )
            }
            .unwrap();
            commands.submit_and_wait().unwrap();
            session.ops.read_f32_into(layout.out, &mut output).unwrap();
            let expected = -256.0 * FFN_WIDTH as f32;
            assert!(
                output.iter().all(|&value| value == expected),
                "tiled={tiled}, expected={expected}, first={}",
                output[0],
            );
        }
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
                    .chain(self.qk_norms.values().map(|bound| &bound.buffer))
                    .chain(self.modulation.iter().map(|bound| &bound.buffer))
                    .chain(
                        self.qk_norms
                            .values()
                            .filter_map(|bound| bound.extra.as_ref()),
                    )
                    .chain(
                        self.weights
                            .values()
                            .filter_map(|bound| bound.extra.as_ref()),
                    )
                    .chain(self.norms.values().filter_map(|bound| bound.extra.as_ref())),
            );
        }
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    /// Q8_0 bytes for one FFN projection: `n_out` rows of `n_in / 32` blocks,
    /// each an f16 scale followed by 32 int8 lanes. Deterministic in `salt` so
    /// W1/W2/W3 differ without needing a GGUF file.
    fn synthetic_ffn_weight(n_in: usize, n_out: usize, salt: usize) -> Vec<u8> {
        let blocks_per_row = n_in / 32;
        let mut bytes = vec![0u8; n_out * blocks_per_row * 34];
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let offset = (row * blocks_per_row + block) * 34;
                let scale = 0.004 + ((row + block + salt) % 13) as f32 * 0.0003;
                bytes[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                for lane in 0..32 {
                    let quant = ((row * 17 + block * 11 + lane * 5 + salt) % 41) as i8 - 20;
                    bytes[offset + 2 + lane] = quant as u8;
                }
            }
        }
        bytes
    }

    /// Does recording the whole FFN (w1 -> w3 -> silu -> w2) into one command
    /// buffer drop its tail on this driver?
    ///
    /// `TokenCommands::flush` exists because drivers "can silently drop the
    /// trailing dispatches of a very long command buffer"; Qwen3.5 hit it at
    /// ~230 dispatches and chunks at 16. The reverted Z-Image FFN batching
    /// reported an all `-inf` activation where submitting the same chain one
    /// dispatch at a time was correct -- the same signature, and it was blamed
    /// on descriptor-set reuse rather than on dispatch chunking.
    ///
    /// This runs the real four-dispatch chain both ways on the real FFN shape.
    /// Matching means this driver honours the batch here, so the reported
    /// `-inf` had another cause and batching is safe to retry. Diverging means
    /// batching needs `flush` chunking first. Either way it is measured.
    #[test]
    fn batched_ffn_chain_matches_incremental_submission() {
        crate::ops::float::enable_gpu();
        let Some(context) = crate::ops::get_vulkan_context() else {
            eprintln!("skipped: no Vulkan context");
            return;
        };

        let rows = if std::env::var("BIG_ROWS").is_ok() {
            3840
        } else {
            256
        };
        let (n_in, n_out) = (HIDDEN, FFN_WIDTH);
        let mut session = DitGpuSession::new(context, rows).expect("session");
        session
            .bind_weight(0, Projection::W1, &synthetic_ffn_weight(n_in, n_out, 3))
            .expect("bind w1");
        session
            .bind_weight(0, Projection::W3, &synthetic_ffn_weight(n_in, n_out, 5))
            .expect("bind w3");
        session
            .bind_weight(0, Projection::W2, &synthetic_ffn_weight(n_out, n_in, 7))
            .expect("bind w2");

        let layout = *session.layout();
        let input: Vec<f32> = (0..rows * n_in)
            .map(|i| ((i % 251) as f32 - 125.0) / 97.0)
            .collect();
        session.ops().write_f32(layout.x, &input).expect("write x");

        // Batched: all four dispatched once, submitted once.
        let before = context.submission_count();
        let mut batched = vec![0.0f32; rows * n_in];
        {
            let commands = TokenCommands::begin(context).expect("begin");
            let commands = session
                .record_ffn(commands, 0, rows * n_out)
                .expect("record FFN");
            commands.submit_and_wait().expect("submit");
        }
        session
            .ops()
            .read_f32_into(layout.out, &mut batched)
            .expect("read batched");
        let batched_submissions = context.submission_count() - before;

        // Incremental: the same four dispatches, one submission each.
        let before = context.submission_count();
        let mut incremental = vec![0.0f32; rows * n_in];
        for step in 0..4 {
            let mut commands = TokenCommands::begin(context).expect("begin");
            match step {
                0 => session
                    .record_projection(
                        &mut commands,
                        session.binding_for(0, Projection::W1).unwrap(),
                        Projection::W1,
                        layout.x,
                        layout.gate,
                    )
                    .expect("w1"),
                1 => session
                    .record_projection(
                        &mut commands,
                        session.binding_for(0, Projection::W3).unwrap(),
                        Projection::W3,
                        layout.x,
                        layout.up,
                    )
                    .expect("w3"),
                2 => session
                    .ops()
                    .record_silu_mul(&commands, layout.gate, layout.up, rows * n_out)
                    .expect("silu"),
                _ => session
                    .record_projection(
                        &mut commands,
                        session.binding_for(0, Projection::W2).unwrap(),
                        Projection::W2,
                        layout.gate,
                        layout.out,
                    )
                    .expect("w2"),
            }
            commands.submit_and_wait().expect("submit");
        }
        session
            .ops()
            .read_f32_into(layout.out, &mut incremental)
            .expect("read incremental");
        let incremental_submissions = context.submission_count() - before;

        let finite = batched.iter().filter(|value| value.is_finite()).count();
        let first_bad = batched
            .iter()
            .zip(incremental.iter())
            .position(|(&a, &b)| !a.is_finite() || (a - b).abs() > 1e-3);
        // `submission_count` is process-global, so concurrent GPU tests inflate
        // these deltas. Report them rather than assert on them; the batching
        // claim rests on the element-wise comparison.
        eprintln!(
            "batched ffn rows={rows} finite={finite}/{} submissions batched={batched_submissions} incremental={incremental_submissions} batched[0]={:.6} incremental[0]={:.6} first_bad={first_bad:?}",
            batched.len(),
            batched[0],
            incremental[0],
        );
        assert_eq!(
            finite,
            batched.len(),
            "batched FFN produced {} non-finite values",
            batched.len() - finite
        );
        assert_eq!(
            first_bad, None,
            "batched FFN diverges from incremental submission"
        );
    }
}
