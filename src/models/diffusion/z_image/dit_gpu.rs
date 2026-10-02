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
    ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TokenCommands,
};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};

use super::dit::{FFN_WIDTH, HIDDEN, QKV_WIDTH};

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

/// Arena regions, laid out once for a fixed maximum row count and grown by
/// rebuilding if a larger render shows up.
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) x: ArenaRegion,
    pub(crate) normed: ArenaRegion,
    pub(crate) out: ArenaRegion,
    pub(crate) gate: ArenaRegion,
    pub(crate) up: ArenaRegion,
    q8: ArenaRegion,
    q8_scales: ArenaRegion,
    q4_1_input_sums: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
}

impl Layout {
    /// Sizes come from `quantize_rows_push` / `matmul_rows_push` in ops.rs:
    /// activations are `rows * width` f32, the Q8_0 staging is `rows * width`
    /// i8 with one f32 scale per 32 values, and `q4_1_input_sums` mirrors the
    /// scales.
    fn build(rows: usize) -> Result<Self, VulkanError> {
        let blocks = HIDDEN / 32;
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
        let rows_hidden = rows.checked_mul(HIDDEN * 4).ok_or(VulkanError::OutOfMemory)?;
        let rows_ffn = rows.checked_mul(FFN_WIDTH * 4).ok_or(VulkanError::OutOfMemory)?;
        let rows = rows as f64;
        Ok(Self {
            x: take(rows_hidden as usize)?,
            normed: take(rows_hidden as usize)?,
            out: take(rows_hidden as usize)?,
            gate: take(rows_ffn as usize)?,
            up: take(rows_ffn as usize)?,
            q8: take((rows * HIDDEN as f64) as usize)?,
            q8_scales: take((rows * blocks as f64) as usize * 4)?,
            q4_1_input_sums: take((rows * blocks as f64) as usize * 4)?,
            q8k: take((rows * HIDDEN as f64) as usize)?,
            q8k_scales: take((rows * blocks as f64) as usize * 4)?,
        })
    }

    fn bytes(&self) -> usize {
        let regions = [
            self.x, self.normed, self.out, self.gate, self.up, self.q8, self.q8_scales,
            self.q4_1_input_sums, self.q8k, self.q8k_scales,
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
    /// Scratch the caller reads back after each dispatch.
    readback: Vec<f32>,
    /// Host-side working buffer for element-wise work between dispatches, so
    /// the fused silu(gate) * up never has to round-trip through the arena.
    pub(crate) scratch: Vec<f32>,
}

impl DitGpuSession {
    /// Build for `rows` sequence rows. Returns `None` if the shape does not fit,
    /// so the caller can fall back rather than fail.
    pub(crate) fn new(context: &'static VulkanContext, rows: usize) -> Result<Self, VulkanError> {
        if rows == 0 || rows % 32 != 0 {
            // The shader stages the input row in 4096 shared words and the
            // sequence is padded to a multiple of 32 by the caller, so this is
            // a caller bug rather than a shape we should reject.
            return Err(VulkanError::UnsupportedShape(format!(
                "Z-Image DiT row count {rows} must be a nonzero multiple of 32"
            )));
        }
        let layout = Layout::build(rows)?;
        // 1.5x headroom over the computed regions, rounded up, so the driver's
        // own alignment does not push the last region past the arena.
        let arena_bytes = layout.bytes().next_multiple_of(1 << 20) + (1 << 20);
        let ops = Qwen3Ops::new_with_size(context, arena_bytes, 64)?;
        Ok(Self {
            context,
            ops,
            layout,
            rows,
            weights: HashMap::new(),
            readback: vec![0f32; rows * QKV_WIDTH],
            scratch: Vec::new(),
        })
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
        if self.weights.contains_key(&(layer, projection)) {
            return Ok(());
        }
        let buffer = unsafe { self.context.upload_static(bytes) }?;
        let bindings = self
            .ops
            .bind_weight_buffers(std::slice::from_ref(&buffer), &[GpuWeightFormat::Q8_0])?;
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
    pub(crate) fn project(
        &mut self,
        layer: usize,
        projection: Projection,
        input_region: ArenaRegion,
        input: &[f32],
        output_region: ArenaRegion,
    ) -> Result<(), VulkanError> {
        let n_out = projection.n_out();
        let output_len = self.rows * n_out;
        if self.readback.len() < output_len {
            self.readback.resize(output_len, 0.0);
        }
        let bindings = self.binding_for(layer, projection)?;
        self.ops.write_f32(input_region, input)?;
        let mut commands = TokenCommands::begin(self.context)?;
        self.ops.record_weight_matmul_rows(
            &commands,
            bindings,
            input_region,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(output_region, n_out, n_out * 4)],
            HIDDEN,
            self.rows,
            HIDDEN,
        )?;
        commands.submit_and_wait()?;
        let values = self.ops.read_f32(output_region, output_len)?;
        self.readback[..output_len].copy_from_slice(&values[..output_len]);
        Ok(())
    }

    /// The last projection's result, valid until the next `project` call.
    pub(crate) fn readback(&self, len: usize) -> &[f32] {
        &self.readback[..len]
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }
}
