use super::*;
use crate::ops::kernel::trellis::TrellisWeight;
use crate::vulkan::qwen3::UploadedBuffers;

// Each recorded chunk uses this many disjoint rows in the scratch allocation.
const CHUNK: usize = 8;
const SHADER: &[u8] = include_bytes!("../../../shaders/bin/trellis.spv");

struct BandBinding {
    bindings: OperatorBindings,
    cols: usize,
    rows: usize,
    bits: u32,
    in_block: u32,
    out_block: u32,
}

#[derive(Default)]
pub(super) struct TrellisBindings {
    pipeline: Option<usize>,
    // The operations are serialized in one command buffer and can reuse scratch.
    scratch: HashMap<(usize, usize), GpuBuffer>,
    projections: HashMap<u64, Vec<BandBinding>>,
}

impl Qwen3Ops<'_> {
    pub(crate) fn bind_trellis(
        &mut self,
        buffers: &mut UploadedBuffers,
        weight: &TrellisWeight<'_>,
    ) -> Result<OperatorBindings, VulkanError> {
        if self.trellis.pipeline.is_none() {
            self.trellis.pipeline = Some(self.add_shader(SHADER)?);
        }
        let mut bands = Vec::with_capacity(weight.bands.len());
        for band in &weight.bands {
            let k = band.pre.len();
            if band.in_block.max(band.out_block) > 4096 {
                return Err(VulkanError::UnsupportedShape(
                    "NariQ Vulkan supports block Hadamards up to 4096".into(),
                ));
            }
            let scratch = match self.trellis.scratch.entry((k, band.rows)) {
                std::collections::hash_map::Entry::Occupied(slot) => *slot.get(),
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let bytes = k
                        .checked_add(band.rows)
                        .and_then(|n| n.checked_mul(CHUNK * 4))
                        .ok_or(VulkanError::OutOfMemory)?;
                    *slot.insert(buffers.upload(&vec![0; bytes])?)
                }
            };
            let mut sides = Vec::new();
            sides.extend(
                band.pre
                    .iter()
                    .chain(&band.s16)
                    .chain(&band.post)
                    .map(|v| v.to_bits()),
            );
            if let Some(pin) = &band.pin {
                sides.extend(pin);
            } else {
                sides.extend(0..as_u32(k, "NariQ width")?);
            }
            sides.extend(&band.scatter);
            let codes = buffers.upload(band.codes)?;
            let transforms = buffers.upload(bytemuck::cast_slice(&sides))?;
            let bindings = self.bind_buffers(&[codes, transforms, scratch])?;
            bands.push(BandBinding {
                bindings,
                cols: k,
                rows: band.rows,
                bits: band.bits() as u32,
                in_block: band.in_block as u32,
                out_block: band.out_block as u32,
            });
        }
        let first = bands[0].bindings;
        self.trellis
            .projections
            .insert(first.descriptor_set_handle(), bands);
        Ok(first)
    }

    pub(super) fn record_trellis(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        outputs: &[(ArenaRegion, usize, usize)],
        k: usize,
        rows: usize,
        input_stride: usize,
    ) -> Result<bool, VulkanError> {
        let Some(pipeline) = self.trellis.pipeline else {
            return Ok(false);
        };
        let Some(bands) = self
            .trellis
            .projections
            .get(&bindings.descriptor_set_handle())
        else {
            return Ok(false);
        };
        let &[(output, n, out_bytes)] = outputs else {
            return Err(VulkanError::UnsupportedShape(
                "NariQ binding requires one output".into(),
            ));
        };
        let output_stride = out_bytes / 4;
        if rows == 0
            || input_stride < k
            || out_bytes % 4 != 0
            || output_stride < n
            || bands.iter().any(|band| band.cols != k)
            || bands.iter().map(|b| b.rows).sum::<usize>() != n
        {
            return Err(VulkanError::UnsupportedShape(
                "NariQ projection shape mismatch".into(),
            ));
        }
        let count = |stride: usize, width: usize| {
            (rows - 1)
                .checked_mul(stride)
                .and_then(|s| s.checked_add(width))
                .ok_or(VulkanError::OutOfMemory)
        };
        let input_word = self.f32_word(input, count(input_stride, k)?, "NariQ input")?;
        let output_word = self.f32_word(output, count(output_stride, n)?, "NariQ output")?;
        for band in bands {
            for first in (0..rows).step_by(CHUNK) {
                let mut push = [
                    0,
                    input_word,
                    output_word,
                    as_u32(input_stride, "NariQ input stride")?,
                    as_u32(output_stride, "NariQ output stride")?,
                    as_u32(first, "NariQ first token")?,
                    as_u32(k, "NariQ width")?,
                    as_u32(band.rows, "NariQ band rows")?,
                    band.bits,
                    band.in_block,
                    band.out_block,
                ];
                let span = |block: u32| if block == 1 { 64 } else { block };
                for (mode, groups) in [
                    push[6].div_ceil(span(band.in_block)),
                    push[7] / 16,
                    push[7].div_ceil(span(band.out_block)),
                ]
                .into_iter()
                .enumerate()
                {
                    push[0] = mode as u32;
                    let grid = [groups, (rows - first).min(CHUNK) as u32, 1];
                    self.record_custom(commands, pipeline, Some(band.bindings), &push, grid)?;
                }
            }
        }
        Ok(true)
    }
}
