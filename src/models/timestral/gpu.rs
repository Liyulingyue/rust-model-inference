//! Resident F32 Transformer; scaling, patch/head projections and forecast folding stay on CPU.
use super::network::Layout;
use super::*;
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{
    ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TimestralOp, TokenCommands,
};
use crate::vulkan::{VulkanContext, VulkanError};

pub(super) struct Transformer {
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    bindings: HashMap<String, OperatorBindings>,
    regions: [ArenaRegion; 12],
    capacity: usize,
    context: &'static VulkanContext,
}

impl Transformer {
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn new(
        context: &'static VulkanContext,
        weights: &HashMap<String, &[f32]>,
        capacity: usize,
    ) -> Result<Self, VulkanError> {
        let mut size = 0usize;
        let mut regions = [ArenaRegion { offset: 0, size: 0 }; 12];
        // x, norm, fused, q, k, v, attention, gate, projection, hidden, valid/roles, rope.
        for (region, width) in regions
            .iter_mut()
            .zip([D, D, 2 * F, D, D, D, D, D, D, F, 2, 32])
        {
            let bytes = capacity
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or(VulkanError::OutOfMemory)?;
            *region = ArenaRegion {
                offset: size,
                size: bytes,
            };
            size = size.checked_add(bytes).ok_or(VulkanError::OutOfMemory)?;
        }
        let mut ops = Qwen3Ops::new_with_size(context, size, weights.len() + 1)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut bindings = HashMap::new();
        // Stable order keeps descriptor/dispatch traces reproducible.
        let mut names: Vec<_> = weights
            .keys()
            .filter(|n| n.starts_with("transformer."))
            .collect();
        names.sort();
        for name in names {
            let matrix = name.ends_with(".weight");
            let buffer = if matrix {
                let n_in = if name.ends_with(".mlp.w3.weight") {
                    F
                } else {
                    D
                };
                buffers.upload_f32_transposed(weights[name], n_in)?
            } else {
                buffers.upload_f32(weights[name])?
            };
            let binding = if matrix {
                ops.bind_weight_buffers(&[buffer], &[GpuWeightFormat::F32])?
            } else {
                ops.bind_buffers(&[buffer])?
            };
            bindings.insert(name.clone(), binding);
        }
        Ok(Self {
            ops,
            _buffers: buffers,
            bindings,
            regions,
            capacity,
            context,
        })
    }

    fn linear(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        input: ArenaRegion,
        output: ArenaRegion,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.ops.record_transposed_float_matmul_rows(
            commands,
            self.bindings[&format!("{name}.weight")],
            input,
            &[(output, n_out, n_out * 4)],
            n_in,
            rows,
            n_in,
        )
    }

    fn norm(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.ops.record_rms_norm_rows(
            commands,
            self.bindings[&format!("{name}.scale")],
            self.regions[0],
            self.regions[1],
            D,
            1e-6,
            rows,
            D,
            D,
        )
    }

    fn checkpoint<'a>(
        &self,
        commands: TokenCommands<'a>,
        name: &str,
        shape: &[usize],
        region: ArenaRegion,
    ) -> Result<TokenCommands<'a>, VulkanError> {
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            let commands = commands.flush()?;
            let values = self.ops.read_f32(region, shape.iter().product())?;
            trace(name, shape, values).map_err(VulkanError::InitFailed)?;
            return Ok(commands);
        }
        let _ = (name, shape, region);
        Ok(commands)
    }

    pub(super) fn forward(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>, VulkanError> {
        let (r, s) = (layout.rows, layout.seq());
        let count = r.checked_mul(s).ok_or(VulkanError::OutOfMemory)?;
        if count > self.capacity || input.len() != count * D || input.iter().any(|v| !v.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Timestral GPU input".into(),
            ));
        }
        let [x, norm, fused, q, k, v, attn, gate, projection, hidden, masks, rope] = self.regions;
        let roles = masks.shifted(count * 4);
        let before = self.ops.recorded_dispatch_count();
        let mut commands = TokenCommands::begin(self.context)?;
        self.ops.write_f32(x, input)?;
        self.ops.write_f32(
            masks,
            &layout
                .valid
                .iter()
                .map(|&b| if b { 1. } else { 0. })
                .collect::<Vec<_>>(),
        )?;
        self.ops.write_f32(
            roles,
            &layout
                .roles
                .iter()
                .map(|&b| f32::from(b))
                .collect::<Vec<_>>(),
        )?;
        let mut coefficients = Vec::with_capacity(s * 32);
        for slot in 0..s {
            for i in 0..16 {
                let theta = layout.position(slot) / 10000f32.powf((2 * i) as f32 / 32.);
                let (sin, cos) = theta.sin_cos();
                coefficients.extend([cos, sin]);
            }
        }
        self.ops.write_f32(rope, &coefficients)?;
        let empty = ArenaRegion { offset: 0, size: 0 };
        for layer in 0..24 {
            let p = format!("transformer.layers.{layer}");
            for axis in 0..if layer % 4 == 3 { 2 } else { 1 } {
                let label = format!("layer.{layer}.attn.{axis}");
                let prefix = format!("{p}.attns.{axis}");
                self.norm(&commands, &format!("{p}.attn_norms.{axis}"), count)?;
                commands = self.checkpoint(commands, &format!("{label}.norm"), &[r, s, D], norm)?;
                self.linear(
                    &commands,
                    &format!("{prefix}.wQKV"),
                    norm,
                    fused,
                    D,
                    3 * D,
                    count,
                )?;
                let geometry = [
                    r,
                    layout.enc,
                    s,
                    layout.context,
                    usize::from(layout.covariates),
                    usize::from(axis == 0),
                ];
                self.ops.record_timestral(
                    &commands,
                    TimestralOp::Qkv,
                    [fused, q, k, v, rope, empty],
                    geometry,
                )?;
                for (name, region) in [("q", q), ("k", k), ("v", v)] {
                    commands = self.checkpoint(
                        commands,
                        &format!("{label}.{name}"),
                        &[r, s, HEADS, HD],
                        region,
                    )?;
                }
                self.ops.record_timestral(
                    &commands,
                    TimestralOp::Attention,
                    [q, k, v, attn, masks, roles],
                    geometry,
                )?;
                self.linear(
                    &commands,
                    &format!("{prefix}._gate"),
                    norm,
                    gate,
                    D,
                    D,
                    count,
                )?;
                self.ops.record_timestral(
                    &commands,
                    TimestralOp::Gate,
                    [attn, gate, empty, empty, empty, empty],
                    geometry,
                )?;
                self.linear(
                    &commands,
                    &format!("{prefix}.wO"),
                    attn,
                    projection,
                    D,
                    D,
                    count,
                )?;
                commands =
                    self.checkpoint(commands, &format!("{label}.output"), &[r, s, D], projection)?;
                self.ops
                    .record_add_rows(&commands, x, projection, D, count)?;
            }
            self.norm(&commands, &format!("{p}.mlp_norm"), count)?;
            self.linear(
                &commands,
                &format!("{p}.mlp.w12"),
                norm,
                fused,
                D,
                2 * F,
                count,
            )?;
            self.ops.record_timestral(
                &commands,
                TimestralOp::SwiGlu,
                [fused, hidden, empty, empty, empty, empty],
                [
                    r,
                    layout.enc,
                    s,
                    layout.context,
                    usize::from(layout.covariates),
                    0,
                ],
            )?;
            self.linear(
                &commands,
                &format!("{p}.mlp.w3"),
                hidden,
                projection,
                F,
                D,
                count,
            )?;
            self.ops
                .record_add_rows(&commands, x, projection, D, count)?;
            commands =
                self.checkpoint(commands, &format!("layer.{layer}.output"), &[r, s, D], x)?;
            // Bound command-buffer length on drivers that truncate long recordings.
            commands = commands.flush()?;
        }
        self.norm(&commands, "transformer.out_norm", count)?;
        commands.submit_and_wait()?;
        let output = self.ops.read_f32(norm, count * D)?.to_vec();
        if output.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "non-finite Timestral GPU output".into(),
            ));
        }
        trace("out_norm", &[r, s, D], &output).map_err(VulkanError::InitFailed)?;
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            eprintln!(
                "compute scope=timestral.transformer gpu_dispatches={}",
                self.ops.recorded_dispatch_count() - before
            );
        }
        Ok(output)
    }
}
