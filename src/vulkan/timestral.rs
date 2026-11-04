//! Resident F32 Transformer; scaling, patch/head projections and forecast folding stay on CPU.
use crate::models::timestral::graph::Slot;
use crate::models::timestral::network::Layout;
#[cfg(feature = "parity-trace")]
use crate::models::timestral::{trace, HD, HEADS};
use crate::models::timestral::{D, F};
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{
    ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TimestralOp, TokenCommands,
};
use crate::vulkan::{VulkanContext, VulkanError};
use std::collections::HashMap;

pub(crate) struct Transformer {
    config: super::super::models::timestral::Config,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    bindings: HashMap<String, OperatorBindings>,
    /// The weight format of every matrix, so the recorder can pick the
    /// matching device kernel.
    formats: HashMap<String, GpuWeightFormat>,
    regions: [ArenaRegion; 17],
    capacity: usize,
    context: &'static VulkanContext,
}

/// Whether the device path can run a weight format.
///
/// The K-quant formats have no vectorised kernel, and F16 has no
/// device kernel at all, so a checkpoint that mixes them in cannot run
/// on the device.
pub(crate) fn gpu_supports(format: crate::core::tensor::GGMLType) -> bool {
    !matches!(
        format,
        crate::core::tensor::GGMLType::Q4K
            | crate::core::tensor::GGMLType::Q5K
            | crate::core::tensor::GGMLType::Q6K
            | crate::core::tensor::GGMLType::F16
    )
}

impl Transformer {
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn new(
        context: &'static VulkanContext,
        config: crate::models::timestral::Config,
        matrices: &HashMap<String, crate::models::timestral::Matrix<'_>>,
        vectors: &HashMap<String, Vec<f32>>,
        capacity: usize,
    ) -> Result<Self, VulkanError> {
        let d = config.d_model;
        let f = config.mlp_hidden_dim;
        let hd = config.hd();
        let mut size = 0usize;
        let max_in = d.max(f);
        let mut regions = [ArenaRegion { offset: 0, size: 0 }; 17];
        // x, norm, fused, q, k, v, attention, gate, projection, hidden,
        // valid/roles, rope, q8 activation, q8 scales, q4_1 input sums,
        // q8k activation, q8k scales.
        for (region, width) in
            regions
                .iter_mut()
                .zip([d, d, 2 * f, d, d, d, d, d, d, f, 2, hd / 2, 0, 0, 0, 0, 0])
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
        // A quantized weight reads a Q8_0 activation, so the arena carries
        // one byte and one scale per value. The widest projection is the
        // `mlp.w3` matrix.
        let q8_bytes = capacity.checked_mul(f).ok_or(VulkanError::OutOfMemory)?;
        regions[12] = ArenaRegion {
            offset: size,
            size: q8_bytes,
        };
        size = size.checked_add(q8_bytes).ok_or(VulkanError::OutOfMemory)?;
        let q8_scale_bytes = capacity
            .checked_mul(f / 32)
            .and_then(|n| n.checked_mul(4))
            .ok_or(VulkanError::OutOfMemory)?;
        regions[13] = ArenaRegion {
            offset: size,
            size: q8_scale_bytes,
        };
        size = size
            .checked_add(q8_scale_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        // The Q8_0 quantizer also writes the per-block input sums.
        let q4_1_sum_bytes = capacity
            .checked_mul(f / 32)
            .and_then(|n| n.checked_mul(4))
            .ok_or(VulkanError::OutOfMemory)?;
        regions[14] = ArenaRegion {
            offset: size,
            size: q4_1_sum_bytes,
        };
        size = size
            .checked_add(q4_1_sum_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        // Q4_K/Q5_K/Q6_K quantize the activation to Q8_K (256-value blocks).
        let q8k_bytes = capacity
            .checked_mul(max_in)
            .ok_or(VulkanError::OutOfMemory)?;
        regions[15] = ArenaRegion {
            offset: size,
            size: q8k_bytes,
        };
        size = size
            .checked_add(q8k_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        let q8k_scale_bytes = capacity
            .checked_mul(max_in / 256)
            .and_then(|n| n.checked_mul(16))
            .ok_or(VulkanError::OutOfMemory)?;
        regions[16] = ArenaRegion {
            offset: size,
            size: q8k_scale_bytes,
        };
        size = size
            .checked_add(q8k_scale_bytes)
            .ok_or(VulkanError::OutOfMemory)?;
        // Stable order keeps descriptor/dispatch traces reproducible.
        let mut names: Vec<_> = matrices
            .keys()
            .filter(|n| n.starts_with("transformer."))
            .collect();
        names.sort();
        let mut vector_names: Vec<_> = vectors
            .keys()
            .filter(|n| n.starts_with("transformer."))
            .collect();
        vector_names.sort();
        // Every upload takes one descriptor set out of the pool.
        let mut ops = Qwen3Ops::new_device_local_with_size(
            context,
            size,
            names.len() + vector_names.len() + 2,
        )?;
        let mut buffers = UploadedBuffers::new(context);
        let mut bindings = HashMap::new();
        let mut formats = HashMap::new();
        for name in &names {
            let weight = &matrices[*name];
            let n_in = if name.ends_with(".mlp.w3.weight") {
                f
            } else {
                d
            };
            let format = GpuWeightFormat::from_ggml_type(weight.ggml_type()).map_err(|_| {
                VulkanError::UnsupportedShape(format!(
                    "Timestral GPU path does not support {:?} weights",
                    weight.ggml_type()
                ))
            })?;
            // Float/half weights are transposed because the device kernel
            // reads them column-major. Quantized tensors keep their block
            // layout and are uploaded as-is.
            let buffer = match weight.f32_slice() {
                Some(rows) => buffers.upload_f32_transposed(rows, n_in)?,
                None if matches!(format, GpuWeightFormat::F16 | GpuWeightFormat::BF16) => {
                    let bytes =
                        weight
                            .weight_bytes()
                            .ok_or(VulkanError::UnsupportedShape(format!(
                                "Timestral GPU path cannot upload {name}"
                            )))?;
                    buffers.upload_half_transposed(bytes, n_in)?
                }
                None => {
                    let bytes =
                        weight
                            .weight_bytes()
                            .ok_or(VulkanError::UnsupportedShape(format!(
                                "Timestral GPU path cannot upload {name}"
                            )))?;
                    buffers.upload(bytes)?
                }
            };
            let binding = ops.bind_weight_buffers(&[buffer], &[format])?;
            bindings.insert((*name).clone(), binding);
            formats.insert((*name).clone(), format);
        }
        // The norms travel with the weights: they are read by the same
        // dispatch that consumes the projection they scale.
        for name in &vector_names {
            let buffer = buffers.upload_f32(&vectors[name.as_str()])?;
            bindings.insert((*name).clone(), ops.bind_buffers(&[buffer])?);
        }
        Ok(Self {
            config,
            ops,
            _buffers: buffers,
            bindings,
            formats,
            regions,
            capacity,
            context,
        })
    }

    pub(crate) fn linear(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        input: ArenaRegion,
        output: ArenaRegion,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let format = self.formats[name];
        if matches!(
            format,
            GpuWeightFormat::F32 | GpuWeightFormat::BF16 | GpuWeightFormat::F16
        ) {
            return self.ops.record_transposed_float_matmul_rows(
                commands,
                self.bindings[name],
                input,
                &[(output, n_out, n_out * 4)],
                n_in,
                rows,
                n_in,
            );
        }
        // A quantized weight keeps its block layout, so the device kernel
        // reads the weight as it is stored and quantizes the activation
        // itself.
        self.ops.record_weight_matmul_rows(
            commands,
            self.bindings[name],
            input,
            self.regions[12],
            self.regions[13],
            self.regions[14],
            self.regions[15],
            self.regions[16],
            &[(output, n_out, n_out * 4)],
            n_in,
            rows,
            n_in,
        )
    }

    pub(crate) fn norm(
        &self,
        commands: &TokenCommands<'_>,
        name: &str,
        input: ArenaRegion,
        output: ArenaRegion,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let d = self.config.d_model;
        self.ops.record_rms_norm_rows(
            commands,
            self.bindings[name],
            input,
            output,
            d,
            1e-6,
            rows,
            d,
            d,
        )
    }

    /// Arena region backing a slot.
    ///
    /// The first twelve slots line up with the region table; the role tags
    /// live inside the mask region.
    fn region(&self, slot: Slot, count: usize) -> ArenaRegion {
        match slot {
            // The role tags follow the mask values inside the same region.
            Slot::Roles => self.regions[Slot::Masks as usize].shifted(count * 4),
            other => self.regions[other as usize],
        }
    }

    #[cfg(feature = "parity-trace")]
    pub(crate) fn checkpoint<'a>(
        &self,
        commands: TokenCommands<'a>,
        name: &str,
        shape: &[usize],
        region: ArenaRegion,
    ) -> Result<TokenCommands<'a>, VulkanError> {
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            commands.submit_and_wait()?;
            let mut values = vec![0.0; shape.iter().product()];
            self.ops.read_f32_into(region, &mut values)?;
            trace(name, shape, &values).map_err(VulkanError::InitFailed)?;
            return TokenCommands::begin(self.context);
        }
        let _ = (name, shape, region);
        Ok(commands)
    }

    pub(crate) fn forward(&self, input: &[f32], layout: &Layout) -> Result<Vec<f32>, VulkanError> {
        let (r, s) = (layout.rows, layout.seq());
        let count = r.checked_mul(s).ok_or(VulkanError::OutOfMemory)?;
        if count > self.capacity
            || input.len() != count * self.config.d_model
            || input.iter().any(|v| !v.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Timestral GPU input".into(),
            ));
        }
        let x = self.region(Slot::X, count);
        let roles = self.region(Slot::Roles, count);
        let before = self.ops.recorded_dispatch_count();
        self.ops.write_f32(x, input)?;
        self.ops.write_f32(
            self.region(Slot::Masks, count),
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
        let hd = self.config.hd();
        let rope_hd_half = hd / 2;
        let mut coefficients = Vec::with_capacity(s * rope_hd_half);
        for slot in 0..s {
            for i in 0..hd / 4 {
                let theta =
                    layout.position(slot) / 10000f32.powf((2 * i) as f32 / rope_hd_half as f32);
                let (sin, cos) = theta.sin_cos();
                coefficients.extend([cos, sin]);
            }
        }
        self.ops
            .write_f32(self.region(Slot::Rope, count), &coefficients)?;
        let mut commands = TokenCommands::begin(self.context)?;
        // The op list is shared with the CPU backend; only the execution
        // differs. The command buffer is flushed at layer boundaries to
        // bound its length on drivers that truncate long recordings.
        for layer in crate::models::timestral::graph::transformer_layer_ops(&self.config) {
            for op in layer {
                let geometry = [
                    r,
                    layout.enc,
                    s,
                    layout.context,
                    usize::from(layout.covariates),
                    usize::from(op.kind.is_time_axis()),
                ];
                self.execute(&commands, &op, geometry, count)?;
                #[cfg(feature = "parity-trace")]
                if std::env::var_os("RMI_PARITY_TRACE").is_some() {
                    for (name, slot) in op.trace {
                        let shape = match slot {
                            Slot::Q | Slot::K | Slot::V => {
                                vec![
                                    geometry[0],
                                    geometry[2],
                                    self.config.num_heads,
                                    self.config.hd(),
                                ]
                            }
                            _ => vec![geometry[0], geometry[2], self.config.d_model],
                        };
                        commands =
                            self.checkpoint(commands, &name, &shape, self.region(slot, count))?;
                    }
                }
            }
            commands = commands.flush()?;
        }
        let geometry = [
            r,
            layout.enc,
            s,
            layout.context,
            usize::from(layout.covariates),
            0,
        ];
        let final_norm = crate::models::timestral::graph::final_norm_op();
        self.execute(&commands, &final_norm, geometry, count)?;
        #[cfg(feature = "parity-trace")]
        if std::env::var_os("RMI_PARITY_TRACE").is_some() {
            for (name, slot) in final_norm.trace {
                let shape = vec![geometry[0], geometry[2], self.config.d_model];
                commands = self.checkpoint(commands, &name, &shape, self.region(slot, count))?;
            }
        }
        commands.submit_and_wait()?;
        let mut output = vec![0.0; count * self.config.d_model];
        self.ops
            .read_f32_into(self.region(Slot::Norm, count), &mut output)?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "non-finite Timestral GPU output".into(),
            ));
        }
        // `out_norm` is traced by the final RMSNorm op in the shared op list.
        if std::env::var_os("RMI_COMPUTE_TRACE").is_some() {
            eprintln!(
                "compute scope=timestral.transformer gpu_dispatches={}",
                self.ops.recorded_dispatch_count() - before
            );
        }
        Ok(output)
    }

    /// Record one shared op on the device.
    fn execute(
        &self,
        commands: &TokenCommands<'_>,
        op: &crate::models::timestral::graph::TimestralOp,
        geometry: [usize; 6],
        count: usize,
    ) -> Result<(), VulkanError> {
        use crate::models::timestral::graph::TimestralOpKind;
        let empty = ArenaRegion { offset: 0, size: 0 };
        match &op.kind {
            TimestralOpKind::RmsNorm {
                input,
                weight,
                output,
            } => {
                self.norm(
                    commands,
                    weight,
                    self.region(*input, count),
                    self.region(*output, count),
                    count,
                )?;
            }
            TimestralOpKind::Linear {
                input,
                weight,
                output,
                n_in,
                n_out,
            } => {
                self.linear(
                    commands,
                    weight,
                    self.region(*input, count),
                    self.region(*output, count),
                    *n_in,
                    *n_out,
                    count,
                )?;
            }
            TimestralOpKind::Qkv {
                fused,
                q,
                k,
                v,
                rope,
                ..
            } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Qkv,
                    [
                        self.region(*fused, count),
                        self.region(*q, count),
                        self.region(*k, count),
                        self.region(*v, count),
                        self.region(*rope, count),
                        empty,
                    ],
                    geometry,
                    self.config.d_model,
                    self.config.mlp_hidden_dim,
                    self.config.num_heads,
                    self.config.hd(),
                )?;
            }
            TimestralOpKind::Attention {
                q,
                k,
                v,
                out,
                masks,
                roles,
                ..
            } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Attention,
                    [
                        self.region(*q, count),
                        self.region(*k, count),
                        self.region(*v, count),
                        self.region(*out, count),
                        self.region(*masks, count),
                        self.region(*roles, count),
                    ],
                    geometry,
                    self.config.d_model,
                    self.config.mlp_hidden_dim,
                    self.config.num_heads,
                    self.config.hd(),
                )?;
            }
            TimestralOpKind::Gate { attn, gate, .. } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::Gate,
                    [
                        self.region(*attn, count),
                        self.region(*gate, count),
                        empty,
                        empty,
                        empty,
                        empty,
                    ],
                    geometry,
                    self.config.d_model,
                    self.config.mlp_hidden_dim,
                    self.config.num_heads,
                    self.config.hd(),
                )?;
            }
            TimestralOpKind::SwiGlu { fused, hidden } => {
                self.ops.record_timestral(
                    commands,
                    TimestralOp::SwiGlu,
                    [
                        self.region(*fused, count),
                        self.region(*hidden, count),
                        empty,
                        empty,
                        empty,
                        empty,
                    ],
                    geometry,
                    self.config.d_model,
                    self.config.mlp_hidden_dim,
                    self.config.num_heads,
                    self.config.hd(),
                )?;
            }
            TimestralOpKind::Add { input, output } => {
                self.ops.record_add_rows(
                    commands,
                    self.region(*output, count),
                    self.region(*input, count),
                    self.config.d_model,
                    count,
                )?;
            }
        }
        Ok(())
    }
}
