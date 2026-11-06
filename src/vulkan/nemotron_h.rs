//! Vulkan resident session for the dense `nemotron_h` trunk
//! (Nemotron-3 Nano 4B).
//!
//! The 4B is a hybrid Mamba-2 / attention tower: 42 layers made of 21
//! SSM-only layers, 17 FFN-only layers and 4 attention-only layers. Every
//! layer still carries a pre-norm on its input, and the three branch types
//! are mutually exclusive per layer.
//!
//! This module follows the shape of `qwen35.rs` (the other hybrid SSM
//! trunk) but the semantics differ in three places that matter:
//!
//! * **SSM**: Nemotron uses canonical Mamba-2 (`state = state * exp(dt*A) +
//!   B*x*dt`, `y = state·C + D*x`, then `silu(z)*y`, then a group RMSNorm),
//!   not Qwen3.5's recurrent form (`beta`/`alpha` gates, RMS-norm'd Q/K).
//!   The Qwen3.5 SSM shaders cannot be reused.
//! * **Attention**: Nemotron's attention has no output gate; the Qwen3.5
//!   attention shader applies a SiLU gate after the value mix.
//! * **FFN**: Nemotron uses `ReLU(x)^2` (LLM_FFN_RELU_SQR), not SwiGLU.
//!
//! Stage 1 of the port keeps the SSM branch on the CPU: the session records
//! the FFN and attention layers on the device and falls back to the existing
//! `forward_layer` SSM path for the 21 SSM layers. Stage 2 adds the Mamba-2
//! conv/scan shaders and moves the whole trunk onto the device.

use crate::models::nemotron_h::trunk::config::NemotronConfig;
use crate::models::nemotron_h::trunk::forward::NemotronModel;
use crate::ops::kernel::Weight;
use crate::vulkan::dense::UploadedBuffers;
use crate::vulkan::ops::{
    row_dispatch, ArenaRegion, GpuMatmulSpec, GpuWeightFormat, OperatorBindings, Qwen3Ops,
    TokenCommands,
};
use crate::vulkan::{GpuBuffer, VulkanContext, VulkanError};

/// Bindings for one attention layer (Q/K/V/O projections).
#[derive(Clone, Copy)]
struct AttentionBindings {
    q: OperatorBindings,
    k: OperatorBindings,
    v: OperatorBindings,
    /// Q/K RMSNorm weights packed as `[q_norm, k_norm]`. `None` when the
    /// layer has no Q/K norm (the 4B Nano attention layers ship neither).
    qk_norm: Option<OperatorBindings>,
    output: OperatorBindings,
}

/// Bindings for one FFN layer (up/down projections).
#[derive(Clone, Copy)]
struct FfnBindings {
    up: OperatorBindings,
    down: OperatorBindings,
}

enum LayerBindings {
    Attention {
        norm: OperatorBindings,
        attention: AttentionBindings,
    },
    Ffn {
        norm: OperatorBindings,
        ffn: FfnBindings,
    },
    /// SSM-only layer. Stage 1 runs it on the CPU; stage 2 replaces this
    /// with device bindings.
    Ssm,
}

/// Arena layout for the Nemotron-3 Nano device session.
///
/// The KV cache is one contiguous region covering every attention layer,
/// because the KV ops index it by layer; `slots` maps a global layer index
/// to its position within that region.
struct NemotronHArenaLayout {
    x: ArenaRegion,
    normed: ArenaRegion,
    q: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    attn: ArenaRegion,
    projection: ArenaRegion,
    ffn_up: ArenaRegion,
    ffn_down: ArenaRegion,
    logits: ArenaRegion,
    rope: ArenaRegion,
    scores: ArenaRegion,
    q8: ArenaRegion,
    q8_scales: ArenaRegion,
    q4_1_input_sums: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
    kv_k: ArenaRegion,
    kv_v: ArenaRegion,
    kv_delta_k: ArenaRegion,
    kv_delta_v: ArenaRegion,
    slots: Vec<Option<usize>>,
    total_size: usize,
}

fn product(label: &str, values: &[usize]) -> Result<usize, VulkanError> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            VulkanError::UnsupportedShape(format!("Nemotron-H {label} size overflows usize"))
        })
    })
}

fn f32_region(cursor: &mut usize, elements: usize) -> Result<ArenaRegion, VulkanError> {
    let bytes = elements.checked_mul(4).ok_or(VulkanError::OutOfMemory)?;
    region(cursor, bytes)
}

fn region(cursor: &mut usize, size: usize) -> Result<ArenaRegion, VulkanError> {
    let offset = cursor
        .checked_add(15)
        .map(|value| value & !15)
        .ok_or(VulkanError::OutOfMemory)?;
    *cursor = offset.checked_add(size).ok_or(VulkanError::OutOfMemory)?;
    Ok(ArenaRegion { offset, size })
}

impl NemotronHArenaLayout {
    fn new(config: &NemotronConfig, capacity: usize, rows: usize) -> Result<Self, VulkanError> {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        let n_head_kv = config.n_head_kv;
        let head_dim_k = config.n_embd_head_k;
        let head_dim_v = config.n_embd_head_v;
        let n_attn_q = n_head * head_dim_k;
        let n_attn_kv = n_head_kv * head_dim_k;
        let n_attn_v_kv = n_head_kv * head_dim_v;
        let n_ff = config.n_ff;
        if [
            n_embd,
            config.n_layer,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff,
            capacity,
            rows,
            config.rope_dim,
            config.vocab_size,
        ]
        .contains(&0)
            || !n_head.is_multiple_of(n_head_kv)
            || config.rope_dim > head_dim_k
            || !config.rope_dim.is_multiple_of(2)
        {
            return Err(VulkanError::UnsupportedShape(
                "invalid Nemotron-H Vulkan arena dimensions".into(),
            ));
        }
        let quant_len = n_embd.max(n_ff).max(n_attn_q).max(n_attn_v_kv);
        let kv_cache = product(
            "KV cache",
            &[config.attention_layers.len(), capacity, n_attn_kv],
        )?;
        let kv_delta = product(
            "KV delta",
            &[config.attention_layers.len(), rows, n_attn_kv],
        )?;
        let mut cursor = 0usize;
        let mut row_region =
            |width| f32_region(&mut cursor, product("row buffer", &[rows, width])?);
        let x = row_region(n_embd)?;
        let normed = row_region(n_embd)?;
        let q = row_region(n_attn_q)?;
        let k = row_region(n_attn_kv)?;
        let v = row_region(n_attn_v_kv)?;
        let attn = row_region(n_attn_q)?;
        let projection = row_region(n_embd)?;
        let ffn_up = row_region(n_ff)?;
        let ffn_down = row_region(n_embd)?;
        let rope = row_region(config.rope_dim)?;
        let scores = f32_region(
            &mut cursor,
            product("attention scores", &[rows, n_head, capacity])?,
        )?;
        let logits = f32_region(&mut cursor, config.vocab_size)?;
        let quant_rows = product("quantized rows", &[rows, quant_len.div_ceil(4) * 4])?;
        let q8 = region(&mut cursor, quant_rows)?;
        let q8_scales = f32_region(
            &mut cursor,
            product("Q8 scales", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q4_1_input_sums = f32_region(
            &mut cursor,
            product("Q4 sums", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q8k = region(&mut cursor, quant_rows)?;
        let q8k_scales = f32_region(
            &mut cursor,
            product("Q8K scales", &[rows, quant_len.div_ceil(256)])?,
        )?;
        let kv_k = f32_region(&mut cursor, kv_cache)?;
        let kv_v = f32_region(&mut cursor, kv_cache)?;
        let kv_delta_k = f32_region(&mut cursor, kv_delta)?;
        let kv_delta_v = f32_region(&mut cursor, kv_delta)?;
        let mut slots = vec![None; config.n_layer];
        for (layer, slot) in slots.iter_mut().enumerate() {
            if config.attention_layers.contains(&layer) {
                *slot = Some(
                    config
                        .attention_layers
                        .iter()
                        .position(|&l| l == layer)
                        .expect("attention layer is in the list"),
                );
            }
        }
        Ok(Self {
            x,
            normed,
            q,
            k,
            v,
            attn,
            projection,
            ffn_up,
            ffn_down,
            logits,
            rope,
            scores,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            kv_k,
            kv_v,
            kv_delta_k,
            kv_delta_v,
            slots,
            total_size: cursor,
        })
    }
}

pub(crate) struct NemotronHVulkanSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    layers: Vec<LayerBindings>,
    output_norm: OperatorBindings,
    output: OperatorBindings,
    layout: NemotronHArenaLayout,
    config: NemotronConfig,
    pub(crate) capacity: usize,
    max_rows: usize,
    rope: Vec<f32>,
    logits: Vec<f32>,
    /// Pipeline for the `ReLU(x)^2` activation, added at construction time.
    relu_square_pipeline: usize,
}

fn required_weight<'a>(
    weight: Option<&'a Weight<'a>>,
    layer: usize,
    name: &str,
) -> Result<&'a Weight<'a>, VulkanError> {
    weight.ok_or_else(|| {
        VulkanError::UnsupportedShape(format!("missing Nemotron-H layer {layer} {name} weight"))
    })
}

fn upload_weight(
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<(GpuBuffer, GpuMatmulSpec), VulkanError> {
    let format = GpuMatmulSpec::prepared(weight)?;
    let bytes = weight.kernel.weight_bytes().ok_or_else(|| {
        VulkanError::UnsupportedShape(format!(
            "{label} does not expose {:?} storage bytes",
            weight.ggml_type
        ))
    })?;
    Ok((buffers.upload(bytes)?, format))
}

fn bind_weight(
    ops: &mut Qwen3Ops<'_>,
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<OperatorBindings, VulkanError> {
    let (buffer, format) = upload_weight(buffers, weight, label)?;
    ops.bind_weight_buffers_mode(
        std::slice::from_ref(&buffer),
        std::slice::from_ref(&format.format),
        format.mode,
    )
}

fn check_eligibility(config: &NemotronConfig, model: &NemotronModel) -> Result<(), String> {
    if config.moe.is_some() {
        return Err("MoE variant is not supported by the dense Vulkan path".into());
    }
    if model.layers.len() != config.n_layer {
        return Err("layer count mismatch".into());
    }
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let is_attention = config.attention_layers.contains(&layer_index);
        let is_ffn = config.ffn_layers.contains(&layer_index);
        if is_attention {
            for weight in [&layer.wq, &layer.wk, &layer.wv, &layer.wo] {
                if let Some(weight) = weight {
                    GpuWeightFormat::from_ggml_type(weight.ggml_type)
                        .map_err(|error| error.to_string())?;
                } else {
                    return Err(format!("layer {layer_index} missing attention weight"));
                }
            }
        } else if is_ffn {
            for weight in [&layer.w_up, &layer.w_down] {
                if let Some(weight) = weight {
                    GpuWeightFormat::from_ggml_type(weight.ggml_type)
                        .map_err(|error| error.to_string())?;
                } else {
                    return Err(format!("layer {layer_index} missing FFN weight"));
                }
            }
        }
    }
    Ok(())
}

impl NemotronHVulkanSession {
    pub(crate) fn try_new_rows(
        model: &NemotronModel,
        capacity: usize,
        max_rows: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        if let Err(reason) = check_device_eligibility(context.supports_shader_float16())
            .and_then(|()| check_eligibility(&model.config, model))
        {
            eprintln!("[GPU] Nemotron-H Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        let config = model.config.clone();
        let layout = NemotronHArenaLayout::new(&config, capacity, max_rows)?;
        let descriptor_capacity = model.layers.iter().try_fold(3usize, |count, _| {
            count.checked_add(6).ok_or(VulkanError::OutOfMemory)
        })?;
        let mut ops =
            Qwen3Ops::new_device_local_with_size(context, layout.total_size, descriptor_capacity)?;
        let relu_square_pipeline =
            ops.add_shader(include_bytes!("../../shaders/bin/relu_square.spv"))?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(config.n_layer);

        for (layer_index, layer) in model.layers.iter().enumerate() {
            let norm_buffer = buffers.upload_f32(&layer.attn_norm)?;
            let norm = ops.bind_buffers(&[norm_buffer])?;
            let is_attention = config.attention_layers.contains(&layer_index);
            let is_ffn = config.ffn_layers.contains(&layer_index);
            if is_attention {
                let wq = required_weight(layer.wq.as_ref(), layer_index, "attn_q")?;
                let wk = required_weight(layer.wk.as_ref(), layer_index, "attn_k")?;
                let wv = required_weight(layer.wv.as_ref(), layer_index, "attn_v")?;
                let wo = required_weight(layer.wo.as_ref(), layer_index, "attn_output")?;
                let q = bind_weight(&mut ops, &mut buffers, wq, "Nemotron-H Q")?;
                let k = bind_weight(&mut ops, &mut buffers, wk, "Nemotron-H K")?;
                let v = bind_weight(&mut ops, &mut buffers, wv, "Nemotron-H V")?;
                // Q/K norm weights are optional and packed into one
                // descriptor set: the shader reads Q from binding 0 and
                // K from binding 1.
                let qk_norm = match (layer.attn_q_norm.as_ref(), layer.attn_k_norm.as_ref()) {
                    (Some(qn), Some(kn)) => {
                        let q_buffer = buffers.upload_f32(qn)?;
                        let k_buffer = buffers.upload_f32(kn)?;
                        Some(ops.bind_buffers(&[q_buffer, k_buffer])?)
                    }
                    (None, None) => None,
                    _ => {
                        return Err(VulkanError::UnsupportedShape(format!(
                            "Nemotron-H layer {layer_index} has only one of Q/K norm"
                        )))
                    }
                };
                let output =
                    bind_weight(&mut ops, &mut buffers, wo, "Nemotron-H attention output")?;
                layers.push(LayerBindings::Attention {
                    norm,
                    attention: AttentionBindings {
                        q,
                        k,
                        v,
                        qk_norm,
                        output,
                    },
                });
            } else if is_ffn {
                let w_up = required_weight(layer.w_up.as_ref(), layer_index, "ffn_up")?;
                let w_down = required_weight(layer.w_down.as_ref(), layer_index, "ffn_down")?;
                let up = bind_weight(&mut ops, &mut buffers, w_up, "Nemotron-H FFN up")?;
                let down = bind_weight(&mut ops, &mut buffers, w_down, "Nemotron-H FFN down")?;
                layers.push(LayerBindings::Ffn {
                    norm,
                    ffn: FfnBindings { up, down },
                });
            } else {
                layers.push(LayerBindings::Ssm);
            }
        }

        let output_norm_buffer = buffers.upload_f32(&model.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output = bind_weight(&mut ops, &mut buffers, &model.output, "Nemotron-H output")?;
        let rope_dim = config.rope_dim;

        Ok(Some(Self {
            context,
            ops,
            _buffers: buffers,
            layers,
            output_norm,
            output,
            layout,
            config,
            capacity,
            max_rows,
            rope: vec![0.0; product("RoPE rows", &[rope_dim, max_rows])?],
            logits: vec![0.0; model.config.vocab_size],
            relu_square_pipeline,
        }))
    }

    /// Run a contiguous range of device-recorded layers over the given
    /// hidden-state rows. Stage 1 uses this to process the FFN/attention
    /// runs between the CPU SSM layers; stage 2 calls it once with the
    /// whole stack.
    ///
    /// The hidden state is written to and read back from the arena, so the
    /// host can run the SSM layers in between two calls.
    pub(crate) fn forward_layers(
        &mut self,
        input: &[f32],
        base_position: usize,
        positions: &[usize],
        rows: usize,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let input_count = product("input rows", &[rows, n_embd])?;
        if rows > self.max_rows
            || positions.len() != rows
            || input.len() != input_count
            || input.iter().any(|v| !v.is_finite())
            || layer_start >= layer_end
            || layer_end > config.n_layer
        {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Nemotron-H token input={} position={base_position}/{} layers={layer_start}..{layer_end}",
                input.len(),
                self.capacity
            )));
        }
        self.ops.write_f32(self.layout.x, input)?;
        // RoPE is only applied when the layer has Q/K norm weights; the
        // 4B Nano attention layers ship neither, so the rope buffer is
        // left untouched for them.
        let has_rope = self
            .layers
            .iter()
            .any(|layer| matches!(layer, LayerBindings::Attention { attention, .. } if attention.qk_norm.is_some()));
        if has_rope && config.rope_dim > 0 {
            for (row, &position) in positions.iter().enumerate() {
                let coeff = &mut self.rope[row * config.rope_dim..(row + 1) * config.rope_dim];
                coeff[..config.rope_dim / 2].fill(1.0);
                crate::ops::rope_neox_inplace_precise(
                    coeff,
                    position,
                    config.rope_dim,
                    config.rope_freq_base,
                );
            }
            self.ops
                .write_f32(self.layout.rope, &self.rope[..rows * config.rope_dim])?;
        }
        let commands = TokenCommands::begin(self.context)?;
        for layer_index in layer_start..layer_end {
            let bindings = &self.layers[layer_index];
            match bindings {
                LayerBindings::Attention { norm, attention } => {
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    self.record_qkv(&commands, attention, rows)?;
                    self.record_attention(&commands, attention, layer_index, base_position, rows)?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.projection,
                        n_embd,
                        rows,
                    )?;
                }
                LayerBindings::Ffn { norm, ffn } => {
                    self.ops.record_rms_norm_rows(
                        &commands,
                        *norm,
                        self.layout.x,
                        self.layout.normed,
                        n_embd,
                        config.norm_eps,
                        rows,
                        n_embd,
                        n_embd,
                    )?;
                    self.record_ffn(&commands, ffn, rows)?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.ffn_down,
                        n_embd,
                        rows,
                    )?;
                }
                LayerBindings::Ssm => {
                    // Stage 1: SSM layers run on the CPU. The device path
                    // cannot proceed without them, so this is a hard error
                    // for now; stage 2 replaces it with device shaders.
                    return Err(VulkanError::UnsupportedShape(
                        "Nemotron-H SSM layers are not yet recorded on the device".into(),
                    ));
                }
            }
        }
        commands.submit_and_wait()?;
        Ok(())
    }

    /// Read the hidden state back from the arena after a `forward_layers`
    /// call, so the host can run the CPU SSM layers in between.
    pub(crate) fn read_hidden(&self, rows: usize) -> Result<Vec<f32>, VulkanError> {
        let mut hidden = vec![0.0; rows * self.config.n_embd];
        self.ops.read_f32_into(self.layout.x, &mut hidden)?;
        Ok(hidden)
    }

    /// The logits produced by the last `finish_logits` call.
    pub(crate) fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Write the hidden state back into the arena before the next
    /// `forward_layers` call.
    pub(crate) fn write_hidden(&self, hidden: &[f32]) -> Result<(), VulkanError> {
        self.ops.write_f32(self.layout.x, hidden)
    }

    /// Run the final norm and the output projection, returning the logits.
    pub(crate) fn finish_logits(&mut self, rows: usize) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let last_offset = (rows - 1) * n_embd * 4;
        let last = ArenaRegion {
            offset: self.layout.x.offset + last_offset,
            size: self.layout.x.size - last_offset,
        };
        let commands = TokenCommands::begin(self.context)?;
        self.ops.record_rms_norm(
            &commands,
            self.output_norm,
            last,
            self.layout.normed,
            n_embd,
            config.norm_eps,
        )?;
        commands.submit_and_wait()?;
        let commands = TokenCommands::begin(self.context)?;
        self.record_weights(
            &commands,
            self.output,
            self.layout.normed,
            &[(self.layout.logits, config.vocab_size)],
            n_embd,
            1,
        )?;
        commands.submit_and_wait()?;
        self.ops
            .read_f32_into(self.layout.logits, &mut self.logits)?;
        if self.logits.iter().any(|v| !v.is_finite()) {
            return Err(VulkanError::UnsupportedShape(
                "Nemotron-H Vulkan chunk produced non-finite logits".into(),
            ));
        }
        Ok(())
    }

    fn record_qkv(
        &self,
        commands: &TokenCommands<'_>,
        attention: &AttentionBindings,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let n_attn_v_kv = config.n_head_kv * config.n_embd_head_v;
        // Three separate single-weight matmuls rather than one grouped
        // three-weight matmul. The grouped kernel writes each output with
        // its own row count, and the Q/K/V widths differ (5120/1024/1024),
        // so the split keeps every output on the plain one-weight path.
        for (bindings, region, n_out) in [
            (attention.q, self.layout.q, n_attn_q),
            (attention.k, self.layout.k, n_attn_kv),
            (attention.v, self.layout.v, n_attn_v_kv),
        ] {
            self.ops.record_weight_matmul_rows(
                commands,
                bindings,
                self.layout.normed,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                &[(region, n_out, n_out * 4)],
                n_embd,
                rows,
                n_embd,
            )?;
        }
        Ok(())
    }

    fn record_attention(
        &self,
        commands: &TokenCommands<'_>,
        attention: &AttentionBindings,
        layer_index: usize,
        base_position: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_attn_q = config.n_head * config.n_embd_head_k;
        let n_attn_kv = config.n_head_kv * config.n_embd_head_k;
        let slot = self.layout.slots[layer_index].ok_or_else(|| {
            VulkanError::UnsupportedShape("attention layer has no KV cache slot".into())
        })?;
        let layer_count = self.config.attention_layers.len();
        let cache_k = self.layout.kv_k;
        let cache_v = self.layout.kv_v;
        // Q/K norm + RoPE. The 4B Nano attention layers ship neither, so
        // both are skipped when the weights are absent.
        if let Some(qk_norm) = attention.qk_norm {
            self.ops.record_qk_norm_rope_rows(
                commands,
                qk_norm,
                self.layout.q,
                self.layout.k,
                config.n_head,
                config.n_head_kv,
                config.n_embd_head_k,
                self.layout.rope,
                config.norm_eps,
                true,
                true,
                rows,
            )?;
        }
        // The KV cache keeps F32. llama.cpp's default cache type is F16, but
        // this model's reference harness runs with an F32 cache, and the F16
        // rounding is amplified by the Mamba-2 layers that follow each
        // attention layer.
        self.ops.record_kv_write_rows_f32(
            commands,
            self.layout.k,
            self.layout.v,
            cache_k,
            cache_v,
            self.layout.kv_delta_k,
            self.layout.kv_delta_v,
            slot,
            base_position,
            layer_count,
            self.capacity,
            n_attn_kv,
            rows,
            n_attn_kv,
        )?;
        // Scores, softmax and values all stay in F32, matching the CPU path.
        self.ops.record_attention_scores_f32_rows(
            commands,
            self.layout.q,
            cache_k,
            self.layout.scores,
            slot,
            layer_count,
            base_position,
            self.capacity,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            rows,
        )?;
        self.ops.record_softmax_f32_rows(
            commands,
            self.layout.scores,
            config.n_head,
            base_position,
            rows,
            false,
        )?;
        self.ops.record_attention_values_f32_kv_rows(
            commands,
            self.layout.scores,
            cache_v,
            self.layout.attn,
            slot,
            layer_count,
            base_position,
            self.capacity,
            config.n_head,
            config.n_head_kv,
            config.n_embd_head_k,
            rows,
            false,
        )?;
        self.ops.record_weight_matmul_rows(
            commands,
            attention.output,
            self.layout.attn,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.projection, n_embd, n_embd * 4)],
            n_attn_q,
            rows,
            n_attn_q,
        )
    }

    fn record_ffn(
        &self,
        commands: &TokenCommands<'_>,
        ffn: &FfnBindings,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let n_embd = config.n_embd;
        let n_ff = config.n_ff;
        self.ops.record_weight_matmul_rows(
            commands,
            ffn.up,
            self.layout.normed,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.ffn_up, n_ff, n_ff * 4)],
            n_embd,
            rows,
            n_embd,
        )?;
        // ReLU(x)^2: llama.cpp's LLM_FFN_RELU_SQR. The shader squares the
        // clamped value, so the result is `max(x, 0)^2`.
        let push = [
            Self::word(self.layout.ffn_up),
            n_ff as u32,
            rows as u32,
            n_ff as u32,
        ];
        let [x, y, z] = row_dispatch(n_ff.div_ceil(64), rows, &self.context.limits)?;
        self.ops
            .record_custom(commands, self.relu_square_pipeline, None, &push, [x, y, z])?;
        self.ops.record_weight_matmul_rows(
            commands,
            ffn.down,
            self.layout.ffn_up,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &[(self.layout.ffn_down, n_embd, n_embd * 4)],
            n_ff,
            rows,
            n_ff,
        )
    }

    fn record_weights(
        &self,
        commands: &TokenCommands<'_>,
        bindings: OperatorBindings,
        input: ArenaRegion,
        outputs: &[(ArenaRegion, usize)],
        n_in: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let outputs = outputs
            .iter()
            .map(|&(region, width)| {
                Ok((
                    region,
                    width,
                    width.checked_mul(4).ok_or(VulkanError::OutOfMemory)?,
                ))
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        self.ops.record_weight_matmul_rows(
            commands,
            bindings,
            input,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            &outputs,
            n_in,
            rows,
            n_in,
        )
    }

    fn word(region: ArenaRegion) -> u32 {
        (region.offset / 4) as u32
    }
}

fn check_device_eligibility(shader_float16: bool) -> Result<(), String> {
    shader_float16
        .then_some(())
        .ok_or_else(|| "selected Vulkan device does not support shaderFloat16".into())
}
