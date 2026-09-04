use super::ops::{ArenaRegion, GpuWeightFormat, OperatorBindings, Qwen3Ops, TokenCommands};
use super::qwen3::{TokenCommitState, UploadedBuffers};
use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::core::scratchpad::KvCache;
use crate::core::tensor::GGMLType;
use crate::models::qwen35::{Qwen35Config, Qwen35Model};
use crate::ops::kernel::Weight;

#[derive(Debug, Clone)]
struct EligibilityFacts {
    architecture: String,
    weight_formats: Vec<GGMLType>,
    unrecorded_operations: Vec<&'static str>,
}

fn check_eligibility(facts: &EligibilityFacts) -> Result<(), String> {
    if facts.architecture != "qwen35" {
        return Err(format!("unsupported architecture {}", facts.architecture));
    }
    if facts.weight_formats.is_empty()
        || facts
            .weight_formats
            .iter()
            .any(|&format| format != GGMLType::BF16)
    {
        return Err("unsupported Qwen3.5 Vulkan weight format".into());
    }
    if let Some(operation) = facts.unrecorded_operations.first() {
        return Err(format!(
            "Qwen3.5 Vulkan operation {operation} has no recorder"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_shadow_state(
    kv_cache: &mut KvCache,
    conv_states: &mut [Vec<f32>],
    ssm_states: &mut [Vec<f32>],
    position: usize,
    capacity: usize,
    kv_stride: usize,
    k_delta: &[f32],
    v_delta: &[f32],
    conv_state: &[f32],
    ssm_state: &[f32],
) -> Result<(), String> {
    let layer_count = conv_states.len();
    if layer_count != ssm_states.len() {
        return Err("Qwen3.5 recurrent shadow layer counts differ".into());
    }
    let delta_len = layer_count
        .checked_mul(kv_stride)
        .ok_or_else(|| "Qwen3.5 KV delta length overflow".to_string())?;
    if position >= capacity || k_delta.len() != delta_len || v_delta.len() != delta_len {
        return Err(format!(
            "Invalid Qwen3.5 Vulkan KV delta: position={position}/{capacity} k={} v={} expected={delta_len}",
            k_delta.len(),
            v_delta.len()
        ));
    }
    let cache_len = layer_count
        .checked_mul(capacity)
        .and_then(|count| count.checked_mul(kv_stride))
        .ok_or_else(|| "Qwen3.5 KV cache length overflow".to_string())?;
    let KvCache::F32(cache) = kv_cache else {
        return Err("Qwen3.5 Vulkan requires an F32 CPU shadow KV cache".into());
    };
    if cache.k.len() != cache_len || cache.v.len() != cache_len {
        return Err("Invalid Qwen3.5 CPU shadow KV cache length".into());
    }

    let conv_len = conv_states.iter().try_fold(0usize, |count, state| {
        count
            .checked_add(state.len())
            .ok_or_else(|| "Qwen3.5 convolution shadow length overflow".to_string())
    })?;
    if conv_state.len() != conv_len {
        return Err(format!(
            "Invalid Qwen3.5 Vulkan convolution state: got {}, expected {conv_len}",
            conv_state.len()
        ));
    }
    let ssm_len = ssm_states.iter().try_fold(0usize, |count, state| {
        count
            .checked_add(state.len())
            .ok_or_else(|| "Qwen3.5 SSM shadow length overflow".to_string())
    })?;
    if ssm_state.len() != ssm_len {
        return Err(format!(
            "Invalid Qwen3.5 Vulkan SSM state: got {}, expected {ssm_len}",
            ssm_state.len()
        ));
    }

    for layer in 0..layer_count {
        let source = layer * kv_stride;
        let target = (layer * capacity + position) * kv_stride;
        cache.k[target..target + kv_stride].copy_from_slice(&k_delta[source..source + kv_stride]);
        cache.v[target..target + kv_stride].copy_from_slice(&v_delta[source..source + kv_stride]);
    }
    let mut offset = 0;
    for shadow in conv_states {
        let end = offset + shadow.len();
        shadow.copy_from_slice(&conv_state[offset..end]);
        offset = end;
    }
    offset = 0;
    for shadow in ssm_states {
        let end = offset + shadow.len();
        shadow.copy_from_slice(&ssm_state[offset..end]);
        offset = end;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Qwen35ArenaLayout {
    x: ArenaRegion,
    normed: ArenaRegion,
    raw_qkv: ArenaRegion,
    raw_k: ArenaRegion,
    q: ArenaRegion,
    k: ArenaRegion,
    v: ArenaRegion,
    z: ArenaRegion,
    beta: ArenaRegion,
    alpha: ArenaRegion,
    attn: ArenaRegion,
    projection: ArenaRegion,
    ffn_gate: ArenaRegion,
    ffn_up: ArenaRegion,
    down: ArenaRegion,
    logits: ArenaRegion,
    rope: ArenaRegion,
    q8: ArenaRegion,
    q8_scales: ArenaRegion,
    q4_1_input_sums: ArenaRegion,
    q8k: ArenaRegion,
    q8k_scales: ArenaRegion,
    kv_k: ArenaRegion,
    kv_v: ArenaRegion,
    kv_delta_k: ArenaRegion,
    kv_delta_v: ArenaRegion,
    conv_state: ArenaRegion,
    ssm_state: ArenaRegion,
    total_size: usize,
}

impl Qwen35ArenaLayout {
    fn new(config: &Qwen35Config, capacity: usize) -> Result<Self, crate::vulkan::VulkanError> {
        let layer_count = config.n_layer_impl();
        let head_dim = config.n_embd_head();
        let value_heads = config.ssm_dt_rank;
        if [
            config.n_embd,
            config.n_ff,
            config.n_head,
            config.n_head_kv,
            head_dim,
            config.vocab_size,
            layer_count,
            capacity,
            config.rope_dimension_count,
            config.ssm_d_conv,
            config.ssm_d_state,
            config.ssm_n_group,
            value_heads,
            config.ssm_d_inner,
        ]
        .contains(&0)
            || config.n_head % config.n_head_kv != 0
            || config.ssm_d_inner % value_heads != 0
        {
            return Err(crate::vulkan::VulkanError::UnsupportedShape(
                "invalid Qwen3.5 Vulkan arena dimensions".into(),
            ));
        }

        let q_raw = qwen35_product("gated query", &[config.n_head, head_dim, 2])?;
        let dense_q = qwen35_product("dense query", &[config.n_head, head_dim])?;
        let dense_kv = qwen35_product("dense KV", &[config.n_head_kv, head_dim])?;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let head_v_dim = config.head_v_dim();
        let kv_cache = qwen35_product("KV cache", &[layer_count, capacity, dense_kv])?;
        let kv_delta = qwen35_product("KV delta", &[layer_count, dense_kv])?;
        let conv_state = qwen35_product(
            "convolution state",
            &[layer_count, config.ssm_d_conv, conv_dim],
        )?;
        let ssm_state = qwen35_product(
            "SSM state",
            &[layer_count, value_heads, head_v_dim, head_v_dim],
        )?;
        let mut cursor = 0usize;
        let x = qwen35_f32_region(&mut cursor, config.n_embd)?;
        let normed = qwen35_f32_region(&mut cursor, config.n_embd)?;
        let raw_qkv = qwen35_f32_region(&mut cursor, q_raw.max(conv_dim))?;
        let raw_k = qwen35_f32_region(&mut cursor, dense_kv)?;
        let q = qwen35_f32_region(&mut cursor, dense_q.max(key_dim))?;
        let k = qwen35_f32_region(&mut cursor, dense_kv.max(key_dim))?;
        let v = qwen35_f32_region(&mut cursor, dense_kv.max(value_dim))?;
        let z = qwen35_f32_region(&mut cursor, dense_q.max(value_dim))?;
        let beta = qwen35_f32_region(&mut cursor, value_heads)?;
        let alpha = qwen35_f32_region(&mut cursor, value_heads)?;
        let attn = qwen35_f32_region(&mut cursor, dense_q.max(value_dim))?;
        let projection = qwen35_f32_region(&mut cursor, config.n_embd)?;
        let ffn_gate = qwen35_f32_region(&mut cursor, config.n_ff)?;
        let ffn_up = qwen35_f32_region(&mut cursor, config.n_ff)?;
        let down = qwen35_f32_region(&mut cursor, config.n_embd)?;
        let logits = qwen35_f32_region(&mut cursor, config.vocab_size)?;
        let rope = qwen35_f32_region(&mut cursor, config.rope_dimension_count)?;
        let q8 = qwen35_region(&mut cursor, 4)?;
        let q8_scales = qwen35_f32_region(&mut cursor, 1)?;
        let q4_1_input_sums = qwen35_f32_region(&mut cursor, 1)?;
        let q8k = qwen35_region(&mut cursor, 4)?;
        let q8k_scales = qwen35_f32_region(&mut cursor, 1)?;
        let kv_k = qwen35_f32_region(&mut cursor, kv_cache)?;
        let kv_v = qwen35_f32_region(&mut cursor, kv_cache)?;
        let kv_delta_k = qwen35_f32_region(&mut cursor, kv_delta)?;
        let kv_delta_v = qwen35_f32_region(&mut cursor, kv_delta)?;
        let conv_state = qwen35_f32_region(&mut cursor, conv_state)?;
        let ssm_state = qwen35_f32_region(&mut cursor, ssm_state)?;
        Ok(Self {
            x,
            normed,
            raw_qkv,
            raw_k,
            q,
            k,
            v,
            z,
            beta,
            alpha,
            attn,
            projection,
            ffn_gate,
            ffn_up,
            down,
            logits,
            rope,
            q8,
            q8_scales,
            q4_1_input_sums,
            q8k,
            q8k_scales,
            kv_k,
            kv_v,
            kv_delta_k,
            kv_delta_v,
            conv_state,
            ssm_state,
            total_size: cursor,
        })
    }

    fn total_size(self) -> usize {
        self.total_size
    }
}

fn qwen35_product(label: &str, values: &[usize]) -> Result<usize, crate::vulkan::VulkanError> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            crate::vulkan::VulkanError::UnsupportedShape(format!(
                "Qwen3.5 {label} size overflows usize"
            ))
        })
    })
}

fn qwen35_f32_region(
    cursor: &mut usize,
    elements: usize,
) -> Result<ArenaRegion, crate::vulkan::VulkanError> {
    let bytes = elements
        .checked_mul(4)
        .ok_or(crate::vulkan::VulkanError::OutOfMemory)?;
    qwen35_region(cursor, bytes)
}

fn qwen35_region(
    cursor: &mut usize,
    size: usize,
) -> Result<ArenaRegion, crate::vulkan::VulkanError> {
    let offset = cursor
        .checked_add(15)
        .map(|value| value & !15)
        .ok_or(crate::vulkan::VulkanError::OutOfMemory)?;
    *cursor = offset
        .checked_add(size)
        .ok_or(crate::vulkan::VulkanError::OutOfMemory)?;
    Ok(ArenaRegion { offset, size })
}

fn fill_mrope(
    coefficients: &mut [f32],
    positions: [usize; 4],
    sections: [i32; 4],
    freq_base: f32,
) -> Result<(), String> {
    if coefficients.is_empty()
        || coefficients.len() % 2 != 0
        || sections.iter().any(|&section| section < 0)
    {
        return Err("Invalid Qwen3.5 mRoPE coefficients".into());
    }
    let half = coefficients.len() / 2;
    let section_count = sections.iter().try_fold(0usize, |sum, &section| {
        sum.checked_add(section as usize)
            .ok_or_else(|| "Qwen3.5 mRoPE section count overflow".to_string())
    })?;
    let boundaries = [
        sections[0] as usize,
        (sections[0] + sections[1]) as usize,
        (sections[0] + sections[1] + sections[2]) as usize,
    ];
    let theta_scale = freq_base.powf(-2.0 / coefficients.len() as f32);
    let mut theta = positions.map(|position| position as f32);
    for index in 0..half {
        let axis = if section_count == 0 {
            0
        } else {
            let sector = index % section_count;
            if sector < boundaries[0] {
                0
            } else if sector < boundaries[1] {
                1
            } else if sector < boundaries[2] {
                2
            } else {
                3
            }
        };
        coefficients[index] = theta[axis].cos();
        coefficients[index + half] = theta[axis].sin();
        for value in &mut theta {
            *value *= theta_scale;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DenseBindings {
    qkv: OperatorBindings,
    prepare: OperatorBindings,
    output: OperatorBindings,
}

#[derive(Clone, Copy)]
struct RecurrentBindings {
    qkv_gate_beta: OperatorBindings,
    alpha: OperatorBindings,
    convolution: OperatorBindings,
    ssm: OperatorBindings,
    output: OperatorBindings,
}

#[derive(Clone, Copy)]
enum AttentionBindings {
    Dense(DenseBindings),
    Recurrent(RecurrentBindings),
}

#[derive(Clone, Copy)]
struct LayerBindings {
    attention_norm: OperatorBindings,
    attention: AttentionBindings,
    post_attention_norm: OperatorBindings,
    gate_up: OperatorBindings,
    down: OperatorBindings,
}

pub(crate) struct Qwen35GpuTokenResult<'a> {
    pub(crate) logits: &'a [f32],
    pub(crate) k_delta: &'a [f32],
    pub(crate) v_delta: &'a [f32],
    pub(crate) conv_state: &'a [f32],
    pub(crate) ssm_state: &'a [f32],
}

pub(crate) struct Qwen35VulkanSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    layers: Vec<LayerBindings>,
    output_norm: OperatorBindings,
    output: OperatorBindings,
    layout: Qwen35ArenaLayout,
    config: Qwen35Config,
    capacity: usize,
    commit_state: TokenCommitState,
    rope: Vec<f32>,
    logits: Vec<f32>,
    k_delta: Vec<f32>,
    v_delta: Vec<f32>,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl Qwen35VulkanSession {
    pub(crate) fn try_new(
        model: &Qwen35Model<'_>,
        capacity: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        let facts = eligibility_facts(model);
        if let Err(reason) = check_eligibility(&facts) {
            eprintln!("[GPU] Qwen3.5 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        if let Err(reason) = validate_executor_shape(&model.config, capacity) {
            eprintln!("[GPU] Qwen3.5 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }

        let config = model.config.clone();
        let layer_count = config.n_layer_impl();
        let layout = Qwen35ArenaLayout::new(&config, capacity)?;
        let descriptor_capacity = layer_count
            .checked_mul(10)
            .and_then(|count| count.checked_add(3))
            .ok_or(VulkanError::OutOfMemory)?;
        let mut ops = Qwen3Ops::new_with_size(context, layout.total_size(), descriptor_capacity)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(layer_count);

        for (layer_index, layer) in model.layers.iter().enumerate() {
            let attention_norm_buffer = buffers.upload_f32(&layer.attn_norm)?;
            let attention_norm = ops.bind_buffers(&[attention_norm_buffer])?;

            let attention = if config.is_recurrent[layer_index] {
                let wqkv = required_weight(layer.wqkv.as_ref(), layer_index, "attn_qkv")?;
                let gate = required_weight(layer.wqkv_gate.as_ref(), layer_index, "attn_gate")?;
                let beta = required_weight(layer.ssm_beta.as_ref(), layer_index, "ssm_beta")?;
                let alpha = required_weight(layer.ssm_alpha.as_ref(), layer_index, "ssm_alpha")?;
                let output = required_weight(layer.ssm_out.as_ref(), layer_index, "ssm_out")?;
                let projection_buffers = [
                    upload_bf16(&mut buffers, wqkv, "Qwen3.5 recurrent QKV")?,
                    upload_bf16(&mut buffers, gate, "Qwen3.5 recurrent gate")?,
                    upload_bf16(&mut buffers, beta, "Qwen3.5 recurrent beta")?,
                ];
                let qkv_gate_beta = bind_bf16(&mut ops, &projection_buffers)?;
                let alpha_buffer = upload_bf16(&mut buffers, alpha, "Qwen3.5 recurrent alpha")?;
                let alpha = bind_bf16(&mut ops, &[alpha_buffer])?;
                let conv_weight = layer.ssm_conv1d.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} convolution weight"
                    ))
                })?;
                let conv_buffer = buffers.upload_f32(conv_weight)?;
                let convolution = ops.bind_buffers(&[conv_buffer])?;
                let dt = layer.ssm_dt.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} SSM dt bias"
                    ))
                })?;
                let a = layer.ssm_a.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} SSM A"
                    ))
                })?;
                let norm = layer.ssm_norm.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} SSM norm"
                    ))
                })?;
                let ssm_buffers = [
                    buffers.upload_f32(dt)?,
                    buffers.upload_f32(a)?,
                    buffers.upload_f32(norm)?,
                ];
                let ssm = ops.bind_buffers(&ssm_buffers)?;
                let output_buffer = upload_bf16(&mut buffers, output, "Qwen3.5 recurrent output")?;
                let output = bind_bf16(&mut ops, &[output_buffer])?;
                AttentionBindings::Recurrent(RecurrentBindings {
                    qkv_gate_beta,
                    alpha,
                    convolution,
                    ssm,
                    output,
                })
            } else {
                let wq = required_weight(layer.wq.as_ref(), layer_index, "attn_q")?;
                let wk = required_weight(layer.wk.as_ref(), layer_index, "attn_k")?;
                let wv = required_weight(layer.wv.as_ref(), layer_index, "attn_v")?;
                let wo = required_weight(layer.wo.as_ref(), layer_index, "attn_output")?;
                let qkv_buffers = [
                    upload_bf16(&mut buffers, wq, "Qwen3.5 dense Q")?,
                    upload_bf16(&mut buffers, wk, "Qwen3.5 dense K")?,
                    upload_bf16(&mut buffers, wv, "Qwen3.5 dense V")?,
                ];
                let qkv = bind_bf16(&mut ops, &qkv_buffers)?;
                let q_norm = layer.attn_q_norm.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} Q norm"
                    ))
                })?;
                let k_norm = layer.attn_k_norm.as_ref().ok_or_else(|| {
                    VulkanError::UnsupportedShape(format!(
                        "missing Qwen3.5 layer {layer_index} K norm"
                    ))
                })?;
                let prepare_buffers = [buffers.upload_f32(q_norm)?, buffers.upload_f32(k_norm)?];
                let prepare = ops.bind_buffers(&prepare_buffers)?;
                let output_buffer = upload_bf16(&mut buffers, wo, "Qwen3.5 dense output")?;
                let output = bind_bf16(&mut ops, &[output_buffer])?;
                AttentionBindings::Dense(DenseBindings {
                    qkv,
                    prepare,
                    output,
                })
            };

            let post_norm_buffer = buffers.upload_f32(&layer.attn_post_norm)?;
            let post_attention_norm = ops.bind_buffers(&[post_norm_buffer])?;
            let gate_up_buffers = [
                upload_bf16(&mut buffers, &layer.ffn_gate, "Qwen3.5 FFN gate")?,
                upload_bf16(&mut buffers, &layer.ffn_up, "Qwen3.5 FFN up")?,
            ];
            let gate_up = bind_bf16(&mut ops, &gate_up_buffers)?;
            let down_buffer = upload_bf16(&mut buffers, &layer.ffn_down, "Qwen3.5 FFN down")?;
            let down = bind_bf16(&mut ops, &[down_buffer])?;
            layers.push(LayerBindings {
                attention_norm,
                attention,
                post_attention_norm,
                gate_up,
                down,
            });
        }

        let output_norm_buffer = buffers.upload_f32(&model.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output_buffer = upload_bf16(&mut buffers, &model.output_weight, "Qwen3.5 output")?;
        let output = bind_bf16(&mut ops, &[output_buffer])?;
        let dense_kv = config
            .n_head_kv
            .checked_mul(config.n_embd_head())
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = layer_count
            .checked_mul(dense_kv)
            .ok_or(VulkanError::OutOfMemory)?;
        let conv_count = layer_count
            .checked_mul(config.ssm_d_conv)
            .and_then(|count| count.checked_mul(config.conv_dim()))
            .ok_or(VulkanError::OutOfMemory)?;
        let ssm_count = layer_count
            .checked_mul(config.ssm_dt_rank)
            .and_then(|count| count.checked_mul(config.head_v_dim()))
            .and_then(|count| count.checked_mul(config.head_v_dim()))
            .ok_or(VulkanError::OutOfMemory)?;
        let vocab = config.vocab_size;
        let rope_dim = config.rope_dimension_count;

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
            commit_state: TokenCommitState::new(0),
            rope: vec![0.0; rope_dim],
            logits: vec![0.0; vocab],
            k_delta: vec![0.0; delta_count],
            v_delta: vec![0.0; delta_count],
            conv_state: vec![0.0; conv_count],
            ssm_state: vec![0.0; ssm_count],
        }))
    }

    pub(crate) fn forward_token<'a>(
        &'a mut self,
        input: &[f32],
        cache_position: usize,
        mrope_position: [usize; 4],
    ) -> Result<Qwen35GpuTokenResult<'a>, VulkanError> {
        self.commit_state
            .begin(cache_position)
            .map_err(VulkanError::UnsupportedShape)?;
        if let Err(error) = self.forward_token_inner(input, cache_position, mrope_position) {
            self.commit_state.abort();
            return Err(error);
        }
        Ok(Qwen35GpuTokenResult {
            logits: &self.logits,
            k_delta: &self.k_delta,
            v_delta: &self.v_delta,
            conv_state: &self.conv_state,
            ssm_state: &self.ssm_state,
        })
    }

    fn forward_token_inner(
        &mut self,
        input: &[f32],
        cache_position: usize,
        mrope_position: [usize; 4],
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        if input.len() != config.n_embd || cache_position >= self.capacity {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Qwen3.5 token input={} position={cache_position}/{}",
                input.len(),
                self.capacity
            )));
        }
        let layer_count = config.n_layer_impl();
        let head_dim = config.n_embd_head();
        let dense_q = config
            .n_head
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let dense_raw_q = dense_q.checked_mul(2).ok_or(VulkanError::OutOfMemory)?;
        let dense_kv = config
            .n_head_kv
            .checked_mul(head_dim)
            .ok_or(VulkanError::OutOfMemory)?;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let value_heads = config.ssm_dt_rank;
        let recurrent_head_dim = config.head_v_dim();

        self.ops.write_f32(self.layout.x, input)?;
        fill_mrope(
            &mut self.rope,
            mrope_position,
            config.rope_dimension_sections,
            config.rope_freq_base,
        )
        .map_err(VulkanError::UnsupportedShape)?;
        self.ops.write_f32(self.layout.rope, &self.rope)?;
        let commands = TokenCommands::begin(self.context)?;

        for (layer_index, bindings) in self.layers.iter().enumerate() {
            self.ops.record_rms_norm(
                &commands,
                bindings.attention_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.norm_eps,
            )?;
            match bindings.attention {
                AttentionBindings::Dense(dense) => {
                    self.ops.record_weight_matvec_group(
                        &commands,
                        dense.qkv,
                        self.layout.normed,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        &[
                            (self.layout.raw_qkv, dense_raw_q),
                            (self.layout.raw_k, dense_kv),
                            (self.layout.v, dense_kv),
                        ],
                        config.n_embd,
                    )?;
                    self.ops.record_qwen35_dense_prepare(
                        &commands,
                        dense.prepare,
                        self.layout.raw_qkv,
                        self.layout.raw_k,
                        self.layout.v,
                        self.layout.q,
                        self.layout.z,
                        self.layout.kv_k,
                        self.layout.kv_v,
                        self.layout.kv_delta_k,
                        self.layout.kv_delta_v,
                        self.layout.rope,
                        layer_index,
                        layer_count,
                        cache_position,
                        self.capacity,
                        config.n_head,
                        config.n_head_kv,
                        head_dim,
                        config.rope_dimension_count,
                        config.norm_eps,
                    )?;
                    self.ops.record_qwen35_attention(
                        &commands,
                        self.layout.q,
                        self.layout.z,
                        self.layout.kv_k,
                        self.layout.kv_v,
                        self.layout.attn,
                        layer_index,
                        layer_count,
                        cache_position + 1,
                        self.capacity,
                        config.n_head,
                        config.n_head_kv,
                        head_dim,
                    )?;
                    self.ops.record_weight_matvec(
                        &commands,
                        dense.output,
                        self.layout.attn,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        self.layout.projection,
                        dense_q,
                        config.n_embd,
                    )?;
                }
                AttentionBindings::Recurrent(recurrent) => {
                    self.ops.record_weight_matvec_group(
                        &commands,
                        recurrent.qkv_gate_beta,
                        self.layout.normed,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        &[
                            (self.layout.raw_qkv, conv_dim),
                            (self.layout.z, value_dim),
                            (self.layout.beta, value_heads),
                        ],
                        config.n_embd,
                    )?;
                    self.ops.record_weight_matvec(
                        &commands,
                        recurrent.alpha,
                        self.layout.normed,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        self.layout.alpha,
                        config.n_embd,
                        value_heads,
                    )?;
                    self.ops.record_qwen35_recurrent_conv(
                        &commands,
                        recurrent.convolution,
                        self.layout.raw_qkv,
                        self.layout.q,
                        self.layout.k,
                        self.layout.v,
                        self.layout.conv_state,
                        layer_index,
                        layer_count,
                        conv_dim,
                        key_dim,
                        value_dim,
                        config.ssm_d_conv,
                        config.ssm_n_group,
                        value_heads,
                        recurrent_head_dim,
                        config.norm_eps,
                    )?;
                    self.ops.record_qwen35_recurrent_ssm(
                        &commands,
                        recurrent.ssm,
                        self.layout.q,
                        self.layout.k,
                        self.layout.v,
                        self.layout.z,
                        self.layout.beta,
                        self.layout.alpha,
                        self.layout.attn,
                        self.layout.ssm_state,
                        layer_index,
                        layer_count,
                        config.ssm_n_group,
                        value_heads,
                        recurrent_head_dim,
                        config.norm_eps,
                    )?;
                    self.ops.record_weight_matvec(
                        &commands,
                        recurrent.output,
                        self.layout.attn,
                        self.layout.q8,
                        self.layout.q8_scales,
                        self.layout.q4_1_input_sums,
                        self.layout.q8k,
                        self.layout.q8k_scales,
                        self.layout.projection,
                        value_dim,
                        config.n_embd,
                    )?;
                }
            }
            self.ops.record_add(
                &commands,
                self.layout.x,
                self.layout.projection,
                config.n_embd,
            )?;
            self.ops.record_rms_norm(
                &commands,
                bindings.post_attention_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.norm_eps,
            )?;
            self.ops.record_weight_matvec_group(
                &commands,
                bindings.gate_up,
                self.layout.normed,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                &[
                    (self.layout.ffn_gate, config.n_ff),
                    (self.layout.ffn_up, config.n_ff),
                ],
                config.n_embd,
            )?;
            self.ops.record_silu_mul(
                &commands,
                self.layout.ffn_gate,
                self.layout.ffn_up,
                config.n_ff,
            )?;
            self.ops.record_weight_matvec(
                &commands,
                bindings.down,
                self.layout.ffn_gate,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                self.layout.down,
                config.n_ff,
                config.n_embd,
            )?;
            self.ops
                .record_add(&commands, self.layout.x, self.layout.down, config.n_embd)?;
        }

        self.ops.record_rms_norm(
            &commands,
            self.output_norm,
            self.layout.x,
            self.layout.normed,
            config.n_embd,
            config.norm_eps,
        )?;
        self.ops.record_weight_matvec(
            &commands,
            self.output,
            self.layout.normed,
            self.layout.q8,
            self.layout.q8_scales,
            self.layout.q4_1_input_sums,
            self.layout.q8k,
            self.layout.q8k_scales,
            self.layout.logits,
            config.n_embd,
            config.vocab_size,
        )?;
        commands.submit_and_wait()?;

        self.logits
            .copy_from_slice(self.ops.read_f32(self.layout.logits, config.vocab_size)?);
        let delta_count = self.k_delta.len();
        self.k_delta
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_k, delta_count)?);
        self.v_delta
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_v, delta_count)?);
        let conv_count = self.conv_state.len();
        self.conv_state
            .copy_from_slice(self.ops.read_f32(self.layout.conv_state, conv_count)?);
        let ssm_count = self.ssm_state.len();
        self.ssm_state
            .copy_from_slice(self.ops.read_f32(self.layout.ssm_state, ssm_count)?);
        Ok(())
    }

    pub(crate) fn commit_token(&mut self) {
        self.commit_state.commit();
    }

    pub(crate) fn abort_token(&mut self) {
        self.commit_state.abort();
    }

    pub(crate) fn reset(&mut self) -> Result<(), VulkanError> {
        self.commit_state.reset();
        for region in [
            self.layout.kv_k,
            self.layout.kv_v,
            self.layout.kv_delta_k,
            self.layout.kv_delta_v,
            self.layout.conv_state,
            self.layout.ssm_state,
        ] {
            self.ops.zero_region(region)?;
        }
        Ok(())
    }
}

fn required_weight<'a>(
    weight: Option<&'a Weight<'a>>,
    layer: usize,
    name: &str,
) -> Result<&'a Weight<'a>, VulkanError> {
    weight.ok_or_else(|| {
        VulkanError::UnsupportedShape(format!("missing Qwen3.5 layer {layer} {name} weight"))
    })
}

fn upload_bf16(
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<GpuBuffer, VulkanError> {
    if weight.ggml_type != GGMLType::BF16 {
        return Err(VulkanError::UnsupportedShape(format!(
            "{label} has unsupported Vulkan format {:?}",
            weight.ggml_type
        )));
    }
    let bytes = weight.kernel.bf16_bytes().ok_or_else(|| {
        VulkanError::UnsupportedShape(format!("{label} does not expose BF16 bytes"))
    })?;
    let expected = weight
        .n_in
        .checked_mul(weight.n_out)
        .and_then(|count| count.checked_mul(2))
        .ok_or(VulkanError::OutOfMemory)?;
    if bytes.len() != expected {
        return Err(VulkanError::UnsupportedShape(format!(
            "{label} has {} BF16 bytes, expected {expected}",
            bytes.len()
        )));
    }
    buffers.upload(bytes)
}

fn bind_bf16(
    ops: &mut Qwen3Ops<'_>,
    buffers: &[GpuBuffer],
) -> Result<OperatorBindings, VulkanError> {
    let formats = vec![GpuWeightFormat::BF16; buffers.len()];
    ops.bind_weight_buffers(buffers, &formats)
}

fn eligibility_facts(model: &Qwen35Model<'_>) -> EligibilityFacts {
    let mut facts = EligibilityFacts {
        architecture: "qwen35".into(),
        weight_formats: vec![model.output_weight.ggml_type],
        unrecorded_operations: Vec::new(),
    };
    let layer_count = model.config.n_layer_impl();
    if model.layers.len() != layer_count || model.config.is_recurrent.len() < layer_count {
        facts.unrecorded_operations.push("layer_layout");
        return facts;
    }
    for (layer_index, layer) in model.layers.iter().enumerate() {
        facts.weight_formats.extend([
            layer.ffn_gate.ggml_type,
            layer.ffn_up.ggml_type,
            layer.ffn_down.ggml_type,
        ]);
        if model.config.is_recurrent[layer_index] {
            let weights = [
                layer.wqkv.as_ref(),
                layer.wqkv_gate.as_ref(),
                layer.ssm_beta.as_ref(),
                layer.ssm_alpha.as_ref(),
                layer.ssm_out.as_ref(),
            ];
            if weights.iter().any(|weight| weight.is_none())
                || layer.ssm_conv1d.is_none()
                || layer.ssm_dt.is_none()
                || layer.ssm_a.is_none()
                || layer.ssm_norm.is_none()
            {
                facts.unrecorded_operations.push("recurrent_layer");
            } else {
                facts
                    .weight_formats
                    .extend(weights.into_iter().flatten().map(|weight| weight.ggml_type));
            }
        } else {
            let weights = [
                layer.wq.as_ref(),
                layer.wk.as_ref(),
                layer.wv.as_ref(),
                layer.wo.as_ref(),
            ];
            if weights.iter().any(|weight| weight.is_none())
                || layer.attn_q_norm.is_none()
                || layer.attn_k_norm.is_none()
            {
                facts.unrecorded_operations.push("dense_attention_layer");
            } else {
                facts
                    .weight_formats
                    .extend(weights.into_iter().flatten().map(|weight| weight.ggml_type));
            }
        }
    }
    facts
}

fn validate_executor_shape(config: &Qwen35Config, capacity: usize) -> Result<(), String> {
    let layer_count = config.n_layer_impl();
    let head_dim = config.n_embd_head();
    let recurrent_head_dim = config.head_v_dim();
    if capacity == 0 || capacity > config.n_ctx {
        return Err(format!(
            "session capacity {capacity} must be within 1..={}",
            config.n_ctx
        ));
    }
    if capacity > 4096 {
        return Err(format!(
            "capacity {capacity} exceeds attention shader limit 4096"
        ));
    }
    if layer_count == 0
        || config.is_recurrent.len() != layer_count
        || config.n_head == 0
        || config.n_head_kv == 0
        || config.n_head % config.n_head_kv != 0
        || head_dim == 0
        || config.rope_dimension_count == 0
        || config.rope_dimension_count > head_dim
        || config.rope_dimension_count % 2 != 0
        || config.ssm_n_group == 0
        || config.ssm_dt_rank == 0
        || config.ssm_d_conv == 0
        || recurrent_head_dim == 0
        || recurrent_head_dim > 128
        || config.ssm_d_state != recurrent_head_dim
        || config.value_dim() != config.ssm_d_inner
    {
        return Err("unsupported Qwen3.5 Vulkan execution shape".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        check_eligibility, commit_shadow_state, fill_mrope, EligibilityFacts, Qwen35ArenaLayout,
    };
    use crate::core::scratchpad::KvCache;
    use crate::core::tensor::GGMLType;
    use crate::models::qwen35::Qwen35Config;

    fn eligible_facts() -> EligibilityFacts {
        EligibilityFacts {
            architecture: "qwen35".into(),
            weight_formats: vec![GGMLType::BF16; 187],
            unrecorded_operations: Vec::new(),
        }
    }

    #[test]
    fn qwen35_bf16_mixed_stack_is_eligible_only_when_every_operation_has_a_recorder() {
        assert_eq!(check_eligibility(&eligible_facts()), Ok(()));

        let mut facts = eligible_facts();
        facts.unrecorded_operations.push("ssm_state_update");
        assert!(check_eligibility(&facts)
            .expect_err("an unrecorded operation must reject the whole model")
            .contains("ssm_state_update"));
    }

    #[test]
    fn invalid_gpu_token_keeps_dense_kv_and_recurrent_shadow_unchanged() {
        let mut cache = KvCache::new_f32(1, 2, 2);
        let mut conv = vec![vec![1.0, 2.0]];
        let mut ssm = vec![vec![3.0, 4.0, 5.0]];

        let error = commit_shadow_state(
            &mut cache,
            &mut conv,
            &mut ssm,
            0,
            2,
            2,
            &[10.0, 11.0],
            &[12.0, 13.0],
            &[20.0, 21.0],
            &[30.0, 31.0],
        )
        .expect_err("short SSM state must abort the complete shadow commit");
        assert!(error.contains("SSM"));
        let KvCache::F32(cache) = cache else {
            unreachable!()
        };
        assert_eq!(cache.k, vec![0.0; 4]);
        assert_eq!(cache.v, vec![0.0; 4]);
        assert_eq!(conv, [vec![1.0, 2.0]]);
        assert_eq!(ssm, [vec![3.0, 4.0, 5.0]]);
    }

    #[test]
    fn complete_gpu_token_commits_dense_kv_and_recurrent_shadow_together() {
        let mut cache = KvCache::new_f32(1, 2, 2);
        let mut conv = vec![vec![1.0, 2.0]];
        let mut ssm = vec![vec![3.0, 4.0, 5.0]];

        commit_shadow_state(
            &mut cache,
            &mut conv,
            &mut ssm,
            1,
            2,
            2,
            &[10.0, 11.0],
            &[12.0, 13.0],
            &[20.0, 21.0],
            &[30.0, 31.0, 32.0],
        )
        .unwrap();

        let KvCache::F32(cache) = cache else {
            unreachable!()
        };
        assert_eq!(cache.k, [0.0, 0.0, 10.0, 11.0]);
        assert_eq!(cache.v, [0.0, 0.0, 12.0, 13.0]);
        assert_eq!(conv, [vec![20.0, 21.0]]);
        assert_eq!(ssm, [vec![30.0, 31.0, 32.0]]);
    }

    fn layout_config() -> Qwen35Config {
        Qwen35Config {
            n_embd: 8,
            n_layer: 3,
            n_nextn: 1,
            n_head: 2,
            n_head_kv: 1,
            n_ff: 12,
            n_ctx: 16,
            vocab_size: 32,
            rope_freq_base: 10_000.0,
            norm_eps: 1e-6,
            rope_dimension_count: 4,
            rope_dimension_sections: [1, 1, 0, 0],
            ssm_d_conv: 2,
            ssm_d_state: 4,
            ssm_n_group: 1,
            ssm_dt_rank: 2,
            ssm_d_inner: 8,
            full_attention_interval: 2,
            is_recurrent: vec![true, false],
            key_length: 4,
            value_length: 4,
        }
    }

    #[test]
    fn arena_layout_carries_complete_transaction_state() {
        let config = layout_config();
        let capacity = 7;
        let layout = Qwen35ArenaLayout::new(&config, capacity).unwrap();

        assert_eq!(layout.kv_k.size, 2 * capacity * 4 * 4);
        assert_eq!(layout.kv_v.size, 2 * capacity * 4 * 4);
        assert_eq!(layout.conv_state.size, 2 * 2 * 16 * 4);
        assert_eq!(layout.ssm_state.size, 2 * 2 * 4 * 4 * 4);
        assert!(layout.total_size() >= layout.ssm_state.end());
    }

    #[test]
    fn mrope_coefficients_reproduce_the_cpu_transform() {
        let positions = [1, 2, 3, 4];
        let sections = [1, 1, 0, 0];
        let mut coefficients = [0.0f32; 4];
        fill_mrope(&mut coefficients, positions, sections, 10_000.0).unwrap();

        let mut expected = [1.0f32, 2.0, 3.0, 4.0];
        crate::ops::rope_mrope(&mut expected, positions, sections, 4, 10_000.0);
        let actual = [
            1.0f32.mul_add(coefficients[0], -(3.0 * coefficients[2])),
            2.0f32.mul_add(coefficients[1], -(4.0 * coefficients[3])),
            1.0f32.mul_add(coefficients[2], 3.0 * coefficients[0]),
            2.0f32.mul_add(coefficients[3], 4.0 * coefficients[1]),
        ];
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
}
