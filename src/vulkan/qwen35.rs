use super::ops::{
    required_activation_contracts, ArenaRegion, ExpertSelection, GpuMatmulSpec, GpuWeightFormat,
    OperatorBindings, Qwen3Ops, TokenCommands,
};
use super::qwen3::UploadedBuffers;
use super::{GpuBuffer, VulkanContext, VulkanError};
use crate::compute::state::TokenCommitState;
use crate::core::scratchpad::KvCache;
use crate::core::tensor::GGMLType;
use crate::models::qwen35::trunk::{config::Qwen35Moe, weights::Qwen35MoeWeights};
use crate::models::qwen35::{Qwen35Config, Qwen35Model};
use crate::ops::kernel::Weight;

mod moe;

#[derive(Debug, Clone)]
struct EligibilityFacts {
    architecture: String,
    weight_formats: Vec<GGMLType>,
    unrecorded_operations: Vec<&'static str>,
}

fn check_eligibility(facts: &EligibilityFacts) -> Result<(), String> {
    if !matches!(facts.architecture.as_str(), "qwen35" | "qwen35moe") {
        return Err(format!("unsupported architecture {}", facts.architecture));
    }
    if facts.weight_formats.is_empty() {
        return Err("unsupported Qwen3.5 Vulkan weight format: no weights".into());
    }
    for &format in &facts.weight_formats {
        GpuWeightFormat::from_ggml_type(format).map_err(|error| error.to_string())?;
    }
    if let Some(operation) = facts.unrecorded_operations.first() {
        return Err(format!(
            "Qwen3.5 Vulkan operation {operation} has no recorder"
        ));
    }
    Ok(())
}

fn check_device_eligibility(shader_float16: bool) -> Result<(), String> {
    shader_float16
        .then_some(())
        .ok_or_else(|| "selected Vulkan device does not support shaderFloat16".into())
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
    commit_shadow_state_chunk(
        kv_cache,
        conv_states,
        ssm_states,
        position,
        1,
        capacity,
        kv_stride,
        k_delta,
        v_delta,
        conv_state,
        ssm_state,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_shadow_state_chunk(
    kv_cache: &mut KvCache,
    conv_states: &mut [Vec<f32>],
    ssm_states: &mut [Vec<f32>],
    position: usize,
    rows: usize,
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
        .and_then(|count| count.checked_mul(rows))
        .ok_or_else(|| "Qwen3.5 KV delta length overflow".to_string())?;
    if rows == 0
        || position.checked_add(rows).is_none_or(|end| end > capacity)
        || k_delta.len() != delta_len
        || v_delta.len() != delta_len
    {
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

    if k_delta
        .iter()
        .chain(v_delta)
        .chain(conv_state)
        .chain(ssm_state)
        .any(|value| !value.is_finite())
    {
        return Err("Qwen3.5 Vulkan chunk contains non-finite shadow state".into());
    }
    for layer in 0..layer_count {
        let source = layer * rows * kv_stride;
        let target = (layer * capacity + position) * kv_stride;
        cache.k[target..target + rows * kv_stride]
            .copy_from_slice(&k_delta[source..source + rows * kv_stride]);
        for row in 0..rows {
            for dimension in 0..kv_stride {
                cache.v[(layer * kv_stride + dimension) * capacity + position + row] =
                    v_delta[source + row * kv_stride + dimension];
            }
        }
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
    router_logits: ArenaRegion,
    router_probs: ArenaRegion,
    expert_ids: ArenaRegion,
    expert_weights: ArenaRegion,
    expert_weights_norm: ArenaRegion,
    expert_gate: ArenaRegion,
    expert_up: ArenaRegion,
    expert_down: ArenaRegion,
    shared_gate: ArenaRegion,
    moe_scores: ArenaRegion,
    moe_status: ArenaRegion,
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
    #[cfg(test)]
    pub(crate) fn new(
        config: &Qwen35Config,
        capacity: usize,
    ) -> Result<Self, crate::vulkan::VulkanError> {
        Self::new_rows(config, capacity, 1)
    }

    fn new_rows(config: &Qwen35Config, capacity: usize, rows: usize) -> Result<Self, VulkanError> {
        let layer_count = config.n_layer_impl();
        let head_dim = config.n_embd_head();
        let value_heads = config.ssm_dt_rank;
        let ffn_width = config
            .moe
            .map_or(config.n_ff, |m| m.n_ff_exp.max(m.n_ff_shexp));
        let expert_slots = config
            .moe
            .map_or(1, |m| m.n_expert_used + usize::from(m.n_ff_shexp > 0));
        if [
            config.n_embd,
            ffn_width,
            config.n_head,
            config.n_head_kv,
            head_dim,
            config.vocab_size,
            layer_count,
            capacity,
            rows,
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
        let (experts, used, expert_ff) = config
            .moe
            .as_ref()
            .map(|m| (m.n_expert, m.n_expert_used, m.n_ff_exp))
            .unwrap_or((1, 1, 1));
        let quant_len = config
            .n_embd
            .max(ffn_width)
            .max(expert_ff)
            .max(dense_q)
            .max(value_dim);
        let kv_cache = qwen35_product("KV cache", &[layer_count, capacity, dense_kv])?;
        let kv_delta = qwen35_product("KV delta", &[layer_count, rows, dense_kv])?;
        let conv_state = qwen35_product(
            "convolution state",
            &[layer_count, config.ssm_d_conv, conv_dim],
        )?;
        let ssm_state = qwen35_product(
            "SSM state",
            &[layer_count, value_heads, head_v_dim, head_v_dim],
        )?;
        let mut cursor = 0usize;
        let mut row_region =
            |width| qwen35_f32_region(&mut cursor, qwen35_product("row buffer", &[rows, width])?);
        let x = row_region(config.n_embd)?;
        let normed = row_region(config.n_embd)?;
        let raw_qkv = row_region(q_raw.max(conv_dim))?;
        let raw_k = row_region(dense_kv)?;
        let q = row_region(dense_q.max(key_dim))?;
        let k = row_region(dense_kv.max(key_dim))?;
        let v = row_region(dense_kv.max(value_dim))?;
        let z = row_region(dense_q.max(value_dim))?;
        let beta = row_region(value_heads)?;
        let alpha = row_region(value_heads)?;
        let attn = row_region(dense_q.max(value_dim))?;
        let projection = row_region(config.n_embd)?;
        let router_logits = row_region(experts)?;
        let router_probs = row_region(experts)?;
        let expert_ids = row_region(used)?;
        let expert_weights = row_region(used)?;
        let expert_weights_norm = row_region(used)?;
        let expert_gate = row_region(qwen35_product("expert gate", &[used, expert_ff])?)?;
        let expert_up = row_region(qwen35_product("expert up", &[used, expert_ff])?)?;
        let expert_down = row_region(qwen35_product("expert down", &[used, config.n_embd])?)?;
        let shared_gate = row_region(1)?;
        let rope = row_region(config.rope_dimension_count)?;
        let moe_scores = row_region(expert_slots)?;
        let moe_status = qwen35_f32_region(&mut cursor, 1)?;
        let ffn_rows = rows.max(expert_slots);
        let ffn_gate = qwen35_f32_region(
            &mut cursor,
            qwen35_product("FFN slots", &[ffn_rows, ffn_width])?,
        )?;
        let ffn_up = qwen35_f32_region(
            &mut cursor,
            qwen35_product("FFN slots", &[ffn_rows, ffn_width])?,
        )?;
        let down = qwen35_f32_region(
            &mut cursor,
            qwen35_product("FFN outputs", &[ffn_rows, config.n_embd])?,
        )?;
        let logits = qwen35_f32_region(&mut cursor, config.vocab_size)?;
        let quant_rows = qwen35_product("quantized rows", &[rows, quant_len.div_ceil(4) * 4])?;
        let q8 = qwen35_region(&mut cursor, quant_rows)?;
        let q8_scales = qwen35_f32_region(
            &mut cursor,
            qwen35_product("Q8 scales", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q4_1_input_sums = qwen35_f32_region(
            &mut cursor,
            qwen35_product("Q4 sums", &[rows, quant_len.div_ceil(32)])?,
        )?;
        let q8k = qwen35_region(&mut cursor, quant_rows)?;
        let q8k_scales = qwen35_f32_region(
            &mut cursor,
            qwen35_product("Q8K scales", &[rows, quant_len.div_ceil(256)])?,
        )?;
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
            router_logits,
            router_probs,
            expert_ids,
            expert_weights,
            expert_weights_norm,
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            moe_scores,
            moe_status,
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
        let (cosine, sine) = crate::ops::rope_sin_cos(theta[axis]);
        coefficients[index] = cosine;
        coefficients[index + half] = sine;
        for value in &mut theta {
            *value *= theta_scale;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum WeightBindings<const N: usize> {
    Grouped(OperatorBindings),
    Split([OperatorBindings; N]),
}

#[derive(Clone, Copy)]
struct DenseBindings {
    qkv: WeightBindings<3>,
    prepare: OperatorBindings,
    output: OperatorBindings,
}

#[derive(Clone, Copy)]
struct RecurrentBindings {
    qkv_gate_beta: WeightBindings<3>,
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
struct SharedExpertBindings {
    gate: OperatorBindings,
    gate_up: WeightBindings<2>,
    down: OperatorBindings,
}

#[derive(Clone, Copy)]
struct MoeBindings {
    router: OperatorBindings,
    gate_up: OperatorBindings,
    down: OperatorBindings,
    shared: Option<SharedExpertBindings>,
}

enum FfnBindings {
    Dense {
        gate_up: WeightBindings<2>,
        down: OperatorBindings,
    },
    Moe(MoeBindings),
    MoeHostRouted(moe::MoeBindings),
}

struct LayerBindings {
    attention_norm: OperatorBindings,
    attention: AttentionBindings,
    post_attention_norm: OperatorBindings,
    ffn: FfnBindings,
}

pub(crate) struct Qwen35GpuChunkResult<'a> {
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
    pub(crate) capacity: usize,
    max_rows: usize,
    #[cfg(test)]
    pub(crate) fail_after_row: Option<usize>,
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
        Self::try_new_rows(model, capacity, 1, context)
    }

    pub(crate) fn try_new_rows(
        model: &Qwen35Model<'_>,
        capacity: usize,
        max_rows: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        if let Err(reason) = check_device_eligibility(context.supports_shader_float16()) {
            eprintln!("[GPU] Qwen3.5 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        let facts = eligibility_facts(model);
        if let Err(reason) = check_eligibility(&facts) {
            eprintln!("[GPU] Qwen3.5 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        if let Err(reason) = validate_executor_shape(&model.config, capacity) {
            eprintln!("[GPU] Qwen3.5 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        context
            .validate_qwen3_core_contract()
            .map_err(VulkanError::UnsupportedShape)?;
        let contracts = required_activation_contracts(&facts.weight_formats)
            .map_err(VulkanError::UnsupportedShape)?;
        if contracts.q8_0 {
            context
                .validate_q8_0_activation_contract()
                .map_err(VulkanError::UnsupportedShape)?;
        }
        if contracts.q8_k {
            context
                .validate_q8_k_activation_contract()
                .map_err(VulkanError::UnsupportedShape)?;
        }

        let config = model.config.clone();
        let layer_count = config.n_layer_impl();
        let layout = Qwen35ArenaLayout::new_rows(&config, capacity, max_rows)?;
        let descriptor_capacity = model.layers.iter().try_fold(3usize, |count, layer| {
            let experts = layer
                .moe
                .as_ref()
                .filter(|weights| device_moe_format(weights).is_none())
                .map_or(0, |weights| weights.n_expert);
            experts
                .checked_mul(3)
                .and_then(|n| n.checked_add(18))
                .and_then(|n| count.checked_add(n))
                .ok_or(VulkanError::OutOfMemory)
        })?;
        // Host routing reads between recordings; keep its arena mapped.
        let host_routed = model
            .layers
            .iter()
            .filter_map(|layer| layer.moe.as_ref())
            .any(|weights| device_moe_format(weights).is_none());
        let mut ops = if host_routed {
            Qwen3Ops::new_with_size(context, layout.total_size(), descriptor_capacity)?
        } else {
            Qwen3Ops::new_device_local_with_size(context, layout.total_size(), descriptor_capacity)?
        };
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
                let qkv_gate_beta = bind_weight_group(
                    &mut ops,
                    &mut buffers,
                    [
                        (wqkv, "Qwen3.5 recurrent QKV"),
                        (gate, "Qwen3.5 recurrent gate"),
                        (beta, "Qwen3.5 recurrent beta"),
                    ],
                )?;
                let alpha = bind_weight(&mut ops, &mut buffers, alpha, "Qwen3.5 recurrent alpha")?;
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
                let output =
                    bind_weight(&mut ops, &mut buffers, output, "Qwen3.5 recurrent output")?;
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
                let qkv = bind_weight_group(
                    &mut ops,
                    &mut buffers,
                    [
                        (wq, "Qwen3.5 dense Q"),
                        (wk, "Qwen3.5 dense K"),
                        (wv, "Qwen3.5 dense V"),
                    ],
                )?;
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
                let output = bind_weight(&mut ops, &mut buffers, wo, "Qwen3.5 dense output")?;
                AttentionBindings::Dense(DenseBindings {
                    qkv,
                    prepare,
                    output,
                })
            };

            let post_norm_buffer = buffers.upload_f32(&layer.attn_post_norm)?;
            let post_attention_norm = ops.bind_buffers(&[post_norm_buffer])?;
            let ffn = if let Some(cfg) = &config.moe {
                let moe = layer
                    .moe
                    .as_ref()
                    .ok_or_else(|| VulkanError::UnsupportedShape("missing MoE weights".into()))?;
                validate_moe(moe, cfg, config.n_embd).map_err(VulkanError::UnsupportedShape)?;
                if let Some(format) = device_moe_format(moe) {
                    let gate_up_buffers =
                        [buffers.upload(moe.gate_exps)?, buffers.upload(moe.up_exps)?];
                    let gate_up = ops.bind_weight_buffers(&gate_up_buffers, &[format; 2])?;
                    let down_buffer = buffers.upload(moe.down_exps)?;
                    let down = ops.bind_weight_buffers(&[down_buffer], &[format])?;
                    let router = bind_weight(&mut ops, &mut buffers, &moe.gate_inp, "MoE router")?;
                    let shared = if cfg.n_ff_shexp > 0 {
                        let gate_weight =
                            Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32 {
                                data: moe.gate_inp_shexp.as_ref().unwrap().clone(),
                                n_in: config.n_embd,
                                n_out: 1,
                            });
                        Some(SharedExpertBindings {
                            gate: bind_weight(
                                &mut ops,
                                &mut buffers,
                                &gate_weight,
                                "shared expert gate",
                            )?,
                            gate_up: bind_weight_group(
                                &mut ops,
                                &mut buffers,
                                [
                                    (
                                        moe.gate_shexp.as_ref().unwrap(),
                                        "shared expert gate projection",
                                    ),
                                    (moe.up_shexp.as_ref().unwrap(), "shared expert up"),
                                ],
                            )?,
                            down: bind_weight(
                                &mut ops,
                                &mut buffers,
                                moe.down_shexp.as_ref().unwrap(),
                                "shared expert down",
                            )?,
                        })
                    } else {
                        None
                    };
                    FfnBindings::Moe(MoeBindings {
                        router,
                        gate_up,
                        down,
                        shared,
                    })
                } else {
                    FfnBindings::MoeHostRouted(moe::MoeBindings::new(
                        &mut ops,
                        &mut buffers,
                        moe,
                        *cfg,
                    )?)
                }
            } else {
                FfnBindings::Dense {
                    gate_up: bind_weight_group(
                        &mut ops,
                        &mut buffers,
                        [
                            (
                                required_weight(layer.ffn_gate.as_ref(), layer_index, "FFN gate")?,
                                "Qwen3.5 FFN gate",
                            ),
                            (
                                required_weight(layer.ffn_up.as_ref(), layer_index, "FFN up")?,
                                "Qwen3.5 FFN up",
                            ),
                        ],
                    )?,
                    down: bind_weight(
                        &mut ops,
                        &mut buffers,
                        required_weight(layer.ffn_down.as_ref(), layer_index, "FFN down")?,
                        "Qwen3.5 FFN down",
                    )?,
                }
            };
            layers.push(LayerBindings {
                attention_norm,
                attention,
                post_attention_norm,
                ffn,
            });
        }

        let output_norm_buffer = buffers.upload_f32(&model.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output = bind_weight(
            &mut ops,
            &mut buffers,
            &model.output_weight,
            "Qwen3.5 output",
        )?;
        let dense_kv = config
            .n_head_kv
            .checked_mul(config.n_embd_head())
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = layer_count
            .checked_mul(dense_kv)
            .and_then(|count| count.checked_mul(max_rows))
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
            max_rows,
            #[cfg(test)]
            fail_after_row: None,
            commit_state: TokenCommitState::new(0, capacity),
            rope: vec![0.0; qwen35_product("mRoPE rows", &[rope_dim, max_rows])?],
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
    ) -> Result<Qwen35GpuChunkResult<'a>, VulkanError> {
        self.forward_chunk(input, cache_position, &[mrope_position], 1)
    }

    pub(crate) fn forward_chunk<'a>(
        &'a mut self,
        input: &[f32],
        base_position: usize,
        positions: &[[usize; 4]],
        rows: usize,
    ) -> Result<Qwen35GpuChunkResult<'a>, VulkanError> {
        let count = qwen35_product(
            "KV delta",
            &[
                self.config.n_layer_impl(),
                rows,
                self.config.n_head_kv,
                self.config.n_embd_head(),
            ],
        )?;
        if let Err(error) = self.commit_state.begin(base_position, rows) {
            self.commit_state.abort();
            return Err(VulkanError::UnsupportedShape(error));
        }
        if let Err(error) = self.forward_chunk_inner(input, base_position, positions, rows) {
            self.commit_state.abort();
            return Err(error);
        }
        Ok(Qwen35GpuChunkResult {
            logits: &self.logits,
            k_delta: &self.k_delta[..count],
            v_delta: &self.v_delta[..count],
            conv_state: &self.conv_state,
            ssm_state: &self.ssm_state,
        })
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

    fn record_weight_group<const N: usize>(
        &self,
        commands: &TokenCommands<'_>,
        bindings: WeightBindings<N>,
        input: ArenaRegion,
        outputs: [(ArenaRegion, usize); N],
        n_in: usize,
        rows: usize,
    ) -> Result<(), VulkanError> {
        match bindings {
            WeightBindings::Grouped(bindings) => {
                self.record_weights(commands, bindings, input, &outputs, n_in, rows)
            }
            WeightBindings::Split(bindings) => {
                for (bindings, output) in bindings.into_iter().zip(outputs) {
                    self.record_weights(commands, bindings, input, &[output], n_in, rows)?;
                }
                Ok(())
            }
        }
    }

    fn record_moe(
        &self,
        commands: &TokenCommands<'_>,
        bindings: MoeBindings,
        rows: usize,
    ) -> Result<(), VulkanError> {
        let cfg = self.config.moe.as_ref().expect("validated MoE config");
        let l = self.layout;
        let width = self.config.n_embd;
        self.record_weights(
            commands,
            bindings.router,
            l.normed,
            &[(l.router_logits, cfg.n_expert)],
            width,
            rows,
        )?;
        self.ops.record_qwen35_moe_route(
            commands,
            l.router_logits,
            l.router_probs,
            l.expert_ids,
            l.expert_weights,
            l.expert_weights_norm,
            cfg.n_expert,
            cfg.n_expert_used,
            cfg.weights_scale,
            rows,
        )?;
        for slot in 0..cfg.n_expert_used {
            let outputs = [
                (
                    l.expert_gate.shifted(slot * cfg.n_ff_exp * 4),
                    cfg.n_ff_exp,
                    cfg.n_expert_used * cfg.n_ff_exp * 4,
                ),
                (
                    l.expert_up.shifted(slot * cfg.n_ff_exp * 4),
                    cfg.n_ff_exp,
                    cfg.n_expert_used * cfg.n_ff_exp * 4,
                ),
            ];
            self.ops.record_selected_weight_matmul_rows(
                commands,
                bindings.gate_up,
                l.normed,
                l.q8,
                l.q8_scales,
                l.q4_1_input_sums,
                l.q8k,
                l.q8k_scales,
                &outputs,
                width,
                rows,
                width,
                Some(ExpertSelection {
                    ids: l.expert_ids,
                    count: cfg.n_expert,
                    used: cfg.n_expert_used,
                    slot,
                }),
            )?;
        }
        self.ops.record_silu_mul_rows_into(
            commands,
            l.expert_gate,
            l.expert_up,
            l.expert_gate,
            cfg.n_ff_exp,
            rows * cfg.n_expert_used,
            false,
        )?;
        for slot in 0..cfg.n_expert_used {
            self.ops.record_selected_weight_matmul_rows(
                commands,
                bindings.down,
                l.expert_gate.shifted(slot * cfg.n_ff_exp * 4),
                l.q8,
                l.q8_scales,
                l.q4_1_input_sums,
                l.q8k,
                l.q8k_scales,
                &[(
                    l.expert_down.shifted(slot * width * 4),
                    width,
                    cfg.n_expert_used * width * 4,
                )],
                cfg.n_ff_exp,
                rows,
                cfg.n_expert_used * cfg.n_ff_exp,
                Some(ExpertSelection {
                    ids: l.expert_ids,
                    count: cfg.n_expert,
                    used: cfg.n_expert_used,
                    slot,
                }),
            )?;
        }
        if let Some(shared) = bindings.shared {
            self.record_weights(
                commands,
                shared.gate,
                l.normed,
                &[(l.shared_gate, 1)],
                width,
                rows,
            )?;
            self.record_weight_group(
                commands,
                shared.gate_up,
                l.normed,
                [(l.ffn_gate, cfg.n_ff_shexp), (l.ffn_up, cfg.n_ff_shexp)],
                width,
                rows,
            )?;
            self.ops.record_silu_mul_rows_into(
                commands,
                l.ffn_gate,
                l.ffn_up,
                l.ffn_gate,
                cfg.n_ff_shexp,
                rows,
                false,
            )?;
            self.record_weights(
                commands,
                shared.down,
                l.ffn_gate,
                &[(l.down, width)],
                cfg.n_ff_shexp,
                rows,
            )?;
        }
        self.ops.record_qwen35_moe_reduce(
            commands,
            l.x,
            l.expert_down,
            l.expert_weights,
            l.down,
            l.shared_gate,
            width,
            cfg.n_expert_used,
            rows,
            bindings.shared.is_some(),
        )
    }

    #[cfg(feature = "parity-trace")]
    fn trace_region(
        &self,
        name: &str,
        layer: Option<usize>,
        step: usize,
        region: ArenaRegion,
        count: usize,
        ids: bool,
    ) -> Result<(), VulkanError> {
        if crate::parity_trace::enabled(name) {
            let mut values = vec![0.0; count];
            self.ops.read_f32_into(region, &mut values)?;
            if ids {
                for value in &mut values {
                    *value = value.to_bits() as f32;
                }
            }
            crate::parity_trace::checkpoint_at(name, layer, Some(step), &[count], &values)
                .map_err(|e| VulkanError::UnsupportedShape(e.to_string()))?;
        }
        Ok(())
    }

    fn forward_chunk_inner(
        &mut self,
        input: &[f32],
        cache_position: usize,
        positions: &[[usize; 4]],
        rows: usize,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        let _numerics = crate::ops::ScalarDispatchScope::new(config.reference_arithmetic());
        let input_count = qwen35_product("input rows", &[rows, config.n_embd])?;
        if rows > self.max_rows
            || positions.len() != rows
            || input.len() != input_count
            || input.iter().any(|v| !v.is_finite())
        {
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
        let delta_bytes = qwen35_product("KV delta bytes", &[layer_count, rows, dense_kv, 4])?;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let value_heads = config.ssm_dt_rank;
        let recurrent_head_dim = config.head_v_dim();

        // Staged transfers take the submission guard themselves, before recording.
        self.ops.write_f32(self.layout.x, input)?;
        self.ops.zero_region(self.layout.moe_status)?;
        // Recurrent layers have no dense delta; active rows change its layer stride.
        for region in [self.layout.kv_delta_k, self.layout.kv_delta_v] {
            self.ops.zero_region(ArenaRegion {
                offset: region.offset,
                size: delta_bytes,
            })?;
        }
        #[cfg(test)]
        let failure = match self.fail_after_row.take() {
            Some(row) if row < rows => Some(row),
            Some(row) => {
                self.fail_after_row = Some(row - rows);
                None
            }
            None => None,
        };
        for (row, &position) in positions.iter().enumerate() {
            fill_mrope(
                &mut self.rope
                    [row * config.rope_dimension_count..(row + 1) * config.rope_dimension_count],
                position,
                config.rope_dimension_sections,
                config.rope_freq_base,
            )
            .map_err(VulkanError::UnsupportedShape)?;
        }
        self.ops.write_f32(
            self.layout.rope,
            &self.rope[..rows * config.rope_dimension_count],
        )?;
        let mut commands = TokenCommands::begin(self.context)?;

        // Flush once a chunk reaches this many dispatches. Recording every layer
        // of a chunk into one command buffer and submitting it as a single unit
        // produces wrong logits on Meteor Lake: the LM head then reads a hidden
        // state that never reflects the recorded layers. Chunking keeps each
        // submission short enough for the drivers to honour every dispatch.
        const MAX_DISPATCHES_PER_CHUNK: usize = 16;
        let mut chunk_start = self.ops.recorded_dispatch_count();
        for (layer_index, bindings) in self.layers.iter().enumerate() {
            if self.ops.recorded_dispatch_count() - chunk_start >= MAX_DISPATCHES_PER_CHUNK {
                commands = commands.flush()?;
                chunk_start = self.ops.recorded_dispatch_count();
            }
            self.ops.record_rms_norm_rows(
                &commands,
                bindings.attention_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.norm_eps,
                rows,
                config.n_embd,
                config.n_embd,
            )?;
            match bindings.attention {
                AttentionBindings::Dense(dense) => {
                    self.record_weight_group(
                        &commands,
                        dense.qkv,
                        self.layout.normed,
                        [
                            (self.layout.raw_qkv, dense_raw_q),
                            (self.layout.raw_k, dense_kv),
                            (self.layout.v, dense_kv),
                        ],
                        config.n_embd,
                        rows,
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
                        rows,
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
                        cache_position,
                        self.capacity,
                        config.n_head,
                        config.n_head_kv,
                        head_dim,
                        rows,
                    )?;
                    self.record_weights(
                        &commands,
                        dense.output,
                        self.layout.attn,
                        &[(self.layout.projection, config.n_embd)],
                        dense_q,
                        rows,
                    )?;
                }
                AttentionBindings::Recurrent(recurrent) => {
                    self.record_weight_group(
                        &commands,
                        recurrent.qkv_gate_beta,
                        self.layout.normed,
                        [
                            (self.layout.raw_qkv, conv_dim),
                            (self.layout.z, value_dim),
                            (self.layout.beta, value_heads),
                        ],
                        config.n_embd,
                        rows,
                    )?;
                    self.record_weights(
                        &commands,
                        recurrent.alpha,
                        self.layout.normed,
                        &[(self.layout.alpha, value_heads)],
                        config.n_embd,
                        rows,
                    )?;
                    #[cfg(test)]
                    let scan_rows = failure.map_or(rows, |row| row + 1);
                    #[cfg(not(test))]
                    let scan_rows = rows;
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
                        scan_rows,
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
                        scan_rows,
                    )?;
                    #[cfg(feature = "parity-trace")]
                    if rows == 1 && std::env::var_os("RMI_PARITY_TRACE").is_some() {
                        commands.submit_and_wait()?;
                        for (name, region, count) in [
                            ("q_conv_predelta", self.layout.q, key_dim),
                            ("k_conv_predelta", self.layout.k, key_dim),
                            ("final_output", self.layout.attn, value_dim),
                        ] {
                            self.trace_region(
                                &format!("{name}-{layer_index}"),
                                Some(layer_index),
                                cache_position,
                                region,
                                count,
                                false,
                            )?;
                        }
                        commands = TokenCommands::begin(self.context)?;
                    }
                    #[cfg(test)]
                    if let Some(row) = failure {
                        commands.submit_and_wait()?;
                        return Err(VulkanError::UnsupportedShape(format!(
                            "injected Qwen3.5 GPU failure after recurrent row {row}"
                        )));
                    }
                    self.record_weights(
                        &commands,
                        recurrent.output,
                        self.layout.attn,
                        &[(self.layout.projection, config.n_embd)],
                        value_dim,
                        rows,
                    )?;
                }
            }
            self.ops.record_add_rows(
                &commands,
                self.layout.x,
                self.layout.projection,
                config.n_embd,
                rows,
            )?;
            self.ops.record_rms_norm_rows(
                &commands,
                bindings.post_attention_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.norm_eps,
                rows,
                config.n_embd,
                config.n_embd,
            )?;
            #[cfg(feature = "parity-trace")]
            if rows == 1 && std::env::var_os("RMI_PARITY_TRACE").is_some() {
                commands.submit_and_wait()?;
                for (name, region) in [
                    ("attn_residual", self.layout.x),
                    ("attn_post_norm", self.layout.normed),
                ] {
                    self.trace_region(
                        &format!("{name}-{layer_index}"),
                        Some(layer_index),
                        cache_position,
                        region,
                        config.n_embd,
                        false,
                    )?;
                }
                commands = TokenCommands::begin(self.context)?;
            }
            match &bindings.ffn {
                FfnBindings::Moe(moe) => self.record_moe(&commands, *moe, rows)?,
                FfnBindings::MoeHostRouted(moe) => {
                    commands = moe.forward(self, commands, rows, layer_index, cache_position)?;
                    chunk_start = self.ops.recorded_dispatch_count();
                }
                FfnBindings::Dense { gate_up, down } => {
                    self.record_weight_group(
                        &commands,
                        *gate_up,
                        self.layout.normed,
                        [
                            (self.layout.ffn_gate, config.n_ff),
                            (self.layout.ffn_up, config.n_ff),
                        ],
                        config.n_embd,
                        rows,
                    )?;
                    self.ops.record_silu_mul_rows(
                        &commands,
                        self.layout.ffn_gate,
                        self.layout.ffn_up,
                        config.n_ff,
                        rows,
                    )?;
                    self.record_weights(
                        &commands,
                        *down,
                        self.layout.ffn_gate,
                        &[(self.layout.down, config.n_embd)],
                        config.n_ff,
                        rows,
                    )?;
                    self.ops.record_add_rows(
                        &commands,
                        self.layout.x,
                        self.layout.down,
                        config.n_embd,
                        rows,
                    )?;
                }
            }
            #[cfg(feature = "parity-trace")]
            if std::env::var_os("RMI_PARITY_TRACE").is_some() {
                if rows != 1 {
                    return Err(VulkanError::UnsupportedShape(
                        "Vulkan parity tracing requires single-token chunks".into(),
                    ));
                }
                commands.submit_and_wait()?;
                self.trace_region(
                    &format!("l_out-{layer_index}"),
                    Some(layer_index),
                    cache_position,
                    self.layout.x,
                    config.n_embd,
                    false,
                )?;
                if let (Some(moe), FfnBindings::Moe(_)) = (&config.moe, &bindings.ffn) {
                    for (name, region, count, ids) in [
                        (
                            "ffn_moe_logits",
                            self.layout.router_logits,
                            moe.n_expert,
                            false,
                        ),
                        (
                            "ffn_moe_probs",
                            self.layout.router_probs,
                            moe.n_expert,
                            false,
                        ),
                        (
                            "ffn_moe_topk",
                            self.layout.expert_ids,
                            moe.n_expert_used,
                            true,
                        ),
                        (
                            "ffn_moe_weights_norm",
                            self.layout.expert_weights_norm,
                            moe.n_expert_used,
                            false,
                        ),
                    ] {
                        self.trace_region(
                            name,
                            Some(layer_index),
                            cache_position,
                            region,
                            count,
                            ids,
                        )?;
                    }
                }
                commands = TokenCommands::begin(self.context)?;
            }
            #[cfg(feature = "parity-trace")]
            {
                let name = format!("l_out-{layer_index}");
                if crate::parity_trace::enabled(&name) {
                    commands = commands.flush()?;
                    crate::parity_trace::report(crate::parity_trace::checkpoint_rows(
                        &name,
                        Some(layer_index),
                        &[rows, config.n_embd],
                        self.ops.read_f32(self.layout.x, rows * config.n_embd)?,
                    ));
                }
            }
        }

        let last_offset = (rows - 1) * config.n_embd * 4;
        let last = ArenaRegion {
            offset: self.layout.x.offset + last_offset,
            size: self.layout.x.size - last_offset,
        };
        self.ops.record_rms_norm(
            &commands,
            self.output_norm,
            last,
            self.layout.normed,
            config.n_embd,
            config.norm_eps,
        )?;
        commands.submit_and_wait()?;
        #[cfg(feature = "parity-trace")]
        self.trace_region(
            "result_norm",
            None,
            cache_position,
            self.layout.normed,
            config.n_embd,
            false,
        )?;
        let commands = TokenCommands::begin(self.context)?;
        self.record_weights(
            &commands,
            self.output,
            self.layout.normed,
            &[(self.layout.logits, config.vocab_size)],
            config.n_embd,
            1,
        )?;
        commands.submit_and_wait()?;

        #[cfg(feature = "parity-trace")]
        self.trace_region(
            "result_output",
            None,
            cache_position,
            self.layout.logits,
            config.vocab_size,
            false,
        )?;
        self.ops
            .read_f32_into(self.layout.logits, &mut self.logits)?;
        let delta_count = qwen35_product("KV delta", &[layer_count, rows, dense_kv])?;
        self.ops
            .read_f32_into(self.layout.kv_delta_k, &mut self.k_delta[..delta_count])?;
        self.ops
            .read_f32_into(self.layout.kv_delta_v, &mut self.v_delta[..delta_count])?;
        self.ops
            .read_f32_into(self.layout.conv_state, &mut self.conv_state)?;
        self.ops
            .read_f32_into(self.layout.ssm_state, &mut self.ssm_state)?;
        let mut hidden = vec![0.0; input_count];
        self.ops.read_f32_into(self.layout.x, &mut hidden)?;
        let mut moe_status = [0.0];
        self.ops
            .read_f32_into(self.layout.moe_status, &mut moe_status)?;
        if moe_status[0].to_bits() != 0
            || self
                .logits
                .iter()
                .chain(&self.k_delta[..delta_count])
                .chain(&self.v_delta[..delta_count])
                .chain(&self.conv_state)
                .chain(&self.ssm_state)
                .any(|v| !v.is_finite())
            || hidden.iter().any(|v| !v.is_finite())
        {
            return Err(VulkanError::UnsupportedShape(
                "Qwen3.5 Vulkan chunk produced non-finite output".into(),
            ));
        }
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

fn upload_weight(
    buffers: &mut UploadedBuffers,
    weight: &Weight<'_>,
    label: &str,
) -> Result<(GpuBuffer, GpuMatmulSpec), VulkanError> {
    let format = GpuMatmulSpec::prepared(weight)?;
    let bytes = validated_weight_bytes(weight, label)?;
    Ok((buffers.upload(bytes)?, format))
}

fn validated_weight_bytes<'a>(
    weight: &'a Weight<'_>,
    label: &str,
) -> Result<&'a [u8], VulkanError> {
    let bytes = weight.kernel.weight_bytes().ok_or_else(|| {
        VulkanError::UnsupportedShape(format!(
            "{label} does not expose {:?} storage bytes",
            weight.ggml_type
        ))
    })?;
    let elements = weight
        .n_in
        .checked_mul(weight.n_out)
        .ok_or(VulkanError::OutOfMemory)?;
    let (block_elements, block_bytes) = weight.ggml_type.type_traits();
    let expected = elements
        .div_ceil(block_elements)
        .checked_mul(block_bytes)
        .ok_or(VulkanError::OutOfMemory)?;
    if bytes.len() != expected {
        return Err(VulkanError::UnsupportedShape(format!(
            "{label} has {} {:?} bytes, expected {expected}",
            bytes.len(),
            weight.ggml_type
        )));
    }
    Ok(bytes)
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

fn bind_weight_group<const N: usize>(
    ops: &mut Qwen3Ops<'_>,
    buffers: &mut UploadedBuffers,
    weights: [(&Weight<'_>, &str); N],
) -> Result<WeightBindings<N>, VulkanError> {
    let mut gpu_buffers = Vec::with_capacity(N);
    let mut formats = Vec::with_capacity(N);
    for (weight, label) in weights {
        let (buffer, format) = upload_weight(buffers, weight, label)?;
        gpu_buffers.push(buffer);
        formats.push(format);
    }
    if formats.iter().all(|format| *format == formats[0]) {
        return ops
            .bind_weight_buffers_mode(
                &gpu_buffers,
                &formats.iter().map(|spec| spec.format).collect::<Vec<_>>(),
                formats[0].mode,
            )
            .map(WeightBindings::Grouped);
    }

    let split = gpu_buffers
        .iter()
        .zip(&formats)
        .map(|(buffer, format)| {
            ops.bind_weight_buffers_mode(
                std::slice::from_ref(buffer),
                std::slice::from_ref(&format.format),
                format.mode,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| VulkanError::OutOfMemory)?;
    Ok(WeightBindings::Split(split))
}

fn device_moe_format(moe: &Qwen35MoeWeights<'_>) -> Option<GpuWeightFormat> {
    (moe.gate_type == moe.up_type
        && moe.gate_type == moe.down_type
        && matches!(moe.gate_type, GGMLType::Q8_0 | GGMLType::F32))
    .then(|| GpuWeightFormat::from_ggml_type(moe.gate_type).unwrap())
}

fn validate_moe(moe: &Qwen35MoeWeights<'_>, cfg: &Qwen35Moe, width: usize) -> Result<(), String> {
    if cfg.n_expert == 0
        || cfg.n_expert_used == 0
        || cfg.n_expert_used > cfg.n_expert
        || cfg.n_ff_exp == 0
        || !cfg.weights_scale.is_finite()
        || (moe.n_expert, moe.n_ff_exp, moe.n_embd) != (cfg.n_expert, cfg.n_ff_exp, width)
        || (moe.gate_inp.n_in, moe.gate_inp.n_out) != (width, cfg.n_expert)
    {
        return Err("invalid or unsupported MoE dimensions/format".into());
    }
    for (data, format, stride, n_in, n_out) in [
        (
            moe.gate_exps,
            moe.gate_type,
            moe.stride_gate,
            width,
            cfg.n_ff_exp,
        ),
        (moe.up_exps, moe.up_type, moe.stride_up, width, cfg.n_ff_exp),
        (
            moe.down_exps,
            moe.down_type,
            moe.stride_down,
            cfg.n_ff_exp,
            width,
        ),
    ] {
        GpuWeightFormat::from_ggml_type(format).map_err(|error| error.to_string())?;
        let (block, bytes) = format.type_traits();
        let expected = (n_in / block)
            .checked_mul(bytes)
            .and_then(|v| v.checked_mul(n_out));
        if n_in % block != 0
            || expected != Some(stride)
            || stride.checked_mul(cfg.n_expert) != Some(data.len())
        {
            return Err("invalid MoE expert payload/stride".into());
        }
    }
    if cfg.n_ff_shexp > 0 {
        if moe
            .gate_inp_shexp
            .as_ref()
            .is_none_or(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
        {
            return Err("missing or invalid shared expert gate".into());
        }
        for (weight, n_in, n_out) in [
            (moe.gate_shexp.as_ref(), width, cfg.n_ff_shexp),
            (moe.up_shexp.as_ref(), width, cfg.n_ff_shexp),
            (moe.down_shexp.as_ref(), cfg.n_ff_shexp, width),
        ] {
            if weight.is_none_or(|w| (w.n_in, w.n_out) != (n_in, n_out)) {
                return Err("missing or invalid shared expert projection".into());
            }
        }
    } else if moe.up_shexp.is_some() || moe.gate_shexp.is_some() || moe.down_shexp.is_some() {
        return Err("unexpected shared expert projections".into());
    }
    Ok(())
}

fn eligibility_facts(model: &Qwen35Model<'_>) -> EligibilityFacts {
    let mut facts = EligibilityFacts {
        architecture: if model.config.moe.is_some() {
            "qwen35moe"
        } else {
            "qwen35"
        }
        .into(),
        weight_formats: vec![model.output_weight.ggml_type],
        unrecorded_operations: Vec::new(),
    };
    let layer_count = model.config.n_layer_impl();
    if model.layers.len() != layer_count || model.config.is_recurrent.len() < layer_count {
        facts.unrecorded_operations.push("layer_layout");
        return facts;
    }
    for (layer_index, layer) in model.layers.iter().enumerate() {
        if let Some(cfg) = &model.config.moe {
            if let Some(moe) = &layer.moe {
                if validate_moe(moe, cfg, model.config.n_embd).is_err() {
                    facts.unrecorded_operations.push("moe_weights");
                }
                facts.weight_formats.extend([
                    moe.gate_type,
                    moe.up_type,
                    moe.down_type,
                    moe.gate_inp.ggml_type,
                ]);
                facts.weight_formats.extend(
                    [
                        moe.gate_shexp.as_ref(),
                        moe.up_shexp.as_ref(),
                        moe.down_shexp.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|w| w.ggml_type),
                );
            } else {
                facts.unrecorded_operations.push("moe_weights");
            }
        } else {
            let ffn_weights = [
                layer.ffn_gate.as_ref(),
                layer.ffn_up.as_ref(),
                layer.ffn_down.as_ref(),
            ];
            if ffn_weights.iter().any(|weight| weight.is_none()) || model.config.n_ff == 0 {
                facts.unrecorded_operations.push("dense_ffn");
            }
            facts.weight_formats.extend(
                ffn_weights
                    .into_iter()
                    .flatten()
                    .map(|weight| weight.ggml_type),
            );
        }
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
        check_device_eligibility, check_eligibility, commit_shadow_state, fill_mrope,
        qwen35_product, validated_weight_bytes, EligibilityFacts, Qwen35ArenaLayout,
    };
    use crate::core::scratchpad::KvCache;
    use crate::core::tensor::GGMLType;
    use crate::models::qwen35::Qwen35Config;
    use crate::ops::kernel::{QuantizedTensor, Weight};

    #[test]
    fn qwen35_product_rejects_size_overflow() {
        let error = qwen35_product("test", &[usize::MAX, 2]).unwrap_err();
        assert!(error.to_string().contains("size overflows usize"));
    }

    #[test]
    fn commit_shadow_state_chunk_is_atomic_and_transposes_values() {
        let mut cache = KvCache::new_f32(2, 5, 2);
        let mut conv = vec![vec![1.0, 2.0]; 2];
        let mut ssm = vec![vec![3.0; 3]; 2];
        let keys = (10..22).map(|v| v as f32).collect::<Vec<_>>();
        let values = (30..42).map(|v| v as f32).collect::<Vec<_>>();
        for bad in 0..8 {
            let mut k = keys.clone();
            let mut v = values.clone();
            let mut c = vec![4.0; 4];
            let mut s = vec![5.0; 6];
            match bad {
                0 => {
                    k.pop();
                }
                1 => {
                    v.pop();
                }
                2 => {
                    c.pop();
                }
                3 => {
                    s.pop();
                }
                4 => k[2] = f32::NAN,
                5 => v[2] = f32::INFINITY,
                6 => c[2] = f32::NAN,
                _ => s[2] = f32::INFINITY,
            }
            assert!(super::commit_shadow_state_chunk(
                &mut cache, &mut conv, &mut ssm, 1, 3, 5, 2, &k, &v, &c, &s,
            )
            .is_err());
            let KvCache::F32(cache) = &cache else {
                unreachable!()
            };
            assert_eq!(cache.k, [0.0; 20]);
            assert_eq!(cache.v, [0.0; 20]);
            assert_eq!(conv, vec![vec![1.0, 2.0]; 2]);
            assert_eq!(ssm, vec![vec![3.0; 3]; 2]);
        }
        super::commit_shadow_state_chunk(
            &mut cache, &mut conv, &mut ssm, 1, 3, 5, 2, &keys, &values, &[4.0; 4], &[5.0; 6],
        )
        .unwrap();
        let KvCache::F32(cache) = cache else {
            unreachable!()
        };
        assert_eq!(
            cache.k,
            [
                0., 0., 10., 11., 12., 13., 14., 15., 0., 0., 0., 0., 16., 17., 18., 19., 20., 21.,
                0., 0.
            ]
        );
        assert_eq!(
            cache.v,
            [
                0., 30., 32., 34., 0., 0., 31., 33., 35., 0., 0., 36., 38., 40., 0., 0., 37., 39.,
                41., 0.
            ]
        );
        assert_eq!(conv, vec![vec![4.0; 2]; 2]);
        assert_eq!(ssm, vec![vec![5.0; 3]; 2]);
    }

    fn eligible_facts() -> EligibilityFacts {
        EligibilityFacts {
            architecture: "qwen35".into(),
            weight_formats: vec![GGMLType::BF16; 187],
            unrecorded_operations: Vec::new(),
        }
    }

    #[test]
    #[ignore = "requires a physical Vulkan GPU"]
    fn dense_rope_respects_reference_arithmetic() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        assert!(!context.is_software_icd());
        let mut model = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
        model.layers[0].attn_q_norm = Some(vec![100_000_008.0, 1.0, 100_000_000.0, 1.0]);
        let gpu = super::Qwen35VulkanSession::try_new(&model, 1, context)
            .unwrap()
            .unwrap();
        let super::AttentionBindings::Dense(dense) = gpu.layers[0].attention else {
            panic!("dense fixture");
        };
        let l = gpu.layout;
        gpu.ops.write_f32(l.raw_qkv, &[1.0; 8]).unwrap();
        gpu.ops.write_f32(l.raw_k, &[1.0; 4]).unwrap();
        gpu.ops.write_f32(l.v, &[1.0; 4]).unwrap();
        let cosine = f32::from_bits(1.0f32.to_bits() - 1);
        gpu.ops.write_f32(l.rope, &[cosine, 1.0, 1.0, 0.0]).unwrap();
        for reference in [false, true] {
            let _scope = crate::ops::ScalarDispatchScope::new(reference);
            let commands = super::TokenCommands::begin(context).unwrap();
            gpu.ops
                .record_qwen35_dense_prepare(
                    &commands,
                    dense.prepare,
                    l.raw_qkv,
                    l.raw_k,
                    l.v,
                    l.q,
                    l.z,
                    l.kv_k,
                    l.kv_v,
                    l.kv_delta_k,
                    l.kv_delta_v,
                    l.rope,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    4,
                    4,
                    0.0,
                    1,
                )
                .unwrap();
            commands.submit_and_wait().unwrap();
            // Cancellation makes fused vs separately rounded rotation observable.
            let expected = if crate::ops::scalar_mode() {
                0.0
            } else {
                100_000_008.0f32.mul_add(cosine, -100_000_000.0)
            };
            let mut output = [0.0; 4];
            gpu.ops.read_f32_into(l.q, &mut output).unwrap();
            let actual = output[0];
            assert!(
                (actual - expected).abs() <= 1e-6,
                "reference={reference} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn eligibility_rejects_missing_moe_weights() {
        let mut model = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
        assert_eq!(check_eligibility(&super::eligibility_facts(&model)), Ok(()));
        model.config.moe = Some(crate::models::qwen35::trunk::config::Qwen35Moe {
            n_expert: 256,
            n_expert_used: 8,
            n_ff_exp: 512,
            n_ff_shexp: 512,
            weights_scale: 1.0,
        });
        let layer = &mut model.layers[0];
        layer.ffn_gate = None;
        layer.ffn_up = None;
        layer.ffn_down = None;
        assert!(check_eligibility(&super::eligibility_facts(&model))
            .expect_err("MoE must not execute a graph that skips its experts")
            .contains("moe"));
    }

    fn tiny_moe_model() -> super::Qwen35Model<'static> {
        use crate::models::qwen35::trunk::{config::Qwen35Moe, weights::Qwen35MoeWeights};
        use crate::ops::kernel::QuantizedTensor;
        let mut model = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
        let cfg = Qwen35Moe {
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 3,
            n_ff_shexp: 5,
            weights_scale: 1.25,
        };
        let weight = |n_in, n_out, seed| {
            Weight::from_quantized(QuantizedTensor::F32 {
                data: (0..n_in * n_out)
                    .map(|i| ((i * 7 + seed) % 19) as f32 / 37.0 - 0.2)
                    .collect(),
                n_in,
                n_out,
            })
        };
        let bytes = |seed| -> &'static [u8] {
            Box::leak(
                (0..4 * 4 * 3)
                    .flat_map(|i| (((i * 7 + seed) % 23) as f32 / 31.0 - 0.3).to_le_bytes())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        };
        let layer = &mut model.layers[0];
        layer.ffn_gate = None;
        layer.ffn_up = None;
        layer.ffn_down = None;
        layer.moe = Some(Qwen35MoeWeights {
            gate_inp: weight(4, 4, 2),
            gate_exps: bytes(1),
            up_exps: bytes(5),
            down_exps: bytes(11),
            up_type: GGMLType::F32,
            gate_type: GGMLType::F32,
            down_type: GGMLType::F32,
            n_expert: 4,
            n_ff_exp: 3,
            n_embd: 4,
            stride_gate: 48,
            stride_up: 48,
            stride_down: 48,
            gate_inp_shexp: Some(vec![0.3, -0.5, 0.7, -0.9]),
            gate_shexp: Some(weight(4, 5, 4)),
            up_shexp: Some(weight(4, 5, 8)),
            down_shexp: Some(weight(5, 4, 12)),
        });
        model.output_weight = weight(4, 8, 3);
        model.config.moe = Some(cfg);
        model.config.n_ff = 5;
        model
    }

    #[test]
    fn eligibility_accepts_complete_moe_and_rejects_partial_shared_expert() {
        let mut model = tiny_moe_model();
        assert_eq!(check_eligibility(&super::eligibility_facts(&model)), Ok(()));
        model.layers[0].moe.as_mut().unwrap().down_shexp = None;
        assert!(check_eligibility(&super::eligibility_facts(&model)).is_err());
    }

    #[test]
    fn eligibility_rejects_malformed_experts() {
        for invalid in 0..5 {
            let mut model = tiny_moe_model();
            let cfg = model.config.moe.as_mut().unwrap();
            let weights = model.layers[0].moe.as_mut().unwrap();
            match invalid {
                0 => cfg.n_expert_used = cfg.n_expert + 1,
                1 => weights.stride_gate -= 1,
                2 => weights.down_exps = &weights.down_exps[1..],
                3 => weights.gate_type = GGMLType::F16,
                _ => cfg.weights_scale = f32::INFINITY,
            }
            assert!(check_eligibility(&super::eligibility_facts(&model)).is_err());
        }
    }

    #[test]
    #[ignore = "requires a physical Vulkan device"]
    fn moe_router_wide_sum_runs_on_vulkan() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        assert!(!context.is_software_icd());
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tools/oracle/sage_a15/router_fixture.json"
        ))
        .unwrap();
        let values = |name: &str| {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                .collect::<Vec<_>>()
        };
        let logits = values("ffn_moe_logits");
        let region = |slot: usize| super::ArenaRegion {
            offset: slot * 1024,
            size: 1024,
        };
        let ops = super::Qwen3Ops::new_with_size(context, 5 * 1024, 1).unwrap();
        ops.write_f32(region(0), &logits).unwrap();
        for scale in [1.0, 0.0] {
            let commands = super::TokenCommands::begin(context).unwrap();
            ops.record_qwen35_moe_route(
                &commands,
                region(0),
                region(1),
                region(2),
                region(3),
                region(4),
                256,
                8,
                scale,
                1,
            )
            .unwrap();
            commands.submit_and_wait().unwrap();
            let expected_ids = values("ffn_moe_topk");
            let ids = ops.read_f32(region(2), 8).unwrap();
            assert_eq!(
                ids.iter().map(|v| v.to_bits() as f32).collect::<Vec<_>>(),
                expected_ids
            );
            for slot in [3, 4] {
                for (&actual, expected) in ops
                    .read_f32(region(slot), 8)
                    .unwrap()
                    .iter()
                    .zip(values("ffn_moe_weights_norm"))
                {
                    assert!(
                        (actual - expected).abs() <= 1e-9 + 1e-8 * expected.abs(),
                        "router weight {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a physical Vulkan device"]
    fn moe_routing_and_experts_run_on_vulkan() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        assert!(!context.is_software_icd());
        let model = tiny_moe_model();
        let gpu = super::Qwen35VulkanSession::try_new_rows(&model, 8, 3, context)
            .unwrap()
            .unwrap();
        let super::FfnBindings::Moe(bindings) = gpu.layers[0].ffn else {
            panic!("missing MoE recorder")
        };
        let input = [
            1.0, -2.0, 0.5, 3.0, -1.5, 0.2, 2.1, -0.7, 0.0, 0.0, 0.0, 0.0,
        ];
        gpu.ops.write_f32(gpu.layout.normed, &input).unwrap();
        let commands = super::TokenCommands::begin(context).unwrap();
        gpu.record_moe(&commands, bindings, 3).unwrap();
        commands.submit_and_wait().unwrap();
        let mut actual = vec![0.0; 12];
        gpu.ops.read_f32_into(gpu.layout.x, &mut actual).unwrap();
        let moe = model.layers[0].moe.as_ref().unwrap();
        let cfg = model.config.moe.as_ref().unwrap();
        for (row, input) in input.chunks_exact(4).enumerate() {
            let mut expected = [0.0; 4];
            {
                let _cpu = crate::compute::ComputePolicy::Cpu.cpu_scope();
                crate::models::qwen35::trunk::moe::forward_qwen35moe_token(
                    moe,
                    cfg,
                    cfg.weights_scale,
                    input,
                    &mut expected,
                    &mut crate::ops::kernel::PreparedRows::new(1, 5),
                    &crate::ComputePool::new(1),
                    0,
                    row,
                )
                .unwrap();
            }
            for (&a, b) in actual[row * 4..row * 4 + 4].iter().zip(expected) {
                assert!(
                    a.is_finite() && (a - b).abs() <= 1e-5 + b.abs() * 1e-4,
                    "row={row}: {a} != {b}"
                );
            }
        }
        let mut ids = [0.0; 6];
        gpu.ops
            .read_f32_into(gpu.layout.expert_ids, &mut ids)
            .unwrap();
        assert_eq!(
            ids[4].to_bits(),
            0,
            "router ties choose ascending expert IDs"
        );
        assert_eq!(ids[5].to_bits(), 1);
        // All-equal logits: normalized weights include the configured scaling.
        let mut weights = [0.0; 6];
        gpu.ops
            .read_f32_into(gpu.layout.expert_weights, &mut weights)
            .unwrap();
        assert!((weights[4] - 0.625).abs() < 1e-7);
        // Non-finite routers propagate failure without forming an invalid expert address.
        gpu.ops
            .write_f32(gpu.layout.router_logits, &[f32::NAN, 1.0, 2.0, 3.0])
            .unwrap();
        let commands = super::TokenCommands::begin(context).unwrap();
        gpu.ops
            .record_qwen35_moe_route(
                &commands,
                gpu.layout.router_logits,
                gpu.layout.router_probs,
                gpu.layout.expert_ids,
                gpu.layout.expert_weights,
                gpu.layout.expert_weights_norm,
                4,
                2,
                1.0,
                1,
            )
            .unwrap();
        commands.submit_and_wait().unwrap();
        gpu.ops
            .read_f32_into(gpu.layout.expert_weights, &mut weights[..2])
            .unwrap();
        assert!(weights[..2].iter().all(|v| v.is_nan()));
        println!("physical GPU MoE: 3 rows, tie ordering, scaled weights, routed/shared output, nonfinite router checked");
    }

    #[test]
    #[ignore = "requires a physical Vulkan device"]
    fn moe_full_session_prefill_and_decode_match_cpu() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        assert!(!context.is_software_icd());
        for mixed in [false, true] {
            let mut model = tiny_moe_model();
            if mixed {
                model.config.n_ff = 64;
                model.config.moe.as_mut().unwrap().n_ff_exp = 64;
                let weights = model.layers[0].moe.as_mut().unwrap();
                // Keep every expert offset aligned for Vulkan storage subranges.
                for data in [
                    &mut weights.gate_exps,
                    &mut weights.up_exps,
                    &mut weights.down_exps,
                ] {
                    *data = Box::leak(
                        data.iter()
                            .copied()
                            .cycle()
                            .take(4 * 4 * 64 * 4)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    );
                }
                weights.n_ff_exp = 64;
                weights.stride_gate = 4 * 64 * 4;
                weights.stride_up = weights.stride_gate;
                weights.stride_down = weights.stride_gate;
                let gate = weights
                    .gate_exps
                    .chunks_exact(4)
                    .flat_map(|bytes| {
                        crate::ops::f32_to_f16(f32::from_le_bytes(bytes.try_into().unwrap()))
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                weights.gate_exps = Box::leak(gate.into_boxed_slice());
                weights.gate_type = GGMLType::F16;
                weights.stride_gate /= 2;
            }
            let mut gpu = super::Qwen35VulkanSession::try_new_rows(&model, 8, 3, context)
                .unwrap()
                .unwrap();
            assert_eq!(
                matches!(gpu.layers[0].ffn, super::FfnBindings::MoeHostRouted(_)),
                mixed
            );
            let mut cpu = crate::models::qwen35::Qwen35Session::new_with_compute(
                &mut model,
                8,
                3,
                std::sync::Arc::new(crate::ComputePool::new(1)),
                crate::compute::ComputePolicy::Cpu,
            )
            .unwrap();
            let input = [
                1.0, -2.0, 0.5, 3.0, -1.5, 0.2, 2.1, -0.7, 0.8, -0.1, 0.9, 0.4,
            ];
            for (base, rows, data) in [
                (0, 3, &input[..]),
                (3, 1, &input[..4]),
                (4, 1, &input[4..8]),
            ] {
                let positions = (base..base + rows).map(|p| [p; 4]).collect::<Vec<_>>();
                let expected = cpu.step(data, rows, &positions).unwrap();
                let actual = gpu
                    .forward_chunk(data, base, &positions, rows)
                    .unwrap()
                    .logits
                    .to_vec();
                gpu.commit_token();
                for (&a, b) in actual.iter().zip(expected) {
                    assert!(
                        a.is_finite() && (a - b).abs() <= 1e-4 + b.abs() * 1e-4,
                        "mixed={mixed} position={base}: {a} != {b}"
                    );
                }
            }
            println!(
            "physical GPU MoE: mixed={mixed}, batched prefill + 2 decode steps match independent CPU session"
        );
        }
    }

    #[test]
    fn eligibility_rejects_incomplete_dense_ffn() {
        for missing in 0..3 {
            let mut model = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
            let layer = &mut model.layers[0];
            match missing {
                0 => layer.ffn_gate = None,
                1 => layer.ffn_up = None,
                _ => layer.ffn_down = None,
            }
            assert!(check_eligibility(&super::eligibility_facts(&model))
                .expect_err("a missing FFN projection must reject the whole graph")
                .contains("ffn"));
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
    fn qwen35_weight_formats_are_vulkan_eligible() {
        for formats in [
            vec![GGMLType::Q8_0],
            vec![GGMLType::Q4_0, GGMLType::Q4_1, GGMLType::Q6K],
            vec![GGMLType::Q4K, GGMLType::Q6K],
            vec![GGMLType::Q5K],
            vec![GGMLType::Q8_0, GGMLType::Q5K, GGMLType::Q6K],
            vec![GGMLType::F32],
            vec![GGMLType::F16],
            vec![GGMLType::Q4_0, GGMLType::Q5K, GGMLType::F32],
        ] {
            let facts = EligibilityFacts {
                architecture: "qwen35".into(),
                weight_formats: formats,
                unrecorded_operations: Vec::new(),
            };
            assert_eq!(check_eligibility(&facts), Ok(()));
        }
    }

    #[test]
    fn qwen35_weights_expose_gpu_upload_bytes() {
        let q8 = [0u8; 34];
        let q4_0 = [0u8; 18];
        let q4_1 = [0u8; 20];
        let q4_k = [0u8; 144];
        let q5_k = [0u8; 176];
        let q6_k = [0u8; 210];
        let f32_values = [1.000_000_1f32, -2.5, 0.0, -0.0, 70_000.0, 1e-8];
        let weights = [
            Weight {
                kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(
                    f32_values.to_vec(),
                    0,
                    0,
                )),
                ggml_type: GGMLType::F32,
                n_in: 3,
                n_out: 2,
            },
            Weight::from_quantized(QuantizedTensor::Q8_0 {
                data: &q8,
                n_cols: 32,
                n_rows: 1,
            }),
            Weight::from_quantized(QuantizedTensor::Q4_0 {
                data: &q4_0,
                n_cols: 32,
                n_rows: 1,
            }),
            Weight::from_quantized(QuantizedTensor::Q4_1 {
                data: &q4_1,
                n_cols: 32,
                n_rows: 1,
            }),
            Weight::from_quantized(QuantizedTensor::Q4_K {
                data: &q4_k,
                n_cols: 256,
                n_rows: 1,
            }),
            Weight::from_quantized(QuantizedTensor::Q5_K {
                data: &q5_k,
                n_cols: 256,
                n_rows: 1,
            }),
            Weight::from_quantized(QuantizedTensor::Q6_K {
                data: &q6_k,
                n_cols: 256,
                n_rows: 1,
            }),
        ];

        for (weight, expected) in weights.iter().zip([
            &f32_values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()[..],
            &q8[..],
            &q4_0[..],
            &q4_1[..],
            &q4_k[..],
            &q5_k[..],
            &q6_k[..],
        ]) {
            assert_eq!(
                validated_weight_bytes(weight, "test weight").unwrap(),
                expected
            );
        }
    }

    #[test]
    fn qwen35_rejects_malformed_weight_storage() {
        let data = [0u8; 1];
        let weight = Weight::from_quantized(QuantizedTensor::Q8_0 {
            data: &data,
            n_cols: 32,
            n_rows: 1,
        });

        let error = validated_weight_bytes(&weight, "test weight")
            .expect_err("short Q8_0 storage must be rejected");
        assert!(format!("{error:?}").contains("expected 34"));

        let mut weight = Weight::from_quantized(QuantizedTensor::F32 {
            data: vec![1.0; 5],
            n_in: 0,
            n_out: 0,
        });
        weight.n_in = 3;
        weight.n_out = 2;
        assert!(validated_weight_bytes(&weight, "short F32")
            .unwrap_err()
            .to_string()
            .contains("expected 24"));
    }

    fn check_gpu_weights(
        context: &'static super::VulkanContext,
        weights: &[(&Weight<'_>, &str)],
        input: &[f32],
    ) {
        assert!(!context.is_software_icd(), "requires a physical Vulkan GPU");
        let n_in = input.len();
        let max_rows = weights
            .iter()
            .map(|(weight, _)| weight.n_out)
            .max()
            .unwrap();
        let layout =
            crate::vulkan::ops::ArenaLayout::for_dims(n_in.max(max_rows), max_rows, 1, 1, 1)
                .unwrap();
        let mut buffers = super::UploadedBuffers::new(context);
        let mut ops = super::Qwen3Ops::new(context, layout, 2).unwrap();
        let (gpu_buffers, formats): (Vec<_>, Vec<_>) = weights
            .iter()
            .map(|(weight, label)| {
                assert!(matches!(weight.ggml_type, GGMLType::F32 | GGMLType::Q8_0));
                assert_eq!(weight.n_in, n_in);
                super::upload_weight(&mut buffers, weight, label).unwrap()
            })
            .unzip();
        let bindings = ops
            .bind_weight_buffers_mode(
                &gpu_buffers,
                &formats.iter().map(|spec| spec.format).collect::<Vec<_>>(),
                formats[0].mode,
            )
            .unwrap();
        let outputs: Vec<_> = [layout.projection, layout.gate, layout.up]
            .into_iter()
            .zip(weights)
            .map(|(region, (weight, _))| (region, weight.n_out))
            .collect();
        ops.write_f32(layout.x, input).unwrap();
        let commands = super::TokenCommands::begin(context).unwrap();
        ops.record_weight_matvec_group(
            &commands,
            bindings,
            layout.x,
            layout.q8,
            layout.q8_scales,
            layout.q4_1_input_sums,
            layout.q8k,
            layout.q8k_scales,
            &outputs,
            n_in,
        )
        .unwrap();
        commands.submit_and_wait().unwrap();
        for ((weight, label), (region, rows)) in weights.iter().zip(outputs) {
            let expected = {
                let _cpu = crate::compute::ComputePolicy::Cpu.cpu_scope();
                weight.matmul(input)
            };
            // GPU/CPU reduction order can differ; use the vk_check numerical bound.
            let mut max_abs = 0.0f32;
            for (row, (&actual, expected)) in ops
                .read_f32(region, rows)
                .unwrap()
                .iter()
                .zip(expected)
                .enumerate()
            {
                let error = (actual - expected).abs();
                assert!(
                    actual.is_finite()
                        && expected.is_finite()
                        && error <= 1e-4 + 1e-4 * expected.abs(),
                    "{label} row {row}, n_in={n_in}: actual={actual}, expected={expected}, error={error}"
                );
                max_abs = max_abs.max(error);
            }
            println!(
                "{:?} {label}: {n_in} -> {rows}, group={}, max_abs={max_abs:.6e}, bounded CPU/GPU values",
                weight.ggml_type, weights.len()
            );
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn f32_weights_run_on_vulkan() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let mut cancellation = vec![0.0; 32 * 65];
        for row in cancellation.chunks_mut(32) {
            row[0] = 1e8;
            row[1] = 1.0;
            row[16] = -1e8;
        }
        let weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32 {
            data: cancellation,
            n_in: 32,
            n_out: 65,
        });
        check_gpu_weights(context, &[(&weight, "SIMD cancellation")], &[1.0; 32]);
        if crate::ops::scalar_mode() {
            let weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32 {
                data: vec![1e8, 1.0, -1e8],
                n_in: 3,
                n_out: 1,
            });
            check_gpu_weights(context, &[(&weight, "wide cancellation")], &[1.0; 3]);
        }
        for n_in in [3, 8, 16, 24, 5120, 17_409] {
            let mut input: Vec<f32> = (0..n_in)
                .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
                .collect();
            input[0] = 1.0;
            input[1] = 1.0 - f32::EPSILON;
            let weights = [48, 33, 65].map(|n_out| {
                let mut values: Vec<f32> = (0..n_in * n_out)
                    .map(|i| ((i * 31 + n_out * 17) % 257) as f32 / 63.0 - 2.0)
                    .collect();
                // Exercise cancellation alongside the wider reduction rows.
                values[..n_in].fill(0.0);
                values[0] = -1.0;
                values[1] = 1.0 + f32::EPSILON;
                Weight {
                    kernel: Box::new(crate::ops::kernel::f32::F32Kernel::new(values, 0, 0)),
                    ggml_type: GGMLType::F32,
                    n_in,
                    n_out,
                }
            });
            for count in 1..=3 {
                let labeled: Vec<_> = weights[..count]
                    .iter()
                    .zip(["alpha", "beta", "third"])
                    .collect();
                check_gpu_weights(context, &labeled, &input);
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device and RMI_QWEN35_MODEL"]
    fn qwen35_real_ssm_weights_run_on_vulkan() {
        let path = std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL must be set");
        let source =
            crate::open_model_source(std::path::Path::new(&path), crate::ComponentRole::Llm)
                .unwrap();
        let config = Qwen35Config::from_source(source.as_ref()).unwrap();
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        let input: Vec<f32> = (0..config.n_embd)
            .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
            .collect();
        let mut checked = 0;
        for layer in 0..config.n_layer_impl() {
            if !config.is_recurrent[layer] {
                continue;
            }
            let names = [
                format!("blk.{layer}.ssm_alpha.weight"),
                format!("blk.{layer}.ssm_beta.weight"),
            ];
            let weights = names.each_ref().map(|name| {
                crate::models::qwen35::trunk::weights::load_weight(source.as_ref(), name).unwrap()
            });
            check_gpu_weights(
                context,
                &[(&weights[0], &names[0]), (&weights[1], &names[1])],
                &input,
            );
            checked += 2;
        }
        assert!(checked > 0, "model must have recurrent SSM matrices");
        println!("verified {checked} real SSM matrices");
    }

    #[test]
    #[ignore = "requires a physical Vulkan GPU and RMI_QWEN35_MODEL"]
    fn qwen35_real_moe_weights_run_on_vulkan() {
        let context = Box::leak(Box::new(super::VulkanContext::new().unwrap()));
        assert!(!context.is_software_icd());
        let source = crate::open_model_source(
            std::path::Path::new(
                &std::env::var("RMI_QWEN35_MODEL").expect("RMI_QWEN35_MODEL required"),
            ),
            crate::ComponentRole::Llm,
        )
        .unwrap();
        let model = super::Qwen35Model::from_source(source.as_ref()).unwrap();
        let gpu = super::Qwen35VulkanSession::try_new_rows(&model, 4, 1, context)
            .unwrap()
            .unwrap();
        let cfg = model.config.moe.as_ref().unwrap();
        let width = model.config.n_embd;
        let input = (0..width)
            .map(|i| ((i * 29 % 251) as f32 - 125.0) / 97.0)
            .collect::<Vec<_>>();
        let pool = crate::ComputePool::new(8);
        let mut prepared =
            crate::ops::kernel::PreparedRows::new(1, width.max(cfg.n_ff_exp).max(cfg.n_ff_shexp));
        let mut worst = 0.0f32;
        for (layer, bindings) in gpu.layers.iter().enumerate() {
            let super::FfnBindings::Moe(bindings) = bindings.ffn else {
                panic!("missing expert recorder")
            };
            gpu.ops.write_f32(gpu.layout.normed, &input).unwrap();
            gpu.ops.zero_region(gpu.layout.x).unwrap();
            let commands = super::TokenCommands::begin(context).unwrap();
            gpu.record_moe(&commands, bindings, 1).unwrap();
            commands.submit_and_wait().unwrap();
            let mut actual = vec![0.0; width];
            gpu.ops.read_f32_into(gpu.layout.x, &mut actual).unwrap();
            let mut expected = vec![0.0; width];
            let moe = model.layers[layer].moe.as_ref().unwrap();
            let _cpu = crate::compute::ComputePolicy::Cpu.cpu_scope();
            crate::models::qwen35::trunk::moe::forward_qwen35moe_token(
                moe,
                cfg,
                cfg.weights_scale,
                &input,
                &mut expected,
                &mut prepared,
                &pool,
                layer,
                0,
            )
            .unwrap();
            let mut probabilities = moe.gate_inp.matmul(&input);
            crate::ops::softmax_inplace(&mut probabilities);
            let mut ids = (0..cfg.n_expert).collect::<Vec<_>>();
            ids.sort_by(|&a, &b| {
                probabilities[b]
                    .total_cmp(&probabilities[a])
                    .then_with(|| a.cmp(&b))
            });
            let mut actual_ids = vec![0.0; cfg.n_expert_used];
            gpu.ops
                .read_f32_into(gpu.layout.expert_ids, &mut actual_ids)
                .unwrap();
            assert_eq!(
                actual_ids
                    .iter()
                    .map(|v| v.to_bits() as usize)
                    .collect::<Vec<_>>(),
                ids[..cfg.n_expert_used]
            );
            let mut max_abs = 0.0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                max_abs = max_abs.max((a - b).abs());
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-4 + b.abs() * 1e-4,
                    "layer {layer}: {a} != {b}"
                );
            }
            worst = worst.max(max_abs);
            println!("real MoE layer={layer}: exact selected IDs, bounded routed+shared output, max_abs={max_abs:.8e}");
        }
        println!(
            "verified {} real MoE blocks, worst max_abs={worst:.8e}",
            gpu.layers.len()
        );
    }

    #[test]
    fn qwen35_full_model_requires_shader_float16() {
        assert!(check_device_eligibility(false)
            .expect_err("full-model Qwen3.5 must stay on CPU without shaderFloat16")
            .contains("shaderFloat16"));
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
        assert_eq!(cache.v, [0.0, 12.0, 0.0, 13.0]);
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
            moe: None,
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
        assert_eq!(layout.q8.size, config.n_ff);
        assert_eq!(layout.q8k.size, config.n_ff);
        assert!(layout.total_size() >= layout.ssm_state.end());

        let mut config = config;
        config.moe = Some(crate::models::qwen35::trunk::config::Qwen35Moe {
            n_expert: 32,
            n_expert_used: 8,
            n_ff_exp: 16,
            n_ff_shexp: 20,
            weights_scale: 1.0,
        });
        for rows in [1, 3, 16] {
            let layout = Qwen35ArenaLayout::new_rows(&config, 16, rows).unwrap();
            assert_eq!(layout.router_logits.size, rows * 32 * 4);
            assert_eq!(layout.shared_gate.size, rows * 4);
            assert_eq!(layout.ffn_gate.size, rows.max(9) * 20 * 4);
            assert_eq!(layout.ffn_up.size, layout.ffn_gate.size);
            assert_eq!(layout.down.size, rows.max(9) * config.n_embd * 4);
            assert!(layout.ffn_gate.end() <= layout.ffn_up.offset);
            assert!(layout.ffn_up.end() <= layout.down.offset);
        }
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
        for dim in [32, 64, 128] {
            for positions in [[0; 4], [3; 4], [7, 11, 13, 17]] {
                let mut coefficients = vec![0.0; dim];
                fill_mrope(&mut coefficients, positions, [16, 24, 24, 0], 1_000_000.0).unwrap();
                let mut expected = vec![0.0; dim];
                expected[..dim / 2].fill(1.0);
                crate::ops::rope_mrope(&mut expected, positions, [16, 24, 24, 0], dim, 1_000_000.0);
                assert_eq!(
                    coefficients.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
