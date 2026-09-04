use crate::core::scratchpad::{KvCache, KvState};
use crate::core::tensor::GGMLType;
use crate::models::qwen3::trunk::{Qwen3Config, Qwen3Model, Qwen3Rope};

use super::ops::{
    fill_rope_neox, ArenaLayout, GpuWeightFormat, OperatorBindings, Qwen3Ops, TokenCommands,
};
use super::{GpuBuffer, VulkanContext, VulkanError};

#[derive(Debug, Clone)]
pub(crate) struct EligibilityFacts {
    pub(crate) architecture: String,
    pub(crate) has_moe: bool,
    pub(crate) n_deepstack_layers: usize,
    pub(crate) has_qkv_bias: bool,
    pub(crate) rope: Qwen3Rope,
    pub(crate) weight_formats: Vec<GGMLType>,
    pub(crate) gate_up_formats: Vec<[GGMLType; 2]>,
}

pub(crate) fn check_eligibility(facts: &EligibilityFacts) -> Result<(), String> {
    if facts.architecture != "qwen3" {
        return Err(format!("unsupported architecture {}", facts.architecture));
    }
    if facts.has_moe {
        return Err("moe models are not supported".into());
    }
    if facts.n_deepstack_layers != 0 {
        return Err("deepstack models are not supported".into());
    }
    if facts.has_qkv_bias {
        return Err("qkv bias is not supported".into());
    }
    if facts.rope != Qwen3Rope::Neox {
        return Err("rope layout is not supported".into());
    }
    if facts.weight_formats.is_empty()
        || facts
            .weight_formats
            .iter()
            .any(|&format| GpuWeightFormat::from_ggml_type(format).is_err())
    {
        return Err("unsupported Vulkan weight format".into());
    }
    if facts
        .gate_up_formats
        .iter()
        .any(|formats| formats[0] != formats[1])
    {
        return Err("heterogeneous gate/up weight formats are not supported".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenCommitState {
    committed_len: usize,
    pending: bool,
}

impl TokenCommitState {
    pub(crate) fn new(committed_len: usize) -> Self {
        Self {
            committed_len,
            pending: false,
        }
    }

    pub(crate) fn begin(&mut self, position: usize) -> Result<(), String> {
        if self.pending {
            return Err("a Vulkan token is already pending".into());
        }
        if position != self.committed_len {
            return Err(format!(
                "Vulkan token position {position} does not match committed length {}",
                self.committed_len
            ));
        }
        self.pending = true;
        Ok(())
    }

    pub(crate) fn commit(&mut self) {
        if self.pending {
            self.committed_len += 1;
            self.pending = false;
        }
    }

    pub(crate) fn abort(&mut self) {
        self.pending = false;
    }

    pub(crate) fn committed_len(&self) -> usize {
        self.committed_len
    }

    pub(crate) fn reset(&mut self) {
        self.committed_len = 0;
        self.pending = false;
    }
}

pub(crate) fn commit_shadow_kv(
    state: &mut KvState,
    position: usize,
    k_delta: &[f32],
    v_delta: &[f32],
) -> Result<(), String> {
    let stride = state
        .arch
        .n_head_kv
        .checked_mul(state.arch.n_embd_head_k.max(state.arch.n_embd_head_v))
        .ok_or_else(|| "KV stride overflow".to_string())?;
    let delta_len = state
        .arch
        .n_layer
        .checked_mul(stride)
        .ok_or_else(|| "KV delta length overflow".to_string())?;
    if position >= state.capacity || k_delta.len() != delta_len || v_delta.len() != delta_len {
        return Err(format!(
            "Invalid Vulkan KV delta: position={position}/{} k={} v={} expected={delta_len}",
            state.capacity,
            k_delta.len(),
            v_delta.len()
        ));
    }
    let cache_len = state
        .arch
        .n_layer
        .checked_mul(state.capacity)
        .and_then(|len| len.checked_mul(stride))
        .ok_or_else(|| "KV cache length overflow".to_string())?;
    let cache_lengths_match = match &state.cache {
        KvCache::F16(cache) => cache.k.len() == cache_len && cache.v.len() == cache_len,
        KvCache::F32(cache) => cache.k.len() == cache_len && cache.v.len() == cache_len,
    };
    if !cache_lengths_match {
        return Err("Invalid CPU shadow KV cache length".into());
    }

    match &mut state.cache {
        KvCache::F16(cache) => {
            for layer in 0..state.arch.n_layer {
                let source = layer * stride;
                let target = (layer * state.capacity + position) * stride;
                for index in 0..stride {
                    cache.k[target + index] = crate::ops::f32_to_f16(k_delta[source + index]);
                    cache.v[target + index] = crate::ops::f32_to_f16(v_delta[source + index]);
                }
            }
        }
        KvCache::F32(cache) => {
            for layer in 0..state.arch.n_layer {
                let source = layer * stride;
                let target = (layer * state.capacity + position) * stride;
                cache.k[target..target + stride].copy_from_slice(&k_delta[source..source + stride]);
                cache.v[target..target + stride].copy_from_slice(&v_delta[source..source + stride]);
            }
        }
    }
    state.seq_len = state.seq_len.max(position + 1);
    state.update_access();
    Ok(())
}

pub(crate) struct GpuTokenResult<'a> {
    pub(crate) logits: &'a [f32],
    pub(crate) k_delta: &'a [f32],
    pub(crate) v_delta: &'a [f32],
}

#[derive(Clone, Copy)]
enum QkvBindings {
    Grouped(OperatorBindings),
    Split([OperatorBindings; 3]),
}

#[derive(Clone, Copy)]
struct LayerBindings {
    attn_norm: OperatorBindings,
    qkv: QkvBindings,
    qk_norm: OperatorBindings,
    wo: OperatorBindings,
    ffn_norm: OperatorBindings,
    gate_up: OperatorBindings,
    down: OperatorBindings,
}

struct UploadedBuffers {
    context: &'static VulkanContext,
    values: Vec<GpuBuffer>,
}

impl UploadedBuffers {
    fn new(context: &'static VulkanContext) -> Self {
        Self {
            context,
            values: Vec::new(),
        }
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<GpuBuffer, VulkanError> {
        let buffer = unsafe { self.context.upload_static(bytes)? };
        self.values.push(buffer);
        Ok(buffer)
    }

    fn upload_f32(&mut self, values: &[f32]) -> Result<GpuBuffer, VulkanError> {
        self.upload(bytemuck::cast_slice(values))
    }

    fn upload_tensor(&mut self, model: &Qwen3Model, name: &str) -> Result<GpuBuffer, VulkanError> {
        let bytes = model
            .source
            .tensor_slice(name)
            .ok_or_else(|| VulkanError::UnsupportedShape(format!("missing Qwen3 tensor {name}")))?;
        self.upload(bytes)
    }
}

impl Drop for UploadedBuffers {
    fn drop(&mut self) {
        let _guard = self.context.mutex.lock().ok();
        unsafe {
            for buffer in &self.values {
                self.context.destroy_buffer(buffer);
            }
        }
    }
}

pub(crate) struct Qwen3VulkanSession {
    context: &'static VulkanContext,
    ops: Qwen3Ops<'static>,
    _buffers: UploadedBuffers,
    layers: Vec<LayerBindings>,
    output_norm: OperatorBindings,
    output: OperatorBindings,
    layout: ArenaLayout,
    config: Qwen3Config,
    capacity: usize,
    commit_state: TokenCommitState,
    rope: Vec<f32>,
    logits: Vec<f32>,
    k_delta: Vec<f32>,
    v_delta: Vec<f32>,
}

impl Qwen3VulkanSession {
    pub(crate) fn try_new(
        model: &Qwen3Model,
        capacity: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        let facts = match eligibility_facts(model) {
            Ok(facts) => facts,
            Err(reason) => {
                eprintln!("[GPU] Qwen3 Vulkan unavailable: {reason}. Falling back to CPU.");
                return Ok(None);
            }
        };
        if let Err(reason) = check_eligibility(&facts) {
            eprintln!("[GPU] Qwen3 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        let config = model.config.clone();
        if config.n_embd_head_k != config.n_embd_head_v
            || config.n_head_kv == 0
            || config.n_head % config.n_head_kv != 0
        {
            eprintln!(
                "[GPU] Qwen3 Vulkan unavailable: unsupported attention shape. Falling back to CPU."
            );
            return Ok(None);
        }

        let layout = ArenaLayout::qwen3(&config, capacity)?;
        let descriptor_capacity = config
            .n_layer
            .checked_mul(9)
            .and_then(|count| count.checked_add(3))
            .ok_or(VulkanError::OutOfMemory)?;
        let mut ops = Qwen3Ops::new(context, layout, descriptor_capacity)?;
        let mut buffers = UploadedBuffers::new(context);
        let mut layers = Vec::with_capacity(config.n_layer);

        for (layer_index, layer) in model.layers.iter().enumerate() {
            let attn_norm_buffer = buffers.upload_f32(&layer.attn_norm)?;
            let attn_norm = ops.bind_buffers(&[attn_norm_buffer])?;

            let qkv_buffers = [
                buffers.upload_tensor(model, &format!("blk.{layer_index}.attn_q.weight"))?,
                buffers.upload_tensor(model, &format!("blk.{layer_index}.attn_k.weight"))?,
                buffers.upload_tensor(model, &format!("blk.{layer_index}.attn_v.weight"))?,
            ];
            let qkv_formats = [
                GpuWeightFormat::from_ggml_type(layer.wq.ggml_type)?,
                GpuWeightFormat::from_ggml_type(layer.wk.ggml_type)?,
                GpuWeightFormat::from_ggml_type(layer.wv.ggml_type)?,
            ];
            let qkv = if qkv_formats.iter().all(|format| *format == qkv_formats[0]) {
                QkvBindings::Grouped(ops.bind_weight_buffers(&qkv_buffers, &qkv_formats)?)
            } else {
                QkvBindings::Split([
                    ops.bind_weight_buffers(&qkv_buffers[0..1], &qkv_formats[0..1])?,
                    ops.bind_weight_buffers(&qkv_buffers[1..2], &qkv_formats[1..2])?,
                    ops.bind_weight_buffers(&qkv_buffers[2..3], &qkv_formats[2..3])?,
                ])
            };

            let qk_norm = match (&layer.q_norm, &layer.k_norm) {
                (Some(q_norm), Some(k_norm)) => {
                    let q_norm = buffers.upload_f32(q_norm)?;
                    let k_norm = buffers.upload_f32(k_norm)?;
                    ops.bind_buffers(&[q_norm, k_norm])?
                }
                (None, None) => ops.bind_buffers(&[])?,
                _ => {
                    return Err(VulkanError::UnsupportedShape(format!(
                        "layer {layer_index} has incomplete Q/K norm weights"
                    )))
                }
            };

            let wo_buffer =
                buffers.upload_tensor(model, &format!("blk.{layer_index}.attn_output.weight"))?;
            let wo = ops.bind_weight_buffers(
                &[wo_buffer],
                &[GpuWeightFormat::from_ggml_type(layer.wo.ggml_type)?],
            )?;

            let ffn_norm_buffer = buffers.upload_f32(&layer.ffn_norm)?;
            let ffn_norm = ops.bind_buffers(&[ffn_norm_buffer])?;

            let gate_up_buffers = [
                buffers.upload_tensor(model, &format!("blk.{layer_index}.ffn_gate.weight"))?,
                buffers.upload_tensor(model, &format!("blk.{layer_index}.ffn_up.weight"))?,
            ];
            let gate_up = ops.bind_weight_buffers(
                &gate_up_buffers,
                &[
                    GpuWeightFormat::from_ggml_type(layer.w_gate.ggml_type)?,
                    GpuWeightFormat::from_ggml_type(layer.w_up.ggml_type)?,
                ],
            )?;

            let down_buffer =
                buffers.upload_tensor(model, &format!("blk.{layer_index}.ffn_down.weight"))?;
            let down = ops.bind_weight_buffers(
                &[down_buffer],
                &[GpuWeightFormat::from_ggml_type(layer.w_down.ggml_type)?],
            )?;
            layers.push(LayerBindings {
                attn_norm,
                qkv,
                qk_norm,
                wo,
                ffn_norm,
                gate_up,
                down,
            });
        }

        let output_norm_buffer = buffers.upload_f32(&model.output_norm)?;
        let output_norm = ops.bind_buffers(&[output_norm_buffer])?;
        let output_name = output_tensor_name(model);
        let output_buffer = buffers.upload_tensor(model, output_name)?;
        let output = ops.bind_weight_buffers(
            &[output_buffer],
            &[GpuWeightFormat::from_ggml_type(model.output.ggml_type)?],
        )?;
        let kv_count = config
            .n_head_kv
            .checked_mul(config.n_embd_head_k)
            .ok_or(VulkanError::OutOfMemory)?;
        let delta_count = config
            .n_layer
            .checked_mul(kv_count)
            .ok_or(VulkanError::OutOfMemory)?;
        let vocab = config.vocab;
        let head_dim = config.n_embd_head_k;

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
            rope: vec![0.0; head_dim],
            logits: vec![0.0; vocab],
            k_delta: vec![0.0; delta_count],
            v_delta: vec![0.0; delta_count],
        }))
    }

    pub(crate) fn forward_token<'a>(
        &'a mut self,
        input: &[f32],
        position: usize,
    ) -> Result<GpuTokenResult<'a>, VulkanError> {
        self.commit_state
            .begin(position)
            .map_err(VulkanError::UnsupportedShape)?;
        if let Err(error) = self.forward_token_inner(input, position, true) {
            self.commit_state.abort();
            return Err(error);
        }
        Ok(GpuTokenResult {
            logits: &self.logits,
            k_delta: &self.k_delta,
            v_delta: &self.v_delta,
        })
    }

    pub(crate) fn forward_hidden_token<'a>(
        &'a mut self,
        input: &[f32],
        position: usize,
    ) -> Result<&'a [f32], VulkanError> {
        self.commit_state
            .begin(position)
            .map_err(VulkanError::UnsupportedShape)?;
        if let Err(error) = self.forward_token_inner(input, position, false) {
            self.commit_state.abort();
            return Err(error);
        }
        self.ops.read_f32(self.layout.normed, self.config.n_embd)
    }

    fn forward_token_inner(
        &mut self,
        input: &[f32],
        position: usize,
        project_logits: bool,
    ) -> Result<(), VulkanError> {
        let config = &self.config;
        if input.len() != config.n_embd || position >= self.capacity {
            return Err(VulkanError::UnsupportedShape(format!(
                "invalid Qwen3 token input={} position={position}/{}",
                input.len(),
                self.capacity
            )));
        }
        let q_count = config
            .n_head
            .checked_mul(config.n_embd_head_k)
            .ok_or(VulkanError::OutOfMemory)?;
        let kv_count = config
            .n_head_kv
            .checked_mul(config.n_embd_head_k)
            .ok_or(VulkanError::OutOfMemory)?;
        let attn_count = config
            .n_head
            .checked_mul(config.n_embd_head_v)
            .ok_or(VulkanError::OutOfMemory)?;
        self.ops.write_f32(self.layout.x, input)?;
        fill_rope_neox(&mut self.rope, position, config.freq_base);
        self.ops.write_f32(self.layout.logits, &self.rope)?;
        let commands = TokenCommands::begin(self.context)?;
        for (layer_index, bindings) in self.layers.iter().enumerate() {
            self.ops.record_rms_norm(
                &commands,
                bindings.attn_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.eps,
            )?;
            let qkv_outputs = [
                (self.layout.q, q_count),
                (self.layout.k, kv_count),
                (self.layout.v, kv_count),
            ];
            match bindings.qkv {
                QkvBindings::Grouped(grouped) => self.ops.record_weight_matvec_group(
                    &commands,
                    grouped,
                    self.layout.normed,
                    self.layout.q8,
                    self.layout.q8_scales,
                    self.layout.q4_1_input_sums,
                    self.layout.q8k,
                    self.layout.q8k_scales,
                    &qkv_outputs,
                    config.n_embd,
                )?,
                QkvBindings::Split(split) => {
                    for (binding, &(output, count)) in split.iter().zip(&qkv_outputs) {
                        self.ops.record_weight_matvec(
                            &commands,
                            *binding,
                            self.layout.normed,
                            self.layout.q8,
                            self.layout.q8_scales,
                            self.layout.q4_1_input_sums,
                            self.layout.q8k,
                            self.layout.q8k_scales,
                            output,
                            config.n_embd,
                            count,
                        )?;
                    }
                }
            }
            self.ops.record_qk_norm_rope(
                &commands,
                bindings.qk_norm,
                self.layout.q,
                self.layout.k,
                config.n_head,
                config.n_head_kv,
                config.n_embd_head_k,
                self.layout.logits,
                config.eps,
                config.has_qk_norm,
                config.has_qk_norm,
            )?;
            self.ops.record_kv_write(
                &commands,
                self.layout.k,
                self.layout.v,
                self.layout.kv_k,
                self.layout.kv_v,
                self.layout.kv_delta_k,
                self.layout.kv_delta_v,
                layer_index,
                position,
                config.n_layer,
                self.capacity,
                kv_count,
            )?;
            self.ops.record_attention(
                &commands,
                self.layout.q,
                self.layout.kv_k,
                self.layout.kv_v,
                self.layout.scores,
                self.layout.attn,
                layer_index,
                config.n_layer,
                position + 1,
                self.capacity,
                config.n_head,
                config.n_head_kv,
                config.n_embd_head_k,
            )?;
            self.ops.record_weight_matvec(
                &commands,
                bindings.wo,
                self.layout.attn,
                self.layout.q8,
                self.layout.q8_scales,
                self.layout.q4_1_input_sums,
                self.layout.q8k,
                self.layout.q8k_scales,
                self.layout.projection,
                attn_count,
                config.n_embd,
            )?;
            self.ops.record_add(
                &commands,
                self.layout.x,
                self.layout.projection,
                config.n_embd,
            )?;
            self.ops.record_rms_norm(
                &commands,
                bindings.ffn_norm,
                self.layout.x,
                self.layout.normed,
                config.n_embd,
                config.eps,
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
                    (self.layout.gate, config.n_ff),
                    (self.layout.up, config.n_ff),
                ],
                config.n_embd,
            )?;
            self.ops
                .record_silu_mul(&commands, self.layout.gate, self.layout.up, config.n_ff)?;
            self.ops.record_weight_matvec(
                &commands,
                bindings.down,
                self.layout.gate,
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
            config.eps,
        )?;
        if project_logits {
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
                config.vocab,
            )?;
        }
        commands.submit_and_wait()?;

        if project_logits {
            self.logits
                .copy_from_slice(self.ops.read_f32(self.layout.logits, config.vocab)?);
        }
        let delta_count = self.k_delta.len();
        self.k_delta
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_k, delta_count)?);
        self.v_delta
            .copy_from_slice(self.ops.read_f32(self.layout.kv_delta_v, delta_count)?);
        Ok(())
    }

    pub(crate) fn commit_token(&mut self) {
        self.commit_state.commit();
    }

    pub(crate) fn abort_token(&mut self) {
        self.commit_state.abort();
    }

    pub(crate) fn reset(&mut self) {
        self.commit_state.reset();
    }
}

fn eligibility_facts(model: &Qwen3Model) -> Result<EligibilityFacts, String> {
    let mut weight_formats = Vec::with_capacity(model.config.n_layer * 7 + 1);
    let mut gate_up_formats = Vec::with_capacity(model.config.n_layer);
    for layer in 0..model.config.n_layer {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let info = model
                .source
                .tensor_info(&name)
                .ok_or_else(|| format!("missing Qwen3 tensor {name}"))?;
            weight_formats.push(info.ggml_type);
        }
        gate_up_formats.push([
            model.layers[layer].w_gate.ggml_type,
            model.layers[layer].w_up.ggml_type,
        ]);
    }
    let output_name = output_tensor_name(model);
    weight_formats.push(
        model
            .source
            .tensor_info(output_name)
            .ok_or_else(|| format!("missing Qwen3 tensor {output_name}"))?
            .ggml_type,
    );
    Ok(EligibilityFacts {
        architecture: model.config.architecture.clone(),
        has_moe: model.config.moe.is_some(),
        n_deepstack_layers: model.config.n_deepstack_layers,
        has_qkv_bias: model.config.has_qkv_bias,
        rope: model.config.rope,
        weight_formats,
        gate_up_formats,
    })
}

fn output_tensor_name(model: &Qwen3Model) -> &'static str {
    if model.source.tensor_info("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scratchpad::{KvArch, KvCache, KvFormat, KvState};
    use crate::core::tensor::GGMLType;
    use crate::models::qwen3::trunk::Qwen3Rope;
    use std::sync::Arc;

    fn eligible_facts() -> EligibilityFacts {
        EligibilityFacts {
            architecture: "qwen3".into(),
            has_moe: false,
            n_deepstack_layers: 0,
            has_qkv_bias: false,
            rope: Qwen3Rope::Neox,
            weight_formats: vec![GGMLType::Q8_0; 198],
            gate_up_formats: vec![[GGMLType::Q8_0, GGMLType::Q8_0]; 28],
        }
    }

    #[test]
    fn qwen3_q8_dense_is_eligible() {
        let facts = eligible_facts();
        assert_eq!(check_eligibility(&facts), Ok(()));
    }

    #[test]
    fn heterogeneous_gate_up_stays_on_cpu_before_session_initialization() {
        let mut facts = eligible_facts();
        facts.gate_up_formats[7] = [GGMLType::Q4K, GGMLType::Q6K];

        assert!(check_eligibility(&facts)
            .expect_err("heterogeneous gate/up must be rejected by preflight")
            .contains("heterogeneous gate/up"));
    }

    #[test]
    fn unsupported_architecture_stays_on_cpu() {
        let mut facts = eligible_facts();
        facts.architecture = "qwen3vl".into();
        assert!(check_eligibility(&facts).is_err());
    }

    #[test]
    fn unsupported_dense_facts_stay_on_cpu() {
        let cases = [
            (
                "moe",
                EligibilityFacts {
                    has_moe: true,
                    ..eligible_facts()
                },
            ),
            (
                "deepstack",
                EligibilityFacts {
                    n_deepstack_layers: 4,
                    ..eligible_facts()
                },
            ),
            (
                "qkv bias",
                EligibilityFacts {
                    has_qkv_bias: true,
                    ..eligible_facts()
                },
            ),
            (
                "rope",
                EligibilityFacts {
                    rope: Qwen3Rope::Interleaved {
                        sections: [16, 24, 24, 0],
                        n_dims: 64,
                    },
                    ..eligible_facts()
                },
            ),
            (
                "weight format",
                EligibilityFacts {
                    weight_formats: vec![GGMLType::Q5K],
                    ..eligible_facts()
                },
            ),
            (
                "weight format",
                EligibilityFacts {
                    weight_formats: Vec::new(),
                    ..eligible_facts()
                },
            ),
        ];
        for (message, facts) in cases {
            assert!(check_eligibility(&facts)
                .expect_err("unsupported facts must be rejected")
                .contains(message));
        }
    }

    #[test]
    fn failed_token_does_not_advance_committed_kv() {
        let mut state = TokenCommitState::new(7);
        state.begin(7).unwrap();
        state.abort();
        assert_eq!(state.committed_len(), 7);
    }

    #[test]
    fn committed_token_advances_exactly_once() {
        let mut state = TokenCommitState::new(7);
        assert!(state.begin(6).is_err());
        state.begin(7).unwrap();
        assert!(state.begin(7).is_err());
        state.commit();
        state.commit();
        assert_eq!(state.committed_len(), 8);
    }

    #[test]
    fn reset_rewinds_committed_length() {
        let mut state = TokenCommitState::new(7);
        state.reset();
        assert_eq!(state.committed_len(), 0);
        state.begin(0).unwrap();
    }

    #[test]
    fn shadow_kv_commits_only_a_complete_token() {
        let arch = Arc::new(KvArch::new(2, 1, 2, 2, 4));
        let mut state = KvState::new(arch, KvFormat::F32, 4);
        let k = [1.0, 2.0, 3.0, 4.0];
        let v = [5.0, 6.0, 7.0, 8.0];

        assert!(commit_shadow_kv(&mut state, 1, &k[..3], &v).is_err());
        assert_eq!(state.seq_len, 0);
        commit_shadow_kv(&mut state, 1, &k, &v).unwrap();

        let KvCache::F32(cache) = &state.cache else {
            panic!("expected F32 cache");
        };
        assert_eq!(&cache.k[2..4], &[1.0, 2.0]);
        assert_eq!(&cache.k[10..12], &[3.0, 4.0]);
        assert_eq!(&cache.v[2..4], &[5.0, 6.0]);
        assert_eq!(&cache.v[10..12], &[7.0, 8.0]);
        assert_eq!(state.seq_len, 2);
    }
}
